/*
 * VulkanDevice compute-pipeline creation helpers.
 * Extracted from VulkanBackend.cpp without changing the public backend API.
 */
#include "Backend/VulkanBackend.h"

namespace {
struct AtmosphereLUTParamsGPU {
    float sunDir_intensity[4];
    float density_intensity[4];
    float physical[4];
    float weather[4];
    float rayleigh[4];
    float mie[4];
};

static AtmosphereLUTParamsGPU makeAtmosphereLUTParamsGPU(const WorldData& world) {
    const NishitaSkyParams& n = world.nishita;
    AtmosphereLUTParamsGPU p{};
    p.sunDir_intensity[0] = n.sun_direction.x;
    p.sunDir_intensity[1] = n.sun_direction.y;
    p.sunDir_intensity[2] = n.sun_direction.z;
    p.sunDir_intensity[3] = n.sun_intensity;
    p.density_intensity[0] = n.air_density;
    p.density_intensity[1] = n.dust_density;
    p.density_intensity[2] = n.ozone_density;
    p.density_intensity[3] = n.atmosphere_intensity;
    p.physical[0] = n.planet_radius;
    p.physical[1] = n.atmosphere_height;
    p.physical[2] = n.altitude;
    p.physical[3] = n.mie_anisotropy;
    p.weather[0] = n.humidity;
    p.weather[1] = n.temperature;
    p.weather[2] = n.ozone_absorption_scale;
    p.weather[3] = 0.0f;
    p.rayleigh[0] = n.rayleigh_scattering.x;
    p.rayleigh[1] = n.rayleigh_scattering.y;
    p.rayleigh[2] = n.rayleigh_scattering.z;
    p.rayleigh[3] = n.rayleigh_density;
    p.mie[0] = n.mie_scattering.x;
    p.mie[1] = n.mie_scattering.y;
    p.mie[2] = n.mie_scattering.z;
    p.mie[3] = n.mie_density;
    return p;
}


}

namespace VulkanRT {
bool VulkanDevice::createSkinningPipeline(const std::vector<uint32_t>& computeSPV) {
    if (computeSPV.empty()) return false;

    // Recreate-safe: free previous skinning resources first.
    // Existing per-BLAS descriptor sets are allocated from the old pool/layout.
    // Invalidate all cached handles so dispatchSkinning reallocates safely.
    for (auto& blas : m_blasList) {
        blas.skinningDescSet = VK_NULL_HANDLE;
    }
    if (m_skinningPipeline) { vkDestroyPipeline(m_device, m_skinningPipeline, nullptr); m_skinningPipeline = VK_NULL_HANDLE; }
    if (m_skinningPipelineLayout) { vkDestroyPipelineLayout(m_device, m_skinningPipelineLayout, nullptr); m_skinningPipelineLayout = VK_NULL_HANDLE; }
    if (m_skinningDescLayout) { vkDestroyDescriptorSetLayout(m_device, m_skinningDescLayout, nullptr); m_skinningDescLayout = VK_NULL_HANDLE; }
    if (m_skinningDescPool) { vkDestroyDescriptorPool(m_device, m_skinningDescPool, nullptr); m_skinningDescPool = VK_NULL_HANDLE; }
    
    // Create descriptor pool — one persistent set per skinned BLAS, no upper bound known at
    // pipeline-creation time so we use a generous cap.  Pool is never reset; sets are
    // allocated once per BLAS and reused every frame (FREE_DESCRIPTOR_SET_BIT not needed).
    // 64 skinned meshes × 7 bindings = 448 descriptors max.
    const uint32_t kMaxSkinnedMeshes = 64;
    VkDescriptorPoolSize poolSizes[] = { {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxSkinnedMeshes * 7} };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = kMaxSkinnedMeshes;
    poolInfo.flags = 0; // no free needed — persistent sets
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_skinningDescPool) != VK_SUCCESS) return false;

    // Create descriptor set layout (7 storage bindings)
    std::vector<VkDescriptorSetLayoutBinding> bindings(7);
    for(int i=0; i<7; ++i){
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 7;
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_skinningDescLayout) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device, m_skinningDescPool, nullptr);
        m_skinningDescPool = VK_NULL_HANDLE;
        return false;
    }

    // Create Pipeline Layout
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = 8; // 2 uints: vertexCount + boneCount

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_skinningDescLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_skinningPipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(m_device, m_skinningDescLayout, nullptr);
        m_skinningDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_skinningDescPool, nullptr);
        m_skinningDescPool = VK_NULL_HANDLE;
        return false;
    }

    // Create shader module
    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = computeSPV.size() * sizeof(uint32_t);
    smInfo.pCode = computeSPV.data();
    VkShaderModule compModule;
    if (vkCreateShaderModule(m_device, &smInfo, nullptr, &compModule) != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device, m_skinningPipelineLayout, nullptr);
        m_skinningPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_skinningDescLayout, nullptr);
        m_skinningDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_skinningDescPool, nullptr);
        m_skinningDescPool = VK_NULL_HANDLE;
        return false;

    }

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.layout = m_skinningPipelineLayout;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = compModule;
    cpInfo.stage.pName = "main";
    
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &m_skinningPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        vkDestroyPipelineLayout(m_device, m_skinningPipelineLayout, nullptr);
        m_skinningPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_skinningDescLayout, nullptr);
        m_skinningDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_skinningDescPool, nullptr);
        m_skinningDescPool = VK_NULL_HANDLE;
        return false;
    }

    vkDestroyShaderModule(m_device, compModule, nullptr);
    return true;
}

bool VulkanDevice::createSculptPipeline(const std::vector<uint32_t>& computeSPV) {
    if (computeSPV.empty()) return false;

    if (m_sculptPipeline) { vkDestroyPipeline(m_device, m_sculptPipeline, nullptr); m_sculptPipeline = VK_NULL_HANDLE; }
    if (m_sculptPipelineLayout) { vkDestroyPipelineLayout(m_device, m_sculptPipelineLayout, nullptr); m_sculptPipelineLayout = VK_NULL_HANDLE; }
    if (m_sculptDescLayout) { vkDestroyDescriptorSetLayout(m_device, m_sculptDescLayout, nullptr); m_sculptDescLayout = VK_NULL_HANDLE; }
    if (m_sculptDescPool) { vkDestroyDescriptorPool(m_device, m_sculptDescPool, nullptr); m_sculptDescPool = VK_NULL_HANDLE; }

    // Descriptor pool: allow up to 64 sets, 3 storage buffers each
    const uint32_t kMaxSculptSets = 64;
    VkDescriptorPoolSize poolSizes[] = { {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxSculptSets * 3} };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = kMaxSculptSets;
    poolInfo.flags = 0;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_sculptDescPool) != VK_SUCCESS) return false;

    // Descriptor layout: 3 storage buffers (positions, normals, weights)
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_sculptDescLayout) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device, m_sculptDescPool, nullptr);
        m_sculptDescPool = VK_NULL_HANDLE;
        return false;
    }

    // Pipeline layout with optional push constants (up to 64 bytes)
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = 64;

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_sculptDescLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_sculptPipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(m_device, m_sculptDescLayout, nullptr);
        m_sculptDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_sculptDescPool, nullptr);
        m_sculptDescPool = VK_NULL_HANDLE;
        return false;

    }

    // Create shader module
    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = computeSPV.size() * sizeof(uint32_t);
    smInfo.pCode = computeSPV.data();
    VkShaderModule compModule;
    if (vkCreateShaderModule(m_device, &smInfo, nullptr, &compModule) != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device, m_sculptPipelineLayout, nullptr);
        m_sculptPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_sculptDescLayout, nullptr);
        m_sculptDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_sculptDescPool, nullptr);
        m_sculptDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.layout = m_sculptPipelineLayout;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = compModule;
    cpInfo.stage.pName = "main";

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &m_sculptPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        vkDestroyPipelineLayout(m_device, m_sculptPipelineLayout, nullptr);
        m_sculptPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_sculptDescLayout, nullptr);
        m_sculptDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_sculptDescPool, nullptr);
        m_sculptDescPool = VK_NULL_HANDLE;
        return false;
    }

    vkDestroyShaderModule(m_device, compModule, nullptr);
    return true;
}

bool VulkanDevice::createTonemapPipeline(const std::vector<uint32_t>& computeSPV) {
    if (computeSPV.empty()) return false;

    if (m_tonemapPipeline)       { vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr); m_tonemapPipeline = VK_NULL_HANDLE; }
    if (m_tonemapPipelineLayout) { vkDestroyPipelineLayout(m_device, m_tonemapPipelineLayout, nullptr); m_tonemapPipelineLayout = VK_NULL_HANDLE; }
    if (m_tonemapDescLayout)     { vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE; }
    if (m_tonemapDescPool)       { vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr); m_tonemapDescPool = VK_NULL_HANDLE; }
    m_tonemapDescSet = VK_NULL_HANDLE;

    // Pool sized for a single persistent set. Aşama 2 binds the same set from both
    // frame slots — images don't change frame-to-frame, only on resize (at which point
    // fences are drained before updateTonemapDescriptors rewrites it).
    const uint32_t kMaxSets = 1;
    VkDescriptorPoolSize poolSizes[] = { {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxSets * 6} };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = kMaxSets;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_tonemapDescPool) != VK_SUCCESS) return false;

    // Layout: 0 = HDR input, 1 = LDR output, 2-5 = Debug Visualizer AOVs
    // (albedo / normal / position / path-stats), all storage images.
    VkDescriptorSetLayoutBinding bindings[6]{};
    for (uint32_t i = 0; i < 6; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 6;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_tonemapDescLayout) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr);
        m_tonemapDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = 40; // w,h,view + exposure,heatScale,bounceScale,overlay + camXYZ

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_tonemapDescLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_tonemapPipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr); m_tonemapDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = computeSPV.size() * sizeof(uint32_t);
    smInfo.pCode = computeSPV.data();
    VkShaderModule compModule;
    if (vkCreateShaderModule(m_device, &smInfo, nullptr, &compModule) != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device, m_tonemapPipelineLayout, nullptr); m_tonemapPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr); m_tonemapDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.layout = m_tonemapPipelineLayout;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = compModule;
    cpInfo.stage.pName = "main";

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &m_tonemapPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        vkDestroyPipelineLayout(m_device, m_tonemapPipelineLayout, nullptr); m_tonemapPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr); m_tonemapDescPool = VK_NULL_HANDLE;
        return false;
    }

    vkDestroyShaderModule(m_device, compModule, nullptr);

    // Pre-allocate the persistent descriptor set. Image views aren't known yet;
    // updateTonemapDescriptors() will populate them once the adapter has its targets.
    {
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = m_tonemapDescPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_tonemapDescLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_tonemapDescSet) != VK_SUCCESS) {
            VK_ERROR() << "[VulkanDevice] Failed to allocate persistent tonemap descriptor set." << std::endl;
            // Tear down the pipeline objects so hasTonemapPipeline() reports false.
            vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr); m_tonemapPipeline = VK_NULL_HANDLE;
            vkDestroyPipelineLayout(m_device, m_tonemapPipelineLayout, nullptr); m_tonemapPipelineLayout = VK_NULL_HANDLE;
            vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE;
            vkDestroyDescriptorPool(m_device, m_tonemapDescPool, nullptr); m_tonemapDescPool = VK_NULL_HANDLE;
            return false;
        }
    }
    return true;
}

bool VulkanDevice::updateTonemapDescriptors(const VulkanRT::ImageHandle& hdrImage, const VulkanRT::ImageHandle& ldrImage,
                                            const VulkanRT::ImageHandle* aovAlbedo,
                                            const VulkanRT::ImageHandle* aovNormal,
                                            const VulkanRT::ImageHandle* aovPosition,
                                            const VulkanRT::ImageHandle* aovStats) {
    if (m_tonemapDescSet == VK_NULL_HANDLE) return false;
    if (!hdrImage.view || !ldrImage.view) return false;

    // 6 bindings: 0=HDR in, 1=LDR out, 2=albedo AOV, 3=normal AOV,
    // 4=position AOV, 5=path-stats AOV. AOVs fall back to the HDR view so
    // every layout binding is always valid (they're only read in debug views).
    VkDescriptorImageInfo infos[6]{};
    infos[0].imageView = hdrImage.view;
    infos[1].imageView = ldrImage.view;
    infos[2].imageView = (aovAlbedo   && aovAlbedo->view)   ? aovAlbedo->view   : hdrImage.view;
    infos[3].imageView = (aovNormal   && aovNormal->view)   ? aovNormal->view   : hdrImage.view;
    infos[4].imageView = (aovPosition && aovPosition->view) ? aovPosition->view : hdrImage.view;
    infos[5].imageView = (aovStats    && aovStats->view)    ? aovStats->view    : hdrImage.view;

    VkWriteDescriptorSet writes[6]{};
    for (uint32_t i = 0; i < 6; ++i) {
        infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_tonemapDescSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(m_device, 6, writes, 0, nullptr);
    return true;
}

bool VulkanDevice::createStylizePipeline(const std::vector<uint32_t>& computeSPV) {
    if (computeSPV.empty()) return false;

    if (m_stylizePipeline)       { vkDestroyPipeline(m_device, m_stylizePipeline, nullptr); m_stylizePipeline = VK_NULL_HANDLE; }
    if (m_stylizePipelineLayout) { vkDestroyPipelineLayout(m_device, m_stylizePipelineLayout, nullptr); m_stylizePipelineLayout = VK_NULL_HANDLE; }
    if (m_stylizeDescLayout)     { vkDestroyDescriptorSetLayout(m_device, m_stylizeDescLayout, nullptr); m_stylizeDescLayout = VK_NULL_HANDLE; }
    if (m_stylizeDescPool)       { vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE; }
    m_stylizeDescSet = VK_NULL_HANDLE;

    // Pool: one persistent set with 2 storage buffers (color + params) and 3 storage images (AOVs).
    const uint32_t kMaxSets = 1;
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kMaxSets * 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  kMaxSets * 3 },
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = kMaxSets;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_stylizeDescPool) != VK_SUCCESS) return false;

    // Layout: 0=color SSBO, 1=position img, 2=albedo img, 3=normal img, 4=params SSBO.
    VkDescriptorSetLayoutBinding bindings[5]{};
    auto setBinding = [](VkDescriptorSetLayoutBinding& b, uint32_t idx, VkDescriptorType type) {
        b.binding = idx; b.descriptorType = type; b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    };
    setBinding(bindings[0], 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    setBinding(bindings[1], 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    setBinding(bindings[2], 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    setBinding(bindings[3], 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    setBinding(bindings[4], 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 5;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_stylizeDescLayout) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_stylizeDescLayout;
    if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_stylizePipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(m_device, m_stylizeDescLayout, nullptr); m_stylizeDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = computeSPV.size() * sizeof(uint32_t);
    smInfo.pCode = computeSPV.data();
    VkShaderModule compModule;
    if (vkCreateShaderModule(m_device, &smInfo, nullptr, &compModule) != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device, m_stylizePipelineLayout, nullptr); m_stylizePipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_stylizeDescLayout, nullptr); m_stylizeDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.layout = m_stylizePipelineLayout;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = compModule;
    cpInfo.stage.pName = "main";
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &m_stylizePipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        vkDestroyPipelineLayout(m_device, m_stylizePipelineLayout, nullptr); m_stylizePipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_stylizeDescLayout, nullptr); m_stylizeDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE;
        return false;
    }
    vkDestroyShaderModule(m_device, compModule, nullptr);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_stylizeDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_stylizeDescLayout;
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_stylizeDescSet) != VK_SUCCESS) {
        vkDestroyPipeline(m_device, m_stylizePipeline, nullptr); m_stylizePipeline = VK_NULL_HANDLE;
        vkDestroyPipelineLayout(m_device, m_stylizePipelineLayout, nullptr); m_stylizePipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_stylizeDescLayout, nullptr); m_stylizeDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_stylizeDescPool, nullptr); m_stylizeDescPool = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool VulkanDevice::updateStylizeDescriptors(const VulkanRT::BufferHandle& colorBuf,
                                            const VulkanRT::BufferHandle& paramsBuf,
                                            VkImageView posView, VkImageView albView, VkImageView nrmView) {
    if (m_stylizeDescSet == VK_NULL_HANDLE) return false;
    if (!colorBuf.buffer || !paramsBuf.buffer || !posView || !albView || !nrmView) return false;

    VkDescriptorBufferInfo colorInfo{ colorBuf.buffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo paramsInfo{ paramsBuf.buffer, 0, VK_WHOLE_SIZE };
    VkDescriptorImageInfo posInfo{}; posInfo.imageView = posView; posInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo albInfo{}; albInfo.imageView = albView; albInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo nrmInfo{}; nrmInfo.imageView = nrmView; nrmInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w[5]{};
    for (int i = 0; i < 5; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = m_stylizeDescSet;
        w[i].dstBinding = (uint32_t)i;
        w[i].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &colorInfo;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  w[1].pImageInfo  = &posInfo;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  w[2].pImageInfo  = &albInfo;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;  w[3].pImageInfo  = &nrmInfo;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[4].pBufferInfo = &paramsInfo;
    vkUpdateDescriptorSets(m_device, 5, w, 0, nullptr);
    return true;
}

bool VulkanDevice::dispatchStylizeCompute(uint32_t w, uint32_t h, VkImage posImg, VkImage albImg, VkImage nrmImg) {
    if (m_stylizePipeline == VK_NULL_HANDLE || m_stylizeDescSet == VK_NULL_HANDLE) return false;

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return false;

    // Make the RT-frame AOV writes available to the compute read. Images stay in
    // GENERAL (they are storage images written by raygen). Coarse but safe.
    VkImageMemoryBarrier barriers[3]{};
    VkImage imgs[3] = { posImg, albImg, nrmImg };
    uint32_t bcount = 0;
    for (int i = 0; i < 3; ++i) {
        if (!imgs[i]) continue;
        VkImageMemoryBarrier& b = barriers[bcount++];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = imgs[i];
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }
    if (bcount > 0) {
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, bcount, barriers);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stylizePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_stylizePipelineLayout, 0, 1, &m_stylizeDescSet, 0, nullptr);
    const uint32_t gx = (w + 7) / 8;
    const uint32_t gy = (h + 7) / 8;
    vkCmdDispatch(cmd, gx, gy, 1);

    endSingleTimeCommands(cmd);   // submits + waits
    return true;
}

bool VulkanDevice::createAtmosphereLUTPipeline(const std::vector<uint32_t>& computeSPV) {
    if (computeSPV.empty()) return false;

    if (m_atmosphereLutPipeline)       { vkDestroyPipeline(m_device, m_atmosphereLutPipeline, nullptr); m_atmosphereLutPipeline = VK_NULL_HANDLE; }
    if (m_atmosphereLutPipelineLayout) { vkDestroyPipelineLayout(m_device, m_atmosphereLutPipelineLayout, nullptr); m_atmosphereLutPipelineLayout = VK_NULL_HANDLE; }
    if (m_atmosphereLutDescLayout)     { vkDestroyDescriptorSetLayout(m_device, m_atmosphereLutDescLayout, nullptr); m_atmosphereLutDescLayout = VK_NULL_HANDLE; }
    if (m_atmosphereLutDescPool)       { vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr); m_atmosphereLutDescPool = VK_NULL_HANDLE; }
    m_atmosphereLutDescSet = VK_NULL_HANDLE;

    VkDescriptorPoolSize poolSizes[2] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}
    };
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = 1;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_atmosphereLutDescPool) != VK_SUCCESS) return false;

    VkDescriptorSetLayoutBinding bindings[4]{};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 4;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_atmosphereLutDescLayout) != VK_SUCCESS) {
        vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr);
        m_atmosphereLutDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset = 0;
    pc.size = 16; // phase, width, height, pad

    VkPipelineLayoutCreateInfo plInfo{};
    plInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_atmosphereLutDescLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pc;
    if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_atmosphereLutPipelineLayout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(m_device, m_atmosphereLutDescLayout, nullptr); m_atmosphereLutDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr); m_atmosphereLutDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkShaderModuleCreateInfo smInfo{};
    smInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smInfo.codeSize = computeSPV.size() * sizeof(uint32_t);
    smInfo.pCode = computeSPV.data();
    VkShaderModule compModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &smInfo, nullptr, &compModule) != VK_SUCCESS) {
        vkDestroyPipelineLayout(m_device, m_atmosphereLutPipelineLayout, nullptr); m_atmosphereLutPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_atmosphereLutDescLayout, nullptr); m_atmosphereLutDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr); m_atmosphereLutDescPool = VK_NULL_HANDLE;
        return false;
    }

    VkComputePipelineCreateInfo cpInfo{};
    cpInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpInfo.layout = m_atmosphereLutPipelineLayout;
    cpInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpInfo.stage.module = compModule;
    cpInfo.stage.pName = "main";

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &m_atmosphereLutPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        vkDestroyPipelineLayout(m_device, m_atmosphereLutPipelineLayout, nullptr); m_atmosphereLutPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_atmosphereLutDescLayout, nullptr); m_atmosphereLutDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr); m_atmosphereLutDescPool = VK_NULL_HANDLE;
        return false;
    }
    vkDestroyShaderModule(m_device, compModule, nullptr);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_atmosphereLutDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_atmosphereLutDescLayout;
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_atmosphereLutDescSet) != VK_SUCCESS) {
        vkDestroyPipeline(m_device, m_atmosphereLutPipeline, nullptr); m_atmosphereLutPipeline = VK_NULL_HANDLE;
        vkDestroyPipelineLayout(m_device, m_atmosphereLutPipelineLayout, nullptr); m_atmosphereLutPipelineLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorSetLayout(m_device, m_atmosphereLutDescLayout, nullptr); m_atmosphereLutDescLayout = VK_NULL_HANDLE;
        vkDestroyDescriptorPool(m_device, m_atmosphereLutDescPool, nullptr); m_atmosphereLutDescPool = VK_NULL_HANDLE;
        return false;
    }
    return true;
}

bool VulkanDevice::updateAtmosphereLUTComputeDescriptors(const ImageHandle* lutImages) {
    if (m_atmosphereLutDescSet == VK_NULL_HANDLE || !lutImages) return false;
    if (!lutImages[0].view || !lutImages[1].view || !lutImages[2].view || !m_atmosphereLutParamsBuffer.buffer) return false;

    VkDescriptorImageInfo imageInfos[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        imageInfos[i].imageView = lutImages[i].view;
        imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkDescriptorBufferInfo paramInfo{};
    paramInfo.buffer = m_atmosphereLutParamsBuffer.buffer;
    paramInfo.offset = 0;
    paramInfo.range = sizeof(AtmosphereLUTParamsGPU);

    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_atmosphereLutDescSet;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].descriptorCount = 1;
        writes[i].pImageInfo = &imageInfos[i];
    }
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = m_atmosphereLutDescSet;
    writes[3].dstBinding = 3;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[3].descriptorCount = 1;
    writes[3].pBufferInfo = &paramInfo;
    vkUpdateDescriptorSets(m_device, 4, writes, 0, nullptr);
    return true;
}

bool VulkanDevice::generateAtmosphereLUTGPU(const WorldData& world) {
    if (m_atmosphereLutPipeline == VK_NULL_HANDLE || m_atmosphereLutDescSet == VK_NULL_HANDLE) return false;

    waitIdle();

    const AtmosphereLUTParamsGPU params = makeAtmosphereLUTParamsGPU(world);
    if (m_atmosphereLutParamsBuffer.size < sizeof(params)) {
        if (m_atmosphereLutParamsBuffer.buffer) destroyBuffer(m_atmosphereLutParamsBuffer);
        BufferCreateInfo ci{};
        ci.size = sizeof(params);
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::CPU_TO_GPU;
        m_atmosphereLutParamsBuffer = createBuffer(ci);
    }
    if (!m_atmosphereLutParamsBuffer.buffer) return false;
    uploadBuffer(m_atmosphereLutParamsBuffer, &params, sizeof(params));

    for (int i = 0; i < 4; ++i) {
        if (m_lutImages[i].image) {
            destroyImage(m_lutImages[i]);
            m_lutImages[i] = {};
        }
    }

    constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ImageHandle lutImgs[4]{};
    lutImgs[0] = createImage2D(TRANSMITTANCE_LUT_W, TRANSMITTANCE_LUT_H, VK_FORMAT_R32G32B32A32_SFLOAT, usage);
    lutImgs[1] = createImage2D(SKYVIEW_LUT_W, SKYVIEW_LUT_H, VK_FORMAT_R32G32B32A32_SFLOAT, usage);
    lutImgs[2] = createImage2D(MULTI_SCATTER_LUT_RES, MULTI_SCATTER_LUT_RES, VK_FORMAT_R32G32B32A32_SFLOAT, usage);
    if (!lutImgs[0].image || !lutImgs[1].image || !lutImgs[2].image) {
        for (auto& img : lutImgs) if (img.image) destroyImage(img);
        return false;
    }

    auto createSampler = [&](ImageHandle& img, bool wrapU) {
        VkSamplerCreateInfo sInfo{};
        sInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sInfo.magFilter = VK_FILTER_LINEAR;
        sInfo.minFilter = VK_FILTER_LINEAR;
        sInfo.addressModeU = wrapU ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sInfo.anisotropyEnable = VK_FALSE;
        sInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        sInfo.unnormalizedCoordinates = VK_FALSE;
        vkCreateSampler(m_device, &sInfo, nullptr, &img.sampler);
    };
    createSampler(lutImgs[0], false);
    createSampler(lutImgs[1], true);
    createSampler(lutImgs[2], false);

    if (!updateAtmosphereLUTComputeDescriptors(lutImgs)) {
        for (auto& img : lutImgs) if (img.image) destroyImage(img);
        return false;
    }

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) {
        for (auto& img : lutImgs) if (img.image) destroyImage(img);
        return false;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atmosphereLutPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atmosphereLutPipelineLayout, 0, 1, &m_atmosphereLutDescSet, 0, nullptr);

    struct PushConstants { uint32_t phase; uint32_t width; uint32_t height; uint32_t pad; } pc{};
    auto dispatchPhase = [&](uint32_t phase, uint32_t w, uint32_t h) {
        pc.phase = phase;
        pc.width = w;
        pc.height = h;
        vkCmdPushConstants(cmd, m_atmosphereLutPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (w + 7u) / 8u, (h + 7u) / 8u, 1);
    };
    dispatchPhase(0, TRANSMITTANCE_LUT_W, TRANSMITTANCE_LUT_H);
    dispatchPhase(1, SKYVIEW_LUT_W, SKYVIEW_LUT_H);
    dispatchPhase(2, MULTI_SCATTER_LUT_RES, MULTI_SCATTER_LUT_RES);

    for (int i = 0; i < 3; ++i) {
        transitionImageLayout(cmd, lutImgs[i].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    endSingleTimeCommands(cmd);

    updateAtmosphereLUTs(lutImgs);
    return true;
}

} // namespace VulkanRT
