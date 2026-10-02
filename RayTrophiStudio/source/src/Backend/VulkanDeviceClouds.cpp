// Cloud field resources on one VkDevice (docs/dev/ATMOSPHERE_CLOUDS.md, Faz 3a):
// generated noise textures, the weather map, and the measurement dispatch
// behind world.sample_clouds. Each Vulkan device builds its own copy with the
// same shader and inputs -- two VkDevices cannot share a texture.

#include "Backend/VulkanBackend.h"
#include "Backend/CloudParams.h"

#include <algorithm>
#include <cstring>

namespace VulkanRT {

namespace {

VkSampler makeRepeatSampler(VkDevice device) {
    VkSamplerCreateInfo s{};
    s.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    s.magFilter = VK_FILTER_LINEAR;
    s.minFilter = VK_FILTER_LINEAR;
    s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    // REPEAT: every cloud texture tiles by construction (cloud_noise.comp).
    s.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    s.maxLod = 0.0f;
    VkSampler out = VK_NULL_HANDLE;
    vkCreateSampler(device, &s, nullptr, &out);
    return out;
}

bool makeComputePipeline(VkDevice device, const std::vector<uint32_t>& spv, VkPipelineLayout layout,
                         VkPipeline& out) {
    VkShaderModuleCreateInfo sm{};
    sm.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    sm.codeSize = spv.size() * sizeof(uint32_t);
    sm.pCode = spv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &sm, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cp{};
    cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cp.layout = layout;
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = module;
    cp.stage.pName = "main";
    const bool ok = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &out) == VK_SUCCESS;
    vkDestroyShaderModule(device, module, nullptr);
    return ok;
}

} // namespace

ImageHandle VulkanDevice::createImage3D(uint32_t w, uint32_t h, uint32_t d, VkFormat format,
                                        VkImageUsageFlags usage) {
    ImageHandle handle{};
    if (w == 0 || h == 0 || d == 0) return {};
    handle.width = w;
    handle.height = h;
    handle.format = format;

    VkImageCreateInfo ii{};
    ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType = VK_IMAGE_TYPE_3D;
    ii.format = format;
    ii.extent = {w, h, d};
    ii.mipLevels = 1;
    ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = usage;
    ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(m_device, &ii, nullptr, &handle.image) != VK_SUCCESS) return {};

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(m_device, handle.image, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX ||
        vkAllocateMemory(m_device, &ai, nullptr, &handle.memory) != VK_SUCCESS) {
        vkDestroyImage(m_device, handle.image, nullptr);
        return {};
    }
    noteMemoryAllocated(handle.memory, req.size, ai.memoryTypeIndex, VramCategory::RenderTarget);
    if (vkBindImageMemory(m_device, handle.image, handle.memory, 0) != VK_SUCCESS) {
        freeTrackedMemory(handle.memory);
        vkDestroyImage(m_device, handle.image, nullptr);
        return {};
    }
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = handle.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_3D;
    vi.format = format;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    if (vkCreateImageView(m_device, &vi, nullptr, &handle.view) != VK_SUCCESS) {
        vkDestroyImage(m_device, handle.image, nullptr);
        freeTrackedMemory(handle.memory);
        return {};
    }
    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) { destroyImage(handle); return {}; }
    transitionImageLayout(cmd, handle.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    endSingleTimeCommands(cmd);
    return handle;
}

void VulkanDevice::destroyCloudResources() {
    destroyCloudRaster();
    destroyCloudPipelines();
    for (ImageHandle* img : {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather, &m_cloudMajorant}) {
        if (img->image || img->view) destroyImage(*img);
        *img = {};
    }
    for (BufferHandle* b : {&m_cloudSampleParamsBuffer, &m_cloudQueryBuffer, &m_cloudResultBuffer,
                            &m_cloudRtParamsBuffer}) {
        if (b->buffer) destroyBuffer(*b);
        *b = {};
    }
    m_cloudQueryCapacity = 0;
    m_cloudNoiseReady = false;
    m_cloudWeatherHash = 0;
}

bool VulkanDevice::ensureCloudTextures() {
    if (m_cloudBase.view && m_cloudMajorant.view && m_cloudRtParamsBuffer.buffer) return true;
    const VkImageUsageFlags use = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VramCategoryScope vram(VramCategory::RenderTarget);
    // createImage2D / createImage3D leave the images in GENERAL: the storage
    // layout for generation and a legal sampled layout for the readers.
    m_cloudBase = createImage3D(Backend::kCloudBaseNoiseSize, Backend::kCloudBaseNoiseSize,
                                Backend::kCloudBaseNoiseSize, VK_FORMAT_R8G8B8A8_UNORM, use);
    m_cloudDetail = createImage3D(Backend::kCloudDetailNoiseSize, Backend::kCloudDetailNoiseSize,
                                  Backend::kCloudDetailNoiseSize, VK_FORMAT_R8G8B8A8_UNORM, use);
    m_cloudCurl = createImage2D(Backend::kCloudCurlNoiseSize, Backend::kCloudCurlNoiseSize,
                                VK_FORMAT_R8G8B8A8_UNORM, use);
    m_cloudWeather = createImage2D(Backend::kCloudWeatherMapSize, Backend::kCloudWeatherMapSize,
                                   VK_FORMAT_R8G8B8A8_UNORM, use);
    m_cloudMajorant = createImage2D(Backend::kCloudMajorantSize, Backend::kCloudMajorantSize,
                                    VK_FORMAT_R8G8B8A8_UNORM, use);
    bool ok = m_cloudBase.view && m_cloudDetail.view && m_cloudCurl.view && m_cloudWeather.view &&
              m_cloudMajorant.view;
    if (ok) {
        for (ImageHandle* img : {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather}) {
            img->sampler = makeRepeatSampler(m_device);
            ok = ok && img->sampler != VK_NULL_HANDLE;
        }
        // The majorant is read with texelFetch; any sampler will do.
        m_cloudMajorant.sampler = makeRepeatSampler(m_device);
        ok = ok && m_cloudMajorant.sampler != VK_NULL_HANDLE;
    }
    if (ok) {
        BufferCreateInfo ci{};
        ci.size = sizeof(Backend::CloudParamsGPU);
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        m_cloudRtParamsBuffer = createBuffer(ci);
        ok = m_cloudRtParamsBuffer.buffer != VK_NULL_HANDLE;
    }
    if (ok) {
        // Zero = every layer off and the render flag clear: RT draws no cloud
        // from uninitialised texture memory.
        const Backend::CloudParamsGPU zero{};
        uploadBuffer(m_cloudRtParamsBuffer, &zero, sizeof(zero));
        return true;
    }
    for (ImageHandle* img : {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather, &m_cloudMajorant}) {
        if (img->image || img->view) destroyImage(*img);
        *img = {};
    }
    if (m_cloudRtParamsBuffer.buffer) destroyBuffer(m_cloudRtParamsBuffer);
    m_cloudRtParamsBuffer = {};
    return false;
}

// RT set bindings 30-35: base, detail, curl, weather, params, majorant.
void VulkanDevice::writeRtCloudDescriptors() {
    if (m_rtDescriptorSet == VK_NULL_HANDLE || !ensureCloudTextures()) return;
    const ImageHandle* imgs[5] = {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather, &m_cloudMajorant};
    const uint32_t bindings[5] = {30, 31, 32, 33, 35};
    VkDescriptorImageInfo ii[5]{};
    VkWriteDescriptorSet w[6]{};
    for (uint32_t i = 0; i < 5; ++i) {
        ii[i].sampler = imgs[i]->sampler;
        ii[i].imageView = imgs[i]->view;
        ii[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = m_rtDescriptorSet;
        w[i].dstBinding = bindings[i];
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[i].pImageInfo = &ii[i];
    }
    VkDescriptorBufferInfo bi{m_cloudRtParamsBuffer.buffer, 0, sizeof(Backend::CloudParamsGPU)};
    w[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[5].dstSet = m_rtDescriptorSet;
    w[5].dstBinding = 34;
    w[5].descriptorCount = 1;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[5].pBufferInfo = &bi;
    vkUpdateDescriptorSets(m_device, 6, w, 0, nullptr);
}

void VulkanDevice::updateCloudRtParams(const Backend::CloudParamsGPU& params) {
    if (!ensureCloudTextures()) return;
    // Ordered transfer (same path as the world buffer): no frame in flight
    // sees a half-written block.
    uploadBuffer(m_cloudRtParamsBuffer, &params, sizeof(params));
}

void VulkanDevice::destroyCloudPipelines() {
    auto kill = [&](VkPipeline& p) { if (p) { vkDestroyPipeline(m_device, p, nullptr); p = VK_NULL_HANDLE; } };
    auto killL = [&](VkPipelineLayout& p) { if (p) { vkDestroyPipelineLayout(m_device, p, nullptr); p = VK_NULL_HANDLE; } };
    auto killD = [&](VkDescriptorSetLayout& p) { if (p) { vkDestroyDescriptorSetLayout(m_device, p, nullptr); p = VK_NULL_HANDLE; } };
    kill(m_cloudNoisePipeline);
    kill(m_cloudSamplePipeline);
    killL(m_cloudNoisePipelineLayout);
    killL(m_cloudSamplePipelineLayout);
    killD(m_cloudNoiseDescLayout);
    killD(m_cloudSampleDescLayout);
    if (m_cloudDescPool) { vkDestroyDescriptorPool(m_device, m_cloudDescPool, nullptr); m_cloudDescPool = VK_NULL_HANDLE; }
    m_cloudNoiseDescSet = VK_NULL_HANDLE;
    m_cloudSampleDescSet = VK_NULL_HANDLE;
}

bool VulkanDevice::createCloudPipelines(const std::vector<uint32_t>& noiseSPV,
                                        const std::vector<uint32_t>& sampleSPV) {
    if (noiseSPV.empty() || sampleSPV.empty()) return false;
    // Pipelines only: the textures may already be bound to the RT set.
    destroyCloudPipelines();
    if (!ensureCloudTextures()) return false;

    // Noise set: 0,1 = 3D storage, 2,3 = 2D storage, 4 = majorant (2D storage).
    VkDescriptorSetLayoutBinding nb[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        nb[i].binding = i;
        nb[i].descriptorCount = 1;
        nb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        nb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // Sample set: 0-3 sampled textures, 4 params, 5 queries, 6 results.
    VkDescriptorSetLayoutBinding sb[7]{};
    for (uint32_t i = 0; i < 7; ++i) {
        sb[i].binding = i;
        sb[i].descriptorCount = 1;
        sb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        sb[i].descriptorType = i < 4 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 5; li.pBindings = nb;
    bool ok = vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_cloudNoiseDescLayout) == VK_SUCCESS;
    li.bindingCount = 7; li.pBindings = sb;
    ok = ok && vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_cloudSampleDescLayout) == VK_SUCCESS;

    if (ok) {
        VkDescriptorPoolSize sizes[3] = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 5},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}};
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.poolSizeCount = 3; pi.pPoolSizes = sizes; pi.maxSets = 2;
        ok = vkCreateDescriptorPool(m_device, &pi, nullptr, &m_cloudDescPool) == VK_SUCCESS;
    }
    if (ok) {
        VkDescriptorSetLayout layouts[2] = {m_cloudNoiseDescLayout, m_cloudSampleDescLayout};
        VkDescriptorSet sets[2] = {};
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = m_cloudDescPool; ai.descriptorSetCount = 2; ai.pSetLayouts = layouts;
        ok = vkAllocateDescriptorSets(m_device, &ai, sets) == VK_SUCCESS;
        m_cloudNoiseDescSet = sets[0];
        m_cloudSampleDescSet = sets[1];
    }
    auto makeLayout = [&](VkDescriptorSetLayout set, uint32_t pushBytes, VkPipelineLayout& out) {
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes};
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1; pl.pSetLayouts = &set;
        pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &range;
        return vkCreatePipelineLayout(m_device, &pl, nullptr, &out) == VK_SUCCESS;
    };
    ok = ok && makeLayout(m_cloudNoiseDescLayout, 16, m_cloudNoisePipelineLayout);
    ok = ok && makeLayout(m_cloudSampleDescLayout, 16, m_cloudSamplePipelineLayout);
    ok = ok && makeComputePipeline(m_device, noiseSPV, m_cloudNoisePipelineLayout, m_cloudNoisePipeline);
    ok = ok && makeComputePipeline(m_device, sampleSPV, m_cloudSamplePipelineLayout, m_cloudSamplePipeline);

    if (ok && !m_cloudSampleParamsBuffer.buffer) {
        BufferCreateInfo ci{};
        ci.size = sizeof(Backend::CloudParamsGPU);
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::CPU_TO_GPU;
        m_cloudSampleParamsBuffer = createBuffer(ci);
        ok = m_cloudSampleParamsBuffer.buffer != VK_NULL_HANDLE;
    }
    if (!ok) {
        destroyCloudPipelines();
        return false;
    }

    // Noise set: fixed.
    VkDescriptorImageInfo storage[5]{};
    const ImageHandle* imgs[5] = {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather, &m_cloudMajorant};
    VkWriteDescriptorSet w[5]{};
    for (uint32_t i = 0; i < 5; ++i) {
        storage[i].imageView = imgs[i]->view;
        storage[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = m_cloudNoiseDescSet;
        w[i].dstBinding = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[i].pImageInfo = &storage[i];
    }
    vkUpdateDescriptorSets(m_device, 5, w, 0, nullptr);

    // Sample set: textures + params fixed; queries/results written per call.
    VkDescriptorImageInfo sampled[4]{};
    VkWriteDescriptorSet sw[5]{};
    for (uint32_t i = 0; i < 4; ++i) {
        sampled[i].sampler = imgs[i]->sampler;
        sampled[i].imageView = imgs[i]->view;
        sampled[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        sw[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        sw[i].dstSet = m_cloudSampleDescSet;
        sw[i].dstBinding = i;
        sw[i].descriptorCount = 1;
        sw[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sw[i].pImageInfo = &sampled[i];
    }
    VkDescriptorBufferInfo pinfo{m_cloudSampleParamsBuffer.buffer, 0, sizeof(Backend::CloudParamsGPU)};
    sw[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    sw[4].dstSet = m_cloudSampleDescSet;
    sw[4].dstBinding = 4;
    sw[4].descriptorCount = 1;
    sw[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sw[4].pBufferInfo = &pinfo;
    vkUpdateDescriptorSets(m_device, 5, sw, 0, nullptr);
    return true;
}

bool VulkanDevice::dispatchCloudNoise(uint32_t mode, uint32_t size, uint32_t seed, uint32_t cells) {
    if (!m_cloudNoisePipeline || !m_cloudNoiseDescSet) return false;
    // Frames of THIS device may be sampling the textures being rewritten. A
    // regeneration is rare (startup, seed/extent change), so a full idle is
    // the simple correct choice.
    waitIdle();
    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return false;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudNoisePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudNoisePipelineLayout,
                            0, 1, &m_cloudNoiseDescSet, 0, nullptr);
    const uint32_t pc[4] = {mode, size, seed, cells};
    vkCmdPushConstants(cmd, m_cloudNoisePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    const uint32_t g = (size + 7u) / 8u;
    const uint32_t gz = (mode <= 1u) ? size : 1u;   // 3D modes: one Z slice per group
    vkCmdDispatch(cmd, g, g, gz);
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    endSingleTimeCommands(cmd);
    return true;
}

bool VulkanDevice::generateCloudNoise() {
    if (m_cloudNoiseReady) return true;
    const bool ok =
        dispatchCloudNoise(0u, Backend::kCloudBaseNoiseSize, 0u, 0u) &&
        dispatchCloudNoise(1u, Backend::kCloudDetailNoiseSize, 0u, 0u) &&
        dispatchCloudNoise(2u, Backend::kCloudCurlNoiseSize, 0u, 0u);
    if (ok) {
        m_cloudNoiseReady = true;
        ++m_cloudNoiseGenerations;
    }
    return ok;
}

bool VulkanDevice::generateCloudWeather(const Backend::CloudWeatherGenParams& w) {
    const uint64_t h = Backend::hashCloudWeatherGen(w);
    if (h == m_cloudWeatherHash) return true;
    // The majorant is derived from the map, so it is rebuilt with it.
    if (!dispatchCloudNoise(3u, Backend::kCloudWeatherMapSize, w.seed, w.feature_cells) ||
        !dispatchCloudNoise(4u, Backend::kCloudMajorantSize, 0u, 0u)) return false;
    m_cloudWeatherHash = h;
    ++m_cloudWeatherGenerations;
    return true;
}

bool VulkanDevice::sampleClouds(const Backend::CloudParamsGPU& params, uint32_t mode, uint32_t steps,
                                const std::vector<Backend::CloudQueryGPU>& queries,
                                std::vector<float>& out) {
    out.clear();
    if (!m_cloudSamplePipeline || !m_cloudNoiseReady || m_cloudWeatherHash == 0 || queries.empty())
        return false;
    const uint32_t n = static_cast<uint32_t>(queries.size());
    if (n > m_cloudQueryCapacity) {
        if (m_cloudQueryBuffer.buffer) destroyBuffer(m_cloudQueryBuffer);
        if (m_cloudResultBuffer.buffer) destroyBuffer(m_cloudResultBuffer);
        const uint32_t cap = std::max<uint32_t>(64u, n);
        BufferCreateInfo q{};
        q.size = uint64_t(cap) * sizeof(Backend::CloudQueryGPU);
        q.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        q.location = MemoryLocation::CPU_TO_GPU;
        m_cloudQueryBuffer = createBuffer(q);
        BufferCreateInfo r{};
        r.size = uint64_t(cap) * sizeof(float);
        r.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_SRC;
        r.location = MemoryLocation::GPU_TO_CPU;
        m_cloudResultBuffer = createBuffer(r);
        if (!m_cloudQueryBuffer.buffer || !m_cloudResultBuffer.buffer) {
            m_cloudQueryCapacity = 0;
            return false;
        }
        m_cloudQueryCapacity = cap;
        VkDescriptorBufferInfo qi{m_cloudQueryBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo ri{m_cloudResultBuffer.buffer, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet w[2]{};
        for (int i = 0; i < 2; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = m_cloudSampleDescSet;
            w[i].dstBinding = 5u + uint32_t(i);
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = i == 0 ? &qi : &ri;
        }
        // The sample set is used only by this synchronous call, so rewriting
        // it cannot race a frame.
        vkUpdateDescriptorSets(m_device, 2, w, 0, nullptr);
    }
    uploadBuffer(m_cloudSampleParamsBuffer, &params, sizeof(params));
    uploadBuffer(m_cloudQueryBuffer, queries.data(), uint64_t(n) * sizeof(Backend::CloudQueryGPU));

    VkCommandBuffer cmd = beginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) return false;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudSamplePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudSamplePipelineLayout,
                            0, 1, &m_cloudSampleDescSet, 0, nullptr);
    const uint32_t pc[4] = {mode, n, std::max(1u, steps), 0u};
    vkCmdPushConstants(cmd, m_cloudSamplePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
    vkCmdDispatch(cmd, (n + 63u) / 64u, 1, 1);
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    endSingleTimeCommands(cmd);

    out.resize(n);
    downloadBuffer(m_cloudResultBuffer, out.data(), uint64_t(n) * sizeof(float));
    ++m_cloudSampleDispatches;
    return true;
}

// ── RayFusion cloud layer (Faz 3c) ──────────────────────────────────────────
// Set: 0-3 cloud textures, 4 CloudParams (the RT params buffer: one block for
// both renderers), 5 majorant, 6 atmosphere LUTs [transmittance, sky-view],
// 7 frame block, 8 previous history (sampled), 9 current history, 10 output.

namespace {
VkSampler makeClampSampler(VkDevice device) {
    VkSamplerCreateInfo s{};
    s.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    s.magFilter = VK_FILTER_LINEAR;
    s.minFilter = VK_FILTER_LINEAR;
    s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    s.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    s.maxLod = 0.0f;
    VkSampler out = VK_NULL_HANDLE;
    vkCreateSampler(device, &s, nullptr, &out);
    return out;
}
} // namespace

void VulkanDevice::destroyCloudRaster() {
    for (ImageHandle* img : {&m_cloudHist[0], &m_cloudHist[1], &m_cloudRasterOut, &m_cloudShadowMap}) {
        if (img->image || img->view) destroyImage(*img);
        *img = {};
    }
    if (m_cloudShadowPipeline) vkDestroyPipeline(m_device, m_cloudShadowPipeline, nullptr);
    m_cloudShadowPipeline = VK_NULL_HANDLE;
    if (m_cloudRasterFrameBuf.buffer) destroyBuffer(m_cloudRasterFrameBuf);
    m_cloudRasterFrameBuf = {};
    if (m_cloudRasterPipeline) vkDestroyPipeline(m_device, m_cloudRasterPipeline, nullptr);
    if (m_cloudRasterLayout) vkDestroyPipelineLayout(m_device, m_cloudRasterLayout, nullptr);
    if (m_cloudRasterDescLayout) vkDestroyDescriptorSetLayout(m_device, m_cloudRasterDescLayout, nullptr);
    if (m_cloudRasterPool) vkDestroyDescriptorPool(m_device, m_cloudRasterPool, nullptr);
    m_cloudRasterPipeline = VK_NULL_HANDLE;
    m_cloudRasterLayout = VK_NULL_HANDLE;
    m_cloudRasterDescLayout = VK_NULL_HANDLE;
    m_cloudRasterPool = VK_NULL_HANDLE;
    m_cloudRasterSets[0] = m_cloudRasterSets[1] = VK_NULL_HANDLE;
    m_cloudRasterW = m_cloudRasterH = 0;
}

bool VulkanDevice::createCloudRasterPipeline(const std::vector<uint32_t>& spv) {
    if (spv.empty() || !ensureCloudTextures()) return false;
    destroyCloudRaster();
    VkDescriptorSetLayoutBinding b[12]{};
    for (uint32_t i = 0; i < 12; ++i) {
        b[i].binding = i;
        b[i].descriptorCount = (i == 6) ? 2u : 1u;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        b[i].descriptorType = (i == 4 || i == 7) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                            : (i >= 9)           ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                 : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    VkDescriptorSetLayoutCreateInfo li{};
    li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    li.bindingCount = 12; li.pBindings = b;
    bool ok = vkCreateDescriptorSetLayout(m_device, &li, nullptr, &m_cloudRasterDescLayout) == VK_SUCCESS;
    if (ok) {
        VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * 9},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * 2},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * 3}};
        VkDescriptorPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pi.poolSizeCount = 3; pi.pPoolSizes = sizes; pi.maxSets = 2;
        ok = vkCreateDescriptorPool(m_device, &pi, nullptr, &m_cloudRasterPool) == VK_SUCCESS;
    }
    if (ok) {
        VkDescriptorSetLayout layouts[2] = {m_cloudRasterDescLayout, m_cloudRasterDescLayout};
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = m_cloudRasterPool; ai.descriptorSetCount = 2; ai.pSetLayouts = layouts;
        ok = vkAllocateDescriptorSets(m_device, &ai, m_cloudRasterSets) == VK_SUCCESS;
    }
    if (ok) {
        VkPipelineLayoutCreateInfo pl{};
        pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pl.setLayoutCount = 1; pl.pSetLayouts = &m_cloudRasterDescLayout;
        ok = vkCreatePipelineLayout(m_device, &pl, nullptr, &m_cloudRasterLayout) == VK_SUCCESS;
    }
    ok = ok && makeComputePipeline(m_device, spv, m_cloudRasterLayout, m_cloudRasterPipeline);
    if (ok) {
        BufferCreateInfo ci{};
        ci.size = sizeof(Backend::CloudRasterFrameGPU);
        ci.usage = BufferUsage::STORAGE | BufferUsage::TRANSFER_DST;
        ci.location = MemoryLocation::GPU_ONLY;
        m_cloudRasterFrameBuf = createBuffer(ci);
        ok = m_cloudRasterFrameBuf.buffer != VK_NULL_HANDLE;
    }
    if (!ok) destroyCloudRaster();
    return ok;
}

// Shadow map (binding 11 of the same set): created with its pipeline, fixed
// size, so the binding is written once for both sets.
bool VulkanDevice::createCloudShadowPipeline(const std::vector<uint32_t>& spv) {
    if (spv.empty() || !m_cloudRasterLayout || !m_cloudRasterSets[0]) return false;
    if (!makeComputePipeline(m_device, spv, m_cloudRasterLayout, m_cloudShadowPipeline)) return false;
    {
        VramCategoryScope vram(VramCategory::RenderTarget);
        m_cloudShadowMap = createImage2D(Backend::kCloudShadowMapSize, Backend::kCloudShadowMapSize,
                                         VK_FORMAT_R16G16B16A16_SFLOAT,
                                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    }
    if (m_cloudShadowMap.view) m_cloudShadowMap.sampler = makeClampSampler(m_device);
    if (!m_cloudShadowMap.sampler) {
        vkDestroyPipeline(m_device, m_cloudShadowPipeline, nullptr);
        m_cloudShadowPipeline = VK_NULL_HANDLE;
        if (m_cloudShadowMap.image || m_cloudShadowMap.view) destroyImage(m_cloudShadowMap);
        m_cloudShadowMap = {};
        return false;
    }
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, m_cloudShadowMap.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2]{};
    for (uint32_t s = 0; s < 2; ++s) {
        w[s].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[s].dstSet = m_cloudRasterSets[s];
        w[s].dstBinding = 11;
        w[s].descriptorCount = 1;
        w[s].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[s].pImageInfo = &ii;
    }
    vkUpdateDescriptorSets(m_device, 2, w, 0, nullptr);
    return true;
}

// LUT slots (binding 6). Called with the targets and again whenever the LUT
// images are rebuilt (writeRtAtmosphereDescriptors runs after a device idle,
// so no in-flight frame holds these sets). Before the LUTs exist the weather
// map stands in; the frame block's lutReady = 0 keeps the shader off them.
void VulkanDevice::writeCloudRasterLutDescriptors() {
    if (!m_cloudRasterSets[0] || !m_cloudWeather.view) return;
    VkDescriptorImageInfo ii[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        const ImageHandle& lut = m_lutImages[i == 0 ? 0 : 1];
        const bool have = lut.view && lut.sampler;
        ii[i].sampler = have ? lut.sampler : m_cloudWeather.sampler;
        ii[i].imageView = have ? lut.view : m_cloudWeather.view;
        ii[i].imageLayout = have ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
    }
    VkWriteDescriptorSet w[2]{};
    for (uint32_t s = 0; s < 2; ++s) {
        w[s].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[s].dstSet = m_cloudRasterSets[s];
        w[s].dstBinding = 6;
        w[s].descriptorCount = 2;
        w[s].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[s].pImageInfo = ii;
    }
    vkUpdateDescriptorSets(m_device, 2, w, 0, nullptr);
}

bool VulkanDevice::ensureCloudRasterTargets(uint32_t w, uint32_t h) {
    if (!m_cloudRasterPipeline || w == 0 || h == 0) return false;
    if (w == m_cloudRasterW && h == m_cloudRasterH && m_cloudRasterOut.view) return false;
    waitIdle();
    for (ImageHandle* img : {&m_cloudHist[0], &m_cloudHist[1], &m_cloudRasterOut}) {
        if (img->image || img->view) destroyImage(*img);
        *img = {};
    }
    const VkImageUsageFlags use = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    {
        VramCategoryScope vram(VramCategory::RenderTarget);
        for (ImageHandle* img : {&m_cloudHist[0], &m_cloudHist[1], &m_cloudRasterOut}) {
            *img = createImage2D(w, h, VK_FORMAT_R16G16B16A16_SFLOAT, use);
            if (img->view) img->sampler = makeClampSampler(m_device);
        }
    }
    if (!m_cloudHist[0].sampler || !m_cloudHist[1].sampler || !m_cloudRasterOut.sampler) {
        m_cloudRasterW = m_cloudRasterH = 0;
        return false;
    }
    m_cloudRasterW = w;
    m_cloudRasterH = h;
    m_cloudRasterParity = 0;

    const ImageHandle* tex[6] = {&m_cloudBase, &m_cloudDetail, &m_cloudCurl, &m_cloudWeather, nullptr, &m_cloudMajorant};
    for (uint32_t s = 0; s < 2; ++s) {
        VkDescriptorImageInfo ti[6]{};
        VkDescriptorImageInfo prev{}, cur{}, out{};
        VkDescriptorBufferInfo params{m_cloudRtParamsBuffer.buffer, 0, sizeof(Backend::CloudParamsGPU)};
        VkDescriptorBufferInfo frame{m_cloudRasterFrameBuf.buffer, 0, sizeof(Backend::CloudRasterFrameGPU)};
        std::vector<VkWriteDescriptorSet> ws;
        auto add = [&](uint32_t binding, VkDescriptorType type, const VkDescriptorImageInfo* ii,
                       const VkDescriptorBufferInfo* bi) {
            VkWriteDescriptorSet x{};
            x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            x.dstSet = m_cloudRasterSets[s];
            x.dstBinding = binding;
            x.descriptorCount = 1;
            x.descriptorType = type;
            x.pImageInfo = ii;
            x.pBufferInfo = bi;
            ws.push_back(x);
        };
        for (uint32_t i = 0; i < 6; ++i) {
            if (i == 4) continue;
            ti[i] = {tex[i]->sampler, tex[i]->view, VK_IMAGE_LAYOUT_GENERAL};
            add(i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &ti[i], nullptr);
        }
        add(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &params);
        add(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &frame);
        // Set s writes history s and reads the other one.
        prev = {m_cloudHist[1 - s].sampler, m_cloudHist[1 - s].view, VK_IMAGE_LAYOUT_GENERAL};
        cur = {VK_NULL_HANDLE, m_cloudHist[s].view, VK_IMAGE_LAYOUT_GENERAL};
        out = {VK_NULL_HANDLE, m_cloudRasterOut.view, VK_IMAGE_LAYOUT_GENERAL};
        add(8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &prev, nullptr);
        add(9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &cur, nullptr);
        add(10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &out, nullptr);
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(ws.size()), ws.data(), 0, nullptr);
    }
    writeCloudRasterLutDescriptors();
    return true;
}

void VulkanDevice::recordCloudRaster(VkCommandBuffer cmd, const Backend::CloudRasterFrameGPU& frame) {
    if (!m_cloudRasterPipeline || !m_cloudRasterOut.view || cmd == VK_NULL_HANDLE) return;
    // Last frame's sky pass read the output and last frame's dispatch wrote
    // the histories: both must finish before this frame overwrites them.
    VkMemoryBarrier pre{};
    pre.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    pre.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    pre.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &pre, 0, nullptr, 0, nullptr);
    vkCmdUpdateBuffer(cmd, m_cloudRasterFrameBuf.buffer, 0, sizeof(frame), &frame);
    VkMemoryBarrier up{};
    up.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    up.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    up.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &up, 0, nullptr, 0, nullptr);
    const uint32_t s = m_cloudRasterParity;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudRasterPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudRasterLayout, 0, 1,
                            &m_cloudRasterSets[s], 0, nullptr);
    vkCmdDispatch(cmd, (m_cloudRasterW + 7u) / 8u, (m_cloudRasterH + 7u) / 8u, 1);
    if (m_cloudShadowPipeline) {
        // Same set (binding 11 + the frame block); independent of the march.
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_cloudShadowPipeline);
        vkCmdDispatch(cmd, (Backend::kCloudShadowMapSize + 7u) / 8u,
                      (Backend::kCloudShadowMapSize + 7u) / 8u, 1);
    }
    VkMemoryBarrier post{};
    post.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    post.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &post, 0, nullptr, 0, nullptr);
    m_cloudRasterParity = 1u - s;
}

} // namespace VulkanRT
