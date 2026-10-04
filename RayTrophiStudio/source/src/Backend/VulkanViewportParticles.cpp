// Raster viewport particle billboards (particle roadmap Phase 1.5).
//
// The pass has its OWN pipeline layouts and descriptor sets:
//   set 0 = appearance LUT (binding 0) + profile-id -> row lookup (binding 1)
//   set 1 = one device-resident system's simulation buffers (pull only)
// Two vertex shaders share the LUT code (shaders/include/particle_appearance_lut.glsl):
//   particle_viewport.vert      - CPU-built quads (host-state systems, fluid domains)
//   particle_viewport_pull.vert - vertex pulling straight from the simulation's
//                                 storage buffers (Batch B, same VkDevice only)
// Before Batch A the pass borrowed the solid pipeline layout, had no descriptor
// set at all, and drew colours the CPU had already lerped.

#include "Backend/VulkanViewportBackend.h"
#include "globals.h"
#include "Viewport/SphereImpostorAvailability.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace Backend {
namespace {

// Must match the push constant block in shaders/include/particle_appearance_lut.glsl.
struct ParticlePushConstants {
    float viewProj[16];
    float cameraRight[4];
    float cameraUp[4];
    uint32_t draw[4];  // lookup offset, lookup count, particle count, blend pass
};
struct SpherePushConstants {
    float viewProj[16];
    float view[16];
    int useMatcap;
    float overrides[9];
};
struct FluidSpherePushConstants {
    SpherePushConstants sphere;
    float childRadius;
    float spreadRadius;
    uint32_t parentCount;
    uint32_t childrenPerParent;
    float sizeVariation;
};
static_assert(sizeof(SpherePushConstants) == 168, "Must match solid pipeline layout");
static_assert(sizeof(FluidSpherePushConstants) == 188,
              "Must match fluid_sphere_proxy.vert push constants");
static_assert(sizeof(ParticlePushConstants) == 112,
              "particle push constants must stay within the guaranteed 128 bytes");

std::vector<uint32_t> loadParticleSPV(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0 || (size % static_cast<std::streamsize>(sizeof(uint32_t))) != 0) return {};
    std::vector<uint32_t> buffer(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) return {};
    return buffer;
}

VkShaderModule createModule(VkDevice device, const std::string& path) {
    const std::vector<uint32_t> spv = loadParticleSPV(path);
    if (spv.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spv.size() * sizeof(uint32_t);
    smci.pCode = spv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return module;
}

VkBuffer toVkBuffer(uint64_t handle) {
    return reinterpret_cast<VkBuffer>(static_cast<uintptr_t>(handle));
}

VkDescriptorSetLayout createStorageLayout(VkDevice device, uint32_t bindingCount) {
    std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
    for (uint32_t i = 0; i < bindingCount; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = bindingCount;
    dslci.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(device, &dslci, nullptr, &layout) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return layout;
}

VkPipelineLayout createParticlePipelineLayout(VkDevice device,
                                              const VkDescriptorSetLayout* sets,
                                              uint32_t setCount) {
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    range.offset = 0;
    range.size = sizeof(ParticlePushConstants);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = setCount;
    plci.pSetLayouts = sets;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(device, &plci, nullptr, &layout) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return layout;
}

VkPipelineLayout createFluidSpherePipelineLayout(
    VkDevice device,
    const VkDescriptorSetLayout sets[2]) {
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    range.offset = 0;
    range.size = sizeof(FluidSpherePushConstants);
    VkPipelineLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.setLayoutCount = 2;
    info.pSetLayouts = sets;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &range;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    if (vkCreatePipelineLayout(device, &info, nullptr, &layout) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return layout;
}

// Builds the additive and alpha variants of one billboard pipeline. With
// `quadVertexInput` the pipeline reads ParticleBillboardVertex; without it the
// vertex shader pulls everything from storage buffers.
void createBillboardPipelines(VkDevice device, VkRenderPass renderPass, VkPipelineLayout layout,
                              VkShaderModule vert, VkShaderModule frag, bool quadVertexInput,
                              VkPipeline& additiveOut, VkPipeline& alphaOut,
                              bool sphereInput = false) {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    // ParticleBillboardVertex: centre.xyz, corner.xy, age, lut_row, size_scale.
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(ParticleBillboardVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attribs[3]{};
    attribs[0].location = 0;
    attribs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attribs[0].offset = offsetof(ParticleBillboardVertex, center);
    attribs[1].location = 1;
    attribs[1].format = VK_FORMAT_R32G32_SFLOAT;
    attribs[1].offset = offsetof(ParticleBillboardVertex, corner);
    attribs[2].location = 2;
    attribs[2].format = VK_FORMAT_R32G32B32_SFLOAT;  // age, lut_row, size_scale
    attribs[2].offset = offsetof(ParticleBillboardVertex, age);
    if (sphereInput) {
        binding.stride = sizeof(SphereImpostorInstance);
        binding.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;
        attribs[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attribs[0].offset = 0;
    }

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (quadVertexInput) {
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = sphereInput ? 1 : 3;
        vi.pVertexAttributeDescriptions = attribs;
    }

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo ras{};
    ras.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    ras.polygonMode = VK_POLYGON_MODE_FILL;
    ras.lineWidth = 1.0f;
    ras.cullMode = VK_CULL_MODE_NONE;
    ras.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE;
    ds.depthWriteEnable = sphereInput ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable = sphereInput ? VK_FALSE : VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.colorBlendOp = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.alphaBlendOp = VK_BLEND_OP_ADD;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &cba;

    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates = dyn;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.stageCount = 2;
    pi.pStages = stages;
    pi.pVertexInputState = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp;
    pi.pRasterizationState = &ras;
    pi.pMultisampleState = &ms;
    pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb;
    pi.pDynamicState = &dynState;
    pi.layout = layout;
    pi.renderPass = renderPass;
    pi.subpass = 0;

    // Additive: dst = ONE (colours accumulate -> glow).
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &additiveOut);
    if (sphereInput) {
        return;
    }

    // Alpha: dst = ONE_MINUS_SRC_ALPHA (standard transparency).
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &alphaOut);
}

} // namespace

// Called from ensureInteractiveViewportResourcesImpl. Creates the descriptor
// layouts/pool/set, both pipeline layouts and the quad + pull pipelines.
void VulkanViewportBackend::ensureParticleBillboardPipelines(const std::string& shaderDir) {
    auto& iv = m_interactiveViewport;
    VkDevice vkDevice = m_device->getDevice();
    if (iv.sphereImpostorPipeline == VK_NULL_HANDLE && iv.pipelineLayout) {
        VkShaderModule vert = createModule(vkDevice, shaderDir + "/sphere_impostor.spv");
        VkShaderModule frag = createModule(vkDevice, shaderDir + "/sphere_impostor_frag.spv");
        if (vert && frag) {
            VkPipeline unused = VK_NULL_HANDLE;
            createBillboardPipelines(vkDevice, iv.renderPass, iv.pipelineLayout, vert, frag,
                                     true, iv.sphereImpostorPipeline, unused, true);
        }
        if (vert) vkDestroyShaderModule(vkDevice, vert, nullptr);
        if (frag) vkDestroyShaderModule(vkDevice, frag, nullptr);
        if (iv.sphereImpostorPipeline) {
            g_sphere_impostor_ready = true;
        } else {
            static bool warned = false;
            if (!warned) {
                SCENE_LOG_WARN("[Particles] Sphere impostor pipeline unavailable; "
                               "using geometry fallback. Compile sphere_impostor shaders.");
                warned = true;
            }
        }
    }

    if (iv.fluidSphereProxyDescLayout == VK_NULL_HANDLE) {
        iv.fluidSphereProxyDescLayout = createStorageLayout(vkDevice, 1);
    }
    if (iv.fluidSphereProxyPipelineLayout == VK_NULL_HANDLE &&
        iv.matcapDescLayout != VK_NULL_HANDLE &&
        iv.fluidSphereProxyDescLayout != VK_NULL_HANDLE) {
        const VkDescriptorSetLayout sets[2] = {
            iv.matcapDescLayout, iv.fluidSphereProxyDescLayout};
        iv.fluidSphereProxyPipelineLayout =
            createFluidSpherePipelineLayout(vkDevice, sets);
    }
    if (iv.fluidSphereProxyPipeline == VK_NULL_HANDLE &&
        iv.fluidSphereProxyPipelineLayout != VK_NULL_HANDLE) {
        VkShaderModule vert = createModule(vkDevice, shaderDir + "/fluid_sphere_proxy.spv");
        VkShaderModule frag = createModule(vkDevice, shaderDir + "/sphere_impostor_frag.spv");
        if (vert && frag) {
            VkPipeline unused = VK_NULL_HANDLE;
            createBillboardPipelines(
                vkDevice, iv.renderPass, iv.fluidSphereProxyPipelineLayout,
                vert, frag, false, iv.fluidSphereProxyPipeline, unused, true);
        }
        if (vert) vkDestroyShaderModule(vkDevice, vert, nullptr);
        if (frag) vkDestroyShaderModule(vkDevice, frag, nullptr);
        if (iv.fluidSphereProxyPipeline == VK_NULL_HANDLE) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                SCENE_LOG_WARN(
                    "[Particles] fluid_sphere_proxy.spv unavailable; using CPU sphere upload.");
            }
        }
    }
    g_fluid_sphere_proxy_ready = iv.fluidSphereProxyPipeline != VK_NULL_HANDLE;

    if (iv.particleDescLayout == VK_NULL_HANDLE) {
        iv.particleDescLayout = createStorageLayout(vkDevice, 2);  // LUT, row lookup
        if (iv.particleDescLayout == VK_NULL_HANDLE) {
            SCENE_LOG_ERROR("[Particles] billboard descriptor layout creation failed");
            return;
        }
    }
    if (iv.particlePullDescLayout == VK_NULL_HANDLE) {
        iv.particlePullDescLayout = createStorageLayout(vkDevice, kPulledStreamCount);
    }
    if (iv.particleDescPool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 2;
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &poolSize;
        dpci.maxSets = 1;
        if (vkCreateDescriptorPool(vkDevice, &dpci, nullptr, &iv.particleDescPool) != VK_SUCCESS) {
            iv.particleDescPool = VK_NULL_HANDLE;
            return;
        }
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool = iv.particleDescPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &iv.particleDescLayout;
        if (vkAllocateDescriptorSets(vkDevice, &dsai, &iv.particleDescSet) != VK_SUCCESS) {
            iv.particleDescSet = VK_NULL_HANDLE;
            return;
        }
        // Buffers uploaded before the set existed still have to be bound.
        iv.particleLutDescriptorStale = true;
        writeParticleLutDescriptor();
    }
    if (iv.particlePipelineLayout == VK_NULL_HANDLE) {
        iv.particlePipelineLayout =
            createParticlePipelineLayout(vkDevice, &iv.particleDescLayout, 1);
        if (iv.particlePipelineLayout == VK_NULL_HANDLE) return;
    }
    if (iv.particlePullPipelineLayout == VK_NULL_HANDLE &&
        iv.particlePullDescLayout != VK_NULL_HANDLE) {
        const VkDescriptorSetLayout sets[2] = {iv.particleDescLayout, iv.particlePullDescLayout};
        iv.particlePullPipelineLayout = createParticlePipelineLayout(vkDevice, sets, 2);
    }

    const bool needQuad = iv.particleAddPipeline == VK_NULL_HANDLE;
    const bool needPull = iv.particlePullAddPipeline == VK_NULL_HANDLE &&
                          iv.particlePullPipelineLayout != VK_NULL_HANDLE;
    if (!needQuad && !needPull) {
        return;
    }
    VkShaderModule frag = createModule(vkDevice, shaderDir + "/particle_viewport_frag.spv");
    if (frag == VK_NULL_HANDLE) {
        return;
    }
    if (needQuad) {
        VkShaderModule vert = createModule(vkDevice, shaderDir + "/particle_viewport.spv");
        if (vert != VK_NULL_HANDLE) {
            createBillboardPipelines(vkDevice, iv.renderPass, iv.particlePipelineLayout, vert, frag,
                                     /*quadVertexInput=*/true, iv.particleAddPipeline,
                                     iv.particleAlphaPipeline);
            vkDestroyShaderModule(vkDevice, vert, nullptr);
        }
    }
    if (needPull) {
        VkShaderModule vert = createModule(vkDevice, shaderDir + "/particle_viewport_pull.spv");
        if (vert != VK_NULL_HANDLE) {
            createBillboardPipelines(vkDevice, iv.renderPass, iv.particlePullPipelineLayout, vert,
                                     frag, /*quadVertexInput=*/false, iv.particlePullAddPipeline,
                                     iv.particlePullAlphaPipeline);
            vkDestroyShaderModule(vkDevice, vert, nullptr);
        } else {
            // Without it every device-resident system is invisible in the
            // viewport while its simulation runs fine — say so, once.
            static bool warned = false;
            if (!warned) {
                warned = true;
                SCENE_LOG_WARN("[Particles] particle_viewport_pull.spv missing: device-resident "
                               "particle systems cannot be drawn (run compile_shaders.bat)");
            }
        }
    }
    vkDestroyShaderModule(vkDevice, frag, nullptr);
}

void VulkanViewportBackend::recordParticleBillboards(VkCommandBuffer cmd,
                                                     const float viewProjGL[16],
                                                     const float viewGL[16]) {
    auto& iv = m_interactiveViewport;
    m_sphereImpostorsDrawn = 0;
    if (iv.fluidSphereProxyPipeline && iv.fluidSphereProxyPipelineLayout &&
        iv.matcapDescSet) {
        bool bound = false;
        const std::size_t count = std::min(
            iv.fluidSphereProxyDraws.size(), iv.fluidSphereProxySlots.size());
        for (std::size_t i = 0; i < count; ++i) {
            const FluidSphereProxyDraw& draw = iv.fluidSphereProxyDraws[i];
            const VkDescriptorSet proxySet = iv.fluidSphereProxySlots[i].set;
            if (!proxySet || draw.parent_count == 0 ||
                draw.children_per_parent == 0 || !(draw.child_radius > 0.0f)) {
                continue;
            }
            if (!bound) {
                vkCmdBindPipeline(
                    cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, iv.fluidSphereProxyPipeline);
                bound = true;
            }
            const VkDescriptorSet sets[2] = {iv.matcapDescSet, proxySet};
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                iv.fluidSphereProxyPipelineLayout, 0, 2, sets, 0, nullptr);
            FluidSpherePushConstants pc{};
            std::memcpy(pc.sphere.viewProj, viewProjGL, sizeof(pc.sphere.viewProj));
            std::memcpy(pc.sphere.view, viewGL, sizeof(pc.sphere.view));
            pc.sphere.useMatcap = m_viewportMode == ViewportMode::Matcap
                ? (iv.matcapUserLoaded ? 1 : iv.matcapPreset) : 0;
            pc.childRadius = draw.child_radius;
            pc.spreadRadius = draw.spread_radius;
            pc.parentCount = draw.parent_count;
            pc.childrenPerParent = draw.children_per_parent;
            pc.sizeVariation = draw.size_variation;
            vkCmdPushConstants(
                cmd, iv.fluidSphereProxyPipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(pc), &pc);
            vkCmdDraw(cmd, 6, draw.parent_count * draw.children_per_parent, 0, 0);
            m_sphereImpostorsDrawn +=
                static_cast<uint64_t>(draw.parent_count) * draw.children_per_parent;
        }
    }
    if (iv.sphereImpostorPipeline && iv.sphereImpostorBuffer.buffer &&
        iv.matcapDescSet && !iv.sphereImpostorsUploaded.empty() &&
        (m_viewportMode == ViewportMode::Solid || m_viewportMode == ViewportMode::Matcap)) {
        SpherePushConstants sphere{};
        std::memcpy(sphere.viewProj, viewProjGL, sizeof(sphere.viewProj));
        std::memcpy(sphere.view, viewGL, sizeof(sphere.view));
        sphere.useMatcap = m_viewportMode == ViewportMode::Matcap
            ? (iv.matcapUserLoaded ? 1 : iv.matcapPreset) : 0;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, iv.sphereImpostorPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, iv.pipelineLayout,
                                0, 1, &iv.matcapDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, iv.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT |
                           VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_GEOMETRY_BIT,
                           0, sizeof(sphere), &sphere);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &iv.sphereImpostorBuffer.buffer, &offset);
        vkCmdDraw(cmd, 6, static_cast<uint32_t>(iv.sphereImpostorsUploaded.size()), 0, 0);
        m_sphereImpostorsDrawn += iv.sphereImpostorsUploaded.size();
    }
    if (iv.particlePipelineLayout == VK_NULL_HANDLE || iv.particleDescSet == VK_NULL_HANDLE ||
        !iv.particleLutBuffer.buffer || !iv.particleRowLookupBuffer.buffer ||
        iv.particleLutDescriptorStale) {
        return;
    }
    ParticlePushConstants pc{};
    std::memcpy(pc.viewProj, viewProjGL, sizeof(pc.viewProj));
    // Camera right / up = first two rows of the view rotation (column-major GL).
    pc.cameraRight[0] = viewGL[0];
    pc.cameraRight[1] = viewGL[4];
    pc.cameraRight[2] = viewGL[8];
    pc.cameraUp[0] = viewGL[1];
    pc.cameraUp[1] = viewGL[5];
    pc.cameraUp[2] = viewGL[9];

    auto drawQuads = [&](VkPipeline pipeline, const VulkanRT::BufferHandle& vbuf,
                         uint32_t vcount) {
        if (pipeline == VK_NULL_HANDLE || !vbuf.buffer || vcount == 0) {
            return;
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, iv.particlePipelineLayout,
                                0, 1, &iv.particleDescSet, 0, nullptr);
        pc.draw[0] = pc.draw[1] = pc.draw[2] = pc.draw[3] = 0u;
        vkCmdPushConstants(cmd, iv.particlePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(ParticlePushConstants), &pc);
        VkBuffer b = vbuf.buffer;
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &b, &offset);
        vkCmdDraw(cmd, vcount, 1, 0, 0);
    };
    auto drawPulled = [&](VkPipeline pipeline, bool alphaPass) {
        if (pipeline == VK_NULL_HANDLE || iv.particlePullPipelineLayout == VK_NULL_HANDLE) {
            return;
        }
        bool bound = false;
        const std::size_t n = std::min(iv.particlePulledDraws.size(), iv.particlePullSlots.size());
        for (std::size_t i = 0; i < n; ++i) {
            const ParticlePulledDraw& d = iv.particlePulledDraws[i];
            const VkDescriptorSet set1 = iv.particlePullSlots[i].set;
            if (d.particle_count == 0 || set1 == VK_NULL_HANDLE || (alphaPass && !d.has_alpha)) {
                continue;
            }
            if (!bound) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                bound = true;
            }
            const VkDescriptorSet sets[2] = {iv.particleDescSet, set1};
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    iv.particlePullPipelineLayout, 0, 2, sets, 0, nullptr);
            pc.draw[0] = d.lookup_offset;
            pc.draw[1] = d.lookup_count;
            pc.draw[2] = d.particle_count;
            pc.draw[3] = alphaPass ? 1u : 0u;
            vkCmdPushConstants(cmd, iv.particlePullPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                               sizeof(ParticlePushConstants), &pc);
            vkCmdDraw(cmd, d.particle_count * 6u, 1, 0, 0);
        }
    };
    // Alpha first, additive on top.
    drawQuads(iv.particleAlphaPipeline, iv.particleAlphaVertexBuffer, iv.particleAlphaVertexCount);
    drawPulled(iv.particlePullAlphaPipeline, /*alphaPass=*/true);
    drawQuads(iv.particleAddPipeline, iv.particleAddVertexBuffer, iv.particleAddVertexCount);
    drawPulled(iv.particlePullAddPipeline, /*alphaPass=*/false);
}

void VulkanViewportBackend::destroyParticleBillboardResources(bool keepPipeline) {
    auto& iv = m_interactiveViewport;
    VkDevice vkDevice = m_device->getDevice();
    if (iv.sphereImpostorBuffer.buffer) {
        m_device->destroyBuffer(iv.sphereImpostorBuffer);
    }
    iv.sphereImpostorsUploaded.clear();
    if (!keepPipeline) {
        g_sphere_impostor_ready = false;
        g_viewport_raster_rebuild_pending = true;
        if (iv.sphereImpostorPipeline) {
            vkDestroyPipeline(vkDevice, iv.sphereImpostorPipeline, nullptr);
            iv.sphereImpostorPipeline = VK_NULL_HANDLE;
        }
        if (iv.fluidSphereProxyPipeline) {
            vkDestroyPipeline(vkDevice, iv.fluidSphereProxyPipeline, nullptr);
            iv.fluidSphereProxyPipeline = VK_NULL_HANDLE;
        }
        g_fluid_sphere_proxy_ready = false;
        if (iv.fluidSphereProxyPipelineLayout) {
            vkDestroyPipelineLayout(
                vkDevice, iv.fluidSphereProxyPipelineLayout, nullptr);
            iv.fluidSphereProxyPipelineLayout = VK_NULL_HANDLE;
        }
        if (iv.fluidSphereProxyDescPool) {
            vkDestroyDescriptorPool(vkDevice, iv.fluidSphereProxyDescPool, nullptr);
            iv.fluidSphereProxyDescPool = VK_NULL_HANDLE;
        }
        if (iv.fluidSphereProxyDescLayout) {
            vkDestroyDescriptorSetLayout(
                vkDevice, iv.fluidSphereProxyDescLayout, nullptr);
            iv.fluidSphereProxyDescLayout = VK_NULL_HANDLE;
        }
        iv.fluidSphereProxyDescCapacity = 0;
        iv.fluidSphereProxySlots.clear();
    }
    if (!keepPipeline) {
        for (VkPipeline* p : {&iv.particleAddPipeline, &iv.particleAlphaPipeline,
                              &iv.particlePullAddPipeline, &iv.particlePullAlphaPipeline}) {
            if (*p != VK_NULL_HANDLE) {
                vkDestroyPipeline(vkDevice, *p, nullptr);
                *p = VK_NULL_HANDLE;
            }
        }
        for (VkPipelineLayout* l : {&iv.particlePipelineLayout, &iv.particlePullPipelineLayout}) {
            if (*l != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(vkDevice, *l, nullptr);
                *l = VK_NULL_HANDLE;
            }
        }
        if (iv.particleDescPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(vkDevice, iv.particleDescPool, nullptr);
            iv.particleDescPool = VK_NULL_HANDLE;
            iv.particleDescSet = VK_NULL_HANDLE;
        }
        if (iv.particlePullDescPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(vkDevice, iv.particlePullDescPool, nullptr);
            iv.particlePullDescPool = VK_NULL_HANDLE;
        }
        iv.particlePullDescCapacity = 0;
        iv.particlePullSlots.clear();
        for (VkDescriptorSetLayout* l : {&iv.particleDescLayout, &iv.particlePullDescLayout}) {
            if (*l != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(vkDevice, *l, nullptr);
                *l = VK_NULL_HANDLE;
            }
        }
    }
    // Pull descriptor sets point at SIMULATION buffers, which a viewport
    // resize does not touch; they survive keepPipeline. The draw list does not:
    // the next upload re-states it.
    iv.particlePulledDraws.clear();
    iv.fluidSphereProxyDraws.clear();
    if (iv.particleAddVertexBuffer.buffer) {
        m_device->destroyBuffer(iv.particleAddVertexBuffer);
    }
    iv.particleAddVertexCount = 0;
    if (iv.particleAlphaVertexBuffer.buffer) {
        m_device->destroyBuffer(iv.particleAlphaVertexBuffer);
    }
    iv.particleAlphaVertexCount = 0;
    if (iv.particleLutBuffer.buffer) {
        m_device->destroyBuffer(iv.particleLutBuffer);
    }
    if (iv.particleRowLookupBuffer.buffer) {
        m_device->destroyBuffer(iv.particleRowLookupBuffer);
    }
    // Forget what was uploaded so the next upload rewrites buffers + descriptor.
    iv.particleLutUploaded.clear();
    iv.particleRowLookupUploaded.clear();
    iv.particleLutDescriptorStale = true;
}

// ── Adapter side (the UI calls these on the base type) ──────────────────────

void VulkanBackendAdapter::writeParticleLutDescriptor() {
    auto& iv = m_interactiveViewport;
    if (!iv.particleLutDescriptorStale || iv.particleDescSet == VK_NULL_HANDLE ||
        !iv.particleLutBuffer.buffer || !iv.particleRowLookupBuffer.buffer) {
        return;
    }
    VkDescriptorBufferInfo infos[2]{};
    infos[0].buffer = iv.particleLutBuffer.buffer;
    infos[0].range = VK_WHOLE_SIZE;
    infos[1].buffer = iv.particleRowLookupBuffer.buffer;
    infos[1].range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wds[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = iv.particleDescSet;
        wds[i].dstBinding = i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(m_device->getDevice(), 2, wds, 0, nullptr);
    iv.particleLutDescriptorStale = false;
}

void VulkanBackendAdapter::uploadParticleBillboards(const ParticleBillboardUpload& data) {
    // Also what makes the pull descriptor rewrites below legal: no frame that
    // uses those sets is in flight after this.
    drainInteractiveViewportInFlight();
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;
    auto& iv = m_interactiveViewport;
    VkDevice vkDevice = m_device->getDevice();

    // Grow-only; reallocates only when the current buffer is too small.
    auto ensureBuffer = [&](VulkanRT::BufferHandle& buffer, uint64_t bytes,
                            VulkanRT::BufferUsage usage) -> bool {
        if (buffer.buffer && buffer.size >= bytes) {
            return false;
        }
        if (buffer.buffer) {
            m_device->destroyBuffer(buffer);
        }
        VulkanRT::BufferCreateInfo bci{};
        bci.size = bytes;
        bci.usage = usage | VulkanRT::BufferUsage::TRANSFER_DST;
        bci.location = VulkanRT::MemoryLocation::GPU_ONLY;
        buffer = m_device->createBuffer(bci);
        return true;
    };

    auto uploadGroup = [&](const std::vector<ParticleBillboardVertex>& verts,
                           VulkanRT::BufferHandle& buffer, uint32_t& countOut) {
        if (verts.empty()) {
            countOut = 0;
            return;
        }
        const uint64_t bytes = verts.size() * sizeof(ParticleBillboardVertex);
        ensureBuffer(buffer, bytes, VulkanRT::BufferUsage::VERTEX);
        if (buffer.buffer) {
            m_device->uploadBuffer(buffer, verts.data(), bytes, 0);
            countOut = static_cast<uint32_t>(verts.size());
        } else {
            countOut = 0;
        }
    };

    const uint32_t prevAdd = iv.particleAddVertexCount;
    const bool spheresChanged = data.spheres.size() != iv.sphereImpostorsUploaded.size() ||
        (!data.spheres.empty() && std::memcmp(data.spheres.data(),
            iv.sphereImpostorsUploaded.data(),
            data.spheres.size() * sizeof(SphereImpostorInstance)) != 0);
    if (spheresChanged) {
        iv.sphereImpostorsUploaded.clear();
        if (!data.spheres.empty()) {
            const uint64_t bytes = data.spheres.size() * sizeof(SphereImpostorInstance);
            ensureBuffer(iv.sphereImpostorBuffer, bytes, VulkanRT::BufferUsage::VERTEX);
            if (iv.sphereImpostorBuffer.buffer) {
                m_device->uploadBuffer(iv.sphereImpostorBuffer, data.spheres.data(), bytes, 0);
                iv.sphereImpostorsUploaded = data.spheres;
            } else {
                g_sphere_impostor_ready = false;
                g_viewport_raster_rebuild_pending = true;
                SCENE_LOG_WARN("[Particles] Sphere instance buffer unavailable; geometry fallback.");
            }
        }
    }
    bool fluidProxyChanged =
        data.fluid_sphere_proxies.size() != iv.fluidSphereProxyDraws.size();
    for (std::size_t i = 0;
         !fluidProxyChanged && i < data.fluid_sphere_proxies.size();
         ++i) {
        const auto& incoming = data.fluid_sphere_proxies[i];
        const auto& current = iv.fluidSphereProxyDraws[i];
        fluidProxyChanged =
            incoming.position_buffer != current.position_buffer ||
            incoming.parent_count != current.parent_count ||
            incoming.children_per_parent != current.children_per_parent ||
            incoming.child_radius != current.child_radius ||
            incoming.spread_radius != current.spread_radius ||
            incoming.state_version != current.state_version;
    }
    iv.fluidSphereProxyDraws = data.fluid_sphere_proxies;
    if (!data.fluid_sphere_proxies.empty() &&
        iv.fluidSphereProxyDescLayout != VK_NULL_HANDLE) {
        const uint32_t needed =
            static_cast<uint32_t>(data.fluid_sphere_proxies.size());
        if (needed > iv.fluidSphereProxyDescCapacity) {
            if (iv.fluidSphereProxyDescPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(
                    vkDevice, iv.fluidSphereProxyDescPool, nullptr);
                iv.fluidSphereProxyDescPool = VK_NULL_HANDLE;
            }
            iv.fluidSphereProxySlots.clear();
            iv.fluidSphereProxyDescCapacity = 0;
            uint32_t capacity = 4;
            while (capacity < needed) capacity *= 2;
            VkDescriptorPoolSize poolSize{};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = capacity;
            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes = &poolSize;
            poolInfo.maxSets = capacity;
            if (vkCreateDescriptorPool(
                    vkDevice, &poolInfo, nullptr,
                    &iv.fluidSphereProxyDescPool) == VK_SUCCESS) {
                std::vector<VkDescriptorSetLayout> layouts(
                    capacity, iv.fluidSphereProxyDescLayout);
                std::vector<VkDescriptorSet> sets(capacity, VK_NULL_HANDLE);
                VkDescriptorSetAllocateInfo alloc{};
                alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                alloc.descriptorPool = iv.fluidSphereProxyDescPool;
                alloc.descriptorSetCount = capacity;
                alloc.pSetLayouts = layouts.data();
                if (vkAllocateDescriptorSets(vkDevice, &alloc, sets.data()) == VK_SUCCESS) {
                    iv.fluidSphereProxySlots.resize(capacity);
                    for (uint32_t i = 0; i < capacity; ++i) {
                        iv.fluidSphereProxySlots[i].set = sets[i];
                    }
                    iv.fluidSphereProxyDescCapacity = capacity;
                }
            }
        }
        const std::size_t proxyCount = std::min(
            data.fluid_sphere_proxies.size(), iv.fluidSphereProxySlots.size());
        for (std::size_t i = 0; i < proxyCount; ++i) {
            auto& slot = iv.fluidSphereProxySlots[i];
            const uint64_t buffer = data.fluid_sphere_proxies[i].position_buffer;
            if (slot.positionBuffer == buffer || !slot.set || buffer == 0) continue;
            VkDescriptorBufferInfo info{};
            info.buffer = toVkBuffer(buffer);
            info.range = VK_WHOLE_SIZE;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = slot.set;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &info;
            vkUpdateDescriptorSets(vkDevice, 1, &write, 0, nullptr);
            slot.positionBuffer = buffer;
        }
    }
    const uint32_t prevAlpha = iv.particleAlphaVertexCount;
    uploadGroup(data.additive, iv.particleAddVertexBuffer, iv.particleAddVertexCount);
    uploadGroup(data.alpha, iv.particleAlphaVertexBuffer, iv.particleAlphaVertexCount);

    // The LUT and the row lookup change only when a profile is edited (or
    // systems come and go): re-upload on content change, not every frame.
    auto uploadTable = [&](const auto& content, VulkanRT::BufferHandle& buffer, auto& uploaded) {
        if (content.empty() || content == uploaded) {
            return;
        }
        const uint64_t bytes = content.size() * sizeof(content[0]);
        if (ensureBuffer(buffer, bytes, VulkanRT::BufferUsage::STORAGE)) {
            iv.particleLutDescriptorStale = true;
        }
        if (buffer.buffer) {
            m_device->uploadBuffer(buffer, content.data(), bytes, 0);
            uploaded = content;
        }
    };
    uploadTable(data.lut, iv.particleLutBuffer, iv.particleLutUploaded);
    uploadTable(data.row_lookup, iv.particleRowLookupBuffer, iv.particleRowLookupUploaded);
    writeParticleLutDescriptor();

    // ── Vertex pulling: one set-1 descriptor set per device-resident system ──
    bool pulledChanged = data.pulled.size() != iv.particlePulledDraws.size();
    for (std::size_t i = 0; !pulledChanged && i < data.pulled.size(); ++i) {
        pulledChanged = data.pulled[i].state_version != iv.particlePulledDraws[i].state_version ||
                        data.pulled[i].particle_count != iv.particlePulledDraws[i].particle_count;
    }
    iv.particlePulledDraws = data.pulled;
    if (!data.pulled.empty() && iv.particlePullDescLayout != VK_NULL_HANDLE) {
        const uint32_t needed = static_cast<uint32_t>(data.pulled.size());
        if (needed > iv.particlePullDescCapacity) {
            if (iv.particlePullDescPool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(vkDevice, iv.particlePullDescPool, nullptr);
                iv.particlePullDescPool = VK_NULL_HANDLE;
            }
            iv.particlePullSlots.clear();
            iv.particlePullDescCapacity = 0;
            uint32_t capacity = 8;
            while (capacity < needed) capacity *= 2;
            VkDescriptorPoolSize poolSize{};
            poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSize.descriptorCount = capacity * kPulledStreamCount;
            VkDescriptorPoolCreateInfo dpci{};
            dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dpci.poolSizeCount = 1;
            dpci.pPoolSizes = &poolSize;
            dpci.maxSets = capacity;
            if (vkCreateDescriptorPool(vkDevice, &dpci, nullptr, &iv.particlePullDescPool) ==
                VK_SUCCESS) {
                std::vector<VkDescriptorSetLayout> layouts(capacity, iv.particlePullDescLayout);
                std::vector<VkDescriptorSet> sets(capacity, VK_NULL_HANDLE);
                VkDescriptorSetAllocateInfo dsai{};
                dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dsai.descriptorPool = iv.particlePullDescPool;
                dsai.descriptorSetCount = capacity;
                dsai.pSetLayouts = layouts.data();
                if (vkAllocateDescriptorSets(vkDevice, &dsai, sets.data()) == VK_SUCCESS) {
                    iv.particlePullSlots.resize(capacity);
                    for (uint32_t s = 0; s < capacity; ++s) iv.particlePullSlots[s].set = sets[s];
                    iv.particlePullDescCapacity = capacity;
                }
            } else {
                iv.particlePullDescPool = VK_NULL_HANDLE;
            }
        }
        const std::size_t n = std::min(data.pulled.size(), iv.particlePullSlots.size());
        for (std::size_t i = 0; i < n; ++i) {
            auto& slot = iv.particlePullSlots[i];
            const ParticlePulledDraw& d = data.pulled[i];
            if (std::memcmp(slot.buffers, d.buffers, sizeof(slot.buffers)) == 0) {
                continue;
            }
            VkDescriptorBufferInfo infos[kPulledStreamCount]{};
            VkWriteDescriptorSet wds[kPulledStreamCount]{};
            for (uint32_t b = 0; b < kPulledStreamCount; ++b) {
                infos[b].buffer = toVkBuffer(d.buffers[b]);
                infos[b].range = VK_WHOLE_SIZE;
                wds[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wds[b].dstSet = slot.set;
                wds[b].dstBinding = b;
                wds[b].descriptorCount = 1;
                wds[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                wds[b].pBufferInfo = &infos[b];
            }
            vkUpdateDescriptorSets(vkDevice, kPulledStreamCount, wds, 0, nullptr);
            std::memcpy(slot.buffers, d.buffers, sizeof(slot.buffers));
        }
    }

    // Re-render only when there is something to show or something just cleared.
    if (iv.particleAddVertexCount || iv.particleAlphaVertexCount || prevAdd || prevAlpha ||
        pulledChanged || spheresChanged || fluidProxyChanged) {
        iv.dirty = true;
        m_currentSamples = 0;
    }
}

} // namespace Backend
