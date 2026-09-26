// Raster viewport particle billboards (particle roadmap Phase 1.5 Batch A).
//
// The pass has its OWN pipeline layout and descriptor set: set 0 binding 0 is
// the appearance LUT (storage buffer, vec4 pairs, see ParticleAppearanceProfile.h)
// and the vertex shader expands each camera-facing quad from the particle
// centre, reading colour / opacity / size / emission from (lut_row, age).
// Before this the pass borrowed the solid pipeline layout, had no descriptor
// set at all, and drew colours the CPU had already lerped — there was no
// channel through which a profile could reach the GPU.

#include "Backend/VulkanViewportBackend.h"
#include "globals.h"

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace Backend {
namespace {

// Must match shaders/particle_viewport.vert.
struct ParticlePushConstants {
    float viewProj[16];
    float view[16];
};
static_assert(sizeof(ParticlePushConstants) == 128,
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

VkShaderModule createModule(VkDevice device, const std::vector<uint32_t>& spv) {
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

} // namespace

// Called from ensureInteractiveViewportResourcesImpl. Creates the descriptor
// layout/pool/set, the pipeline layout and the additive + alpha pipelines.
void VulkanViewportBackend::ensureParticleBillboardPipelines(const std::string& shaderDir) {
    auto& iv = m_interactiveViewport;
    VkDevice vkDevice = m_device->getDevice();

    if (iv.particleDescLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding lut{};
        lut.binding = 0;
        lut.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lut.descriptorCount = 1;
        lut.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo dslci{};
        dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dslci.bindingCount = 1;
        dslci.pBindings = &lut;
        if (vkCreateDescriptorSetLayout(vkDevice, &dslci, nullptr, &iv.particleDescLayout) !=
            VK_SUCCESS) {
            SCENE_LOG_ERROR("[Particles] billboard descriptor layout creation failed");
            iv.particleDescLayout = VK_NULL_HANDLE;
            return;
        }
    }
    if (iv.particleDescPool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 1;
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
        // A buffer uploaded before the set existed still has to be bound.
        iv.particleLutDescriptorStale = true;
        writeParticleLutDescriptor();
    }
    if (iv.particlePipelineLayout == VK_NULL_HANDLE) {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        range.offset = 0;
        range.size = sizeof(ParticlePushConstants);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &iv.particleDescLayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &range;
        if (vkCreatePipelineLayout(vkDevice, &plci, nullptr, &iv.particlePipelineLayout) !=
            VK_SUCCESS) {
            iv.particlePipelineLayout = VK_NULL_HANDLE;
            return;
        }
    }
    if (iv.particleAddPipeline != VK_NULL_HANDLE) {
        return;
    }

    const std::string vertPath = shaderDir + "/particle_viewport.spv";
    const std::string fragPath = shaderDir + "/particle_viewport_frag.spv";
    if (!std::filesystem::exists(vertPath) || !std::filesystem::exists(fragPath)) {
        return;
    }
    VkShaderModule vert = createModule(vkDevice, loadParticleSPV(vertPath));
    VkShaderModule frag = createModule(vkDevice, loadParticleSPV(fragPath));
    if (vert && frag) {
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

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 3;
        vi.pVertexAttributeDescriptions = attribs;

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
        ds.depthWriteEnable = VK_FALSE;  // transparent: test against scene, don't occlude
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState cba{};
        cba.blendEnable = VK_TRUE;
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
        pi.layout = iv.particlePipelineLayout;
        pi.renderPass = iv.renderPass;
        pi.subpass = 0;

        // Additive: dst = ONE (colours accumulate -> glow).
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pi, nullptr,
                                  &iv.particleAddPipeline);

        // Alpha: dst = ONE_MINUS_SRC_ALPHA (standard transparency).
        cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pi, nullptr,
                                  &iv.particleAlphaPipeline);
    }
    if (vert) vkDestroyShaderModule(vkDevice, vert, nullptr);
    if (frag) vkDestroyShaderModule(vkDevice, frag, nullptr);
}

void VulkanViewportBackend::recordParticleBillboards(VkCommandBuffer cmd,
                                                     const float viewProjGL[16],
                                                     const float viewGL[16]) {
    auto& iv = m_interactiveViewport;
    if (iv.particlePipelineLayout == VK_NULL_HANDLE || iv.particleDescSet == VK_NULL_HANDLE ||
        !iv.particleLutBuffer.buffer || iv.particleLutDescriptorStale) {
        return;
    }
    ParticlePushConstants pc{};
    std::memcpy(pc.viewProj, viewProjGL, sizeof(pc.viewProj));
    std::memcpy(pc.view, viewGL, sizeof(pc.view));

    auto drawGroup = [&](VkPipeline pipeline, const VulkanRT::BufferHandle& vbuf,
                         uint32_t vcount) {
        if (pipeline == VK_NULL_HANDLE || !vbuf.buffer || vcount == 0) {
            return;
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, iv.particlePipelineLayout,
                                0, 1, &iv.particleDescSet, 0, nullptr);
        vkCmdPushConstants(cmd, iv.particlePipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0,
                           sizeof(ParticlePushConstants), &pc);
        VkBuffer b = vbuf.buffer;
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &b, &offset);
        vkCmdDraw(cmd, vcount, 1, 0, 0);
    };
    // Alpha first, additive on top.
    drawGroup(iv.particleAlphaPipeline, iv.particleAlphaVertexBuffer, iv.particleAlphaVertexCount);
    drawGroup(iv.particleAddPipeline, iv.particleAddVertexBuffer, iv.particleAddVertexCount);
}

void VulkanViewportBackend::destroyParticleBillboardResources(bool keepPipeline) {
    auto& iv = m_interactiveViewport;
    VkDevice vkDevice = m_device->getDevice();
    if (!keepPipeline) {
        if (iv.particleAddPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(vkDevice, iv.particleAddPipeline, nullptr);
            iv.particleAddPipeline = VK_NULL_HANDLE;
        }
        if (iv.particleAlphaPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(vkDevice, iv.particleAlphaPipeline, nullptr);
            iv.particleAlphaPipeline = VK_NULL_HANDLE;
        }
        if (iv.particlePipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(vkDevice, iv.particlePipelineLayout, nullptr);
            iv.particlePipelineLayout = VK_NULL_HANDLE;
        }
        if (iv.particleDescPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(vkDevice, iv.particleDescPool, nullptr);
            iv.particleDescPool = VK_NULL_HANDLE;
            iv.particleDescSet = VK_NULL_HANDLE;
        }
        if (iv.particleDescLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(vkDevice, iv.particleDescLayout, nullptr);
            iv.particleDescLayout = VK_NULL_HANDLE;
        }
    }
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
    // Forget what was uploaded so the next upload rewrites buffer + descriptor.
    iv.particleLutUploaded.clear();
    iv.particleLutDescriptorStale = true;
}

// ── Adapter side (the UI calls these on the base type) ──────────────────────

void VulkanBackendAdapter::writeParticleLutDescriptor() {
    auto& iv = m_interactiveViewport;
    if (!iv.particleLutDescriptorStale || iv.particleDescSet == VK_NULL_HANDLE ||
        !iv.particleLutBuffer.buffer) {
        return;
    }
    VkDescriptorBufferInfo info{};
    info.buffer = iv.particleLutBuffer.buffer;
    info.offset = 0;
    info.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet wds{};
    wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wds.dstSet = iv.particleDescSet;
    wds.dstBinding = 0;
    wds.descriptorCount = 1;
    wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wds.pBufferInfo = &info;
    vkUpdateDescriptorSets(m_device->getDevice(), 1, &wds, 0, nullptr);
    iv.particleLutDescriptorStale = false;
}

void VulkanBackendAdapter::uploadParticleBillboards(const ParticleBillboardUpload& data) {
    drainInteractiveViewportInFlight();
    std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_device) return;
    auto& iv = m_interactiveViewport;

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
    const uint32_t prevAlpha = iv.particleAlphaVertexCount;
    uploadGroup(data.additive, iv.particleAddVertexBuffer, iv.particleAddVertexCount);
    uploadGroup(data.alpha, iv.particleAlphaVertexBuffer, iv.particleAlphaVertexCount);

    // The LUT changes only when a profile is edited (or systems come and go):
    // re-upload on content change, not every frame.
    if (!data.lut.empty() && data.lut != iv.particleLutUploaded) {
        const uint64_t bytes = data.lut.size() * sizeof(float);
        if (ensureBuffer(iv.particleLutBuffer, bytes, VulkanRT::BufferUsage::STORAGE)) {
            iv.particleLutDescriptorStale = true;
        }
        if (iv.particleLutBuffer.buffer) {
            m_device->uploadBuffer(iv.particleLutBuffer, data.lut.data(), bytes, 0);
            iv.particleLutUploaded = data.lut;
        }
        writeParticleLutDescriptor();
    }

    // Re-render only when there is something to show or something just cleared.
    if (iv.particleAddVertexCount || iv.particleAlphaVertexCount || prevAdd || prevAlpha) {
        iv.dirty = true;
        m_currentSamples = 0;
    }
}

} // namespace Backend
