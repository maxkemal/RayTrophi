#include "Backend/VulkanBackend.h"
#include "globals.h"

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

extern RenderSettings render_settings;

namespace Backend {

class MaterialPreviewSdfSurfaceResources {
public:
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkBuffer boundVolumeBuffer = VK_NULL_HANDLE;
    bool missingShaderReported = false;
};

namespace {

std::vector<uint32_t> loadSdfSpv(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const std::streamsize size = file.tellg();
    if (size <= 0 || (size % 4) != 0) return {};
    std::vector<uint32_t> words(static_cast<size_t>(size) / sizeof(uint32_t));
    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(words.data()), size)) return {};
    return words;
}

void matrixToGL(const Matrix4x4& matrix, float out[16]) {
    const Matrix4x4 transposed = matrix.transpose();
    std::memcpy(out, transposed.m, sizeof(float) * 16u);
}

struct PreviewPush {
    float viewProj[16];
    float view[16];
    float cameraPos[4];
    float lightDir0[4];
    float lightDir1[4];
    float lightDir2[4];
    uint32_t materialMeta[4];
};
static_assert(sizeof(PreviewPush) == 208u, "Material preview push ABI changed");

} // namespace

void VulkanBackendAdapter::destroyMaterialPreviewSdfSurfaceResources() {
    if (!m_materialPreviewSdfSurface || !m_device) return;
    if (m_materialPreviewSdfSurface->pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device->getDevice(),
                          m_materialPreviewSdfSurface->pipeline, nullptr);
    }
    m_materialPreviewSdfSurface.reset();
}

bool VulkanBackendAdapter::ensureMaterialPreviewSdfSurfaceResources(
    const std::string& shaderDir) {
    if (!m_device || m_interactiveViewport.renderPass == VK_NULL_HANDLE ||
        m_interactiveViewport.materialPreviewPipelineLayout == VK_NULL_HANDLE ||
        m_interactiveViewport.materialPreviewDescSet == VK_NULL_HANDLE) return false;

    if (m_materialPreviewSdfSurface &&
        m_materialPreviewSdfSurface->pipeline != VK_NULL_HANDLE) {
        updateMaterialPreviewSdfSurfaceBinding();
        return true;
    }
    if (!m_materialPreviewSdfSurface) {
        m_materialPreviewSdfSurface =
            std::make_shared<MaterialPreviewSdfSurfaceResources>();
    }
    auto& state = *m_materialPreviewSdfSurface;

    const std::vector<uint32_t> vert =
        loadSdfSpv(shaderDir + "/material_preview_sky_vert.spv");
    const std::vector<uint32_t> frag =
        loadSdfSpv(shaderDir + "/material_preview_sdf_surface.spv");
    if (vert.empty() || frag.empty()) {
        if (!state.missingShaderReported) {
            state.missingShaderReported = true;
            SCENE_LOG_WARN(
                "[MaterialPreview] Realtime SurfaceSDF shader is missing; "
                "fluid SDF stays on the existing particle proxy until shaders are compiled.");
        }
        return false;
    }

    VkDevice device = m_device->getDevice();
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = vert.size() * sizeof(uint32_t);
    smci.pCode = vert.data();
    VkShaderModule vertModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smci, nullptr, &vertModule) != VK_SUCCESS)
        return false;
    smci.codeSize = frag.size() * sizeof(uint32_t);
    smci.pCode = frag.data();
    VkShaderModule fragModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smci, nullptr, &fragModule) != VK_SUCCESS) {
        vkDestroyShaderModule(device, vertModule, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragModule;
    stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    // The ray-marched depth is reconstructed through the inverse projection
    // and can differ by an ulp from coincident raster depth as the camera moves.
    // Equality is not occlusion; rejecting it made the complete SDF flicker out.
    depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &attachment;
    const VkDynamicState dynamicStates[2] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount = 2;
    pci.pStages = stages;
    pci.pVertexInputState = &vertexInput;
    pci.pInputAssemblyState = &assembly;
    pci.pViewportState = &viewport;
    pci.pRasterizationState = &raster;
    pci.pMultisampleState = &multisample;
    pci.pDepthStencilState = &depth;
    pci.pColorBlendState = &blend;
    pci.pDynamicState = &dynamic;
    pci.layout = m_interactiveViewport.materialPreviewPipelineLayout;
    pci.renderPass = m_interactiveViewport.renderPass;
    pci.subpass = 0;
    const VkResult result = vkCreateGraphicsPipelines(
        device, VK_NULL_HANDLE, 1, &pci, nullptr, &state.pipeline);
    vkDestroyShaderModule(device, fragModule, nullptr);
    vkDestroyShaderModule(device, vertModule, nullptr);
    if (result != VK_SUCCESS) {
        state.pipeline = VK_NULL_HANDLE;
        SCENE_LOG_WARN(
            "[MaterialPreview] Realtime SurfaceSDF pipeline creation failed; "
            "the raster viewport remains usable without the optional pass.");
        return false;
    }

    updateMaterialPreviewSdfSurfaceBinding();
    return true;
}

void VulkanBackendAdapter::updateMaterialPreviewSdfSurfaceBinding() {
    if (!m_materialPreviewSdfSurface || !m_device ||
        m_interactiveViewport.materialPreviewDescSet == VK_NULL_HANDLE) return;
    const VkBuffer volume = m_device->m_volumeBuffer.buffer;
    if (volume == VK_NULL_HANDLE ||
        volume == m_materialPreviewSdfSurface->boundVolumeBuffer) return;

    // The descriptor set is shared with submitted material-preview frames.
    // Drain only when the backing allocation changed; ordinary per-frame volume
    // uploads keep the same handle and pay no descriptor synchronization cost.
    drainInteractiveViewportInFlight();
    VkDescriptorBufferInfo info{};
    info.buffer = volume;
    info.offset = 0;
    info.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_interactiveViewport.materialPreviewDescSet;
    write.dstBinding = 20;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(m_device->getDevice(), 1, &write, 0, nullptr);
    m_materialPreviewSdfSurface->boundVolumeBuffer = volume;
}

void VulkanBackendAdapter::recordMaterialPreviewSdfSurfacePass(
    VkCommandBuffer cmd, const Matrix4x4& viewProj, const Matrix4x4& view,
    uint32_t width, uint32_t height, bool opaqueSnapshotReady) {
    // ★ Same five gates, same reason, as recordMaterialPreviewVolumePass: an
    // empty frame is produced BOTH by "the pass never recorded" and by "the pass
    // recorded and the shader found nothing", and the screen cannot tell them
    // apart. The SurfaceSDF has a second, temporal failure mode the gas does not
    // — a slot published INACTIVE for a frame while its grid is regenerating
    // (see the mid-rebuild branch in VulkanBackend_Volumes.cpp) — which reads as
    // the surface blinking out. Name the gate so a disappearance is attributable
    // without a bisect. Change-gated: one line per state transition.
    const bool gateMode  = (m_viewportMode == ViewportMode::MaterialPreview);
    const bool gatePipe  = m_materialPreviewSdfSurface &&
                           m_materialPreviewSdfSurface->pipeline != VK_NULL_HANDLE;
    const bool gateSet   = m_interactiveViewport.materialPreviewDescSet != VK_NULL_HANDLE;
    const bool gateCount = m_device && m_device->m_volumeCount > 0u;
    const bool gateBound = m_materialPreviewSdfSurface &&
                           m_materialPreviewSdfSurface->boundVolumeBuffer != VK_NULL_HANDLE;
    SCENE_LOG_ON_CHANGE("mpsdf.gate",
        (long long)((gateMode ? 1 : 0) | (gatePipe ? 2 : 0) | (gateSet ? 4 : 0) |
                    (gateCount ? 8 : 0) | (gateBound ? 16 : 0)) * 1000ll +
            (long long)(m_device ? m_device->m_volumeCount : 0u),
        std::string("[MPSdf] pass gates: mode=") + (gateMode ? "1" : "0") +
        " pipeline=" + (gatePipe ? "1" : "0") +
        " descSet=" + (gateSet ? "1" : "0") +
        " volumeCount=" + std::to_string(m_device ? m_device->m_volumeCount : 0u) +
        " bound=" + (gateBound ? "1" : "0") +
        (gateMode && gatePipe && gateSet && gateCount && gateBound
            ? "  -> RECORDED (a surface missing from here on is the SSBO slot or "
              "the shader, not the gates)"
            : "  -> SKIPPED"));

    if (!cmd || !gateMode || !gatePipe || !gateSet || !gateCount || !gateBound) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      m_materialPreviewSdfSurface->pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_interactiveViewport.materialPreviewPipelineLayout,
                            0, 1, &m_interactiveViewport.materialPreviewDescSet,
                            0, nullptr);
    VkViewport viewport{0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {width, height}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    PreviewPush push{};
    matrixToGL(viewProj, push.viewProj);
    matrixToGL(view, push.view);
    push.cameraPos[0] = m_camera.origin.x;
    push.cameraPos[1] = m_camera.origin.y;
    push.cameraPos[2] = m_camera.origin.z;
    // w is unused by the shared preview ABI. It tells only this shader that
    // bindings 17/18 contain the opaque image captured before the SDF draw.
    push.lightDir0[3] = opaqueSnapshotReady ? 1.0f : 0.0f;
    uint32_t quality = 2u;
    if (::render_settings.raster_viewport_quality_preset ==
        ::RasterViewportQualityPreset::Performance) quality = 1u;
    else if (::render_settings.raster_viewport_quality_preset ==
                 ::RasterViewportQualityPreset::Quality ||
             ::render_settings.raster_viewport_quality_preset ==
                 ::RasterViewportQualityPreset::Full) quality = 3u;
    push.materialMeta[0] = m_interactiveViewport.materialPreviewBoundMaterialCount;
    push.materialMeta[1] = quality;
    push.materialMeta[2] =
        static_cast<uint32_t>(::render_settings.material_preview_lighting_preset);
    push.materialMeta[3] = m_device->m_volumeCount;
    vkCmdPushConstants(cmd, m_interactiveViewport.materialPreviewPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(push), &push);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

} // namespace Backend
