#include "RayFusion/ProbeOverlay.h"
#include "Backend/VulkanViewportBackend.h"
#include <cstring>
#include <filesystem>
#include <fstream>

namespace RayFusion {
namespace {
struct Push { float viewProj[16], centerRadius[4], color[4]; };
static_assert(sizeof(Push) == 96, "Probe overlay push ABI");
VkShaderModule load(VkDevice device, const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return VK_NULL_HANDLE;
    const std::streamsize size = file.tellg();
    if (size <= 0 || size % 4 != 0) return VK_NULL_HANDLE;
    std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(words.data()), size)) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = words.size() * 4; ci.pCode = words.data();
    VkShaderModule result = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &result) != VK_SUCCESS) return VK_NULL_HANDLE;
    return result;
}
}
ProbeOverlay::~ProbeOverlay() {
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
}
bool ProbeOverlay::initialize(VkRenderPass pass, const std::string& dir) {
    if (pipeline_) return true;
    if (attempted_) return false;
    attempted_ = true;
    auto vert = load(device_, dir + "/rayfusion_probe_overlay.spv");
    auto frag = load(device_, dir + "/rayfusion_probe_overlay_frag.spv");
    if (!vert || !frag) {
        if (vert) vkDestroyShaderModule(device_, vert, nullptr);
        if (frag) vkDestroyShaderModule(device_, frag, nullptr);
        reason_ = "Probe overlay shaders missing or invalid; rebuild shaders and restart";
        return false;
    }
    VkPushConstantRange range{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo lc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lc.pushConstantRangeCount = 1; lc.pPushConstantRanges = &range;
    auto result = vkCreatePipelineLayout(device_, &lc, nullptr, &layout_);
    if (result == VK_SUCCESS) {
        VkPipelineShaderStageCreateInfo stages[2]{};
        for (int i = 0; i < 2; ++i) {
            stages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[i].pName = "main";
        }
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vert;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = frag;
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1; vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = VK_TRUE;
        ds.depthWriteEnable = VK_FALSE;
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        VkPipelineColorBlendAttachmentState attachment{};
        attachment.colorWriteMask = 0xf;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1; cb.pAttachments = &attachment;
        VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = 2; dy.pDynamicStates = dynamic;
        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.stageCount = 2; ci.pStages = stages; ci.pVertexInputState = &vi;
        ci.pInputAssemblyState = &ia; ci.pViewportState = &vp;
        ci.pRasterizationState = &rs; ci.pMultisampleState = &ms;
        ci.pDepthStencilState = &ds; ci.pColorBlendState = &cb; ci.pDynamicState = &dy;
        ci.layout = layout_; ci.renderPass = pass;
        result = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline_);
    }
    vkDestroyShaderModule(device_, vert, nullptr);
    vkDestroyShaderModule(device_, frag, nullptr);
    reason_ = result == VK_SUCCESS ? "" : "Probe overlay graphics pipeline creation failed";
    return result == VK_SUCCESS;
}
void ProbeOverlay::record(VkCommandBuffer cmd, const float viewProj[16],
                          const std::vector<ProbeMarker>& markers) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    Push push{};
    std::memcpy(push.viewProj, viewProj, sizeof(push.viewProj));
    for (const auto& marker : markers) {
        std::memcpy(push.centerRadius, marker.position.data(), 3 * sizeof(float));
        push.centerRadius[3] = marker.radius;
        const float colors[3][4] = {{1,0.65f,0.08f,1},{0.1f,0.95f,0.35f,1},{1,0.15f,0.2f,1}};
        std::memcpy(push.color, colors[marker.state <= 2 ? marker.state : 0], sizeof(push.color));
        vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
        vkCmdDraw(cmd, 24, 1, 0, 0);
    }
}
}

namespace Backend {
void VulkanViewportBackend::recordRayFusionProbeOverlay(VkCommandBuffer cmd, const Matrix4x4& viewProj) {
    if (!rayFusionProbeOverlayRequested()) return;
    auto markers = rayFusionProbeMarkers();
    if (markers.empty()) {
        setRayFusionProbeOverlayResult(false, 0, "Probe field not configured");
        return;
    }
    if (!m_probeOverlay) {
        m_probeOverlay = std::make_shared<RayFusion::ProbeOverlay>(m_device->getDevice());
        std::string dir = "shaders";
        for (const char* candidate : {"shaders", "source/shaders", "../shaders"}) {
            if (std::filesystem::exists(std::string(candidate) + "/rayfusion_probe_overlay.spv")) {
                dir = candidate; break;
            }
        }
        m_probeOverlay->initialize(m_interactiveViewport.renderPass, dir);
    }
    if (!m_probeOverlay->ready()) {
        setRayFusionProbeOverlayResult(false, 0, m_probeOverlay->reason());
        return;
    }
    float matrix[16];
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col) matrix[col * 4 + row] = viewProj.m[row][col];
    m_probeOverlay->record(cmd, matrix, markers);
    setRayFusionProbeOverlayResult(true, static_cast<uint32_t>(markers.size()), "");
}
}
