#include "Backend/VulkanBackend.h"
#include "globals.h"
#include <array>
#include <filesystem>
#include <fstream>

namespace Backend {
void VulkanBackendAdapter::createMaterialPreviewCoveredPipeline(
    const VkGraphicsPipelineCreateInfo& base, const std::string& shaderDir) {
    if (!m_device || base.stageCount != 2 || !base.pDepthStencilState) return;
    const auto path = std::filesystem::path(shaderDir) / "material_preview_covered.spv";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { SCENE_LOG_WARN("[Coverage] material_preview_covered.spv missing; using regular material shading."); return; }
    const auto length = file.tellg();
    if (length <= 0 || static_cast<std::size_t>(length) % sizeof(uint32_t)) return;
    std::vector<uint32_t> words(static_cast<std::size_t>(length) / sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(length))) return;
    VkShaderModuleCreateInfo shader{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    shader.codeSize = words.size() * sizeof(uint32_t);
    shader.pCode = words.data();
    VkDevice device = m_device->getDevice();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &shader, nullptr, &module) != VK_SUCCESS) return;
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{base.pStages[0], base.pStages[1]};
    for (auto& stage : stages) if (stage.stage == VK_SHADER_STAGE_FRAGMENT_BIT) stage.module = module;
    auto depth = *base.pDepthStencilState;
    depth.depthWriteEnable = VK_FALSE;
    depth.depthCompareOp = VK_COMPARE_OP_EQUAL;
    auto info = base;
    info.pStages = stages.data();
    info.pDepthStencilState = &depth;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) == VK_SUCCESS) {
        m_interactiveViewport.materialPreviewCoveredPipeline = pipeline;
        SCENE_LOG_INFO("[Coverage] EQUAL / early-tests material pipeline ready.");
    } else {
        if (pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline, nullptr);
        SCENE_LOG_WARN("[Coverage] Covered pipeline creation failed; using regular material shading.");
    }
    vkDestroyShaderModule(device, module, nullptr);
}

void VulkanBackendAdapter::destroyMaterialPreviewCoveredPipeline() {
    if (!m_device) return;
    auto& pipeline = m_interactiveViewport.materialPreviewCoveredPipeline;
    if (pipeline) vkDestroyPipeline(m_device->getDevice(), pipeline, nullptr);
    pipeline = VK_NULL_HANDLE;
}

VkPipeline VulkanBackendAdapter::materialPreviewShadingPipeline(
    const RasterMeshBuffer& mesh, bool depthPrepassActive) const {
    if (depthPrepassActive && m_interactiveViewport.materialPreviewCoveredPipeline &&
        rasterMeshHasExactCoverage(mesh.materialUsage, mesh.cpuMatIds, mesh.vertexCount,
            m_cachedGpuMaterials, m_interactiveViewport.materialPreviewBoundMaterialCount,
            m_rasterMaterialPrograms, m_interactiveViewport.materialPreviewUsesExternalMaterials))
        return m_interactiveViewport.materialPreviewCoveredPipeline;
    return m_interactiveViewport.materialPreviewPipeline;
}
} // namespace Backend
