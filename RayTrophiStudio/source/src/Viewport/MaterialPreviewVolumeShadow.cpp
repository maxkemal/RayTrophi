#include "Viewport/MaterialPreviewVolumeShadow.h"
#include "globals.h"
#include <cstring>
#include <fstream>
#include <vector>

namespace Backend {
VolumeShadowBudget volumeShadowBudget(RasterViewportQualityPreset preset) {
    const uint32_t side = preset == RasterViewportQualityPreset::Performance ? 512u : 1024u;
    const uint32_t layers = preset == RasterViewportQualityPreset::Performance ? 8u :
        (preset == RasterViewportQualityPreset::Quality || preset == RasterViewportQualityPreset::Full ? 16u : 12u);
    const uint32_t tilesPerRow = rasterShadowAtlasSize() / rasterShadowTileSize(preset);
    return {side, side / tilesPerRow, layers,
            layers * (preset == RasterViewportQualityPreset::Performance ? 2u : 4u)};
}
namespace {
struct Push {
    float invViewProj[16];
    uint32_t tile, tilesPerRow, tileSize, layers;
    uint32_t volumeCount, steps, perspective, padding;
};
static_assert(sizeof(Push) == 96, "Volume shadow compute push ABI");
}

MaterialPreviewVolumeShadow::~MaterialPreviewVolumeShadow() {
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    if (layout_) vkDestroyPipelineLayout(device_, layout_, nullptr);
    if (pool_) vkDestroyDescriptorPool(device_, pool_, nullptr);
    if (descriptorLayout_) vkDestroyDescriptorSetLayout(device_, descriptorLayout_, nullptr);
}

bool MaterialPreviewVolumeShadow::initialize(const std::string& shaderDir) {
    if (pipeline_) return true;
    std::ifstream file(shaderDir + "/material_preview_volume_shadow.spv",
                       std::ios::binary | std::ios::ate);
    if (!file) return false;
    const std::streamsize bytes = file.tellg();
    if (bytes <= 0 || (static_cast<size_t>(bytes) % 4) != 0) return false;
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(words.data()), bytes)) return false;

    VkDescriptorSetLayoutBinding bindings[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl.bindingCount = 2; dsl.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device_, &dsl, nullptr, &descriptorLayout_) != VK_SUCCESS) return false;
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1; pl.pSetLayouts = &descriptorLayout_;
    pl.pushConstantRangeCount = 1; pl.pPushConstantRanges = &range;
    if (vkCreatePipelineLayout(device_, &pl, nullptr, &layout_) != VK_SUCCESS) return false;
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool.maxSets = 1; pool.poolSizeCount = 1; pool.pPoolSizes = &size;
    if (vkCreateDescriptorPool(device_, &pool, nullptr, &pool_) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = pool_; alloc.descriptorSetCount = 1; alloc.pSetLayouts = &descriptorLayout_;
    if (vkAllocateDescriptorSets(device_, &alloc, &set_) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    sm.codeSize = words.size() * 4; sm.pCode = words.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device_, &sm, nullptr, &module) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module; ci.stage.pName = "main"; ci.layout = layout_;
    const auto result = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline_);
    vkDestroyShaderModule(device_, module, nullptr);
    return result == VK_SUCCESS;
}

bool MaterialPreviewVolumeShadow::bindingsChanged(VkBuffer volumes, VkBuffer shadows) const {
    return volumes != volume_ || shadows != shadow_;
}
void MaterialPreviewVolumeShadow::bind(VkBuffer volumes, VkBuffer shadows) {
    VkDescriptorBufferInfo info[2]{{volumes, 0, VK_WHOLE_SIZE}, {shadows, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set_; writes[i].dstBinding = i;
        writes[i].descriptorCount = 1; writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &info[i];
    }
    vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    volume_ = volumes; shadow_ = shadows;
}
void MaterialPreviewVolumeShadow::record(VkCommandBuffer cmd, const float inverseViewProj[16],
    uint32_t tile, uint32_t tilesPerRow, uint32_t tileSize, uint32_t layers,
    uint32_t volumeCount, uint32_t steps, bool perspective) {
    Push push{};
    std::memcpy(push.invViewProj, inverseViewProj, sizeof(push.invViewProj));
    push.tile = tile; push.tilesPerRow = tilesPerRow; push.tileSize = tileSize;
    push.layers = layers; push.volumeCount = volumeCount; push.steps = steps;
    push.perspective = perspective ? 1u : 0u;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set_, 0, nullptr);
    vkCmdPushConstants(cmd, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (tileSize + 7) / 8, (tileSize + 7) / 8, 1);
}
} // namespace Backend
