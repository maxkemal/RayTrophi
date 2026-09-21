#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>

enum class RasterViewportQualityPreset;
namespace Backend {

struct VolumeShadowBudget {
    uint32_t atlasSize, tileSize, layers, steps;
};
VolumeShadowBudget volumeShadowBudget(RasterViewportQualityPreset preset);

// Compute companion to the raster shadow atlas. It writes the depth-dependent
// optical depth into the tail of the existing shadow-record SSBO (binding 7).
class MaterialPreviewVolumeShadow {
public:
    explicit MaterialPreviewVolumeShadow(VkDevice device) : device_(device) {}
    ~MaterialPreviewVolumeShadow();
    MaterialPreviewVolumeShadow(const MaterialPreviewVolumeShadow&) = delete;
    MaterialPreviewVolumeShadow& operator=(const MaterialPreviewVolumeShadow&) = delete;
    bool initialize(const std::string& shaderDir);
    bool bindingsChanged(VkBuffer volumes, VkBuffer shadows) const;
    void bind(VkBuffer volumes, VkBuffer shadows); // Caller drains in-flight frames.
    void record(VkCommandBuffer cmd, const float inverseViewProj[16],
                uint32_t tile, uint32_t tilesPerRow, uint32_t tileSize,
                uint32_t layers, uint32_t volumeCount, uint32_t steps,
                bool perspective);
private:
    VkDevice device_;
    VkDescriptorSetLayout descriptorLayout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkBuffer volume_ = VK_NULL_HANDLE, shadow_ = VK_NULL_HANDLE;
};
} // namespace Backend
