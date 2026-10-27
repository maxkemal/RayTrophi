#pragma once

#include <memory>
#include <string>
#include <vulkan/vulkan.h>

namespace VulkanRT {
class VulkanDevice;
struct ImageHandle;

class VulkanUpscaleInputs {
public:
    VulkanUpscaleInputs();
    ~VulkanUpscaleInputs();
    bool ensure(VulkanDevice& device, const ImageHandle& raw, const ImageHandle& position,
                const ImageHandle& motion, std::string& reason);
    void destroy(VulkanDevice& device);
    // rejectionReactive: 0..1 weight of raygen's clustered history rejections
    // in FSR's reactive mask (UpscaleSettings::rejection_reactive).
    void recordInputs(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                      const float camera[4][4], float rejectionReactive);
    void compose(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                 float bridgeWeight, bool bridge, const float camera[4][4]);
    void savePosition(VkCommandBuffer cmd, uint32_t width, uint32_t height);
    const ImageHandle& depth() const;
    const ImageHandle& velocity() const;
    const ImageHandle& reactive() const;
    const ImageHandle& output() const;
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace VulkanRT
