#pragma once

#include "Audio/AcousticGpu.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace VulkanRT {
class VulkanDevice;

// Main/render thread only. Borrows the renderer's fence and AS, never submits or waits.
class AcousticGpuPass {
public:
    explicit AcousticGpuPass(VulkanDevice& device);
    ~AcousticGpuPass();
    bool enqueue(audio::AcousticGpuBatch batch, const std::string& shaderDirectory);
    bool take(audio::AcousticGpuCompletion& result);
    void cancel();
    void collect(VkFence fence = VK_NULL_HANDLE);
    bool record(VkCommandBuffer cmd, VkFence fence);
    bool hasPending() const;
    void submitted(VkFence fence, bool success);
    void retireFence(VkFence fence);
    const std::string& error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace VulkanRT
