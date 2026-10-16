#pragma once
#include "Audio/AcousticSurface.h"
#include <vulkan/vulkan.h>

namespace VulkanRT {
class VulkanDevice;
// Records only into an already acquired renderer frame slot. Never submits or waits.
class AcousticSurfacePass {
public:
    explicit AcousticSurfacePass(VulkanDevice& device);
    ~AcousticSurfacePass();
    bool update(audio::AcousticSurfaceBatch batch, const std::string& directory);
    void disable();
    bool record(VkCommandBuffer cmd, unsigned slot, VkImageView position,
        VkImageView normal, VkImageView output, unsigned width, unsigned height,
        float exposure, float overlay);
    const std::string& error() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace VulkanRT
