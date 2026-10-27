#pragma once

#include <memory>
#include <vulkan/vulkan.h>

namespace VulkanRT {

class VulkanDevice;
struct ImageHandle;

// Display-only composition guard for the raw accumulator. A realtime denoiser
// (SVGF) and/or the upscaler compose into raw so the existing tonemap shows
// their result; save() backs raw up first and restore() puts it back after
// tonemap in the same submission. Nothing composed can reach Welford
// accumulation or OIDN's HDR input.
//
// Owned by the backend, not by a denoiser: the upscaler needs the same
// protection when no denoiser runs (upscale input provider "none").
class VulkanRawGuard {
public:
    VulkanRawGuard();
    ~VulkanRawGuard();
    // Allocates a backup matching raw (size + format). Idempotent per raw view.
    bool ensure(VulkanDevice& device, const ImageHandle& raw);
    const ImageHandle& backup() const;
    void save(VkCommandBuffer cmd);
    void restore(VkCommandBuffer cmd);
    void destroy(VulkanDevice& device);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace VulkanRT
