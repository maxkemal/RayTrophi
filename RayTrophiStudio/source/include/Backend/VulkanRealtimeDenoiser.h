#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vulkan/vulkan.h>
#include "Backend/RealtimeDenoiserTypes.h"

namespace VulkanRT {

class VulkanDevice;
struct ImageHandle;

// Display-only SVGF, the "svgf" upscale input provider. VulkanRawGuard backs
// the raw accumulator up before begin() composes into it and restores it after
// tonemap in the same queue submission. No filtered value can enter Welford
// accumulation or OIDN's HDR input.
class VulkanRealtimeDenoiser {
public:
    VulkanRealtimeDenoiser();
    ~VulkanRealtimeDenoiser();
    void configure(const RealtimeDenoiserSettings& settings, bool moving);
    // record0/1: raygen's per-dispatch primary-hit records (VulkanMotionHistory
    // history slots). Temporal reprojection tests its 2x2 / 3x3 taps against them.
    // backup: VulkanRawGuard's copy of raw (the unfiltered input). variance:
    // raygen's per-pixel Welford M2 (still-frame noise measurement).
    bool ensure(VulkanDevice& device, const ImageHandle& raw, const ImageHandle& backup,
                const ImageHandle& variance, const ImageHandle& albedo,
                const ImageHandle& normal, const ImageHandle& position,
                const ImageHandle& motion, const ImageHandle& record0,
                const ImageHandle& record1);
    void destroy(VulkanDevice& device);
    void invalidate();
    void setActiveExtent(uint32_t width, uint32_t height);
    // staticAllowed: viewport OIDN is off, so a still frame may be filtered
    // (RealtimeDenoiserSettings::still_noise). targetSamples caps it; epoch is
    // the accumulation about to be traced (a completed still accumulation
    // stops filtering only for its own epoch).
    bool wantsFrame(uint64_t epoch, int sample, int targetSamples, bool finalRender,
                    bool debug, bool staticAllowed) const;
    // Call after VulkanRawGuard::save in the same command buffer, and after
    // this frame slot's fence (the slot's still-noise counter is read here).
    // rtHistory: the RT record slot not written this dispatch holds the previous
    // dispatch. rtWriteSlot: the slot raygen writes this dispatch. camera:
    // origin xyz + pixel size per unit distance (VulkanMotionHistory).
    void begin(VulkanDevice& device, VkCommandBuffer cmd, uint32_t slot,
               uint64_t epoch, int sample, uint64_t rtDispatch,
               bool rtHistory, uint32_t rtWriteSlot, const float camera[4],
               bool forceMoving = false);
    void end(VkCommandBuffer cmd);
    RealtimeDenoiserStats stats(VulkanDevice& device, bool synchronize);
    bool ownsViewport() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace VulkanRT
