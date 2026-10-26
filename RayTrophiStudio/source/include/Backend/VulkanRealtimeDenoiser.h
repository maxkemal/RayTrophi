#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vulkan/vulkan.h>
#include "Backend/RealtimeDenoiserTypes.h"

namespace VulkanRT {

class VulkanDevice;
struct ImageHandle;

// Display-only SVGF. The raw accumulator is backed up before composition and
// restored after tonemap in the same queue submission. No filtered value can
// enter Welford accumulation or OIDN's HDR input.
class VulkanRealtimeDenoiser {
public:
    VulkanRealtimeDenoiser();
    ~VulkanRealtimeDenoiser();
    void configure(const RealtimeDenoiserSettings& settings, bool moving);
    // record0/1: raygen's per-dispatch primary-hit records (VulkanMotionHistory
    // history slots). Temporal reprojection tests its 2x2 / 3x3 taps against them.
    bool ensure(VulkanDevice& device, const ImageHandle& raw, const ImageHandle& albedo,
                const ImageHandle& normal, const ImageHandle& position,
                const ImageHandle& motion, const ImageHandle& record0,
                const ImageHandle& record1);
    void destroy(VulkanDevice& device);
    void invalidate();
    // staticAllowed: viewport OIDN is off, so a still frame may be filtered
    // (RealtimeDenoiserSettings::static_samples). targetSamples caps it.
    bool wantsFrame(int sample, int targetSamples, bool finalRender, bool debug,
                    bool staticAllowed) const;
    // rtHistory: the RT record slot not written this dispatch holds the previous
    // dispatch. rtWriteSlot: the slot raygen writes this dispatch. camera:
    // origin xyz + pixel size per unit distance (VulkanMotionHistory).
    void begin(VulkanDevice& device, VkCommandBuffer cmd, uint32_t slot,
               uint64_t epoch, int sample, uint64_t rtDispatch,
               bool rtHistory, uint32_t rtWriteSlot, const float camera[4]);
    void end(VkCommandBuffer cmd);
    RealtimeDenoiserStats stats(VulkanDevice& device, bool synchronize);
    bool ownsViewport() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace VulkanRT
