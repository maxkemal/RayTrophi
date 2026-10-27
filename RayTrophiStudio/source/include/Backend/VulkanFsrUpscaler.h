#pragma once

#include "Backend/UpscaleTypes.h"
#include <memory>
#include <vulkan/vulkan.h>

namespace VulkanRT {
class VulkanDevice;
struct ImageHandle;

// One context per display allocation, independent of active render size.
// Owns GPU inputs, presentation/bridge history and per-slot timestamps.
class VulkanFsrUpscaler {
public:
    VulkanFsrUpscaler();
    ~VulkanFsrUpscaler();
    void configure(const UpscaleSettings& settings, bool moving);
    bool prepare(VulkanDevice& device, const ImageHandle& raw,
                 const ImageHandle& position, const ImageHandle& motion);
    void setEligible(bool eligible, const char* reason);
    bool needsReset() const;
    void acceptReset(uint64_t epoch);
    void invalidate();
    bool active() const;
    bool protectsDenoiser() const;
    uint32_t renderWidth(uint32_t display) const;
    uint32_t renderHeight(uint32_t display) const;
    bool jitter(uint64_t frame, float out[2]);
    bool bridgeWanted(int sample, bool finalRender, bool debug) const;
    // Called inside the current frame slot, under VulkanRawGuard, after the
    // input provider (if any) composed into raw. FSR reads raw as its colour
    // input whoever wrote it: the provider contract is "denoised HDR in raw at
    // render resolution", and with no provider it is the raw accumulation.
    bool record(VulkanDevice& device, VkCommandBuffer cmd, uint32_t slot,
                uint64_t epoch, int sample, const float camera[4][4], float fov,
                float preExposure, bool motionHistory);
    void recordBridge(VkCommandBuffer cmd, uint64_t epoch, int sample,
                      const float camera[4][4]);
    void setSlotValid(uint32_t slot, bool valid);
    bool slotValid(uint32_t slot) const;
    UpscaleStats stats(VulkanDevice& device, bool synchronize);
    void destroy(VulkanDevice& device);
private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
} // namespace VulkanRT
