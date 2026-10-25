#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace VulkanRT {

struct RayGpuTimingSample {
    uint32_t captureIndex = 0;
    uint32_t sampleIndex = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    double gpuMs = 0.0;
};

struct RayGpuTimingSnapshot {
    bool enabled = false;
    bool capturing = false;
    bool supported = false;
    uint32_t pending = 0;
    uint32_t recorded = 0;
    uint32_t failed = 0;
    uint32_t hostWaits = 0;
    double hostWaitMs = 0.0;
    double maxHostWaitMs = 0.0;
    uint32_t asyncSlotWaits = 0;
    double asyncSlotWaitMs = 0.0;
    double maxAsyncSlotWaitMs = 0.0;
    std::string error;
    std::vector<RayGpuTimingSample> samples;
};

// An explicitly armed, bounded capture. The disabled frame path does not
// allocate, issue timestamps, poll or synchronize. Explicit snapshot requests
// may still retrieve a disabled capture's pending results, without WAIT_BIT.
class VulkanRayGpuTiming {
public:
    static constexpr uint32_t capacity = 64;
    static constexpr uint32_t invalid = UINT32_MAX;

    bool configure(VkDevice device, VkPhysicalDevice physical, uint32_t queueFamily,
        bool enabled, std::string& error);
    void destroy(); // owner must have drained its existing in-flight work
    void resetQueries(VkCommandBuffer command);
    uint32_t begin(VkCommandBuffer command, uint32_t width, uint32_t height,
        uint32_t sampleIndex);
    void end(VkCommandBuffer command, uint32_t token);
    void cancel(VkCommandBuffer command);
    RayGpuTimingSnapshot snapshot();
    uint64_t beginHostWait() const;
    void endHostWait(uint64_t started, bool asyncSlot = false);
    bool enabled() const { return m_enabled; }

private:
    void collect();
    struct Slot {
        VkCommandBuffer command = VK_NULL_HANDLE;
        bool ended = false;
        bool consumed = false;
        RayGpuTimingSample sample;
    };
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueryPool m_pool = VK_NULL_HANDLE;
    uint32_t m_validBits = 0;
    double m_periodNs = 0.0;
    bool m_enabled = false;
    bool m_supported = false;
    uint32_t m_recorded = 0;
    uint32_t m_failed = 0;
    uint32_t m_hostWaits = 0;
    double m_hostWaitMs = 0.0;
    double m_maxHostWaitMs = 0.0;
    bool m_hostCaptureFinished = false;
    uint32_t m_asyncSlotWaits = 0;
    double m_asyncSlotWaitMs = 0.0;
    double m_maxAsyncSlotWaitMs = 0.0;
    std::string m_error;
    std::array<Slot, capacity> m_slots{};
    std::vector<RayGpuTimingSample> m_samples;
};

} // namespace VulkanRT
