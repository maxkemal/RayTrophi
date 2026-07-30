#pragma once

#include <cstdint>
#include <memory>

namespace VulkanRT {

class VulkanDevice;
struct BufferHandle;

struct VolumePerformanceStats {
    uint32_t volumeRays = 0;
    uint32_t densitySamples = 0;
    uint32_t emptySegmentsSkipped = 0;
    uint32_t topologySegmentsSkipped = 0;
    uint32_t majorantSegmentsSkipped = 0;
    uint32_t shadowDensitySamples = 0;
    uint32_t extinctionTerminations = 0;
    uint32_t stepBudgetExhausted = 0;
    uint32_t completedIntervals = 0;
    uint32_t temporalAccepted = 0;
    uint32_t temporalRejected = 0;
    uint32_t majorantQueries = 0;
    uint32_t majorantAvailableQueries = 0;
    uint32_t reserved[3]{};
};
static_assert(sizeof(VolumePerformanceStats) == 64,
              "Volume instrumentation ABI must remain 16 uint32 words.");

class VulkanVolumeInstrumentation {
public:
    VulkanVolumeInstrumentation();
    ~VulkanVolumeInstrumentation();

    VulkanVolumeInstrumentation(const VulkanVolumeInstrumentation&) = delete;
    VulkanVolumeInstrumentation& operator=(const VulkanVolumeInstrumentation&) = delete;

    bool ensure(VulkanDevice& device);
    void destroy(VulkanDevice& device);
    void reset(VulkanDevice& device, bool enabled);
    VolumePerformanceStats read(VulkanDevice& device) const;
    const BufferHandle* buffer() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace VulkanRT
