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
    // Embedded-solid probe accounting (volume_instrumentation.glsl writes these
    // as reserved2/reserved3). They were named `reserved` here and never shown
    // in the panel, so the shader has been recording them into a void.
    //
    // ★ Read them as a PAIR. runs == 0 means the gate suppressed the probe
    // entirely; runs > 0 with hits == 0 means the probe ran and did not see the
    // geometry. On screen both produce the same missing surface, which is why a
    // single "probe" counter cannot diagnose it.
    //
    // ★★ A zero here is not proof of absence: the gas march's probe mask (0xF1)
    // excludes splats and SurfaceSDF by construction, so those representations
    // can never raise `hits` no matter how many are in the box.
    uint32_t solidProbeRuns = 0;
    uint32_t solidProbeHits = 0;
    // The shader's `enabled` word (volume_instrumentation.glsl, last of the 16).
    // Was spelled reserved[2] here, which is why renaming the probe counters had
    // to touch it: the gate and the two counters shared one anonymous array.
    uint32_t enabled = 0;
    // Handoff accounting — see volume_instrumentation.glsl for what separates
    // these four. They exist because a gas segment can leave its box four ways
    // and three of them look the same on screen.
    uint32_t gasHandoffs = 0;
    uint32_t layeredHandoffs = 0;
    uint32_t arbiterRejects = 0;
    uint32_t teleports = 0;
    uint32_t arbiterCandidates = 0;
    uint32_t arbiterGateOpen = 0;
    // The three silent exits of nearestSurfaceSDFCrossing. arbiterCandidates is
    // counted BEFORE all three, which is why it reads 100% while the arbiter is
    // in fact failing — see volume_instrumentation.glsl for the partition.
    uint32_t arbiterNoBox = 0;
    uint32_t arbiterEmptyRange = 0;
    uint32_t arbiterNoCrossing = 0;
};
static_assert(sizeof(VolumePerformanceStats) == 100,
              "Volume instrumentation ABI must remain 25 uint32 words "
              "and must match volume_instrumentation.glsl exactly.");

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
