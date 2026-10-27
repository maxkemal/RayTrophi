#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace VulkanRT {

class VulkanDevice;
struct ImageHandle;

// Per-dispatch MotionParams record read by raygen.rgen through a buffer
// reference. std430, mirrors `MotionParamsRef` there (7 x 16 = 112 bytes).
struct alignas(16) MotionParamsGPU {
    float prevOrigin[4];
    float prevLowerLeft[4];
    float prevHorizontal[4];
    float prevVertical[4];
    float jitter[4];          // xy: sub-pixel jitter of this dispatch, pixels
    uint32_t instAddr[4];     // current instance array lo/hi, previous lo/hi
    uint32_t info[4];         // instance count, flags, dispatch index, unused
};
static_assert(sizeof(MotionParamsGPU) == 112, "MotionParamsGPU must match raygen.rgen");

enum MotionFlags : uint32_t {
    kMotionHistoryValid = 1u,   // the read history slot holds the previous dispatch
    kMotionObjects      = 2u,   // previous instance transforms are valid
    kMotionHalton       = 4u,   // the jitter drives the primary ray
    kMotionPrevCamera   = 8u,   // prev camera vectors are valid
};

// Running counters of what the motion AOV did (render.motion_stats).
struct MotionCounters {
    uint64_t dispatches = 0;
    uint64_t object_motion_dispatches = 0;  // dispatches with valid previous transforms
    uint64_t history_resets = 0;            // resize / device rebuild / explicit
};

// What render.motion_stats reports per device (VulkanMotionAov.cpp).
struct MotionAovStatus {
    bool rt_available = false;      // images + params allocated on this device
    bool rt_enabled = false;
    bool halton_jitter = false;
    bool rt_history_valid = false;
    float jitter[2] = {0.0f, 0.0f};
    uint32_t width = 0, height = 0;
    MotionCounters counters;
    // Raster viewport (RayFusion / material preview): the TAA pass writes the
    // same motion contract, CAMERA motion only (no per-object vectors yet).
    bool raster_objects_available = false;
    bool raster_available = false;
    std::string raster_reason;
};

// One synchronous readback of a motion image, summarised on the CPU.
struct MotionImageSummary {
    bool sampled = false;
    std::string reason;             // why not sampled
    uint64_t pixels = 0;
    double rejected_fraction = 0.0; // w = 1
    double reactive_fraction = 0.0; // z > 0
    double mean_dx = 0.0, mean_dy = 0.0;      // over accepted pixels
    double mean_motion_px = 0.0, max_motion_px = 0.0;
};

// Owns the motion AOV of the Vulkan RT path (RT_REALTIME_DENOISE_UPSCALE_PLAN.md
// B2): the motion image, the ping-pong primary-hit history and a small ring of
// MotionParams records. Unlike VulkanVolumeTemporal its history is NOT
// invalidated by an accumulation reset: a camera move is exactly what it is for;
// the shader's history test rejects what no longer matches.
class VulkanMotionHistory {
public:
    VulkanMotionHistory();
    ~VulkanMotionHistory();

    VulkanMotionHistory(const VulkanMotionHistory&) = delete;
    VulkanMotionHistory& operator=(const VulkanMotionHistory&) = delete;

    bool ensure(VulkanDevice& device, uint32_t width, uint32_t height);
    void destroy(VulkanDevice& device);
    void invalidate();   // next dispatch has no usable history or previous camera

    // Writes this dispatch's record and returns its device address (0 when the
    // AOV is off or unallocated). `camera` = origin, lowerLeft, horizontal,
    // vertical of THIS dispatch, as pushed to raygen.
    uint64_t prepareDispatch(VulkanDevice& device, const float camera[4][4],
                             uint64_t curInstances, uint64_t prevInstances,
                             uint32_t instanceCount, bool objectMotion);
    // Call once the dispatch was submitted: this camera becomes "previous".
    void advance();

    void setEnabled(bool enabled);   // re-enabling drops the history
    bool enabled() const { return m_enabled; }
    void setHaltonJitter(bool on) { m_halton = on; }
    bool haltonJitter() const { return m_halton; }
    const float* currentJitter() const { return m_jitter; }

    uint32_t writeIndex() const;
    bool hasHistory() const;
    void setActiveExtent(uint32_t width, uint32_t height);
    void setExternalJitter(bool enabled, const float jitter[2]);
    void cameraVectors(float out[4][4]) const;
    // This dispatch's camera origin (xyz) and the world size of one pixel per
    // unit of distance (w), from the vectors given to prepareDispatch.
    void cameraFootprint(float out[4]) const;
    const ImageHandle* motionImage() const;
    const ImageHandle* historySlot(uint32_t index) const;
    const MotionCounters& counters() const { return m_counters; }
    uint32_t width() const;
    uint32_t height() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    bool m_enabled = true;
    bool m_halton = false;
    float m_jitter[2] = {0.0f, 0.0f};
    MotionCounters m_counters;
};

} // namespace VulkanRT
