#pragma once
// ---------------------------------------------------------------------------
// Motion AOV (render.motion_stats / render.set_motion).
// docs/dev/RT_REALTIME_DENOISE_UPSCALE_PLAN.md B2.
//
// The Vulkan RT path writes, per dispatch, a motion image (previous-frame pixel
// minus this pixel's centre, a reactive mask and a history-rejected flag) and
// keeps a ping-pong primary-hit history. The raster viewport's TAA pass writes
// the same contract in Material/RayFusion (camera + global-instance model motion).
// SVGF consumes the RT vectors; this surface measures both paths independently.
//
// ★ A sample is one synchronous GPU readback of the whole image, summarised on
// the CPU: it stalls a frame. Ask for it in a measurement, not in a loop.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace rtapi {

struct Result;

struct MotionDeviceInfo {
    std::string role;                 // "render" | "viewport"
    bool rt_available = false;
    bool rt_enabled = false;
    bool rt_history_valid = false;
    bool halton_jitter = false;
    float jitter_x = 0.0f, jitter_y = 0.0f;   // last dispatch, pixels
    uint32_t width = 0, height = 0;
    uint64_t dispatches = 0;
    uint64_t object_motion_dispatches = 0;    // first dispatch after a TLAS build with valid previous transforms
    uint64_t history_resets = 0;
    bool raster_objects_available = false;
    bool raster_available = false;
    std::string raster_reason;
};

struct MotionStatsInfo {
    bool enabled = true;              // RenderSettings::motion_aov_enabled
    bool halton_jitter = false;
    std::vector<MotionDeviceInfo> devices;
    // Image summary, only when a sample was requested.
    bool sampled = false;
    std::string sample_source;        // "rt" | "raster"
    std::string sample_role;          // device the image came from
    std::string sample_reason;        // why not sampled
    uint64_t pixels = 0;
    double rejected_fraction = 0.0;
    double reactive_fraction = 0.0;
    double mean_dx = 0.0, mean_dy = 0.0;
    double mean_motion_px = 0.0, max_motion_px = 0.0;
};
// sample: "" (no readback), "rt" or "raster". Unknown values are an error,
// reported in sample_reason with sampled = false.
MotionStatsInfo motionStats(const std::string& sample);
// Unset optionals are left alone. halton_jitter changes the primary-ray
// sub-pixel pattern and so resets accumulation.
Result setMotion(std::optional<bool> enabled, std::optional<bool> halton_jitter);

} // namespace rtapi
