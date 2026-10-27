#pragma once

#include <cstdint>
#include <string>

namespace VulkanRT {
struct RealtimeDenoiserSettings {
    // On by default since B3b: display-only while the camera moves, the
    // converged image is bit-identical with it off (B3 live test).
    bool enabled = true;
    int atrous_iterations = 4;
    int history_max = 24;
    // Firefly clamp (sigmas) for pixels starting a new history. Replaces
    // B3's variance_clip, which clamped ALL history and darkened the image.
    float firefly_clip = 3.0f;
    // Still camera, viewport OIDN OFF: spatial-only SVGF over the ACCUMULATED
    // image, faded out per pixel by MEASURED noise (display only). The raw
    // accumulation's relative standard error (raygen's Welford M2, 3x3
    // same-material mean) is shown unfiltered at or below this value and fully
    // filtered at twice it. 0 = off. Applies through the converged frame:
    // pixels still above the target stay filtered on the final view (display
    // only, the accumulator is untouched); there is no forced raw ramp.
    // Replaces static_samples (fixed linear fade to raw by sample 64): on an
    // indirect-lit interior the shown noise grew as sqrt(N)/64 with the sample
    // count, and from 64 on the raw 64+ spp image was all that was left.
    float still_noise = 0.02f;
    // Diagnostic: SVGF frames show each pixel's history length instead of the
    // filtered image (red = no history, yellow = < 4, green = long). Session only.
    bool debug_history = false;
    // Scene animation counts as motion: consecutive realtime traces that each
    // start a new accumulation while the camera is still (skinning, timeline,
    // simulation, rigid objects) run SVGF's temporal path and let FSR engage.
    // Off = the pre-2026-10-27 behaviour, where such frames ran the still
    // filter at 1 spp with the history wiped every frame (skin scene: 11.8 ms
    // per frame at full resolution, mean history 1.0).
    bool scene_motion = true;
};

struct RealtimeDenoiserStats {
    bool available = false;
    bool moving = false;
    bool presenting = false;
    bool static_active = false;   // last dispatch was a still-camera (spatial-only) frame
    // Every pixel of this accumulation reached still_noise: still-frame SVGF
    // stopped dispatching until the next accumulation reset.
    bool still_complete = false;
    bool gpu_timing_available = false;
    uint64_t dispatches = 0;
    uint64_t history_resets = 0;
    uint64_t epoch = 0;
    int sample = 0;
    double temporal_ms = 0.0;
    double variance_ms = 0.0;
    double atrous_ms = 0.0;
    double total_ms = 0.0;
    double history_length_mean = 0.0;
    // Fraction of active pixels still (partly) filtered in the last measured
    // still frame; -1 = not measured for this accumulation yet. Read back one
    // frame-slot later, never by stalling.
    double still_filtered_fraction = -1.0;
    // Why the policy says moving: "camera", "scene" (animation, see
    // scene_motion) or "none". Filled by the backend adapter.
    std::string motion_source = "none";
    // Consecutive realtime traces that started a new accumulation with the
    // camera still; scene motion needs at least 2.
    int scene_reset_streak = 0;
    std::string reason = "not initialized";
};
} // namespace VulkanRT
