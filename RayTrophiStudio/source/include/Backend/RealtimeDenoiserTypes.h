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
    // image until this many samples, fading to the raw accumulation as it
    // approaches the limit (display only). 0 = off; capped below the target
    // sample count, so a converged frame is always shown unfiltered.
    // Default 64 (B3c live, interior): 8 filtered samples were as close to the
    // converged image as ~48 unfiltered ones; cost ~13% per sample, only below
    // the limit and only with OIDN off.
    int static_samples = 64;
    // Diagnostic: SVGF frames show each pixel's history length instead of the
    // filtered image (red = no history, yellow = < 4, green = long). Session only.
    bool debug_history = false;
};

struct RealtimeDenoiserStats {
    bool available = false;
    bool moving = false;
    bool presenting = false;
    bool static_active = false;   // last dispatch was a still-camera (spatial-only) frame
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
    std::string reason = "not initialized";
};
} // namespace VulkanRT
