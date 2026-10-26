#pragma once
// ---------------------------------------------------------------------------
// Denoiser settings + cost (render.get_denoiser / set_denoiser /
// denoiser_stats / reset_denoiser_stats).
//
// ★ Until 2026-10-26 the denoiser was panel-only: an agent could measure the
// RT dispatch to the millisecond (render.gpu_timing_stats) but could neither
// see whether OIDN ran nor what it cost, so "the denoiser takes 10 fps to 8"
// was a number only a human could produce.
//
// The stats are recorded by the main loop around the whole viewport denoise
// block (frame fetch + OIDN + tonemap + copy into the display surface). The
// GPU path ends in a synchronous D2H copy, so its wall time IS its GPU cost.
//
// ★★ `repeat_sample_calls` counts denoises of a sample count that was already
// denoised at the same size - i.e. an identical input. A rate close to
// `calls` means the denoiser runs on the frame loop's cadence, not on new
// samples (feedback: a counter's rate proves what it counts).
// ---------------------------------------------------------------------------
#include <cstdint>
#include <optional>
#include <string>

namespace rtapi {

struct Result;

struct DenoiserInfo {
    bool viewport_enabled = false;   // RenderSettings::use_denoiser
    bool render_enabled = false;     // final render (OIDN quality forced to high)
    bool timeline_enabled = false;   // timeline playback
    std::string mode;                // "fast" = beauty only, "quality" = + albedo/normal AOVs
    std::string quality;             // viewport OIDN tier: "fast" | "balanced" | "high"
    std::string schedule;            // viewport: "every_sample" | "progressive" | "converged_only"
    float blend = 1.0f;              // 0 = raw, 1 = fully denoised
};
DenoiserInfo denoiser();
// Unset optionals are left alone. Unknown mode/quality names and a blend
// outside [0, 1] are rejected, not clamped.
Result setDenoiser(std::optional<bool> viewport_enabled,
                   std::optional<bool> render_enabled,
                   std::optional<bool> timeline_enabled,
                   const std::optional<std::string>& mode,
                   const std::optional<std::string>& quality,
                   std::optional<float> blend,
                   const std::optional<std::string>& schedule = std::nullopt);

struct DenoiserStatsInfo {
    uint64_t calls = 0;               // denoise blocks entered
    uint64_t succeeded = 0;           // a denoised image reached the display surface
    uint64_t gpu_path = 0;            // CUDA zero-copy OIDN
    uint64_t cpu_path = 0;            // host-buffer OIDN fallback
    uint64_t surface_path = 0;        // SDL-surface fallback (no backend frame)
    uint64_t repeat_sample_calls = 0; // same sample count + size as the previous call
    uint64_t converged_calls = 0;     // accumulation was already complete
    uint64_t skipped_by_schedule = 0; // denoiser on, sample not due (no OIDN run)
    uint64_t cache_presents = 0;      // ...of which re-showed the last denoised image
    double total_ms = 0.0;
    double max_ms = 0.0;
    double last_ms = 0.0;
    int last_width = 0;
    int last_height = 0;
    int last_sample_count = 0;
    std::string last_path = "none";   // gpu | cpu | surface | failed | none
};
DenoiserStatsInfo denoiserStats();
void resetDenoiserStats();

// Main loop hook. `path` is one of gpu | cpu | surface | failed.
void recordDenoiserCall(double ms, const char* path, int width, int height,
                        int sample_count, bool accumulation_complete);
// Main loop hook for a sample the schedule skipped. `reused` = the last
// denoised image was re-shown (false: the raw frame stayed on screen).
void recordDenoiserSkip(bool reused);

} // namespace rtapi
