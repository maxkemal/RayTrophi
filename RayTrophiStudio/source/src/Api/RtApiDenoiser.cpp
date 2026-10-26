/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiDenoiser.cpp
* License:       MIT
* =========================================================================
*
* Denoiser settings and per-call cost. See include/Api/RtApiDenoiser.h.
*/
#include "Api/RtApi.h"
#include "RtApiInternal.h"
#include "globals.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rtapi {

namespace {

// Written and read on the UI thread only: the main loop records, and the
// IPC/Python readers run from the same loop's command queue.
DenoiserStatsInfo s_stats;

const char* modeName(DenoiserMode m) {
    return m == DenoiserMode::Fast ? "fast" : "quality";
}

const char* scheduleName(DenoiserSchedule s) {
    switch (s) {
        case DenoiserSchedule::EverySample:   return "every_sample";
        case DenoiserSchedule::ConvergedOnly: return "converged_only";
        case DenoiserSchedule::Progressive:
        default:                              return "progressive";
    }
}

const char* qualityName(DenoiserQuality q) {
    switch (q) {
        case DenoiserQuality::Balanced: return "balanced";
        case DenoiserQuality::High:     return "high";
        case DenoiserQuality::Fast:
        default:                        return "fast";
    }
}

} // namespace

DenoiserInfo denoiser() {
    DenoiserInfo out;
    if (!g_ctx) return out;
    const auto& rs = g_ctx->render_settings;
    out.viewport_enabled = rs.use_denoiser;
    out.render_enabled = rs.render_use_denoiser;
    out.timeline_enabled = rs.timeline_use_denoiser;
    out.mode = modeName(rs.denoiser_mode);
    out.quality = qualityName(rs.denoiser_quality);
    out.schedule = scheduleName(rs.denoiser_schedule);
    out.blend = rs.denoiser_blend_factor;
    return out;
}

Result setDenoiser(std::optional<bool> viewport_enabled,
                   std::optional<bool> render_enabled,
                   std::optional<bool> timeline_enabled,
                   const std::optional<std::string>& mode,
                   const std::optional<std::string>& quality,
                   std::optional<float> blend,
                   const std::optional<std::string>& schedule) {
    if (!g_ctx) return notBound();

    // Validate everything before writing anything.
    std::optional<DenoiserMode> newMode;
    if (mode) {
        if (*mode == "fast") newMode = DenoiserMode::Fast;
        else if (*mode == "quality") newMode = DenoiserMode::Quality;
        else return Result::fail("mode must be 'fast' or 'quality'");
    }
    std::optional<DenoiserQuality> newQuality;
    if (quality) {
        if (*quality == "fast") newQuality = DenoiserQuality::Fast;
        else if (*quality == "balanced") newQuality = DenoiserQuality::Balanced;
        else if (*quality == "high") newQuality = DenoiserQuality::High;
        else return Result::fail("quality must be 'fast', 'balanced' or 'high'");
    }
    if (blend && (!std::isfinite(*blend) || *blend < 0.0f || *blend > 1.0f))
        return Result::fail("blend must be in [0, 1]");
    std::optional<DenoiserSchedule> newSchedule;
    if (schedule) {
        if (*schedule == "every_sample") newSchedule = DenoiserSchedule::EverySample;
        else if (*schedule == "progressive") newSchedule = DenoiserSchedule::Progressive;
        else if (*schedule == "converged_only") newSchedule = DenoiserSchedule::ConvergedOnly;
        else return Result::fail("schedule must be 'every_sample', 'progressive' or 'converged_only'");
    }

    auto& rs = g_ctx->render_settings;
    if (viewport_enabled) rs.use_denoiser = *viewport_enabled;
    if (render_enabled) rs.render_use_denoiser = *render_enabled;
    if (timeline_enabled) rs.timeline_use_denoiser = *timeline_enabled;
    if (newMode) rs.denoiser_mode = *newMode;
    if (newQuality) rs.denoiser_quality = *newQuality;
    if (blend) rs.denoiser_blend_factor = *blend;
    if (newSchedule) rs.denoiser_schedule = *newSchedule;
    // No accumulation reset: the denoiser reads the accumulated buffer every
    // frame, exactly as when the panel edits these fields.
    return Result::success();
}

DenoiserStatsInfo denoiserStats() { return s_stats; }

void resetDenoiserStats() { s_stats = DenoiserStatsInfo{}; }

void recordDenoiserCall(double ms, const char* path, int width, int height,
                        int sample_count, bool accumulation_complete) {
    DenoiserStatsInfo& s = s_stats;
    const bool repeat = s.calls > 0 && sample_count == s.last_sample_count &&
                        width == s.last_width && height == s.last_height;
    ++s.calls;
    if (repeat) ++s.repeat_sample_calls;
    if (accumulation_complete) ++s.converged_calls;
    if (std::strcmp(path, "gpu") == 0) { ++s.gpu_path; ++s.succeeded; }
    else if (std::strcmp(path, "cpu") == 0) { ++s.cpu_path; ++s.succeeded; }
    else if (std::strcmp(path, "surface") == 0) { ++s.surface_path; ++s.succeeded; }
    s.total_ms += ms;
    s.max_ms = std::max(s.max_ms, ms);
    s.last_ms = ms;
    s.last_width = width;
    s.last_height = height;
    s.last_sample_count = sample_count;
    s.last_path = path;
}

void recordDenoiserSkip(bool reused) {
    ++s_stats.skipped_by_schedule;
    if (reused) ++s_stats.cache_presents;
}

} // namespace rtapi
