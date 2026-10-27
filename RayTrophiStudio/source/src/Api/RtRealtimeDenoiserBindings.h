#pragma once

#include "Api/RtApiRealtimeDenoiser.h"

template<class Dict> Dict realtimeDenoiserDictionary(bool synchronize = true) {
    const auto settings = rtapi::realtimeDenoiserSettings();
    const auto s = rtapi::realtimeDenoiserStats(synchronize);
    Dict out;
    out["mode"] = settings.enabled ? "svgf" : "off";
    out["atrous_iterations"] = settings.atrous_iterations;
    out["history_max"] = settings.history_max;
    out["firefly_clip"] = settings.firefly_clip;
    out["still_noise"] = settings.still_noise;
    out["debug_history"] = settings.debug_history;
    out["scene_motion"] = settings.scene_motion;
    out["motion_source"] = s.motion_source;
    out["scene_reset_streak"] = s.scene_reset_streak;
    out["static_active"] = s.static_active;
    out["still_complete"] = s.still_complete;
    out["still_filtered_fraction"] = s.still_filtered_fraction;
    out["available"] = s.available;
    out["moving"] = s.moving;
    out["presenting"] = s.presenting;
    out["gpu_timing_available"] = s.gpu_timing_available;
    out["dispatches"] = s.dispatches;
    out["history_resets"] = s.history_resets;
    out["epoch"] = s.epoch;
    out["sample"] = s.sample;
    out["temporal_ms"] = s.temporal_ms;
    out["variance_ms"] = s.variance_ms;
    out["atrous_ms"] = s.atrous_ms;
    out["total_ms"] = s.total_ms;
    out["history_length_mean"] = s.history_length_mean;
    out["reason"] = s.reason;
    return out;
}

namespace pybind11 { class module_; }
void bindRealtimeDenoiser(pybind11::module_& render);
