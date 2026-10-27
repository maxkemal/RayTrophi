#pragma once
#include "Api/RtApiUpscale.h"

template<class Dict> Dict upscaleDictionary(bool synchronize = true) {
    using namespace VulkanRT;
    const auto settings = rtapi::upscaleSettings();
    const auto s = rtapi::upscaleStats(synchronize);
    Dict out;
    out["mode"] = settings.mode == UpscaleMode::Off ? "off" :
        settings.mode == UpscaleMode::Fsr ? "fsr" : "auto";
    const char* quality[] = {"quality", "balanced", "performance", "ultra_performance"};
    out["quality"] = quality[static_cast<int>(settings.quality) - 1];
    out["sharpness"] = settings.sharpness;
    out["rejection_reactive"] = settings.rejection_reactive;
    out["available"] = s.available;
    out["reason"] = s.reason;
    out["inactive_reason"] = s.inactive_reason;
    out["dll_loaded"] = s.dll_loaded;
    out["ffx_version"] = s.ffx_version;
    out["active"] = s.active;
    out["moving"] = s.moving;
    out["bridge_active"] = s.bridge_active;
    out["render_w"] = s.render_w;
    out["render_h"] = s.render_h;
    out["display_w"] = s.display_w;
    out["display_h"] = s.display_h;
    out["effective_scale"] = s.effective_scale;
    out["jitter_phase_count"] = s.jitter_phase_count;
    out["jitter_x"] = s.jitter_x;
    out["jitter_y"] = s.jitter_y;
    out["gpu_timing_available"] = s.gpu_timing_available;
    out["gpu_ms"] = s.gpu_ms;
    out["svgf_filter_ms"] = s.svgf_filter_ms;
    out["input_provider"] = s.input_provider;
    out["dispatches"] = s.dispatches;
    out["epoch"] = s.epoch;
    out["sample"] = s.sample;
    out["history_resets"] = s.history_resets;
    out["failures"] = s.failures;
    out["rejected_slots"] = s.rejected_slots;
    return out;
}
namespace pybind11 { class module_; }
void bindUpscale(pybind11::module_& render);
