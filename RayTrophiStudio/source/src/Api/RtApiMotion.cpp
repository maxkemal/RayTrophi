/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Api/RtApiMotion.cpp
* License:       MIT
* =========================================================================
*
* Motion AOV settings and measurement. See include/Api/RtApiMotion.h.
*/
#include "Api/RtApi.h"
#include "RtApiInternal.h"
#include "globals.h"
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanBackend.h"

#include <memory>

extern std::unique_ptr<Backend::IBackend> g_backend;
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

namespace rtapi {

namespace {

// Named devices, never derived from backend_ptr (see vulkanBackendsForStats).
std::vector<std::pair<std::string, Backend::VulkanBackendAdapter*>> motionDevices() {
    std::vector<std::pair<std::string, Backend::VulkanBackendAdapter*>> out;
    auto* render = dynamic_cast<Backend::VulkanBackendAdapter*>(::g_backend.get());
    auto* viewport = dynamic_cast<Backend::VulkanBackendAdapter*>(::g_viewport_backend.get());
    if (render) out.emplace_back("render", render);
    if (viewport && viewport != render) out.emplace_back("viewport", viewport);
    return out;
}

} // namespace

MotionStatsInfo motionStats(const std::string& sample) {
    MotionStatsInfo out;
    if (g_ctx) {
        out.enabled = g_ctx->render_settings.motion_aov_enabled;
        out.halton_jitter = g_ctx->render_settings.motion_halton_jitter;
    }
    const auto devices = motionDevices();
    for (const auto& [role, vk] : devices) {
        VulkanRT::MotionAovStatus s;
        vk->getMotionAovStatus(s);
        MotionDeviceInfo d;
        d.role = role;
        d.rt_available = s.rt_available;
        d.rt_enabled = s.rt_enabled;
        d.rt_history_valid = s.rt_history_valid;
        d.halton_jitter = s.halton_jitter;
        d.jitter_x = s.jitter[0];
        d.jitter_y = s.jitter[1];
        d.width = s.width;
        d.height = s.height;
        d.dispatches = s.counters.dispatches;
        d.object_motion_dispatches = s.counters.object_motion_dispatches;
        d.history_resets = s.counters.history_resets;
        d.raster_available = s.raster_available;
        d.raster_objects_available = s.raster_objects_available;
        d.raster_reason = s.raster_reason;
        out.devices.push_back(d);
    }

    if (sample.empty()) return out;
    if (sample != "rt" && sample != "raster") {
        out.sample_reason = "sample must be '', 'rt' or 'raster'";
        return out;
    }
    out.sample_source = sample;
    const bool raster = sample == "raster";
    // RT traces on the render device; the raster TAA runs on the viewport device
    // when there is one, otherwise on the render device itself.
    Backend::VulkanBackendAdapter* pick = nullptr;
    std::string pickRole;
    for (const auto& [role, vk] : devices) {
        if (raster ? role == "viewport" : role == "render") { pick = vk; pickRole = role; }
    }
    if (!pick && !devices.empty()) { pick = devices.front().second; pickRole = devices.front().first; }
    if (!pick) {
        out.sample_reason = "no Vulkan device";
        return out;
    }
    out.sample_role = pickRole;
    VulkanRT::MotionImageSummary s;
    pick->sampleMotionImage(raster, s);
    out.sampled = s.sampled;
    out.sample_reason = s.reason;
    out.pixels = s.pixels;
    out.rejected_fraction = s.rejected_fraction;
    out.reactive_fraction = s.reactive_fraction;
    out.mean_dx = s.mean_dx;
    out.mean_dy = s.mean_dy;
    out.mean_motion_px = s.mean_motion_px;
    out.max_motion_px = s.max_motion_px;
    return out;
}

Result setMotion(std::optional<bool> enabled, std::optional<bool> halton_jitter) {
    if (!g_ctx) return notBound();
    if (enabled && !*enabled && g_ctx->render_settings.realtime_denoiser.enabled) {
        return Result::fail("disable SVGF before disabling the motion AOV");
    }
    // FSR needs the motion AOV on its own: SVGF is no longer the gate that
    // kept it on (the upscale input provider may be "none").
    if (enabled && !*enabled &&
        g_ctx->render_settings.upscale.mode != VulkanRT::UpscaleMode::Off) {
        return Result::fail("disable upscale before disabling the motion AOV");
    }
    auto& rs = g_ctx->render_settings;
    if (enabled) rs.motion_aov_enabled = *enabled;
    if (halton_jitter) rs.motion_halton_jitter = *halton_jitter;
    // The main loop pushes the change to every Vulkan device on its next frame.
    g_ctx->start_render = true;
    return Result::success();
}

} // namespace rtapi
