#include "Api/RtApi.h"
#include "RtApiInternal.h"
#include "Backend/VulkanBackend.h"
#include "globals.h"

#include <cmath>
#include <memory>

extern std::unique_ptr<Backend::IBackend> g_backend;

namespace rtapi {
VulkanRT::UpscaleSettings upscaleSettings() {
    return g_ctx ? g_ctx->render_settings.upscale : VulkanRT::UpscaleSettings{};
}

Result setUpscale(const std::optional<std::string>& mode,
    const std::optional<std::string>& quality, std::optional<float> sharpness,
    std::optional<bool> rejection_reactive) {
    if (!g_ctx) {
        return notBound();
    }
    using namespace VulkanRT;
    auto next = upscaleSettings();
    if (mode) {
        if (*mode == "off") {
            next.mode = UpscaleMode::Off;
        } else if (*mode == "fsr") {
            next.mode = UpscaleMode::Fsr;
        } else if (*mode == "auto") {
            next.mode = UpscaleMode::Auto;
        } else {
            return Result::fail("mode must be 'off', 'fsr' or 'auto'");
        }
    }
    if (quality) {
        if (*quality == "quality") {
            next.quality = UpscaleQuality::Quality;
        } else if (*quality == "balanced") {
            next.quality = UpscaleQuality::Balanced;
        } else if (*quality == "performance") {
            next.quality = UpscaleQuality::Performance;
        } else if (*quality == "ultra_performance") {
            next.quality = UpscaleQuality::UltraPerformance;
        } else {
            return Result::fail("quality must be quality, balanced, performance or ultra_performance");
        }
    }
    if (sharpness) {
        next.sharpness = *sharpness;
    }
    if (rejection_reactive) {
        next.rejection_reactive = *rejection_reactive;
    }
    if (!std::isfinite(next.sharpness) || next.sharpness < 0.0f || next.sharpness > 1.0f) {
        return Result::fail("sharpness must be finite and in [0, 1]");
    }
    // SVGF is an optional input provider, not a requirement: without it FSR
    // upscales the raw accumulation (upscale_stats input_provider "none").
    if (next.mode != UpscaleMode::Off && !g_ctx->render_settings.motion_aov_enabled) {
        return Result::fail("FSR requires the motion AOV enabled");
    }
    // Project-persisted preference. Hardware availability is an observed status,
    // never a reason to silently replace the requested mode with 'off'.
    g_ctx->render_settings.upscale = next;
    if (auto* vk = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get())) {
        vk->configureUpscale(next, vk->realtimeDenoiserStats(false).moving,
            ui.viewport_settings.shading_mode == 2 && !g_ctx->render_settings.is_final_render_mode);
    }
    g_ctx->start_render = true;
    return Result::success();
}

VulkanRT::UpscaleStats upscaleStats(bool synchronize) {
    if (auto* vk = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get())) {
        return vk->upscaleStats(synchronize);
    }
    VulkanRT::UpscaleStats status;
    status.reason = "Vulkan render backend required";
    return status;
}
} // namespace rtapi
