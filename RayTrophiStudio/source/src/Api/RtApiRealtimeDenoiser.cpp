#include "Api/RtApi.h"
#include "RtApiInternal.h"
#include "Backend/VulkanBackend.h"
#include "globals.h"

#include <cmath>
#include <memory>

extern std::unique_ptr<Backend::IBackend> g_backend;

namespace rtapi {
VulkanRT::RealtimeDenoiserSettings realtimeDenoiserSettings() {
    return g_ctx ? g_ctx->render_settings.realtime_denoiser :
        VulkanRT::RealtimeDenoiserSettings{};
}

Result setRealtimeDenoiser(const std::optional<std::string>& mode,
    std::optional<int> atrous_iterations, std::optional<int> history_max,
    std::optional<float> firefly_clip, std::optional<int> static_samples,
    std::optional<bool> debug_history) {
    if (!g_ctx) {
        return notBound();
    }
    auto next = realtimeDenoiserSettings();
    if (mode) {
        if (*mode != "off" && *mode != "svgf") {
            return Result::fail("mode must be 'off' or 'svgf'");
        }
        next.enabled = *mode == "svgf";
    }
    if (atrous_iterations) {
        next.atrous_iterations = *atrous_iterations;
    }
    if (history_max) {
        next.history_max = *history_max;
    }
    if (firefly_clip) {
        next.firefly_clip = *firefly_clip;
    }
    if (static_samples) {
        next.static_samples = *static_samples;
    }
    if (debug_history) {
        next.debug_history = *debug_history;
    }
    if (next.atrous_iterations < 1 || next.atrous_iterations > 5) {
        return Result::fail("atrous_iterations must be in [1, 5]");
    }
    if (next.history_max < 1 || next.history_max > 64) {
        return Result::fail("history_max must be in [1, 64]");
    }
    if (!std::isfinite(next.firefly_clip) ||
        next.firefly_clip < 0.1f || next.firefly_clip > 10.0f) {
        return Result::fail("firefly_clip must be finite and in [0.1, 10]");
    }
    if (next.static_samples < 0 || next.static_samples > 1024) {
        return Result::fail("static_samples must be in [0, 1024] (0 = off)");
    }
    if (next.enabled && !g_ctx->render_settings.motion_aov_enabled) {
        return Result::fail("SVGF requires render.set_motion enabled=true");
    }
    // Commit only after every field has passed validation. Session-only, like
    // B2 motion controls; no project serializer owns a second copy.
    g_ctx->render_settings.realtime_denoiser = next;
    if (auto* vk = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get())) {
        vk->requestPresentRefresh();
    }
    g_ctx->start_render = true;
    return Result::success();
}

VulkanRT::RealtimeDenoiserStats realtimeDenoiserStats(bool synchronize) {
    if (auto* vk = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get())) {
        return vk->realtimeDenoiserStats(synchronize);
    }
    VulkanRT::RealtimeDenoiserStats result;
    result.reason = "Vulkan render backend required";
    return result;
}

void updateRealtimeDenoiserPolicy(bool cameraMoving, bool finalRender) {
    auto settings = realtimeDenoiserSettings();
    if (finalRender) {
        settings.enabled = false;
    }
    if (auto* vk = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get())) {
        const bool wasMoving = vk->realtimeDenoiserStats(false).moving;
        vk->configureRealtimeDenoiser(settings, cameraMoving);
        if (wasMoving && !vk->realtimeDenoiserStats(false).moving && settings.enabled) {
            vk->requestPresentRefresh();
            if (g_ctx) {
                g_ctx->start_render = true;
            }
        }
    }
}

bool realtimeDenoiserOwnsViewport(Backend::IBackend* backend) {
    auto* vk = dynamic_cast<Backend::VulkanBackendAdapter*>(backend);
    return vk && vk->realtimeDenoiserOwnsViewport();
}
} // namespace rtapi
