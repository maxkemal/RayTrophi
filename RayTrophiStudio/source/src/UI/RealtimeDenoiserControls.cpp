#include "Api/RtApi.h"
#include "imgui.h"

namespace rtapi {
void drawRealtimeDenoiserControls() {
    static std::string error;
    auto apply = [&](const Result& result) {
        error = result.ok ? std::string() : result.error;
    };
    auto settings = realtimeDenoiserSettings();
    bool enabled = settings.enabled;
    if (ImGui::Checkbox("Denoise While Moving (SVGF)", &enabled)) {
        apply(setRealtimeDenoiser(std::string(enabled ? "svgf" : "off"), {}, {}, {}));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Fast temporal denoiser for the frames shown while the camera or an "
                          "object moves (Vulkan RT). Independent of the OIDN switch; the "
                          "converged image is unchanged. Needs Motion Vectors.");
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    if (!settings.enabled) {
        return;
    }
    if (ImGui::SliderInt("SVGF Iterations", &settings.atrous_iterations, 1, 5)) {
        apply(setRealtimeDenoiser({}, settings.atrous_iterations, {}, {}));
    }
    if (ImGui::SliderInt("SVGF History", &settings.history_max, 1, 64)) {
        apply(setRealtimeDenoiser({}, {}, settings.history_max, {}));
    }
    if (ImGui::SliderInt("SVGF On Still Frames", &settings.static_samples, 0, 256)) {
        apply(setRealtimeDenoiser({}, {}, {}, {}, settings.static_samples));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("With the OIDN denoiser OFF: filter the still image while it "
                          "accumulates, fading to the raw result by this sample count. "
                          "0 = off. The converged image is never filtered.");
    }
    bool debugHistory = settings.debug_history;
    if (ImGui::Checkbox("SVGF History Debug", &debugHistory)) {
        apply(setRealtimeDenoiser({}, {}, {}, {}, {}, debugHistory));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Show each pixel's temporal history while SVGF runs: red = none, "
                          "yellow = under 4 frames, green = long. Diagnostic only.");
    }
    if (ImGui::SliderFloat("SVGF Firefly Clip", &settings.firefly_clip, 0.1f, 10.0f)) {
        apply(setRealtimeDenoiser({}, {}, {}, settings.firefly_clip));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Sigmas above the neighbourhood at which a pixel with no usable "
                          "history is scaled down. Lower removes more bright dots on edges.");
    }
    const auto stats = realtimeDenoiserStats(false);
    if (!stats.reason.empty()) {
        ImGui::TextWrapped("%s", stats.reason.c_str());
    }
}
} // namespace rtapi
