#include "Api/RtApi.h"
#include "imgui.h"

namespace rtapi {
void drawUpscaleControls() {
    static std::string error;
    auto apply = [&](const Result& result) {
        error = result.ok ? std::string() : result.error;
    };
    const auto settings = upscaleSettings();
    int mode = static_cast<int>(settings.mode);
    const char* modes[] = {"Off", "FSR 3.1", "Auto (FSR 3.1)"};
    if (ImGui::Combo("Upscale While Moving", &mode, modes, 3)) {
        const char* names[] = {"off", "fsr", "auto"};
        apply(setUpscale(std::string(names[mode]), {}, {}));
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    if (settings.mode == VulkanRT::UpscaleMode::Off) {
        return;
    }
    int quality = static_cast<int>(settings.quality) - 1;
    const char* qualities[] = {"Quality", "Balanced", "Performance", "Ultra Performance"};
    if (ImGui::Combo("FSR Quality", &quality, qualities, 4)) {
        const char* names[] = {"quality", "balanced", "performance", "ultra_performance"};
        apply(setUpscale({}, std::string(names[quality]), {}));
    }
    float sharpness = settings.sharpness;
    if (ImGui::SliderFloat("FSR Sharpness", &sharpness, 0.0f, 1.0f)) {
        apply(setUpscale({}, {}, sharpness));
    }
    bool rejection = settings.rejection_reactive;
    if (ImGui::Checkbox("FSR: Rejected History Is Reactive", &rejection)) {
        apply(setUpscale({}, {}, {}, rejection));
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Where the renderer found that a pixel's previous content does not "
                          "match (moving particles, deforming surfaces), FSR uses the new frame "
                          "instead of dragging old history along. Stops ghost trails.");
    }
    const auto s = upscaleStats(false);
    ImGui::Text("%u x %u -> %u x %u%s", s.render_w, s.render_h, s.display_w,
        s.display_h, s.active ? " (active)" : "");
    // FSR reads whatever the input provider composed; with none it upscales
    // the raw accumulation, which on a noisy path-traced view keeps the noise.
    ImGui::Text("Input: %s", s.input_provider == "svgf" ? "SVGF (denoised)" :
                             "raw accumulation (no denoiser)");
    if (!s.available) {
        ImGui::TextWrapped("%s", s.reason.c_str());
    }
    if (!s.active && !s.inactive_reason.empty()) {
        ImGui::TextWrapped("%s", s.inactive_reason.c_str());
    }
}
} // namespace rtapi
