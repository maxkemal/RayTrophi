#include "Audio/AcousticRenderDebug.h"
#include "Audio/AcousticSurface.h"
#include "Audio/AcousticBoundary.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "SceneSelection.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "imgui.h"
#include "UI/AudioPropertyLayout.h"

#include <exception>
#include <string>

namespace audio {
void drawAcousticRenderDebugControls(UIContext& ctx) {
    if (ctx.render_settings.debug_view != kAcousticSurfaceDebugView) {
        drawAcousticBoundaryProbe(ctx);
        return;
    }
    const auto data = ctx.scene.audio->snapshot();
    const auto selected = ctx.selection.selected.type == SelectableType::AudioSource ?
        ctx.selection.selected.audio_source_id : std::string();
    std::string label = "Select a spatial manual source";
    for (const auto& emitter : data.at("emitters")) {
        if (emitter.at("id") == selected) {
            label = emitter.value("name", selected);
        }
    }
    static std::string error;
    auto workspace = ctx.scene.audio->invoke("workspace_get", nlohmann::json::object());
    bool all = workspace.value("acoustic_all_sources", false);
    int band = workspace.value("acoustic_band", -1) + 1;
    int reflections = workspace.value("acoustic_reflection_samples", 1);
    nlohmann::json patch = nlohmann::json::object();
    if (ImGui::Checkbox("Sum sources (max 4)", &all)) {
        patch["acoustic_all_sources"] = all;
    }
    const char* bands[] = {"Broadband mean", "125 Hz", "250 Hz", "500 Hz", "1 kHz", "2 kHz", "4 kHz"};
    if (ImGui::Combo(AudioUI::property("Acoustic band").c_str(), &band, bands, 7)) {
        patch["acoustic_band"] = band - 1;
    }
    if (ImGui::SliderInt(AudioUI::property("First-bounce samples").c_str(),
        &reflections, 0, 2)) {
        patch["acoustic_reflection_samples"] = reflections;
    }
    if (!patch.empty()) {
        try {
            author(ctx, nullptr, "workspace_update", patch);
            ctx.start_render = true;
            error.clear();
        } catch (const std::exception& exception) {
            error = exception.what();
        }
    }
    ImGui::BeginDisabled(all);
    if (ImGui::BeginCombo(AudioUI::property("Acoustic source").c_str(), label.c_str())) {
        for (const auto& emitter : data.at("emitters")) {
            if (!emitter.value("spatial", false) ||
                emitter.at("reaction").value("driver", "manual") != "manual") {
                continue;
            }
            const auto id = emitter.at("id").get<std::string>();
            const auto name = emitter.value("name", id) + "##" + id;
            if (ImGui::Selectable(name.c_str(), id == selected)) {
                try {
                    author(ctx, nullptr, "select", {{"id", id}});
                    ctx.start_render = true;
                    error.clear();
                } catch (const std::exception& exception) {
                    error = exception.what();
                }
            }
        }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    ImGui::TextWrapped("Six-band transmission, diffuse scattering and bounded specular reflection. "
        "Purple -60 dB, yellow 0 dB; magenta = unknown material / query limit. "
        "Grey = unavailable. Diffuse sampling is noisy; specular candidate coverage is incomplete.");
    ImGui::TextWrapped("Relative authored energy, not SPL or live PCM. "
        "No refraction, delayed audible echo or phase interference. Vulkan RT only.");
    drawAcousticBoundaryProbe(ctx);
    const auto status = acousticSurfaceStatus(ctx);
    const auto failure = status.value("error", std::string());
    if (!failure.empty()) {
        ImGui::TextWrapped("%s", failure.c_str());
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
}
} // namespace audio
