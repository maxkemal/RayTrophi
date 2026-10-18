#include "Audio/AcousticScene.h"
#include "Audio/AcousticPreview.h"
#include "Audio/AudioAuthoring.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "Camera.h"
#include "SubstanceLibrary.h"
#include "ObjectSubstance.h"
#include "globals.h"
#include "imgui.h"
#include "UI/AudioPropertyLayout.h"
#include <exception>
#include <utility>

namespace audio {
void drawAcousticDiagnostics(UIContext& ctx, const nlohmann::json& emitter) {
    using json = nlohmann::json;
    ImGui::TextUnformatted("Scene acoustics");
    static const SceneData* owner = nullptr;
    static json report;
    static std::string key, error;
    static float metersPerUnit = 1;
    if (owner != &ctx.scene) {
        owner = &ctx.scene;
        report = nullptr;
        key.clear();
        error.clear();
    }
    ImGui::TextWrapped("Hear how walls and surfaces change this sound.");
    const auto& policy = emitter.at("acoustics");
    const auto sourceMode = policy.at("mode").get<std::string>();
    const char* modeLabel = sourceMode == "scene" ? "Use scene settings" :
        sourceMode == "custom" ? "Custom settings" : "Off for this source";
    if (ImGui::BeginCombo(AudioUI::property("Acoustic settings").c_str(), modeLabel)) {
        for (const auto& choice : {std::pair{"scene", "Use scene settings"},
            std::pair{"custom", "Custom settings"}, std::pair{"off", "Off for this source"}}) {
            if (ImGui::Selectable(choice.second, sourceMode == choice.first)) {
                auto edited = policy;
                edited["mode"] = choice.first;
                if (std::string(choice.first) == "custom" && sourceMode == "scene") {
                    const auto settings = ctx.scene.audio->invoke("acoustics_get", json::object());
                    for (const auto& source : settings.at("sources")) {
                        if (source.at("id") == emitter.at("id")) {
                            edited["config"] = source.at("effective");
                        }
                    }
                }
                try {
                    author(ctx, ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr,
                        "update", {{"id", emitter.at("id")}, {"acoustics", edited}});
                    error.clear();
                } catch (const std::exception& exception) {
                    error = exception.what();
                }
            }
        }
        ImGui::EndCombo();
    }
    if (sourceMode == "scene") {
        ImGui::TextWrapped("Following saved scene defaults. Edit them in Audio > Scene.");
    }
    if (!emitter.at("spatial").get<bool>()) {
        ImGui::TextWrapped("This sound is non-spatial; geometry acoustics does not apply.");
    }
    const auto config = sourceMode == "custom" ? policy.at("config") :
        ctx.scene.audio->snapshot().at("acoustics");
    if (sourceMode == "scene" && config.at("diffuse").get<bool>() &&
        !supportsDiffuseSource(emitter)) {
        ImGui::TextWrapped("Diffuse scattering is excluded for this directional source. "
            "Direct sound and supported reflections still use the scene settings.");
    }
    json live;
    for (const auto& entry : ctx.scene.audio->invoke("acoustics_preview_status", json::object())) {
        if (entry.at("id") == emitter.at("id")) {
            live = entry;
        }
    }
    bool enabled = sourceMode != "off" && config.at("enabled").get<bool>();
    int model = config.at("boundary_mode") == "object_barrier" ? 0 : 1;
    metersPerUnit = config.at("meters_per_unit").get<float>();
    ImGui::BeginDisabled(sourceMode != "custom" || !emitter.at("spatial").get<bool>());
    const bool changedEnabled = ImGui::Checkbox(sourceMode == "custom" ?
        "Enable custom acoustics" : "Scene default enabled", &enabled);
    bool changedUnits = false;
    bool changedModel = false;
    ImGui::BeginDisabled(!enabled);
    bool spectral = config.at("spectral");
    const bool changedSpectral = ImGui::Checkbox("Frequency dependent transmission", &spectral);
    bool reflections = config.at("reflections");
    const bool changedReflections = ImGui::Checkbox("Early specular reflections", &reflections);
    bool diffuse = config.at("diffuse");
    const bool changedDiffuse = ImGui::Checkbox("Diffuse surface scattering", &diffuse);
    if (ImGui::TreeNode("Scene scale and geometry")) {
        changedUnits = AudioUI::slider("Meters per scene unit", metersPerUnit, 0.001f,
            1000.0f, "%.3f m", ImGuiSliderFlags_Logarithmic);
        changedModel = ImGui::Combo(AudioUI::property("Surface interpretation").c_str(), &model,
            "One barrier per object\0Independent thin surfaces\0");
        ImGui::TreePop();
    }
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    try {
        if (changedEnabled || (enabled && (changedUnits || changedModel ||
                                          changedSpectral || changedReflections || changedDiffuse))) {
            author(ctx, ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr,
                "acoustics_preview", {{"id", emitter.at("id")},
                {"enabled", enabled}, {"meters_per_unit", metersPerUnit}, {"spectral", spectral},
                {"reflections", reflections},
                {"diffuse", diffuse},
                {"boundary_mode", model == 0 ? "object_barrier" : "thin_surface"}});
            error.clear();
        }
    } catch (const std::exception& exception) {
        error = exception.what();
    }
    if (!live.is_null()) {
        ImGui::Text("Live gain: %.3f | %s", live.at("amplitude_gain").get<double>(),
            live.at("status").get<std::string>().c_str());
        if (live.contains("report") && live.at("report").contains("error")) {
            ImGui::TextWrapped("%s", live.at("report").at("error").get<std::string>().c_str());
        }
        if (live.contains("report") && live.at("report").contains("diffuse")) {
            const auto& scatter = live.at("report").at("diffuse");
            if (scatter.value("enabled", false)) {
                ImGui::Text("Diffuse: %u / %u samples contributed | %s",
                    scatter.value("contributions", 0u), scatter.value("samples", 0u),
                    scatter.value("complete", false) ? "estimated" : "partial estimate");
            }
        }
    }
    if (!ImGui::TreeNode("Technical diagnostics")) {
        return;
    }
    ImGui::TextWrapped("Six complementary frequency bands; optional broadband comparison. "
        "Specular reflections sample six planes. Diffuse scattering uses 64 source rays, "
        "Lambert weighting and eight delay bins; omnidirectional sources only. "
        "Both models are single bounce, with up to one second of excess delay. "
        "Object model applies transmission once per named object, regardless of thickness. "
        "Keep separate walls as separate objects. Up to 32 sources, about 4 updates/second.");
    std::string current;
    try {
        if (ctx.scene.camera) {
            const auto& listener = ctx.scene.camera->lookfrom;
            current = json{{"id", emitter.at("id")},
                {"position", ctx.scene.audio->invoke("source_runtime",
                    {{"id", emitter.at("id")}}).at("position")},
                {"listener", {listener.x, listener.y, listener.z}}, {"units", metersPerUnit},
                {"boundary_model", model},
                {"geometry", g_scene_geometry_generation.load()},
                {"materials", matter::objectSubstanceFingerprint(ctx.scene)},
                {"audio", ctx.scene.audio->generation()},
                {"frame", ctx.scene.timeline.current_frame}}.dump();
        }
        if (ImGui::Button("Trace direct path")) {
            report = author(ctx, nullptr, "acoustics_trace",
                {{"id", emitter.at("id")}, {"meters_per_unit", metersPerUnit},
                 {"boundary_mode", model == 0 ? "object_barrier" : "thin_surface"}});
            key = current;
            error.clear();
        }
    } catch (const std::exception& exception) {
        report = nullptr;
        error = exception.what();
    }
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    if (report.is_null()) {
        ImGui::TreePop();
        return;
    }
    if (current.empty() || current != key) {
        ImGui::TextWrapped("Scene, source or listener changed. Trace again for a current snapshot.");
        ImGui::TreePop();
        return;
    }
    ImGui::Text("Path: %.3f m | travel time: %.2f ms",
        report.at("distance_meters").get<double>(), report.at("delay_seconds").get<double>() * 1000);
    ImGui::Text("Surface crossings: %zu | triangles: %zu", report.at("hits").size(),
        report.at("triangles").get<std::size_t>());
    ImGui::TextWrapped("Transmission: %s", report.at("transmission_status").get<std::string>().c_str());
    ImGui::TextDisabled("Coverage: authored flat meshes only");
    if (!report.at("snapshot_complete").get<bool>()) {
        ImGui::TextWrapped("Coverage is limited to authored flat meshes; other geometry was skipped.");
    }
    for (const auto& hit : report.at("hits")) {
        const auto& material = hit.at("material");
        const auto label = material.at("binding").is_null() ? std::string("Unassigned / conflict") :
            material.at("binding").at("substance").get<std::string>();
        ImGui::TextWrapped("%.3f m: %s | %s", hit.at("distance_meters").get<double>(),
            material.at("object").get<std::string>().c_str(), label.c_str());
    }
    ImGui::TextWrapped("Trace is a snapshot. Object barriers are an explicit assembly approximation; "
                       "thickness and internal propagation are not solved.");
    ImGui::TreePop();
}
} // namespace audio
