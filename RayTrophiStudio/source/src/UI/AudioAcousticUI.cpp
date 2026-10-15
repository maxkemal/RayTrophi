#include "Audio/AcousticScene.h"
#include "Audio/AudioAuthoring.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "Camera.h"
#include "SubstanceLibrary.h"
#include "ObjectSubstance.h"
#include "globals.h"
#include "imgui.h"
#include <exception>

namespace audio {
void drawAcousticDiagnostics(UIContext& ctx, const nlohmann::json& emitter) {
    using json = nlohmann::json;
    if (!ImGui::TreeNode("Scene acoustics / Direct path")) {
        return;
    }
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
    ImGui::TextWrapped("Direct path preview affects spatial manual sources in this session. "
                       "Unknown materials or incomplete geometry bypass attenuation.");
    json live;
    for (const auto& entry : ctx.scene.audio->invoke("acoustics_preview_status", json::object())) {
        if (entry.at("id") == emitter.at("id")) {
            live = entry;
        }
    }
    bool enabled = !live.is_null();
    int model = live.is_null() || live.at("boundary_mode") == "object_barrier" ? 0 : 1;
    if (!live.is_null()) {
        metersPerUnit = live.at("meters_per_unit").get<float>();
    }
    const bool changedUnits = ImGui::InputFloat("Meters per scene unit", &metersPerUnit,
        0, 0, "%.6g");
    const bool changedModel = ImGui::Combo("Boundary model", &model,
        "One barrier per object\0Independent thin surfaces\0");
    const bool changedEnabled = ImGui::Checkbox("Audition scene obstruction", &enabled);
    try {
        if (changedEnabled || (enabled && (changedUnits || changedModel))) {
            author(ctx, nullptr, "acoustics_preview", {{"id", emitter.at("id")},
                {"enabled", enabled}, {"meters_per_unit", metersPerUnit},
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
    }
    ImGui::TextWrapped("Broadband level approximation; no muffling, echo or diffraction yet. "
        "Object model applies transmission once per named object, regardless of thickness. "
        "Keep separate walls as separate objects. Up to four sources, about 4 updates/second.");
    std::string current;
    try {
        if (ctx.scene.camera) {
            const auto& listener = ctx.scene.camera->lookfrom;
            current = json{{"id", emitter.at("id")},
                {"position", worldPose(ctx, emitter).at("position")},
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
