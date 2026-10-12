#include "UI/AudioThunderUI.h"
#include "Audio/AudioAuthoring.h"
#include "Audio/AudioService.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "Camera.h"
#include "imgui.h"
#include <utility>
#include <stdexcept>

namespace AudioUI {
void drawThunderAudio(UIContext& ctx) {
    if (!ImGui::TreeNode("Lightning / Thunder")) {
        return;
    }
    static std::string error;
    auto call = [&](const std::string& op, const nlohmann::json& params) {
        try {
            audio::author(ctx, ctx.scene_ui_ptr ? &ctx.scene_ui_ptr->history : nullptr, op, params);
            error.clear();
        } catch (const std::exception& e) {
            error = e.what();
        }
    };
    if (ImGui::Button("Connect CC0 thunder variations")) {
        call("thunder_setup", nlohmann::json::object());
    }
    if (ImGui::Button("Choose thunder Assets folder")) {
        const auto folder = SceneUI::selectFolderDialogW(L"Assets root containing audio/cc0_thunder");
        if (!folder.empty()) {
            call("thunder_setup", {{"assets_root", folder}});
        }
    }
    auto config = ctx.scene.audio->snapshot().at("thunder");
    bool enabled = config.at("enabled");
    if (ImGui::Checkbox("Automatic thunder", &enabled)) {
        call("thunder_update", {{"config", {{"enabled", enabled}}}});
    }
    for (const auto& field : {std::pair{"gain", "Thunder gain"},
        std::pair{"sound_speed_mps", "Sound speed (m/s)"},
        std::pair{"reference_distance_m", "Reference distance (m)"},
        std::pair{"max_distance_m", "Audible range (m)"},
        std::pair{"crossover_m", "Near / far transition (m)"}}) {
        float value = config.at(field.first);
        if (ImGui::InputFloat(field.second, &value, 0, 0, "%.2f",
                              ImGuiInputTextFlags_EnterReturnsTrue)) {
            call("thunder_update", {{"config", {{field.first, value}}}});
        }
    }
    ImGui::TextWrapped("Uses the scene lightning clock. Distance / sound speed delays each event; "
        "near and far recordings vary. Paused visual time emits no new flashes. "
        "Pending thunder continues until Stop All, a seek or scene change.");
    const auto status = ctx.scene.audio->invoke("status", nlohmann::json::object());
    const auto& runtime = status.at("thunder_runtime");
    ImGui::Text("Pending: %zu, voices: %zu", runtime.at("pending").size(),
                status.at("thunder_voices").size());
    ImGui::Text("Detected event identities: %zu", runtime.at("seen_count").get<std::size_t>());
    if (!runtime.at("pending").empty()) {
        const nlohmann::json* nearestArrival = nullptr;
        for (const auto& event : runtime.at("pending")) {
            if (!nearestArrival || event.at("remaining_seconds").get<double>() <
                nearestArrival->at("remaining_seconds").get<double>()) {
                nearestArrival = &event;
            }
        }
        ImGui::Text("Next thunder: %.1f km, arrives in %.1f s",
            nearestArrival->at("distance_m").get<double>() / 1000,
            nearestArrival->at("remaining_seconds").get<double>());
    }
    if (runtime.at("suspended").get<bool>()) {
        ImGui::TextWrapped("Thunder suspended by Stop All / asset preview. Resume automatic "
                           "audio or restart timeline Play.");
    }
    static int previewId = 0;
    ImGui::BeginDisabled(!ctx.scene.camera || !enabled || runtime.at("suspended").get<bool>() ||
                        ctx.scene.audio->transportPaused());
    if (ImGui::Button("Test thunder at 1 km (about 2.9 s)")) {
        const auto position = ctx.scene.camera->lookfrom;
        call("thunder_event", {{"key", "preview/" + std::to_string(++previewId)},
            {"position", nlohmann::json::array({position.x + 1000, position.y, position.z})},
            {"intensity", 30}});
    }
    ImGui::EndDisabled();
    const auto runtimeError = status.at("thunder_error").get<std::string>();
    if (!error.empty() || !runtimeError.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 0.4f, 0.3f, 1));
        ImGui::TextWrapped("Thunder: %s", (error.empty() ? runtimeError : error).c_str());
        ImGui::PopStyleColor();
    }
    ImGui::TreePop();
}
}
