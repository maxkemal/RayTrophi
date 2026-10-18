#pragma once
#include "UI/AudioWorkspaceUI.h"
#include "UI/AudioVisualLanguage.h"
#include "Audio/AudioService.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "SceneSelection.h"
#include "imgui.h"
#include <algorithm>
#include <memory>
#include <set>
#include <string>

namespace AudioUI {
inline void drawAudioScopeControls(UIContext& ctx) {
    using json = nlohmann::json;
    static std::weak_ptr<audio::AudioService> owner;
    static std::set<std::string> group;
    static int scope = 0;
    if (owner.lock() != ctx.scene.audio) {
        owner = ctx.scene.audio;
        group.clear();
        scope = 0;
    }
    const auto data = ctx.scene.audio->snapshot();
    std::set<std::string> known;
    for (const auto& source : data.at("emitters")) {
        known.insert(source.at("id").get<std::string>());
    }
    std::erase_if(group, [&](const std::string& id) { return !known.contains(id); });
    ImGui::Combo("Source scope", &scope, "Selected source\0Source group\0Scene sources\0");
    if (scope == 1 && ImGui::TreeNode("Group membership")) {
        ImGui::BeginChild("Audio scope members", ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 6),
                          true);
        for (const auto& source : data.at("emitters")) {
            const auto id = source.at("id").get<std::string>();
            const auto name = source.at("name").get<std::string>();
            bool included = group.contains(id);
            ImGui::PushID(id.c_str());
            if (ImGui::Checkbox(name.empty() ? id.c_str() : name.c_str(), &included)) {
                if (included) {
                    group.insert(id);
                } else {
                    group.erase(id);
                }
            }
            ImGui::PopID();
        }
        ImGui::EndChild();
        ImGui::TreePop();
    }
    json params = {{"scope", scope == 0 ? "selected" : scope == 1 ? "group" : "scene"}};
    json ids = json::array();
    if (scope == 0 && ctx.selection.selected.type == SelectableType::AudioSource &&
        known.contains(ctx.selection.selected.audio_source_id)) {
        ids.push_back(ctx.selection.selected.audio_source_id);
    } else if (scope == 1) {
        for (const auto& id : group) {
            ids.push_back(id);
        }
    }
    if (scope != 2) {
        params["ids"] = ids;
    }
    ImGui::TextDisabled("Target: %d sources", static_cast<int>(scope == 2 ? known.size() : ids.size()));
    ImGui::BeginDisabled(scope == 2 ? known.empty() : ids.empty());
    for (const auto* action : {"play", "pause", "stop"}) {
        if (action != std::string("play")) {
            ImGui::SameLine();
        }
        if (ImGui::Button(action)) {
            auto request = params;
            request["action"] = action;
            workspaceCall(ctx, "transport_batch", request);
        }
    }
    const auto entries = ctx.scene.audio->invoke("acoustics_preview_status", json::object());
    std::size_t configured = 0;
    std::size_t dry = 0;
    for (const auto& entry : entries) {
        if (scope != 2 && std::find(ids.begin(), ids.end(), entry.at("id")) == ids.end()) {
            continue;
        }
        ++configured;
        if (entry.value("bypassed", false)) {
            ++dry;
        }
    }
    const auto targets = scope == 2 ? known.size() : ids.size();
    const bool complete = configured > 0 && configured == targets;
    ImGui::TextDisabled("Acoustics: %s", configured == 0 ? "Not configured" :
        !complete ? "Partially configured" :
        dry == configured ? "Dry" : dry == 0 ? "Processed" : "Mixed");
    ImGui::BeginDisabled(!complete);
    if (stateButton("Dry", complete && dry == configured,
        "Bypass configured acoustic processing without changing its settings")) {
        auto request = params;
        request["action"] = "dry";
        workspaceCall(ctx, "acoustics_batch", request);
    }
    ImGui::SameLine();
    if (stateButton("Processed", complete && dry == 0,
        "Restore the configured acoustic processing")) {
        auto request = params;
        request["action"] = "processed";
        workspaceCall(ctx, "acoustics_batch", request);
    }
    ImGui::EndDisabled();
    if (ImGui::Button("Configure acoustics...")) {
        ImGui::OpenPopup("Scope acoustic configuration");
    }
    if (ImGui::BeginPopup("Scope acoustic configuration")) {
        static bool enabled = true;
        static bool reflections = true;
        static bool diffuse = false;
        ImGui::TextDisabled("Settings to apply to this target");
        ImGui::TextWrapped("Saves custom acoustic settings on the targeted sources.");
        ImGui::Checkbox("Enable acoustics", &enabled);
        ImGui::Checkbox("Specular reflections", &reflections);
        ImGui::Checkbox("Diffuse scattering", &diffuse);
        ImGui::TextWrapped("Applies spectral transmission at 1 meter per scene unit.");
        if (ImGui::Button("Apply to target")) {
            auto request = params;
            request["action"] = "configure";
            request["config"] = {{"enabled", enabled}, {"spectral", true},
                {"reflections", reflections}, {"diffuse", diffuse}};
            const auto result = workspaceCall(ctx, "acoustics_batch", request);
            if (!result.is_null() && !result.empty()) {
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }
    ImGui::EndDisabled();
    if (!ImGui::TreeNode("Target diagnostics")) {
        return;
    }
    ImGui::BeginChild("Scope diagnostics", ImVec2(0, 180), true);
    for (const auto& source : data.at("emitters")) {
        const auto id = source.at("id").get<std::string>();
        if (scope != 2 && std::find(ids.begin(), ids.end(), source.at("id")) == ids.end()) {
            continue;
        }
        const auto runtime = ctx.scene.audio->invoke("source_runtime", {{"id", id}});
        ImGui::TextWrapped("%s: %s", id.c_str(),
                          runtime.at("playback_state").get<std::string>().c_str());
    }
    for (const auto& entry : entries) {
        const auto id = entry.at("id").get<std::string>();
        if (scope != 2 && std::find(ids.begin(), ids.end(), entry.at("id")) == ids.end()) {
            continue;
        }
        ImGui::TextWrapped("%s: %s | %s", id.c_str(),
            entry.value("bypassed", false) ? "dry bypass" : "processed",
            entry.value("status", "pending").c_str());
    }
    ImGui::EndChild();
    ImGui::TreePop();
}
}
