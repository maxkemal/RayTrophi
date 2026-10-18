#pragma once
#include "imgui.h"
#include "json.hpp"

namespace AudioUI {
inline bool stateButton(const char* label, bool active, const char* hint = nullptr) {
    if (active) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_Header));
        ImGui::PushStyleColor(ImGuiCol_Border, ImGui::GetStyleColorVec4(ImGuiCol_CheckMark));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    }
    const bool pressed = ImGui::Button(label);
    if (active) {
        ImGui::PopStyleVar();
        ImGui::PopStyleColor(2);
    }
    if (hint && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("%s", hint);
    }
    return pressed;
}

inline void sourceStatus(const nlohmann::json& source, const nlohmann::json& runtime) {
    const bool valid = runtime.value("valid", false);
    const bool playing = runtime.value("playing", false);
    const bool suspended = runtime.value("automatic_suspended", false);
    const bool reactive = source.at("reaction").at("driver") != "manual";
    const bool automatic = source.at("reaction").at("automatic").get<bool>();
    const bool procedural = !source.value("generator", nlohmann::json()).is_null();
    const bool draining = runtime.contains("generator_runtime") &&
        runtime.at("generator_runtime").value("state", "") == "draining";
    const char* label = !valid ? "Position unavailable" :
        runtime.value("timeline_held", false) ? "Timeline paused" :
        suspended && automatic ? "Automatic playback suspended" : draining ? "Sound tail fading" :
        playing ? "Playing" :
        runtime.value("playback_state", "stopped") == "paused" ? "Paused" :
        reactive && automatic ? "Waiting for activity" : "Stopped";
    const auto color = ImGui::GetStyleColorVec4(!valid ? ImGuiCol_PlotHistogram :
        playing ? ImGuiCol_CheckMark : ImGuiCol_TextDisabled);
    ImGui::TextColored(color, "%s", label);
    ImGui::TextDisabled("%s / %s", procedural ? "Generated sound" : "Audio file",
        source.at("bus").get<std::string>().c_str());
    if (!valid) {
        ImGui::TextWrapped("Check the source's parent and position in Space.");
    } else if (suspended && automatic) {
        ImGui::TextWrapped("Play this source to resume its automatic control.");
    } else if (reactive && automatic && !playing) {
        ImGui::TextWrapped("Playback depends on the linked activity and scene transport.");
    }
    if (playing && runtime.value("effective_gain", 0.0f) <= 0) {
        ImGui::TextWrapped("Playback is running, but the source gain is zero.");
    }
}
}
