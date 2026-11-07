#pragma once

#include "imgui.h"

#include <algorithm>

// Shared presentation contract. Labels and callbacks retain their existing IDs
// and business behavior; callers choose a layout role rather than pixel sizes.
namespace PanelControls {

inline ImVec2 padding() {
    const float font = ImGui::GetFontSize();
    return ImVec2(font * 0.5f, font * 0.25f);
}

inline float height() {
    return ImGui::GetFontSize() + padding().y * 2.0f;
}

inline float actionWidth(int columns = 1) {
    const float available = ImGui::GetContentRegionAvail().x;
    const int count = (std::max)(1, columns);
    return (std::max)(1.0f,
        (available - ImGui::GetStyle().ItemSpacing.x * (count - 1)) / count);
}

inline void heading(const char* text) {
    ImGui::Spacing();
    ImGui::TextDisabled("%s", text);
}

inline bool beginCombo(const char* id, const char* preview, bool fullWidth = true) {
    if (fullWidth) {
        ImGui::SetNextItemWidth(-1.0f);
    }
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, padding());
    const bool open = ImGui::BeginCombo(id, preview);
    ImGui::PopStyleVar();
    return open;
}

inline bool button(const char* label, const ImVec2& requested = ImVec2(0, 0),
                   bool enabled = true) {
    ImGui::BeginDisabled(!enabled);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, padding());
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_FrameBg));
    ImGui::PushStyleColor(
        ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));
    ImGui::PushStyleColor(
        ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgActive));
    const ImVec2 size(requested.x, requested.y == 0.0f ? height() : requested.y);
    const bool clicked = ImGui::Button(label, size);
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar();
    ImGui::EndDisabled();
    return clicked && enabled;
}

} // namespace PanelControls
