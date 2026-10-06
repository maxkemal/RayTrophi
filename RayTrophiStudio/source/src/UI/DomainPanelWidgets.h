#pragma once

// Standard value rows of the simulation domain panel.
//
// One width rule (label right, field 55% of the section, at least 140 px) and a
// mandatory tooltip on every value: what it does, unit and typical range, what
// it interacts with, and the script key that writes the same field. The
// tooltip is shown on disabled rows too, so a locked control can say why.
// scripts/test/check_domain_panel_fields.py inventories these calls, and
// check_matter_grain_contracts.py fails on a raw ImGui value widget in the
// helper files that use them.

#include "imgui.h"

#include <algorithm>

namespace DomainUi {

inline float itemWidth() {
    return std::max(140.0f, ImGui::GetContentRegionAvail().x * .55f);
}

inline void tooltip(const char* text) {
    if (text && *text && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("%s", text);
    }
}

inline bool Float(const char* label, float* value, float speed, float minimum, float maximum,
                  const char* format, const char* tip, ImGuiSliderFlags flags = 0) {
    ImGui::SetNextItemWidth(itemWidth());
    const bool changed = ImGui::DragFloat(label, value, speed, minimum, maximum, format,
        flags | ImGuiSliderFlags_AlwaysClamp);
    tooltip(tip);
    return changed;
}

inline bool Slider(const char* label, float* value, float minimum, float maximum,
                   const char* format, const char* tip) {
    ImGui::SetNextItemWidth(itemWidth());
    const bool changed = ImGui::SliderFloat(label, value, minimum, maximum, format,
        ImGuiSliderFlags_AlwaysClamp);
    tooltip(tip);
    return changed;
}

inline bool Int(const char* label, int* value, float speed, int minimum, int maximum,
                const char* tip) {
    ImGui::SetNextItemWidth(itemWidth());
    const bool changed = ImGui::DragInt(label, value, speed, minimum, maximum, "%d",
        ImGuiSliderFlags_AlwaysClamp);
    tooltip(tip);
    return changed;
}

inline bool Bool(const char* label, bool* value, const char* tip) {
    const bool changed = ImGui::Checkbox(label, value);
    tooltip(tip);
    return changed;
}

// `items` is ImGui's zero-separated list ("A\0B\0").
inline bool Choice(const char* label, int* index, const char* items, const char* tip) {
    ImGui::SetNextItemWidth(itemWidth());
    const bool changed = ImGui::Combo(label, index, items);
    tooltip(tip);
    return changed;
}

// Read-only value row in the same layout as an editable one.
inline void Value(const char* label, const char* text, const char* tip) {
    ImGui::TextDisabled("%s", text);
    tooltip(tip);
    ImGui::SameLine(itemWidth() + ImGui::GetStyle().ItemInnerSpacing.x);
    ImGui::TextUnformatted(label);
}

// The reason a control or section is locked, under it.
inline void Reason(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
}

} // namespace DomainUi
