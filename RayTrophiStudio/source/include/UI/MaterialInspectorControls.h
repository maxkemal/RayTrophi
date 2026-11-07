#pragma once

#include "UI/MaterialInspectorSchema.h"
#include "UI/PanelControls.h"
#include "UI/PropertyLayout.h"
#include "UI/ColorControls.h"

#include <string>

namespace MaterialInspectorControls {

inline std::string labelFor(const char* id) {
    if (const auto* field = MaterialInspectorSchema::find(id)) {
        return field->label;
    }
    const std::string label(id);
    return label.substr(0, label.find("##"));
}

inline void unavailableKey() {
    ImGui::SameLine();
    ImGui::BeginDisabled();
    PropertyLayout::keyframeButton(false);
    ImGui::EndDisabled();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip("No keyframe action is connected for this field in this editor.");
    }
}

template <typename Editor> bool field(const char* id, Editor&& editor) {
    const std::string label = labelFor(id);
    ImGui::BeginGroup();
    bool changed = false;
    {
        PropertyLayout::Row row(id, label.c_str(), true);
        changed = editor();
        unavailableKey();
    }
    ImGui::EndGroup();
    return changed;
}

inline bool sliderFloat(const char* id, float* value, float minimum, float maximum,
                        const char* format = "%.3f") {
    return field(id,
                 [&]() { return ImGui::SliderFloat("##value", value, minimum, maximum, format); });
}

inline bool sliderInt(const char* id, int* value, int minimum, int maximum) {
    return field(id, [&]() { return ImGui::SliderInt("##value", value, minimum, maximum); });
}

inline bool dragFloat3(const char* id, float* value, float speed, float minimum, float maximum,
                       const char* format) {
    return field(
        id, [&]() { return ImGui::DragFloat3("##value", value, speed, minimum, maximum, format); });
}

inline bool checkbox(const char* id, bool* value) {
    return field(id, [&]() { return ImGui::Checkbox("##value", value); });
}

inline bool combo(const char* id, int* value, const char* const* names, int count) {
    return field(id, [&]() {
        bool changed = false;
        const char* preview = *value >= 0 && *value < count ? names[*value] : "Choose a preset...";
        if (PanelControls::beginCombo("##value", preview, false)) {
            for (int i = 0; i < count; ++i) {
                if (ImGui::Selectable(names[i], *value == i)) {
                    *value = i;
                    changed = true;
                }
                if (*value == i) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }
        return changed;
    });
}

inline bool color(const char* id, float* value, bool keyed = false,
                  const std::function<void()>& onKey = {}) {
    const std::string label = labelFor(id);
    ImGui::BeginGroup();
    bool changed = false;
    {
        ImGui::PushID(id);
        const float actionWidth = ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.x;
        ImGui::SetNextItemWidth((std::max)(1.0f, ImGui::CalcItemWidth() - actionWidth));
        changed = ColorControls::field3(label.c_str(), value);
        if (onKey) {
            ImGui::SameLine();
            if (PropertyLayout::keyframeButton(keyed)) {
                onKey();
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(keyed ? "Remove %s keyframe" : "Add %s keyframe", label.c_str());
            }
        } else {
            unavailableKey();
        }
        ImGui::PopID();
    }
    ImGui::EndGroup();
    return changed;
}

inline bool colorEdit3(const char* id, float* value) {
    return color(id, value);
}

inline bool smartFloat(const char* id, const char* legacyLabel, float* value, float minimum,
                       float maximum, const char* format, bool keyed,
                       const std::function<void()>& onKey, int /*segments*/) {
    const std::string label =
        MaterialInspectorSchema::find(id) ? labelFor(id) : labelFor(legacyLabel);
    if (onKey) {
        return PropertyLayout::slider(id, label.c_str(), value, minimum, maximum, format, keyed,
                                      onKey);
    }
    ImGui::PushID(id);
    const bool changed =
        field(id, [&]() { return ImGui::SliderFloat("##value", value, minimum, maximum, format); });
    ImGui::PopID();
    return changed;
}

inline void nextPreset() {
    const float needed = ImGui::GetFontSize() * 7.0f;
    const float end = ImGui::GetItemRectMax().x;
    const float right = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    if (right - end > needed) {
        ImGui::SameLine();
    }
}

} // namespace MaterialInspectorControls
