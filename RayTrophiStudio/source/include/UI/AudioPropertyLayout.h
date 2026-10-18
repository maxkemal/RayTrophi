#pragma once
#include "imgui.h"
#include <cfloat>
#include <string>
#include <array>

namespace AudioUI {
// A label above its full-width field never consumes a fixed right-side gutter.
inline std::string property(const char* label) {
    ImGui::TextWrapped("%s", label);
    ImGui::SetNextItemWidth(-FLT_MIN);
    return std::string("##") + label;
}

// Keep an uncommitted gesture locally. Commit through the authoring service once
// on release, so a slider drag does not create an undo entry for every frame.
inline bool slider(const char* label, float& value, float low, float high,
                   const char* format = "%.2f", ImGuiSliderFlags flags = 0) {
    static ImGuiID editing = 0;
    static float draft = 0;
    static int lastFrame = -2;
    const auto field = property(label);
    const auto id = ImGui::GetID(field.c_str());
    const auto frame = ImGui::GetFrameCount();
    if (editing == id && lastFrame == frame - 1) {
        value = draft;
    }
    ImGui::SliderFloat(field.c_str(), &value, low, high, format, flags);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Drag to adjust; release to apply. Ctrl+click to enter a precise value.");
    }
    const bool committed = ImGui::IsItemDeactivatedAfterEdit();
    if (ImGui::IsItemActive()) {
        editing = id;
        draft = value;
        lastFrame = frame;
    } else if (editing == id) {
        editing = 0;
    }
    return committed;
}

inline bool dragVector(const char* label, float* value, float speed) {
    static ImGuiID editing = 0;
    static std::array<float, 3> draft{};
    static int lastFrame = -2;
    const auto field = property(label);
    const auto id = ImGui::GetID(field.c_str());
    const auto frame = ImGui::GetFrameCount();
    if (editing == id && lastFrame == frame - 1) {
        for (int axis = 0; axis < 3; ++axis) {
            value[axis] = draft[axis];
        }
    }
    ImGui::DragFloat3(field.c_str(), value, speed, 0, 0, "%.3f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Drag an axis value; release to apply. Ctrl+click for numeric entry.");
    }
    const bool committed = ImGui::IsItemDeactivatedAfterEdit();
    if (ImGui::IsItemActive()) {
        editing = id;
        for (int axis = 0; axis < 3; ++axis) {
            draft[axis] = value[axis];
        }
        lastFrame = frame;
    } else if (editing == id) {
        editing = 0;
    }
    return committed;
}
}
