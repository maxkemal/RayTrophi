#pragma once

#include "imgui.h"

#include <algorithm>
#include <functional>

// Presentation only. Callers own values, validation, keyframes and commits.
namespace PropertyLayout {

struct Metrics {
    static float labelWidth(float available) {
        const float font = ImGui::GetFontSize();
        return (std::min)(available * 0.38f, font * 9.0f);
    }

    static bool stacked(float available, float actionWidth) {
        const float gap = ImGui::GetStyle().ItemSpacing.x;
        return available - labelWidth(available) - actionWidth - gap * 2.0f
            < ImGui::GetFontSize() * 7.0f;
    }
};

// One field per scope. The action slot follows the editor, independently of
// the label width. No persistent state or copied authoring values live here.
class Row {
public:
    Row(const char* id, const char* label, bool hasAction = false) {
        ImGui::PushID(id);
        const float available = ImGui::GetContentRegionAvail().x;
        const float gap = ImGui::GetStyle().ItemSpacing.x;
        const float actionWidth = hasAction ? ImGui::GetFrameHeight() + gap : 0.0f;
        const float start = ImGui::GetCursorPosX();

        if (Metrics::stacked(available, actionWidth)) {
            ImGui::PushTextWrapPos(start + available);
            ImGui::TextUnformatted(label);
            ImGui::PopTextWrapPos();
        } else {
            ImGui::AlignTextToFramePadding();
            ImGui::PushTextWrapPos(start + Metrics::labelWidth(available));
            ImGui::TextUnformatted(label);
            ImGui::PopTextWrapPos();
            ImGui::SameLine(start + Metrics::labelWidth(available) + gap);
        }

        ImGui::SetNextItemWidth(
            (std::max)(1.0f, ImGui::GetContentRegionAvail().x - actionWidth));
    }

    ~Row() {
        ImGui::PopID();
    }

    Row(const Row&) = delete;
    Row& operator=(const Row&) = delete;
};

inline bool keyframeButton(bool keyed) {
    const float size = ImGui::GetFrameHeight();
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    const bool clicked = ImGui::Button("##keyframe", ImVec2(size, size));
    ImGui::PopStyleColor();

    const ImVec2 origin = ImGui::GetItemRectMin();
    const ImVec2 center(origin.x + size * 0.5f, origin.y + size * 0.5f);
    const float radius = size * 0.18f;
    const ImVec2 top(center.x, center.y - radius);
    const ImVec2 right(center.x + radius, center.y);
    const ImVec2 bottom(center.x, center.y + radius);
    const ImVec2 left(center.x - radius, center.y);
    const ImU32 color = ImGui::GetColorU32(
        keyed ? ImGuiCol_CheckMark : ImGuiCol_TextDisabled);
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    if (keyed) {
        drawList->AddQuadFilled(top, right, bottom, left, color);
    } else {
        drawList->AddQuad(top, right, bottom, left, color);
    }
    return clicked;
}

inline bool slider(const char* id, const char* label, float* value,
                   float minimum, float maximum, const char* format,
                   bool keyed = false, const std::function<void()>& onKey = {}) {
    Row row(id, label, static_cast<bool>(onKey));
    const bool changed = ImGui::SliderFloat("##value", value, minimum, maximum, format);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("%s: drag to adjust; Ctrl+click to enter a value", label);
    }
    if (onKey) {
        ImGui::SameLine();
        if (keyframeButton(keyed)) {
            onKey();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(keyed ? "Remove %s keyframe" : "Add %s keyframe", label);
        }
    }
    return changed;
}

// An opt-in group for migrated panels: no enclosing rectangle, no per-domain
// accent. Existing section wrappers remain unchanged elsewhere.
inline bool beginGroup(const char* title, bool defaultOpen = true) {
    ImGui::PushID(title);
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(
        ImGuiCol_HeaderHovered, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));
    ImGui::PushStyleColor(
        ImGuiCol_HeaderActive, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgActive));
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_SpanAvailWidth
        | ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    if (defaultOpen) {
        flags |= ImGuiTreeNodeFlags_DefaultOpen;
    }
    const bool open = ImGui::TreeNodeEx(title, flags);
    ImGui::PopStyleColor(3);
    if (open) {
        ImGui::Spacing();
    } else {
        ImGui::PopID();
    }
    return open;
}

inline void endGroup() {
    ImGui::Spacing();
    ImGui::PopID();
}

} // namespace PropertyLayout
