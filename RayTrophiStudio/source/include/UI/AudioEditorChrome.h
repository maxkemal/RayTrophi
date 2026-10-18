#pragma once
#include "imgui.h"
#include <algorithm>

namespace AudioUI {
// Match Terrain node editor: a 13 px seam with a centered 3.5 px chevron.
inline bool audioPanelToggleStrip(const char* id, bool visible, bool leftPanel,
                                 const char* panelName) {
    constexpr float width = 13.0f;
    ImGui::BeginChild(id, ImVec2(width, 0), false,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    const auto p = ImGui::GetCursorScreenPos();
    const auto size = ImGui::GetContentRegionAvail();
    ImGui::InvisibleButton("##toggle", ImVec2(width, std::max(24.0f, size.y)));
    const bool clicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);
    const bool hovered = ImGui::IsItemHovered();
    auto* draw = ImGui::GetWindowDrawList();
    if (hovered) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        ImGui::SetTooltip("%s %s", visible ? "Hide" : "Show", panelName);
        draw->AddRectFilled(p, ImVec2(p.x + width, p.y + size.y),
            IM_COL32(90, 130, 200, 60));
    }
    const float cx = p.x + width * 0.5f;
    const float cy = p.y + size.y * 0.5f;
    constexpr float arrow = 3.5f;
    const auto color = hovered ? IM_COL32(220, 230, 245, 255) :
        IM_COL32(130, 135, 145, 255);
    const bool pointLeft = leftPanel ? visible : !visible;
    const float direction = pointLeft ? -1.0f : 1.0f;
    draw->AddTriangleFilled(ImVec2(cx - direction * arrow * 0.7f, cy - arrow),
        ImVec2(cx - direction * arrow * 0.7f, cy + arrow),
        ImVec2(cx + direction * arrow * 0.9f, cy), color);
    ImGui::EndChild();
    return clicked;
}

inline void audioPanelSplitter(const char* id, float& width, bool leftPanel,
                               float minimum, float maximum) {
    ImGui::SameLine(0, 0);
    ImGui::InvisibleButton(id, ImVec2(6, std::max(24.0f, ImGui::GetContentRegionAvail().y)));
    if (ImGui::IsItemActive()) {
        width = std::clamp(width + (leftPanel ? 1.0f : -1.0f) *
            ImGui::GetIO().MouseDelta.x, minimum, maximum);
    }
    const bool hovered = ImGui::IsItemHovered() || ImGui::IsItemActive();
    if (hovered) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
    ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(),
        ImGui::GetItemRectMax(), hovered ? IM_COL32(100, 150, 255, 200) :
        IM_COL32(80, 80, 90, 120));
}
} // namespace AudioUI
