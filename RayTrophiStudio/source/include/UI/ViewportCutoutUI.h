#pragma once
#include "Api/RtApiViewportCutout.h"
#include "imgui.h"
inline void DrawViewportAutomaticCutout() {
    bool enabled = rtapi::viewportAutomaticCutout();
    static std::string error;
    if (ImGui::Checkbox("Automatic opacity cutout", &enabled)) {
        const auto r = rtapi::setViewportAutomaticCutout(enabled);
        error = r.ok ? std::string{} : r.error;
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip(
        "Scene-wide viewport cutout at alpha 0.5, including scattered assets.\n"
        "Keeps physical transmission. Disable for smooth opacity blending.\n"
        "Off restores authored material coverage; final renders are unchanged.");
    if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
}
