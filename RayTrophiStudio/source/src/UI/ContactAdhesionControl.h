#pragma once

#include "imgui.h"
#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid {

inline bool drawContactAdhesionControl(float& adhesion_pa) {
    constexpr float maximum_pa = 1.0e6f;
    const float before = adhesion_pa;
    // A logarithmic DragFloat normalizes its drag speed by the full range.
    // Starting at zero in a 1 MPa range made ordinary dragging nearly inert.
    // A real slider addresses the complete range with its absolute position.
    bool changed = ImGui::SliderFloat("Contact Adhesion (Pa)##CollTab",
        &adhesion_pa, 0.0f, maximum_pa, "%.0f Pa",
        ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("MPM surface adhesion, 0 to 1,000,000 Pa (1 MPa).\n"
            "0 = off; 1,000 Pa = 1 kPa. This is a stress, not a 0..1 factor.\n"
            "The logarithmic slider reaches the whole range. Use the numeric\n"
            "field below for an exact value without Ctrl+click.\n"
            "Higher stress resists separation; the material must carry the load.");
    }
    changed |= ImGui::InputFloat("Exact Adhesion (Pa)##CollTab", &adhesion_pa,
        100.0f, 1000.0f, "%.0f");
    if (changed) {
        adhesion_pa = std::isfinite(adhesion_pa)
            ? std::clamp(adhesion_pa, 0.0f, maximum_pa) : before;
    }
    return changed && adhesion_pa != before;
}

} // namespace RayTrophiSim::Fluid
