#pragma once

#include "Fluid/DomainWallSettings.h"
#include "imgui.h"

namespace RayTrophiSim::Fluid {

inline bool drawDomainWallControls(APICSolverParams& params, bool active) {
    ImGui::SeparatorText("Domain Walls (Liquid / MPM / DEM)");
    ImGui::BeginDisabled(!active);
    float friction = params.wall_friction;
    float restitution = params.wall_damping;
    bool changed = ImGui::DragFloat("Wall Friction", &friction, 0.01f,
        0.0f, 2.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Coulomb coefficient for all six domain walls.\n"
            "0 = free slip. Higher values resist sliding under contact load.\n"
            "Mesh colliders use their own surface settings.");
    }
    changed |= ImGui::DragFloat("Wall Restitution", &restitution, 0.01f,
        0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Liquid / DEM normal rebound: 0 = no bounce, 1 = elastic bounce.\n"
            "MPM snow and elastic solids rebound through their material stiffness;\n"
            "their grid wall contact removes inward motion without extra bounce.");
    }
    float adhesion = params.wall_adhesion_pa;
    changed |= ImGui::DragFloat("Wall Adhesion (Pa)", &adhesion, 10.0f,
        0.0f, 1.0e9f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Finite tensile and cohesive wall traction. 0 = no adhesion.\n"
            "Resists separation and sliding near closed walls; does not set viscosity or wetting angle.");
    }
    ImGui::EndDisabled();
    if (!active) {
        ImGui::TextDisabled("Active only with Closed boundaries.");
    }
    if (changed) {
        std::string error;
        return patchDomainWallSettings(params,
            {{"friction", friction}, {"restitution", restitution}, {"adhesion_pa", adhesion}}, error);
    }
    return false;
}

inline bool drawDomainWallControls(APICSolverParams& params) {
    return drawDomainWallControls(params, domainWallContact(params).closed);
}

} // namespace RayTrophiSim::Fluid
