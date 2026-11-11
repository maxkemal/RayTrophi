#pragma once

#include "Fluid/LiquidEmissionAuthoring.h"
#include "imgui.h"

namespace RayTrophiSim::Fluid {

inline bool drawLiquidEmissionControls(SimulationFlowSourceDesc& source) {
    const char* labels[] = { "Particle Rate", "Liquid Inflow", "Liquid Outflow" };
    const char* names[] = { "rate", "inflow", "outflow" };
    int selected = static_cast<int>(source.liquid_emission);
    bool changed = false;
    if (ImGui::Combo("Liquid Emission", &selected, labels, IM_ARRAYSIZE(labels))) {
        std::string error;
        changed = setLiquidEmission(source, names[selected], error);
        if (!changed) {
            ImGui::TextWrapped("%s", error.c_str());
        }
    }
    if (source.liquid_emission == SimulationFlowSourceDesc::LiquidEmission::Inflow) {
        ImGui::TextWrapped("Nozzle flow is determined by area and speed; rate keys are ignored.");
        ImGui::Text("Target: %.6g m3/s", source.fluid_flux_target_m3_s);
        ImGui::Text("Added volume / dt: %.6g m3/s", source.fluid_flux_m3_s);
        ImGui::TextWrapped("First fill includes the upstream reservoir. Compare steady flow after filling.");
    } else if (source.liquid_emission == SimulationFlowSourceDesc::LiquidEmission::Outflow) {
        ImGui::Text("Drained: %.6g kg / step", source.drained_mass_kg_step);
        ImGui::Text("Total drained: %.6g kg", source.drained_mass_kg_total);
    }
    if (source.fluid_overpack_dropped_total > 0) {
        ImGui::Text("Rate overpack dropped: %d", source.fluid_overpack_dropped_total);
    }
    if (!source.liquid_emission_status.empty()) {
        ImGui::TextWrapped("%s", source.liquid_emission_status.c_str());
    }
    return changed;
}

} // namespace RayTrophiSim::Fluid
