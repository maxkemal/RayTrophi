#include "Fluid/MatterGrain.h"
#include "ParticleSimulation.h"
#include "../Api/RtMatterModels.h"
#include "imgui.h"

namespace RayTrophiSim::Fluid {

void drawMatterGrainControls(const SimulationGridDomainDesc& domain) {
    if (domain.type != SimulationDomainType::Matter ||
        !ImGui::CollapsingHeader("Dry Grain Solver (candidate)")) {
        return;
    }
    ImGui::PushID(domain.name.c_str());
    auto p = domain.fluid_params.grain;
    bool changed = ImGui::Checkbox("Enable discrete grains", &p.enabled);
    changed |= ImGui::DragFloat("Physical radius (m)", &p.radius_m, .001f, .001f, 1.0f);
    changed |= ImGui::DragFloat("Contact stiffness (N/m)", &p.stiffness_n_m,
        100.0f, 1.0f, 1e8f, "%.0f");
    changed |= ImGui::DragFloat("Normal damping (Ns/m)", &p.normal_damping_n_s_m,
        .1f, 0.0f, 1e5f);
    changed |= ImGui::DragFloat("Sliding damping (Ns/m)", &p.sliding_damping_n_s_m,
        .1f, 0.0f, 1e5f);
    changed |= ImGui::SliderFloat("Sliding friction", &p.friction, 0.0f, 2.0f);
    changed |= ImGui::SliderFloat("Rolling friction", &p.rolling_friction, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Twisting friction", &p.twisting_friction, 0.0f, 1.0f);
    changed |= ImGui::SliderFloat("Static friction stiffness (x normal)",
        &p.tangential_stiffness_ratio, 0.0f, 1.0f, "%.3f");
    changed |= ImGui::SliderFloat("Packing fraction", &p.packing_fraction, .3f, .74f, "%.2f");
    changed |= ImGui::DragInt("Substeps per collision", &p.contact_resolution, 1.0f, 8, 200);
    changed |= ImGui::DragInt("Contact substep limit", &p.max_substeps, 1.0f, 1, 4096);
    int solver = p.solver_kind == "xpbd" ? 1 : 0;
    if (ImGui::Combo("Solver (comparison)", &solver, "DEM (force, history)\0XPBD (positional)\0")) {
        p.solver_kind = solver == 1 ? "xpbd" : "dem";
        changed = true;
    }
    if (solver == 1) {
        changed |= ImGui::DragInt("XPBD substeps per frame", &p.xpbd_substeps, 1.0f, 4, 512);
    }
    changed |= ImGui::Checkbox("Liquid drag + buoyancy coupling", &p.fluid_coupling);
    changed |= ImGui::DragFloat("Liquid viscosity for drag (Pa s)", &p.drag_viscosity_pa_s,
        1e-4f, 1e-6f, 1e3f, "%.5f", ImGuiSliderFlags_Logarithmic);
    ImGui::BeginDisabled(!p.fluid_coupling);
    changed |= ImGui::Checkbox("Liquid sees grain volume (pressure force)", &p.volume_exclusion);
    ImGui::EndDisabled();
    changed |= ImGui::Checkbox("Wet grains (absorb water, liquid bridges)", &p.wet_grains);
    ImGui::BeginDisabled(!p.wet_grains);
    changed |= ImGui::SliderFloat("Water capacity (x grain volume)", &p.water_capacity_fraction,
        0.0f, .5f, "%.3f");
    changed |= ImGui::DragFloat("Absorption rate (1/s)", &p.absorption_rate_per_s, .1f, 0.0f, 1000.0f);
    changed |= ImGui::DragFloat("Drying rate (1/s)", &p.drying_rate_per_s, .01f, 0.0f, 100.0f);
    changed |= ImGui::SliderFloat("Surface tension (N/m)", &p.surface_tension_n_m, 0.0f, 1.0f, "%.4f");
    changed |= ImGui::SliderFloat("Contact angle (deg)", &p.contact_angle_deg, 0.0f, 89.0f);
    changed |= ImGui::DragFloat("Represented grain radius (m, 0 = same)",
        &p.represented_grain_radius_m, 1e-5f, 0.0f, p.radius_m, "%.5f");
    changed |= ImGui::SliderFloat("Birth saturation", &p.birth_saturation, 0.0f, 1.0f);
    ImGui::EndDisabled();
    static std::string error;
    if (changed) {
        try {
            rtapi::matterGrainSettings(domain.name, matterGrainParamsToJson(p), true);
            error.clear();
        } catch (const std::exception& e) {
            error = e.what();
        }
    }
    ImGui::TextWrapped("Granular emitters become grains; liquid emitters in the same domain "
        "stay liquid parcels (one transport owner each). Closed Vulkan, static PlaneY/flat mesh "
        "colliders. Reset particles before editing. Disable Pore Water and thermal physics.");
    ImGui::TextWrapped("Liquid coupling: Di Felice drag against the liquid in the grain's cells; "
        "the liquid gets the opposite impulse. With grain volume on, the liquid's projection "
        "sees the pore fraction and the grains feel its pressure gradient (otherwise "
        "hydrostatic buoyancy, and a pile is drag-only to the liquid).");
    ImGui::TextWrapped("Each carrier is one physical sphere; its mass is the substance bulk "
        "density / packing fraction x sphere volume. Static friction stiffness 0.286 (2/7) "
        "holds piles with a Cundall-Strack spring; 0 is kinetic-only sliding. Wet grains hold "
        "water from the liquid (or from birth) and pull on each other through pendular liquid "
        "bridges; a coarse grain standing for small real grains keeps their Bond number via "
        "the represented radius.");
    if (!error.empty()) {
        ImGui::TextWrapped("%s", error.c_str());
    }
    ImGui::PopID();
}

} // namespace RayTrophiSim::Fluid
