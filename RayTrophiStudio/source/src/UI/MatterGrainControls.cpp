#include "Fluid/MatterGrain.h"
#include "ParticleSimulation.h"
#include "../Api/RtMatterModels.h"
#include "DomainPanelWidgets.h"
#include "ui_modern.h"
#include "imgui.h"

#include <cmath>
#include <map>
#include <string>

namespace RayTrophiSim::Fluid {
namespace {

// Last authoring error per domain (transient feedback, not domain state).
std::map<std::string, std::string>& grainErrors() {
    static std::map<std::string, std::string> errors;
    return errors;
}

void apply(const SimulationGridDomainDesc& domain, const MatterGrainParams& p) {
    try {
        rtapi::matterGrainSettings(domain.name, matterGrainParamsToJson(p), true);
        grainErrors().erase(domain.name);
    } catch (const std::exception& e) {
        grainErrors()[domain.name] = e.what();
    }
}

void showError(const SimulationGridDomainDesc& domain) {
    const auto found = grainErrors().find(domain.name);
    if (found != grainErrors().end()) {
        DomainUi::Reason(found->second.c_str());
    }
}

} // namespace

// Matter tab: what the grains are made of. Drawn only when the domain has a
// granular phase (grains enabled in Solvers).
void drawMatterGrainMaterial(const SimulationGridDomainDesc& domain) {
    if (domain.type != SimulationDomainType::Matter || !domain.fluid_params.grain.enabled ||
        !UIWidgets::CollapsingHeader("Granular Material (grains)##MatterGrainMaterial",
            ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    ImGui::PushID(domain.name.c_str());
    auto p = domain.fluid_params.grain;
    ImGui::TextWrapped("Transport is selected per substance: dem uses grains, mpm uses "
        "the continuum skeleton. Both can share this domain with bidirectional contact.");
    bool changed = false;
    changed |= DomainUi::Float("Real grain radius (m, 0 = simulation)", &p.represented_grain_radius_m,
        1e-5f, 0.0f, p.radius_m, "%.5f",
        "Radius of the real grains one simulated grain stands for (0.0005 m = 1 mm sand).\n"
        "Range 0..simulation radius; 0 = the simulated grain is the real grain.\n"
        "Keeps the real Bond number: liquid bridge force x (simulation / real radius)^2.\n"
        "Script: fluid.set_grain_settings(represented_grain_radius_m=...)");
    changed |= DomainUi::Slider("Packing fraction", &p.packing_fraction, .3f, .74f, "%.2f",
        "Solid fraction of a settled bed of these grains (random packing ~0.6).\n"
        "Grain density = substance bulk density / packing fraction (sand 1600 / 0.6 = 2667 kg/m^3).\n"
        "Changes grain mass, so reset particles after editing.\n"
        "Script: fluid.set_grain_settings(packing_fraction=...)");
    changed |= DomainUi::Slider("Sliding friction", &p.friction, 0.0f, 2.0f, "%.2f",
        "Coulomb friction coefficient mu between grains and against walls/meshes.\n"
        "Typical: glass beads 0.1-0.3, sand 0.5-0.7. Equivalent angle atan(mu).\n"
        "Together with rolling friction sets the angle of repose.\n"
        "Script: fluid.set_grain_settings(friction=...)");
    changed |= DomainUi::Slider("Rolling friction", &p.rolling_friction, 0.0f, 1.0f, "%.3f",
        "Rolling resistance mu_r (grain shape): round 0.02, angular sand 0.1-0.3.\n"
        "A grain holds on a slope while tan(slope) <= mu_r.\n"
        "Drives the EPSD2 rolling spring; raises the repose angle.\n"
        "Script: fluid.set_grain_settings(rolling_friction=...)");
    changed |= DomainUi::Slider("Restitution", &p.restitution, .01f, 1.0f, "%.2f",
        "Coefficient of restitution of a normal impact: rebound speed / impact speed.\n"
        "Typical: sand 0.4-0.6, glass beads 0.9; 0.01..1.\n"
        "Every contact derives its damping from it and its own effective mass.\n"
        "Script: fluid.set_grain_settings(restitution=...)");
    if (ImGui::TreeNode("Advanced material##GrainMaterialAdvanced")) {
        changed |= DomainUi::Slider("Twisting friction", &p.twisting_friction, 0.0f, 1.0f, "%.3f",
            "Torsional resistance about the contact normal (fraction of mu_r torque).\n"
            "0 = free spin about the normal; 0.1 damps spinning piles.\n"
            "Script: fluid.set_grain_settings(twisting_friction=...)");
        changed |= DomainUi::Slider("Static friction stiffness (x normal)",
            &p.tangential_stiffness_ratio, 0.0f, 1.0f, "%.3f",
            "Cundall-Strack tangential spring as a fraction of the normal stiffness.\n"
            "2/7 (0.286) matches a solid sphere's contact periods; 0 = kinetic-only sliding.\n"
            "Holds piles and slopes with static friction.\n"
            "Script: fluid.set_grain_settings(tangential_stiffness_ratio=...)");
        ImGui::TreePop();
    }
    ImGui::SeparatorText("Water");
    changed |= DomainUi::Bool("Wet grains (absorb water, liquid bridges)", &p.wet_grains,
        "Grains take water from the liquid around them and pull on each other through\n"
        "pendular liquid bridges (Willett 2000). Water is held in the grain, counted once.\n"
        "Needs liquid in the domain or birth saturation.\n"
        "Script: fluid.set_grain_settings(wet_grains=true)");
    ImGui::BeginDisabled(!p.wet_grains);
    changed |= DomainUi::Slider("Water capacity (x grain volume)", &p.water_capacity_fraction,
        0.0f, .5f, "%.3f",
        "Water one grain can hold, as a fraction of its volume (0..0.5).\n"
        "Sets saturation = held water / capacity.\n"
        "Script: fluid.set_grain_settings(water_capacity_fraction=...)");
    changed |= DomainUi::Float("Absorption rate (1/s)", &p.absorption_rate_per_s, .1f, 0.0f,
        1000.0f, "%.2f",
        "Fraction of the free capacity filled per second while submerged (0..1000).\n"
        "At most half a cell's liquid per frame; momentum is exact.\n"
        "Script: fluid.set_grain_settings(absorption_rate_per_s=...)");
    changed |= DomainUi::Float("Drying rate (1/s)", &p.drying_rate_per_s, .01f, 0.0f, 100.0f,
        "%.3f",
        "Fraction of the held water lost per second (0..100). The water leaves the\n"
        "domain and is reported as evaporated.\n"
        "Script: fluid.set_grain_settings(drying_rate_per_s=...)");
    changed |= DomainUi::Slider("Surface tension (N/m)", &p.surface_tension_n_m, 0.0f, 1.0f,
        "%.4f",
        "Liquid surface tension of the bridges (water 0.072 N/m).\n"
        "Scales the bridge force F = 2 pi R gamma cos(theta) / (1 + 2.1 s + 10 s^2).\n"
        "Script: fluid.set_grain_settings(surface_tension_n_m=...)");
    changed |= DomainUi::Slider("Contact angle (deg)", &p.contact_angle_deg, 0.0f, 89.0f, "%.1f",
        "Wetting angle of the liquid on the grain (water on quartz ~20 deg).\n"
        "0 = perfect wetting, strongest bridges.\n"
        "Script: fluid.set_grain_settings(contact_angle_deg=...)");
    changed |= DomainUi::Slider("Birth saturation", &p.birth_saturation, 0.0f, 1.0f, "%.2f",
        "Water a newly emitted grain already holds, as a fraction of capacity\n"
        "(an emitter of damp sand).\n"
        "Script: fluid.set_grain_settings(birth_saturation=...)");
    ImGui::EndDisabled();
    if (changed) {
        apply(domain, p);
    }
    showError(domain);
    ImGui::PopID();
}

// Solvers tab: whether and how the granular phase is stepped.
void drawMatterGrainSolver(const SimulationGridDomainDesc& domain,
                           const SimulationGridDomainState* state) {
    if (domain.type != SimulationDomainType::Matter ||
        !UIWidgets::CollapsingHeader("Granular Solver (grains)##MatterGrainSolver",
            ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    ImGui::PushID(domain.name.c_str());
    auto p = domain.fluid_params.grain;
    bool changed = DomainUi::Bool("Enable discrete grains", &p.enabled,
        "Substances with granular_transport=dem use physical DEM grains;\n"
        "granular_transport=mpm retains the MPM skeleton and exchanges contact impulses.\n"
        "liquid emitters in the same domain stay liquid parcels (one owner each).\n"
        "Needs Vulkan, a Closed boundary, and pore water / thermal liquid off.\n"
        "Script: fluid.set_grain_settings(enabled=true)");
    const auto blockers = matterGrainBlockers(domain);
    if (p.enabled || !blockers.empty()) {
        if (blockers.empty()) {
            const bool held = state && state->fluid_stats.mixed_step_held;
            if (held) {
                // Not greyed: a held step freezes every grain where it was born,
                // and this line is the only place that says why.
                ImGui::TextWrapped("Step held - grains do not move: %s",
                                   state->fluid_stats.gpu_status.c_str());
            } else {
                DomainUi::Reason("Ready.");
            }
        }
        for (const auto& blocker : blockers) {
            DomainUi::Reason(blocker.message.c_str());
        }
    }
    if (p.enabled) {
        changed |= DomainUi::Float("Simulation grain radius (m)", &p.radius_m, .001f, .001f, 1.0f,
            "%.4f",
            "Radius of one simulated grain (0.001..1 m). Sets cost: grain count ~ 1/r^3.\n"
            "Mass falls with r^3 while Contact stiffness stays in N/m, so a smaller\n"
            "grain is a stiffer contact: the substep shrinks ~ r^1.5. Lower stiffness\n"
            "with the radius (k ~ r keeps the same impact overlap fraction).\n"
            "A real grain size below it is set in Matter (coarse graining).\n"
            "Reset particles after editing.\n"
            "Script: fluid.set_grain_settings(radius_m=...)");
        changed |= DomainUi::Float("Contact stiffness (N/m)", &p.stiffness_n_m, 100.0f, 1.0f, 1e8f,
            "%.0f",
            "Normal contact spring (1..1e8 N/m). Softer = more overlap, fewer substeps\n"
            "(substeps ~ sqrt(k)); keep overlap under ~1% of the radius.\n"
            "Script: fluid.set_grain_settings(stiffness_n_m=...)");
        int accuracy = p.contact_resolution == 12 ? 0 : p.contact_resolution == 24 ? 1
            : p.contact_resolution == 48 ? 2 : 3;
        if (DomainUi::Choice("Accuracy", &accuracy, "Draft\0Production\0Reference\0Custom\0",
                "Substeps per binary collision: Draft 12, Production 24, Reference 48.\n"
                "Production passed the convergence test (24 vs 48 within tolerance).\n"
                "Custom keeps the value under Advanced.\n"
                "Script: fluid.set_grain_settings(contact_resolution=...)")) {
            static const int presets[] = {12, 24, 48};
            if (accuracy < 3) {
                p.contact_resolution = presets[accuracy];
                changed = true;
            }
        }
        if (ImGui::TreeNode("Advanced solver##GrainSolverAdvanced")) {
            changed |= DomainUi::Int("Substeps per collision", &p.contact_resolution, 1.0f, 8, 200,
                "Substeps per binary collision duration (8..200): accuracy, not stability.\n"
                "Script: fluid.set_grain_settings(contact_resolution=...)");
            changed |= DomainUi::Int("Contact substep limit", &p.max_substeps, 1.0f, 1, 4096,
                "Safety cap on substeps per frame (1..4096); a step that needs more is held.\n"
                "Script: fluid.set_grain_settings(max_substeps=...)");
            changed |= DomainUi::Float("Sliding damping (Ns/m)", &p.sliding_damping_n_s_m, .1f,
                0.0f, 1e5f, "%.2f",
                "Viscous part of the sliding force inside the Coulomb cone (0..1e5 Ns/m).\n"
                "Script: fluid.set_grain_settings(sliding_damping_n_s_m=...)");
            changed |= DomainUi::Bool("Sleeping grains", &p.sleep,
                "A grain that stays slower than the sleep speed for the sleep time stops\n"
                "computing contacts and holds still until a faster grain touches it.\n"
                "Saves GPU time on settled piles. Never with moving colliders, MPM contact,\n"
                "liquid coupling or a force field on the grain. Off = A/B reference.\n"
                "Script: fluid.set_grain_settings(sleep=...)");
            ImGui::BeginDisabled(!p.sleep);
            changed |= DomainUi::Float("Sleep speed (m/s)", &p.sleep_speed_m_s, .0005f,
                0.0f, 1.0f, "%.4f",
                "Translation speed, and spin x radius, below which a grain counts as still\n"
                "(0..1 m/s, absolute). A slope creeping slower than this freezes.\n"
                "Script: fluid.set_grain_settings(sleep_speed_m_s=...)");
            changed |= DomainUi::Float("Sleep time (s)", &p.sleep_time_s, .01f, .01f, 10.0f,
                "%.2f",
                "How long a grain must stay still before it sleeps (0.01..10 s).\n"
                "Script: fluid.set_grain_settings(sleep_time_s=...)");
            ImGui::EndDisabled();
            ImGui::TreePop();
        }
        ImGui::SeparatorText("Liquid coupling");
        changed |= DomainUi::Bool("Liquid drag + buoyancy coupling", &p.fluid_coupling,
            "Grains and liquid parcels of this domain exchange momentum: Di Felice drag\n"
            "with the liquid's own viscosity (from its substance) and buoyancy; the liquid\n"
            "gets the opposite impulse. Off = they pass through each other (A/B only).\n"
            "Script: fluid.set_grain_settings(fluid_coupling=...)");
        ImGui::BeginDisabled(!p.fluid_coupling);
        changed |= DomainUi::Bool("Liquid sees grain volume (pressure force)", &p.volume_exclusion,
            "The liquid's projection sees the grains' pore fraction (water is displaced),\n"
            "and grains take the force of the liquid's measured acceleration instead of\n"
            "hydrostatic buoyancy.\n"
            "Script: fluid.set_grain_settings(volume_exclusion=...)");
        ImGui::EndDisabled();
    }
    if (changed) {
        apply(domain, p);
    }
    // A rejected enable repeats the blocker lines above; show only other errors.
    if (blockers.empty()) {
        showError(domain);
    }
    ImGui::PopID();
}

// Measure tab: what the last grain step did.
void drawMatterGrainReport(const SimulationGridDomainDesc& domain,
                           const SimulationGridDomainState* state) {
    if (domain.type != SimulationDomainType::Matter || !domain.fluid_params.grain.enabled ||
        !UIWidgets::CollapsingHeader("Grain Step##MatterGrainReport")) {
        return;
    }
    if (!state || !state->fluid_stats.mixed_model_step) {
        ImGui::TextDisabled("No grain step yet.");
        return;
    }
    const auto& r = state->fluid_stats.grain_report;
    ImGui::Text("DEM: %d substeps (%.2e s), limited by %s", r.substeps, r.substep_dt,
        r.limit.c_str());
    ImGui::Text("Grains %zu, liquid parcels %zu, MPM parcels %zu",
        r.grains, r.liquid_parcels, r.mpm_parcels);
    if (r.mpm_parcels > 0) {
        ImGui::Text("MPM contact events %llu, max neighbours %u, residual %.3g N s",
            static_cast<unsigned long long>(r.mpm_contact_events), r.mpm_contact_max_neighbours,
            r.mpm_contact_momentum_residual);
        if (r.common_clock) {
            ImGui::Text("Shared clock: %d transport ticks, %d continuum grid updates",
                r.substeps, r.liquid_substeps);
        } else {
            ImGui::TextDisabled("Continuum frame, then contact at each DEM substep.");
        }
        if (r.liquid_parcels > 0) {
            ImGui::TextWrapped("Liquid uses drag and buoyancy while MPM is present; "
                "porous projection awaits separate owner weights.");
            if (r.liquid_reaction_on_gpu) {
                ImGui::TextDisabled(r.liquid_support_dynamic
                    ? "Liquid support and reaction rebuilt on GPU each tick."
                    : "Liquid reaction on GPU each tick; support sampled per frame.");
            }
        }
    }
    ImGui::Text("Contacts %u (sticking %u), max per grain %u", r.contacts, r.sticking_contacts,
        r.max_contacts);
    ImGui::Text("History: %s%s", r.history_reset ? r.history_reset_reason.c_str() : "carried",
        r.history_remapped ? ", remapped" : "");
    ImGui::Text("Transfers: %s, up %zu B, down %zu B, %d batches",
        r.state_resident ? "resident" : "uploaded", r.upload_bytes, r.download_bytes,
        r.transfer_batches);
    if (r.coupling_enabled) {
        ImGui::SeparatorText("Liquid coupling");
        ImGui::Text("Coupled grains %zu, max submerged %.2f", r.coupled_grains,
            r.max_submerged_fraction);
        ImGui::Text("Drag impulse y %.4g N s, %s y %.4g N s", r.drag_impulse.y,
            r.pressure_force ? "pressure force" : "buoyancy", r.buoyancy_impulse.y);
        ImGui::Text("Momentum residual %.3g N s, unmatched %.3g N s", r.momentum_residual,
            r.unmatched_impulse);
        if (r.volume_exclusion) {
            ImGui::Text("Porous cells %zu, max solid fraction %.2f", r.porous_cells,
                r.max_solid_fraction);
        }
    }
    if (r.wet_grains) {
        ImGui::SeparatorText("Water");
        ImGui::Text("Wet grains %zu, bridges %u, max saturation %.2f", r.wet_grain_count,
            r.liquid_bridges, r.max_grain_saturation);
        ImGui::Text("Held %.4g kg, absorbed %.3g kg, evaporated %.3g kg, balance error %.2g kg",
            r.grain_water_kg, r.absorbed_kg, r.evaporated_kg, r.water_balance_error_kg);
    }
}

} // namespace RayTrophiSim::Fluid
