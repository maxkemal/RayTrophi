#include "Fluid/MatterGrain.h"
#include "Fluid/MatterSubstanceState.h"
#include "ParticleSimulation.h"
#include "MaterialStateField.h"
#include "../Api/RtMatterModels.h"
#include "DomainPanelWidgets.h"
#include "ui_modern.h"
#include "imgui.h"

#include <algorithm>
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
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.42f, 0.25f, 1.0f));
        ImGui::TextWrapped("SETTINGS NOT APPLIED");
        ImGui::TextWrapped("%s", found->second.c_str());
        ImGui::PopStyleColor();
        ImGui::Separator();
    }
}

void drawGrainSizes(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState* state) {
    struct Draft {
        MatterGrainParams value;
        uint64_t saved_signature = 0;
        bool initialized = false;
        bool dirty = false;
    };
    static std::map<std::string, Draft> drafts;
    auto& draft = drafts[domain.name];
    const uint64_t signature = matterGrainSizeSignature(domain.fluid_params.grain);
    if (!draft.initialized || signature != draft.saved_signature) {
        draft.value = domain.fluid_params.grain;
        draft.saved_signature = signature;
        draft.initialized = true;
        draft.dirty = false;
    }
    auto& sizes = draft.value;
    ImGui::SeparatorText("Physical grain sizes");
    const bool live = state && !state->particles.empty();
    if (live) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.75f, 0.25f, 1.0f));
        ImGui::TextWrapped("RESET REQUIRED: existing grains keep their birth size. "
            "You can prepare settings below, then reset particles and apply.");
        ImGui::PopStyleColor();
    }
    draft.dirty |= DomainUi::Float("Minimum radius (m)", &sizes.radius_m,
        .001f, .001f, 1.0f, "%.4f", "Edit both radii, then Apply size settings.");
    draft.dirty |= DomainUi::Float("Maximum radius (m)", &sizes.radius_max_m,
        .001f, 0.0f, 1.0f, "%.4f", "0 = fixed size. Maximum up to 4x minimum.");
    ImGui::TextWrapped("Allowed maximum: %.4f to %.4f m, or 0 for fixed size.",
        sizes.radius_m, std::min(1.0f, 4.0f * sizes.radius_m));
    int distribution = sizes.size_distribution == "bimodal" ? 2 :
        sizes.size_distribution == "log_uniform" ? 1 : 0;
    if (DomainUi::Choice("Size distribution", &distribution,
            "Uniform\0Log uniform\0Sand + large stones\0",
            "Sand + large stones gives direct control over the small/large count ratio.")) {
        const char* modes[] = {"uniform", "log_uniform", "bimodal"};
        sizes.size_distribution = modes[distribution];
        draft.dirty = true;
    }
    if (distribution == 2) {
        float percent = sizes.large_grain_fraction * 100.0f;
        if (ImGui::SliderFloat("Large stones (%)", &percent, 0.0f, 100.0f, "%.1f%%")) {
            sizes.large_grain_fraction = percent * 0.01f;
            draft.dirty = true;
        }
        ImGui::TextWrapped("By count: small sand %.1f%% / large stones %.1f%%.",
            100.0f - percent, percent);
        const double ratio = matterGrainMaximumRadius(sizes) / sizes.radius_m;
        const double weighted = sizes.large_grain_fraction * ratio * ratio * ratio;
        const double total = weighted + 1.0 - sizes.large_grain_fraction;
        ImGui::TextWrapped("Large stones occupy about %.1f%% of volume and mass "
            "at equal density. A few large stones can look dominant.",
            total > 0.0 ? 100.0 * weighted / total : 0.0);
    } else {
        ImGui::TextWrapped("For mostly small sand with occasional large stones, "
            "choose Sand + large stones and set the large-stone percentage.");
    }
    if (ImGui::Button("Mostly sand: 95% / 5%")) {
        sizes.size_distribution = "bimodal";
        sizes.large_grain_fraction = 0.05f;
        draft.dirty = true;
    }
    draft.dirty |= DomainUi::Int("Size seed", &sizes.size_seed, 1.0f, 0, 2147483647,
        "Stable per grain identity. Reset particles before applying.");
    auto candidate = domain.fluid_params.grain;
    const nlohmann::json patch = {{"radius_m", sizes.radius_m},
        {"radius_max_m", sizes.radius_max_m}, {"size_distribution", sizes.size_distribution},
        {"large_grain_fraction", sizes.large_grain_fraction}, {"size_seed", sizes.size_seed}};
    std::string error;
    const bool valid = patchMatterGrainParams(patch, candidate, error);
    if (!valid) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.42f, 0.25f, 1.0f));
        ImGui::TextWrapped("CHECK SIZE VALUES: %s", error.c_str());
        ImGui::PopStyleColor();
    }
    if (draft.dirty) {
        ImGui::TextWrapped("Size changes are pending; the simulation still uses saved settings.");
    }
    ImGui::BeginDisabled(live || !valid || !draft.dirty);
    if (ImGui::Button("Apply size settings")) {
        apply(domain, candidate);
        if (grainErrors().find(domain.name) == grainErrors().end()) {
            draft.dirty = false;
        }
    }
    ImGui::EndDisabled();
    if (draft.dirty && ImGui::Button("Discard size changes")) {
        draft.value = domain.fluid_params.grain;
        draft.dirty = false;
    }
    showError(domain);
    ImGui::Separator();
}

} // namespace

// Matter tab: what the grains are made of. The material is the DEM
// substance's and is edited in its substance editor (Default Substance or the
// source's); this block says which substance runs and what it gives the
// grains, and never holds a value of its own (MADDE_UI_TEK_OTORITE U2/U4).
void drawMatterGrainMaterial(const SimulationGridDomainDesc& domain,
                             const MatterGrainOwnership& ownership) {
    if (domain.type != SimulationDomainType::Matter || !ownership.wanted ||
        !UIWidgets::CollapsingHeader("Granular Material (grains)##MatterGrainMaterial",
            ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    const auto* s = tryFindSubstance(ownership.substance);
    if (!s) {
        DomainUi::Reason(("Substance '" + ownership.substance + "' is not in the library").c_str());
        return;
    }
    ImGui::PushID(domain.name.c_str());
    ImGui::TextWrapped("Grain material: %s. Edit it in the substance editor under the source or "
        "the Default Substance (Derive to customize a built-in).", ownership.substance.c_str());
    ImGui::Text("Friction %.2f   Rolling %.3f   Restitution %.2f",
                s->grain_friction, s->grain_rolling_friction, s->grain_restitution);
    const float packing = std::clamp(s->grain_packing_fraction, .3f, .74f);
    ImGui::Text("Packing %.2f   Grain density %.0f kg/m^3", packing, s->density / packing);
    if (s->grain_real_radius_m > 0.0f) {
        ImGui::Text("Real grain radius %.5f m (coarse graining)", s->grain_real_radius_m);
    }
    if (ownership.wet_grains) {
        ImGui::Text("Wet grains: on (water capacity %.3f of the grain volume)",
                    s->grain_water_capacity_fraction);
    } else if (s->grain_water_capacity_fraction <= 0.0f) {
        ImGui::TextDisabled("Wet grains: off - %s holds no water (grain_water_capacity_fraction = 0)",
                            ownership.substance.c_str());
    } else {
        ImGui::TextDisabled("Wet grains: off - no liquid here and no source pours wet grains");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Wet grains follow the substances: on when the grain substance holds\n"
                          "water and the domain has liquid (or a source pours wet grains).\n"
                          "Script: fluid.matter_models(domain=...)['grain_ownership']");
    }
    for (const auto& note : ownership.notes) {
        DomainUi::Reason(note.c_str());
    }
    ImGui::PopID();
}

// Solvers tab: whether and how the granular phase is stepped.
void drawMatterGrainSolver(const SimulationGridDomainDesc& domain,
                           const SimulationGridDomainState* state,
                           const MatterGrainOwnership& ownership) {
    if (domain.type != SimulationDomainType::Matter ||
        !UIWidgets::CollapsingHeader("Granular Solver (grains)##MatterGrainSolver",
            ImGuiTreeNodeFlags_DefaultOpen)) {
        return;
    }
    ImGui::PushID(domain.name.c_str());
    auto p = domain.fluid_params.grain;
    bool changed = false;
    // No switch: grains follow the substances. This block only says which
    // solver the granular material gets and, when it is not DEM, why.
    if (!ownership.wanted) {
        ImGui::TextWrapped("Solver: MPM continuum - no substance here uses DEM grains.");
    } else if (ownership.reset_pending) {
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "%s", ownership.enabled
            ? "Solver: DEM grains until particles are reset (no substance asks for them now)."
            : "Reset particles to start DEM grains - live particles keep their solver.");
    } else if (!ownership.blockers.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f),
            "%s asks for DEM grains but runs as MPM:", ownership.substance.c_str());
        for (const auto& blocker : ownership.blockers) {
            DomainUi::Reason(blocker.c_str());
        }
    } else if (state && state->fluid_stats.mixed_step_held) {
        // Not greyed: a held step freezes every grain where it was born,
        // and this line is the only place that says why.
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.42f, 0.25f, 1.0f));
        ImGui::TextWrapped("SIMULATION STOPPED - grains cannot move");
        ImGui::TextWrapped("%s", state->fluid_stats.gpu_status.c_str());
        ImGui::PopStyleColor();
        ImGui::TextWrapped("If more substeps are required, increase Contact substep limit "
            "under Advanced solver. Size changes require a particle reset.");
    } else {
        ImGui::TextWrapped("Solver: DEM grains (%s).", ownership.substance.c_str());
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(
            "Grains (DEM) run when a substance in this domain is granular with\n"
            "granular_transport = dem: Sand, Gravel, Ice or a substance derived from them,\n"
            "as the Default Substance or on a flow source. granular_transport = mpm keeps\n"
            "the continuum skeleton (cheaper for large flows).\n"
            "Script: fluid.matter_models(domain=...)['grain_ownership']");
    }
    if (ownership.wanted) {
        drawGrainSizes(domain, state);
        p = domain.fluid_params.grain;
        ImGui::TextDisabled("Contact stiffness %.0f N/m (from the radius)",
            p.stiffness_scale * kMatterGrainStiffnessPerRadius * p.radius_m);
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
            changed |= DomainUi::Float("Stiffness scale", &p.stiffness_scale, .01f, .01f, 1000.0f,
                "%.2f",
                "Multiplies the contact stiffness the radius gives (k = scale x 8e5 N/m per m x r;\n"
                "1 = 2e4 N/m at 25 mm). Softer = more overlap, fewer substeps (~ sqrt(k));\n"
                "keep overlap under ~1% of the radius.\n"
                "Script: fluid.set_grain_settings(stiffness_scale=...)");
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
                "Grains sleep after staying slow and force/torque balanced for the sleep time.\n"
                "Meaningful impacts or lost support trigger a local contact check.\n"
                "A supported grain can stay asleep while a weak neighbour moves.\n"
                "Saves GPU time on settled piles. Never with moving colliders, MPM contact,\n"
                "liquid coupling or a force field on the grain. Off = A/B reference.\n"
                "Script: fluid.set_grain_settings(sleep=...)");
            ImGui::BeginDisabled(!p.sleep);
            changed |= DomainUi::Float("Sleep speed (m/s)", &p.sleep_speed_m_s, .0005f,
                0.0f, 1.0f, "%.4f",
                "Translation speed, and spin x radius, below which a grain counts as still\n"
                "(0..1 m/s, absolute). Also sets the recipient's impulse-response deadband.\n"
                "Low speed alone cannot freeze an unsupported grain.\n"
                "Script: fluid.set_grain_settings(sleep_speed_m_s=...)");
            changed |= DomainUi::Float("Sleep time (s)", &p.sleep_time_s, .01f, .01f, 10.0f,
                "%.2f",
                "Physical seconds of slow, balanced motion before sleep (0.01..10 s).\n"
                "Adaptive substep changes do not restart this timer.\n"
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
    ImGui::PopID();
}

void drawMatterGrainSourceLine(const ParticleSimulationSystem& system,
                               SimulationFlowSourceDesc& source) {
    const auto& domains = system.gridDomains();
    if (source.domain_index < 0 || source.domain_index >= static_cast<int>(domains.size())) {
        return;
    }
    const auto& domain = domains[static_cast<std::size_t>(source.domain_index)];
    const std::string& name = source.fluid_substance.empty()
        ? domain.fluid_params.default_substance : source.fluid_substance;
    const auto* profile = name.empty() ? nullptr : tryFindSubstance(name);
    if (domain.type != SimulationDomainType::Matter || !profile ||
        profile->default_constitutive_model != MatterConstitutiveModel::Granular) {
        return;
    }
    if (profile->granular_transport != MatterGranularTransport::Dem) {
        ImGui::TextDisabled("Solver here: MPM continuum (%s: granular_transport = mpm).",
                            name.c_str());
        return;
    }
    const auto ownership = matterGrainOwnership(system,
        static_cast<std::size_t>(source.domain_index));
    if (!ownership.blockers.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f),
            "Asks for DEM grains but runs as MPM: %s", ownership.blockers.front().c_str());
    } else if (ownership.reset_pending && !ownership.enabled) {
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
            "Reset particles to start DEM grains.");
    } else {
        ImGui::TextDisabled("Solver here: DEM grains (settings in the domain's Solvers tab).");
    }
    // Wet at birth is a state of what this source pours (damp sand), so it is
    // the source's; it needs a grain substance that holds water.
    ImGui::BeginDisabled(profile->grain_water_capacity_fraction <= 0.0f);
    float saturation = source.grain_birth_saturation;
    if (DomainUi::Slider("Wet at birth", &saturation, 0.0f, 1.0f, "%.2f",
            "Water each poured grain already holds, as a fraction of its\n"
            "substance's water capacity (0..1). Needs a substance with\n"
            "grain_water_capacity_fraction > 0.\n"
            "Script: flow_source.update(grain_birth_saturation=...)")) {
        source.grain_birth_saturation = std::clamp(saturation, 0.0f, 1.0f);
    }
    ImGui::EndDisabled();
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
    if (r.gas_coupled_grains > 0) {
        ImGui::SeparatorText("Gas coupling");
        ImGui::Text("Grains in moving air %zu, max relative speed %.3g m/s",
            r.gas_coupled_grains, r.gas_max_relative_speed);
        ImGui::Text("Drag impulse %.4g N s, residual %.3g N s, held cells %zu",
            r.gas_drag_impulse.length(), r.gas_momentum_residual, r.gas_limited_cells);
    }
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
