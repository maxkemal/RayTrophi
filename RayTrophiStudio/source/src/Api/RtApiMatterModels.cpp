#include "RtMatterModels.h"
#include "RtApiInternal.h"
#include "Fluid/MatterModelService.h"
#include "Fluid/MatterPoreAuthoring.h"
#include "Fluid/MatterWetResponse.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/MatterAcceptanceMetrics.h"
#include "Fluid/MatterGrain.h"
#include "Fluid/MatterSubstanceState.h"

#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <sstream>

namespace rtapi {

nlohmann::json getMatterModels(const std::string& domain, bool include_transfer) {
    if (!g_ctx) {
        throw std::runtime_error("rtapi is not bound to a UIContext");
    }
    auto& runtime = scriptSimulationRuntime();
    const auto& domains = runtime.gridDomains();
    const auto& states = runtime.gridDomainStates();
    for (std::size_t index = 0; index < domains.size(); ++index) {
        if (domains[index].name != domain) {
            continue;
        }
        const auto* state = index < states.size() && states[index].valid
            ? &states[index] : nullptr;
        RayTrophiSim::Fluid::MatterTransferFrame frame;
        std::string error;
        if (!RayTrophiSim::Fluid::inspectMatterModels(
                domains[index], state, include_transfer, frame, error)) {
            throw std::runtime_error(error);
        }
        const auto momentum = [](const RayTrophiSim::Fluid::MatterMomentum& value) {
            return nlohmann::json::array({value.x, value.y, value.z});
        };
        nlohmann::json models = nlohmann::json::array();
        const char* names[] = {"fluid", "granular", "elastic", "unresolved"};
        for (std::size_t model = 0; model < frame.totals.size(); ++model) {
            const auto& totals = frame.totals[model];
            models.push_back({{"model", names[model]}, {"particles", totals.particles},
                {"mass_kg", totals.mass_kg}, {"momentum_kg_m_s", momentum(totals.momentum)}});
        }
        uint64_t identity_hash = 14695981039346656037ull;
        double pore_mass = 0.0, pore_capacity = 0.0;
        float maximum_saturation = 0.0f;
        if (state) {
            for (std::size_t i = 0; i < state->particles.size(); ++i) {
                const auto& particles = state->particles;
                const float mass = i < particles.pore_water_mass_kg.size()
                    ? particles.pore_water_mass_kg[i] : 0.0f;
                const float capacity = i < particles.pore_capacity_kg.size()
                    ? particles.pore_capacity_kg[i] : 0.0f;
                pore_mass += mass;
                pore_capacity += capacity;
                if (capacity > 0.0f) {
                    maximum_saturation = std::max(maximum_saturation, mass / capacity);
                }
            }
            for (uint64_t identity : state->particles.particle_id) {
                for (int byte = 0; byte < 8; ++byte) {
                    identity_hash ^= (identity >> (byte * 8)) & 0xffu;
                    identity_hash *= 1099511628211ull;
                }
            }
        }
        std::size_t wet_particles = 0;
        float maximum_pore_pressure = 0.0f;
        float maximum_capillary_cohesion = 0.0f;
        if (state) {
            std::vector<RayTrophiSim::Fluid::MatterWetResponse> wet;
            if (!RayTrophiSim::Fluid::buildMatterWetResponses(state->particles,
                    domains[index].fluid_params.granular_enabled,
                    RayTrophiSim::Fluid::liquidGrid(*state).voxel_size,
                    domains[index].fluid_params.pore_exchange, wet, error)) {
                throw std::runtime_error(error);
            }
            for (std::size_t i = 0; i < wet.size(); ++i) {
                if (wet[i][2] > 0.0f || wet[i][3] > 0.0f || wet[i][0] < 1.0f) {
                    ++wet_particles;
                }
                maximum_capillary_cohesion = std::max(maximum_capillary_cohesion, wet[i][2]);
                maximum_pore_pressure = std::max(maximum_pore_pressure, wet[i][3]);
            }
        }
        std::ostringstream hash_text;
        hash_text << std::hex << std::setw(16) << std::setfill('0') << identity_hash;
        nlohmann::json acceptance_metrics{{"measured", false},
            {"reason", "Canonical mass/model/pore sidecars are not available"}};
        if (state) {
            const auto& particles = state->particles;
            const auto count = particles.size();
            const bool canonical = particles.constitutive_model.size() == count &&
                particles.rest_mass_kg.size() == count && particles.mass_fraction.size() == count &&
                particles.pore_water_mass_kg.size() == count &&
                particles.pore_capacity_kg.size() == count &&
                std::all_of(particles.rest_mass_kg.begin(), particles.rest_mass_kg.end(),
                    [](float mass) { return mass > 0.0f; });
            if (canonical) {
                acceptance_metrics = RayTrophiSim::Fluid::inspectMatterAcceptanceMetrics(
                    particles, domains[index].fluid_params.granular_enabled,
                    domains[index].fluid_params.pore_exchange.wet_appearance_full_saturation);
            }
        }
        return {{"domain", domain}, {"measured", state && state->valid},
            {"acceptance_metrics", acceptance_metrics},
            {"granular_mechanics", {
                {"measured", state && state->fluid_stats.granular_load_measured},
                {"load_proxy", "granular_extent_rho_g_h"},
                {"contact_pressure_measured", false},
                {"overburden_estimate_pa", state
                    ? state->fluid_stats.granular_overburden_pressure : 0.0f},
                {"young_modulus_for_load_pa", state
                    ? state->fluid_stats.granular_young_modulus_for_load : 0.0f},
                {"stiffness_below_estimated_load", state
                    && state->fluid_stats.granular_stiffness_below_load},
                {"wave_substeps", state ? state->fluid_stats.granular_wave_substeps : 1},
                {"strain_substeps", state ? state->fluid_stats.granular_strain_substeps : 1},
                {"strain_rate_per_s", state ? state->fluid_stats.granular_strain_rate : 0.0f}}},
            {"pore_exchange", {
                {"settings", RayTrophiSim::Fluid::matterPoreParamsToJson(
                    domains[index].fluid_params.pore_exchange)},
                {"measured", state && state->fluid_stats.pore_exchange.measured},
                {"held", state && state->fluid_stats.pore_exchange.held},
                {"status", state ? state->fluid_stats.pore_exchange.status : "No synchronized state"},
                {"pore_water_kg", pore_mass}, {"capacity_kg", pore_capacity},
                {"maximum_saturation", maximum_saturation},
                {"absorbed_kg", state ? state->fluid_stats.pore_exchange.absorbed_kg : 0.0},
                {"drained_kg", state ? state->fluid_stats.pore_exchange.drained_kg : 0.0},
                {"drainage_births", state ? state->fluid_stats.pore_exchange.drainage_births : 0},
                {"drainage_refills", state ? state->fluid_stats.pore_exchange.drainage_refills : 0},
                {"drainage_budget_blocked_carriers", state
                    ? state->fluid_stats.pore_exchange.drainage_budget_blocked_carriers : 0},
                {"mass_error_kg", state ? state->fluid_stats.pore_exchange.mass_error_kg : 0.0},
                {"momentum_error_kg_m_s", state
                    ? state->fluid_stats.pore_exchange.momentum_error_kg_m_s : 0.0},
                {"thermal_energy_error_j", state
                    ? state->fluid_stats.pore_exchange.thermal_energy_error_j : 0.0}}},
            {"wet_response", {
                {"enabled", domains[index].fluid_params.pore_exchange.wet_response_enabled},
                {"appearance_enabled", domains[index].fluid_params.pore_exchange.wet_appearance_enabled},
                {"model", "empirical_cell_head"}, {"appearance", "granular_splat_8_bands"},
                {"appearance_quantization", RayTrophiSim::Fluid::kMatterWetAppearanceQuantization},
                {"field_sampled", state && state->valid},
                {"gpu_step_measured", state && state->fluid_stats.mixed_model_step &&
                    !state->fluid_stats.mixed_step_held},
                {"wet_particles", wet_particles},
                {"maximum_pore_pressure_pa", maximum_pore_pressure},
                {"maximum_capillary_cohesion_pa", maximum_capillary_cohesion}}},
            {"models", models}, {"identity_scope", "domain"},
            {"transport_owners", [&] {
                const auto owners = state ? RayTrophiSim::Fluid::inspectMatterDomainOwners(
                    state->particles, domains[index])
                    : RayTrophiSim::Fluid::MatterOwnerSummary{};
                return nlohmann::json{{"measured", state != nullptr},
                    {"fluid", owners.particles[0]}, {"grain", owners.particles[1]},
                    {"mpm", owners.particles[2]}, {"obstacle", owners.particles[3]},
                    {"ready", state != nullptr && owners.ready}, {"reason", owners.reason}};
            }()},
            {"grain_settings", RayTrophiSim::Fluid::matterGrainParamsToJson(
                domains[index].fluid_params.grain)},
            {"grain_readiness", [&] {
                // Same rule the panel locks and set_grain_settings use.
                nlohmann::json blockers = nlohmann::json::array();
                for (const auto& b : RayTrophiSim::Fluid::matterGrainBlockers(domains[index])) {
                    blockers.push_back({{"code", b.code}, {"message", b.message}});
                }
                return nlohmann::json{{"enabled", domains[index].fluid_params.grain.enabled},
                    {"ready", blockers.empty()}, {"blockers", blockers},
                    {"step_held", state && state->fluid_stats.mixed_step_held},
                    {"status", state ? state->fluid_stats.gpu_status : std::string()}};
            }()},
            {"grain_diagnostics", state && domains[index].fluid_params.grain.enabled
                ? RayTrophiSim::Fluid::matterGrainDiagnostics(
                    state->particles, domains[index].fluid_params.grain,
                    &state->fluid_stats.grain_report,
                    &domains[index].fluid_params) : nlohmann::json(nullptr)},
            {"mixed_transport_available", true},
            {"mixed_transport_ready", state && state->fluid_stats.mixed_model_step},
            {"mixed_execution", {
                {"common_substeps", state ? state->fluid_stats.mixed_common_substeps : 0},
                {"contact_pairs", state ? state->fluid_stats.mixed_contact_pairs : 0},
                {"working_set_bytes", state ? state->fluid_stats.mixed_working_set_bytes : 0},
                {"p2g_on_gpu", state && state->fluid_stats.p2g_on_gpu},
                {"pressure_on_gpu", state && state->fluid_stats.pressure_on_gpu},
                {"g2p_on_gpu", state && state->fluid_stats.g2p_on_gpu},
                {"step_held", state && state->fluid_stats.mixed_step_held},
                {"status", state ? state->fluid_stats.gpu_status : "No synchronized state"}}},
            {"transfer_requested", include_transfer},
            {"transfer_cells", frame.cells.size()}, {"overlapping_cells", frame.overlapping_cells},
            {"outside_particles", frame.outside_particles}, {"outside_mass_kg", frame.outside_mass_kg},
            {"deposited_mass_kg", frame.deposited_mass_kg},
            {"deposited_momentum_kg_m_s", momentum(frame.deposited_momentum)},
            {"particle_id_hash", hash_text.str()},
            {"next_particle_id", state ? std::to_string(state->particles.next_particle_id) : "1"},
            {"particle_id_bytes", state ? state->particles.particle_id.size() * sizeof(uint64_t) : 0}};
    }
    throw std::runtime_error("grid domain not found: " + domain);
}

} // namespace rtapi
