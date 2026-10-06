#include "Fluid/MatterGrain.h"
#include "Fluid/FluidParticles.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace RayTrophiSim::Fluid {

nlohmann::json matterGrainPileProfile(const std::vector<Vec3>& centres, float radius) {
    if (centres.size() < 32 || !(radius > 0.0f)) {
        return {{"measured", false}, {"grains", centres.size()}};
    }
    double floor = std::numeric_limits<double>::max(), cx = 0.0, cz = 0.0;
    for (const auto& c : centres) {
        floor = std::min(floor, double(c.y) - radius);
        cx += c.x;
        cz += c.z;
    }
    cx /= double(centres.size());
    cz /= double(centres.size());
    const double ring = 2.0 * radius;
    std::vector<double> top;
    for (const auto& c : centres) {
        const double distance = std::hypot(double(c.x) - cx, double(c.z) - cz);
        const auto k = static_cast<std::size_t>(distance / ring);
        if (k >= top.size()) {
            top.resize(k + 1, 0.0);
        }
        top[k] = std::max(top[k], double(c.y) + radius - floor);
    }
    const double peak = *std::max_element(top.begin(), top.end());
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    int used = 0;
    double base_radius = 0.0;
    for (std::size_t k = 0; k < top.size(); ++k) {
        const double x = (double(k) + .5) * ring;
        if (top[k] >= 2.0 * ring) {
            base_radius = x;
        }
        if (k == 0 || top[k] < .2 * peak || top[k] > .8 * peak) {
            continue;
        }
        sx += x;
        sy += top[k];
        sxx += x * x;
        sxy += x * top[k];
        ++used;
    }
    nlohmann::json angle = nullptr;
    const double denominator = used * sxx - sx * sx;
    if (used >= 3 && denominator > 0.0) {
        const double slope = (used * sxy - sx * sy) / denominator;
        if (slope < 0.0) {
            angle = std::atan(-slope) * 57.29577951308232;
        }
    }
    return {{"measured", angle != nullptr}, {"grains", centres.size()},
        {"centre_xz", {cx, cz}}, {"peak_height_m", peak},
        {"base_radius_m", base_radius}, {"rings_fit", used},
        {"repose_angle_deg", angle}};
}

nlohmann::json matterGrainDiagnostics(const FluidParticles& p, const MatterGrainParams& params,
                                     const MatterGrainStepReport* report) {
    // Liquid parcels of the domain: count, centre height and the 95th
    // percentile height (the free surface, robust to a few splash parcels).
    std::vector<float> liquid_heights;
    double liquid_mass = 0.0, liquid_moment = 0.0;
    for (std::size_t i = 0; i < p.size() && i < p.constitutive_model.size(); ++i) {
        if (p.constitutive_model[i] != static_cast<uint8_t>(MatterConstitutiveModel::Fluid) ||
            i >= p.rest_mass_kg.size() || i >= p.mass_fraction.size()) {
            continue;
        }
        const double mass = double(p.rest_mass_kg[i]) * p.mass_fraction[i];
        liquid_heights.push_back(p.position[i].y);
        liquid_mass += mass;
        liquid_moment += mass * p.position[i].y;
    }
    nlohmann::json liquid_shape = {{"parcels", liquid_heights.size()}};
    if (!liquid_heights.empty()) {
        const auto rank = static_cast<std::size_t>(.95 * double(liquid_heights.size() - 1));
        std::nth_element(liquid_heights.begin(), liquid_heights.begin() + rank, liquid_heights.end());
        liquid_shape["surface_p95_m"] = liquid_heights[rank];
        liquid_shape["center_y_m"] = liquid_mass > 0.0 ? liquid_moment / liquid_mass : 0.0;
    }
    double spin_energy = 0.0, maximum_spin = 0.0;
    double angular[3] = {};
    std::vector<Vec3> centres;
    centres.reserve(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        // Liquid parcels of the same domain are not grains: no spin, no pile.
        if (i >= p.rest_mass_kg.size() || i >= p.mass_fraction.size() || i >= p.affine.size() ||
            i >= p.constitutive_model.size() || p.constitutive_model[i] !=
                static_cast<uint8_t>(MatterConstitutiveModel::Granular)) {
            continue;
        }
        centres.push_back(p.position[i]);
        const auto& a = p.affine[i];
        const Vec3 w = Vec3(a.col1.z - a.col2.y, a.col2.x - a.col0.z,
                            a.col0.y - a.col1.x) * .5f;
        const double mass = double(p.rest_mass_kg[i]) * p.mass_fraction[i];
        const double inertia = .4 * mass * params.radius_m * params.radius_m;
        const double w2 = double(w.x) * w.x + double(w.y) * w.y + double(w.z) * w.z;
        spin_energy += .5 * inertia * w2;
        maximum_spin = std::max(maximum_spin, std::sqrt(w2));
        const auto orbital = p.position[i].cross(p.velocity[i]);
        angular[0] += mass * orbital.x + inertia * w.x;
        angular[1] += mass * orbital.y + inertia * w.y;
        angular[2] += mass * orbital.z + inertia * w.z;
    }
    const bool history = params.tangential_stiffness_ratio > 0.0f;
    nlohmann::json runtime = nullptr;
    if (report && report->substeps > 0) {
        runtime = {{"substeps", report->substeps}, {"dispatches", report->dispatches},
            {"substep_dt_s", report->substep_dt}, {"substep_limit", report->limit},
            {"max_contacts_per_grain", report->max_contacts},
            {"contacts_last_substep", report->contacts},
            {"sticking_contacts_last_substep", report->sticking_contacts},
            {"history_reset_this_step", report->history_reset},
            {"history_reset_reason", report->history_reset_reason},
            {"working_set_bytes", report->working_set_bytes}};
    }
    nlohmann::json liquid = nullptr;
    if (report) {
        const auto vec = [](const Vec3& v) { return nlohmann::json{v.x, v.y, v.z}; };
        liquid = {{"grains", report->grains}, {"liquid_parcels", report->liquid_parcels},
            {"coupling_enabled", report->coupling_enabled},
            {"coupled_grains", report->coupled_grains},
            {"drag_impulse_n_s", vec(report->drag_impulse)},
            {"buoyancy_impulse_n_s", vec(report->buoyancy_impulse)},
            {"liquid_reaction_n_s", vec(report->liquid_reaction)},
            {"momentum_residual_n_s", report->momentum_residual},
            {"unmatched_impulse_n_s", report->unmatched_impulse},
            {"max_drag_coefficient_kg_s", report->max_drag_coefficient},
            {"max_submerged_fraction", report->max_submerged_fraction},
            {"model", report->volume_exclusion
                ? "di_felice_implicit_pair_drag+porous_projection_pressure_force"
                : "di_felice_implicit_pair_drag+hydrostatic_buoyancy"},
            {"volume_exclusion", report->volume_exclusion},
            {"pressure_force", report->pressure_force},
            {"pressure_impulse_n_s", vec(report->pressure_impulse)},
            {"porous_cells", report->porous_cells},
            {"max_solid_fraction", report->max_solid_fraction},
            {"shape", liquid_shape},
            {"wet", {{"enabled", report->wet_grains},
                {"wet_grains", report->wet_grain_count},
                {"liquid_bridges_last_substep", report->liquid_bridges},
                {"grain_water_kg", report->grain_water_kg},
                {"absorbed_kg", report->absorbed_kg},
                {"evaporated_kg", report->evaporated_kg},
                {"water_balance_error_kg", report->water_balance_error_kg},
                {"max_saturation", report->max_grain_saturation},
                {"bridge_model", "willett_2000_pendular"}}}};
    }
    return {{"transport_owner", params.enabled ? "grain" : "mpm"},
        {"solver", history ? "force_dem_cundall_strack_candidate"
                           : "force_dem_viscous_sliding_candidate"},
        {"collider_structure", "flat_bvh"}, {"max_support_contacts", 4},
        {"max_surface_patches", 8}, {"contact_shader_revision", kMatterGrainShaderRevision},
        {"contact_budget", kMatterGrainContactBudget}, {"runtime", runtime},
        {"rolling_model", params.rolling_friction > 0.0f ? "epsd2_spring" : "none"},
        {"pile", matterGrainPileProfile(centres, params.radius_m)},
        {"neighbour_structure", "rotating_fixed_buckets"},
        {"radius_m", params.radius_m}, {"spin_energy_j", spin_energy},
        {"max_spin_rad_s", maximum_spin},
        {"angular_momentum_kg_m2_s", {angular[0], angular[1], angular[2]}},
        {"history_static_friction", history}, {"wet_coupling", params.wet_grains},
        {"liquid", liquid}, {"fluid_coupling_setting", params.fluid_coupling},
        {"dissipation_heat_coupled", false}, {"frame_end_host_publication", true}};
}

} // namespace RayTrophiSim::Fluid
