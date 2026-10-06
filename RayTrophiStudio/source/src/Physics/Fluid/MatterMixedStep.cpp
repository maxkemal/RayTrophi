#include "Fluid/MatterMixedStep.h"
#include "Fluid/MatterModelBatch.h"
#include "Fluid/MatterMacContact.h"
#include "Fluid/GranularGpuDispatch.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace RayTrophiSim::Fluid {
namespace {

template <typename T>
std::size_t bytesOf(const std::vector<T>& values) {
    return values.capacity() * sizeof(T);
}

} // namespace

MatterConstitutiveModel resolveSingleMatterModel(const FluidParticles& particles,
                                                 bool legacy_granular) {
    MatterConstitutiveModel resolved = MatterConstitutiveModel::Auto;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        auto model = i < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[i])
            : MatterConstitutiveModel::Auto;
        if (model == MatterConstitutiveModel::Auto) {
            model = legacy_granular ? MatterConstitutiveModel::Granular
                                   : MatterConstitutiveModel::Fluid;
        }
        if (resolved == MatterConstitutiveModel::Auto) {
            resolved = model;
        } else if (resolved != model) {
            return MatterConstitutiveModel::Auto;
        }
    }
    return resolved;
}

bool hasMixedMatterModels(const FluidParticles& particles, bool legacy_granular) {
    bool fluid = false, granular = false;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        auto model = i < particles.constitutive_model.size()
            ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[i])
            : MatterConstitutiveModel::Auto;
        if (model == MatterConstitutiveModel::Auto) {
            model = legacy_granular ? MatterConstitutiveModel::Granular
                                   : MatterConstitutiveModel::Fluid;
        }
        fluid |= model == MatterConstitutiveModel::Fluid;
        granular |= model == MatterConstitutiveModel::Granular;
    }
    return fluid && granular;
}

std::size_t estimateMixedMatterWorkingSet(const FluidParticles& particles,
                                         const FluidSim::FluidGrid& grid) {
    std::size_t bytes = 0;
    bytes += bytesOf(grid.active_tiles);
    bytes += bytesOf(grid.tile_active_mask);
    bytes += bytesOf(grid.vel_x);
    bytes += bytesOf(grid.vel_y);
    bytes += bytesOf(grid.vel_z);
    bytes += bytesOf(grid.density);
    bytes += bytesOf(grid.temperature);
    bytes += bytesOf(grid.fuel);
    bytes += bytesOf(grid.interaction);
    bytes += bytesOf(grid.pressure);
    bytes += bytesOf(grid.divergence);
    bytes += bytesOf(grid.solid);
    bytes += bytesOf(grid.solid_vel);
    bytes += bytesOf(grid.solid_cells);
    bytes += bytesOf(grid.substance_solid_cells);
    bytes += bytesOf(grid.substance_solid_prev_cells);
    bytes += bytesOf(grid.substance_solid_prev_vel);
    bytes += bytesOf(grid.solid_gas_density);
    bytes += bytesOf(grid.solid_gas_temperature);
    bytes += bytesOf(grid.solid_gas_fuel);
    bytes += bytesOf(grid.solid_gas_flame);
    bytes += bytesOf(grid.solid_gas_band);
    bytes += bytesOf(grid.u_weight);
    bytes += bytesOf(grid.v_weight);
    bytes += bytesOf(grid.w_weight);
    bytes += bytesOf(grid.fluid_phi);
    bytes += bytesOf(grid.surface_dust_supply);
    bytes += bytesOf(particles.particle_id);
    bytes += bytesOf(particles.position);
    bytes += bytesOf(particles.velocity);
    bytes += bytesOf(particles.affine);
    bytes += bytesOf(particles.flags);
    bytes += bytesOf(particles.mass_fraction);
    bytes += bytesOf(particles.rest_mass_kg);
    bytes += bytesOf(particles.temperature);
    bytes += bytesOf(particles.combustible_fraction);
    bytes += bytesOf(particles.substance_tag);
    bytes += bytesOf(particles.constitutive_model);
    bytes += bytesOf(particles.granular_deformation_col0);
    bytes += bytesOf(particles.granular_deformation_col1);
    bytes += bytesOf(particles.granular_deformation_col2);
    bytes += bytesOf(particles.granular_plastic_volume);
    bytes += bytesOf(particles.granular_softening);
    bytes += bytesOf(particles.granular_bond_scale);
    bytes += bytesOf(particles.granular_hardening);
    bytes += bytesOf(particles.granular_material_flags);
    bytes += bytesOf(particles.granular_stress_diag);
    bytes += bytesOf(particles.granular_stress_shear);
    bytes += bytesOf(particles.granular_yield_value);
    bytes += bytesOf(particles.granular_plastic_increment);
    bytes += bytesOf(particles.granular_damage);
    bytes += bytesOf(particles.granular_fracture_history);
    bytes += bytesOf(particles.uvw);
    bytes += bytesOf(particles.uvw_b);

    // Original data, frame rollback, model partitions, two grids, merge and
    // CPU projection scratch. Include sparse map nodes/occupant allocations
    // at worst-case 27 touched cells per parcel, plus contact face buffers.
    const auto cells = static_cast<std::size_t>(grid.nx) * grid.ny * grid.nz;
    const auto support = std::min(cells, particles.size() * 27);
    return bytes * 16 + support * 1024 + particles.size() * 1024;
}

bool stepMixedMatter(FluidParticles& particles, FluidSim::FluidGrid& grid,
    const APICSolverParams& params, float dt, const SimulationForceFieldSnapshot* forces,
    float time_seconds, APICSolverStats& stats, std::string& error) {
    error.clear();
    if (params.pore_exchange.enabled || needsMatterPoreTransport(particles, params.pore_exchange)) {
        error = "C5 wet-carrier transport requires Vulkan; CPU reference is not enabled";
        return false;
    }
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        !std::isfinite(grid.voxel_size) || grid.voxel_size <= 0.0f ||
        !std::isfinite(dt) || dt <= 0.0f || !std::isfinite(time_seconds)) {
        error = "mixed step requires valid grid and time";
        return false;
    }
    if (params.mixed_working_set_budget_bytes > 0 &&
        estimateMixedMatterWorkingSet(particles, grid) > params.mixed_working_set_budget_bytes) {
        error = "mixed CPU working set exceeds domain resource budget";
        return false;
    }
    double speed = 0.0, strain = 0.0;
    for (std::size_t i = 0; i < particles.size(); ++i) {
        if (i >= particles.velocity.size() || i >= particles.affine.size()) {
            error = "mixed step requires velocity and affine sidecars";
            return false;
        }
        speed = std::max(speed, static_cast<double>(particles.velocity[i].length()));
        const auto& c = particles.affine[i];
        strain = std::max(strain, static_cast<double>(std::sqrt(
            c.col0.length_squared() + c.col1.length_squared() + c.col2.length_squared())));
    }
    const double h = grid.voxel_size;
    const double cfl = std::clamp(static_cast<double>(params.cfl), 0.05, 1.0);
    const auto elastic = Granular::elasticStepInfo(params.granular_young_modulus,
        grid.voxel_size, dt, static_cast<float>(strain));
    // Include acceleration over the whole frame rather than measuring only
    // the initially stationary pile. Never reduce authored stiffness to fit.
    const double acceleration = params.gravity.length();
    const double request = std::max(static_cast<double>(elastic.required_substeps),
        std::ceil((speed + acceleration * dt) * dt / (cfl * h)));
    if (!std::isfinite(request) || request > 4096.0) {
        error = "mixed common CFL request exceeds CPU execution safety limit";
        return false;
    }
    const int substeps = std::max(1, static_cast<int>(request));
    const float sub_dt = dt / static_cast<float>(substeps);
    auto candidate_particles = particles;
    auto candidate_grid = grid;
    std::array<APICSolverParams, 2> models{params, params};
    for (auto& model : models) {
        model.mixed_model_substep = true;
        model.reseed_enabled = false;
        model.uvw_refresh_period = std::numeric_limits<int>::max();
        // Birth/reseed requires a domain allocator shared by both lanes.
        // Preserve outflow compaction while preventing private lane births.
    }
    models[0].granular_enabled = false;
    models[1].granular_enabled = true;
    const auto legacy = params.granular_enabled ? MatterConstitutiveModel::Granular
                                              : MatterConstitutiveModel::Fluid;
    APICSolverStats aggregate{};
    MatterContactResult contact_stats;
    const double friction = std::tan(std::clamp(
        static_cast<double>(params.granular_friction_angle_degrees), 0.0, 80.0) *
        3.14159265358979323846 / 180.0);
    for (int substep = 0; substep < substeps; ++substep) {
        if (candidate_particles.empty()) {
            break;
        }
        if (!hasMixedMatterModels(candidate_particles, params.granular_enabled)) {
            const auto remaining = resolveSingleMatterModel(candidate_particles,
                                                            params.granular_enabled);
            auto single = models[remaining == MatterConstitutiveModel::Granular ? 1 : 0];
            APICSolverStats tail_stats;
            step(candidate_particles, candidate_grid, single, sub_dt, forces,
                 time_seconds + substep * sub_dt, &tail_stats);
            aggregate.total_ms += tail_stats.total_ms;
            continue;
        }
        const MatterBatchContact contact = [&](const FluidParticles&, FluidSim::FluidGrid& liquid,
            const FluidParticles&, FluidSim::FluidGrid& granular, std::string& message) {
            return applyMatterMacContact(candidate_particles, liquid, granular,
                                         legacy, friction, contact_stats, message);
        };
        MatterModelBatchResult batch;
        if (!stepMatterModelBatch(candidate_particles, candidate_grid, models, legacy,
            sub_dt, forces, time_seconds + substep * sub_dt, contact, batch, error)) {
            return false;
        }
        for (const auto& lane : batch.model_stats) {
            aggregate.forces_ms += lane.forces_ms;
            aggregate.p2g_ms += lane.p2g_ms;
            aggregate.pressure_ms += lane.pressure_ms;
            aggregate.viscosity_ms += lane.viscosity_ms;
            aggregate.g2p_ms += lane.g2p_ms;
            aggregate.advect_ms += lane.advect_ms;
            aggregate.total_ms += lane.total_ms;
        }
        aggregate.granular_constitutive_ms += batch.model_stats[1].granular_constitutive_ms;
        aggregate.mixed_contact_pairs += contact_stats.pairs;
        aggregate.mixed_contact_energy_loss += contact_stats.kinetic_energy_loss;
    }
    // UVW is a frame epoch; private common substeps must not age it N times.
    candidate_particles.uvw_step = particles.uvw_step;
    candidate_particles.uvw_refresh_period = params.uvw_refresh_period;
    candidate_particles.advanceMaterialCoordinates();
    aggregate.particle_count = candidate_particles.size();
    aggregate.grid_cell_count = static_cast<std::size_t>(grid.nx) * grid.ny * grid.nz;
    aggregate.mixed_model_step = true;
    aggregate.mixed_common_substeps = substeps;
    aggregate.mixed_working_set_bytes = estimateMixedMatterWorkingSet(particles, grid);
    aggregate.granular_solver_substeps = substeps;
    aggregate.granular_requested_young_modulus = params.granular_young_modulus;
    aggregate.granular_effective_young_modulus = params.granular_young_modulus;
    particles = std::move(candidate_particles);
    grid = std::move(candidate_grid);
    stats = aggregate;
    return true;
}

} // namespace RayTrophiSim::Fluid
