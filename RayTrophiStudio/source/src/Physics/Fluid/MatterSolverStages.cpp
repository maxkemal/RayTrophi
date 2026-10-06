#include "Fluid/MatterSolverStages.h"

#include <cmath>

namespace RayTrophiSim::Fluid {

bool prepareMatterModelGrid(FluidParticles& particles, FluidSim::FluidGrid& grid,
                           const APICSolverParams& params, float dt,
                           const SimulationForceFieldSnapshot* forces,
                           float time_seconds, MatterModelGridStep& workspace,
                           std::string& error) {
    error.clear();
    if (workspace.ready) {
        error = "model grid step is already prepared";
        return false;
    }
    if (!std::isfinite(dt) || dt <= 0.0f || !std::isfinite(time_seconds) ||
        !std::isfinite(grid.voxel_size) || grid.voxel_size < 1e-6f ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        error = "invalid model grid step layout or time";
        return false;
    }
    particles.ensureParticleIdentities();
    MatterModelGridStep candidate;
    candidate.params = params;
    candidate.params.p2g_precomputed = false;
    candidate.params.external_forces_preintegrated = params.external_forces_preintegrated;
    candidate.params.viscosity_precomputed = false;
    candidate.params.pressure_precomputed = false;
    candidate.params.pressure_g2p_precomputed = false;
    candidate.params.particle_tail_precomputed = false;
    candidate.params.stop_before_pressure = false;
    candidate.params.stop_after_projection = true;
    candidate.params.model_flip_snapshot = nullptr;
    step(particles, grid, candidate.params, dt, forces,
         time_seconds, &candidate.projection_stats);
    const bool flip = !params.granular_enabled && params.free_surface &&
        params.flip_blend > 0.0f && !particles.empty();
    if (flip) {
        if (!hasLastFlipPreSnapshot()) {
            error = "model FLIP snapshot was not published";
            return false;
        }
        candidate.flip.x.assign(getLastFlipPreSnapshotX(),
            getLastFlipPreSnapshotX() + grid.vel_x.size());
        candidate.flip.y.assign(getLastFlipPreSnapshotY(),
            getLastFlipPreSnapshotY() + grid.vel_y.size());
        candidate.flip.z.assign(getLastFlipPreSnapshotZ(),
            getLastFlipPreSnapshotZ() + grid.vel_z.size());
    }
    candidate.particles = &particles;
    candidate.grid = &grid;
    candidate.dimensions = {grid.nx, grid.ny, grid.nz};
    candidate.origin = grid.origin;
    candidate.voxel = grid.voxel_size;
    candidate.dt = dt;
    candidate.time_seconds = time_seconds;
    candidate.allocator = particles.next_particle_id;
    candidate.identities = particles.particle_id;
    candidate.particle_count = particles.size();
    candidate.ready = true;
    workspace = std::move(candidate);
    return true;
}

bool finishMatterModelGrid(MatterModelGridStep& workspace, APICSolverStats& stats,
                          std::string& error) {
    error.clear();
    if (!workspace.ready || !workspace.grid || !workspace.particles) {
        error = "model grid step is not prepared or already consumed";
        return false;
    }
    auto& grid = *workspace.grid;
    auto& particles = *workspace.particles;
    if (workspace.dimensions != std::array<int, 3>{grid.nx, grid.ny, grid.nz} ||
        workspace.voxel != grid.voxel_size ||
        workspace.origin.x != grid.origin.x || workspace.origin.y != grid.origin.y ||
        workspace.origin.z != grid.origin.z || workspace.particle_count != particles.size() ||
        workspace.allocator != particles.next_particle_id ||
        workspace.identities != particles.particle_id) {
        error = "model layout or topology changed between projection and gather";
        return false;
    }
    auto params = workspace.params;
    if (!params.granular_enabled && params.free_surface && params.flip_blend > 0.0f &&
        !particles.empty() && (workspace.flip.x.size() != grid.vel_x.size() ||
        workspace.flip.y.size() != grid.vel_y.size() ||
        workspace.flip.z.size() != grid.vel_z.size())) {
        error = "model FLIP baseline no longer matches the MAC fields";
        return false;
    }
    params.stop_after_projection = false;
    params.p2g_precomputed = true;
    params.external_forces_preintegrated = true;
    params.viscosity_precomputed = true;
    params.pressure_precomputed = true;
    params.model_flip_snapshot = &workspace.flip;
    step(particles, grid, params, workspace.dt, nullptr,
         workspace.time_seconds, &stats);
    const auto& prepared = workspace.projection_stats;
    stats.forces_ms = prepared.forces_ms;
    stats.p2g_ms = prepared.p2g_ms;
    stats.p2g_on_gpu = false;
    stats.boundary_ms = prepared.boundary_ms;
    stats.viscosity_ms = prepared.viscosity_ms;
    stats.viscosity_sweeps_run = prepared.viscosity_sweeps_run;
    stats.pressure_ms = prepared.pressure_ms;
    stats.active_fluid_cells = prepared.active_fluid_cells;
    stats.sealed_pockets = prepared.sealed_pockets;
    stats.sealed_pocket_cells = prepared.sealed_pocket_cells;
    stats.sealed_pockets_measured = prepared.sealed_pockets_measured;
    stats.interior_fluid_cells = prepared.interior_fluid_cells;
    stats.total_ms += prepared.total_ms;
    workspace.ready = false;
    return true;
}

} // namespace RayTrophiSim::Fluid
