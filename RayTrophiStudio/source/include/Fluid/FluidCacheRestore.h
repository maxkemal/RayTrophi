#pragma once

#include "Fluid/MatterPhaseConfig.h"
#include "PerfProfile.h"

#include <algorithm>
#include <vector>

namespace RayTrophiSim::Fluid {

// RAM and disk restores publish a new host snapshot, even when particle count
// and the version recorded by the bake are unchanged.
inline void prepareFluidCacheRestore(
    std::vector<SimulationGridDomainState>& restored,
    const std::vector<SimulationGridDomainState>& previous,
    const std::vector<SimulationGridDomainDesc>& domains,
    std::vector<SimulationGridDomainComputeBuffers>& primary_buffers,
    std::vector<SimulationGridDomainComputeBuffers>& liquid_buffers) {
    RTPERF_FRAME_SCOPE("sim.cache.restore.prepare");
    for (std::size_t i = 0; i < restored.size(); ++i) {
        auto& state = restored[i];
        const uint64_t previous_version = i < previous.size() ? previous[i].version : 0;
        state.version = std::max(state.version, previous_version) + 1;

        // Render-only disk caches predate the secondary Matter liquid grid.
        // Recover its layout from the baked logical box and authored phase
        // settings; the default empty grid has a misleading 1-metre voxel.
        if (state.valid && state.type == SimulationDomainType::Matter &&
            !state.matter_liquid_active && state.matter_liquid_grid.nx == 0 &&
            i < domains.size()) {
            const auto layouts = resolvePhaseLayouts(
                domains[i], state.bounds_min, state.bounds_max,
                state.resolution_x, state.resolution_y, state.resolution_z,
                state.voxel_size);
            const auto& liquid = layouts.liquid;
            state.matter_liquid_grid.allocate_gas_channels = false;
            state.matter_liquid_grid.sparse_mode_enabled = domains[i].use_sparse_tiles ||
                domains[i].backend == SimulationDomainBackend::CPU_SparseVDB;
            // Playback only needs this grid's layout, not APIC velocity,
            // pressure, divergence or solid work arrays. Live synchronization
            // detects the missing storage and allocates it before solving.
            state.matter_liquid_grid.nx = liquid.nx;
            state.matter_liquid_grid.ny = liquid.ny;
            state.matter_liquid_grid.nz = liquid.nz;
            state.matter_liquid_grid.voxel_size = liquid.voxel;
            state.matter_liquid_grid.origin = liquid.origin;
            state.matter_liquid_grid.tiles_x = (liquid.nx + FluidSim::TILE_SIZE - 1) /
                FluidSim::TILE_SIZE;
            state.matter_liquid_grid.tiles_y = (liquid.ny + FluidSim::TILE_SIZE - 1) /
                FluidSim::TILE_SIZE;
            state.matter_liquid_grid.tiles_z = (liquid.nz + FluidSim::TILE_SIZE - 1) /
                FluidSim::TILE_SIZE;
            state.matter_liquid_grid.total_tiles = state.matter_liquid_grid.tiles_x *
                state.matter_liquid_grid.tiles_y * state.matter_liquid_grid.tiles_z;
        }
    }
    // A valid handle/count alone cannot prove these positions belong to the
    // restored frame. Live simulation uploads will republish the count.
    for (auto& buffers : primary_buffers) {
        buffers.fluid_uploaded_particle_count = 0;
    }
    for (auto& buffers : liquid_buffers) {
        buffers.fluid_uploaded_particle_count = 0;
    }
}

} // namespace RayTrophiSim::Fluid
