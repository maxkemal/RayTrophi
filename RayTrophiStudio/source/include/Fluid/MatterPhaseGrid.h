#pragma once

#include "ParticleSimulation.h"

#include <utility>

namespace RayTrophiSim::Fluid {

// The liquid solver temporarily owns state.grid. Consumers must select by
// phase identity rather than assume the gas or liquid is in a fixed slot.
inline bool liquidUsesSecondaryGrid(const SimulationGridDomainState& state) {
    return state.type == SimulationDomainType::Matter && !state.matter_liquid_active;
}

inline FluidSim::FluidGrid& liquidGrid(SimulationGridDomainState& state) {
    return liquidUsesSecondaryGrid(state) ? state.matter_liquid_grid : state.grid;
}

inline const FluidSim::FluidGrid& liquidGrid(const SimulationGridDomainState& state) {
    return liquidUsesSecondaryGrid(state) ? state.matter_liquid_grid : state.grid;
}

inline FluidSim::FluidGrid& gasGrid(SimulationGridDomainState& state) {
    return state.type == SimulationDomainType::Matter && state.matter_liquid_active
        ? state.matter_liquid_grid : state.grid;
}

inline const FluidSim::FluidGrid& gasGrid(const SimulationGridDomainState& state) {
    return state.type == SimulationDomainType::Matter && state.matter_liquid_active
        ? state.matter_liquid_grid : state.grid;
}

inline void translatePhaseGrids(SimulationGridDomainState& state, const Vec3& delta) {
    state.grid.origin += delta;
    if (state.type == SimulationDomainType::Matter) {
        state.matter_liquid_grid.origin += delta;
    }
}

inline void translateLiquidParticles(SimulationGridDomainState& state, const Vec3& delta) {
    for (Vec3& position : state.particles.position) {
        position += delta;
    }

}

inline Vec3 gridBoundsMax(const FluidSim::FluidGrid& grid) {
    return grid.origin + Vec3(
        static_cast<float>(grid.nx),
        static_cast<float>(grid.ny),
        static_cast<float>(grid.nz)) * grid.voxel_size;
}

inline bool gridContains(const FluidSim::FluidGrid& grid, const Vec3& position) {
    const Vec3 hi = gridBoundsMax(grid);
    return grid.nx > 0 && grid.ny > 0 && grid.nz > 0 && grid.voxel_size > 0.0f &&
        position.x >= grid.origin.x && position.y >= grid.origin.y &&
        position.z >= grid.origin.z && position.x < hi.x &&
        position.y < hi.y && position.z < hi.z;
}

inline bool gridsOverlap(const FluidSim::FluidGrid& a, const FluidSim::FluidGrid& b) {
    if (a.nx <= 0 || a.ny <= 0 || a.nz <= 0 || !(a.voxel_size > 0.0f) ||
        b.nx <= 0 || b.ny <= 0 || b.nz <= 0 || !(b.voxel_size > 0.0f)) {
        return false;
    }
    const Vec3 lo = Vec3::max(a.origin, b.origin);
    const Vec3 hi = Vec3::min(gridBoundsMax(a), gridBoundsMax(b));
    return lo.x < hi.x && lo.y < hi.y && lo.z < hi.z;
}

// Keep storage, GPU residency and active solver coordinates in one transaction.
// The destructor restores the gas view on continue, return and exception paths.
class MatterLiquidScope {
public:
    MatterLiquidScope(
        SimulationGridDomainState& state,
        SimulationGridDomainComputeBuffers* primary,
        SimulationGridDomainComputeBuffers* secondary)
        : state_(state), primary_(primary), secondary_(secondary),
          bounds_min_(state.bounds_min), bounds_max_(state.bounds_max),
          nx_(state.resolution_x), ny_(state.resolution_y), nz_(state.resolution_z),
          voxel_size_(state.voxel_size) {
        if (state.type != SimulationDomainType::Matter || state.matter_liquid_active) {
            return;
        }
        std::swap(state.grid, state.matter_liquid_grid);
        if (primary_ && secondary_) {
            std::swap(*primary_, *secondary_);
        }
        state.matter_liquid_active = true;
        state.bounds_min = state.grid.origin;
        state.bounds_max = gridBoundsMax(state.grid);
        state.resolution_x = state.grid.nx;
        state.resolution_y = state.grid.ny;
        state.resolution_z = state.grid.nz;
        state.voxel_size = state.grid.voxel_size;
        active_ = true;
    }

    ~MatterLiquidScope() {
        restore();
    }

    MatterLiquidScope(const MatterLiquidScope&) = delete;
    MatterLiquidScope& operator=(const MatterLiquidScope&) = delete;
    MatterLiquidScope(MatterLiquidScope&&) = delete;
    MatterLiquidScope& operator=(MatterLiquidScope&&) = delete;

    void restore() {
        if (!active_) {
            return;
        }
        std::swap(state_.grid, state_.matter_liquid_grid);
        if (primary_ && secondary_) {
            std::swap(*primary_, *secondary_);
        }
        state_.bounds_min = bounds_min_;
        state_.bounds_max = bounds_max_;
        state_.resolution_x = nx_;
        state_.resolution_y = ny_;
        state_.resolution_z = nz_;
        state_.voxel_size = voxel_size_;
        state_.matter_liquid_active = false;
        active_ = false;
    }

private:
    SimulationGridDomainState& state_;
    SimulationGridDomainComputeBuffers* primary_;
    SimulationGridDomainComputeBuffers* secondary_;
    Vec3 bounds_min_;
    Vec3 bounds_max_;
    int nx_;
    int ny_;
    int nz_;
    float voxel_size_;
    bool active_ = false;
};

} // namespace RayTrophiSim::Fluid
