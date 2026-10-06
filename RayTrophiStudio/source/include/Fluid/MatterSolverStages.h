#pragma once

#include "APICFluidSolver.h"

#include <array>
#include <string>

namespace RayTrophiSim::Fluid {

// One workspace per model: preparing another model must not overwrite this
// model's FLIP baseline. Not serialized; rebuilt at each common substep.
struct MatterModelGridStep {
    APICFlipSnapshot flip;
    APICSolverParams params;
    APICSolverStats projection_stats;
    FluidParticles* particles = nullptr;
    FluidSim::FluidGrid* grid = nullptr;
    std::array<int, 3> dimensions{};
    Vec3 origin;
    float voxel = 0.0f;
    float dt = 0.0f;
    float time_seconds = 0.0f;
    uint64_t allocator = 0;
    std::vector<uint64_t> identities;
    std::size_t particle_count = 0;
    bool ready = false;
};

// Prepare = external forces/P2G/boundary/viscosity + liquid pressure or
// granular stress field. No G2P, particle advection, reseed or UVW aging.
bool prepareMatterModelGrid(FluidParticles& particles, FluidSim::FluidGrid& grid,
                           const APICSolverParams& params, float dt,
                           const SimulationForceFieldSnapshot* forces,
                           float time_seconds, MatterModelGridStep& workspace,
                           std::string& error);

// Contact may modify model grid velocity fields between these calls.
// Finish consumes the projected/contact field exactly once, without rerunning
// force integration/P2G/viscosity/pressure, then performs constitutive + tail.
bool finishMatterModelGrid(MatterModelGridStep& workspace, APICSolverStats& stats,
                          std::string& error);

} // namespace RayTrophiSim::Fluid
