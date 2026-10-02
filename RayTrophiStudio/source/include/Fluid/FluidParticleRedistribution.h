#pragma once

#include <cstddef>
#include <cstdint>

namespace FluidSim { class FluidGrid; }

namespace RayTrophiSim::Fluid {
class FluidParticles;
struct APICSolverParams;

// Locally relocate existing liquid parcels. Never emit, delete, reorder or
// overwrite parcel state; only position changes. Returns the number moved.
std::size_t redistributeFluidParticles(FluidParticles& particles,
                                      const FluidSim::FluidGrid& grid,
                                      const APICSolverParams& params,
                                      uint32_t step_seed);
}
