#pragma once

#include <cstddef>

namespace FluidSim { class FluidGrid; }

namespace RayTrophiSim::Fluid {
class FluidParticles;

// Relocates particles that are already inside a collider solid cell to the
// nearest non-solid cell. Mass/chemistry sidecars remain untouched.
std::size_t recoverParticlesFromSolidCells(FluidParticles& particles,
                                           const FluidSim::FluidGrid& grid,
                                           int maximum_search_radius = 6);
}
