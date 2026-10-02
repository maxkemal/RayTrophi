#pragma once

#include "../Vec3.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace RayTrophiSim {

struct SimulationGridDomainDesc;
struct SimulationGridDomainState;

namespace Fluid {

struct FluidGasMovingBoundaryStats {
    std::size_t source_domains = 0;
    std::size_t source_particles = 0;
    std::size_t boundary_cells = 0;
    Vec3 mean_velocity;
};

// Rasterizes all live liquid in domains overlapping gas_domain_index into the
// gas grid. A cell becomes a boundary at 25% physical volume fill and carries
// the mass-weighted liquid velocity. The caller applies the returned cells via
// the shared solid-overlay path used by every CPU/GPU solver.
FluidGasMovingBoundaryStats buildFluidGasMovingBoundary(
    const std::vector<SimulationGridDomainDesc>& domains,
    const std::vector<SimulationGridDomainState>& states,
    std::size_t gas_domain_index,
    std::vector<uint32_t>& cells_out,
    std::vector<Vec3>& velocities_out);

} // namespace Fluid
} // namespace RayTrophiSim
