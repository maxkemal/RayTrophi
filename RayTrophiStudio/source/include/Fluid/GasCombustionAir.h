#pragma once

// FG1 (2026-10-18): the air a gas domain's fire burns in. See
// GridFluid::CombustionAir for the physics; this only resolves its inputs.

#include "SimulationCompute.h"
#include "GridFluidSolver.h"

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
struct WorldThermalState;
namespace Fluid {

// Vapour molar mass, cell volume, the ambient/temperature scale and the oxygen
// boundary condition of this gas domain. The host vapour pointer is set only
// while the host copy is the authority (no device inventory); on the device
// path the shader reads the inventory's vapour pool instead (see
// gasCombustionVapourBuffer). vapour_molar stays 0 when the domain has no
// condensable vapour: dry air, the solver before FG1.
GridFluid::CombustionAir gasCombustionAir(const SimulationGridDomainDesc& domain,
                                          const SimulationGridDomainState& state,
                                          const WorldThermalState& world);

// The device vapour pool, one float per cell. Invalid when the domain has no
// device inventory or its storage does not exist yet (then there is no vapour).
ComputeBufferHandle gasCombustionVapourBuffer(SimulationGridDomainState& state);

} // namespace Fluid
} // namespace RayTrophiSim
