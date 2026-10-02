#pragma once

#include <cstddef>

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;

namespace Fluid {

// One-way subgrid drag for low-mass mist parcels inside an overlapping gas
// domain. The liquid parcel remains the mass owner; this only relaxes its
// velocity toward the carrier gas and therefore cannot duplicate mass.
std::size_t applyMistGasDrag(SimulationGridDomainState& fluid_state,
                             const SimulationGridDomainDesc& gas_desc,
                             const SimulationGridDomainState& gas_state,
                             float dt);

} // namespace Fluid
} // namespace RayTrophiSim
