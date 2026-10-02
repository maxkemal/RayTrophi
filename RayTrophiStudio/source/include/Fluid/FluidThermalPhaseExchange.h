#pragma once

#include "FluidThermalLiquid.h"

#include <cstdint>

namespace RayTrophiSim {

class MatterExchangeLedger;
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;

namespace Fluid {

struct FluidThermalPhaseExchangeStats {
    std::size_t frozen_particles = 0;
    std::size_t melted_particles = 0;
    double frozen_mass_kg = 0.0;
    double melted_mass_kg = 0.0;
};

// Runs the thermal freeze pass and records every liquid/solid phase change in
// the shared matter ledger. Mass, sensible heat and latent heat use the same
// physical parcel/profile data as combustion and mist transfer.
FluidThermalPhaseExchangeStats updateThermalFreezeAndRecord(
    const SimulationGridDomainDesc& domain,
    SimulationGridDomainState& state,
    const APICSolverParams& params,
    uint64_t event_id_base,
    MatterExchangeLedger& ledger);

} // namespace Fluid
} // namespace RayTrophiSim
