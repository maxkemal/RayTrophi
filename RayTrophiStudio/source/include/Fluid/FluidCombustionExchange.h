#pragma once

#include <cstddef>
#include <cstdint>

namespace RayTrophiSim {

class MatterExchangeLedger;
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;

namespace Fluid {

struct FluidCombustionExchangeStats {
    bool ran = false;
    std::size_t particles_changed = 0;
    std::size_t particles_removed = 0;
    double transferred_mass_kg = 0.0;
    double transferred_energy_j = 0.0;
};

// Debits APIC parcel mass and deposits the exact same kg into one gas domain.
// Gas solver fields receive dimensionless tracer values; physical kg/J remain
// in explicit sidecars and the MatterExchangeLedger.
FluidCombustionExchangeStats transferFluidCombustionToGas(
    const SimulationGridDomainDesc& fluid_domain,
    SimulationGridDomainState& fluid_state,
    const SimulationGridDomainDesc& gas_domain,
    SimulationGridDomainState& gas_state,
    float dt,
    uint64_t event_id_base,
    MatterExchangeLedger& ledger);

} // namespace Fluid
} // namespace RayTrophiSim
