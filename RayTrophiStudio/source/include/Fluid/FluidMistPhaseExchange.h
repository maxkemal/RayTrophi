#pragma once

#include <cstddef>
#include <cstdint>

namespace RayTrophiSim {

class MatterExchangeLedger;
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;

namespace GridFluid {
struct SolverParams;
}

namespace Fluid {

struct FluidMistPhaseExchangeStats {
    bool ran = false;
    std::size_t particles_removed = 0;
    double transferred_mass_kg = 0.0;
    double transferred_energy_j = 0.0;
};

// Converts every labelled mist parcel inside one gas domain into gas-phase
// inventory. The remaining APIC mass is debited exactly once and recorded as
// a balanced mist_to_gas exchange.
FluidMistPhaseExchangeStats transferMistToGas(
    const SimulationGridDomainDesc& fluid_domain,
    SimulationGridDomainState& fluid_state,
    const SimulationGridDomainDesc& gas_domain,
    SimulationGridDomainState& gas_state,
    uint64_t event_id_base,
    MatterExchangeLedger& ledger);

// Carries physical gas kg/J with the solved gas velocity and the same advection
// scheme and boundary rule as the gas scalar channels. Visual dissipation does
// not destroy physical inventory; only transport through an open boundary can.
bool advectGasPhaseInventory(
    SimulationGridDomainState& gas_state,
    const GridFluid::SolverParams& params,
    float dt);

} // namespace Fluid
} // namespace RayTrophiSim
