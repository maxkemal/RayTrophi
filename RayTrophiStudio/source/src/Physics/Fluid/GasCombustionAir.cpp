#include "Fluid/GasCombustionAir.h"

#include "Fluid/GasInventoryGpu.h"
#include "Fluid/GasSaturation.h"
#include "Fluid/MatterPhaseGrid.h"
#include "MaterialStateField.h"
#include "ParticleSimulation.h"

#include <algorithm>

namespace RayTrophiSim::Fluid {

GridFluid::CombustionAir gasCombustionAir(const SimulationGridDomainDesc& domain,
                                          const SimulationGridDomainState& state,
                                          const WorldThermalState& world) {
    GridFluid::CombustionAir air;
    const auto scale = world.scale();
    air.kelvin_per_unit = scale.kelvin_per_unit;
    air.ambient_kelvin = fluidDomainAmbientKelvin(domain, world);
    air.gas_ambient = world.ambientNormalized();
    // The same boundary condition MSF pyrolysis reads: the domain overrides the
    // world inside its own box.
    air.oxygen = domain.thermal_override_enabled ? domain.thermal_oxygen
                                                 : world.oxygen_availability;
    const auto& grid = gasGrid(state);
    air.cell_volume = grid.voxel_size * grid.voxel_size * grid.voxel_size;
    const CondensableVapour vapour = resolveCondensableVapour(domain);
    if (!vapour.valid()) {
        return air;
    }
    air.vapour_molar = vapour.molar_kg_mol;
    if (!gasInventoryDevice(state) &&
        state.gas_phase_mass_kg.size() == grid.getCellCount()) {
        air.vapour_kg = state.gas_phase_mass_kg.data();
        air.vapour_cells = state.gas_phase_mass_kg.size();
    }
    return air;
}

ComputeBufferHandle gasCombustionVapourBuffer(SimulationGridDomainState& state) {
    // ★ Before storage exists the read buffers are ONE zero float; a kernel
    // indexing it per cell would read past its end. No storage = no vapour.
    const auto owner = gasInventoryDevice(state);
    if (!owner || !owner->storage_initialized) {
        return {};
    }
    return gasInventoryReadBuffers(state)[0];
}

} // namespace RayTrophiSim::Fluid
