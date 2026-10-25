#include "Fluid/MatterFogSnapshot.h"
#include "Fluid/GasInventoryGpu.h"
#include "Fluid/GasSaturation.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"
#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid {
namespace {
FogFieldGrid fogGrid(const FluidSim::FluidGrid& grid) {
    return {grid.nx, grid.ny, grid.nz, grid.voxel_size, grid.origin};
}
}
bool prepareMatterFogSnapshot(ParticleSimulationSystem& runtime, std::size_t index,
    const SimulationComputeContext& compute, MatterFogFields& output,
    std::string& snapshot, std::string& error) {
    const auto refuse = [&](const std::string& message) {
        error = message;
        return false;
    };
    if (index >= runtime.gridDomains().size()) {
        return refuse("Matter Fog domain index is out of range");
    }
    const auto& desc = runtime.gridDomains()[index];
    const auto& states = runtime.gridDomainStates();
    if (desc.type != RayTrophiSim::SimulationDomainType::Matter) {
        return refuse("Fog field inspection requires a Matter domain");
    }
    if (index >= states.size() || !states[index].valid) {
        return refuse("Matter domain has no live state; initialize the scene first");
    }
    const auto& state = states[index];
    if (state.matter_liquid_active) {
        return refuse("Fog inspection requires a completed simulation step");
    }
    const auto& grid = gasGrid(state);
    // Bound both the independent builder and this diagnostic's snapshot/reference
    // buffers. No per-frame readback or persistent dense allocation is introduced.
    constexpr std::size_t budget = 128u * 1024u * 1024u;
    std::size_t cells = 1;
    for (const int dimension : {grid.nx, grid.ny, grid.nz}) {
        if (dimension <= 0 || cells > budget / sizeof(float) /
                static_cast<std::size_t>(dimension)) {
            return refuse("Fog inspection grid exceeds diagnostic memory budget");
        }
        cells *= static_cast<std::size_t>(dimension);
    }
    // Up to six inventory sidecars, two gas snapshot fields and one reference.
    constexpr std::size_t snapshot_bytes_per_cell = 9u * sizeof(float);
    if (cells > budget / snapshot_bytes_per_cell) {
        return refuse("Fog inspection snapshot exceeds diagnostic memory budget");
    }
    if (state.particles.size() > budget - cells * snapshot_bytes_per_cell) {
        return refuse("Fog inspection selection exceeds diagnostic memory budget");
    }
    const auto plan = resolveFluidViews(desc,
        cachedDistinctViewKeys(state.particles, state.version));
    std::vector<float> density;
    std::vector<float> temperature;
    GasFogFieldView gas;
    gas.grid = fogGrid(grid);
    gas.temperature_units = GasFogTemperatureUnits::Kelvin;
    snapshot = "hidden";
    if (plan.phase_output.gas != LabelRoute::Hidden) {
        if (runtime.gasGpuFieldView(index, compute).valid()) {
            if (!runtime.downloadGasDenseFields(index, compute, density, temperature)) {
                return refuse("Cannot read authoritative GPU gas fields");
            }
            gas.density = &density;
            snapshot = "device_readback";
        } else {
            gas.density = &grid.density;
            temperature = grid.temperature;
            snapshot = "host";
        }
        if (!temperature.empty()) {
            if (temperature.size() != cells) {
                return refuse("Gas temperature snapshot has mismatched dimensions");
            }
            // Matter stores a buoyancy/virtual-temperature field, not Kelvin/3000.
            // Use the same physical scale and humidity inversion as gas probes
            // and heat coupling. Signed field values are valid around ambient.
            const auto& world = runtime.worldThermal();
            const float ambient = RayTrophiSim::fluidDomainAmbientKelvin(desc, world);
            const float gas_ambient = world.ambientNormalized();
            const float per_unit = world.kelvin_per_unit;
            if (!std::isfinite(ambient) || !std::isfinite(gas_ambient) ||
                !std::isfinite(per_unit) || !(per_unit > 0.0f)) {
                return refuse("Gas temperature scale is invalid");
            }
            const auto inventory = captureGasInventoryHostImage(state);
            if (!inventory.mass.empty() && inventory.mass.size() != cells) {
                return refuse("Gas vapour inventory has mismatched dimensions");
            }
            const auto* profile = RayTrophiSim::tryFindSubstance(
                state.gas_stats.condensable_substance.empty() ? std::string("Water") :
                    state.gas_stats.condensable_substance);
            const float molar = profile && profile->vapor_molar_mass_kg_mol > 0.0f
                ? profile->vapor_molar_mass_kg_mol : 0.018015f;
            const float volume = grid.voxel_size * grid.voxel_size * grid.voxel_size;
            for (std::size_t i = 0; i < cells; ++i) {
                if (!std::isfinite(temperature[i])) {
                    return refuse("Gas temperature field contains nonfinite values");
                }
                const float virtual_kelvin = ambient +
                    (temperature[i] - gas_ambient) * per_unit;
                if (!std::isfinite(virtual_kelvin)) {
                    return refuse("Gas temperature conversion overflowed");
                }
                const float vapour = inventory.mass.empty() ? 0.0f : inventory.mass[i];
                if (!std::isfinite(vapour) || vapour < 0.0f) {
                    return refuse("Gas vapour inventory is invalid");
                }
                temperature[i] = airKelvinFromVirtual(
                    std::max(1.0f, virtual_kelvin), vapour, volume, molar, 1.0f);
            }
            gas.temperature = &temperature;
        }
        gas.authoritative = true;
    }
    MatterFogFieldInput input;
    input.particles = &state.particles;
    input.plan = &plan;
    input.carrier_grid = fogGrid(grid);
    input.parcel_density = 1.0f /
        static_cast<float>(std::max(1, desc.fluid_params.particles_per_cell));
    input.default_substance_tag = substanceTag(desc.fluid_params.default_substance);
    input.gas = plan.phase_output.gas == LabelRoute::Hidden ? nullptr : &gas;
    input.max_dense_bytes = budget;
    return buildMatterFogFields(input, output, error);
}
} // namespace RayTrophiSim::Fluid
