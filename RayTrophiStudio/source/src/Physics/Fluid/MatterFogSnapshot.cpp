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
    std::string& snapshot, std::string& error, MatterFogSnapshotPurpose purpose) {
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
    const auto plan = resolveFluidViews(desc,
        cachedDistinctViewKeys(state.particles, state.version));
    const bool gas_visible = plan.phase_output.gas != LabelRoute::Hidden;
    // Inspection retains its small on-demand ceiling. Publication has a distinct
    // domain-bounded policy; a diagnostic guard must not silently hide live Fog.
    const std::size_t budget = matterFogSnapshotBudgetBytes(purpose,
        desc.resource_budget_mb, desc.enforce_resource_budget);
    std::size_t cells = 1;
    for (const int dimension : {grid.nx, grid.ny, grid.nz}) {
        if (dimension <= 0 || cells > budget / sizeof(float) /
                static_cast<std::size_t>(dimension)) {
            return refuse("Fog snapshot grid exceeds memory budget");
        }
        cells *= static_cast<std::size_t>(dimension);
    }
    // Up to six inventory sidecars, two gas snapshot fields and one reference.
    const std::size_t snapshot_bytes_per_cell =
        gas_visible ? (9u + 2u * state.gas_source_images.size()) * sizeof(float) : 0u;
    if (snapshot_bytes_per_cell != 0u && cells > budget / snapshot_bytes_per_cell) {
        return refuse("Fog snapshot exceeds memory budget: " +
            std::to_string(budget / (1024u * 1024u)) + " MiB");
    }
    if (state.particles.size() > budget - cells * snapshot_bytes_per_cell) {
        return refuse("Fog snapshot selection exceeds memory budget");
    }
    std::vector<float> density;
    std::vector<float> temperature;
    SimulationGridDomainState source_snapshot;
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
        if (!state.gas_source_images.empty()) {
            source_snapshot.gas_source_images = state.gas_source_images;
            source_snapshot.gas_source_device = state.gas_source_device;
            source_snapshot.gas_source_device_current = state.gas_source_device_current;
            if (!snapshotGasSources(source_snapshot)) {
                return refuse("Cannot read authoritative gas source contributions");
            }
            for (const auto& source : runtime.flowSources()) {
                auto found = source_snapshot.gas_source_images.find(source.timeline_uid);
                if (found != source_snapshot.gas_source_images.end()) {
                    found->second.name = source.name;
                }
            }
            gas.sources = &source_snapshot.gas_source_images;
        }
    }
    MatterFogFieldInput input;
    input.particles = &state.particles;
    input.plan = &plan;
    input.carrier_grid = fogGrid(grid);
    input.spread_voxels = desc.fluid_fog_spread_voxels;
    input.parcel_density = 1.0f /
        static_cast<float>(std::max(1, desc.fluid_params.particles_per_cell));
    input.default_substance_tag = substanceTag(desc.fluid_params.default_substance);
    input.gas = plan.phase_output.gas == LabelRoute::Hidden ? nullptr : &gas;
    // Gas snapshots survive into the builder. Inventory conversion scratch
    // above is already destroyed; account retained density/temperature/source
    // images here so fields + gather scratch cannot each reuse the full limit.
    const std::size_t retained_snapshot_bytes = gas_visible
        ? cells * (2u + 2u * state.gas_source_images.size()) * sizeof(float) : 0u;
    input.max_dense_bytes = budget - retained_snapshot_bytes;
    return buildMatterFogFields(input, output, error);
}
} // namespace RayTrophiSim::Fluid
