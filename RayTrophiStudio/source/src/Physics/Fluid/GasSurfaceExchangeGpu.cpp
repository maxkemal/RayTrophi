#include "Fluid/GasSurfaceExchangeGpu.h"
#include "Fluid/GasInventoryGpu.h"
#include "Fluid/GasSurfaceExchangePlan.h"
#include "Fluid/GasPhaseMomentum.h"
#include "Fluid/FluidDomainSubstance.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/FluidThermalLiquid.h"
#include "Fluid/MatterPhaseGrid.h"
#include "Fluid/SubstanceTag.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <vector>

namespace RayTrophiSim::Fluid {
namespace {
using Parcel = GasSurfaceParcel;
using Group = GasSurfaceGroup;
using Result = GasSurfaceResult;
static_assert(sizeof(Parcel) == 64 && offsetof(Parcel, capacity) == 40);
static_assert(sizeof(Group) == 16);
static_assert(sizeof(Result) == 64 && offsetof(Result, kelvin) == 32);

struct Constants {
    uint32_t groups;
    float cell_volume;
    float ambient;
    float gas_ambient;
    float kelvin_per_unit;
    float humidity;
    float dt;
    uint32_t inventory;
    uint32_t stamp;
};
static_assert(sizeof(Constants) == 36);

struct CommitConstants {
    uint32_t count;
    uint32_t inventory;
};
static_assert(sizeof(CommitConstants) == 8);

bool reserve(SimulationComputeContext& compute, ComputeBufferHandle& handle,
             std::size_t bytes, const char* name) {
    if (!handle.valid()) {
        const auto usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                           ComputeBufferUsage::Download | ComputeBufferUsage::ReadWrite;
        handle = compute.createBuffer({name, bytes, usage});
        return handle.valid();
    }
    return compute.getBufferSize(handle) >= bytes || compute.resizeBuffer(handle, bytes);
}
} // namespace

bool exchangeGasSurfaceGpu(
    const SimulationGridDomainDesc& fluid_domain, SimulationGridDomainState& fluid,
    const SimulationGridDomainDesc& gas_domain, SimulationGridDomainState& gas,
    SimulationGridDomainComputeBuffers& gas_buffers, SimulationComputeContext& compute,
    const MaterialTemperatureScale& scale, float ambient_kelvin,
    float gas_ambient_normalized, float drying_scale, float dt,
    uint64_t event_id_base, MatterExchangeLedger& ledger) {
    if (!(dt > 0.0f) || !fluid_domain.enabled || !gas_domain.enabled ||
        !simulationDomainHasLiquid(fluid_domain.type) ||
        !simulationDomainHasGas(gas_domain.type) || !fluid.valid || !gas.valid ||
        fluid.particles.empty() || !gridsOverlap(liquidGrid(fluid), gasGrid(gas)) ||
        !(scale.kelvin_per_unit > 0.0f)) {
        return true; // No overlap is a legitimate no-op.
    }
    const auto fail = [&](const char* reason) {
        holdGasInventory(gas, reason);
        return false;
    };
    const auto& gas_grid = gasGrid(gas);
    if (!(gas_grid.voxel_size > 0.0f) || !std::isfinite(dt) ||
        gas_grid.temperature.size() != gas_grid.getCellCount() ||
        gas_grid.density.size() != gas_grid.getCellCount() ||
        !gas_buffers.deviceHasCurrent(SimulationGridDomainComputeBuffers::GridField::Temperature) ||
        !gas_buffers.deviceHasCurrent(SimulationGridDomainComputeBuffers::GridField::Density)) {
        return fail("gas surface device temperature/density is unavailable or stale; reset required");
    }
    auto owner = gasInventoryDevice(gas);
    if (!owner || !flushGasInventoryDeposits(gas)) {
        return fail("gas surface inventory owner/deposit preparation failed; reset required");
    }
    const auto plan = buildGasSurfaceExchangePlan(fluid_domain, fluid, gas_domain, gas);
    if (!plan.error.empty()) {
        holdGasInventory(gas, plan.error.c_str());
        return false;
    }
    const auto& records = plan.records;
    const auto& groups = plan.groups;
    if (records.empty()) {
        return true;
    }
    const auto& grid = gasGrid(gas);
    const auto inventory_inputs = gasInventoryReadBuffers(gas);
    const auto rain_inputs = gasInventoryRainBuffers(gas);
    const auto landing = gasInventoryRainLanding(gas);
    if (!inventory_inputs[0].valid() || !inventory_inputs[1].valid() ||
        !rain_inputs[0].valid() || !rain_inputs[1].valid() || !landing.valid()) {
        return fail("gas surface inventory input allocation failed; reset required");
    }
    std::vector<Result> results(records.size());
    auto& work = owner->surface;
    if (!reserve(compute, work[0], records.size() * sizeof(Parcel), "GasSurfaceParcels") ||
        !reserve(compute, work[1], groups.size() * sizeof(Group), "GasSurfaceColumns") ||
        !reserve(compute, work[2], results.size() * sizeof(Result), "GasSurfaceResults") ||
        !reserve(compute, work[3], groups.size() * 8 * sizeof(float), "GasSurfaceCommit")) {
        return fail("gas surface work buffer allocation failed; reset required");
    }
    compute.beginTransferBatch();
    bool ok = compute.uploadBuffer(work[0], records.data(), records.size() * sizeof(Parcel));
    ok = compute.uploadBuffer(work[1], groups.data(), groups.size() * sizeof(Group)) && ok;
    ok = compute.endTransferBatch() && ok;
    if (!ok) {
        return fail("gas surface parcel upload failed; reset required");
    }
    const Constants pc{static_cast<uint32_t>(groups.size()),
                       grid.voxel_size * grid.voxel_size * grid.voxel_size,
                       ambient_kelvin, gas_ambient_normalized, scale.kelvin_per_unit,
                       std::clamp(1.0f - drying_scale, 0.0f, 1.0f), dt,
                       owner->storage_initialized ? 1u : 0u, gas.gas_rain_stamp};
    ComputeBufferHandle bindings[11] = {
        work[0], work[1], work[2], inventory_inputs[0], inventory_inputs[1],
        gas_buffers.temperature, gas_buffers.density, work[3],
        rain_inputs[0], rain_inputs[1], landing
    };
    ComputeDispatch command;
    command.kernel = "sim_inventory_surface_exchange";
    command.buffers = bindings;
    command.buffer_count = 11;
    command.constants = &pc;
    command.constants_size = sizeof(pc);
    command.groups = gasInventoryDispatch(pc.groups, 128);
    if (!compute.dispatch(command)) {
        return fail("gas surface exchange dispatch failed; reset required");
    }
    // No gas kg/J grid comes back. Only the liquid parcel publication/ledger
    // result is needed by the existing canonical host SoA liquid consumers.
    // Record the readback after the pending exchange dispatch. A standalone
    // immediate copy can otherwise run before the recorded compute stream.
    compute.beginTransferBatch();
    const bool downloaded = compute.downloadBuffer(
        work[2], results.data(), results.size() * sizeof(Result));
    const bool completed = compute.endTransferBatch();
    if (!downloaded || !completed) {
        return fail("gas surface result download failed; reset required");
    }
    if (!validateGasSurfaceResults(plan, results)) {
        for (std::size_t r = 0; r < results.size(); ++r) {
            const auto& value = results[r];
            const auto& input = plan.records[r];
            const double budget = static_cast<double>(input.rest_mass) * input.fraction;
            if (value.error != 0.0f || !(value.kelvin > 0.0f) ||
                !std::isfinite(value.kelvin) || !std::isfinite(value.fraction) ||
                value.fraction < 0.0f || value.fraction > input.fraction + 1.0e-5f ||
                !std::isfinite(value.mass) || value.mass < -(budget + 1.0e-8) ||
                value.mass > budget + 1.0e-8 || !std::isfinite(value.sensible) ||
                !std::isfinite(value.latent) || !std::isfinite(value.heat_to_gas)) {
                const std::string reason = "gas surface invalid result: parcel=" +
                    std::to_string(r) + " K=" + std::to_string(value.kelvin) +
                    " fraction=" + std::to_string(value.fraction) +
                    " mass=" + std::to_string(value.mass) +
                    " heat=" + std::to_string(value.heat_to_gas) +
                    " error=" + std::to_string(value.error) + "; reset required";
                holdGasInventory(gas, reason.c_str());
                return false;
            }
        }
        return fail("gas surface result count is inconsistent; reset required");
    }
    const bool evaporating = std::any_of(results.begin(), results.end(),
                                        [](const Result& value) { return value.mass > 0.0; });
    if (evaporating && !ensureGasInventoryStorage(gas)) {
        return fail("gas surface inventory output allocation failed; reset required");
    }
    const auto inventory_outputs = gasInventoryReadBuffers(gas);
    const auto rain_outputs = gasInventoryRainBuffers(gas);
    // A pool allocated just now (evaporation) held no rain: the exchange read
    // zero there and the commit writes zero back.
    const CommitConstants commit_pc{pc.groups, owner->storage_initialized ? 1u : 0u};
    ComputeBufferHandle commit[8] = {work[1], work[3], inventory_outputs[0], inventory_outputs[1],
                                    gas_buffers.temperature, gas_buffers.density,
                                    rain_outputs[0], rain_outputs[1]};
    command.kernel = "sim_inventory_surface_commit";
    command.buffers = commit;
    command.buffer_count = 8;
    command.constants = &commit_pc;
    command.constants_size = sizeof(commit_pc);
    if (!compute.dispatch(command)) {
        return fail("gas surface commit dispatch failed; reset required");
    }
    publishGasSurfaceResults(plan, results, fluid_domain, fluid, gas_domain, gas,
                             event_id_base, ledger, true);
    owner->populated = owner->populated || evaporating;
    markGasInventoryDeviceChanged(gas);
    gas_buffers.markDeviceWrote(SimulationGridDomainComputeBuffers::GridField::Temperature);
    gas_buffers.markDeviceWrote(SimulationGridDomainComputeBuffers::GridField::Density);
    return true;
}
bool exchangeGasSurfaceGpuDomains(
    const std::vector<std::size_t>& targets,
    const std::vector<SimulationGridDomainDesc>& domains,
    std::vector<SimulationGridDomainState>& states, std::size_t gas_index,
    SimulationGridDomainComputeBuffers& buffers, SimulationComputeContext& compute,
    const WorldThermalState& world, float dt, uint64_t step, MatterExchangeLedger& ledger) {
    for (const std::size_t source : targets) {
        const uint64_t event = (step << 32u) |
            (static_cast<uint64_t>(source & 0xffffu) << 16u) |
            static_cast<uint64_t>(gas_index & 0xffffu);
        if (!exchangeGasSurfaceGpu(domains[source], states[source],
                                   domains[gas_index], states[gas_index], buffers, compute,
                                   world.scale(), fluidDomainAmbientKelvin(domains[source], world),
                                   world.ambientNormalized(), world.dryingScale(), dt,
                                   event ^ (uint64_t{1} << 61u), ledger)) {
            return false;
        }
    }
    return flushGasInventoryDeposits(states[gas_index]) &&
        flushGasPhaseMomentum(states[gas_index], domains[gas_index], world.scale(),
            fluidDomainAmbientKelvin(domains[gas_index], world), world.ambientNormalized(),
            &compute, &buffers);
}
} // namespace RayTrophiSim::Fluid
