#include "Fluid/GasInventoryGpu.h"
#include "Fluid/MatterPhaseGrid.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace RayTrophiSim::Fluid {
namespace {
constexpr uint32_t kThreads = 256;

struct AdvectionConstants {
    int nx;
    int ny;
    int nz;
    float voxel_size;
    float dt;
    int boundary;
    float background = 0.0f;
};
static_assert(sizeof(AdvectionConstants) == 28);

struct ReductionConstants {
    uint32_t cells;
    uint32_t blocks;
    uint32_t closed;
    uint32_t phase;
};
static_assert(sizeof(ReductionConstants) == 16);

} // namespace

bool advectGasInventoryGpu(SimulationGridDomainState& state,
                           const GridFluid::SolverParams& params,
                           float dt,
                           SimulationComputeContext& compute,
                           SimulationGridDomainComputeBuffers& buffers,
                           bool carrier_valid) {
    auto& grid = gasGrid(state);
    const std::size_t cells = grid.getCellCount();
    if (compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch() || !state.valid ||
        !simulationDomainHasGas(state.type) || !(dt > 0.0f) || !std::isfinite(dt) ||
        grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 || !(grid.voxel_size > 0.0f) ||
        cells > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    auto owner = gasInventoryDevice(state);
    if (!owner || owner->cells != cells || !flushGasInventoryDeposits(state)) {
        return false;
    }
    if (!owner->populated) {
        state.gas_stats.inventory_transport = "gpu_resident_idle";
        return true;
    }
    if (!carrier_valid || !buffers.vel_x.valid() || !buffers.vel_y.valid() ||
        !buffers.vel_z.valid() || !buffers.gas_solid_mask.valid()) {
        return false;
    }
    auto& scratch = *owner;
    const uint32_t blocks = static_cast<uint32_t>((cells + kThreads - 1) / kThreads);

    const AdvectionConstants pc{grid.nx, grid.ny, grid.nz, grid.voxel_size,
                                dt, static_cast<int>(params.boundary)};
    std::array<ComputeBufferHandle, 3> outputs{};
    for (std::size_t field = 0; field < 3; ++field) {
        const std::size_t base = field * 3;
        ComputeBufferHandle bindings[7] = {
            buffers.vel_x, buffers.vel_y, buffers.vel_z,
            scratch.fields[base], scratch.fields[base + 1],
            scratch.fields[base + 2], buffers.gas_solid_mask
        };
        ComputeDispatch command;
        command.kernel = "sim_inventory_advect_scalar";
        command.buffers = bindings;
        command.buffer_count = 7;
        command.constants = &pc;
        command.constants_size = sizeof(pc);
        command.groups = gasInventoryDispatch(cells);
        if (!compute.dispatch(command)) {
            return false;
        }
        outputs[field] = scratch.fields[base + 1];
        if (params.advection == GridFluid::Advection::MacCormack) {
            command.kernel = "sim_inventory_maccormack_scalar";
            if (!compute.dispatch(command)) {
                return false;
            }
            outputs[field] = scratch.fields[base + 2];
        }
    }

    // Conservation is a reduction and normalization on the device. In an open
    // domain only numerical creation is removed; physical outflow is retained.
    ComputeBufferHandle reduction[6] = {
        scratch.fields[0], outputs[0], scratch.fields[3], outputs[1],
        scratch.partials, scratch.totals
    };
    ReductionConstants rc{static_cast<uint32_t>(cells), blocks,
                           params.boundary == GridFluid::Boundary::Open ? 0u : 1u, 0u};
    ComputeDispatch command;
    command.kernel = "sim_inventory_conserve";
    command.buffers = reduction;
    command.buffer_count = 6;
    command.constants = &rc;
    command.constants_size = sizeof(rc);
    command.groups = gasInventoryDispatch(cells);
    if (!compute.dispatch(command)) {
        return false;
    }
    rc.phase = 1;
    command.groups = {1, 1, 1};
    if (!compute.dispatch(command)) {
        return false;
    }
    rc.phase = 2;
    command.groups = gasInventoryDispatch(cells);
    if (!compute.dispatch(command)) {
        return false;
    }
    // Fog water: the same conservation over the condensate pool. The kernel
    // normalizes a (mass, energy) pair; the energy pair here is the pool's
    // free scratch slot, read and written in place and then discarded.
    const std::size_t result_slot = params.advection == GridFluid::Advection::MacCormack ? 2 : 1;
    const ComputeBufferHandle free_slot = scratch.fields[6 + (result_slot == 2 ? 1 : 2)];
    ComputeBufferHandle water[6] = {
        scratch.fields[6], outputs[2], free_slot, free_slot,
        scratch.partials, scratch.totals
    };
    command.buffers = water;
    for (uint32_t phase = 0; phase < 3; ++phase) {
        rc.phase = phase;
        command.groups = phase == 1 ? ComputeDispatchSize{1, 1, 1} : gasInventoryDispatch(cells);
        if (!compute.dispatch(command)) {
            return false;
        }
    }
    // No kg/J download and no host normalization. Cache/API consumers request
    // a separate snapshot/reduction through the ownership service.
    std::swap(scratch.fields[0], outputs[0]);
    // outputs held a copy of the result handle; replace its scratch slot with
    // the old input so every allocation still has exactly one owner.
    scratch.fields[result_slot] = outputs[0];
    std::swap(scratch.fields[3], outputs[1]);
    scratch.fields[3 + result_slot] = outputs[1];
    std::swap(scratch.fields[6], outputs[2]);
    scratch.fields[6 + result_slot] = outputs[2];

    // Single-slot pools: rain mass [9], rain energy [10], non-condensable gas
    // [11]. Each advects into a pool scratch slot that is now free in either
    // scheme (1, 4, 7: the old input or the semi-Lagrangian intermediate).
    // A drop moves with the air plus its terminal fall: this is the air part
    // (wind drift; an updraft holds rain up), the fall is the saturation
    // pass. Semi-Lagrangian only - MacCormack overshoots at a rain shaft's
    // sharp edge. ★ Before 2026-10-11 part 4 rain did not move with the wind.
    if (scratch.storage_initialized) {
        constexpr std::size_t kSingles[3][2] = {{9, 1}, {10, 4}, {11, 7}};
        for (const auto& single : kSingles) {
            ComputeBufferHandle bindings[7] = {
                buffers.vel_x, buffers.vel_y, buffers.vel_z,
                scratch.fields[single[0]], scratch.fields[single[1]],
                scratch.fields[single[1]], buffers.gas_solid_mask
            };
            ComputeDispatch advect;
            advect.kernel = "sim_inventory_advect_scalar";
            advect.buffers = bindings;
            advect.buffer_count = 7;
            advect.constants = &pc;
            advect.constants_size = sizeof(pc);
            advect.groups = gasInventoryDispatch(cells);
            if (!compute.dispatch(advect)) {
                return false;
            }
        }
        // Rain mass and its energy normalize as one pair; the non-condensable
        // gas alone, its "energy" pair a free slot read and written in place.
        ComputeBufferHandle rain[6] = {
            scratch.fields[9], scratch.fields[1], scratch.fields[10], scratch.fields[4],
            scratch.partials, scratch.totals
        };
        ComputeBufferHandle foreign[6] = {
            scratch.fields[11], scratch.fields[7], scratch.fields[8], scratch.fields[8],
            scratch.partials, scratch.totals
        };
        ComputeBufferHandle* const pairs[2] = {rain, foreign};
        for (ComputeBufferHandle* pair : pairs) {
            command.buffers = pair;
            for (uint32_t phase = 0; phase < 3; ++phase) {
                rc.phase = phase;
                command.groups = phase == 1 ? ComputeDispatchSize{1, 1, 1}
                                            : gasInventoryDispatch(cells);
                if (!compute.dispatch(command)) {
                    return false;
                }
            }
        }
        std::swap(scratch.fields[9], scratch.fields[1]);
        std::swap(scratch.fields[10], scratch.fields[4]);
        std::swap(scratch.fields[11], scratch.fields[7]);
    }
    markGasInventoryDeviceChanged(state);
    state.gas_stats.inventory_transport = "gpu_resident";
    return true;
}
} // namespace RayTrophiSim::Fluid
