#include "Fluid/MatterAudioTelemetry.h"
#include "Fluid/GasInventoryGpu.h"
#include "Fluid/MatterPhaseGrid.h"
#include "ParticleSimulation.h"
#include <cmath>
#include <limits>

namespace RayTrophiSim::Fluid {
bool measureGasAudioGpu(SimulationGridDomainState& state, SimulationComputeContext& compute,
                       SimulationGridDomainComputeBuffers& buffers, double duration,
                       DomainAudioMeasurement& output) {
    const auto inventory = gasInventoryDevice(state);
    using Field = SimulationGridDomainComputeBuffers::GridField;
    const auto& grid = gasGrid(state);
    if (!inventory || !inventory->populated || inventory->revision != state.gas_inventory_revision ||
        !buffers.deviceHasCurrent(Field::VelX) || !buffers.deviceHasCurrent(Field::VelY) ||
        !buffers.deviceHasCurrent(Field::VelZ) || grid.voxel_size <= 0 || duration <= 0 ||
        inventory->cells > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const uint32_t blocks = static_cast<uint32_t>((inventory->cells + 255) / 256);
    if (blocks == 0) {
        return false;
    }
    struct Scratch {
        SimulationComputeContext& compute;
        ComputeBufferHandle partials, total;
        ~Scratch() {
            if (partials.valid()) {
                compute.destroyBuffer(partials);
            }
            if (total.valid()) {
                compute.destroyBuffer(total);
            }
        }
    } scratch{compute};
    ComputeBufferDesc desc;
    desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::ReadWrite |
                 ComputeBufferUsage::Download;
    desc.debug_name = "MatterAudioGasPartials";
    desc.size_bytes = static_cast<std::size_t>(blocks) * 8 * sizeof(double);
    scratch.partials = compute.createBuffer(desc);
    desc.debug_name = "MatterAudioGasTotal";
    desc.size_bytes = 8 * sizeof(double);
    scratch.total = compute.createBuffer(desc);
    if (!scratch.partials.valid() || !scratch.total.valid()) {
        return false;
    }
    struct Constants {
        uint32_t cells, blocks, phase, nx, ny, nz;
        float voxel, origin_x, origin_y, origin_z;
    } pc{static_cast<uint32_t>(inventory->cells), blocks, 0,
         static_cast<uint32_t>(grid.nx), static_cast<uint32_t>(grid.ny),
         static_cast<uint32_t>(grid.nz), grid.voxel_size, grid.origin.x, grid.origin.y, grid.origin.z};
    static_assert(sizeof(Constants) == 40);
    ComputeBufferHandle bindings[] = {inventory->fields[0], inventory->fields[11],
        buffers.vel_x, buffers.vel_y, buffers.vel_z, scratch.partials, scratch.total};
    ComputeDispatch dispatch;
    dispatch.kernel = "sim_audio_gas_flow";
    dispatch.buffers = bindings;
    dispatch.buffer_count = 7;
    dispatch.constants = &pc;
    dispatch.constants_size = sizeof(pc);
    dispatch.groups = gasInventoryDispatch(inventory->cells);
    if (!compute.dispatch(dispatch)) {
        return false;
    }
    pc.phase = 1;
    dispatch.groups = {1, 1, 1};
    if (!compute.dispatch(dispatch)) {
        return false;
    }
    compute.synchronize();
    double result[8]{};
    if (!compute.downloadBuffer(scratch.total, result, sizeof(result)) || result[5] != 0 ||
        result[4] <= 0 || !std::isfinite(result[0]) || result[0] < 0 || result[0] > 1e9) {
        return false;
    }
    output.measured = true;
    output.value = result[0];
    output.energy_j = 0;
    output.duration_seconds = duration;
    output.position_weight = result[0];
    output.carriers = static_cast<std::uint64_t>(result[4]);
    output.reason = "available";
    if (result[0] > 0) {
        output.position = Vec3(static_cast<float>(result[1] / result[0]),
            static_cast<float>(result[2] / result[0]), static_cast<float>(result[3] / result[0]));
        if (!std::isfinite(output.position.x) || !std::isfinite(output.position.y) ||
            !std::isfinite(output.position.z)) {
            output = {};
            return false;
        }
    }
    return true;
}
} // namespace RayTrophiSim::Fluid
