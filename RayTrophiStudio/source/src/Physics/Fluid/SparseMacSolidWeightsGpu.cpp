#include "Fluid/SparseMacSolidWeightsGpu.h"
#include "Fluid/SparseMacSolidWeightSource.h"
#include "Fluid/SparseMacTransferGpu.h"
#include "Fluid/FluidGpuDispatch.h"
#include "FluidGrid.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <limits>

namespace RayTrophiSim::Fluid {

bool uploadSparseMacSolidWeights(SimulationComputeContext& compute,
                                 SimulationGridDomainComputeBuffers& buffers,
                                 const FluidSim::FluidGrid& grid,
                                 std::string& error, uint64_t budget_bytes) {
    auto& storage = buffers.sparse_mac_transfer;
    storage.solid_weights_ready = false;
    error.clear();
    if (!storage.used || !storage.canonical || storage.nx != grid.nx ||
        storage.ny != grid.ny || storage.nz != grid.nz ||
        storage.active_tiles > uint64_t(std::numeric_limits<int32_t>::max()) / 576u) {
        error = "compact solid weights: canonical MAC layout unavailable";
        return false;
    }
    const auto active = static_cast<uint32_t>(storage.active_tiles);
    const auto values = std::size_t(active) * 576u;
    const auto bytes = std::max(values, std::size_t(1)) * sizeof(float);
    uint64_t page_bytes = 0;
    for (const auto handle : storage.owned) {
        page_bytes += compute.getBufferSize(handle);
    }
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!storage.owned[11u + axis].valid() ||
            compute.getBufferSize(storage.owned[11u + axis]) < bytes) {
            error = "compact solid weights: P2G did not allocate variational pages";
            return false;
        }
    }
    // Mixed-frame CPU collider weights are immutable throughout its subcycle.
    // A standalone caller has no such lease and refreshes on every invocation.
    if (!buffers.matter_model.enabled || !storage.solid_weight_source_valid) {
        const auto limit = compute.caps().max_storage_buffer_bytes;
        uint64_t source_limit = limit;
        if (budget_bytes != 0) {
            if (page_bytes >= budget_bytes) {
                error = "compact solid weights: no metadata budget remains";
                return false;
            }
            const auto remaining = budget_bytes - page_bytes;
            source_limit = limit == 0 ? remaining : std::min<uint64_t>(limit, remaining);
        }
        std::vector<uint32_t> source;
        if (!packSparseMacSolidWeightSource({grid.nx, grid.ny, grid.nz},
                {&grid.u_weight, &grid.v_weight, &grid.w_weight}, source, error, source_limit)) {
            return false;
        }
        const auto required = source.size() * sizeof(uint32_t);
        if ((limit != 0 && required > limit) ||
            (budget_bytes != 0 && (page_bytes > budget_bytes ||
                required > budget_bytes - page_bytes))) {
            error = "compact solid weights: metadata exceeds buffer/working budget";
            return false;
        }
        auto handle = storage.solid_weight_source;
        const auto previous = handle;
        const auto capacity = compute.getBufferSize(handle);
        const bool trim = budget_bytes != 0 && capacity > budget_bytes - page_bytes;
        if (!handle.valid() || capacity < required || trim) {
            handle = compute.createBuffer({"SparseSolidWeightSource", required,
                ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                    ComputeBufferUsage::ReadOnly});
            if (!handle.valid()) {
                error = "compact solid weights: metadata allocation failed";
                return false;
            }
        }
        compute.beginTransferBatch();
        bool uploaded = compute.uploadBuffer(handle, source.data(), required);
        uploaded = compute.endTransferBatch() && uploaded;
        if (!uploaded) {
            if (handle.id != previous.id) {
                compute.destroyBuffer(handle);
            }
            error = "compact solid weights: metadata upload failed";
            return false;
        }
        if (handle.id != previous.id && previous.valid()) {
            compute.destroyBuffer(previous);
        }
        storage.solid_weight_source = handle;
        storage.solid_weight_source_valid = true;
    }
    struct Constants { int nx, ny, nz; uint32_t active; } constants{
        grid.nx, grid.ny, grid.nz, active};
    static_assert(sizeof(Constants) == 16);
    const ComputeBufferHandle bindings[] = {storage.solid_weight_source, storage.owned[1],
        storage.owned[11], storage.owned[12], storage.owned[13]};
    ComputeDispatch command;
    command.kernel = "sim_sparse_mac_solid_weights";
    command.buffers = bindings;
    command.buffer_count = 5;
    command.constants = &constants;
    command.constants_size = sizeof(constants);
    command.groups = FluidGpuDispatch::groups256(static_cast<uint32_t>(values));
    if (!compute.dispatch(command)) {
        error = "compact solid weights: GPU slot remap failed";
        return false;
    }
    storage.resident_bytes = page_bytes + compute.getBufferSize(storage.solid_weight_source);
    storage.solid_weights_ready = true;
    return true;
}
} // namespace RayTrophiSim::Fluid
