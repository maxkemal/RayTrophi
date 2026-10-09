#include "Fluid/SparseMacSolidWeightsGpu.h"
#include "Fluid/SparseMacTransferGpu.h"
#include "FluidGrid.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <limits>

namespace RayTrophiSim::Fluid {

bool uploadSparseMacSolidWeights(SimulationComputeContext& compute,
                                 SimulationGridDomainComputeBuffers& buffers,
                                 const FluidSim::FluidGrid& grid,
                                 std::string& error) {
    auto& storage = buffers.sparse_mac_transfer;
    storage.solid_weights_ready = false;
    error.clear();
    if (!storage.used || !storage.canonical || storage.nx != grid.nx ||
        storage.ny != grid.ny || storage.nz != grid.nz ||
        storage.active_tiles > uint64_t(std::numeric_limits<int32_t>::max()) / 576u) {
        error = "compact solid weights: canonical MAC layout unavailable";
        return false;
    }
    const auto active = static_cast<std::size_t>(storage.active_tiles);
    const std::size_t values = active * 576u;
    const std::size_t bytes = std::max(values, std::size_t(1)) * sizeof(float);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const auto handle = storage.owned[11u + axis];
        if (!handle.valid() || compute.getBufferSize(handle) < bytes) {
            error = "compact solid weights: P2G did not allocate variational pages";
            return false;
        }
    }
    std::vector<uint32_t> list(active + 1u);
    compute.beginTransferBatch();
    bool ok = compute.downloadBuffer(storage.owned[1], list.data(),
                                     list.size() * sizeof(uint32_t));
    ok = compute.endTransferBatch() && ok;
    if (!ok || list[0] != active) {
        error = "compact solid weights: MAC tile list readback failed or changed";
        return false;
    }
    const std::vector<uint32_t> keys(list.begin() + 1, list.end());
    std::array<std::vector<float>, 3> pages;
    if (!packSparseMacSolidWeights({grid.nx, grid.ny, grid.nz}, keys,
                                  {&grid.u_weight, &grid.v_weight, &grid.w_weight},
                                  pages, error)) {
        return false;
    }
    // Empty topologies still bind one initialized float to every descriptor.
    const float open = 1.0f;
    compute.beginTransferBatch();
    ok = true;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const void* data = values == 0 ? static_cast<const void*>(&open)
                                      : static_cast<const void*>(pages[axis].data());
        ok = compute.uploadBuffer(storage.owned[11u + axis], data, bytes) && ok;
    }
    ok = compute.endTransferBatch() && ok;
    if (!ok) {
        error = "compact solid weights: page upload failed";
        return false;
    }
    storage.solid_weights_ready = true;
    return true;
}

} // namespace RayTrophiSim::Fluid
