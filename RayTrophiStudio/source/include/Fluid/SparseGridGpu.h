#pragma once

#include "Fluid/SparseGridStorage.h"
#include "SimulationCompute.h"

#include <string>

namespace RayTrophiSim::Fluid::Sparse {

struct DeviceGrid {
    std::shared_ptr<const Topology> topology;
    ChannelDescriptions descriptions{};
    ComputeBufferHandle map;
    ComputeBufferHandle keys;
    std::array<ComputeBufferHandle, channelCount> fields{};
    uint64_t resident_bytes = 0;
    uint64_t generation = 0;
};

struct DeviceGridConstants {
    int32_t nx = 0;
    int32_t ny = 0;
    int32_t nz = 0;
    int32_t face_axis = -1;
    uint32_t tiles_x = 0;
    uint32_t tiles_y = 0;
    uint32_t tiles_z = 0;
    uint32_t old_active = 0;
    uint32_t new_active = 0;
    uint32_t page_values = 512;
    float background = 0.0f;
    uint32_t padding = 0;
};
static_assert(sizeof(DeviceGridConstants) == 48, "Sparse grid transaction push ABI");

// Bootstrap from explicit host pages. Device fields remain authoritative after
// this operation; topology changes then copy on the GPU, not through host grids.
bool uploadGrid(SimulationComputeContext& compute, const GridStorage& host,
                std::shared_ptr<const Topology> topology, DeviceGrid& destination,
                uint64_t working_budget_bytes, std::string& error);

// Includes old + candidate GPU capacity in the authored working budget. Rejects
// retirement if ANY live physical channel would be discarded. Publication is a
// single move after the GPU validation flag has been read; no partial remap.
bool rebindGrid(SimulationComputeContext& compute, DeviceGrid& grid,
                std::shared_ptr<const Topology> next, uint64_t working_budget_bytes,
                std::string& error);

bool downloadGrid(SimulationComputeContext& compute, const DeviceGrid& device,
                  GridStorage& host, std::string& error);
void releaseGrid(SimulationComputeContext& compute, DeviceGrid& grid);

} // namespace RayTrophiSim::Fluid::Sparse
