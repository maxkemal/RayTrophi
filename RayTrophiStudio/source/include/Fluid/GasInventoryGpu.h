#pragma once

#include "SimulationCompute.h"
#include "Fluid/GasPhaseMomentum.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "Vec3.h"

struct SceneData;

namespace RayTrophiSim {
struct SimulationGridDomainState;
struct SimulationGridDomainComputeBuffers;
namespace GridFluid {
struct SolverParams;
}
namespace Fluid {

inline ComputeDispatchSize gasInventoryDispatch(std::size_t items, uint32_t threads = 256) {
    const uint32_t blocks = static_cast<uint32_t>((items + threads - 1) / threads);
    const uint32_t x = std::max(1u, std::min(blocks, 65535u));
    return {x, std::max(1u, (blocks + x - 1) / x), 1u};
}

struct GasInventorySummary {
    double mass_kg = 0.0;            // vapour
    double energy_j = 0.0;           // all water (vapour + condensate)
    uint64_t active_cells = 0;       // cells holding vapour
    Vec3 centroid;
    double condensate_kg = 0.0;      // fog water
    uint64_t condensate_cells = 0;   // cells holding fog water
    double rain_kg = 0.0;            // rain water (falling or pooled)
    uint64_t rain_cells = 0;
};

// The compute-buffer bank owns the device allocation. Live states only hold a
// weak view; cache snapshots detach it after their explicit materialization.
struct GasInventoryDevice {
    SimulationComputeContext* compute = nullptr;
    // Three advected pools x three slots (current, advection scratch,
    // MacCormack scratch): vapour mass [0..2], energy [3..5], condensate
    // [6..8]. Rain water is not carried by the wind (it falls through it), so
    // it has one slot each: rain mass [9], rain energy [10].
    std::array<ComputeBufferHandle, 11> fields{};
    ComputeBufferHandle partials;
    ComputeBufferHandle totals;
    ComputeBufferHandle zero;
    ComputeBufferHandle deposits;
    ComputeBufferHandle condensate_deposits;   // own buffer: uploads are not stream-ordered
    ComputeBufferHandle phase_momentum;
    std::map<uint32_t, GasPhaseMomentumDeposit> pending_momentum;
    ComputeBufferHandle summary_partials;
    ComputeBufferHandle summary_totals;
    std::array<ComputeBufferHandle, 4> surface{};
    std::size_t cells = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    float voxel = 0.0f;
    uint64_t revision = 0;
    uint64_t grid_upload_bytes = 0;
    uint64_t grid_download_bytes = 0;
    uint64_t snapshot_download_bytes = 0;
    uint64_t query_download_bytes = 0;
    bool populated = false;
    bool storage_initialized = false;
    bool summary_valid = false;
    GasInventorySummary summary;
    Vec3 summary_origin;
    float summary_voxel = 0.0f;
    std::map<uint32_t, std::pair<float, float>> pending;
    // Liquid water entering the air (mist parcels): condensate kg, energy J.
    std::map<uint32_t, std::pair<float, float>> pending_condensate;
    // Saturation adjustment scratch: rain-out records and their counter.
    ComputeBufferHandle saturation_records;
    ComputeBufferHandle saturation_counter;
    // Rain landing stamps, one uint per cell (the device side of
    // SimulationGridDomainState::gas_rain_landing). Zeroed at creation.
    ComputeBufferHandle rain_landing;
};

struct GasInventoryGpuBuffers {
    std::shared_ptr<GasInventoryDevice> owner;
};

void releaseGasInventoryGpu(SimulationComputeContext& compute, GasInventoryGpuBuffers& buffers);

bool beginGasInventoryDevice(SimulationGridDomainState& state,
                             SimulationComputeContext& compute,
                             SimulationGridDomainComputeBuffers& buffers);
bool ensureGasInventoryStorage(SimulationGridDomainState& state);
std::array<ComputeBufferHandle, 2> gasInventoryReadBuffers(SimulationGridDomainState& state);
void prepareGasInventoryHostProducer(SimulationGridDomainState& state);
void depositGasInventory(SimulationGridDomainState& state, std::size_t cell,
                         float mass_kg, float energy_j);
// Liquid water carried into the air as fog water (no phase change): kg into
// the condensate pool, its sensible energy into the shared energy pool.
void depositGasCondensate(SimulationGridDomainState& state, std::size_t cell,
                          float mass_kg, float energy_j);
// Current condensate device buffer (the 1-float zero buffer before storage).
ComputeBufferHandle gasInventoryCondensateBuffer(SimulationGridDomainState& state);
// Rain mass/energy device buffers (the zero buffer before storage).
std::array<ComputeBufferHandle, 2> gasInventoryRainBuffers(SimulationGridDomainState& state);
// The rain landing stamp buffer, created and zeroed on first use.
ComputeBufferHandle gasInventoryRainLanding(SimulationGridDomainState& state);
// Free device scratch for a per-cell reduction: the two non-current slots of
// the vapour pool and one of the energy pool. Valid only between advections;
// null before storage.
std::array<ComputeBufferHandle, 3> gasInventoryScratchBuffers(SimulationGridDomainState& state);
std::shared_ptr<GasInventoryDevice> gasInventoryDevice(const SimulationGridDomainState& state);
void markGasInventoryDeviceChanged(SimulationGridDomainState& state);
bool flushGasInventoryDeposits(SimulationGridDomainState& state);
bool gasInventoryHasContent(const SimulationGridDomainState& state);
void holdGasInventory(SimulationGridDomainState& state, const char* reason);

// Explicit consumers only. Physics does not call either operation on the
// healthy Vulkan path. Snapshots materialize into a COPY and detach its view.
bool snapshotGasInventory(SimulationGridDomainState& snapshot);
bool materializeGasInventoryForBackendChange(SimulationGridDomainState& state);
void prepareGasInventoryRestore(std::vector<SimulationGridDomainState>& states);
void releaseGasInventoryDomainsForBackendChange(
    std::vector<SimulationGridDomainState>& states,
    std::vector<SimulationGridDomainComputeBuffers>& buffers,
    SimulationComputeContext& compute);
GasInventorySummary readGasInventorySummary(const SimulationGridDomainState& state);
std::string gasInventoryStepError(const ::SceneData& scene);

struct GasInventoryHostImage {
    std::vector<float> mass;
    std::vector<float> energy;
    std::vector<float> condensate;
    std::vector<float> rain;
    std::vector<float> rain_energy;
};
GasInventoryHostImage captureGasInventoryHostImage(const SimulationGridDomainState& state);
bool captureGasInventoryHostImages(const std::vector<SimulationGridDomainState>& states,
                                   std::vector<GasInventoryHostImage>& images);

// Both fields commit by device handle swap, with no inventory readback.
// Failure leaves the original authoritative device fields in place.
bool advectGasInventoryGpu(SimulationGridDomainState& state,
                           const GridFluid::SolverParams& params,
                           float dt,
                           SimulationComputeContext& compute,
                           SimulationGridDomainComputeBuffers& buffers,
                           bool carrier_valid = true);

} // namespace Fluid
} // namespace RayTrophiSim
