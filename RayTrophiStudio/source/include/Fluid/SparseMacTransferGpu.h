#pragma once

#include "SimulationCompute.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace RayTrophiSim {
struct SimulationGridDomainComputeBuffers;
namespace Fluid {
struct APICSolverStats;
struct APICSolverParams;

// Sparse transfer storage. With `compact_owner` the pages are the canonical
// device MAC velocity of the liquid lane and no dense velocity/FLIP bank exists
// (docs/dev/MATTER_SPARSE_S1_SIVI_GPU.md); otherwise projection/contact consume
// a dense publication and the pages are additional memory.
struct SparseMacTransferGpuStorage {
    // map, list/count, velocity XYZ, weight XYZ, FLIP baseline XYZ.
    std::array<ComputeBufferHandle, 11> owned{};
    uint64_t active_tiles = 0;
    uint64_t allocated_tiles = 0;
    uint64_t resident_bytes = 0;
    int nx = 0;
    int ny = 0;
    int nz = 0;
    bool used = false;
    bool snapshot_valid = false;
    bool flip_gather_used = false;
    // Persistent: this domain's liquid lane runs on compact pages, so the
    // buffer allocator skips the dense velocity and FLIP banks. Cleared by any
    // path that needs them (pure-liquid step, compact failure, sparse off).
    bool compact_owner = false;
    // A compact P2G failed with the bank released: the lane stays dense until
    // sparse mode is switched off and on again. Survives a pool release, so a
    // persistent failure does not reallocate the dense bank every frame.
    bool compact_blocked = false;
    // This substep's P2G left the canonical field on the pages (no publication).
    bool canonical = false;
    std::string status;
};

// What a MAC velocity consumer binds this substep: the dense bank, or the
// compact pages plus the tile map and list it must add after its own buffers.
struct MacVelocityBinding {
    std::array<ComputeBufferHandle, 3> velocity{};
    std::array<ComputeBufferHandle, 3> weight{};
    std::array<ComputeBufferHandle, 3> flip{};
    ComputeBufferHandle map;
    ComputeBufferHandle list;
    bool compact = false;
    // Compact face lanes (active tiles x 576); a face kernel dispatches these.
    uint32_t face_lanes = 0;
};

MacVelocityBinding macVelocityBinding(const SimulationGridDomainComputeBuffers& buffers);

// Allocator hook: true for a compact owner, after releasing any dense velocity
// and FLIP bank left from an earlier dense step. The caller then skips them.
bool releaseDenseMacForCompactOwner(SimulationComputeContext& compute,
                                    SimulationGridDomainComputeBuffers& buffers);

// The dense kernel name, or its compact twin when the binding is compact.
inline const char* macKernel(const MacVelocityBinding& binding, const char* dense,
                             const char* compact) {
    return binding.compact ? compact : dense;
}

// Host MAC arrays from the compact pages: one map + page download, then a
// scatter into the dense host layout (the host grid stays dense until S2).
// Faces without a page are 0, which is what the dense device field held.
bool publishCompactMacToHost(SimulationComputeContext& compute,
                             const SimulationGridDomainComputeBuffers& buffers,
                             const std::array<std::vector<float>*, 3>& host,
                             std::string& error);

struct SparseMacTransferConstants {
    int32_t nx = 0;
    int32_t ny = 0;
    int32_t nz = 0;
    int32_t particle_count = 0;
    int32_t component = 0;
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    float origin_z = 0.0f;
    float voxel_size = 1.0f;
};
static_assert(sizeof(SparseMacTransferConstants) == 36, "Sparse MAC transfer ABI");

// `canonical`: leave the field on the pages (no dense publication; the dense
// velocity/weight banks need not exist). Otherwise publish as before.
bool runSparseMacP2G(SimulationComputeContext& compute,
                     SimulationGridDomainComputeBuffers& buffers,
                     const APICSolverParams& params,
                     SparseMacTransferConstants constants,
                     bool canonical,
                     std::string& error);

// Capture the actual caller-selected pre-projection field, including its
// boundary policy. Never replace it with a post-viscosity/projection field.
bool captureSparseMacFlip(SimulationComputeContext& compute,
                          SimulationGridDomainComputeBuffers& buffers);
bool captureSparseMacPost(SimulationComputeContext& compute,
                          SimulationGridDomainComputeBuffers& buffers);
void publishSparseMacTransferStats(const SparseMacTransferGpuStorage& storage,
                                   APICSolverStats& stats);
void releaseSparseMacTransfer(SimulationComputeContext& compute,
                              SparseMacTransferGpuStorage& storage);

} // namespace Fluid
} // namespace RayTrophiSim
