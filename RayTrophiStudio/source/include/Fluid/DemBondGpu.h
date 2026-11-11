#pragma once

#include "DemBondNetwork.h"
#include "../SimulationCompute.h"
#include <array>

namespace RayTrophiSim::Fluid {
struct MatterGrainGpuRuntime;
struct MatterGrainParams;

// One speculative frame attempt. Canonical state changes only in publish(),
// after all existing DEM validity/retry checks. Device reset/retry starts again
// from the canonical graph. Buffers outlive every queued download pointer.
class DemBondGpuFrame {
public:
    DemBondGpuFrame(SimulationComputeContext& compute, MatterGrainGpuRuntime& runtime);
    ~DemBondGpuFrame();
    DemBondGpuFrame(const DemBondGpuFrame&) = delete;
    DemBondGpuFrame& operator=(const DemBondGpuFrame&) = delete;
    bool prepare(const FluidParticles& particles, const MatterGrainParams& params,
                 std::string& error);
    bool before(MatterGrainGpuRuntime& runtime, uint32_t substep, float dt,
                std::string& error);
    bool after(MatterGrainGpuRuntime& runtime, uint32_t substep, float dt,
               std::string& error);
    bool readback(std::string& error);
    bool publish(FluidParticles& particles, std::string& error);
    double maximum_dt = 1.0e30;
    uint64_t working_bytes = 0;
    uint64_t upload_bytes = 0;
    uint64_t download_bytes = 0;
    bool active = false;
    bool constrains_motion = false;
    uint32_t extraDispatchesPerStep() const {
        return active ? (constrains_motion ? 3u : 1u) : 0u;
    }
private:
    struct alignas(16) EdgeRow {
        std::array<uint32_t, 4> meta{};
        std::array<float, 4> rest{};
        std::array<float, 4> qa{};
        std::array<float, 4> qb{};
        std::array<float, 4> material{};
        std::array<float, 4> history{};
    };
    bool dispatch(const char* kernel, MatterGrainGpuRuntime& runtime,
                  uint32_t substep, float dt, uint32_t threads, std::string& error);
    SimulationComputeContext& compute;
    std::array<ComputeBufferHandle, 5>& owned;
    std::vector<EdgeRow> edges;
    std::vector<std::size_t> edge_indices;
    std::size_t canonical_edge_count = 0;
    std::vector<std::array<float, 4>> poses;
    std::vector<uint32_t> adjacency;
    std::vector<uint64_t> identities;
    uint32_t status = 0;
    bool completed = false;
    bool uploaded = false;
};
} // namespace RayTrophiSim::Fluid
