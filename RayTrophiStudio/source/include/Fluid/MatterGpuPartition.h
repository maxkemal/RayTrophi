#pragma once

#include "FluidParticles.h"
#include "../SimulationCompute.h"
#include <array>
#include <string>
#include <cstdint>
#include <vector>

namespace RayTrophiSim::Fluid {

// Lists borrow canonical flat particle indices. Partition never copies/reorders
// position, identity, substance or granular state. Counters stay on device.
struct MatterGpuPartition {
    ComputeBufferHandle models;
    std::array<ComputeBufferHandle, 2> indices;
    ComputeBufferHandle counters;
    std::size_t capacity = 0;

    bool valid() const;
};

void destroyMatterGpuPartition(SimulationComputeContext& compute, MatterGpuPartition& buffers);
bool ensureMatterGpuPartition(SimulationComputeContext& compute, MatterGpuPartition& buffers,
    std::size_t count, std::size_t budget_bytes, std::string& error);
constexpr uint32_t kMatterGpuHeldModel = 0xFFu;
// `held` (optional, one byte per parcel): parcels in NO lane -- frozen
// carriers / static blocks stay pinned where they are (the grain domain's
// "held ice" rule) instead of refusing the whole mixed step.
bool dispatchMatterGpuPartition(SimulationComputeContext& compute,
    MatterGpuPartition& buffers, const FluidParticles& particles,
    MatterConstitutiveModel legacy_model, std::string& error,
    const std::vector<uint8_t>* held = nullptr);

} // namespace RayTrophiSim::Fluid
