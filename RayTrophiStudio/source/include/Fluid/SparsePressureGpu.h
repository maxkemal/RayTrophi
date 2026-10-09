#pragma once

#include "SimulationCompute.h"

#include <array>
#include <cstdint>
#include <string>

namespace RayTrophiSim {
struct SimulationGridDomainComputeBuffers;
namespace Fluid {
struct APICSolverParams;
struct APICSolverStats;

struct SparsePressureGpuStorage {
    // map/list + six compact float fields + double partials/scalars.
    std::array<ComputeBufferHandle, 10> owned{};
    uint64_t allocated_tiles = 0;
    uint64_t active_tiles = 0;
    uint64_t resident_bytes = 0;
    uint64_t download_bytes = 0;
    uint64_t dispatches = 0;
    bool used = false;
};

bool usesSparsePressure(const SimulationComputeContext& compute, bool sparse,
                        const APICSolverParams& params);

bool solveSparsePressure(SimulationComputeContext& compute,
                         SimulationGridDomainComputeBuffers& buffers,
                         const APICSolverParams& params, int nx, int ny, int nz,
                         float voxel, float dt, bool variational,
                         APICSolverStats* stats, std::string& error);

void releaseSparsePressure(SimulationComputeContext& compute,
                           SparsePressureGpuStorage& storage);

bool ensurePressureScratch(SimulationComputeContext& compute,
                           SimulationGridDomainComputeBuffers& buffers,
                           std::size_t cells, bool compact);

} // namespace Fluid
} // namespace RayTrophiSim
