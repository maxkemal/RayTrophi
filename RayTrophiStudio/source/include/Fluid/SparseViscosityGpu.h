#pragma once

#include "SimulationCompute.h"
#include <array>
#include <string>

namespace RayTrophiSim {
struct SimulationGridDomainComputeBuffers;
namespace Fluid {
struct APICSolverParams;

struct ViscosityGpuConstants {
    int nx = 0, ny = 0, nz = 0, boundary = 1;
    float voxel_size = 1.0f, dt = 0.0f, alpha = 0.0f, solid_weight = 0.0f;
    int parity = 0, has_nu = 0, pad1 = 0, pad2 = 0, pad3 = 0;
};
static_assert(sizeof(ViscosityGpuConstants) == 52, "Viscosity base push ABI");

struct SparseViscosityGpuStorage {
    std::array<ComputeBufferHandle, 5> owned{};
    uint64_t active_tiles = 0;
    uint64_t allocated_tiles = 0;
    uint64_t resident_bytes = 0;
    bool used = false;
};

bool ensureMacRhsScratch(SimulationComputeContext& compute,
                         SimulationGridDomainComputeBuffers& buffers,
                         const std::array<std::size_t, 3>& faces, bool compact);

bool runSparseViscosity(SimulationComputeContext& compute,
                        SimulationGridDomainComputeBuffers& buffers,
                        const APICSolverParams& params,
                        const ViscosityGpuConstants& viscosity_constants, int sweeps,
                        std::string& error);

void releaseSparseViscosity(SimulationComputeContext& compute,
                            SparseViscosityGpuStorage& storage);

} // namespace Fluid
} // namespace RayTrophiSim
