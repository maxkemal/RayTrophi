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

// Implicit viscosity solved to a tolerance (sim_fluid_viscosity_pcg.glsl):
// Jacobi-preconditioned CG with device scalars, a convergence readback every
// few iterations. Replaced fixed red-black GS sweeps on the GPU, which applied
// 1/9..1/25 of honey's viscosity at h 2 mm (docs/dev/VISKOZ_SIVI_URETIM_KALITESI.md).
constexpr float kViscosityPcgTolerance = 1.0e-4f;  // sqrt(r.z / r0.z0)
constexpr int kViscosityPcgMaxIterations = 500;

struct ViscosityPcgStorage {
    // r, p, q, d (3 * lanes floats), scalars (8 doubles), partials (doubles)
    std::array<ComputeBufferHandle, 6> owned{};
};

struct ViscosityPcgResult {
    int iterations = 0;
    float relative_residual = 0.0f;
    bool converged = false;
};

// base: the 11 bindings the RBGS kernel takes (velocity x3, mask, rhs x3,
// solid velocity x3, nu). tiles / mac: the sparse viscosity tile map+list and
// the compact MAC map+list, nullptr when the variant has none.
bool solveViscosityPcg(SimulationComputeContext& compute, ViscosityPcgStorage& storage,
                       const char* kernel, const ComputeBufferHandle* base,
                       const ComputeBufferHandle* tiles, const ComputeBufferHandle* mac,
                       const ViscosityGpuConstants& constants,
                       const std::array<uint32_t, 3>& tile_dims, uint32_t compact_faces,
                       uint32_t lanes_per_component, float particles_per_cell,
                       ViscosityPcgResult& result, std::string& error);

void releaseViscosityPcg(SimulationComputeContext& compute, ViscosityPcgStorage& storage);

bool ensureMacRhsScratch(SimulationComputeContext& compute,
                         SimulationGridDomainComputeBuffers& buffers,
                         const std::array<std::size_t, 3>& faces, bool compact);

bool runSparseViscosity(SimulationComputeContext& compute,
                        SimulationGridDomainComputeBuffers& buffers,
                        const APICSolverParams& params,
                        const ViscosityGpuConstants& viscosity_constants,
                        ViscosityPcgStorage& pcg, ViscosityPcgResult& result,
                        std::string& error);

void releaseSparseViscosity(SimulationComputeContext& compute,
                            SparseViscosityGpuStorage& storage);

} // namespace Fluid
} // namespace RayTrophiSim
