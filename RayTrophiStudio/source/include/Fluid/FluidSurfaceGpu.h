#pragma once

// GPU surface reconstruction: the liquid/snow level set, its render density
// proxy and the material-coordinate residual, built on the simulation Vulkan
// compute device (docs/dev/GPU_YUZEY_LEVEL_SET.md, Y1).
//
// Device twin of buildLevelSet + the proxy loop + buildMaterialCoordinateGrid.
// The CPU builders stay the reference (and the route for what this one refuses);
// fluid.surface_parity compares the two on the same particles.
//
// ★ Inputs are UPLOADED from the host particles (~28 bytes per parcel), not
// taken from the step's device buffers: the host edits particles after the step
// (open-boundary removal, pore exchange, labels, mist drag), so the device copy
// is not the state being drawn.

#include "Fluid/FluidLevelSet.h"
#include "SimulationCompute.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace RayTrophiSim::Fluid {

// Persistent per-domain device storage. Grown geometrically, never shrunk.
struct FluidSurfaceGpuBuffers {
    ComputeBufferHandle positions, mass, disp, support_radii;
    ComputeBufferHandle bin1, bin2, counts, cursor, offsets1, offsets2, items1, items2;
    ComputeBufferHandle centres, sdf_a, sdf_b, sdf_c, confidence, density, stats;
    ComputeBufferHandle uvw_a, uvw_b, valid_a, valid_b;
    ComputeBufferHandle slots, slot_table, composition;
    std::size_t particle_capacity = 0;
    std::size_t bin_capacity = 0;
    std::size_t cell_capacity = 0;
    std::size_t uvw_cell_capacity = 0;

    void release(SimulationComputeContext& compute);
};

// What the scene layer keeps per domain between frames: the storage, and the
// last live publication (route 2), re-published every frame until a rebuild
// replaces it — the render-volume sync clears live fields it is not told about.
struct FluidSurfaceGpuDomain {
    FluidSurfaceGpuBuffers buffers;
    bool live = false;          // the Vulkan RT backend reads the device buffers
    bool host_stale = false;    // host proxy / NanoVDB grid older than the device
    uint64_t density_address = 0;
    uint64_t uvw_address = 0;
    uint64_t composition_address = 0;  // 0 = no mixture (single material)
    int dims[3] = {0, 0, 0};
    float origin[3] = {0, 0, 0};
    float voxel = 0.0f;
    bool occupied = false;
    int cell_min[3] = {0, 0, 0};
    int cell_max[3] = {-1, -1, -1};
    uint64_t version = 0;       // bumped per live build (light-cache stamp)
};

struct FluidSurfaceGpuRequest {
    const FluidParticles* particles = nullptr;
    const FluidSim::FluidGrid* grid = nullptr;     // the sim grid (render_grid)
    LevelSetParams params;
    float surface_band_voxels = 1.0f;              // domain fluid_surface_band_voxels
    const FluidViewSelection* selection = nullptr; // same selection the CPU gets
    bool want_uvw = true;
    // Per-substance materials (buildCompositionGrid's inputs). With none, or
    // when every substance resolves to the fallback, no composition is built,
    // exactly as on the CPU.
    const SubstanceMaterialEntry* materials = nullptr;
    std::size_t material_count = 0;
    int fallback_material = -1;
    // Download the proxy (and uvw) into host vectors: every consumer that is not
    // the Vulkan RT backend owning this device (raster viewport, OptiX, Embree).
    bool download_host = false;
};

struct FluidSurfaceGpuResult {
    bool ok = false;
    std::string status;            // why not, when !ok
    // Proxy grid (refined by the surface multiplier).
    int nx = 0, ny = 0, nz = 0;
    float voxel = 0.0f;
    // Occupied box in grid-local units (VDBVolume::occupiedLocalBox rules).
    bool occupied = false;
    float occupied_min[3] = {0, 0, 0};
    float occupied_max[3] = {0, 0, 0};
    // The same occupancy as inclusive cell indices, unrounded (the live dense
    // publication pads these itself).
    int occupied_cell_min[3] = {0, 0, 0};
    int occupied_cell_max[3] = {-1, -1, -1};
    std::size_t active_cells = 0;
    std::size_t surface_cells = 0;
    std::size_t eligible_particles = 0;
    // Device results, valid until the next build on these buffers.
    ComputeBufferHandle density;
    ComputeBufferHandle uvw;       // sim grid, xyz residual; invalid when !uvw_built
    bool uvw_built = false;
    ComputeBufferHandle composition; // sim grid, (slot A, slot B, weight B)
    bool composition_built = false;
    int uvw_nx = 0, uvw_ny = 0, uvw_nz = 0;
    double build_ms = 0.0;         // host wall time including the device wait
    std::size_t upload_bytes = 0;
    std::size_t download_bytes = 0;
};

// The configurations the GPU route covers; `why` names the first refusal.
// Anisotropic kernels are CPU-only for now (eigensolve per parcel not ported).
bool fluidSurfaceGpuSupports(const LevelSetParams& params, std::string& why);

// Builds the surface on `compute`. On success `density_host` / `uvw_host` are
// filled when request.download_host is set (sized like the CPU buffers:
// proxy cells, 3 * sim cells). `stats` mirrors LevelSetStats for the panel/IPC.
bool buildFluidSurfaceGpu(SimulationComputeContext& compute,
                          FluidSurfaceGpuBuffers& buffers,
                          const FluidSurfaceGpuRequest& request,
                          FluidSurfaceGpuResult& result,
                          std::vector<float>* density_host,
                          std::vector<float>* uvw_host,
                          std::vector<float>* composition_host,
                          LevelSetStats* stats);

// SDF only (no proxy transform), downloaded, for fluid.surface_parity: the same
// field buildLevelSet writes, so the two can be compared cell by cell.
bool buildFluidSurfaceGpuSdf(SimulationComputeContext& compute,
                             FluidSurfaceGpuBuffers& buffers,
                             const FluidSurfaceGpuRequest& request,
                             std::vector<float>& sdf_out,
                             std::vector<float>* uvw_out,
                             std::vector<float>* composition_out,
                             FluidSurfaceGpuResult& result);

} // namespace RayTrophiSim::Fluid
