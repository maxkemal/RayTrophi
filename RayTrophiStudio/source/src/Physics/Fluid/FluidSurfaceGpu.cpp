#include "Fluid/FrozenSurfaceMaterial.h"
#include "Fluid/FluidSurfaceGpu.h"

#include "PerfProfile.h"
#include "Fluid/SubstanceTag.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>

namespace RayTrophiSim::Fluid {
namespace {

// Host mirror of SurfacePC in shaders/sim_surface_common.glsl. Same order,
// same size; every surface kernel is registered with 128 push-constant bytes.
struct SurfaceGpuConstants {
    int32_t grid[4] = {};    // nx, ny, nz, particle_count (or element count)
    int32_t bins[4] = {};    // bnx, bny, bnz, bin_count
    float origin[4] = {};    // origin xyz, inverse bin width
    float cell[4] = {};      // voxel, sim voxel, R^2, R
    float centre[4] = {};    // smoothing radius^2, lambda, particle radius, offset
    float band[4] = {};      // far, 0.5/gradient width, displacement limit, -
    int32_t reach[4] = {};   // bin reach, neighbour minimum, smoothing stride, mode
    int32_t sim[4] = {};     // sim nx, ny, nz, -
};
static_assert(sizeof(SurfaceGpuConstants) == 128, "SurfacePC is 128 bytes");

constexpr uint32_t kThreads = 256;

uint32_t groupsFor(std::size_t n) {
    return static_cast<uint32_t>((std::max<std::size_t>(n, 1) + kThreads - 1) / kThreads);
}

bool ensureBuffer(SimulationComputeContext& compute, ComputeBufferHandle& handle,
                  std::size_t required, const char* name, bool render_read) {
    if (handle.valid() && handle.backend == compute.backendType() &&
        compute.getBufferSize(handle) >= required) {
        return true;
    }
    if (handle.valid()) {
        compute.destroyBuffer(handle);
        handle = {};
    }
    std::size_t grown = std::max<std::size_t>(required, 4096u);
    grown += grown / 2u;
    ComputeBufferDesc desc;
    desc.debug_name = name;
    desc.size_bytes = grown;
    desc.usage = ComputeBufferUsage::Storage | ComputeBufferUsage::Upload |
                 ComputeBufferUsage::Download | ComputeBufferUsage::ReadWrite;
    // The proxy and the coordinate field are read by the Vulkan RT backend
    // through their device addresses (Y2); only address-capable allocations
    // answer bufferDeviceAddress.
    if (render_read) desc.usage = desc.usage | ComputeBufferUsage::AccelInput;
    handle = compute.createBuffer(desc);
    return handle.valid();
}

struct Dispatcher {
    SimulationComputeContext& compute;
    bool ok = true;
    void run(const char* kernel, std::initializer_list<ComputeBufferHandle> list,
             const SurfaceGpuConstants& c, uint32_t groups) {
        if (!ok) return;
        ComputeBufferHandle bound[8];
        std::size_t n = 0;
        for (const auto& h : list) bound[n++] = h;
        ComputeDispatch cmd;
        cmd.kernel = kernel;
        cmd.buffers = bound;
        cmd.buffer_count = n;
        cmd.constants = &c;
        cmd.constants_size = sizeof(c);
        cmd.groups.groups_x = groups;
        ok = compute.dispatch(cmd);
    }
};

// Everything the CPU builder derives from the parameters, derived the same way.
struct Derived {
    int m = 1;
    float sim_voxel = 0.0f, voxel = 0.0f;
    int nx = 0, ny = 0, nz = 0;            // refined proxy grid
    int snx = 0, sny = 0, snz = 0;         // sim grid
    float kernel_R = 0.0f, particle_r = 0.0f, surface_offset = 0.0f, far_value = 0.0f;
    float bin_width = 0.0f;                // max(R, h): raw bins == support bins (no aniso)
    int bnx = 0, bny = 0, bnz = 0;
    int reach = 1;                         // support bin reach
    float rs = 0.0f; int reach_centres = 1; float lambda = 0.0f; int nmin = 1;
    int smoothing = 0; int stride = 1; float displacement_limit = 0.0f;
    int extension_sweeps = 3;
};

Derived derive(const FluidSim::FluidGrid& grid, const LevelSetParams& p) {
    Derived d;
    d.m = std::clamp(p.surface_resolution_multiplier, 1, 4);
    d.sim_voxel = grid.voxel_size;
    d.snx = grid.nx; d.sny = grid.ny; d.snz = grid.nz;
    d.nx = grid.nx * d.m; d.ny = grid.ny * d.m; d.nz = grid.nz * d.m;
    d.voxel = d.m > 1 ? d.sim_voxel / static_cast<float>(d.m) : d.sim_voxel;
    d.kernel_R = std::max(1e-4f, p.kernel_radius_voxels * d.sim_voxel);
    d.particle_r = std::max(0.0f, p.particle_radius_voxels * d.sim_voxel);
    d.surface_offset = std::clamp(p.surface_offset_voxels, -0.75f, 1.25f) * d.sim_voxel;
    d.far_value = std::max(d.kernel_R, p.narrow_band_voxels * d.sim_voxel);
    d.bin_width = std::max(d.kernel_R, d.sim_voxel);
    const float inv = 1.0f / d.bin_width;
    d.bnx = std::max(1, static_cast<int>(std::ceil(grid.nx * d.sim_voxel * inv)));
    d.bny = std::max(1, static_cast<int>(std::ceil(grid.ny * d.sim_voxel * inv)));
    d.bnz = std::max(1, static_cast<int>(std::ceil(grid.nz * d.sim_voxel * inv)));
    d.reach = std::max(1, static_cast<int>(std::ceil(d.kernel_R * inv)));
    d.rs = std::max(1e-4f, p.anisotropy_radius_voxels * d.sim_voxel);
    d.reach_centres = std::max(1, static_cast<int>(std::ceil(d.rs * inv)));
    d.lambda = std::clamp(p.position_smoothing, 0.0f, 1.0f);
    d.nmin = std::max(1, p.anisotropy_neighbor_min);
    d.smoothing = std::max(0, p.smoothing_iterations);
    d.stride = std::max(1, static_cast<int>(std::lround(d.sim_voxel / d.voxel)));
    d.displacement_limit = 0.2f * d.sim_voxel;
    const float shift = d.lambda * d.rs;
    d.extension_sweeps = std::max(3, static_cast<int>(std::ceil(shift / d.sim_voxel)) + 2);
    return d;
}

SurfaceGpuConstants baseConstants(const FluidSim::FluidGrid& grid, const Derived& d,
                                  std::size_t particles) {
    SurfaceGpuConstants c;
    c.grid[0] = d.nx; c.grid[1] = d.ny; c.grid[2] = d.nz;
    c.grid[3] = static_cast<int32_t>(particles);
    c.bins[0] = d.bnx; c.bins[1] = d.bny; c.bins[2] = d.bnz;
    c.bins[3] = d.bnx * d.bny * d.bnz;
    c.origin[0] = grid.origin.x; c.origin[1] = grid.origin.y; c.origin[2] = grid.origin.z;
    c.origin[3] = 1.0f / d.bin_width;
    c.cell[0] = d.voxel; c.cell[1] = d.sim_voxel;
    c.cell[2] = d.kernel_R * d.kernel_R; c.cell[3] = d.kernel_R;
    c.centre[0] = d.rs * d.rs; c.centre[1] = d.lambda;
    c.centre[2] = d.particle_r; c.centre[3] = d.surface_offset;
    c.band[0] = d.far_value; c.band[2] = d.displacement_limit;
    c.reach[0] = d.reach; c.reach[1] = d.nmin; c.reach[2] = d.smoothing;
    c.sim[0] = d.snx; c.sim[1] = d.sny; c.sim[2] = d.snz;
    return c;
}

// One build. `sdf_download` = parity mode (raw SDF, no proxy download).
bool run(SimulationComputeContext& compute, FluidSurfaceGpuBuffers& b,
         const FluidSurfaceGpuRequest& request, FluidSurfaceGpuResult& result,
         std::vector<float>* sdf_download, std::vector<float>* density_download,
         std::vector<float>* uvw_download, std::vector<float>* composition_download) {
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    result = {};
    if (!request.particles || !request.grid) {
        result.status = "no particles or grid";
        return false;
    }
    const auto& particles = *request.particles;
    const auto& grid = *request.grid;
    const std::size_t count = particles.size();
    if (compute.backendType() != ComputeBackendType::VulkanCompute ||
        !compute.supportsDispatch()) {
        result.status = "no Vulkan compute device";
        return false;
    }
    if (!fluidSurfaceGpuSupports(request.params, result.status)) return false;
    if (count == 0 || grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
        !(grid.voxel_size > 0.0f)) {
        result.status = "empty domain";
        return false;
    }
    const Derived d = derive(grid, request.params);
    const uint64_t cells64 = uint64_t(d.nx) * d.ny * d.nz;
    const uint64_t sim_cells64 = uint64_t(d.snx) * d.sny * d.snz;
    const uint64_t bins64 = uint64_t(d.bnx) * d.bny * d.bnz;
    constexpr uint64_t kMax = std::numeric_limits<int32_t>::max();
    if (cells64 >= kMax || bins64 >= kMax || count >= kMax) {
        result.status = "grid or particle count past the 32-bit index range";
        return false;
    }
    const std::size_t cells = static_cast<std::size_t>(cells64);
    const std::size_t sim_cells = static_cast<std::size_t>(sim_cells64);
    const std::size_t bins = static_cast<std::size_t>(bins64);
    const bool want_uvw = request.want_uvw && particles.uvw.size() >= count;

    // ── Host staging: mass carries the view selection (excluded = 0) ────────
    static thread_local std::vector<float> s_mass;
    static thread_local std::vector<float> s_radii;
    static thread_local std::vector<float> s_disp;
    s_mass.resize(count);
    s_radii.resize(count);
    const bool has_mass = particles.mass_fraction.size() >= count;
    for (std::size_t p = 0; p < count; ++p) {
        const bool keep = !request.selection || request.selection->keeps(particles, p);
        s_mass[p] = keep ? (has_mass ? particles.mass_fraction[p] : 1.0f) : 0.0f;
        const float fraction = std::clamp(s_mass[p], 0.0f, 1.0f);
        s_radii[p] = d.particle_r * std::cbrt(std::max(fraction, 0.02f));
    }
    if (want_uvw) {
        float w_a = 1.0f, w_b = 0.0f;
        particles.materialCoordWeights(w_a, w_b);
        const bool has_gen_b = particles.uvw_b.size() >= count;
        if (!has_gen_b) { w_a = 1.0f; w_b = 0.0f; }
        s_disp.resize(count * 3u);
        for (std::size_t p = 0; p < count; ++p) {
            const Vec3& x = particles.position[p];
            const Vec3 a = particles.uvw[p] - x;
            const Vec3 bgen = has_gen_b ? (particles.uvw_b[p] - x) : a;
            s_disp[p * 3u + 0] = w_a * a.x + w_b * bgen.x;
            s_disp[p * 3u + 1] = w_a * a.y + w_b * bgen.y;
            s_disp[p * 3u + 2] = w_a * a.z + w_b * bgen.z;
        }
    }

    // ── Composition slot table (buildCompositionGrid, same rules) ──────────
    // Slot 0 is the fallback; a tag with no binding takes it. Untagged liquid
    // resolves through its alias entry (collectSubstanceMaterialEntries).
    constexpr std::size_t kSlotCapacity = kSubstanceMaterialEntryCapacity;
    int slot_material[kSlotCapacity];
    float slot_miscibility[kSlotCapacity];
    std::size_t slot_count = 1;
    slot_material[0] = request.fallback_material;
    const bool composition_inputs =
        request.materials && request.material_count > 0;
    if (composition_inputs) {
        for (std::size_t i = 0; i < request.material_count &&
                                slot_count < kSlotCapacity; ++i) {
            bool seen = false;
            for (std::size_t s = 0; s < slot_count; ++s)
                if (slot_material[s] == request.materials[i].material_id) { seen = true; break; }
            if (!seen) slot_material[slot_count++] = request.materials[i].material_id;
        }
    }
    const bool want_composition = composition_inputs && slot_count > 1;
    static thread_local std::vector<uint32_t> s_slots;
    float slot_table[kSlotCapacity * 2] = {};
    if (want_composition) {
        for (std::size_t s = 0; s < kSlotCapacity; ++s) {
            slot_table[s] = s < slot_count ? static_cast<float>(slot_material[s]) : -1.0f;
            slot_miscibility[s] = 1.0f;
        }
        for (std::size_t i = 0; i < request.material_count; ++i) {
            for (std::size_t s = 0; s < slot_count; ++s) {
                if (slot_material[s] != request.materials[i].material_id) continue;
                slot_miscibility[s] = std::min(slot_miscibility[s],
                    std::clamp(request.materials[i].miscibility, 0.0f, 1.0f));
                break;
            }
        }
        for (std::size_t s = 0; s < kSlotCapacity; ++s)
            slot_table[kSlotCapacity + s] = slot_miscibility[s];
        s_slots.resize(count);
        for (std::size_t p = 0; p < count; ++p) {
            const int material = surfaceMaterialForParticle(particles, p,
                request.materials, request.material_count, request.fallback_material);
            uint32_t slot = 0;
            for (std::size_t candidate = 0; candidate < slot_count; ++candidate) {
                if (slot_material[candidate] == material) {
                    slot = static_cast<uint32_t>(candidate);
                    break;
                }
            }
            s_slots[p] = slot;
        }
    }

    // ── Storage ─────────────────────────────────────────────────────────────
    const std::size_t pf = count * sizeof(float);
    const std::size_t pu = count * sizeof(uint32_t);
    bool ok =
        ensureBuffer(compute, b.positions, count * sizeof(Vec3), "SurfacePositions", false) &&
        ensureBuffer(compute, b.mass, pf, "SurfaceMass", false) &&
        ensureBuffer(compute, b.support_radii, pf, "SurfaceSupportRadii", false) &&
        ensureBuffer(compute, b.bin1, pu, "SurfaceBin1", false) &&
        ensureBuffer(compute, b.bin2, pu, "SurfaceBin2", false) &&
        ensureBuffer(compute, b.items1, pu, "SurfaceItems1", false) &&
        ensureBuffer(compute, b.items2, pu, "SurfaceItems2", false) &&
        ensureBuffer(compute, b.centres, count * sizeof(Vec3), "SurfaceCentres", false) &&
        ensureBuffer(compute, b.counts, bins * 4u, "SurfaceBinCounts", false) &&
        ensureBuffer(compute, b.cursor, bins * 4u, "SurfaceBinCursor", false) &&
        ensureBuffer(compute, b.offsets1, (bins + 1u) * 4u, "SurfaceOffsets1", false) &&
        ensureBuffer(compute, b.offsets2, (bins + 1u) * 4u, "SurfaceOffsets2", false) &&
        ensureBuffer(compute, b.sdf_a, cells * 4u, "SurfaceSdfA", false) &&
        ensureBuffer(compute, b.sdf_b, cells * 4u, "SurfaceSdfB", false) &&
        ensureBuffer(compute, b.sdf_c, cells * 4u, "SurfaceSdfC", false) &&
        ensureBuffer(compute, b.confidence, cells * 4u, "SurfaceConfidence", false) &&
        ensureBuffer(compute, b.density, cells * 4u, "SurfaceDensityProxy", true) &&
        ensureBuffer(compute, b.stats, 8u * 4u, "SurfaceStats", false);
    if (ok && want_uvw) {
        ok = ensureBuffer(compute, b.disp, count * sizeof(Vec3), "SurfaceDisp", false) &&
             ensureBuffer(compute, b.uvw_a, sim_cells * 12u, "SurfaceUvwA", true) &&
             ensureBuffer(compute, b.uvw_b, sim_cells * 12u, "SurfaceUvwB", true) &&
             ensureBuffer(compute, b.valid_a, sim_cells * 4u, "SurfaceValidA", false) &&
             ensureBuffer(compute, b.valid_b, sim_cells * 4u, "SurfaceValidB", false);
    }
    if (ok && want_composition) {
        ok = ensureBuffer(compute, b.slots, pu, "SurfaceSlots", false) &&
             ensureBuffer(compute, b.slot_table, sizeof(slot_table), "SurfaceSlotTable", false) &&
             ensureBuffer(compute, b.composition, sim_cells * 12u, "SurfaceComposition", true);
    }
    if (!ok) {
        result.status = "device allocation failed";
        return false;
    }
    b.particle_capacity = std::max(b.particle_capacity, count);
    b.bin_capacity = std::max(b.bin_capacity, bins);
    b.cell_capacity = std::max(b.cell_capacity, cells);
    if (want_uvw) b.uvw_cell_capacity = std::max(b.uvw_cell_capacity, sim_cells);

    const uint32_t stats_init[8] = {0u, 0u, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0u, 0u, 0u};
    compute.beginTransferBatch();
    ok = compute.uploadBuffer(b.positions, particles.position.data(), count * sizeof(Vec3)) &&
         compute.uploadBuffer(b.mass, s_mass.data(), pf) &&
         compute.uploadBuffer(b.support_radii, s_radii.data(), pf) &&
         compute.uploadBuffer(b.stats, stats_init, sizeof(stats_init));
    if (ok && want_uvw) ok = compute.uploadBuffer(b.disp, s_disp.data(), count * sizeof(Vec3));
    if (ok && want_composition) {
        ok = compute.uploadBuffer(b.slots, s_slots.data(), pu) &&
             compute.uploadBuffer(b.slot_table, slot_table, sizeof(slot_table));
    }
    ok = compute.endTransferBatch() && ok;
    result.upload_bytes = count * sizeof(Vec3) + 2u * pf + sizeof(stats_init) +
                          (want_uvw ? count * sizeof(Vec3) : 0u) +
                          (want_composition ? pu + sizeof(slot_table) : 0u);
    if (!ok) {
        result.status = "upload failed";
        return false;
    }

    // ── Dispatch chain ──────────────────────────────────────────────────────
    Dispatcher chain{compute};
    const SurfaceGpuConstants base = baseConstants(grid, d, count);
    const auto clearBins = [&]() {
        SurfaceGpuConstants c = base;
        c.grid[3] = static_cast<int32_t>(bins);
        c.reach[3] = 0;
        chain.run("sim_surface_clear", {b.counts}, c, groupsFor(bins));
    };
    // Raw-position bins (centre filter + coordinate gather).
    clearBins();
    {
        SurfaceGpuConstants c = base;
        c.reach[3] = 0;
        chain.run("sim_surface_bin_assign", {b.positions, b.mass, b.bin1, b.bin1, b.counts},
                c, groupsFor(count));
    }
    chain.run("sim_surface_scan", {b.counts, b.offsets1, b.cursor}, base, 1);
    chain.run("sim_surface_scatter", {b.bin1, b.cursor, b.items1}, base, groupsFor(count));
    {
        SurfaceGpuConstants c = base;
        c.reach[0] = d.reach_centres;
        chain.run("sim_surface_centres", {b.positions, b.bin1, b.offsets1, b.items1, b.centres},
                c, groupsFor(count));
    }
    // Filtered-centre bins (SDF gather).
    clearBins();
    {
        SurfaceGpuConstants c = base;
        c.reach[3] = 1;
        chain.run("sim_surface_bin_assign", {b.centres, b.mass, b.bin1, b.bin2, b.counts},
                c, groupsFor(count));
    }
    chain.run("sim_surface_scan", {b.counts, b.offsets2, b.cursor}, base, 1);
    chain.run("sim_surface_scatter", {b.bin2, b.cursor, b.items2}, base, groupsFor(count));
    chain.run("sim_surface_gather_supported", {b.centres, b.mass, b.offsets2, b.items2,
                b.sdf_a, b.confidence, b.positions, b.support_radii}, base, groupsFor(cells));
    ComputeBufferHandle field = b.sdf_a;
    {
        SurfaceGpuConstants c = base;
        c.reach[2] = d.stride;
        ComputeBufferHandle ping[2] = {b.sdf_b, b.sdf_c};
        for (int it = 0; it < d.smoothing; ++it) {
            const ComputeBufferHandle dst = ping[it & 1];
            chain.run("sim_surface_smooth", {field, dst, b.sdf_a, b.confidence}, c,
                    groupsFor(cells));
            field = dst;
        }
    }
    {
        SurfaceGpuConstants c = base;
        const float grad_width = std::max(1.0f, request.surface_band_voxels) * d.sim_voxel;
        c.band[1] = 0.5f / grad_width;
        chain.run("sim_surface_proxy", {field, b.density, b.stats}, c, groupsFor(cells));
    }
    ComputeBufferHandle uvw_field = b.uvw_a;
    if (want_uvw) {
        SurfaceGpuConstants c = base;
        c.grid[0] = d.snx; c.grid[1] = d.sny; c.grid[2] = d.snz;
        c.cell[0] = d.sim_voxel;
        chain.run("sim_surface_uvw_gather", {b.positions, b.disp, b.offsets1, b.items1,
                                           b.uvw_a, b.valid_a}, c, groupsFor(sim_cells));
        ComputeBufferHandle uvw_in = b.uvw_a, valid_in = b.valid_a;
        ComputeBufferHandle uvw_out = b.uvw_b, valid_out = b.valid_b;
        const auto sweep = [&](int mode) {
            c.reach[3] = mode;
            chain.run("sim_surface_uvw_extend", {uvw_in, valid_in, uvw_out, valid_out}, c,
                    groupsFor(sim_cells));
            std::swap(uvw_in, uvw_out);
            std::swap(valid_in, valid_out);
        };
        for (int s = 0; s < d.extension_sweeps; ++s) sweep(0);
        for (int s = 0; s < 2; ++s) sweep(1);
        uvw_field = uvw_in;
    }
    if (want_composition) {
        SurfaceGpuConstants c = base;
        c.grid[0] = d.snx; c.grid[1] = d.sny; c.grid[2] = d.snz;
        c.cell[0] = d.sim_voxel;
        c.reach[1] = static_cast<int32_t>(slot_count);
        chain.run("sim_surface_composition_phase", {b.positions, b.slots, b.offsets1, b.items1,
                                             b.slot_table, b.composition}, c,
                  groupsFor(sim_cells));
    }
    if (!chain.ok) {
        compute.synchronize();
        result.status = "dispatch refused (surface kernel missing: compile shaders?)";
        return false;
    }

    // ── Readback: counters always (32 bytes), fields only when asked ────────
    uint32_t stats[8] = {};
    compute.beginTransferBatch();
    ok = compute.downloadBuffer(b.stats, stats, sizeof(stats));
    result.download_bytes = sizeof(stats);
    if (ok && sdf_download) {
        sdf_download->resize(cells);
        ok = compute.downloadBuffer(field, sdf_download->data(), cells * 4u);
        result.download_bytes += cells * 4u;
    }
    if (ok && density_download) {
        density_download->resize(cells);
        ok = compute.downloadBuffer(b.density, density_download->data(), cells * 4u);
        result.download_bytes += cells * 4u;
    }
    if (ok && want_uvw && uvw_download) {
        uvw_download->resize(sim_cells * 3u);
        ok = compute.downloadBuffer(uvw_field, uvw_download->data(), sim_cells * 12u);
        result.download_bytes += sim_cells * 12u;
    }
    if (ok && want_composition && composition_download) {
        composition_download->resize(sim_cells * 3u);
        ok = compute.downloadBuffer(b.composition, composition_download->data(), sim_cells * 12u);
        result.download_bytes += sim_cells * 12u;
    }
    ok = compute.endTransferBatch() && ok;
    if (!ok) {
        result.status = "readback failed";
        return false;
    }

    result.ok = true;
    result.nx = d.nx; result.ny = d.ny; result.nz = d.nz;
    result.voxel = d.voxel;
    result.active_cells = stats[0];
    result.surface_cells = stats[1];
    result.density = b.density;
    result.uvw_built = want_uvw;
    result.uvw = want_uvw ? uvw_field : ComputeBufferHandle{};
    result.uvw_nx = want_uvw ? d.snx : 0;
    result.uvw_ny = want_uvw ? d.sny : 0;
    result.uvw_nz = want_uvw ? d.snz : 0;
    result.composition_built = want_composition;
    result.composition = want_composition ? b.composition : ComputeBufferHandle{};
    // VDBVolume::occupiedLocalBox: occupied cells padded by one, rounded out to
    // 8-cell blocks, in grid-local units.
    if (stats[5] > 0u) {
        constexpr int kBlock = 8;
        const int n[3] = {d.nx, d.ny, d.nz};
        for (int a = 0; a < 3; ++a) {
            const int lo = static_cast<int>(stats[2 + a]);
            const int hi = static_cast<int>(stats[5 + a]) - 1;
            result.occupied_cell_min[a] = lo;
            result.occupied_cell_max[a] = hi;
            const int first = std::max(0, ((lo - 1) / kBlock) * kBlock);
            const int end_cell = std::min(n[a], ((hi + 2 + kBlock - 1) / kBlock) * kBlock);
            result.occupied_min[a] = static_cast<float>(first) * d.voxel;
            result.occupied_max[a] = static_cast<float>(end_cell) * d.voxel;
        }
        result.occupied = true;
    }
    result.build_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    return true;
}

} // namespace

void FluidSurfaceGpuBuffers::release(SimulationComputeContext& compute) {
    ComputeBufferHandle* all[] = {
        &positions, &mass, &disp, &support_radii, &bin1, &bin2, &counts, &cursor, &offsets1, &offsets2,
        &items1, &items2, &centres, &sdf_a, &sdf_b, &sdf_c, &confidence, &density, &stats,
        &uvw_a, &uvw_b, &valid_a, &valid_b, &slots, &slot_table, &composition};
    for (auto* h : all) {
        if (h->valid() && h->backend == compute.backendType()) compute.destroyBuffer(*h);
        *h = {};
    }
    particle_capacity = bin_capacity = cell_capacity = uvw_cell_capacity = 0;
}

bool fluidSurfaceGpuSupports(const LevelSetParams& params, std::string& why) {
    if (params.anisotropy_enabled) {
        why = "anisotropic kernels are CPU-only";
        return false;
    }
    why.clear();
    return true;
}

bool buildFluidSurfaceGpu(SimulationComputeContext& compute,
                          FluidSurfaceGpuBuffers& buffers,
                          const FluidSurfaceGpuRequest& request,
                          FluidSurfaceGpuResult& result,
                          std::vector<float>* density_host,
                          std::vector<float>* uvw_host,
                          std::vector<float>* composition_host,
                          LevelSetStats* stats) {
    RTPERF_FRAME_SCOPE("sim.render.surface.gpu");
    const bool built = run(compute, buffers, request, result, nullptr,
                           request.download_host ? density_host : nullptr,
                           request.download_host ? uvw_host : nullptr,
                           request.download_host ? composition_host : nullptr);
    if (built && stats) {
        *stats = {};
        stats->particle_count = request.particles->size();
        stats->grid_cell_count = static_cast<std::size_t>(result.nx) * result.ny * result.nz;
        stats->work_cells = stats->grid_cell_count;
        stats->active_cells = result.active_cells;
        stats->surface_cells = result.surface_cells;
        stats->build_ms = static_cast<float>(result.build_ms);
        stats->eff_nx = result.nx;
        stats->eff_ny = result.ny;
        stats->eff_nz = result.nz;
        stats->eff_voxel = result.voxel;
    }
    return built;
}

bool buildFluidSurfaceGpuSdf(SimulationComputeContext& compute,
                             FluidSurfaceGpuBuffers& buffers,
                             const FluidSurfaceGpuRequest& request,
                             std::vector<float>& sdf_out,
                             std::vector<float>* uvw_out,
                             std::vector<float>* composition_out,
                             FluidSurfaceGpuResult& result) {
    return run(compute, buffers, request, result, &sdf_out, nullptr, uvw_out, composition_out);
}

} // namespace RayTrophiSim::Fluid
