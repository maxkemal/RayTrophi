/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          APICFluidSolver.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 */

#include "Fluid/APICFluidSolver.h"
#include "GridFluidSolver.h"
#include "SimulationWorld.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// =====================================================================
// DEBUG BISECT FLAGS — temporarily flip to isolate the 0.036-voxel hang.
// Uncomment ONE at a time, rebuild, and seed at voxel 0.036.
// =====================================================================
// #define APIC_DEBUG_FORCE_SERIAL_SCATTER   // disables OMP scatter, runs single-thread
// #define APIC_DEBUG_SKIP_STEP              // makes Fluid::step a no-op (sim runs but does nothing)
// =====================================================================

namespace RayTrophiSim {
namespace Fluid {

// Shared static storage exposed by getLastFlipPreSnapshot*().
// Populated by the most recent Fluid::step call that ran pressure with skip_g2p=true.
static std::vector<float> g_flip_snap_x, g_flip_snap_y, g_flip_snap_z;
static bool g_flip_snap_valid = false;

bool         hasLastFlipPreSnapshot()      { return g_flip_snap_valid; }
std::size_t  getLastFlipPreSnapshotSize()  { return g_flip_snap_x.size(); }
const float* getLastFlipPreSnapshotX()     { return g_flip_snap_x.data(); }
const float* getLastFlipPreSnapshotY()     { return g_flip_snap_y.data(); }
const float* getLastFlipPreSnapshotZ()     { return g_flip_snap_z.data(); }

namespace {

int solverThreadCount(const APICSolverParams& params) {
#ifdef _OPENMP
    const int max_threads = std::max(1, omp_get_max_threads());
    const int app_threads = std::min(max_threads, 8);
    if (params.cpu_threads > 0) return std::clamp(params.cpu_threads, 1, app_threads);
    return app_threads;
#else
    (void)params;
    return 1;
#endif
}

bool shouldParallelParticles(const APICSolverParams& params, size_t count) {
    return solverThreadCount(params) > 1 &&
           count >= static_cast<size_t>(std::max(1, params.parallel_particle_threshold));
}

bool shouldParallelGrid(const APICSolverParams& params, size_t count) {
    constexpr size_t grid_threshold = 32768;
    return solverThreadCount(params) > 1 && count >= grid_threshold;
}

inline void clampAffine(AffineC& c, float max_abs) {
    const float limit = std::max(0.0f, max_abs);
    c.col0.x = std::clamp(c.col0.x, -limit, limit);
    c.col0.y = std::clamp(c.col0.y, -limit, limit);
    c.col0.z = std::clamp(c.col0.z, -limit, limit);
    c.col1.x = std::clamp(c.col1.x, -limit, limit);
    c.col1.y = std::clamp(c.col1.y, -limit, limit);
    c.col1.z = std::clamp(c.col1.z, -limit, limit);
    c.col2.x = std::clamp(c.col2.x, -limit, limit);
    c.col2.y = std::clamp(c.col2.y, -limit, limit);
    c.col2.z = std::clamp(c.col2.z, -limit, limit);
}

using SolverClock = std::chrono::steady_clock;

inline float elapsedMs(SolverClock::time_point start, SolverClock::time_point end) {
    return std::chrono::duration<float, std::milli>(end - start).count();
}

// Quadratic B-spline weights centred at the fractional grid coordinate `fx`.
// Returns the three weights for nodes (base-1, base, base+1) where
// base = floor(fx - 0.5). Standard APIC kernel (Jiang et al. 2015).
inline void quadraticWeights(float fx, int& base, float w[3]) {
    base = static_cast<int>(std::floor(fx - 0.5f));
    float d = fx - (base + 1.0f);          // d in [-0.5, 0.5]
    w[0] = 0.5f * (0.5f - d) * (0.5f - d);
    w[1] = 0.75f - d * d;
    w[2] = 0.5f * (0.5f + d) * (0.5f + d);
}

// Sample the staggered MAC velocity at an arbitrary grid-space position using
// trilinear interpolation. `comp` selects which component (0=x, 1=y, 2=z).
// Position is in grid units; staggering offsets are applied internally.
inline float sampleMACComponent(const FluidSim::FluidGrid& g,
                                int comp,
                                const Vec3& gridPos) {
    Vec3 p = gridPos;
    // Shift to the face-centred coordinate frame for this component.
    if (comp == 0)      { p.y -= 0.5f; p.z -= 0.5f; }
    else if (comp == 1) { p.x -= 0.5f; p.z -= 0.5f; }
    else                { p.x -= 0.5f; p.y -= 0.5f; }

    int i0 = static_cast<int>(std::floor(p.x));
    int j0 = static_cast<int>(std::floor(p.y));
    int k0 = static_cast<int>(std::floor(p.z));
    float fx = p.x - i0;
    float fy = p.y - j0;
    float fz = p.z - k0;

    auto clamp = [](int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); };
    int xmax = (comp == 0) ? g.nx     : g.nx - 1;
    int ymax = (comp == 1) ? g.ny     : g.ny - 1;
    int zmax = (comp == 2) ? g.nz     : g.nz - 1;
    int i1 = clamp(i0,     0, xmax);
    int i2 = clamp(i0 + 1, 0, xmax);
    int j1 = clamp(j0,     0, ymax);
    int j2 = clamp(j0 + 1, 0, ymax);
    int k1 = clamp(k0,     0, zmax);
    int k2 = clamp(k0 + 1, 0, zmax);

    auto fetch = [&](int i, int j, int k) -> float {
        if (comp == 0) return g.vel_x[g.velXIndex(i, j, k)];
        if (comp == 1) return g.vel_y[g.velYIndex(i, j, k)];
        return                 g.vel_z[g.velZIndex(i, j, k)];
    };

    float c000 = fetch(i1, j1, k1);
    float c100 = fetch(i2, j1, k1);
    float c010 = fetch(i1, j2, k1);
    float c110 = fetch(i2, j2, k1);
    float c001 = fetch(i1, j1, k2);
    float c101 = fetch(i2, j1, k2);
    float c011 = fetch(i1, j2, k2);
    float c111 = fetch(i2, j2, k2);

    float c00 = c000 * (1 - fx) + c100 * fx;
    float c10 = c010 * (1 - fx) + c110 * fx;
    float c01 = c001 * (1 - fx) + c101 * fx;
    float c11 = c011 * (1 - fx) + c111 * fx;
    float c0  = c00  * (1 - fy) + c10  * fy;
    float c1  = c01  * (1 - fy) + c11  * fy;
    return    c0  * (1 - fz) + c1  * fz;
}

// Tile-binned particle index list (CSR). Built once per P2G call and reused
// for all three velocity components. The bin key is the integer cell index
// of the particle position divided by FluidSim::TILE_SIZE — same tiling the
// sparse grid uses elsewhere, so the bin coordinates line up with the
// active_tiles list and any future GPU/NanoVDB scatter plumbing.
//
// Per-particle quadratic B-spline stencil reach is ±2 cells; with
// TILE_SIZE=8, two same-color tiles separated by one tile (8 cells apart)
// have stencil write sets ≥4 cells apart, so an 8-way (odd/even per axis)
// coloring is conflict-free. That coloring is the hook for the upcoming
// OpenMP step — this commit only sets up the data and the serial scatter.
struct ParticleTileBins {
    int tile_nx = 0, tile_ny = 0, tile_nz = 0;
    size_t tile_count = 0;
    std::vector<int> offsets;             // size tile_count + 1
    std::vector<uint32_t> indices;        // size parts.size()
    std::vector<int> active_tile_linear;  // tile linear indices with ≥1 particle
    // 8-color (odd/even per axis) partition of active_tile_linear. Color =
    // (tx&1) | ((ty&1)<<1) | ((tz&1)<<2). Within one color class, no two
    // tiles share a stencil neighbor (TILE_SIZE=8, ±2 reach → ≥4-cell gap),
    // so the inner scatter writes are conflict-free across threads.
    std::vector<int> color_buckets[8];
    // Scratch buffers reused across calls so the per-step allocator churn
    // doesn't explode at small voxel sizes (tens of millions of particles
    // would otherwise trigger 100MB+ heap traffic per frame and serialize
    // on the Windows heap lock — which can stall OMP workers entering
    // malloc inside the scatter).
    std::vector<int> tile_of_particle;
    std::vector<int> cursor;
};

static void buildParticleTileBins(const FluidParticles& parts,
                                  const FluidSim::FluidGrid& grid,
                                  ParticleTileBins& bins) {
    bins.tile_nx = grid.tiles_x;
    bins.tile_ny = grid.tiles_y;
    bins.tile_nz = grid.tiles_z;
    bins.tile_count = static_cast<size_t>(bins.tile_nx) *
                      static_cast<size_t>(bins.tile_ny) *
                      static_cast<size_t>(bins.tile_nz);
    bins.offsets.assign(bins.tile_count + 1, 0);
    bins.indices.resize(parts.size());
    bins.active_tile_linear.clear();

    if (parts.empty() || bins.tile_count == 0) return;

    const float invH = 1.0f / grid.voxel_size;
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int tnx = bins.tile_nx;
    const int tnxy = bins.tile_nx * bins.tile_ny;

    // Pass 1: assign each particle to a tile and count occupancy. Particles
    // outside the grid are clamped to the nearest cell — they will not write
    // useful contributions but they still need a bucket so the index array
    // stays dense and offsets line up.
    // tile_of_particle is a persistent scratch buffer in `bins` — `resize`
    // here is amortized O(1) once capacity stabilizes.
    bins.tile_of_particle.resize(parts.size());
    auto& tile_of_particle = bins.tile_of_particle;
    for (size_t pi = 0; pi < parts.size(); ++pi) {
        const Vec3 gp = (parts.position[pi] - grid.origin) * invH;
        int i = static_cast<int>(std::floor(gp.x));
        int j = static_cast<int>(std::floor(gp.y));
        int k = static_cast<int>(std::floor(gp.z));
        i = std::clamp(i, 0, nx - 1);
        j = std::clamp(j, 0, ny - 1);
        k = std::clamp(k, 0, nz - 1);
        const int tx = i / FluidSim::TILE_SIZE;
        const int ty = j / FluidSim::TILE_SIZE;
        const int tz = k / FluidSim::TILE_SIZE;
        const int t = tx + ty * tnx + tz * tnxy;
        tile_of_particle[pi] = t;
        ++bins.offsets[static_cast<size_t>(t) + 1];
    }

    // Exclusive prefix sum -> CSR offsets.
    for (size_t i = 1; i <= bins.tile_count; ++i) {
        bins.offsets[i] += bins.offsets[i - 1];
    }

    // Active tile list (linear indices, ordered) — touched only where the
    // running count actually grew. Same pass also bins each active tile by
    // its 8-color class so the OpenMP scatter can iterate one color at a
    // time without per-pass filtering. `clear()` keeps capacity; we
    // intentionally don't `reserve` here — push_back amortization handles
    // growth once and capacity persists across frames.
    for (int c = 0; c < 8; ++c) {
        bins.color_buckets[c].clear();
    }
    for (int tz = 0; tz < bins.tile_nz; ++tz)
    for (int ty = 0; ty < bins.tile_ny; ++ty)
    for (int tx = 0; tx < bins.tile_nx; ++tx) {
        const int t = tx + ty * tnx + tz * tnxy;
        if (bins.offsets[static_cast<size_t>(t) + 1] > bins.offsets[static_cast<size_t>(t)]) {
            bins.active_tile_linear.push_back(t);
            const int color = (tx & 1) | ((ty & 1) << 1) | ((tz & 1) << 2);
            bins.color_buckets[color].push_back(t);
        }
    }

    // Pass 2: scatter particle indices into their bins. `cursor` walks the
    // write head inside each tile's slice of `indices`. Persistent scratch
    // in `bins` — `assign` reuses capacity, only the value-fill runs.
    bins.cursor.assign(bins.tile_count, 0);
    auto& cursor = bins.cursor;
    for (size_t pi = 0; pi < parts.size(); ++pi) {
        const int t = tile_of_particle[pi];
        const int slot = bins.offsets[t] + cursor[t]++;
        bins.indices[static_cast<size_t>(slot)] = static_cast<uint32_t>(pi);
    }
}

} // namespace

static bool seedBoxCellRange(const Vec3& grid_origin,
                 int nx,
                 int ny,
                 int nz,
                 float voxel_size,
                 const Vec3& min_world,
                 const Vec3& max_world,
                 int& i0,
                 int& j0,
                 int& k0,
                 int& i1,
                 int& j1,
                 int& k1) {
    Vec3 gmin = grid_origin;
    Vec3 gmax = grid_origin + Vec3(nx * voxel_size, ny * voxel_size, nz * voxel_size);
    Vec3 lo(std::max(min_world.x, gmin.x),
        std::max(min_world.y, gmin.y),
        std::max(min_world.z, gmin.z));
    Vec3 hi(std::min(max_world.x, gmax.x),
        std::min(max_world.y, gmax.y),
        std::min(max_world.z, gmax.z));
    if (lo.x >= hi.x || lo.y >= hi.y || lo.z >= hi.z) return false;

    const float h = voxel_size;
    i0 = std::max(0, static_cast<int>(std::floor((lo.x - grid_origin.x) / h)));
    j0 = std::max(0, static_cast<int>(std::floor((lo.y - grid_origin.y) / h)));
    k0 = std::max(0, static_cast<int>(std::floor((lo.z - grid_origin.z) / h)));
    i1 = std::min(nx - 1, static_cast<int>(std::floor((hi.x - grid_origin.x) / h)));
    j1 = std::min(ny - 1, static_cast<int>(std::floor((hi.y - grid_origin.y) / h)));
    k1 = std::min(nz - 1, static_cast<int>(std::floor((hi.z - grid_origin.z) / h)));
    return i0 <= i1 && j0 <= j1 && k0 <= k1;
}

static bool seedBoxCellRange(const FluidSim::FluidGrid& grid,
                 const Vec3& min_world,
                 const Vec3& max_world,
                 int& i0,
                 int& j0,
                 int& k0,
                 int& i1,
                 int& j1,
                 int& k1) {
    return seedBoxCellRange(grid.origin,
                grid.nx,
                grid.ny,
                grid.nz,
                grid.voxel_size,
                min_world,
                max_world,
                i0,
                j0,
                k0,
                i1,
                j1,
                k1);
}

size_t estimateSeedBoxParticleCount(const Vec3& grid_origin,
                    int nx,
                    int ny,
                    int nz,
                    float voxel_size,
                    const Vec3& min_world,
                    const Vec3& max_world,
                    int particles_per_cell) {
    if (particles_per_cell <= 0 || nx <= 0 || ny <= 0 || nz <= 0 || voxel_size <= 0.0f) return 0;

    int i0, j0, k0, i1, j1, k1;
    if (!seedBoxCellRange(grid_origin,
              nx,
              ny,
              nz,
              voxel_size,
              min_world,
              max_world,
              i0,
              j0,
              k0,
              i1,
              j1,
              k1)) {
    return 0;
    }
    return static_cast<size_t>(std::max(0, i1 - i0 + 1)) *
       static_cast<size_t>(std::max(0, j1 - j0 + 1)) *
       static_cast<size_t>(std::max(0, k1 - k0 + 1)) *
       static_cast<size_t>(particles_per_cell);
}

size_t estimateSeedBoxParticleCount(const FluidSim::FluidGrid& grid,
                                    const Vec3& min_world,
                                    const Vec3& max_world,
                                    int particles_per_cell) {
    return estimateSeedBoxParticleCount(grid.origin,
                    grid.nx,
                    grid.ny,
                    grid.nz,
                    grid.voxel_size,
                    min_world,
                    max_world,
                    particles_per_cell);
}

void seedBox(FluidParticles& particles,
             const FluidSim::FluidGrid& grid,
             const Vec3& min_world,
             const Vec3& max_world,
             int particles_per_cell,
             uint32_t seed,
             size_t max_new_particles) {
    if (particles_per_cell <= 0 || max_new_particles == 0) return;

    int i0, j0, k0, i1, j1, k1;
    if (!seedBoxCellRange(grid, min_world, max_world, i0, j0, k0, i1, j1, k1)) return;

    std::mt19937 rng(seed ? seed : 0xC0FFEEu);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    const float h = grid.voxel_size;

    size_t added = static_cast<size_t>(std::max(0, i1 - i0 + 1)) *
                   static_cast<size_t>(std::max(0, j1 - j0 + 1)) *
                   static_cast<size_t>(std::max(0, k1 - k0 + 1)) *
                   static_cast<size_t>(particles_per_cell);
    added = std::min(added, max_new_particles);
    particles.reserve(particles.size() + added);

    size_t emitted = 0;
    for (int k = k0; k <= k1; ++k)
    for (int j = j0; j <= j1; ++j)
    for (int i = i0; i <= i1; ++i) {
        if (grid.isSolid(i, j, k)) continue;
        Vec3 cellMin = grid.origin + Vec3(i * h, j * h, k * h);
        for (int p = 0; p < particles_per_cell; ++p) {
            if (emitted >= max_new_particles) return;
            Vec3 jitter(U(rng), U(rng), U(rng));
            particles.emit(cellMin + jitter * h, Vec3(0, 0, 0));
            ++emitted;
        }
    }
}

// Scatter particle velocities (with APIC affine term) onto the MAC grid.
// One pass per velocity component, each on its own face-centred coordinate
// frame. Weights are quadratic B-spline (3x3x3 stencil per particle).
static void particleToGrid(const FluidParticles& parts,
                           FluidSim::FluidGrid& grid,
                           const APICSolverParams& params) {
    std::fill(grid.vel_x.begin(), grid.vel_x.end(), 0.0f);
    std::fill(grid.vel_y.begin(), grid.vel_y.end(), 0.0f);
    std::fill(grid.vel_z.begin(), grid.vel_z.end(), 0.0f);

    std::vector<float> wx(grid.vel_x.size(), 0.0f);
    std::vector<float> wy(grid.vel_y.size(), 0.0f);
    std::vector<float> wz(grid.vel_z.size(), 0.0f);

    const float h    = grid.voxel_size;
    const float invH = 1.0f / h;

    // Build the particle→tile CSR bin once. All three component passes
    // iterate over the same active tile list, so writes for a given tile
    // land in a contiguous region of vel_*/w_* and stay cache-warm across
    // its particle range. `bins` is function-static (NOT thread_local) so
    // its vectors keep their capacity across calls — steady-state has zero
    // allocator traffic. thread_local would be WRONG here: only the main
    // thread populates bins, then the OMP scatter reads it read-only from
    // many workers; with thread_local each worker would see its own empty
    // copy and crash on bins.offsets[active_t]. particleToGrid is only
    // ever called from the single-threaded SimulationWorld tick, so the
    // static is safe.
    static ParticleTileBins bins;
    buildParticleTileBins(parts, grid, bins);

    // Per-tile scatter worker. Iterates the particles inside one tile's
    // CSR slice and runs the quadratic B-spline stencil into the
    // face-centred velocity/weight fields. No write conflicts across
    // tiles that share a color class (see ParticleTileBins::color_buckets).
    auto scatterTile = [&](int comp,
                           int active_t,
                           std::vector<float>& vfield,
                           std::vector<float>& wfield) {
        const int slot_begin = bins.offsets[static_cast<size_t>(active_t)];
        const int slot_end   = bins.offsets[static_cast<size_t>(active_t) + 1];
        const int xmax = (comp == 0) ? grid.nx     : grid.nx - 1;
        const int ymax = (comp == 1) ? grid.ny     : grid.ny - 1;
        const int zmax = (comp == 2) ? grid.nz     : grid.nz - 1;

        for (int slot = slot_begin; slot < slot_end; ++slot) {
            const size_t pi = static_cast<size_t>(bins.indices[static_cast<size_t>(slot)]);
            Vec3 gp = (parts.position[pi] - grid.origin) * invH;
            if (comp == 0)      { gp.y -= 0.5f; gp.z -= 0.5f; }
            else if (comp == 1) { gp.x -= 0.5f; gp.z -= 0.5f; }
            else                { gp.x -= 0.5f; gp.y -= 0.5f; }

            int bx, by, bz;
            float wxk[3], wyk[3], wzk[3];
            quadraticWeights(gp.x, bx, wxk);
            quadraticWeights(gp.y, by, wyk);
            quadraticWeights(gp.z, bz, wzk);

            const float vp = (comp == 0) ? parts.velocity[pi].x
                           : (comp == 1) ? parts.velocity[pi].y
                                         : parts.velocity[pi].z;
            const AffineC& C = parts.affine[pi];

            for (int dk = 0; dk < 3; ++dk)
            for (int dj = 0; dj < 3; ++dj)
            for (int di = 0; di < 3; ++di) {
                int gi = bx + di, gj = by + dj, gk = bz + dk;
                if (gi < 0 || gi > xmax || gj < 0 || gj > ymax || gk < 0 || gk > zmax) continue;
                const float w = wxk[di] * wyk[dj] * wzk[dk];
                Vec3 dx_grid(static_cast<float>(gi) - gp.x,
                             static_cast<float>(gj) - gp.y,
                             static_cast<float>(gk) - gp.z);
                Vec3 dx_world = dx_grid * h;
                float apic;
                if (comp == 0)      apic = C.col0.x * dx_world.x + C.col1.x * dx_world.y + C.col2.x * dx_world.z;
                else if (comp == 1) apic = C.col0.y * dx_world.x + C.col1.y * dx_world.y + C.col2.y * dx_world.z;
                else                apic = C.col0.z * dx_world.x + C.col1.z * dx_world.y + C.col2.z * dx_world.z;

                const size_t idx = (comp == 0) ? grid.velXIndex(gi, gj, gk)
                                 : (comp == 1) ? grid.velYIndex(gi, gj, gk)
                                               : grid.velZIndex(gi, gj, gk);
                vfield[idx] += w * (vp + apic);
                wfield[idx] += w;
            }
        }
    };

    // Drive scatter over all active tiles. In parallel mode we run 8
    // sequential color passes (odd/even per axis); inside each color
    // class every tile's stencil write set is ≥4 cells from every other
    // tile in the same class, so the OpenMP loop has no atomics and no
    // per-thread scratch grids.
    auto scatter = [&](int comp, std::vector<float>& vfield, std::vector<float>& wfield) {
        // Recompute thread/parallel locals INSIDE the lambda — MSVC's
        // OpenMP frontend cannot read lambda-captured values from a
        // num_threads(...) clause (yields C2326). Locals declared in
        // lambda scope avoid the capture machinery entirely.
        const int  threads = solverThreadCount(params);
#ifdef APIC_DEBUG_FORCE_SERIAL_SCATTER
        const bool par = false;
#else
        const bool par     = threads > 1 && shouldParallelParticles(params, parts.size());
#endif

        if (par) {
            for (int color = 0; color < 8; ++color) {
                const std::vector<int>& bucket = bins.color_buckets[color];
                const int n = static_cast<int>(bucket.size());
                if (n == 0) continue;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) num_threads(threads)
#endif
                for (int i = 0; i < n; ++i) {
                    scatterTile(comp, bucket[static_cast<size_t>(i)], vfield, wfield);
                }
            }
        } else {
            for (int active_t : bins.active_tile_linear) {
                scatterTile(comp, active_t, vfield, wfield);
            }
        }

        // Weight normalization: per-index, no cross-cell dependency, so
        // safe to parallelize over the whole field.
        const int64_t nf = static_cast<int64_t>(vfield.size());
        const bool grid_par = shouldParallelGrid(params, vfield.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if(grid_par)
#endif
        for (int64_t i = 0; i < nf; ++i) {
            if (wfield[static_cast<size_t>(i)] > 1e-8f) {
                vfield[static_cast<size_t>(i)] /= wfield[static_cast<size_t>(i)];
            } else {
                vfield[static_cast<size_t>(i)] = 0.0f;
            }
        }
    };

    scatter(0, grid.vel_x, wx);
    scatter(1, grid.vel_y, wy);
    scatter(2, grid.vel_z, wz);
}

// Gather grid velocity back to particles and reconstruct the APIC affine C.
static void gridToParticle(FluidParticles& parts,
                           const FluidSim::FluidGrid& grid,
                           const APICSolverParams& params,
                           float dt,
                           const float* vel_x_pre,
                           const float* vel_y_pre,
                           const float* vel_z_pre) {
    const float h    = grid.voxel_size;
    const float invH = 1.0f / h;
    // Inverse second moment of the quadratic B-spline kernel: 4/h^2.
    const float D_inv = 4.0f * invH * invH;

    const bool parallel = shouldParallelParticles(params, parts.size());
    const int thread_count = solverThreadCount(params);
    // FLIP blend: 0 = pure PIC (current grid velocity), 1 = pure FLIP (old
    // particle velocity + pressure impulse). Requires a pre-projection grid
    // snapshot; if none was supplied (gas debug path, GPU branch) we
    // degenerate to PIC regardless of the parameter.
    const bool has_flip_snapshot = (vel_x_pre != nullptr) &&
                                   (vel_y_pre != nullptr) &&
                                   (vel_z_pre != nullptr);
    const float flip_ratio = has_flip_snapshot
        ? std::clamp(params.flip_blend, 0.0f, 1.0f)
        : 0.0f;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(parallel)
#endif
    for (int64_t raw_pi = 0; raw_pi < static_cast<int64_t>(parts.size()); ++raw_pi) {
        const size_t pi = static_cast<size_t>(raw_pi);
        Vec3 v_new(0, 0, 0);
        Vec3 v_pre(0, 0, 0); // gathered pre-projection grid velocity (for FLIP)
        AffineC C_new;

        for (int comp = 0; comp < 3; ++comp) {
            Vec3 gp = (parts.position[pi] - grid.origin) * invH;
            if (comp == 0)      { gp.y -= 0.5f; gp.z -= 0.5f; }
            else if (comp == 1) { gp.x -= 0.5f; gp.z -= 0.5f; }
            else                { gp.x -= 0.5f; gp.y -= 0.5f; }

            int bx, by, bz;
            float wxk[3], wyk[3], wzk[3];
            quadraticWeights(gp.x, bx, wxk);
            quadraticWeights(gp.y, by, wyk);
            quadraticWeights(gp.z, bz, wzk);

            int xmax = (comp == 0) ? grid.nx     : grid.nx - 1;
            int ymax = (comp == 1) ? grid.ny     : grid.ny - 1;
            int zmax = (comp == 2) ? grid.nz     : grid.nz - 1;

            float v_acc = 0.0f;
            float v_pre_acc = 0.0f;
            Vec3  c_acc(0, 0, 0);

            for (int dk = 0; dk < 3; ++dk)
            for (int dj = 0; dj < 3; ++dj)
            for (int di = 0; di < 3; ++di) {
                int gi = bx + di, gj = by + dj, gk = bz + dk;
                if (gi < 0 || gi > xmax || gj < 0 || gj > ymax || gk < 0 || gk > zmax) continue;
                float w = wxk[di] * wyk[dj] * wzk[dk];
                size_t face_idx;
                float vn;
                float vn_pre = 0.0f;
                if (comp == 0) {
                    face_idx = grid.velXIndex(gi, gj, gk);
                    vn = grid.vel_x[face_idx];
                    if (has_flip_snapshot) vn_pre = vel_x_pre[face_idx];
                } else if (comp == 1) {
                    face_idx = grid.velYIndex(gi, gj, gk);
                    vn = grid.vel_y[face_idx];
                    if (has_flip_snapshot) vn_pre = vel_y_pre[face_idx];
                } else {
                    face_idx = grid.velZIndex(gi, gj, gk);
                    vn = grid.vel_z[face_idx];
                    if (has_flip_snapshot) vn_pre = vel_z_pre[face_idx];
                }
                v_acc     += w * vn;
                v_pre_acc += w * vn_pre;

                Vec3 dx_grid(static_cast<float>(gi) - gp.x,
                             static_cast<float>(gj) - gp.y,
                             static_cast<float>(gk) - gp.z);
                Vec3 dx_world = dx_grid * h;
                c_acc = c_acc + dx_world * (w * vn);
            }

            if (comp == 0) { v_new.x = v_acc; v_pre.x = v_pre_acc; C_new.col0.x = c_acc.x * D_inv; C_new.col1.x = c_acc.y * D_inv; C_new.col2.x = c_acc.z * D_inv; }
            if (comp == 1) { v_new.y = v_acc; v_pre.y = v_pre_acc; C_new.col0.y = c_acc.x * D_inv; C_new.col1.y = c_acc.y * D_inv; C_new.col2.y = c_acc.z * D_inv; }
            if (comp == 2) { v_new.z = v_acc; v_pre.z = v_pre_acc; C_new.col0.z = c_acc.x * D_inv; C_new.col1.z = c_acc.y * D_inv; C_new.col2.z = c_acc.z * D_inv; }
        }

        // Linear velocity blend:
        //   PIC  = v_new
        //   FLIP = v_old_particle + (v_new - v_pre)
        //   out  = lerp(PIC, FLIP, flip_ratio)
        //        = v_new + flip_ratio * (v_old_particle - v_pre)
        // The APIC angular contribution lives in C_new and rides on either
        // mix; it is scaled by apic_blend (NOT flip_ratio) — they are
        // independent knobs.
        Vec3 v_out;
        if (flip_ratio > 0.0f) {
            const Vec3 v_old = parts.velocity[pi];
            v_out = v_new + (v_old - v_pre) * flip_ratio;
        } else {
            v_out = v_new;
        }

        // Internal friction (viscous decay). Exponential rate-based energy
        // loss: dv/dt = -ν·v  ⇒  v *= exp(-ν·dt). This is the per-particle
        // analogue of momentum loss to heat in a real viscous fluid.
        //
        // Earlier this was implemented as a lerp toward v_new (the PIC
        // sample), which seemed natural — PIC IS the locally-averaged
        // neighbour velocity, so dragging toward it should equilibrate
        // momentum. The problem: in a coherent flow (everyone moving the
        // same direction) PIC ≈ self, and lerp-to-self is a no-op. Pure
        // variance-based schemes can't dissipate uniform motion.
        //
        // Exponential decay sidesteps that entirely. rate=0 disables;
        // rate=0.5 → ~50% energy lost per ~1.4 s; rate=10 → per ~0.07 s.
        if (params.internal_friction > 0.0f && dt > 0.0f) {
            const float decay = std::exp(-params.internal_friction * dt);
            v_out = v_out * decay;
        }
        parts.velocity[pi] = v_out;

        const float affine_blend = std::clamp(params.apic_blend, 0.0f, 1.0f) *
                                   std::clamp(params.affine_damping, 0.0f, 1.0f);
        C_new.col0 = C_new.col0 * affine_blend;
        C_new.col1 = C_new.col1 * affine_blend;
        C_new.col2 = C_new.col2 * affine_blend;
        clampAffine(C_new, params.max_affine);
        parts.affine[pi] = C_new;
    }
}

// Free-surface pressure projection. Cells containing at least one particle
// are FLUID; cells flagged solid in the grid are SOLID; everything else is
// AIR (Dirichlet p = 0). Iterative Gauss-Seidel with successive over-
// relaxation, then a single velocity-gradient subtraction pass.
//
// Reference: Bridson "Fluid Simulation for Computer Graphics", ch. 5
// (variational pressure solve, simplified for uniform-density water).
static size_t projectPressureFreeSurface(const FluidParticles& parts,
                                         FluidSim::FluidGrid& grid,
                                         const APICSolverParams& params,
                                         float dt) {
    if (dt <= 0.0f) return 0;

    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float h    = grid.voxel_size;
    const float invH = 1.0f / h;
    const size_t total = static_cast<size_t>(nx) * ny * nz;

    enum CellType : uint8_t { CELL_AIR = 0, CELL_FLUID = 1, CELL_SOLID = 2 };
    std::vector<uint8_t> cell(total, CELL_AIR);
    // Per-cell particle population. Used for the Bridson density-targeted
    // pressure correction; ALSO used as a cheap fluid mask elsewhere.
    // Function-static for heap reuse across frames.
    static std::vector<int> proj_cell_count_buf;
    if (proj_cell_count_buf.size() < total) proj_cell_count_buf.assign(total, 0);
    else std::fill(proj_cell_count_buf.begin(), proj_cell_count_buf.begin() + total, 0);
    int* const proj_cell_count = proj_cell_count_buf.data();
    const int thread_count = solverThreadCount(params);
    const bool grid_parallel = shouldParallelGrid(params, total);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(grid_parallel)
#endif
    for (int64_t raw_i = 0; raw_i < static_cast<int64_t>(total); ++raw_i) {
        const size_t i = static_cast<size_t>(raw_i);
        if (grid.solid[i]) cell[i] = CELL_SOLID;
    }
    int min_i = nx, min_j = ny, min_k = nz;
    int max_i = -1, max_j = -1, max_k = -1;

    // Mark fluid cells from particle positions AND accumulate per-cell counts
    // for the density-targeted pressure correction below.
    for (size_t pi = 0; pi < parts.size(); ++pi) {
        Vec3 gp = (parts.position[pi] - grid.origin) * invH;
        int i = static_cast<int>(std::floor(gp.x));
        int j = static_cast<int>(std::floor(gp.y));
        int k = static_cast<int>(std::floor(gp.z));
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
        size_t idx = grid.cellIndex(i, j, k);
        if (cell[idx] != CELL_SOLID) {
            if (cell[idx] != CELL_FLUID) {
                cell[idx] = CELL_FLUID;
                min_i = std::min(min_i, i);
                min_j = std::min(min_j, j);
                min_k = std::min(min_k, k);
                max_i = std::max(max_i, i);
                max_j = std::max(max_j, j);
                max_k = std::max(max_k, k);
            }
            ++proj_cell_count[idx];
        }
    }

    if (max_i < min_i || max_j < min_j || max_k < min_k) {
        return 0;
    }

    min_i = std::max(0, min_i - 1);
    min_j = std::max(0, min_j - 1);
    min_k = std::max(0, min_k - 1);
    max_i = std::min(nx - 1, max_i + 1);
    max_j = std::min(ny - 1, max_j + 1);
    max_k = std::min(nz - 1, max_k + 1);

    const int active_nx = max_i - min_i + 1;
    const int active_ny = max_j - min_j + 1;
    const int active_nz = max_k - min_k + 1;
    const int64_t active_cell_count_i64 =
        static_cast<int64_t>(active_nx) *
        static_cast<int64_t>(active_ny) *
        static_cast<int64_t>(active_nz);
    const bool active_grid_parallel = shouldParallelGrid(params, static_cast<size_t>(active_cell_count_i64));

    // Count fluid cells via OMP reduction. Originally written as
    // `omp parallel num_threads(N) { omp for ... }` with a manual
    // per-thread local_count array + post-loop sum; that pattern
    // deadlocks MSVC vcomp inside the team barrier at large active
    // grids (observed: full UI freeze at voxel ~0.022 with ~57k
    // active cells, stack pointed straight at this OMP region).
    // `parallel for ... reduction(+:x)` uses vcomp's built-in reducer
    // and avoids the nested parallel/for split that triggers the bug.
    int64_t fluid_cell_count_i64 = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) reduction(+:fluid_cell_count_i64) if(active_grid_parallel)
#endif
    for (int64_t raw = 0; raw < active_cell_count_i64; ++raw) {
        const int i = min_i + static_cast<int>(raw % active_nx);
        const int j = min_j + static_cast<int>((raw / active_nx) % active_ny);
        const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * active_ny));
        if (cell[grid.cellIndex(i, j, k)] == CELL_FLUID) {
            ++fluid_cell_count_i64;
        }
    }
    const size_t fluid_cell_count = static_cast<size_t>(fluid_cell_count_i64);

    // Build right-hand side: scaled divergence at fluid cells. Solid faces
    // contribute zero (no flow through walls).
    const float scale_div = invH;
    std::vector<float>& p = grid.pressure;
    std::vector<float>& div = grid.divergence;
    std::fill(p.begin(),   p.end(),   0.0f);
    std::fill(div.begin(), div.end(), 0.0f);

    auto isFluid = [&](int i, int j, int k) {
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return false;
        return cell[grid.cellIndex(i, j, k)] == CELL_FLUID;
    };
    auto isSolid = [&](int i, int j, int k) {
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return true;
        return cell[grid.cellIndex(i, j, k)] == CELL_SOLID;
    };

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(active_grid_parallel)
#endif
    for (int64_t raw = 0; raw < active_cell_count_i64; ++raw) {
        const int i = min_i + static_cast<int>(raw % active_nx);
        const int j = min_j + static_cast<int>((raw / active_nx) % active_ny);
        const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * active_ny));
        if (cell[grid.cellIndex(i, j, k)] != CELL_FLUID) continue;
        float vx_lo = grid.vel_x[grid.velXIndex(i,     j, k)];
        float vx_hi = grid.vel_x[grid.velXIndex(i + 1, j, k)];
        float vy_lo = grid.vel_y[grid.velYIndex(i, j,     k)];
        float vy_hi = grid.vel_y[grid.velYIndex(i, j + 1, k)];
        float vz_lo = grid.vel_z[grid.velZIndex(i, j, k)];
        float vz_hi = grid.vel_z[grid.velZIndex(i, j, k + 1)];

        // Cancel divergence across solid faces (their velocity is treated as 0).
        if (isSolid(i - 1, j, k)) vx_lo = 0.0f;
        if (isSolid(i + 1, j, k)) vx_hi = 0.0f;
        if (isSolid(i, j - 1, k)) vy_lo = 0.0f;
        if (isSolid(i, j + 1, k)) vy_hi = 0.0f;
        if (isSolid(i, j, k - 1)) vz_lo = 0.0f;
        if (isSolid(i, j, k + 1)) vz_hi = 0.0f;

        float div_val = -scale_div *
            ((vx_hi - vx_lo) + (vy_hi - vy_lo) + (vz_hi - vz_lo));

        // Bridson density-targeted projection. Over-populated cells get a
        // positive contribution to the RHS, which the PCG converts to a
        // pressure rise → outward velocity gradient → particles expelled.
        // Sign matches the existing convention (div = -scale_div * div_v;
        // positive RHS pushes pressure up). Scaled by 1/target so the knob
        // is dimensionless and independent of seed density.
        if (params.density_correction > 0.0f) {
            const int target = std::max(1, params.particles_per_cell);
            const int count = proj_cell_count[grid.cellIndex(i, j, k)];
            const int over = count - target;
            if (over > 0) {
                div_val += params.density_correction * scale_div *
                           static_cast<float>(over) / static_cast<float>(target);
            }
        }

        div[grid.cellIndex(i, j, k)] = div_val;
    }

    // PCG (Preconditioned Conjugate Gradient) with MIC(0) (Modified Incomplete
    // Cholesky, zero fill-in) preconditioner. Replaces the previous SOR
    // Gauss-Seidel relaxation: for typical liquid Poisson systems PCG converges
    // in 10-30 iterations to 1e-6 residual, vs. SOR which needs O(N) iterations
    // to propagate boundary information and stalls visibly on splashes/walls.
    //
    // System being solved (Bridson ch.5, uniform-density free-surface form):
    //   A[c,c] = #non-solid neighbours of c     (FLUID + AIR contribute)
    //   A[c,n] = -1 if neighbour n is FLUID, 0 if AIR (Dirichlet p=0) or SOLID
    //   b[c]   = div[c] / pressure_scale         (already-scaled negative div)
    // After PCG: subtract dt*grad(p)/h from faces; identical to the old code.
    const float pressure_scale = dt / (h * h);
    const int   max_iterations = std::max(1, params.pressure_iterations);

    // Function-static scratch — per-call vector<float> allocation stalls on the
    // heap lock under contention; reuse across frames and grow as needed.
    static std::vector<float>  Adiag_buf;   // diagonal coefficient (non-solid neighbour count)
    static std::vector<float>  Aplusi_buf;  // off-diagonal to (i+1,j,k); -1 if both fluid, else 0
    static std::vector<float>  Aplusj_buf;  // off-diagonal to (i,j+1,k)
    static std::vector<float>  Aplusk_buf;  // off-diagonal to (i,j,k+1)
    static std::vector<float>  precon_buf;  // MIC(0) E^{-1/2}
    static std::vector<float>  r_buf;       // residual
    static std::vector<float>  z_buf;       // preconditioned residual (M^{-1} r)
    static std::vector<float>  s_buf;       // search direction
    static std::vector<float>  As_buf;      // A * s
    static std::vector<float>  q_buf;       // MIC(0) forward-sub intermediate
    static std::vector<uint8_t> is_fluid_buf; // fast O(1) fluid mask

    if (Adiag_buf.size()    < total) Adiag_buf.assign(total, 0.0f);    else std::fill(Adiag_buf.begin(),  Adiag_buf.begin()  + total, 0.0f);
    if (Aplusi_buf.size()   < total) Aplusi_buf.assign(total, 0.0f);   else std::fill(Aplusi_buf.begin(), Aplusi_buf.begin() + total, 0.0f);
    if (Aplusj_buf.size()   < total) Aplusj_buf.assign(total, 0.0f);   else std::fill(Aplusj_buf.begin(), Aplusj_buf.begin() + total, 0.0f);
    if (Aplusk_buf.size()   < total) Aplusk_buf.assign(total, 0.0f);   else std::fill(Aplusk_buf.begin(), Aplusk_buf.begin() + total, 0.0f);
    if (precon_buf.size()   < total) precon_buf.assign(total, 0.0f);   else std::fill(precon_buf.begin(), precon_buf.begin() + total, 0.0f);
    if (r_buf.size()        < total) r_buf.assign(total, 0.0f);        else std::fill(r_buf.begin(),      r_buf.begin()      + total, 0.0f);
    if (z_buf.size()        < total) z_buf.assign(total, 0.0f);        else std::fill(z_buf.begin(),      z_buf.begin()      + total, 0.0f);
    if (s_buf.size()        < total) s_buf.assign(total, 0.0f);        else std::fill(s_buf.begin(),      s_buf.begin()      + total, 0.0f);
    if (As_buf.size()       < total) As_buf.assign(total, 0.0f);       else std::fill(As_buf.begin(),     As_buf.begin()     + total, 0.0f);
    if (q_buf.size()        < total) q_buf.assign(total, 0.0f);        else std::fill(q_buf.begin(),      q_buf.begin()      + total, 0.0f);
    if (is_fluid_buf.size() < total) is_fluid_buf.assign(total, 0u);   else std::fill(is_fluid_buf.begin(), is_fluid_buf.begin() + total, 0u);

    float* const Adiag  = Adiag_buf.data();
    float* const Aplusi = Aplusi_buf.data();
    float* const Aplusj = Aplusj_buf.data();
    float* const Aplusk = Aplusk_buf.data();
    float* const precon = precon_buf.data();
    float* const r      = r_buf.data();
    float* const z      = z_buf.data();
    float* const s      = s_buf.data();
    float* const As     = As_buf.data();
    float* const q      = q_buf.data();
    uint8_t* const is_fluid = is_fluid_buf.data();

    // 1. Build coefficient matrix (only fluid cells contribute rows). Air
    //    neighbours raise Adiag (they belong to the non-solid count) but their
    //    off-diagonal coefficient stays 0 — they're the Dirichlet ghosts that
    //    give the free surface its zero-pressure boundary.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(active_grid_parallel)
#endif
    for (int64_t raw = 0; raw < active_cell_count_i64; ++raw) {
        const int i = min_i + static_cast<int>(raw % active_nx);
        const int j = min_j + static_cast<int>((raw / active_nx) % active_ny);
        const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * active_ny));
        const size_t c = grid.cellIndex(i, j, k);
        if (cell[c] != CELL_FLUID) continue;
        is_fluid[c] = 1u;

        float diag = 0.0f;
        if (!isSolid(i - 1, j, k)) diag += 1.0f;
        if (!isSolid(i + 1, j, k)) { diag += 1.0f; if (isFluid(i + 1, j, k)) Aplusi[c] = -1.0f; }
        if (!isSolid(i, j - 1, k)) diag += 1.0f;
        if (!isSolid(i, j + 1, k)) { diag += 1.0f; if (isFluid(i, j + 1, k)) Aplusj[c] = -1.0f; }
        if (!isSolid(i, j, k - 1)) diag += 1.0f;
        if (!isSolid(i, j, k + 1)) { diag += 1.0f; if (isFluid(i, j, k + 1)) Aplusk[c] = -1.0f; }
        Adiag[c] = diag;
    }

    // 2. MIC(0) preconditioner. E[c] = A[c,c] - sum_{n in -i,-j,-k}
    //    (A[n,c])^2 * precon[n]^2 - tau * A[n,c] * precon[n]^2 *
    //    (sum of other-axis A[n,*] coefficients), with the standard safety
    //    fallback when E becomes too small relative to A[c,c].
    //    Reference: Bridson & Müller-Fischer SIGGRAPH 2007 course notes,
    //    "Fluid Simulation for Computer Graphics" 2nd ed. §5.2.
    constexpr float tau = 0.97f;       // MIC tuning constant
    constexpr float safety = 0.25f;    // restart threshold (Bridson)
    for (int k = min_k; k <= max_k; ++k) {
        for (int j = min_j; j <= max_j; ++j) {
            for (int i = min_i; i <= max_i; ++i) {
                const size_t c = grid.cellIndex(i, j, k);
                if (!is_fluid[c]) continue;

                float e = Adiag[c];

                if (i > 0) {
                    const size_t cm = grid.cellIndex(i - 1, j, k);
                    if (is_fluid[cm]) {
                        const float a = Aplusi[cm];   // A[(i-1,j,k), c]
                        const float pre_m = precon[cm];
                        e -= (a * pre_m) * (a * pre_m);
                        e -= tau * a * pre_m * pre_m * (Aplusj[cm] + Aplusk[cm]);
                    }
                }
                if (j > 0) {
                    const size_t cm = grid.cellIndex(i, j - 1, k);
                    if (is_fluid[cm]) {
                        const float a = Aplusj[cm];
                        const float pre_m = precon[cm];
                        e -= (a * pre_m) * (a * pre_m);
                        e -= tau * a * pre_m * pre_m * (Aplusi[cm] + Aplusk[cm]);
                    }
                }
                if (k > 0) {
                    const size_t cm = grid.cellIndex(i, j, k - 1);
                    if (is_fluid[cm]) {
                        const float a = Aplusk[cm];
                        const float pre_m = precon[cm];
                        e -= (a * pre_m) * (a * pre_m);
                        e -= tau * a * pre_m * pre_m * (Aplusi[cm] + Aplusj[cm]);
                    }
                }

                if (e < safety * Adiag[c]) {
                    e = Adiag[c]; // fall back to Jacobi diagonal for this cell
                }
                precon[c] = (e > 1e-12f) ? (1.0f / std::sqrt(e)) : 0.0f;
            }
        }
    }

    // Build the initial residual r = b - A*p; since we restart with p=0 each
    // step (warm-start is risky for free-surface — air cell topology changes),
    // r = b = div / pressure_scale. Reset pressure to zero.
    std::fill(p.begin(), p.end(), 0.0f);
    float r_inf = 0.0f;
    for (int k = min_k; k <= max_k; ++k) {
        for (int j = min_j; j <= max_j; ++j) {
            for (int i = min_i; i <= max_i; ++i) {
                const size_t c = grid.cellIndex(i, j, k);
                if (!is_fluid[c]) continue;
                const float rhs = div[c] / pressure_scale;
                r[c] = rhs;
                r_inf = std::max(r_inf, std::abs(rhs));
            }
        }
    }

    const float abs_tol = 1e-6f;
    if (r_inf <= abs_tol) {
        // Already divergence-free (e.g. fresh seed before any forces). Skip
        // straight to gradient subtraction — pressure remains zero.
    } else {
        const float tolerance = std::max(abs_tol, 1e-5f * r_inf);

        auto applyPreconditioner = [&]() {
            // Forward substitution: solve (L) q = r, where L is the lower
            // triangular part of MIC(0). Sweep in lex order; serial — MIC(0)
            // has data dependencies that block vectorisation/OMP without a
            // colouring pass (deferred).
            for (int k = min_k; k <= max_k; ++k) {
                for (int j = min_j; j <= max_j; ++j) {
                    for (int i = min_i; i <= max_i; ++i) {
                        const size_t c = grid.cellIndex(i, j, k);
                        if (!is_fluid[c]) continue;
                        float t = r[c];
                        if (i > 0) {
                            const size_t cm = grid.cellIndex(i - 1, j, k);
                            if (is_fluid[cm]) t -= Aplusi[cm] * precon[cm] * q[cm];
                        }
                        if (j > 0) {
                            const size_t cm = grid.cellIndex(i, j - 1, k);
                            if (is_fluid[cm]) t -= Aplusj[cm] * precon[cm] * q[cm];
                        }
                        if (k > 0) {
                            const size_t cm = grid.cellIndex(i, j, k - 1);
                            if (is_fluid[cm]) t -= Aplusk[cm] * precon[cm] * q[cm];
                        }
                        q[c] = t * precon[c];
                    }
                }
            }
            // Backward substitution: solve (L^T) z = q.
            for (int k = max_k; k >= min_k; --k) {
                for (int j = max_j; j >= min_j; --j) {
                    for (int i = max_i; i >= min_i; --i) {
                        const size_t c = grid.cellIndex(i, j, k);
                        if (!is_fluid[c]) continue;
                        float t = q[c];
                        if (i + 1 < nx) {
                            const size_t cp = grid.cellIndex(i + 1, j, k);
                            if (is_fluid[cp]) t -= Aplusi[c] * precon[c] * z[cp];
                        }
                        if (j + 1 < ny) {
                            const size_t cp = grid.cellIndex(i, j + 1, k);
                            if (is_fluid[cp]) t -= Aplusj[c] * precon[c] * z[cp];
                        }
                        if (k + 1 < nz) {
                            const size_t cp = grid.cellIndex(i, j, k + 1);
                            if (is_fluid[cp]) t -= Aplusk[c] * precon[c] * z[cp];
                        }
                        z[c] = t * precon[c];
                    }
                }
            }
        };

        // applyA computes As = A * s, restricted to fluid cells. Off-diagonal
        // entries from the lower-neighbour side use symmetry: A[n,c] is the
        // same as A[c,n] stored at the lower cell (Aplusi[i-1] == A[(i-1,c)]).
        auto applyA = [&]() {
            // MSVC vcomp: num_threads()/if() clauses can't read lambda-captured
            // variables (C2326). Re-declare as locals inside the lambda scope.
            const int  omp_threads = thread_count;
            const bool omp_parallel = active_grid_parallel;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(omp_threads) if(omp_parallel)
#endif
            for (int64_t raw = 0; raw < active_cell_count_i64; ++raw) {
                const int i = min_i + static_cast<int>(raw % active_nx);
                const int j = min_j + static_cast<int>((raw / active_nx) % active_ny);
                const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * active_ny));
                const size_t c = grid.cellIndex(i, j, k);
                if (!is_fluid[c]) { As[c] = 0.0f; continue; }

                float v = Adiag[c] * s[c];
                if (i > 0) {
                    const size_t cm = grid.cellIndex(i - 1, j, k);
                    if (is_fluid[cm]) v += Aplusi[cm] * s[cm];
                }
                if (i + 1 < nx) {
                    const size_t cp = grid.cellIndex(i + 1, j, k);
                    if (is_fluid[cp]) v += Aplusi[c] * s[cp];
                }
                if (j > 0) {
                    const size_t cm = grid.cellIndex(i, j - 1, k);
                    if (is_fluid[cm]) v += Aplusj[cm] * s[cm];
                }
                if (j + 1 < ny) {
                    const size_t cp = grid.cellIndex(i, j + 1, k);
                    if (is_fluid[cp]) v += Aplusj[c] * s[cp];
                }
                if (k > 0) {
                    const size_t cm = grid.cellIndex(i, j, k - 1);
                    if (is_fluid[cm]) v += Aplusk[cm] * s[cm];
                }
                if (k + 1 < nz) {
                    const size_t cp = grid.cellIndex(i, j, k + 1);
                    if (is_fluid[cp]) v += Aplusk[c] * s[cp];
                }
                As[c] = v;
            }
        };

        // z = M^{-1} r; s = z; sigma = z·r
        applyPreconditioner();
        double sigma = 0.0;
        for (int64_t raw = 0; raw < active_cell_count_i64; ++raw) {
            const int i = min_i + static_cast<int>(raw % active_nx);
            const int j = min_j + static_cast<int>((raw / active_nx) % active_ny);
            const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * active_ny));
            const size_t c = grid.cellIndex(i, j, k);
            if (!is_fluid[c]) continue;
            s[c] = z[c];
            sigma += static_cast<double>(z[c]) * static_cast<double>(r[c]);
        }

        int iter_used = 0;
        for (int iter = 0; iter < max_iterations; ++iter) {
            iter_used = iter + 1;
            applyA();

            double sAs = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) reduction(+:sAs) if(active_grid_parallel)
#endif
            for (int64_t raw = 0; raw < active_cell_count_i64; ++raw) {
                const int i = min_i + static_cast<int>(raw % active_nx);
                const int j = min_j + static_cast<int>((raw / active_nx) % active_ny);
                const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * active_ny));
                const size_t c = grid.cellIndex(i, j, k);
                if (!is_fluid[c]) continue;
                sAs += static_cast<double>(s[c]) * static_cast<double>(As[c]);
            }
            if (std::abs(sAs) < 1e-30) break; // degenerate — orthogonal search dir

            const float alpha = static_cast<float>(sigma / sAs);

            // p += alpha * s ; r -= alpha * As ; track new residual inf-norm
            float new_r_inf = 0.0f;
            for (int64_t raw = 0; raw < active_cell_count_i64; ++raw) {
                const int i = min_i + static_cast<int>(raw % active_nx);
                const int j = min_j + static_cast<int>((raw / active_nx) % active_ny);
                const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * active_ny));
                const size_t c = grid.cellIndex(i, j, k);
                if (!is_fluid[c]) continue;
                p[c] += alpha * s[c];
                r[c] -= alpha * As[c];
                new_r_inf = std::max(new_r_inf, std::abs(r[c]));
            }
            if (new_r_inf <= tolerance) break;

            applyPreconditioner();

            double sigma_new = 0.0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) reduction(+:sigma_new) if(active_grid_parallel)
#endif
            for (int64_t raw = 0; raw < active_cell_count_i64; ++raw) {
                const int i = min_i + static_cast<int>(raw % active_nx);
                const int j = min_j + static_cast<int>((raw / active_nx) % active_ny);
                const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * active_ny));
                const size_t c = grid.cellIndex(i, j, k);
                if (!is_fluid[c]) continue;
                sigma_new += static_cast<double>(z[c]) * static_cast<double>(r[c]);
            }
            const float beta = (sigma > 1e-30) ? static_cast<float>(sigma_new / sigma) : 0.0f;

            // s = z + beta * s
            for (int64_t raw = 0; raw < active_cell_count_i64; ++raw) {
                const int i = min_i + static_cast<int>(raw % active_nx);
                const int j = min_j + static_cast<int>((raw / active_nx) % active_ny);
                const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * active_ny));
                const size_t c = grid.cellIndex(i, j, k);
                if (!is_fluid[c]) continue;
                s[c] = z[c] + beta * s[c];
            }
            sigma = sigma_new;
        }
        (void)iter_used; // available for future stats plumbing
    }

    // Subtract pressure gradient from velocities. Only faces with at least
    // one fluid cell on either side are updated; solid faces are zeroed.
    const float grad_scale = dt / h;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(active_grid_parallel)
#endif
    for (int64_t raw = 0; raw < static_cast<int64_t>(active_nx + 1) * active_ny * active_nz; ++raw) {
        const int i = min_i + static_cast<int>(raw % (active_nx + 1));
        const int j = min_j + static_cast<int>((raw / (active_nx + 1)) % active_ny);
        const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx + 1) * active_ny));
        bool lo_solid = isSolid(i - 1, j, k);
        bool hi_solid = isSolid(i,     j, k);
        if (lo_solid || hi_solid) {
            grid.vel_x[grid.velXIndex(i, j, k)] = 0.0f;
            continue;
        }
        bool lo_fluid = isFluid(i - 1, j, k);
        bool hi_fluid = isFluid(i,     j, k);
        if (!lo_fluid && !hi_fluid) continue;
        float p_lo = lo_fluid ? p[grid.cellIndex(i - 1, j, k)] : 0.0f;
        float p_hi = hi_fluid ? p[grid.cellIndex(i,     j, k)] : 0.0f;
        grid.vel_x[grid.velXIndex(i, j, k)] -= grad_scale * (p_hi - p_lo);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(active_grid_parallel)
#endif
    for (int64_t raw = 0; raw < static_cast<int64_t>(active_nx) * (active_ny + 1) * active_nz; ++raw) {
        const int i = min_i + static_cast<int>(raw % active_nx);
        const int j = min_j + static_cast<int>((raw / active_nx) % (active_ny + 1));
        const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * (active_ny + 1)));
        bool lo_solid = isSolid(i, j - 1, k);
        bool hi_solid = isSolid(i, j,     k);
        if (lo_solid || hi_solid) {
            grid.vel_y[grid.velYIndex(i, j, k)] = 0.0f;
            continue;
        }
        bool lo_fluid = isFluid(i, j - 1, k);
        bool hi_fluid = isFluid(i, j,     k);
        if (!lo_fluid && !hi_fluid) continue;
        float p_lo = lo_fluid ? p[grid.cellIndex(i, j - 1, k)] : 0.0f;
        float p_hi = hi_fluid ? p[grid.cellIndex(i, j,     k)] : 0.0f;
        grid.vel_y[grid.velYIndex(i, j, k)] -= grad_scale * (p_hi - p_lo);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(active_grid_parallel)
#endif
    for (int64_t raw = 0; raw < static_cast<int64_t>(active_nx) * active_ny * (active_nz + 1); ++raw) {
        const int i = min_i + static_cast<int>(raw % active_nx);
        const int j = min_j + static_cast<int>((raw / active_nx) % active_ny);
        const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * active_ny));
        bool lo_solid = isSolid(i, j, k - 1);
        bool hi_solid = isSolid(i, j, k);
        if (lo_solid || hi_solid) {
            grid.vel_z[grid.velZIndex(i, j, k)] = 0.0f;
            continue;
        }
        bool lo_fluid = isFluid(i, j, k - 1);
        bool hi_fluid = isFluid(i, j, k);
        if (!lo_fluid && !hi_fluid) continue;
        float p_lo = lo_fluid ? p[grid.cellIndex(i, j, k - 1)] : 0.0f;
        float p_hi = hi_fluid ? p[grid.cellIndex(i, j, k)]     : 0.0f;
        grid.vel_z[grid.velZIndex(i, j, k)] -= grad_scale * (p_hi - p_lo);
    }

    return fluid_cell_count;
}

// Enforce no-flow through solid cells on the staggered velocity field. This
// matches the gas solver's wall handling.
static void enforceSolidBoundaries(FluidSim::FluidGrid& grid, const APICSolverParams& params) {
    const int thread_count = solverThreadCount(params);
    const bool grid_parallel = shouldParallelGrid(params,
        static_cast<size_t>(grid.nx) * static_cast<size_t>(grid.ny) * static_cast<size_t>(grid.nz));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(grid_parallel)
#endif
    for (int64_t raw = 0; raw < static_cast<int64_t>(grid.nx + 1) * grid.ny * grid.nz; ++raw) {
        const int i = static_cast<int>(raw % (grid.nx + 1));
        const int j = static_cast<int>((raw / (grid.nx + 1)) % grid.ny);
        const int k = static_cast<int>(raw / (static_cast<int64_t>(grid.nx + 1) * grid.ny));
        if ((i > 0 && grid.isSolid(i - 1, j, k)) ||
            (i < grid.nx && grid.isSolid(i, j, k))) {
            grid.vel_x[grid.velXIndex(i, j, k)] = 0.0f;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(grid_parallel)
#endif
    for (int64_t raw = 0; raw < static_cast<int64_t>(grid.nx) * (grid.ny + 1) * grid.nz; ++raw) {
        const int i = static_cast<int>(raw % grid.nx);
        const int j = static_cast<int>((raw / grid.nx) % (grid.ny + 1));
        const int k = static_cast<int>(raw / (static_cast<int64_t>(grid.nx) * (grid.ny + 1)));
        if ((j > 0 && grid.isSolid(i, j - 1, k)) ||
            (j < grid.ny && grid.isSolid(i, j, k))) {
            grid.vel_y[grid.velYIndex(i, j, k)] = 0.0f;
        }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(grid_parallel)
#endif
    for (int64_t raw = 0; raw < static_cast<int64_t>(grid.nx) * grid.ny * (grid.nz + 1); ++raw) {
        const int i = static_cast<int>(raw % grid.nx);
        const int j = static_cast<int>((raw / grid.nx) % grid.ny);
        const int k = static_cast<int>(raw / (static_cast<int64_t>(grid.nx) * grid.ny));
        if ((k > 0 && grid.isSolid(i, j, k - 1)) ||
            (k < grid.nz && grid.isSolid(i, j, k))) {
            grid.vel_z[grid.velZIndex(i, j, k)] = 0.0f;
        }
    }
}

static void applyViscosity(FluidSim::FluidGrid& grid, const APICSolverParams& params, float dt) {
    if (params.viscosity <= 0.0f || dt <= 0.0f) {
        return;
    }

    const int iterations = std::clamp(params.viscosity_iterations, 1, 16);
    const float strength = std::clamp(params.viscosity * dt, 0.0f, 0.45f);
    if (strength <= 0.0f) {
        return;
    }

    auto smoothField = [&](std::vector<float>& field, int sx, int sy, int sz) {
        std::vector<float> scratch(field.size(), 0.0f);
        const int64_t count = static_cast<int64_t>(field.size());
        auto idx = [sx, sy](int i, int j, int k) {
            return static_cast<size_t>(i + j * sx + k * sx * sy);
        };

        // Keep this serial for the CPU reference path. MSVC OpenMP rejects
        // num_threads() over lambda-captured locals here, and prior OMP regions
        // in this solver already hit vcomp stability issues on large domains.
        for (int it = 0; it < iterations; ++it) {
            for (int64_t raw = 0; raw < count; ++raw) {
                const int i = static_cast<int>(raw % sx);
                const int j = static_cast<int>((raw / sx) % sy);
                const int k = static_cast<int>(raw / (static_cast<int64_t>(sx) * sy));
                const float center = field[static_cast<size_t>(raw)];
                float sum = 0.0f;
                int n = 0;
                if (i > 0)      { sum += field[idx(i - 1, j, k)]; ++n; }
                if (i + 1 < sx) { sum += field[idx(i + 1, j, k)]; ++n; }
                if (j > 0)      { sum += field[idx(i, j - 1, k)]; ++n; }
                if (j + 1 < sy) { sum += field[idx(i, j + 1, k)]; ++n; }
                if (k > 0)      { sum += field[idx(i, j, k - 1)]; ++n; }
                if (k + 1 < sz) { sum += field[idx(i, j, k + 1)]; ++n; }
                const float avg = (n > 0) ? (sum / static_cast<float>(n)) : center;
                scratch[static_cast<size_t>(raw)] = center + (avg - center) * strength;
            }
            field.swap(scratch);
        }
    };

    smoothField(grid.vel_x, grid.nx + 1, grid.ny,     grid.nz);
    smoothField(grid.vel_y, grid.nx,     grid.ny + 1, grid.nz);
    smoothField(grid.vel_z, grid.nx,     grid.ny,     grid.nz + 1);
}

// Step particle positions through the grid velocity field, subdivided to
// keep each move under one cell (CFL).
static int advectParticles(FluidParticles& parts,
                            const FluidSim::FluidGrid& grid,
                            float dt,
                            const APICSolverParams& params,
                            float max_velocity,
                            float wall_damping,
                            float cfl,
                            int max_substeps) {
    const float h    = grid.voxel_size;
    const float invH = 1.0f / h;

    const int thread_count = solverThreadCount(params);
    const bool parallel = shouldParallelParticles(params, parts.size());

    // vmax reduction runs serially — even at 1M particles this is ~1ms,
    // below OMP team-creation overhead. The previous nested
    // `omp parallel num_threads(N) { omp for }` form deadlocks MSVC vcomp
    // at large particle counts (same bug as the pressure projection fix);
    // MSVC OpenMP 2.0 doesn't support `reduction(max:)` so there's no
    // clean single-directive parallel form here. Sequential is safe and
    // fast enough — the actual advect loop below is still parallel.
    float vmax = 0.0f;
    for (int64_t raw_i = 0; raw_i < static_cast<int64_t>(parts.velocity.size()); ++raw_i) {
        const Vec3& v = parts.velocity[static_cast<size_t>(raw_i)];
        vmax = std::max(vmax, std::max(std::abs(v.x), std::max(std::abs(v.y), std::abs(v.z))));
    }
    int substeps = 1;
    if (vmax > 1e-6f) {
        substeps = static_cast<int>(std::ceil(vmax * dt * invH / cfl));
        substeps = std::max(1, std::min(substeps, max_substeps));
    }
    float sub_dt = dt / substeps;

    Vec3 gmin, gmax;
    grid.getWorldBounds(gmin, gmax);
    // Keep particles a hair inside the domain to avoid sampling outside.
    const float pad = 0.01f * h;
    Vec3 lo = gmin + Vec3(pad, pad, pad);
    Vec3 hi = gmax - Vec3(pad, pad, pad);

    for (int s = 0; s < substeps; ++s) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(parallel)
#endif
        for (int64_t raw_pi = 0; raw_pi < static_cast<int64_t>(parts.size()); ++raw_pi) {
            const size_t pi = static_cast<size_t>(raw_pi);
            // RK2 midpoint using grid velocity (more stable than forward Euler
            // for free-surface flow, negligible cost on CPU).
            Vec3 p0 = parts.position[pi];
            Vec3 g0 = (p0 - grid.origin) * invH;
            Vec3 v0(sampleMACComponent(grid, 0, g0),
                    sampleMACComponent(grid, 1, g0),
                    sampleMACComponent(grid, 2, g0));
            Vec3 pmid = p0 + v0 * (0.5f * sub_dt);
            Vec3 gm   = (pmid - grid.origin) * invH;
            Vec3 vm(sampleMACComponent(grid, 0, gm),
                    sampleMACComponent(grid, 1, gm),
                    sampleMACComponent(grid, 2, gm));
            Vec3 pn = p0 + vm * sub_dt;

            // Clamp to domain and dissipate wall-normal velocity. Without this
            // early preview guard, particles stuck at the floor can feed energy
            // back into the next pressure solve and explode outward.
            Vec3& pv = parts.velocity[pi];
            if (pn.x < lo.x) { pn.x = lo.x; pv.x = std::abs(pv.x) * wall_damping; }
            if (pn.x > hi.x) { pn.x = hi.x; pv.x = -std::abs(pv.x) * wall_damping; }
            if (pn.y < lo.y) { pn.y = lo.y; pv.y = std::abs(pv.y) * wall_damping; }
            if (pn.y > hi.y) { pn.y = hi.y; pv.y = -std::abs(pv.y) * wall_damping; }
            if (pn.z < lo.z) { pn.z = lo.z; pv.z = std::abs(pv.z) * wall_damping; }
            if (pn.z > hi.z) { pn.z = hi.z; pv.z = -std::abs(pv.z) * wall_damping; }

            parts.position[pi] = pn;
        }
    }

    // Velocity cap for sanity during early development.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(parallel)
#endif
    for (int64_t raw_i = 0; raw_i < static_cast<int64_t>(parts.velocity.size()); ++raw_i) {
        Vec3& v = parts.velocity[static_cast<size_t>(raw_i)];
        float speed = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        if (speed > max_velocity) v = v * (max_velocity / speed);
    }

    return substeps;
}

// Per-cell particle redistribution. Maintains the seed-time particle count
// invariant: cells whose population has drifted out of [min_per, max_per]
// are pulled back toward `target`. Without this, FLIP-style energy injection
// and pressure-driven motion eventually pile particles up in low-pressure
// pockets and starve high-pressure ones, producing visible density bands
// and apparent volume loss.
//
// Removal is in-place (write-cursor compaction; no per-particle erase).
// Addition samples grid velocity at the new particle's position — this is
// the post-projection (incompressible) field, so injected particles do not
// re-introduce divergence.
static void redistributeParticles(FluidParticles& parts,
                                  const FluidSim::FluidGrid& grid,
                                  const APICSolverParams& params,
                                  uint32_t step_seed) {
    if (!params.reseed_enabled || parts.empty()) return;
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) return;

    const int  ppc    = std::max(1, params.particles_per_cell);
    const int  target = (params.reseed_target_per_cell > 0)
        ? params.reseed_target_per_cell
        : ppc;
    const int  min_per = std::clamp(params.reseed_min_per_cell, 1, target);
    const int  max_per = std::max(target + 1, params.reseed_max_per_cell);

    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float h = grid.voxel_size;
    const float invH = 1.0f / h;
    const std::size_t total = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz);

    // 1. Map each particle to its containing cell (or -1 if outside / solid).
    //    Function-static buffers — per-call vector<int> would stall under
    //    contention (see feedback_persistent_per_step_buffers.md).
    static std::vector<int> particle_cell_buf;
    static std::vector<int> cell_count_buf;
    static std::vector<int> cell_cursor_buf;
    static std::vector<uint8_t> remove_mask_buf;

    particle_cell_buf.assign(parts.size(), -1);
    if (cell_count_buf.size() < total) cell_count_buf.assign(total, 0);
    else std::fill(cell_count_buf.begin(), cell_count_buf.begin() + total, 0);
    if (cell_cursor_buf.size() < total) cell_cursor_buf.assign(total, 0);
    else std::fill(cell_cursor_buf.begin(), cell_cursor_buf.begin() + total, 0);
    remove_mask_buf.assign(parts.size(), 0u);

    for (std::size_t pi = 0; pi < parts.size(); ++pi) {
        const Vec3 gp = (parts.position[pi] - grid.origin) * invH;
        const int i = static_cast<int>(std::floor(gp.x));
        const int j = static_cast<int>(std::floor(gp.y));
        const int k = static_cast<int>(std::floor(gp.z));
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
        const std::size_t c = grid.cellIndex(i, j, k);
        if (grid.solid[c]) continue; // particle is inside a solid; wall path handles it
        particle_cell_buf[static_cast<int>(pi)] = static_cast<int>(c);
        ++cell_count_buf[c];
    }

    // 2. Mark surplus particles for removal. Walk in particle order; within
    //    each over-populated cell (count > max_per), keep the first `max_per`
    //    and flag the rest. (Order-agnostic — particle SoA has no persistent
    //    identity beyond the current step.)
    //
    //    Trim threshold is `max_per`, NOT `target`. The previous code trimmed
    //    over-populated cells all the way back to `target` (= particles_per_cell
    //    = 8 by default), which made accumulating fluid (e.g. water piling at
    //    the domain floor) literally disappear: as particles compressed and
    //    hit count > max_per, the reseed culled 22 out of 30 in one tick.
    //    Trimming only to max_per preserves natural accumulation up to that
    //    cap (16 by default) while still removing pathological spikes.
    int removed = 0;
    for (std::size_t pi = 0; pi < parts.size(); ++pi) {
        const int c = particle_cell_buf[static_cast<int>(pi)];
        if (c < 0) continue;
        if (cell_count_buf[c] <= max_per) continue;
        const int seen = cell_cursor_buf[c]++;
        if (seen >= max_per) {
            remove_mask_buf[pi] = 1u;
            ++removed;
        }
    }

    // 3. Compact in place. Write cursor advances only for kept particles.
    if (removed > 0) {
        std::size_t write = 0;
        for (std::size_t pi = 0; pi < parts.size(); ++pi) {
            if (remove_mask_buf[pi]) continue;
            if (write != pi) {
                parts.position[write] = parts.position[pi];
                parts.velocity[write] = parts.velocity[pi];
                parts.affine[write]   = parts.affine[pi];
                parts.flags[write]    = parts.flags[pi];
            }
            ++write;
        }
        parts.position.resize(write);
        parts.velocity.resize(write);
        parts.affine.resize(write);
        parts.flags.resize(write);
    }

    // 4. Top up starved fluid cells. An empty cell (count == 0) is treated
    //    as AIR and left untouched — adding particles there would create
    //    mass from nothing. New particles inherit grid-sampled velocity, no
    //    affine history (AffineC default = zero); the next P2G rebuilds C.
    std::mt19937 rng(step_seed ^ 0x9E3779B9u);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = grid.cellIndex(i, j, k);
                if (grid.solid[c]) continue;
                const int count = cell_count_buf[c];
                if (count == 0) continue;            // air — do not seed
                if (count >= min_per) continue;       // healthy

                // To prevent stray spray/droplets from multiplying exponentially into dense water chunks:
                // We only reseed if there is a minimum collective particle presence in the 3x3x3 neighborhood,
                // or if at least one direct neighbor is a healthy fluid cell.
                int neighborhood_sum = count;
                bool has_healthy_neighbor = false;
                for (int dk = -1; dk <= 1; ++dk) {
                    for (int dj = -1; dj <= 1; ++dj) {
                        for (int di = -1; di <= 1; ++di) {
                            if (di == 0 && dj == 0 && dk == 0) continue;
                            int ni = i + di, nj = j + dj, nk = k + dk;
                            if (ni >= 0 && ni < nx && nj >= 0 && nj < ny && nk >= 0 && nk < nz) {
                                size_t nc = grid.cellIndex(ni, nj, nk);
                                int n_count = cell_count_buf[nc];
                                neighborhood_sum += n_count;
                                if (n_count >= min_per) {
                                    has_healthy_neighbor = true;
                                }
                            }
                        }
                    }
                }

                // If this is an isolated spray cell (fewer than min_per + 2 particles in 3x3x3 neighborhood)
                // and has no healthy neighbors, bypass reseeding (treat as isolated spray to preserve mass!).
                if (neighborhood_sum < std::max(4, min_per + 2) && !has_healthy_neighbor) {
                    continue;
                }

                const int need = target - count;
                if (parts.size() >= params.max_particles) {
                    continue;
                }
                const int actual_need = std::min<int>(need, static_cast<int>(params.max_particles - parts.size()));
                if (actual_need <= 0) continue;

                const Vec3 cell_min = grid.origin + Vec3(static_cast<float>(i) * h,
                                                          static_cast<float>(j) * h,
                                                          static_cast<float>(k) * h);
                for (int n = 0; n < actual_need; ++n) {
                    const Vec3 p = cell_min + Vec3(dist01(rng), dist01(rng), dist01(rng)) * h;
                    const Vec3 v = grid.sampleVelocity(p);
                    parts.emit(p, v);
                }
            }
        }
    }
}

void step(FluidParticles& particles,
          FluidSim::FluidGrid& grid,
          const APICSolverParams& params,
          float dt,
          const SimulationForceFieldSnapshot* forces,
          float time_seconds,
          APICSolverStats* stats) {
    APICSolverStats local_stats{};
    APICSolverStats& out_stats = stats ? *stats : local_stats;
    out_stats = APICSolverStats{};
    out_stats.cpu_threads = solverThreadCount(params);
    out_stats.particle_count = particles.size();
    out_stats.grid_cell_count = static_cast<size_t>(grid.nx) *
                                static_cast<size_t>(grid.ny) *
                                static_cast<size_t>(grid.nz);

    if (particles.empty() || dt <= 0.0f) return;

#ifdef APIC_DEBUG_SKIP_STEP
    // Solver disabled for bisect — sim ticks, fluid stage no-ops.
    return;
#endif

    const auto total_begin = SolverClock::now();

    auto clampVelocity = [&](Vec3& v) {
        const float speed = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        if (speed > params.max_velocity && speed > 1e-6f) {
            v = v * (params.max_velocity / speed);
        }
    };
    const int thread_count = solverThreadCount(params);
    const bool particle_parallel = shouldParallelParticles(params, particles.size());

    // 1. External forces on particles. Force fields are sampled on the APIC
    // particles, so emitters/fields steer the true liquid mass before P2G.
    auto stage_begin = SolverClock::now();
    if (!params.external_forces_preintegrated) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(particle_parallel)
#endif
        for (int64_t raw_i = 0; raw_i < static_cast<int64_t>(particles.velocity.size()); ++raw_i) {
            const size_t pi = static_cast<size_t>(raw_i);
            Vec3& v = particles.velocity[pi];
            Vec3 acceleration = params.gravity;
            if (forces && !forces->empty()) {
                acceleration = acceleration +
                    forces->evaluateAt(particles.position[pi],
                                       time_seconds,
                                       v,
                                       SimulationSystemKind::Fluid);
            }
            v = v + acceleration * dt;
            clampVelocity(v);
        }
    }
    auto stage_end = SolverClock::now();
    out_stats.forces_ms = elapsedMs(stage_begin, stage_end);

    // 2. Particle -> grid (APIC scatter)
    stage_begin = SolverClock::now();
    if (!params.p2g_precomputed) {
        particleToGrid(particles, grid, params);
    }
    stage_end = SolverClock::now();
    out_stats.p2g_ms = elapsedMs(stage_begin, stage_end);
    out_stats.p2g_on_gpu = params.p2g_precomputed;

    // 3. Solid boundary enforcement + viscosity.
    //    Skipped in GPU split-step's second call (pressure_g2p_precomputed=true)
    //    because these were already run in the first call (stop_after_viscosity=true).
    if (!params.pressure_g2p_precomputed) {
        stage_begin = SolverClock::now();
        enforceSolidBoundaries(grid, params);
        stage_end = SolverClock::now();
        out_stats.boundary_ms = elapsedMs(stage_begin, stage_end);

        stage_begin = SolverClock::now();
        applyViscosity(grid, params, dt);
        enforceSolidBoundaries(grid, params);
        stage_end = SolverClock::now();
        out_stats.viscosity_ms = elapsedMs(stage_begin, stage_end);
    }

    // GPU split-step call 1: return after boundary+viscosity so caller can
    // run GPU pressure → GPU G2P before the second call does advect+reseed.
    if (params.stop_after_viscosity) {
        out_stats.total_ms = elapsedMs(total_begin, SolverClock::now());
        return;
    }

    // 4–6. FLIP snapshot + pressure projection + G2P.
    //       When pressure_g2p_precomputed is set, the GPU path has already
    //       run these three stages and downloaded updated particle velocities
    //       and affine matrices to CPU. Skip them here; advect + reseed still
    //       run on CPU below using the GPU-written velocity data.
    static std::vector<float> vel_x_pre_buf;
    static std::vector<float> vel_y_pre_buf;
    static std::vector<float> vel_z_pre_buf;

    stage_begin = SolverClock::now();
    if (!params.pressure_g2p_precomputed) {
        // 4. Snapshot the MAC velocity field BEFORE pressure projection (FLIP).
        const bool want_flip = params.flip_blend > 0.0f &&
                               params.free_surface;
        if (want_flip) {
            if (vel_x_pre_buf.size() < grid.vel_x.size()) vel_x_pre_buf.resize(grid.vel_x.size());
            if (vel_y_pre_buf.size() < grid.vel_y.size()) vel_y_pre_buf.resize(grid.vel_y.size());
            if (vel_z_pre_buf.size() < grid.vel_z.size()) vel_z_pre_buf.resize(grid.vel_z.size());
            std::copy(grid.vel_x.begin(), grid.vel_x.end(), vel_x_pre_buf.begin());
            std::copy(grid.vel_y.begin(), grid.vel_y.end(), vel_y_pre_buf.begin());
            std::copy(grid.vel_z.begin(), grid.vel_z.end(), vel_z_pre_buf.begin());
        }

        // 5. Pressure projection (incompressibility).
        auto pressure_begin = SolverClock::now();
        if (params.free_surface) {
            out_stats.active_fluid_cells = projectPressureFreeSurface(particles,
                                                                      grid,
                                                                      params,
                                                                      dt);
        } else {
            GridFluid::SolverParams pp{};
            pp.pressure_iterations = params.pressure_iterations;
            pp.sor_omega           = params.sor_omega;
            pp.boundary            = GridFluid::Boundary::Closed;
            GridFluid::projectPressure(grid, pp, dt);
            out_stats.active_fluid_cells = out_stats.grid_cell_count;
        }
        out_stats.pressure_ms = elapsedMs(pressure_begin, SolverClock::now());

        if (!params.skip_g2p) {
            // 6. Grid -> particle (APIC affine + PIC/FLIP linear blend + friction)
            gridToParticle(particles,
                           grid,
                           params,
                           dt,
                           want_flip ? vel_x_pre_buf.data() : nullptr,
                           want_flip ? vel_y_pre_buf.data() : nullptr,
                           want_flip ? vel_z_pre_buf.data() : nullptr);
        } else {
            // Expose FLIP snapshot for GPU G2P. Copy static buffers to
            // namespace-level g_ statics so the caller can access them after
            // this call returns. vel_x_pre_buf is valid only when want_flip.
            g_flip_snap_valid = want_flip;
            if (want_flip) {
                g_flip_snap_x.assign(vel_x_pre_buf.begin(),
                                     vel_x_pre_buf.begin() + grid.vel_x.size());
                g_flip_snap_y.assign(vel_y_pre_buf.begin(),
                                     vel_y_pre_buf.begin() + grid.vel_y.size());
                g_flip_snap_z.assign(vel_z_pre_buf.begin(),
                                     vel_z_pre_buf.begin() + grid.vel_z.size());
            } else {
                g_flip_snap_x.clear(); g_flip_snap_y.clear(); g_flip_snap_z.clear();
            }
            out_stats.g2p_ms = 0.0f;
            // Caller will do GPU G2P and then call Fluid::step with
            // pressure_g2p_precomputed=true for the tail (air_drag+advect+reseed).
            out_stats.total_ms = elapsedMs(total_begin, SolverClock::now());
            return;
        }
        // When skip_g2p=true early return is above. Normal flow continues below.
    } else {
        // pressure_g2p_precomputed: GPU already handled pressure+G2P,
        // particle velocities+affine are downloaded — only tail stages remain.
    }

    // 6b. Air drag for spray/droplet particles. A particle whose containing
    //     cell has fewer than `reseed_min_per_cell` neighbours is treated as
    //     "in air"; quadratic drag F = -k|v|v is integrated implicitly via
    //     v *= 1/(1 + k|v|dt) — unconditionally stable, no inner iterate.
    //     Bulk particles are skipped (their dissipation comes from
    //     internal_friction inside gridToParticle).
    if (params.air_drag > 0.0f && dt > 0.0f) {
        const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
        const float invH = (grid.voxel_size > 1e-6f) ? (1.0f / grid.voxel_size) : 0.0f;
        const int air_threshold = std::max(1, params.reseed_min_per_cell);
        const float air_k = params.air_drag;

        // Build per-cell particle counts (fresh from post-G2P positions —
        // pre-advect). Function-static to avoid heap stalls under contention.
        static std::vector<int> air_cell_count_buf;
        const std::size_t total = static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny) *
                                   static_cast<std::size_t>(nz);
        if (air_cell_count_buf.size() < total) air_cell_count_buf.assign(total, 0);
        else std::fill(air_cell_count_buf.begin(), air_cell_count_buf.begin() + total, 0);

        for (std::size_t pi = 0; pi < particles.size(); ++pi) {
            const Vec3 gp = (particles.position[pi] - grid.origin) * invH;
            const int i = static_cast<int>(std::floor(gp.x));
            const int j = static_cast<int>(std::floor(gp.y));
            const int k = static_cast<int>(std::floor(gp.z));
            if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
            ++air_cell_count_buf[grid.cellIndex(i, j, k)];
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(particle_parallel)
#endif
        for (int64_t raw_i = 0; raw_i < static_cast<int64_t>(particles.size()); ++raw_i) {
            const std::size_t pi = static_cast<std::size_t>(raw_i);
            const Vec3 gp = (particles.position[pi] - grid.origin) * invH;
            const int i = static_cast<int>(std::floor(gp.x));
            const int j = static_cast<int>(std::floor(gp.y));
            const int k = static_cast<int>(std::floor(gp.z));
            if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
            const int count = air_cell_count_buf[grid.cellIndex(i, j, k)];
            if (count >= air_threshold) continue; // bulk fluid — skip air drag

            Vec3& v = particles.velocity[pi];
            const float speed_sq = v.x * v.x + v.y * v.y + v.z * v.z;
            if (speed_sq < 1e-8f) continue;
            const float speed = std::sqrt(speed_sq);
            const float decay = 1.0f / (1.0f + air_k * speed * dt);
            v = v * decay;
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(particle_parallel)
#endif
    for (int64_t raw_i = 0; raw_i < static_cast<int64_t>(particles.velocity.size()); ++raw_i) {
        Vec3& v = particles.velocity[static_cast<size_t>(raw_i)];
        v = v * std::clamp(params.velocity_damping, 0.0f, 1.0f);
        clampVelocity(v);
    }
    stage_end = SolverClock::now();
    out_stats.g2p_ms = elapsedMs(stage_begin, stage_end); // includes air drag + damping

    // 7. Advect particle positions
    stage_begin = SolverClock::now();
    out_stats.advect_substeps = advectParticles(particles,
                                                grid,
                                                dt,
                                                params,
                                                params.max_velocity,
                                                params.wall_damping,
                                                params.cfl,
                                                params.max_substeps);
    stage_end = SolverClock::now();
    out_stats.advect_ms = elapsedMs(stage_begin, stage_end);

    // 8. Per-cell redistribution. Cull over-populated cells and top up
    //    starved ones so the seed-time particle density stays bounded. Uses
    //    the post-advect positions and the post-projection grid (sampled by
    //    new particles for their initial velocity).
    const uint32_t reseed_seed =
        static_cast<uint32_t>(out_stats.particle_count) * 2654435761u ^
        static_cast<uint32_t>(std::llround(time_seconds * 1000.0));
    redistributeParticles(particles, grid, params, reseed_seed);
    out_stats.particle_count = particles.size();

    out_stats.total_ms = elapsedMs(total_begin, stage_end);
}

} // namespace Fluid
} // namespace RayTrophiSim
