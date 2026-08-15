/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          APICFluidSolver.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 */

#include "Fluid/APICFluidSolver.h"
#include "Fluid/SubstanceTag.h"   // kSubstanceUntagged (solid-phase resolution)
#include "GridFluidSolver.h"
#include "SimulationWorld.h"
#include "ForceField.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
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
// The post-P2G MAC velocity field, published by every Fluid::step that runs the
// transfer. This is the FLIP reference state: the delta the particles receive is
// (post-pressure grid - this), so every grid-side change in between (solid
// boundaries, viscosity, pressure) reaches them. The GPU split-step uploads THIS,
// not grid.vel_*, which by upload time already carries the viscous solve.
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

    // Stratified placement. Pure uniform jitter (one random point per particle
    // anywhere in the cell) clumps like Poisson noise: even though the integer
    // count per cell is exactly particles_per_cell, the sub-cell clustering
    // means the first advect step redistributes mass and some cells spike above
    // target — the density-targeted pressure projection then expels them, so a
    // freshly seeded "resting tank" gushes upward on the first frames. Splitting
    // each cell into a regular sub-lattice (sub^3 slots) with one jittered point
    // per occupied slot gives blue-noise-like uniform coverage, so the tank
    // starts near hydrostatic equilibrium and the surface stays put.
    const int sub = std::max(1, static_cast<int>(std::ceil(std::cbrt(
        static_cast<double>(std::max(1, particles_per_cell))))));
    const float inv_sub = 1.0f / static_cast<float>(sub);

    // Y-major iteration (j outer). When max_new_particles truncates the seed
    // (budget too small to fill the whole region), the cells that go unseeded
    // are the TOP layers — leaving complete horizontal layers from the floor up
    // plus at most one partial top layer. This is the graceful failure for a
    // "resting tank": the fill level just drops to whatever the budget affords,
    // never the vertical -Z wall slab a Z-major fill would leave.
    size_t emitted = 0;
    for (int j = j0; j <= j1; ++j)
    for (int k = k0; k <= k1; ++k)
    for (int i = i0; i <= i1; ++i) {
        if (grid.isSolid(i, j, k)) continue;
        Vec3 cellMin = grid.origin + Vec3(i * h, j * h, k * h);
        int placed = 0;
        for (int sz = 0; sz < sub && placed < particles_per_cell; ++sz)
        for (int sy = 0; sy < sub && placed < particles_per_cell; ++sy)
        for (int sx = 0; sx < sub && placed < particles_per_cell; ++sx) {
            if (emitted >= max_new_particles) return;
            // Sub-cell base corner + jitter confined to that sub-cell.
            Vec3 frac(
                (static_cast<float>(sx) + U(rng)) * inv_sub,
                (static_cast<float>(sy) + U(rng)) * inv_sub,
                (static_cast<float>(sz) + U(rng)) * inv_sub);
            particles.emit(cellMin + frac * h, Vec3(0, 0, 0));
            ++emitted;
            ++placed;
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
                           const float* vel_z_pre,
                           // Per-particle "this parcel is solid phase" mask, or
                           // null when the domain declares no solid substance.
                           const uint8_t* solid_particle) {
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
        // ── Solid phase: the grid does not tell this parcel how to move ──────
        // ★★★ A solid parcel keeps the velocity it entered the step with (its
        // own gravity and force-field integration). Gathering the grid here
        // would read the field the parcel ITSELF imposed through solid_vel and
        // then blend the surrounding liquid into it, so the chunk would be
        // carried by the flow it is supposed to obstruct — and it would do so
        // at whatever rate the interpolation happened to give, which is not a
        // force anyone authored. Cohesive, dragged clusters are the NEXT tier
        // and belong to Jolt; see Fluid::SubstancePhase.
        //
        // ★ The affine matrix is left alone rather than zeroed: it is APIC's
        // velocity gradient, and a parcel that never gathers never accumulates
        // one. Zeroing it every step would be the same value at more cost.
        if (solid_particle && solid_particle[pi]) continue;
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
                                         float dt,
                                         size_t& out_sealed_pockets,
                                         size_t& out_sealed_cells,
                                         size_t& out_interior_cells) {
    out_sealed_pockets = 0;
    out_sealed_cells = 0;
    out_interior_cells = 0;
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

    // Open domains let fluid drain through the bounding walls: an out-of-grid
    // neighbour is then treated as AIR (Dirichlet p=0 → outflow) instead of a
    // SOLID wall, so the projection no longer cancels wall-normal velocity and
    // the liquid actually leaves (paired with the advection-step outflow cull).
    // Closed seals every wall. Periodic connects each wall to the OPPOSITE one:
    // an out-of-grid neighbour wraps to the far side (the seam becomes interior),
    // so the Laplacian, the face weights and the gradient update all couple
    // across it — the advection wrap then actually fires because the pressure
    // solve no longer dams the fluid at the boundary (without this, Periodic was
    // indistinguishable from Closed).
    const bool open_walls = (params.boundary == APICSolverParams::BoundaryMode::Open);
    const bool periodic   = (params.boundary == APICSolverParams::BoundaryMode::Periodic);
    // Wrap a (possibly out-of-range) index into [0,n) — used only on the periodic
    // path. For non-periodic these helpers are never reached with out-of-range
    // arguments (the isFluid guard short-circuits first), so the closed/open
    // behaviour stays bit-identical.
    auto wrapIdx = [](int v, int n) { v %= n; return v < 0 ? v + n : v; };
    auto cIdx = [&](int i, int j, int k) -> size_t {
        if (periodic) { i = wrapIdx(i, nx); j = wrapIdx(j, ny); k = wrapIdx(k, nz); }
        return grid.cellIndex(i, j, k);
    };
    auto isFluid = [&](int i, int j, int k) {
        if (periodic) { i = wrapIdx(i, nx); j = wrapIdx(j, ny); k = wrapIdx(k, nz); }
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return false;
        return cell[grid.cellIndex(i, j, k)] == CELL_FLUID;
    };
    // ── Variational solid coupling ───────────────────────────────────────────
    // Unified MAC-face open weight in [0,1]. With variational_solids on it reads
    // the analytic fractional weights (sub-grid solid boundary); otherwise it
    // reduces EXACTLY to the binary "blocked if an adjacent cell is solid" test,
    // so the flag-off path is bit-identical to the old projection. Domain-boundary
    // faces follow the wall mode (closed = 0, open = 1 Dirichlet). svX/Y/Z give the
    // solid's normal velocity at a face so a MOVING collider enters the divergence
    // RHS (real splash); 0 in binary mode and for static colliders.
    const bool use_var = params.variational_solids &&
        grid.u_weight.size() == grid.vel_x.size() &&
        grid.v_weight.size() == grid.vel_y.size() &&
        grid.w_weight.size() == grid.vel_z.size();
    const bool has_svel = grid.solid_vel.size() == grid.solid.size();
    // A periodic boundary face (i==0 ≡ i==nx) is a genuine interior face linking
    // cell (n-1) and cell (0); it stays open unless either wrapped cell is solid.
    // (Variational sub-grid weights aren't tracked across the seam, so it uses the
    // binary open/closed test there — periodic + cut-cell solids on the seam is an
    // edge case not worth a second weight array.)
    auto fWx = [&](int i, int j, int k) -> float {
        if (i <= 0 || i >= nx)
            return periodic ? ((grid.solid[grid.cellIndex(nx - 1, j, k)] || grid.solid[grid.cellIndex(0, j, k)]) ? 0.0f : 1.0f)
                            : (open_walls ? 1.0f : 0.0f);
        if (use_var) return FluidSim::FluidGrid::weightToFloat(grid.u_weight[grid.velXIndex(i, j, k)]);
        return (grid.solid[grid.cellIndex(i - 1, j, k)] || grid.solid[grid.cellIndex(i, j, k)]) ? 0.0f : 1.0f;
    };
    auto fWy = [&](int i, int j, int k) -> float {
        if (j <= 0 || j >= ny)
            return periodic ? ((grid.solid[grid.cellIndex(i, ny - 1, k)] || grid.solid[grid.cellIndex(i, 0, k)]) ? 0.0f : 1.0f)
                            : (open_walls ? 1.0f : 0.0f);
        if (use_var) return FluidSim::FluidGrid::weightToFloat(grid.v_weight[grid.velYIndex(i, j, k)]);
        return (grid.solid[grid.cellIndex(i, j - 1, k)] || grid.solid[grid.cellIndex(i, j, k)]) ? 0.0f : 1.0f;
    };
    auto fWz = [&](int i, int j, int k) -> float {
        if (k <= 0 || k >= nz)
            return periodic ? ((grid.solid[grid.cellIndex(i, j, nz - 1)] || grid.solid[grid.cellIndex(i, j, 0)]) ? 0.0f : 1.0f)
                            : (open_walls ? 1.0f : 0.0f);
        if (use_var) return FluidSim::FluidGrid::weightToFloat(grid.w_weight[grid.velZIndex(i, j, k)]);
        return (grid.solid[grid.cellIndex(i, j, k - 1)] || grid.solid[grid.cellIndex(i, j, k)]) ? 0.0f : 1.0f;
    };
    auto svX = [&](int i, int j, int k) -> float {
        if (!use_var || !has_svel) return 0.0f;
        if (i - 1 >= 0 && grid.solid[grid.cellIndex(i - 1, j, k)]) return grid.solid_vel[grid.cellIndex(i - 1, j, k)].x;
        if (i < nx && grid.solid[grid.cellIndex(i, j, k)])         return grid.solid_vel[grid.cellIndex(i, j, k)].x;
        return 0.0f;
    };
    auto svY = [&](int i, int j, int k) -> float {
        if (!use_var || !has_svel) return 0.0f;
        if (j - 1 >= 0 && grid.solid[grid.cellIndex(i, j - 1, k)]) return grid.solid_vel[grid.cellIndex(i, j - 1, k)].y;
        if (j < ny && grid.solid[grid.cellIndex(i, j, k)])         return grid.solid_vel[grid.cellIndex(i, j, k)].y;
        return 0.0f;
    };
    auto svZ = [&](int i, int j, int k) -> float {
        if (!use_var || !has_svel) return 0.0f;
        if (k - 1 >= 0 && grid.solid[grid.cellIndex(i, j, k - 1)]) return grid.solid_vel[grid.cellIndex(i, j, k - 1)].z;
        if (k < nz && grid.solid[grid.cellIndex(i, j, k)])         return grid.solid_vel[grid.cellIndex(i, j, k)].z;
        return 0.0f;
    };

    // ── Ghost-fluid free surface ─────────────────────────────────────────────
    // Cheap per-step liquid level set (union of particle balls). phi < 0 inside
    // the fluid, > 0 in air; the zero-crossing is the sub-cell surface. Used to
    // place the p=0 boundary at the real surface instead of the air cell centre.
    const float gfm_big = 3.0f * h;
    const bool use_gfm = params.ghost_fluid_surface;
    if (use_gfm) {
        if (grid.fluid_phi.size() != total) grid.fluid_phi.assign(total, gfm_big);
        std::fill(grid.fluid_phi.begin(), grid.fluid_phi.begin() + total, gfm_big);
        const float rball = std::max(0.1f * h, params.surface_ball_radius * h);
        // Serial scatter (3x3x3 per particle): write-conflicting min, cheap vs PCG.
        for (size_t pi = 0; pi < parts.size(); ++pi) {
            const Vec3 pos = parts.position[pi];
            const Vec3 gp = (pos - grid.origin) * invH;
            const int ci = static_cast<int>(std::floor(gp.x));
            const int cj = static_cast<int>(std::floor(gp.y));
            const int ck = static_cast<int>(std::floor(gp.z));
            for (int dk = -1; dk <= 1; ++dk)
            for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di) {
                const int ni = ci + di, nj = cj + dj, nk = ck + dk;
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) continue;
                const Vec3 cc = grid.origin + Vec3((ni + 0.5f) * h, (nj + 0.5f) * h, (nk + 0.5f) * h);
                const Vec3 d = cc - pos;
                const float dist = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z) - rball;
                float& ph = grid.fluid_phi[grid.cellIndex(ni, nj, nk)];
                if (dist < ph) ph = dist;
            }
        }
    }
    const bool gfm_active = use_gfm && grid.fluid_phi.size() == total;
    // Sub-cell fluid fraction along a fluid→air face (1 = surface at air centre).
    // theta = phi_f / (phi_f - phi_a), valid only when phi_f < 0 < phi_a; clamped
    // for stability. Returns 1 (= plain p=0 Dirichlet) when GFM can't apply.
    auto airTheta = [&](size_t c_fluid, int ni, int nj, int nk) -> float {
        if (!gfm_active) return 1.0f;
        if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) return 1.0f;
        const size_t n = grid.cellIndex(ni, nj, nk);
        if (grid.solid[n]) return 1.0f;
        const float pf = grid.fluid_phi[c_fluid], pa = grid.fluid_phi[n];
        if (!(pf < 0.0f && pa > 0.0f)) return 1.0f;
        const float t = pf / (pf - pa);
        return t < 0.1f ? 0.1f : (t > 1.0f ? 1.0f : t);
    };

    // Periodic seam merge: the low face (index 0) and the high face (index N)
    // of each axis are the SAME physical MAC face, but P2G fills them
    // independently from particles near each wall. Collapse them to a single
    // value before building divergence — otherwise the two sides feed
    // inconsistent normal velocities into the solve (mass leak at the seam) and
    // the gradient update can't keep them equal. Averaging is the cheap, mass-
    // symmetric merge; the projection below then writes both faces identically.
    if (periodic) {
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
                const size_t a = grid.velXIndex(0,  j, k);
                const size_t b = grid.velXIndex(nx, j, k);
                const float m = 0.5f * (grid.vel_x[a] + grid.vel_x[b]);
                grid.vel_x[a] = grid.vel_x[b] = m;
            }
        for (int k = 0; k < nz; ++k)
            for (int i = 0; i < nx; ++i) {
                const size_t a = grid.velYIndex(i, 0,  k);
                const size_t b = grid.velYIndex(i, ny, k);
                const float m = 0.5f * (grid.vel_y[a] + grid.vel_y[b]);
                grid.vel_y[a] = grid.vel_y[b] = m;
            }
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t a = grid.velZIndex(i, j, 0);
                const size_t b = grid.velZIndex(i, j, nz);
                const float m = 0.5f * (grid.vel_z[a] + grid.vel_z[b]);
                grid.vel_z[a] = grid.vel_z[b] = m;
            }
    }

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

        // Weighted face velocities: the open fraction carries the fluid velocity,
        // the closed fraction carries the solid's velocity (0 for a static wall →
        // identical to the old "zero the solid face" behaviour; non-zero for a
        // MOVING collider → the wall pushes the fluid through the pressure solve).
        const float wxl = fWx(i,     j, k), wxh = fWx(i + 1, j, k);
        const float wyl = fWy(i, j,     k), wyh = fWy(i, j + 1, k);
        const float wzl = fWz(i, j, k),     wzh = fWz(i, j, k + 1);
        const float Uxl = wxl * vx_lo + (1.0f - wxl) * svX(i,     j, k);
        const float Uxh = wxh * vx_hi + (1.0f - wxh) * svX(i + 1, j, k);
        const float Uyl = wyl * vy_lo + (1.0f - wyl) * svY(i, j,     k);
        const float Uyh = wyh * vy_hi + (1.0f - wyh) * svY(i, j + 1, k);
        const float Uzl = wzl * vz_lo + (1.0f - wzl) * svZ(i, j, k);
        const float Uzh = wzh * vz_hi + (1.0f - wzh) * svZ(i, j, k + 1);

        float div_val = -scale_div *
            ((Uxh - Uxl) + (Uyh - Uyl) + (Uzh - Uzl));

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

        // Variational Laplacian: each face contributes its open weight to the
        // diagonal; the +i/+j/+k off-diagonals carry -weight to FLUID neighbours
        // (symmetric, so the -i/-j/-k coupling is the neighbour's +coefficient).
        // AIR neighbours add weight to the diagonal but no off-diagonal → the
        // Dirichlet p=0 free surface. Reduces to the old integer counts when the
        // weights are binary. (fWx/fWy/fWz already fold in the domain-wall mode.)
        // Each face adds its open weight to the diagonal. FLUID neighbours also
        // get the -weight off-diagonal. AIR neighbours instead add weight/theta —
        // the ghost-fluid surface scaling that moves the p=0 boundary to the real
        // sub-cell surface (theta=1 → ordinary Dirichlet at the air centre, i.e.
        // GFM off). SOLID/closed-wall faces have weight ~0 so they drop out.
        float diag = 0.0f;
        float w;
        w = fWx(i,     j, k); if (isFluid(i - 1, j, k)) diag += w;                         else diag += w / airTheta(c, i - 1, j, k);
        w = fWx(i + 1, j, k); if (isFluid(i + 1, j, k)) { diag += w; Aplusi[c] = -w; }     else diag += w / airTheta(c, i + 1, j, k);
        w = fWy(i, j,     k); if (isFluid(i, j - 1, k)) diag += w;                         else diag += w / airTheta(c, i, j - 1, k);
        w = fWy(i, j + 1, k); if (isFluid(i, j + 1, k)) { diag += w; Aplusj[c] = -w; }     else diag += w / airTheta(c, i, j + 1, k);
        w = fWz(i, j, k);     if (isFluid(i, j, k - 1)) diag += w;                         else diag += w / airTheta(c, i, j, k - 1);
        w = fWz(i, j, k + 1); if (isFluid(i, j, k + 1)) { diag += w; Aplusk[c] = -w; }     else diag += w / airTheta(c, i, j, k + 1);
        // A fluid cell hemmed in by solid (all weights ~0) would give a singular
        // row → NaN in PCG. Floor it; an isolated cell just holds p≈0.
        if (diag < 1e-6f) diag = 1.0f;
        Adiag[c] = diag;
    }

    // 1b. Sealed pockets: give every fluid region a pressure reference.
    //
    // ★★★ THIS IS THE CURE FOR "MORE PRESSURE ITERATIONS = HARDER KICK".
    // A connected group of fluid cells whose every face is either fluid-fluid or
    // closed (solid, or a closed domain wall) has no Dirichlet row anywhere in
    // it. Its matrix block is SINGULAR and the null space is the constant
    // pressure over the region. PCG does not blow up on such a block in one
    // iteration -- it grows the null-space component a little on EVERY
    // iteration, because the right-hand side is never exactly orthogonal to it
    // (a moving wall, the fractional variational face weights and plain
    // floating-point rounding each break the exact flux balance). The pressure
    // therefore ramps with the ITERATION COUNT, the gradient at the pocket's
    // boundary ramps with it, and the particles sitting there are thrown clear.
    //
    // ★★ The measurement is what identified this, not a hunch: the symptom
    // scales with `pressure_iterations` (quiet at 8, a few particles at 9, a
    // one-frame explosion at 60+) and disappears when reseeding is switched off,
    // because reseeding is what finally tops up the last starved cell and SEALS
    // the pocket against the collider. Nothing else in this projection has
    // "converge harder, kick harder" as a property -- a genuine physical impulse
    // gets SMALLER as the solve converges, not larger. That single property
    // rules out every candidate that was guessed at before.
    //
    // Two standard repairs per sealed region:
    //   1. Subtract its mean divergence. That removes the part of the RHS lying
    //      in the null space -- the part no pressure field can answer, because
    //      it is asking a closed box to change volume. Spreading the leftover
    //      over the region is the gentlest place to put an error that physically
    //      cannot be removed.
    //   2. Pin one cell to p = 0. With a reference the block is non-singular and
    //      the answer is bounded however long PCG runs.
    //
    // ★ INERT WHEN THE DIAGNOSIS IS WRONG. A region touching air, an open wall,
    // or any partially open solid face already has a reference and is left
    // exactly as it was. `out_sealed_cells` reports what this actually fired on:
    // zero means no pocket existed and this pass changed nothing -- which is the
    // measurement that keeps this from being one more blind fix.
    {
        struct PocketCell { int i, j, k; uint32_t c; };
        static std::vector<uint8_t>    pocket_seen_buf;
        static std::vector<PocketCell> pocket_stack;
        static std::vector<uint32_t>   pocket_members;
        if (pocket_seen_buf.size() < total) pocket_seen_buf.assign(total, 0u);
        else std::fill(pocket_seen_buf.begin(), pocket_seen_buf.begin() + total, 0u);
        uint8_t* const seen = pocket_seen_buf.data();
        constexpr float kOpenEps = 1e-6f;
        const int face_off[6][3] = { {-1,0,0}, {1,0,0}, {0,-1,0},
                                     {0,1,0},  {0,0,-1}, {0,0,1} };

        for (int k0 = min_k; k0 <= max_k; ++k0)
        for (int j0 = min_j; j0 <= max_j; ++j0)
        for (int i0 = min_i; i0 <= max_i; ++i0) {
            const size_t c0 = grid.cellIndex(i0, j0, k0);
            if (!is_fluid[c0] || seen[c0]) continue;

            pocket_stack.clear();
            pocket_members.clear();
            seen[c0] = 1u;
            pocket_stack.push_back(PocketCell{ i0, j0, k0, static_cast<uint32_t>(c0) });
            bool has_reference = false;

            // Flood fill along exactly the couplings the matrix has: a face
            // carries an off-diagonal only when its open weight is non-zero AND
            // the neighbour is fluid. Following anything else here would merge
            // regions the solver keeps apart, and the singularity test would be
            // answering a question about a different matrix.
            while (!pocket_stack.empty()) {
                const PocketCell cur = pocket_stack.back();
                pocket_stack.pop_back();
                pocket_members.push_back(cur.c);
                const int i = cur.i, j = cur.j, k = cur.k;
                bool cell_touches_air = false;
                const float fw[6] = { fWx(i, j, k), fWx(i + 1, j, k),
                                      fWy(i, j, k), fWy(i, j + 1, k),
                                      fWz(i, j, k), fWz(i, j, k + 1) };
                for (int f = 0; f < 6; ++f) {
                    if (fw[f] <= kOpenEps) continue;   // closed: no coupling, no ghost
                    const int ni = i + face_off[f][0];
                    const int nj = j + face_off[f][1];
                    const int nk = k + face_off[f][2];
                    if (!isFluid(ni, nj, nk)) {
                        cell_touches_air = true;
                        // Open face onto a non-fluid cell = the Dirichlet ghost
                        // that gives this region its p = 0 reference. One is
                        // enough; the region is well posed and we are done.
                        has_reference = true;
                        continue;
                    }
                    int wi = ni, wj = nj, wk = nk;
                    if (periodic) { wi = wrapIdx(ni, nx); wj = wrapIdx(nj, ny); wk = wrapIdx(nk, nz); }
                    const size_t nc = grid.cellIndex(wi, wj, wk);
                    if (seen[nc]) continue;
                    seen[nc] = 1u;
                    pocket_stack.push_back(PocketCell{ wi, wj, wk, static_cast<uint32_t>(nc) });
                }
                // \u2605\u2605\u2605 INTERIOR = a fluid cell with liquid on all six sides. It is
                // the only kind of cell the pressure solve can do anything with:
                // a cell that touches air is held at p \u2248 0 by the free-surface
                // condition, so a body made ENTIRELY of such cells has no
                // pressure field, no incompressibility, and nothing to eject a
                // droplet with on impact. It falls as loose particles and lands
                // in a heap -- which reads exactly as "thick, blobby, no splash".
                //
                // \u2605\u2605 THIS IS THE NUMBER THAT SAYS "YOUR JET IS TOO THIN FOR THIS
                // VOXEL SIZE", and its absence is why that diagnosis took days.
                // A stream one or two cells across is all surface; the fix is
                // more mass per second or a smaller voxel -- NOT viscosity, not
                // damping, not the surface settings, all of which were tried.
                // The eye cannot tell an under-resolved jet from a viscous one;
                // this ratio can.
                if (!cell_touches_air) ++out_interior_cells;
            }

            if (has_reference || pocket_members.empty()) continue;

            ++out_sealed_pockets;
            out_sealed_cells += pocket_members.size();

            double div_sum = 0.0;
            for (uint32_t c : pocket_members) div_sum += static_cast<double>(div[c]);
            const float div_mean =
                static_cast<float>(div_sum / static_cast<double>(pocket_members.size()));
            for (uint32_t c : pocket_members) div[c] -= div_mean;

            // Pin the seed cell. Both directions of every incident off-diagonal
            // have to go: the PCG reads Aplus* from the lower cell for both
            // sides of a face, so clearing only this cell's entries would leave
            // an asymmetric matrix and CG would silently stop being CG.
            const size_t pc = static_cast<size_t>(pocket_members[0]);
            div[pc]    = 0.0f;
            Adiag[pc]  = 1.0f;
            Aplusi[pc] = 0.0f;
            Aplusj[pc] = 0.0f;
            Aplusk[pc] = 0.0f;
            const int mi = (i0 > 0) ? i0 - 1 : (periodic ? nx - 1 : -1);
            const int mj = (j0 > 0) ? j0 - 1 : (periodic ? ny - 1 : -1);
            const int mk = (k0 > 0) ? k0 - 1 : (periodic ? nz - 1 : -1);
            if (mi >= 0) Aplusi[grid.cellIndex(mi, j0, k0)] = 0.0f;
            if (mj >= 0) Aplusj[grid.cellIndex(i0, mj, k0)] = 0.0f;
            if (mk >= 0) Aplusk[grid.cellIndex(i0, j0, mk)] = 0.0f;
        }
    }

    // 2. MIC(0) preconditioner. E[c] = A[c,c] - sum_{n in -i,-j,-k}
    //    (A[n,c])^2 * precon[n]^2 - tau * A[n,c] * precon[n]^2 *
    //    (sum of other-axis A[n,*] coefficients), with the standard safety
    //    fallback when E becomes too small relative to A[c,c].
    //    Reference: Bridson & Müller-Fischer SIGGRAPH 2007 course notes,
    //    "Fluid Simulation for Computer Graphics" 2nd ed. §5.2.
    constexpr float tau = 0.97f;       // MIC tuning constant
    constexpr float safety = 0.25f;    // restart threshold (Bridson)
    if (periodic) {
        // MIC(0) factorises the matrix into banded lower/upper triangular sweeps
        // in lexicographic order; the periodic wrap adds couplings (cell 0 ↔ cell
        // N-1) that fall OUTSIDE that band, so the forward/backward substitution
        // would silently drop them and the preconditioner would no longer match A.
        // Use a Jacobi (diagonal) preconditioner instead: it has no ordering
        // dependency, stays correct under wrap, and PCG still converges — just in
        // more iterations. precon holds 1/diag here (Jacobi), not the MIC 1/√E.
        for (int k = min_k; k <= max_k; ++k)
            for (int j = min_j; j <= max_j; ++j)
                for (int i = min_i; i <= max_i; ++i) {
                    const size_t c = grid.cellIndex(i, j, k);
                    if (!is_fluid[c]) continue;
                    precon[c] = (Adiag[c] > 1e-12f) ? (1.0f / Adiag[c]) : 0.0f;
                }
    } else
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
        const float rel_tol = std::clamp(params.pressure_relative_residual, 1.0e-8f, 1.0e-2f);
        const float tolerance = std::max(abs_tol, rel_tol * r_inf);

        auto applyPreconditioner = [&]() {
            if (periodic) {
                // Jacobi: z = D^{-1} r (precon holds 1/diag on the periodic path).
                for (int k = min_k; k <= max_k; ++k)
                    for (int j = min_j; j <= max_j; ++j)
                        for (int i = min_i; i <= max_i; ++i) {
                            const size_t c = grid.cellIndex(i, j, k);
                            if (!is_fluid[c]) continue;
                            z[c] = r[c] * precon[c];
                        }
                return;
            }
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
                // Lower-neighbour coupling uses the neighbour's stored upper
                // coefficient (symmetry: A[c,c-1] == Aplus*[c-1]). On the periodic
                // path the lower neighbour of index 0 wraps to N-1 and the upper
                // neighbour of N-1 wraps to 0; the seam coefficient lives in
                // Aplus*[N-1] for both directions, so symmetry still holds.
                const int im = (i > 0) ? i - 1 : (periodic ? nx - 1 : -1);
                const int ip = (i + 1 < nx) ? i + 1 : (periodic ? 0 : -1);
                const int jm = (j > 0) ? j - 1 : (periodic ? ny - 1 : -1);
                const int jp = (j + 1 < ny) ? j + 1 : (periodic ? 0 : -1);
                const int km = (k > 0) ? k - 1 : (periodic ? nz - 1 : -1);
                const int kp = (k + 1 < nz) ? k + 1 : (periodic ? 0 : -1);
                if (im >= 0) {
                    const size_t cm = grid.cellIndex(im, j, k);
                    if (is_fluid[cm]) v += Aplusi[cm] * s[cm];
                }
                if (ip >= 0) {
                    const size_t cp = grid.cellIndex(ip, j, k);
                    if (is_fluid[cp]) v += Aplusi[c] * s[cp];
                }
                if (jm >= 0) {
                    const size_t cm = grid.cellIndex(i, jm, k);
                    if (is_fluid[cm]) v += Aplusj[cm] * s[cm];
                }
                if (jp >= 0) {
                    const size_t cp = grid.cellIndex(i, jp, k);
                    if (is_fluid[cp]) v += Aplusj[c] * s[cp];
                }
                if (km >= 0) {
                    const size_t cm = grid.cellIndex(i, j, km);
                    if (is_fluid[cm]) v += Aplusk[cm] * s[cm];
                }
                if (kp >= 0) {
                    const size_t cp = grid.cellIndex(i, j, kp);
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
        // Periodic seam faces (i==0 and i==N are the same physical face) are
        // handled once in the dedicated post-pass below — skipping them here
        // avoids the active-region loop touching only one copy when fluid sits
        // on a single side of the seam.
        if (periodic && (i <= 0 || i >= nx)) continue;
        // Fully-closed face → take the solid's normal velocity (moving wall);
        // open/partial face → subtract the pressure gradient (the weight already
        // lives in the matrix, so the update itself is the plain MAC gradient).
        const float wf = fWx(i, j, k);
        if (wf <= 0.0f) {
            grid.vel_x[grid.velXIndex(i, j, k)] = svX(i, j, k);
            continue;
        }
        bool lo_fluid = isFluid(i - 1, j, k);
        bool hi_fluid = isFluid(i,     j, k);
        if (!lo_fluid && !hi_fluid) continue;
        float p_lo = lo_fluid ? p[cIdx(i - 1, j, k)] : 0.0f;
        float p_hi = hi_fluid ? p[cIdx(i,     j, k)] : 0.0f;
        // Ghost-fluid: the air-side pressure is the extrapolated p that puts p=0
        // at the sub-cell surface (p_ghost = p_fluid*(1 - 1/theta)) instead of 0.
        if (gfm_active) {
            if (lo_fluid && !hi_fluid)      p_hi = p_lo * (1.0f - 1.0f / airTheta(cIdx(i - 1, j, k), i,     j, k));
            else if (hi_fluid && !lo_fluid) p_lo = p_hi * (1.0f - 1.0f / airTheta(cIdx(i,     j, k), i - 1, j, k));
        }
        grid.vel_x[grid.velXIndex(i, j, k)] -= grad_scale * (p_hi - p_lo);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(active_grid_parallel)
#endif
    for (int64_t raw = 0; raw < static_cast<int64_t>(active_nx) * (active_ny + 1) * active_nz; ++raw) {
        const int i = min_i + static_cast<int>(raw % active_nx);
        const int j = min_j + static_cast<int>((raw / active_nx) % (active_ny + 1));
        const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * (active_ny + 1)));
        if (periodic && (j <= 0 || j >= ny)) continue;
        const float wf = fWy(i, j, k);
        if (wf <= 0.0f) {
            grid.vel_y[grid.velYIndex(i, j, k)] = svY(i, j, k);
            continue;
        }
        bool lo_fluid = isFluid(i, j - 1, k);
        bool hi_fluid = isFluid(i, j,     k);
        if (!lo_fluid && !hi_fluid) continue;
        float p_lo = lo_fluid ? p[cIdx(i, j - 1, k)] : 0.0f;
        float p_hi = hi_fluid ? p[cIdx(i, j,     k)] : 0.0f;
        if (gfm_active) {
            if (lo_fluid && !hi_fluid)      p_hi = p_lo * (1.0f - 1.0f / airTheta(cIdx(i, j - 1, k), i, j,     k));
            else if (hi_fluid && !lo_fluid) p_lo = p_hi * (1.0f - 1.0f / airTheta(cIdx(i, j,     k), i, j - 1, k));
        }
        grid.vel_y[grid.velYIndex(i, j, k)] -= grad_scale * (p_hi - p_lo);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(active_grid_parallel)
#endif
    for (int64_t raw = 0; raw < static_cast<int64_t>(active_nx) * active_ny * (active_nz + 1); ++raw) {
        const int i = min_i + static_cast<int>(raw % active_nx);
        const int j = min_j + static_cast<int>((raw / active_nx) % active_ny);
        const int k = min_k + static_cast<int>(raw / (static_cast<int64_t>(active_nx) * active_ny));
        if (periodic && (k <= 0 || k >= nz)) continue;
        const float wf = fWz(i, j, k);
        if (wf <= 0.0f) {
            grid.vel_z[grid.velZIndex(i, j, k)] = svZ(i, j, k);
            continue;
        }
        bool lo_fluid = isFluid(i, j, k - 1);
        bool hi_fluid = isFluid(i, j, k);
        if (!lo_fluid && !hi_fluid) continue;
        float p_lo = lo_fluid ? p[cIdx(i, j, k - 1)] : 0.0f;
        float p_hi = hi_fluid ? p[cIdx(i, j, k)]     : 0.0f;
        if (gfm_active) {
            if (lo_fluid && !hi_fluid)      p_hi = p_lo * (1.0f - 1.0f / airTheta(cIdx(i, j, k - 1), i, j, k));
            else if (hi_fluid && !lo_fluid) p_lo = p_hi * (1.0f - 1.0f / airTheta(cIdx(i, j, k),     i, j, k - 1));
        }
        grid.vel_z[grid.velZIndex(i, j, k)] -= grad_scale * (p_hi - p_lo);
    }

    // Periodic seam gradient pass. The seam faces still hold the merged
    // pre-projection velocity m (the main loops skipped them); apply the seam
    // pressure gradient ONCE and write both copies identically so the wrap stays
    // single-valued regardless of which side carried fluid. The lo cell is the
    // far wall (N-1), the hi cell is the near wall (0). GFM isn't applied at the
    // seam (the wrapped air-extrapolation theta collapses to 1 → plain p=0).
    if (periodic) {
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j) {
                const size_t f0 = grid.velXIndex(0,  j, k);
                const size_t fN = grid.velXIndex(nx, j, k);
                if (fWx(0, j, k) <= 0.0f) { grid.vel_x[f0] = grid.vel_x[fN] = svX(0, j, k); continue; }
                const bool lo_fluid = isFluid(nx - 1, j, k);
                const bool hi_fluid = isFluid(0,      j, k);
                if (!lo_fluid && !hi_fluid) continue;
                const float p_lo = lo_fluid ? p[grid.cellIndex(nx - 1, j, k)] : 0.0f;
                const float p_hi = hi_fluid ? p[grid.cellIndex(0,      j, k)] : 0.0f;
                const float nv = grid.vel_x[f0] - grad_scale * (p_hi - p_lo);
                grid.vel_x[f0] = grid.vel_x[fN] = nv;
            }
        for (int k = 0; k < nz; ++k)
            for (int i = 0; i < nx; ++i) {
                const size_t f0 = grid.velYIndex(i, 0,  k);
                const size_t fN = grid.velYIndex(i, ny, k);
                if (fWy(i, 0, k) <= 0.0f) { grid.vel_y[f0] = grid.vel_y[fN] = svY(i, 0, k); continue; }
                const bool lo_fluid = isFluid(i, ny - 1, k);
                const bool hi_fluid = isFluid(i, 0,      k);
                if (!lo_fluid && !hi_fluid) continue;
                const float p_lo = lo_fluid ? p[grid.cellIndex(i, ny - 1, k)] : 0.0f;
                const float p_hi = hi_fluid ? p[grid.cellIndex(i, 0,      k)] : 0.0f;
                const float nv = grid.vel_y[f0] - grad_scale * (p_hi - p_lo);
                grid.vel_y[f0] = grid.vel_y[fN] = nv;
            }
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const size_t f0 = grid.velZIndex(i, j, 0);
                const size_t fN = grid.velZIndex(i, j, nz);
                if (fWz(i, j, 0) <= 0.0f) { grid.vel_z[f0] = grid.vel_z[fN] = svZ(i, j, 0); continue; }
                const bool lo_fluid = isFluid(i, j, nz - 1);
                const bool hi_fluid = isFluid(i, j, 0);
                if (!lo_fluid && !hi_fluid) continue;
                const float p_lo = lo_fluid ? p[grid.cellIndex(i, j, nz - 1)] : 0.0f;
                const float p_hi = hi_fluid ? p[grid.cellIndex(i, j, 0)]      : 0.0f;
                const float nv = grid.vel_z[f0] - grad_scale * (p_hi - p_lo);
                grid.vel_z[f0] = grid.vel_z[fN] = nv;
            }
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

// ── Implicit viscous diffusion (CPU reference) ──────────────────────────────
// Solves (I - ν·dt·∇²) u_new = u_old independently per MAC component with
// red-black Gauss-Seidel. This is the ORACLE the Vulkan kernel
// (sim_fluid_viscosity_rbgs.comp) must match face for face; keep the two
// stencils and boundary rules in step, or a device/host mismatch shows up as
// "the GPU one is a bit thinner", which nobody reports as a bug.
//
// Three things the previous neighbour-average smoother got wrong, all of which
// made the viscosity dial mean different things in different scenes:
//   1. no 1/h² — the SAME ν behaved differently at every voxel size;
//   2. explicit, so it had to clamp its strength to 0.45 and saturated: 20 and
//      30 produced nearly identical motion and honey could never be honey;
//   3. it averaged AIR faces (velocity ≈ 0) into the surface, which brakes the
//      free surface against nothing — a "skin" that is not a physical stress.
//
// Boundary rules, one per face classification:
//   fluid neighbour → ordinary coupling;
//   air neighbour   → stress-free: DROPPED from both the sum and the diagonal
//                     (Neumann). Dropping, not zeroing — a zero would be a wall;
//   solid neighbour → Dirichlet toward the solid's own velocity, weighted by
//                     (1 - viscosity_wall_slip). slip=1 degenerates to the air
//                     rule (free-slip), slip=0 is full no-slip, which is what
//                     makes a viscous fluid coat a collider and get dragged by
//                     a moving one.
namespace {

// Face classification shared by all three components. The fluid mask lives in
// the cell-centred grid: solid cells are solid, cells holding particles are
// fluid, everything else is air.
enum class FaceKind { Fluid, Air, Solid };

} // namespace

// Cell-centred liquid occupancy for the viscous stencil: -1 solid, >0.5 fluid,
// 0 air. Deliberately the SAME encoding the GPU pressure path uploads
// (ParticleSimulation.cpp:buildFluidMaskFromParticles) so the host and device
// viscosity kernels classify identical faces — a mask that disagreed by one
// cell would show up as a slightly different surface, not as an error.
static void buildViscosityFluidMask(const FluidSim::FluidGrid& grid,
                                    const FluidParticles& particles,
                                    std::vector<float>& mask) {
    const std::size_t cells = grid.getCellCount();
    if (mask.size() < cells) mask.assign(cells, 0.0f);
    else std::fill(mask.begin(), mask.begin() + static_cast<std::ptrdiff_t>(cells), 0.0f);
    const bool has_solid = grid.solid.size() == cells;
    if (has_solid) {
        for (std::size_t c = 0; c < cells; ++c)
            if (grid.solid[c]) mask[c] = -1.0f;
    }
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float invH = grid.voxel_size > 1.0e-6f ? 1.0f / grid.voxel_size : 0.0f;
    for (std::size_t p = 0; p < particles.size(); ++p) {
        const Vec3 gp = (particles.position[p] - grid.origin) * invH;
        const int i = static_cast<int>(std::floor(gp.x));
        const int j = static_cast<int>(std::floor(gp.y));
        const int k = static_cast<int>(std::floor(gp.z));
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
        const std::size_t c = grid.cellIndex(i, j, k);
        if (has_solid && grid.solid[c]) continue;
        const float mass = p < particles.mass_fraction.size()
            ? std::clamp(particles.mass_fraction[p], 0.0f, 1.0f)
            : 1.0f;
        if (mass > 0.02f) mask[c] += mass;
    }
}

static void applyViscosity(FluidSim::FluidGrid& grid,
                           const APICSolverParams& params,
                           float dt,
                           const std::vector<float>* fluid_mask,
                           int& out_sweeps_run) {
    out_sweeps_run = 0;
    // A per-cell field switches the stage on by itself: it is only ever built
    // when a substance actually overrides the domain, and ignoring it because
    // the domain happens to be inviscid would make an authored number a no-op.
    const bool has_nu_field =
        params.substance_viscosity != nullptr &&
        params.substance_viscosity->size() == grid.getCellCount();
    if ((params.kinematic_viscosity <= 0.0f && !has_nu_field) || dt <= 0.0f) {
        return;
    }
    const float h = (grid.voxel_size > 1.0e-6f) ? grid.voxel_size : 1.0f;
    // a = ν·dt/h² — the dimensionless diffusion number. THIS is the term the old
    // code was missing; without it ν is not a viscosity, it is a screen-space
    // blur radius.
    const float a = params.kinematic_viscosity * dt / (h * h);
    if (a <= 1.0e-8f && !has_nu_field) {
        return;
    }
    const float dt_over_h2 = dt / (h * h);
    const int sweeps = std::clamp(params.viscosity_sweeps, 1, 64);
    const float slip = std::clamp(params.viscosity_wall_slip, 0.0f, 1.0f);
    const float solid_weight = 1.0f - slip;   // 0 = free-slip, 1 = no-slip

    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const bool closed = params.boundary != APICSolverParams::BoundaryMode::Open;
    const bool has_mask = fluid_mask && fluid_mask->size() == grid.getCellCount();
    const bool has_solid = !grid.solid.empty() && grid.solid.size() == grid.getCellCount();
    const bool has_solid_vel = grid.solid_vel.size() == grid.getCellCount();

    auto cellSolid = [&](int i, int j, int k) -> bool {
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return false;
        return has_solid && grid.solid[grid.cellIndex(i, j, k)] != 0u;
    };
    auto cellFluid = [&](int i, int j, int k) -> bool {
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return false;
        if (cellSolid(i, j, k)) return false;
        // No mask (non-free-surface solve) means the whole grid is liquid.
        if (!has_mask) return true;
        return (*fluid_mask)[grid.cellIndex(i, j, k)] > 0.5f;
    };
    auto solidVel = [&](int i, int j, int k, int comp) -> float {
        if (!has_solid_vel) return 0.0f;
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return 0.0f;
        const Vec3& v = grid.solid_vel[grid.cellIndex(i, j, k)];
        return (comp == 0) ? v.x : (comp == 1) ? v.y : v.z;
    };

    // ── Variable viscosity ──────────────────────────────────────────────────
    // ν lives at CELL CENTRES; the unknowns live on FACES. A face's viscosity is
    // the mean of the two cells it straddles, and the coefficient coupling two
    // faces is the mean of THEIR viscosities.
    //
    // ★★★ SYMMETRIC ON PURPOSE. Taking the coefficient from the updating face
    // alone would make the operator non-self-adjoint: the milk side would pull
    // on the chocolate side harder than the chocolate side pulls back, so
    // momentum would leak across the interface. Gauss-Seidel does not diverge
    // on that — it converges to the wrong answer, and the wrong answer looks
    // like a mixture that slowly drifts one way.
    auto clampCell = [&](int& i, int& j, int& k) {
        i = std::clamp(i, 0, nx - 1);
        j = std::clamp(j, 0, ny - 1);
        k = std::clamp(k, 0, nz - 1);
    };
    auto cellNu = [&](int i, int j, int k) -> float {
        clampCell(i, j, k);
        return (*params.substance_viscosity)[grid.cellIndex(i, j, k)];
    };
    auto faceNu = [&](int comp, int i, int j, int k) -> float {
        int li = i, lj = j, lk = k;
        if (comp == 0)      li = i - 1;
        else if (comp == 1) lj = j - 1;
        else                lk = k - 1;
        return 0.5f * (cellNu(li, lj, lk) + cellNu(i, j, k));
    };

    // Classify a face of the given component and, when it is usable, hand back
    // the value to couple against.
    auto classify = [&](int comp, int i, int j, int k,
                        const std::vector<float>& field,
                        float& out_value) -> FaceKind {
        const int fsx = (comp == 0) ? nx + 1 : nx;
        const int fsy = (comp == 1) ? ny + 1 : ny;
        const int fsz = (comp == 2) ? nz + 1 : nz;
        if (i < 0 || i >= fsx || j < 0 || j >= fsy || k < 0 || k >= fsz) {
            // Outside the face grid: a closed domain wall is a solid at rest,
            // an open one is outflow (treat as air / stress-free).
            out_value = 0.0f;
            return closed ? FaceKind::Solid : FaceKind::Air;
        }
        int li, lj, lk, hi, hj, hk;
        li = i; lj = j; lk = k; hi = i; hj = j; hk = k;
        if (comp == 0)      li = i - 1;
        else if (comp == 1) lj = j - 1;
        else                lk = k - 1;

        if (cellSolid(li, lj, lk) || cellSolid(hi, hj, hk)) {
            out_value = cellSolid(li, lj, lk) ? solidVel(li, lj, lk, comp)
                                              : solidVel(hi, hj, hk, comp);
            return FaceKind::Solid;
        }
        if (cellFluid(li, lj, lk) || cellFluid(hi, hj, hk)) {
            const size_t idx = static_cast<size_t>(i) +
                               static_cast<size_t>(j) * fsx +
                               static_cast<size_t>(k) * static_cast<size_t>(fsx) * fsy;
            out_value = field[idx];
            return FaceKind::Fluid;
        }
        out_value = 0.0f;
        return FaceKind::Air;
    };

    // Right-hand side: the pre-diffusion field. GS updates in place, so the
    // original values must be kept somewhere. Function-static and grow-only,
    // like the rest of this solver's scratch.
    static std::vector<float> rhs_x, rhs_y, rhs_z;
    rhs_x.assign(grid.vel_x.begin(), grid.vel_x.end());
    rhs_y.assign(grid.vel_y.begin(), grid.vel_y.end());
    rhs_z.assign(grid.vel_z.begin(), grid.vel_z.end());

    auto sweepComponent = [&](int comp,
                              std::vector<float>& field,
                              const std::vector<float>& rhs,
                              int parity) {
        const int fsx = (comp == 0) ? nx + 1 : nx;
        const int fsy = (comp == 1) ? ny + 1 : ny;
        const int fsz = (comp == 2) ? nz + 1 : nz;
        const int64_t count = static_cast<int64_t>(fsx) * fsy * fsz;
        // Serial on purpose, same reason the old smoother was: MSVC OpenMP
        // rejects num_threads() over lambda-captured locals here, and vcomp has
        // already been a stability problem in this file on large domains. Vulkan
        // is the primary path for this solve; the host copy is the reference and
        // the fallback, so correctness outranks its throughput.
        for (int64_t raw = 0; raw < count; ++raw) {
            const int i = static_cast<int>(raw % fsx);
            const int j = static_cast<int>((raw / fsx) % fsy);
            const int k = static_cast<int>(raw / (static_cast<int64_t>(fsx) * fsy));
            if (((i + j + k) & 1) != parity) continue;

            float self_value = 0.0f;
            if (classify(comp, i, j, k, field, self_value) != FaceKind::Fluid) {
                continue; // only fluid faces carry the viscous unknown
            }

            float sum = 0.0f;
            float diag = 1.0f;
            const float nu_self = has_nu_field ? faceNu(comp, i, j, k) : 0.0f;
            const int offsets[6][3] = {
                {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}
            };
            for (const auto& o : offsets) {
                float nv = 0.0f;
                const FaceKind kind = classify(comp, i + o[0], j + o[1], k + o[2],
                                               field, nv);
                // Uniform domains keep the precomputed scalar; the branch is on
                // a loop-invariant bool and costs nothing there.
                const float a_link = has_nu_field
                    ? 0.5f * (nu_self + faceNu(comp, i + o[0], j + o[1], k + o[2]))
                          * dt_over_h2
                    : a;
                if (kind == FaceKind::Fluid) {
                    sum  += a_link * nv;
                    diag += a_link;
                } else if (kind == FaceKind::Solid) {
                    // Dirichlet at the solid's velocity, scaled by how much of
                    // the no-slip condition the material asks for.
                    sum  += a_link * solid_weight * nv;
                    diag += a_link * solid_weight;
                }
                // Air: stress-free — contributes to neither side.
            }
            field[static_cast<size_t>(raw)] = (rhs[static_cast<size_t>(raw)] + sum) / diag;
        }
    };

    for (int s = 0; s < sweeps; ++s) {
        for (int parity = 0; parity < 2; ++parity) {
            sweepComponent(0, grid.vel_x, rhs_x, parity);
            sweepComponent(1, grid.vel_y, rhs_y, parity);
            sweepComponent(2, grid.vel_z, rhs_z, parity);
        }
    }
    out_sweeps_run = sweeps;
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
                            int max_substeps,
                            // Per-particle "this parcel is solid phase" mask, or
                            // null when the domain declares no solid substance.
                            const uint8_t* solid_particle) {
    const float h    = grid.voxel_size;
    const float invH = 1.0f / h;

    const int snx = grid.nx, sny = grid.ny, snz = grid.nz;
    const bool has_solid = !grid.solid.empty();
    const bool track_solid_vel = has_solid && grid.solid_vel.size() == grid.solid.size();

    // Interior solid (collider) collision helpers. advectParticles only clamps
    // to the DOMAIN walls; voxelized colliders are enforced on the GRID velocity
    // by the pressure projection, but FLIP/APIC particles carry their own
    // velocity and will tunnel through a thin solid band or get swept into a
    // MOVING collider's cells. These resolve penetration at the particle level.
    auto solidCellAt = [&](int i, int j, int k) -> bool {
        if (i < 0 || i >= snx || j < 0 || j >= sny || k < 0 || k >= snz) return false;
        return grid.solid[grid.cellIndex(i, j, k)] != 0u;
    };
    auto solidAtPos = [&](const Vec3& p) -> bool {
        const Vec3 g = (p - grid.origin) * invH;
        return solidCellAt(static_cast<int>(std::floor(g.x)),
                           static_cast<int>(std::floor(g.y)),
                           static_cast<int>(std::floor(g.z)));
    };
    // Linear velocity of the solid at a world position (0 if not solid / not
    // tracked). Lets a MOVING collider hand its momentum to the fluid.
    auto solidVelAtPos = [&](const Vec3& p) -> Vec3 {
        if (!track_solid_vel) return Vec3(0.0f, 0.0f, 0.0f);
        const Vec3 g = (p - grid.origin) * invH;
        const int i = static_cast<int>(std::floor(g.x));
        const int j = static_cast<int>(std::floor(g.y));
        const int k = static_cast<int>(std::floor(g.z));
        if (i < 0 || i >= snx || j < 0 || j >= sny || k < 0 || k >= snz) return Vec3(0.0f, 0.0f, 0.0f);
        return grid.solid_vel[grid.cellIndex(i, j, k)];
    };

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

    // Wall behaviour. Closed clamps + bounces (sealed box). Open lets particles
    // leave through the walls (flagged here, culled after the substep loop) so
    // an "Open (Outflow)" domain actually drains. Periodic wraps to the far
    // wall. Bit 1 of `flags` marks a particle as having flowed out.
    const APICSolverParams::BoundaryMode boundary = params.boundary;
    const Vec3 span = gmax - gmin;
    constexpr uint32_t FLAG_OUTFLOW = 1u << 1;

    for (int s = 0; s < substeps; ++s) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(parallel)
#endif
        for (int64_t raw_pi = 0; raw_pi < static_cast<int64_t>(parts.size()); ++raw_pi) {
            const size_t pi = static_cast<size_t>(raw_pi);
            // A particle that already flowed out this frame is frozen until the
            // post-loop cull removes it — skip so it never samples outside.
            if (parts.flags[pi] & FLAG_OUTFLOW) continue;
            Vec3 p0 = parts.position[pi];
            const bool is_solid_parcel = solid_particle && solid_particle[pi];
            Vec3 pn;
            if (is_solid_parcel) {
                // ★ A solid parcel moves by its OWN velocity, not by the grid
                // field — the same reason gridToParticle skips it. Sampling the
                // MAC field here would advect the chunk with the liquid it is
                // blocking, and because its own cells carry solid_vel the
                // result would look nearly right at rest and drift only under
                // flow, which is the hardest version of this bug to see.
                pn = p0 + parts.velocity[pi] * sub_dt;
            } else {
                // RK2 midpoint using grid velocity (more stable than forward Euler
                // for free-surface flow, negligible cost on CPU).
                Vec3 g0 = (p0 - grid.origin) * invH;
                Vec3 v0(sampleMACComponent(grid, 0, g0),
                        sampleMACComponent(grid, 1, g0),
                        sampleMACComponent(grid, 2, g0));
                Vec3 pmid = p0 + v0 * (0.5f * sub_dt);
                Vec3 gm   = (pmid - grid.origin) * invH;
                Vec3 vm(sampleMACComponent(grid, 0, gm),
                        sampleMACComponent(grid, 1, gm),
                        sampleMACComponent(grid, 2, gm));
                pn = p0 + vm * sub_dt;
            }

            // ── Interior solid collision ─────────────────────────────────────
            // Resolved BEFORE the domain-wall handling so nothing ever slides
            // into a voxelized solid. Two branches, because a parcel that IS
            // the solid asks a different question than one caught inside it.
            if (has_solid && is_solid_parcel) {
                // ── Solid parcel vs. everything else that is solid ────────────
                // ★★★ ITS OWN CELL IS NOT AN OBSTACLE. The liquid branch below
                // treats a parcel standing in a solid cell as swallowed by a
                // moving collider and ejects it; for the parcel that IS the
                // solid that reads as the chunk blowing itself apart the moment
                // it becomes thick enough to fill a cell — a "collision
                // instability" nobody would trace back to a missing exception.
                //
                // ★★ Blocked against every OTHER solid cell, including cells
                // held by sibling parcels: that mutual blocking is the only
                // thing that makes chunks stack instead of collapsing into one
                // layer on the floor. It is contact, not cohesion — a pile
                // still spreads sideways under load, which is the honest limit
                // of an obstacle field (see Fluid::SubstancePhase).
                Vec3& pv = parts.velocity[pi];
                const Vec3 g0c = (p0 - grid.origin) * invH;
                const int own_i = static_cast<int>(std::floor(g0c.x));
                const int own_j = static_cast<int>(std::floor(g0c.y));
                const int own_k = static_cast<int>(std::floor(g0c.z));
                auto blockedAt = [&](const Vec3& p) -> bool {
                    const Vec3 g = (p - grid.origin) * invH;
                    const int i = static_cast<int>(std::floor(g.x));
                    const int j = static_cast<int>(std::floor(g.y));
                    const int k = static_cast<int>(std::floor(g.z));
                    if (i == own_i && j == own_j && k == own_k) return false; // own cell
                    return solidCellAt(i, j, k);
                };
                // Separated-axis slide with a RELATIVE-velocity contact test.
                //
                // ★★★ "THE CELL AHEAD IS SOLID" IS NOT ENOUGH, and getting this
                // wrong tears a falling chunk apart. Inside any solid body the
                // cell below a parcel is occupied by its own siblings, so a
                // purely positional test blocks every layer except the bottom
                // one: the bottom falls, the rest stay put, and the chunk
                // smears into a column. Blocking only when the parcel is
                // APPROACHING the obstacle in the obstacle's own frame makes
                // co-moving matter transparent to itself while a floor (or a
                // chunk that has already stopped) still stops it. The failure
                // this avoids looks like melting, not like a collision bug.
                //
                // ★ A blocked axis takes the obstacle's velocity — 0 for a
                // static collider, so a landing chunk comes to rest with no
                // restitution nobody authored. Tangential components survive,
                // so it still slides down a slope.
                auto resolveAxis = [&](int axis, float p0_a, float pn_a,
                                       Vec3& out_pos, float& out_vel) {
                    Vec3 probe = out_pos;
                    if (axis == 0) probe.x = pn_a; else if (axis == 1) probe.y = pn_a; else probe.z = pn_a;
                    if (!blockedAt(probe)) {
                        if (axis == 0) out_pos.x = pn_a; else if (axis == 1) out_pos.y = pn_a; else out_pos.z = pn_a;
                        return;
                    }
                    const Vec3 sv = solidVelAtPos(probe);
                    const float sv_a = (axis == 0) ? sv.x : (axis == 1) ? sv.y : sv.z;
                    const float dir = pn_a - p0_a;
                    const float approach = (out_vel - sv_a) * dir;
                    if (approach <= 0.0f) {
                        // Separating or co-moving: the obstacle is this parcel's
                        // own body travelling with it. Let it through.
                        if (axis == 0) out_pos.x = pn_a; else if (axis == 1) out_pos.y = pn_a; else out_pos.z = pn_a;
                        return;
                    }
                    out_vel = sv_a;
                };
                Vec3 resolved = p0;
                resolveAxis(0, p0.x, pn.x, resolved, pv.x);
                resolveAxis(1, p0.y, pn.y, resolved, pv.y);
                resolveAxis(2, p0.z, pn.z, resolved, pv.z);
                pn = resolved;
            } else if (has_solid) {
                // Liquid. Two cases:
                //   (a) the origin is already solid — a MOVING collider swept
                //       its cells over the particle this step; eject to the
                //       nearest free cell so the collider shoves the fluid
                //       instead of swallowing it.
                //   (b) the destination is solid — separated-axis slide from p0
                //       so the blocked component stops while motion along the
                //       surface is preserved (no tunneling).
                Vec3& pv = parts.velocity[pi];
                if (solidAtPos(p0)) {
                    const Vec3 g0c = (p0 - grid.origin) * invH;
                    const int ci = static_cast<int>(std::floor(g0c.x));
                    const int cj = static_cast<int>(std::floor(g0c.y));
                    const int ck = static_cast<int>(std::floor(g0c.z));
                    // -- Which KIND of solid swallowed this parcel? ------------
                    // ★★★ A CHUNK OF MATTER IS NOT A COLLIDER, and resolving them
                    // the same way is the second half of the reported burst.
                    // Below, a swallowed parcel is teleported to the CENTRE of
                    // the nearest free cell, up to three cells away -- correct
                    // for a collider, which is a hard surface that genuinely
                    // swept through and whose interior no liquid may occupy.
                    // A solid-phase substance is liquid's own neighbour: its
                    // mask changes shape every step, so parcels drift in and out
                    // of it constantly, and a three-cell teleport per crossing
                    // is an explosion sourced at the interface.
                    //
                    // ★ So a substance solid pushes across ONE FACE only, by a
                    // hair, keeping the other two axes where they were. Worst
                    // case the parcel is deep inside a chunk and no face is
                    // free: then it is left ALONE this substep rather than flung
                    // outward -- the mask moves on the next step and it leaves
                    // on its own. Doing nothing is the honest answer to "there
                    // is nowhere to go"; teleporting is a guess that always
                    // looks like a bug.
                    const bool in_bounds_cell =
                        ci >= 0 && ci < snx && cj >= 0 && cj < sny && ck >= 0 && ck < snz;
                    const bool substance_solid = in_bounds_cell &&
                        grid.solid[grid.cellIndex(ci, cj, ck)] ==
                            FluidSim::FluidGrid::kSolidSubstance;
                    if (substance_solid) {
                        static const int kFaceOffsets[6][3] = {
                            { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 },
                            { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }
                        };
                        int best = -1;
                        float best_gap = -1.0f;
                        for (int f = 0; f < 6; ++f) {
                            const int di = kFaceOffsets[f][0];
                            const int dj = kFaceOffsets[f][1];
                            const int dk = kFaceOffsets[f][2];
                            const int ni = ci + di, nj = cj + dj, nk = ck + dk;
                            // ★ Out of grid is NOT a free cell here, even though
                            // solidCellAt says "not solid" for it. Taking that
                            // answer would push the parcel through the domain
                            // wall — past the clamp this branch skips — and it
                            // would only ever happen to a chunk touching the
                            // boundary, i.e. rarely and inexplicably.
                            if (ni < 0 || ni >= snx || nj < 0 || nj >= sny ||
                                nk < 0 || nk >= snz) continue;
                            if (solidCellAt(ni, nj, nk)) continue;
                            // Prefer the face the parcel is already closest to:
                            // it is the shortest way out and the one its own
                            // motion was heading for.
                            const float frac = (di != 0) ? (g0c.x - static_cast<float>(ci))
                                             : (dj != 0) ? (g0c.y - static_cast<float>(cj))
                                                         : (g0c.z - static_cast<float>(ck));
                            const float gap = (di + dj + dk > 0) ? (1.0f - frac) : frac;
                            if (best < 0 || gap < best_gap) { best = f; best_gap = gap; }
                        }
                        if (best >= 0) {
                            const int di = kFaceOffsets[best][0];
                            const int dj = kFaceOffsets[best][1];
                            const int dk = kFaceOffsets[best][2];
                            const float eps = 0.02f * h;
                            pn = p0;
                            if (di > 0)      pn.x = grid.origin.x + static_cast<float>(ci + 1) * h + eps;
                            else if (di < 0) pn.x = grid.origin.x + static_cast<float>(ci) * h - eps;
                            if (dj > 0)      pn.y = grid.origin.y + static_cast<float>(cj + 1) * h + eps;
                            else if (dj < 0) pn.y = grid.origin.y + static_cast<float>(cj) * h - eps;
                            if (dk > 0)      pn.z = grid.origin.z + static_cast<float>(ck + 1) * h + eps;
                            else if (dk < 0) pn.z = grid.origin.z + static_cast<float>(ck) * h - eps;
                            // Only the velocity going INTO the chunk is removed,
                            // in the chunk's own frame. The parcel is not given
                            // an outward kick: the push above already separated
                            // them, and adding speed here is how a contact
                            // resolution turns into a spray.
                            const Vec3 sv = solidVelAtPos(p0);
                            Vec3 rel = pv - sv;
                            if (di != 0 && rel.x * static_cast<float>(di) < 0.0f) rel.x = 0.0f;
                            if (dj != 0 && rel.y * static_cast<float>(dj) < 0.0f) rel.y = 0.0f;
                            if (dk != 0 && rel.z * static_cast<float>(dk) < 0.0f) rel.z = 0.0f;
                            pv = sv + rel;
                        } else {
                            pn = p0;  // fully enclosed: wait, do not fling
                        }
                        parts.position[pi] = pn;
                        continue;
                    }
                    bool found = false;
                    int fi = ci, fj = cj, fk = ck;
                    for (int r = 1; r <= 3 && !found; ++r) {
                        for (int dk = -r; dk <= r && !found; ++dk)
                        for (int dj = -r; dj <= r && !found; ++dj)
                        for (int di = -r; di <= r && !found; ++di) {
                            if (std::max(std::abs(di), std::max(std::abs(dj), std::abs(dk))) != r) continue;
                            if (!solidCellAt(ci + di, cj + dj, ck + dk)) {
                                fi = ci + di; fj = cj + dj; fk = ck + dk;
                                found = true;
                            }
                        }
                    }
                    if (found) {
                        pn = grid.origin + Vec3((fi + 0.5f) * h, (fj + 0.5f) * h, (fk + 0.5f) * h);
                        Vec3 nrm((float)(fi - ci), (float)(fj - cj), (float)(fk - ck));
                        const float nl = nrm.length();
                        if (nl > 1e-6f) {
                            nrm = nrm * (1.0f / nl);
                            // Carry the collider's momentum: give the particle the
                            // solid velocity plus its own tangential slip, with no
                            // velocity going back INTO the solid (relative frame).
                            const Vec3 sv = solidVelAtPos(p0);
                            Vec3 rel = pv - sv;
                            const float rn = rel.x * nrm.x + rel.y * nrm.y + rel.z * nrm.z;
                            if (rn < 0.0f) rel = rel - nrm * rn;
                            pv = sv + rel;
                        }
                    }
                } else if (solidAtPos(pn)) {
                    // Separated-axis slide. A blocked axis takes the solid's
                    // velocity (a MOVING wall drags the fluid along) instead of a
                    // dead stop; a static collider's solid_vel is 0 → stop.
                    Vec3 resolved = p0;
                    Vec3 tx = resolved; tx.x = pn.x;
                    if (!solidAtPos(tx)) resolved.x = pn.x; else pv.x = solidVelAtPos(tx).x;
                    Vec3 ty = resolved; ty.y = pn.y;
                    if (!solidAtPos(ty)) resolved.y = pn.y; else pv.y = solidVelAtPos(ty).y;
                    Vec3 tz = resolved; tz.z = pn.z;
                    if (!solidAtPos(tz)) resolved.z = pn.z; else pv.z = solidVelAtPos(tz).z;
                    pn = resolved;
                }
            }

            if (boundary == APICSolverParams::BoundaryMode::Open) {
                // Crossed any wall → flow out. Leave the position untouched (it
                // is still in-bounds, so sampling stays safe) and flag for cull.
                if (pn.x < gmin.x || pn.x > gmax.x ||
                    pn.y < gmin.y || pn.y > gmax.y ||
                    pn.z < gmin.z || pn.z > gmax.z) {
                    parts.flags[pi] |= FLAG_OUTFLOW;
                    continue;
                }
                parts.position[pi] = pn;
            } else if (boundary == APICSolverParams::BoundaryMode::Periodic) {
                // Wrap to the opposite wall so the particle re-enters in-bounds.
                if (span.x > 1e-6f) { while (pn.x < gmin.x) pn.x += span.x; while (pn.x > gmax.x) pn.x -= span.x; }
                if (span.y > 1e-6f) { while (pn.y < gmin.y) pn.y += span.y; while (pn.y > gmax.y) pn.y -= span.y; }
                if (span.z > 1e-6f) { while (pn.z < gmin.z) pn.z += span.z; while (pn.z > gmax.z) pn.z -= span.z; }
                parts.position[pi] = pn;
            } else {
                // Closed: clamp to domain and dissipate wall-normal velocity.
                // Without this guard, particles stuck at the floor can feed
                // energy back into the next pressure solve and explode outward.
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
    }

    // Cull particles that flowed out of an open domain (serial swap-remove;
    // order is irrelevant to the solver). Iterate backwards so swapped-in
    // survivors are not skipped.
    if (boundary == APICSolverParams::BoundaryMode::Open && !parts.empty()) {
        for (int64_t pi = static_cast<int64_t>(parts.size()) - 1; pi >= 0; --pi) {
            if (parts.flags[static_cast<size_t>(pi)] & FLAG_OUTFLOW) {
                parts.removeSwap(static_cast<size_t>(pi));
            }
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
                                  uint32_t step_seed,
                                  std::size_t* added_out,
                                  std::size_t* removed_out) {
    if (added_out) *added_out = 0u;
    if (removed_out) *removed_out = 0u;
    // Granular particles carry persistent stress/plastic history and represent
    // material points. Liquid reseeding invents/removes samples and therefore
    // destroys both mass and constitutive history; never run it for sand.
    if (params.granular_enabled) return;
    if (!params.reseed_enabled || parts.empty()) return;
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) return;

    const int  ppc    = std::max(1, params.particles_per_cell);
    const int  target = (params.reseed_target_per_cell > 0)
        ? params.reseed_target_per_cell
        : ppc;
    const int  min_per = std::clamp(params.reseed_min_per_cell, 1, target);
    const int  max_per = std::max(target + 1, params.reseed_max_per_cell);

    // ★★ Reseed CREATES particles (parts.emit below); it does not move them.
    // Topping a starved cell from 1 up to `target` invents 7 particles' worth of
    // liquid. That is fine for the drift it was written to correct — INTERIOR
    // cells whose sample count wandered — but catastrophic at a free surface.
    //
    // A spreading sheet (a spill hitting the floor, a jet flattening out) is a
    // moving frontier of cells holding 1-2 particles, each with a healthy
    // neighbour behind it. Every step that whole frontier is amplified 4-8x, the
    // now-denser sheet spreads further, and a new frontier appears. The result
    // is exponential growth that is bounded by nothing but max_particles: the
    // liquid looks like it "suddenly dumps its entire budget" a moment after it
    // lands, and lowering the emitter's rate changes nothing because the emitter
    // is not what is producing the particles.
    //
    // The primary guard is GEOMETRIC and lives in the loop: a surface cell is
    // lifted only to `min_per`, never to the bulk `target`. That alone bounds a
    // spreading film at ~min_per particles per wetted cell — a finite number set
    // by the floor area — instead of letting the frontier reach full bulk
    // density, push further, and compound.
    // Particle count is mass in the current unit-mass P2G formulation. This
    // budget starts empty and is funded only by crowded-cell removals below;
    // reseeding may redistribute samples but cannot increase total mass.
    std::size_t reseed_budget = 0u;

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
    static std::vector<int> cell_seed_particle_buf;
    static std::vector<uint8_t> remove_mask_buf;

    particle_cell_buf.assign(parts.size(), -1);
    if (cell_count_buf.size() < total) cell_count_buf.assign(total, 0);
    else std::fill(cell_count_buf.begin(), cell_count_buf.begin() + total, 0);
    if (cell_cursor_buf.size() < total) cell_cursor_buf.assign(total, 0);
    else std::fill(cell_cursor_buf.begin(), cell_cursor_buf.begin() + total, 0);
    if (cell_seed_particle_buf.size() < total) cell_seed_particle_buf.assign(total, -1);
    else std::fill(cell_seed_particle_buf.begin(), cell_seed_particle_buf.begin() + total, -1);
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
        if (cell_seed_particle_buf[c] < 0)
            cell_seed_particle_buf[c] = static_cast<int>(pi);
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

    if (removed_out) *removed_out = static_cast<std::size_t>(removed);
    reseed_budget = static_cast<std::size_t>(removed);
    if (removed == 0) return;

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
                parts.mass_fraction[write] = parts.mass_fraction[pi];
                parts.temperature[write] = parts.temperature[pi];
                parts.combustible_fraction[write] = parts.combustible_fraction[pi];
                parts.substance_tag[write] = parts.substance_tag[pi];
                parts.uvw[write] = parts.uvw[pi];
                // Both generations move together. Compacting one and not the
                // other would misalign them by the number of removed particles,
                // and since they are blended per index the result would be a
                // liquid textured with two unrelated coordinates.
                if (pi < parts.uvw_b.size()) parts.uvw_b[write] = parts.uvw_b[pi];
            }
            ++write;
        }
        parts.position.resize(write);
        parts.velocity.resize(write);
        parts.affine.resize(write);
        parts.flags.resize(write);
        parts.mass_fraction.resize(write);
        parts.temperature.resize(write);
        parts.combustible_fraction.resize(write);
        parts.substance_tag.resize(write);
        parts.uvw.resize(write);
        if (!parts.uvw_b.empty()) parts.uvw_b.resize(write);
    }

    // Compaction changes particle indices. Rebuild the per-cell chemistry donor
    // map before any reseed emits children; using the pre-compaction index can
    // copy an unrelated substance or read past the shortened sidecars.
    std::fill(cell_seed_particle_buf.begin(), cell_seed_particle_buf.begin() + total, -1);
    for (std::size_t pi = 0; pi < parts.size(); ++pi) {
        const Vec3 gp = (parts.position[pi] - grid.origin) * invH;
        const int i = static_cast<int>(std::floor(gp.x));
        const int j = static_cast<int>(std::floor(gp.y));
        const int k = static_cast<int>(std::floor(gp.z));
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
        const std::size_t c = grid.cellIndex(i, j, k);
        if (!grid.solid[c] && cell_seed_particle_buf[c] < 0)
            cell_seed_particle_buf[c] = static_cast<int>(pi);
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
                // NOTE: this only catches DETACHED droplets. A spreading sheet
                // always has healthy neighbours behind it, so it sails straight
                // through — which is exactly the runaway case the surface test
                // below exists to stop.
                if (neighborhood_sum < std::max(4, min_per + 2) && !has_healthy_neighbor) {
                    continue;
                }

                // ★ Free-surface test: a face neighbour that is in-bounds, not
                // solid and EMPTY means this cell borders air, so it is surface,
                // not interior. Out-of-bounds and solid neighbours are walls, not
                // air — a cell resting on the floor is still interior.
                bool borders_air = false;
                const int face[6][3] = { {-1,0,0}, {1,0,0}, {0,-1,0},
                                         {0,1,0},  {0,0,-1}, {0,0,1} };
                for (const auto& f : face) {
                    const int ni = i + f[0], nj = j + f[1], nk = k + f[2];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny ||
                        nk < 0 || nk >= nz) {
                        continue;                       // wall
                    }
                    const std::size_t nc = grid.cellIndex(ni, nj, nk);
                    if (grid.solid[nc]) continue;       // wall
                    if (cell_count_buf[nc] == 0) { borders_air = true; break; }
                }

                // ★★ Surface cells are never topped up. Every particle this
                // function emits is INVENTED liquid (P2G is unit-mass and never
                // reads mass_fraction, so there is no way to split instead), and
                // at a free surface a low count is the truth: a sheet one cell
                // deep really does hold a couple of particles per cell. Filling
                // those cells is what made a spill gain mass the moment it
                // landed — drops hit the floor, spread into fresh cells, and
                // each fresh cell was inflated to bulk density.
                //
                // Interior cells are a different question and keep the original
                // behaviour: they are enclosed, they cannot form an expanding
                // frontier, and a count that has drifted there really is drift.
                //
                // ★ This is only safe because the air-drag stage no longer keys
                // off `reseed_min_per_cell`. It used to, so starving the surface
                // also classified the whole outer layer as spray — the two
                // faults hid each other. Air drag now tests ISOLATION (3x3x3
                // neighbourhood) instead; keep them decoupled.
                if (borders_air) continue;

                const int need = target - count;
                if (parts.size() >= params.max_particles) {
                    continue;
                }
                if (reseed_budget == 0u) continue;
                const int actual_need = std::min<int>(
                    std::min<int>(need, static_cast<int>(params.max_particles - parts.size())),
                    static_cast<int>(reseed_budget));
                if (actual_need <= 0) continue;
                reseed_budget -= static_cast<std::size_t>(actual_need);

                const Vec3 cell_min = grid.origin + Vec3(static_cast<float>(i) * h,
                                                          static_cast<float>(j) * h,
                                                          static_cast<float>(k) * h);
                for (int n = 0; n < actual_need; ++n) {
                    const Vec3 p = cell_min + Vec3(dist01(rng), dist01(rng), dist01(rng)) * h;
                    const Vec3 v = grid.sampleVelocity(p);
                    const int parent = cell_seed_particle_buf[c];
                    const float temperature = parent >= 0 &&
                        static_cast<std::size_t>(parent) < parts.temperature.size()
                        ? parts.temperature[static_cast<std::size_t>(parent)] : 0.0f;
                    const float combustible = parent >= 0 &&
                        static_cast<std::size_t>(parent) < parts.combustible_fraction.size()
                        ? parts.combustible_fraction[static_cast<std::size_t>(parent)] : 0.0f;
                    const uint32_t substance = parent >= 0 &&
                        static_cast<std::size_t>(parent) < parts.substance_tag.size()
                        ? parts.substance_tag[static_cast<std::size_t>(parent)] : 0u;
                    // ★ Reseed does not TRANSPORT liquid, it CREATES particles —
                    // and a created particle with no material coordinate is a
                    // hole in the texture. Inherit the donor's, carrying the
                    // spawn OFFSET across so the child lands at the coordinate
                    // its own position implies rather than stacking every child
                    // of one donor onto a single texel (which would show as a
                    // smeared blob wherever the top-up is busy).
                    //
                    // No donor means no continuity to inherit: fall through to
                    // emit's default (uvw = birth position). That is a seam, but
                    // it is an HONEST one — there is no neighbour to be
                    // continuous with — and it can only happen in a cell whose
                    // only particles left the grid this step.
                    //
                    // ★ BOTH generations are inherited, each from its own donor
                    // value. They were reset at different times, so the donor's
                    // two coordinates genuinely differ; copying one into both
                    // would fuse the pair into a single generation for every
                    // reseeded particle — the stretch cure would keep reporting
                    // as enabled while quietly not applying in splashes.
                    Vec3 child_uvw, child_uvw_b;
                    const Vec3* child_uvw_ptr = nullptr;
                    const Vec3* child_uvw_b_ptr = nullptr;
                    if (parent >= 0 &&
                        static_cast<std::size_t>(parent) < parts.uvw.size()) {
                        const std::size_t pidx = static_cast<std::size_t>(parent);
                        const Vec3 offset = p - parts.position[pidx];
                        child_uvw = parts.uvw[pidx] + offset;
                        child_uvw_ptr = &child_uvw;
                        if (pidx < parts.uvw_b.size()) {
                            child_uvw_b = parts.uvw_b[pidx] + offset;
                            child_uvw_b_ptr = &child_uvw_b;
                        }
                    }
                    parts.emit(p, v, temperature, combustible, substance,
                               child_uvw_ptr, child_uvw_b_ptr);
                }
                if (added_out) *added_out += static_cast<std::size_t>(actual_need);
            }
        }
    }
}

void applyExternalForces(FluidParticles& particles,
                         const FluidSim::FluidGrid& grid,
                         const APICSolverParams& params,
                         const SimulationForceFieldSnapshot* forces,
                         float time_seconds,
                         float dt) {
    if (particles.empty() || dt <= 0.0f) return;

    const int thread_count = solverThreadCount(params);
    const bool particle_parallel = shouldParallelParticles(params, particles.size());
    auto clampVelocity = [&](Vec3& v) {
        const float speed = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        if (speed > params.max_velocity && speed > 1e-6f) {
            v = v * (params.max_velocity / speed);
        }
    };

    // Wind is special on liquids. Applied as a uniform body acceleration it makes
    // water feel weightless: every particle (surface and deep) speeds up without
    // limit. So for Wind fields flagged fluid_surface_drag we instead apply a
    // relative-velocity surface drag — the horizontal velocity is pulled toward
    // the wind speed and saturates there (weight), and the push fades out over a
    // band below the free surface (deep water stays calm). Every other field type,
    // and Wind on non-fluid systems, keeps the body-force path.
    const uint32_t fluid_mask = toSimulationSystemMask(SimulationSystemKind::Fluid);
    const std::vector<const Physics::ForceField*>* active_fields =
        (forces && !forces->empty()) ? &forces->activeFields() : nullptr;
    const std::vector<PackedForceField>* packed_fields =
        (forces && !forces->empty()) ? &forces->packedFields() : nullptr;

    // Collect Wind fields that drive this liquid as a surface drag.
    std::vector<const Physics::ForceField*> wind_drag_fields;
    if (active_fields) {
        for (size_t fi = 0; fi < active_fields->size(); ++fi) {
            const Physics::ForceField* f = (*active_fields)[fi];
            if (!f) continue;
            if (packed_fields && fi < packed_fields->size() &&
                ((*packed_fields)[fi].affect_mask & fluid_mask) == 0u) continue;
            if (f->type == Physics::ForceFieldType::Wind && f->fluid_surface_drag)
                wind_drag_fields.push_back(f);
        }
    }

    // Free-surface height per XZ column (highest particle.y), built once and
    // sampled per particle to confine the wind push to the surface band.
    const int cnx = std::max(1, grid.nx);
    const int cnz = std::max(1, grid.nz);
    const float inv_h = grid.voxel_size > 1e-6f ? 1.0f / grid.voxel_size : 0.0f;
    std::vector<float> column_top_y;
    const bool wind_drag_active = !wind_drag_fields.empty() && inv_h > 0.0f;
    if (wind_drag_active) {
        column_top_y.assign(static_cast<size_t>(cnx) * static_cast<size_t>(cnz),
                            -std::numeric_limits<float>::max());
        for (size_t pi = 0; pi < particles.position.size(); ++pi) {
            const Vec3& p = particles.position[pi];
            int gx = std::clamp(static_cast<int>(std::floor((p.x - grid.origin.x) * inv_h)), 0, cnx - 1);
            int gz = std::clamp(static_cast<int>(std::floor((p.z - grid.origin.z) * inv_h)), 0, cnz - 1);
            float& top = column_top_y[static_cast<size_t>(gz) * cnx + gx];
            if (p.y > top) top = p.y;
        }
    }

    auto windDragAccel = [&](const Vec3& pos, const Vec3& v) -> Vec3 {
        int gx = std::clamp(static_cast<int>(std::floor((pos.x - grid.origin.x) * inv_h)), 0, cnx - 1);
        int gz = std::clamp(static_cast<int>(std::floor((pos.z - grid.origin.z) * inv_h)), 0, cnz - 1);
        const float surf = column_top_y[static_cast<size_t>(gz) * cnx + gx];
        Vec3 a(0.0f, 0.0f, 0.0f);
        for (const Physics::ForceField* f : wind_drag_fields) {
            // Surface band weight: 1 at/above the free surface, 0 once the
            // particle is fluid_surface_depth below it.
            const float band = std::max(1e-4f, f->fluid_surface_depth);
            const float depth = surf - pos.y;          // >0 below the surface
            const float w = 1.0f - std::clamp(depth / band, 0.0f, 1.0f);
            if (w <= 0.0f) continue;

            // Shaped (non-Infinite) wind falls off radially from its centre.
            float fall = 1.0f;
            if (f->shape != Physics::ForceFieldShape::Infinite)
                fall = f->calculateFalloff((pos - f->position).length());
            if (fall <= 0.0f) continue;

            // strength = target surface speed (m/s) along the wind direction.
            const float dl = f->direction.length();
            Vec3 wind_vel = dl > 1e-6f ? f->direction * (f->strength / dl) : Vec3(0,0,0);
            if (f->fluid_curl_detail > 0.0f)
                wind_vel = wind_vel + f->sampleCurlDetail(pos, time_seconds) *
                                      (f->strength * f->fluid_curl_detail);

            Vec3 rel = wind_vel * (w * fall) - v;
            rel.y = 0.0f;  // horizontal only — never fight gravity/buoyancy
            a = a + rel * (f->fluid_drag_coupling * w * fall);
        }
        return a;
    };

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(particle_parallel)
#endif
    for (int64_t raw_i = 0; raw_i < static_cast<int64_t>(particles.velocity.size()); ++raw_i) {
        const size_t pi = static_cast<size_t>(raw_i);
        Vec3& v = particles.velocity[pi];
        const Vec3& pos = particles.position[pi];
        Vec3 acceleration = params.gravity;
        if (active_fields) {
            // Body-force fields (everything except surface-drag wind). Mirrors
            // SimulationForceFieldSnapshot::evaluateAt's mask + dispatch.
            for (size_t fi = 0; fi < active_fields->size(); ++fi) {
                const Physics::ForceField* f = (*active_fields)[fi];
                if (!f) continue;
                if (packed_fields && fi < packed_fields->size() &&
                    ((*packed_fields)[fi].affect_mask & fluid_mask) == 0u) continue;
                if (f->type == Physics::ForceFieldType::Wind && f->fluid_surface_drag) continue;
                acceleration = acceleration + f->evaluate(pos, time_seconds, v);
            }
            if (wind_drag_active)
                acceleration = acceleration + windDragAccel(pos, v);
        }
        v = v + acceleration * dt;
        clampVelocity(v);
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

    // ── Solid-phase parcels ─────────────────────────────────────────────────
    // Resolved ONCE per step into a per-particle mask. The tag list is short
    // (kMaxFluidSubstanceMaterials at most), but the particle stages that need
    // the answer run per substep, so a scan there would multiply by CFL.
    //
    // ★ Null mask, not an empty one, when the domain declares no solid: every
    // consumer then compiles down to exactly the code it ran before this
    // existed, which is what keeps liquid-only domains bit-identical.
    static std::vector<uint8_t> s_solid_particle;
    const uint8_t* solid_particle = nullptr;
    if (params.solid_substance_tags && !params.solid_substance_tags->empty() &&
        particles.substance_tag.size() >= particles.size()) {
        const auto& tags = *params.solid_substance_tags;
        s_solid_particle.assign(particles.size(), 0u);
        bool any = false;
        for (std::size_t pi = 0; pi < particles.size(); ++pi) {
            const uint32_t tag = particles.substance_tag[pi];
            // ★ Tag 0 is untagged liquid, never solid — see
            // buildSubstanceSolidCells for why that matters.
            if (tag == kSubstanceUntagged) continue;
            for (uint32_t st : tags) {
                if (st != tag) continue;
                s_solid_particle[pi] = 1u;
                any = true;
                break;
            }
        }
        if (any) solid_particle = s_solid_particle.data();
    }

    // 1. External forces on particles (gravity + force fields, incl. the wind
    // surface-drag model). Factored into applyExternalForces so the GPU pipeline
    // can run it on the CPU then upload the post-force velocities for a GPU P2G.
    auto stage_begin = SolverClock::now();
    if (!params.external_forces_preintegrated) {
        applyExternalForces(particles, grid, params, forces, time_seconds, dt);
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

    // ── 3. FLIP snapshot ────────────────────────────────────────────────────
    // ★ ROOT CAUSE OF "THE VISCOSITY DIAL DOES NOTHING".
    // The FLIP update is  v_p = v_grid_post + flip·(v_p_old - v_snapshot).
    // This snapshot used to be taken AFTER boundary+viscosity, so writing
    // v_grid_post = v_visc + Δp and v_snapshot = v_visc gives
    //     v_p = (1-flip)·v_visc + flip·v_p_old + Δp
    // — the viscous field survives with weight (1-flip). At the water default
    // flip_blend=0.97 that is THREE PERCENT, and the FLIP fraction simply kept
    // its own old velocity, untouched by any viscosity at all. Every viscous
    // preset was therefore forced to fake thickness with per-particle friction
    // and damping, which drags the whole body instead of resisting shear.
    // Snapshotting here — right after the transfer, before any grid-side force —
    // is the standard FLIP contract: EVERY grid change (solid boundaries,
    // viscosity, pressure) becomes part of the delta the particles receive.
    static std::vector<float> vel_x_pre_buf;
    static std::vector<float> vel_y_pre_buf;
    static std::vector<float> vel_z_pre_buf;
    const bool want_flip = params.flip_blend > 0.0f && params.free_surface;
    if (!params.pressure_g2p_precomputed && want_flip) {
        if (vel_x_pre_buf.size() < grid.vel_x.size()) vel_x_pre_buf.resize(grid.vel_x.size());
        if (vel_y_pre_buf.size() < grid.vel_y.size()) vel_y_pre_buf.resize(grid.vel_y.size());
        if (vel_z_pre_buf.size() < grid.vel_z.size()) vel_z_pre_buf.resize(grid.vel_z.size());
        std::copy(grid.vel_x.begin(), grid.vel_x.end(), vel_x_pre_buf.begin());
        std::copy(grid.vel_y.begin(), grid.vel_y.end(), vel_y_pre_buf.begin());
        std::copy(grid.vel_z.begin(), grid.vel_z.end(), vel_z_pre_buf.begin());
        // Publish for the GPU split-step: the caller uploads THIS, not
        // grid.vel_*, which by then carries the viscous solve as well. Only on
        // the split-step path — a CPU-only step reads vel_*_pre_buf directly, so
        // copying three more face arrays for it would be pure waste.
        if (params.stop_before_pressure) {
            g_flip_snap_valid = true;
            g_flip_snap_x.assign(vel_x_pre_buf.begin(),
                                 vel_x_pre_buf.begin() + grid.vel_x.size());
            g_flip_snap_y.assign(vel_y_pre_buf.begin(),
                                 vel_y_pre_buf.begin() + grid.vel_y.size());
            g_flip_snap_z.assign(vel_z_pre_buf.begin(),
                                 vel_z_pre_buf.begin() + grid.vel_z.size());
        }
    } else if (!params.pressure_g2p_precomputed) {
        g_flip_snap_valid = false;
        g_flip_snap_x.clear(); g_flip_snap_y.clear(); g_flip_snap_z.clear();
    }

    // 3b. Solid boundary enforcement + viscous diffusion.
    //     Skipped in GPU split-step's second call (pressure_g2p_precomputed=true)
    //     because these were already run in the first call.
    if (!params.pressure_g2p_precomputed) {
        stage_begin = SolverClock::now();
        enforceSolidBoundaries(grid, params);
        stage_end = SolverClock::now();
        out_stats.boundary_ms = elapsedMs(stage_begin, stage_end);

        stage_begin = SolverClock::now();
        if (!params.viscosity_precomputed) {
            // The viscous stencil needs to tell fluid faces from air faces —
            // that distinction IS the stress-free surface condition. Build the
            // same cell mask the pressure solve and the GPU kernel use.
            static std::vector<float> visc_mask;
            const std::vector<float>* mask_ptr = nullptr;
            // ★ The per-substance field counts as "there is viscosity here"
            // exactly like the scalar does. Without this, an inviscid domain
            // with a thick substance would run the stencil with NO free-surface
            // mask, so it would treat air faces as walls and brake the surface
            // against nothing — a viscous skin, not a viscous liquid.
            const bool nu_field_present =
                params.substance_viscosity != nullptr &&
                params.substance_viscosity->size() == grid.getCellCount();
            if (params.free_surface &&
                (params.kinematic_viscosity > 0.0f || nu_field_present)) {
                buildViscosityFluidMask(grid, particles, visc_mask);
                mask_ptr = &visc_mask;
            }
            int sweeps_run = 0;
            applyViscosity(grid, params, dt, mask_ptr, sweeps_run);
            out_stats.viscosity_sweeps_run = sweeps_run;
            if (sweeps_run > 0) {
                enforceSolidBoundaries(grid, params);
            }
        }
        stage_end = SolverClock::now();
        out_stats.viscosity_ms = elapsedMs(stage_begin, stage_end);
    }

    // GPU split-step call 1: return after P2G + snapshot + boundary (+ CPU
    // viscosity when the device did not take it) so the caller can run the
    // device viscosity → pressure → G2P before the second call does the tail.
    if (params.stop_before_pressure) {
        out_stats.total_ms = elapsedMs(total_begin, SolverClock::now());
        return;
    }

    // 4–6. Pressure projection + G2P.
    //       When pressure_g2p_precomputed is set, the GPU path has already
    //       run these stages and downloaded updated particle velocities
    //       and affine matrices to CPU. Skip them here; advect + reseed still
    //       run on CPU below using the GPU-written velocity data.
    stage_begin = SolverClock::now();
    if (!params.pressure_g2p_precomputed) {
        // 5. Pressure projection (incompressibility).
        auto pressure_begin = SolverClock::now();
        if (params.free_surface) {
            out_stats.active_fluid_cells = projectPressureFreeSurface(particles,
                                                                      grid,
                                                                      params,
                                                                      dt,
                                                                      out_stats.sealed_pockets,
                                                                      out_stats.sealed_pocket_cells,
                                                                      out_stats.interior_fluid_cells);
            out_stats.sealed_pockets_measured = true;
        } else {
            GridFluid::SolverParams pp{};
            pp.pressure_iterations = params.pressure_iterations;
            pp.sor_omega           = params.sor_omega;
            pp.boundary            = GridFluid::Boundary::Closed;
            GridFluid::projectPressure(grid, pp, dt);
            out_stats.active_fluid_cells = out_stats.grid_cell_count;
        }
        out_stats.pressure_ms = elapsedMs(pressure_begin, SolverClock::now());

        // 6. Grid -> particle (APIC affine + PIC/FLIP linear blend + friction)
        gridToParticle(particles,
                       grid,
                       params,
                       dt,
                       want_flip ? vel_x_pre_buf.data() : nullptr,
                       want_flip ? vel_y_pre_buf.data() : nullptr,
                       want_flip ? vel_z_pre_buf.data() : nullptr,
                       solid_particle);
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
    if (!params.particle_tail_precomputed &&
        params.air_drag > 0.0f && dt > 0.0f) {
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
            if (count >= air_threshold) continue; // dense cell — certainly bulk

            // ★★ A sparse CELL is not the same thing as a detached DROPLET.
            //
            // This used to drag every particle whose own cell held fewer than
            // air_threshold particles. A thin sheet of liquid lying on the floor
            // is exactly that — one cell deep, a couple of particles per cell —
            // so the entire spread of a spill was being treated as airborne
            // spray and given quadratic drag. It stopped following a moving
            // container and visibly scattered. Reseed used to hide this by
            // pumping every sparse cell up to the bulk target, which is how the
            // liquid gained mass on impact; the two faults were propping each
            // other up.
            //
            // Spray is defined by ISOLATION, not by local thinness: sum the
            // 3x3x3 neighbourhood, and only drag a particle whose neighbourhood
            // is also nearly empty. A film has plenty of in-plane neighbours; a
            // droplet flying through air has none. Only reached by particles
            // that already failed the cheap test above, so the 27 taps cost
            // nothing on bulk fluid.
            const int isolation_threshold = std::max(4, air_threshold + 2);
            int neighbourhood = 0;
            for (int dk = -1; dk <= 1 && neighbourhood < isolation_threshold; ++dk) {
                const int nk2 = k + dk;
                if (nk2 < 0 || nk2 >= nz) continue;
                for (int dj = -1; dj <= 1 && neighbourhood < isolation_threshold; ++dj) {
                    const int nj2 = j + dj;
                    if (nj2 < 0 || nj2 >= ny) continue;
                    for (int di = -1; di <= 1; ++di) {
                        const int ni2 = i + di;
                        if (ni2 < 0 || ni2 >= nx) continue;
                        neighbourhood += air_cell_count_buf[grid.cellIndex(ni2, nj2, nk2)];
                    }
                }
            }
            if (neighbourhood >= isolation_threshold) continue; // sheet/film — bulk

            Vec3& v = particles.velocity[pi];
            const float speed_sq = v.x * v.x + v.y * v.y + v.z * v.z;
            if (speed_sq < 1e-8f) continue;
            const float speed = std::sqrt(speed_sq);
            const float decay = 1.0f / (1.0f + air_k * speed * dt);
            v = v * decay;
        }
    }

    if (!params.particle_tail_precomputed) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(thread_count) if(particle_parallel)
#endif
        for (int64_t raw_i = 0; raw_i < static_cast<int64_t>(particles.velocity.size()); ++raw_i) {
            Vec3& v = particles.velocity[static_cast<size_t>(raw_i)];
            v = v * std::clamp(params.velocity_damping, 0.0f, 1.0f);
            clampVelocity(v);
        }
    }
    stage_end = SolverClock::now();
    out_stats.g2p_ms = elapsedMs(stage_begin, stage_end); // includes air drag + damping

    // 7. Advect particle positions
    stage_begin = SolverClock::now();
    out_stats.advect_substeps = params.particle_tail_precomputed ? 0 :
        advectParticles(particles,
                        grid,
                        dt,
                        params,
                        params.max_velocity,
                        params.wall_damping,
                        params.cfl,
                        params.max_substeps,
                        solid_particle);
    stage_end = SolverClock::now();
    out_stats.advect_ms = elapsedMs(stage_begin, stage_end);

    // 8. Per-cell redistribution. Cull over-populated cells and top up
    //    starved ones so the seed-time particle density stays bounded. Uses
    //    the post-advect positions and the post-projection grid (sampled by
    //    new particles for their initial velocity).
    const uint32_t reseed_seed =
        static_cast<uint32_t>(out_stats.particle_count) * 2654435761u ^
        static_cast<uint32_t>(std::llround(time_seconds * 1000.0));
    redistributeParticles(particles, grid, params, reseed_seed,
                          &out_stats.reseed_added_particles,
                          &out_stats.reseed_removed_particles);
    out_stats.particle_count = particles.size();

    // 9. Advance the material-coordinate refresh schedule and reset whichever
    //    generation is due.
    //
    //    ★ LAST, and after reseed on purpose. A generation reset assigns
    //    uvw = position, so it must see the FINAL particle set of this step:
    //    reset before advection and the coordinate describes where the liquid
    //    was, not where it is; reset before reseed and the particles created
    //    this step never get the fresh generation and stay one period stale.
    //
    //    ★ The early-out at the top of this function (empty or dt <= 0) skips
    //    this too, which is correct — a step that did not advance the liquid
    //    must not age its coordinates either, or a paused sim would keep
    //    refreshing a body that is not deforming.
    particles.uvw_refresh_period = params.uvw_refresh_period;
    particles.advanceMaterialCoordinates();

    out_stats.total_ms = elapsedMs(total_begin, stage_end);
}

} // namespace Fluid
} // namespace RayTrophiSim
