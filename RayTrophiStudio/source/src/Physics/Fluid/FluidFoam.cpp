/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FluidFoam.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Ihmsen et al. 2012 whitewater. See FluidFoam.h for the model overview.
 */

#include "Fluid/FluidFoam.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace RayTrophiSim {
namespace Fluid {

namespace {

inline float lengthSq(const Vec3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }

// Normalise a criterion into [0,1] over [lo,hi] (Ihmsen clamp).
inline float clamp01(float v, float lo, float hi) {
    if (hi <= lo) return v > lo ? 1.0f : 0.0f;
    float t = (v - lo) / (hi - lo);
    return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}

} // namespace

void stepFoam(const FluidParticles& fluid,
              const FluidSim::FluidGrid& grid,
              FoamParticles& foam,
              const FoamParams& params,
              const Vec3& gravity,
              float dt,
              uint32_t seed,
              FoamStats* stats)
{
    using clock = std::chrono::steady_clock;
    if (stats) *stats = FoamStats{};

    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float voxel = grid.voxel_size;
    if (!params.enabled || nx <= 0 || ny <= 0 || nz <= 0 || voxel <= 0.0f || dt <= 0.0f) {
        if (stats) { stats->alive = foam.size(); }
        return;
    }

    const Vec3  origin = grid.origin;
    const Vec3  dom_min = origin;
    const Vec3  dom_max = origin + Vec3(nx * voxel, ny * voxel, nz * voxel);
    const std::size_t cell_count = static_cast<std::size_t>(nx) * ny * nz;
    const std::size_t fluid_count = fluid.size();

    const float h     = std::max(1e-4f, params.neighbor_radius_voxels * voxel);
    const float h_sq  = h * h;
    const float inv_h = 1.0f / voxel;
    const int   reach = std::max(1, static_cast<int>(std::ceil(params.neighbor_radius_voxels)));

    int thread_cap = 1;
#ifdef _OPENMP
    thread_cap = std::max(1, omp_get_max_threads());
#endif

    auto cellIndex = [nx, ny](int i, int j, int k) -> std::size_t {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    };

    // ── 1. CSR bin of the FLUID particles (used by both generation + foam
    //       classification). Function-static scratch reused across steps.
    constexpr std::size_t kInvalid = static_cast<std::size_t>(-1);
    static std::vector<std::size_t> particle_cell;
    static std::vector<int>         cell_count_per;
    static std::vector<std::size_t> cell_offset;
    static std::vector<std::size_t> cell_csr;
    static std::vector<std::size_t> cursor;

    if (fluid_count == 0) {
        // No source liquid: still advect/age existing foam below (it can persist).
    }

    if (particle_cell.size() < fluid_count) particle_cell.assign(fluid_count, kInvalid);
    else std::fill(particle_cell.begin(), particle_cell.begin() + fluid_count, kInvalid);
    if (cell_count_per.size() < cell_count) cell_count_per.assign(cell_count, 0);
    else std::fill(cell_count_per.begin(), cell_count_per.begin() + cell_count, 0);

    for (std::size_t p = 0; p < fluid_count; ++p) {
        const Vec3& wp = fluid.position[p];
        if (!std::isfinite(wp.x) || !std::isfinite(wp.y) || !std::isfinite(wp.z)) continue;
        const Vec3 local = (wp - origin) * inv_h;
        const int i = static_cast<int>(std::floor(local.x));
        const int j = static_cast<int>(std::floor(local.y));
        const int k = static_cast<int>(std::floor(local.z));
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
        const std::size_t ci = cellIndex(i, j, k);
        particle_cell[p] = ci;
        ++cell_count_per[ci];
    }
    cell_offset.assign(cell_count + 1, 0);
    for (std::size_t c = 0; c < cell_count; ++c)
        cell_offset[c + 1] = cell_offset[c] + static_cast<std::size_t>(cell_count_per[c]);
    cell_csr.assign(cell_offset.back(), 0);
    cursor.assign(cell_count, 0);
    for (std::size_t p = 0; p < fluid_count; ++p) {
        const std::size_t ci = particle_cell[p];
        if (ci == kInvalid) continue;
        cell_csr[cell_offset[ci] + cursor[ci]++] = p;
    }

    // Count fluid neighbours of a world point (submersion test for foam).
    auto countFluidNeighbours = [&](const Vec3& x) -> int {
        const Vec3 lp = (x - origin) * inv_h;
        const int ci = static_cast<int>(std::floor(lp.x));
        const int cj = static_cast<int>(std::floor(lp.y));
        const int ck = static_cast<int>(std::floor(lp.z));
        if (ci < -reach || ci > nx + reach) return 0;
        int n = 0;
        const int i0 = std::max(0, ci - reach), i1 = std::min(nx - 1, ci + reach);
        const int j0 = std::max(0, cj - reach), j1 = std::min(ny - 1, cj + reach);
        const int k0 = std::max(0, ck - reach), k1 = std::min(nz - 1, ck + reach);
        for (int kk = k0; kk <= k1; ++kk)
        for (int jj = j0; jj <= j1; ++jj)
        for (int ii = i0; ii <= i1; ++ii) {
            const std::size_t nci = cellIndex(ii, jj, kk);
            for (std::size_t a = cell_offset[nci]; a < cell_offset[nci + 1]; ++a) {
                if (lengthSq(fluid.position[cell_csr[a]] - x) < h_sq) ++n;
            }
        }
        return n;
    };

    // ── 2. Advect / classify / age existing foam ────────────────────────────
    const auto t_adv0 = clock::now();
    const std::size_t foam_count = foam.size();
    if (foam_count > 0) {
        const int spray_max  = params.spray_max_neighbors;
        const int bubble_min = params.bubble_min_neighbors;
        const float drag_f   = std::max(0.0f, params.fluid_drag);
        const float drag_s   = std::max(0.0f, params.spray_drag);
        const float buoy     = std::max(0.0f, params.buoyancy);

        // Interior solid (collider) collision — foam advects in the grid velocity
        // field, but like the liquid particles it carries its own momentum and
        // would otherwise tunnel through / sit inside a voxelized collider. Mirror
        // the liquid advect's eject + separated-axis slide so foam respects the
        // collider too (and a MOVING collider shoves it via solid_vel).
        const bool foam_has_solid = !grid.solid.empty();
        const bool foam_has_svel  = grid.solid_vel.size() == grid.solid.size();
        auto solidCellAt = [&](int i, int j, int k) -> bool {
            if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return false;
            return grid.solid[cellIndex(i, j, k)] != 0u;
        };
        auto solidAtPos = [&](const Vec3& p) -> bool {
            const Vec3 g = (p - origin) * inv_h;
            return solidCellAt(static_cast<int>(std::floor(g.x)),
                               static_cast<int>(std::floor(g.y)),
                               static_cast<int>(std::floor(g.z)));
        };
        auto solidVelAtPos = [&](const Vec3& p) -> Vec3 {
            if (!foam_has_svel) return Vec3(0.0f, 0.0f, 0.0f);
            const Vec3 g = (p - origin) * inv_h;
            const int i = static_cast<int>(std::floor(g.x));
            const int j = static_cast<int>(std::floor(g.y));
            const int k = static_cast<int>(std::floor(g.z));
            if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return Vec3(0.0f, 0.0f, 0.0f);
            return grid.solid_vel[cellIndex(i, j, k)];
        };

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 512) num_threads(thread_cap)
#endif
        for (int64_t f = 0; f < static_cast<int64_t>(foam_count); ++f) {
            const std::size_t fi = static_cast<std::size_t>(f);
            Vec3 x = foam.position[fi];
            const Vec3 p0 = x;
            Vec3 v = foam.velocity[fi];

            const int neigh = countFluidNeighbours(x);
            FoamType type;
            if      (neigh <= spray_max)  type = FoamType::Spray;
            else if (neigh >= bubble_min) type = FoamType::Bubble;
            else                          type = FoamType::Foam;
            foam.type[fi] = static_cast<uint8_t>(type);

            const Vec3 vf = grid.sampleVelocity(x);
            if (type == FoamType::Spray) {
                // Ballistic: gravity + air drag.
                v = v + gravity * dt;
                const float k = 1.0f / (1.0f + drag_s * dt);
                v = v * k;
            } else if (type == FoamType::Bubble) {
                // Buoyant rise against gravity, dragged toward the local flow.
                v = v + gravity * (-buoy) * dt;
                const float a = std::min(1.0f, drag_f * dt);
                v = v + (vf - v) * a;
            } else {
                // Foam: advected with the fluid (strong coupling).
                const float a = std::min(1.0f, drag_f * dt);
                v = v + (vf - v) * a;
            }

            x = x + v * dt;

            // Solid (collider) collision — same two cases as the liquid advect.
            if (foam_has_solid) {
                if (solidAtPos(p0)) {
                    // Collider swept over the foam particle → eject to nearest free cell.
                    const Vec3 g0 = (p0 - origin) * inv_h;
                    const int ci = static_cast<int>(std::floor(g0.x));
                    const int cj = static_cast<int>(std::floor(g0.y));
                    const int ck = static_cast<int>(std::floor(g0.z));
                    bool found = false; int ffi = ci, ffj = cj, ffk = ck;
                    for (int r = 1; r <= 3 && !found; ++r) {
                        for (int dk = -r; dk <= r && !found; ++dk)
                        for (int dj = -r; dj <= r && !found; ++dj)
                        for (int di = -r; di <= r && !found; ++di) {
                            if (std::max(std::abs(di), std::max(std::abs(dj), std::abs(dk))) != r) continue;
                            if (!solidCellAt(ci + di, cj + dj, ck + dk)) { ffi = ci + di; ffj = cj + dj; ffk = ck + dk; found = true; }
                        }
                    }
                    if (found) {
                        x = origin + Vec3((ffi + 0.5f) * voxel, (ffj + 0.5f) * voxel, (ffk + 0.5f) * voxel);
                        Vec3 nrm((float)(ffi - ci), (float)(ffj - cj), (float)(ffk - ck));
                        const float nl = nrm.length();
                        if (nl > 1e-6f) {
                            nrm = nrm * (1.0f / nl);
                            const Vec3 sv = solidVelAtPos(p0);
                            Vec3 rel = v - sv;
                            const float rn = rel.x * nrm.x + rel.y * nrm.y + rel.z * nrm.z;
                            if (rn < 0.0f) rel = rel - nrm * rn;
                            v = sv + rel;
                        }
                    }
                } else if (solidAtPos(x)) {
                    Vec3 resolved = p0;
                    Vec3 tx = resolved; tx.x = x.x;
                    if (!solidAtPos(tx)) resolved.x = x.x; else v.x = solidVelAtPos(tx).x;
                    Vec3 ty = resolved; ty.y = x.y;
                    if (!solidAtPos(ty)) resolved.y = x.y; else v.y = solidVelAtPos(ty).y;
                    Vec3 tz = resolved; tz.z = x.z;
                    if (!solidAtPos(tz)) resolved.z = x.z; else v.z = solidVelAtPos(tz).z;
                    x = resolved;
                }
            }

            float life = foam.lifetime[fi] - dt;
            const bool oob = (x.x < dom_min.x || x.x > dom_max.x ||
                              x.y < dom_min.y || x.y > dom_max.y ||
                              x.z < dom_min.z || x.z > dom_max.z);
            if (oob) life = -1.0f;

            foam.position[fi] = x;
            foam.velocity[fi] = v;
            foam.lifetime[fi] = life;
        }

        // Serial compaction: swap-remove dead (lifetime <= 0). Forward scan that
        // re-checks the swapped-in element (removeSwap brings the last element to
        // i), so a dead element moved down is never skipped.
        std::size_t fi = 0;
        while (fi < foam.size()) {
            if (foam.lifetime[fi] <= 0.0f) foam.removeSwap(fi);
            else ++fi;
        }
    }
    if (stats) stats->advect_ms = std::chrono::duration<float, std::milli>(clock::now() - t_adv0).count();

    // ── 3. Generation potentials (Ihmsen) per fluid particle ────────────────
    const auto t_gen0 = clock::now();
    if (fluid_count > 0 && (params.trapped_air_rate > 0.0f || params.wave_crest_rate > 0.0f)) {
        static std::vector<float> expected;     // expected spawn count per particle
        if (expected.size() < fluid_count) expected.resize(fluid_count);

        const float k_ta = params.trapped_air_rate;
        const float k_wc = params.wave_crest_rate;
        const float crest_cos = params.crest_cos;

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 512) num_threads(thread_cap)
#endif
        for (int64_t p = 0; p < static_cast<int64_t>(fluid_count); ++p) {
            const std::size_t pi = static_cast<std::size_t>(p);
            expected[pi] = 0.0f;
            if (particle_cell[pi] == kInvalid) continue;
            const Vec3 xi = fluid.position[pi];
            const Vec3 vi = fluid.velocity[pi];

            const Vec3 lp = (xi - origin) * inv_h;
            const int ci = static_cast<int>(std::floor(lp.x));
            const int cj = static_cast<int>(std::floor(lp.y));
            const int ck = static_cast<int>(std::floor(lp.z));
            const int i0 = std::max(0, ci - reach), i1 = std::min(nx - 1, ci + reach);
            const int j0 = std::max(0, cj - reach), j1 = std::min(ny - 1, cj + reach);
            const int k0 = std::max(0, ck - reach), k1 = std::min(nz - 1, ck + reach);

            float wsum = 0.0f; Vec3 mean(0, 0, 0); int ncount = 0;
            float ita = 0.0f;
            for (int kk = k0; kk <= k1; ++kk)
            for (int jj = j0; jj <= j1; ++jj)
            for (int ii = i0; ii <= i1; ++ii) {
                const std::size_t nci = cellIndex(ii, jj, kk);
                for (std::size_t a = cell_offset[nci]; a < cell_offset[nci + 1]; ++a) {
                    const std::size_t pj = cell_csr[a];
                    if (pj == pi) continue;
                    const Vec3 xij = xi - fluid.position[pj];
                    const float d2 = lengthSq(xij);
                    if (d2 >= h_sq) continue;
                    const float d = std::sqrt(d2);
                    const float W = 1.0f - d / h;     // linear falloff
                    wsum += W;
                    const Vec3& xj = fluid.position[pj];
                    mean.x += W * xj.x; mean.y += W * xj.y; mean.z += W * xj.z;
                    ++ncount;
                    // Trapped air: relative speed weighted by how convergent the
                    // pair is. v_ij anti-parallel to x_ij (approaching) → factor ~2.
                    const Vec3 vij = vi - fluid.velocity[pj];
                    const float sv = std::sqrt(lengthSq(vij));
                    if (sv > 1e-5f && d > 1e-5f) {
                        const float cosvx = (vij.x * xij.x + vij.y * xij.y + vij.z * xij.z) / (sv * d);
                        ita += sv * (1.0f - cosvx) * W;
                    }
                }
            }
            if (wsum < 1e-8f || ncount == 0) continue;
            const float invw = 1.0f / wsum;
            mean.x *= invw; mean.y *= invw; mean.z *= invw;

            // Outward surface normal proxy = particle minus neighbour mean.
            const Vec3 nrm(xi.x - mean.x, xi.y - mean.y, xi.z - mean.z);
            const float nlen = std::sqrt(lengthSq(nrm));
            const float convexity = nlen / h;            // 0 interior … ~1 surface lip

            // Wave crest: convex AND moving along its own outward normal.
            float ik = 0.0f;
            const float vlen = std::sqrt(lengthSq(vi));
            if (vlen > 1e-5f && nlen > 1e-5f) {
                const float cvn = (vi.x * nrm.x + vi.y * nrm.y + vi.z * nrm.z) / (vlen * nlen);
                if (cvn > crest_cos) ik = convexity;
            }

            const float ek = 0.5f * vlen * vlen;

            const float Ita = clamp01(ita, params.ta_min, params.ta_max);
            const float Ik  = clamp01(ik,  params.wc_min, params.wc_max);
            const float Iek = clamp01(ek,  params.ke_min, params.ke_max);

            const float rate = k_ta * Ita + k_wc * Ik;   // spawns/sec at full Iek
            expected[pi] = rate * Iek * dt;
        }

        // ── 4. Serial stochastic emit (RNG can't run race-free in the parallel
        //       pass; this is cheap — only high-potential particles spawn). ──
        std::mt19937 rng(seed * 2654435761u + 12345u);
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        const float jit = params.spawn_jitter_voxels * voxel;
        std::size_t spawned = 0;
        for (std::size_t pi = 0; pi < fluid_count; ++pi) {
            float e = expected[pi];
            if (e <= 0.0f) continue;
            if (foam.size() >= params.max_foam) break;
            int n = static_cast<int>(e);
            if (u01(rng) < (e - static_cast<float>(n))) ++n;
            if (n <= 0) continue;
            const Vec3 xi = fluid.position[pi];
            const Vec3 vi = fluid.velocity[pi];
            for (int s = 0; s < n && foam.size() < params.max_foam; ++s) {
                // Jittered spawn inside the particle's cell.
                const Vec3 off((u01(rng) - 0.5f) * 2.0f * jit,
                               (u01(rng) - 0.5f) * 2.0f * jit,
                               (u01(rng) - 0.5f) * 2.0f * jit);
                const float life = params.lifetime * (0.75f + 0.5f * u01(rng));
                // New foam carries the source velocity (reclassified next step).
                foam.emit(xi + off, vi, life, FoamType::Foam);
                ++spawned;
            }
        }
        if (stats) stats->spawned = spawned;
    }
    if (stats) stats->gen_ms = std::chrono::duration<float, std::milli>(clock::now() - t_gen0).count();

    // ── 5. Stats ────────────────────────────────────────────────────────────
    if (stats) {
        stats->alive = foam.size();
        for (std::size_t i = 0; i < foam.size(); ++i) {
            switch (static_cast<FoamType>(foam.type[i])) {
                case FoamType::Spray:  ++stats->spray;  break;
                case FoamType::Bubble: ++stats->bubble; break;
                default:               ++stats->foam;   break;
            }
        }
    }
}

void splatFoamDensity(const FoamParticles& foam,
                      const FluidSim::FluidGrid& grid,
                      std::vector<float>& density_out,
                      float density_per_particle)
{
    splatFoamDensity(foam, grid.nx, grid.ny, grid.nz, grid.voxel_size, grid.origin,
                     density_out, density_per_particle);
}

void splatFoamDensity(const FoamParticles& foam,
                      int nx, int ny, int nz, float voxel, const Vec3& origin,
                      std::vector<float>& density_out,
                      float density_per_particle,
                      float bubble_weight,
                      float spray_weight)
{
    const std::size_t cells = (nx > 0 && ny > 0 && nz > 0)
        ? static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz)
        : 0;
    density_out.assign(cells, 0.0f);
    if (nx <= 0 || ny <= 0 || nz <= 0 || voxel <= 0.0f || foam.empty() || cells == 0) return;

    const float inv_h = 1.0f / voxel;
    auto idx = [nx, ny](int i, int j, int k) -> std::size_t {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    };

    const std::size_t n = foam.size();
    const bool has_type = (foam.type.size() == n);
    for (std::size_t p = 0; p < n; ++p) {
        const Vec3& x = foam.position[p];
        if (!std::isfinite(x.x) || !std::isfinite(x.y) || !std::isfinite(x.z)) continue;
        // Per-class deposit weight: surface FOAM full, BUBBLE/SPRAY scaled. Bubbles
        // ride the same channel but are depth-tinted by the water for free → distinct
        // subsurface froth without a second volume.
        float dpp = density_per_particle;
        if (has_type) {
            const uint8_t t = foam.type[p];
            if (t == static_cast<uint8_t>(FoamType::Bubble))      dpp *= bubble_weight;
            else if (t == static_cast<uint8_t>(FoamType::Spray))  dpp *= spray_weight;
        }
        if (dpp <= 0.0f) continue;  // class fully suppressed
        // Cell-centred coords (shift by -0.5 so trilinear weights span cell centres).
        const float gx = (x.x - origin.x) * inv_h - 0.5f;
        const float gy = (x.y - origin.y) * inv_h - 0.5f;
        const float gz = (x.z - origin.z) * inv_h - 0.5f;
        const int i0 = static_cast<int>(std::floor(gx));
        const int j0 = static_cast<int>(std::floor(gy));
        const int k0 = static_cast<int>(std::floor(gz));
        const float fx = gx - i0, fy = gy - j0, fz = gz - k0;
        for (int dk = 0; dk < 2; ++dk)
        for (int dj = 0; dj < 2; ++dj)
        for (int di = 0; di < 2; ++di) {
            const int i = i0 + di, j = j0 + dj, k = k0 + dk;
            if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
            const float w = (di ? fx : 1.0f - fx) * (dj ? fy : 1.0f - fy) * (dk ? fz : 1.0f - fz);
            density_out[idx(i, j, k)] += dpp * w;
        }
    }
}

} // namespace Fluid
} // namespace RayTrophiSim
