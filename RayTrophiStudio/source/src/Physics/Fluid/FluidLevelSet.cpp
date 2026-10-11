/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FluidLevelSet.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Zhu-Bridson narrow-band SDF reconstruction from APIC particle positions.
 * See FluidLevelSet.h for algorithm / surface definition.
 *
 * Implementation:
 *   1. CSR bin: particles -> containing cell. Cells with no particles get an
 *      empty range. Single pass, O(N).
 *   2. Output pass: for each cell c in parallel, iterate the stencil of
 *      neighbour cells whose particles can fall inside the kernel radius R.
 *      Accumulate kernel-weighted positions; compute phi = |x - x_bar| - r.
 *      Far cells -> +narrow_band sentinel.
 *
 * Race-free by construction: parallelisation is over output cells, each
 * reads disjoint inputs (the CSR is built sequentially before the kernel).
 * Mirrors the APIC P2G tile-bin pattern from MEMORY (no atomics, no
 * per-thread scratch grid).
 */

#include "Fluid/FluidLevelSet.h"
#include "Fluid/SurfaceWorkTiles.h"
#include "Fluid/SurfaceParticleBins.h"
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/SubstanceTag.h"
#include "PerfProfile.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace RayTrophiSim {
namespace Fluid {

namespace {

inline float lengthSq(const Vec3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// ── Anisotropic-kernel helpers (Yu & Turk 2013) ─────────────────────────────
// Symmetric 3x3 matrix (covariance / anisotropy G) stored as 6 unique entries.
struct Sym3 { float xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0; };

inline Vec3 symMul(const Sym3& A, const Vec3& v) {
    return Vec3(A.xx * v.x + A.xy * v.y + A.xz * v.z,
                A.xy * v.x + A.yy * v.y + A.yz * v.z,
                A.xz * v.x + A.yz * v.y + A.zz * v.z);
}

inline Sym3 isoSym(float s) { Sym3 G; G.xx = G.yy = G.zz = s; return G; }

// G = sum_k lk * e_k e_k^T  (e_k = eigenvector columns).
inline Sym3 symFromEigen(const Vec3 e[3], float l0, float l1, float l2) {
    const float L[3] = { l0, l1, l2 };
    Sym3 G;
    for (int k = 0; k < 3; ++k) {
        const Vec3& ek = e[k];
        const float lk = L[k];
        G.xx += lk * ek.x * ek.x; G.yy += lk * ek.y * ek.y; G.zz += lk * ek.z * ek.z;
        G.xy += lk * ek.x * ek.y; G.xz += lk * ek.x * ek.z; G.yz += lk * ek.y * ek.z;
    }
    return G;
}

inline Vec3 cross3(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x);
}

// Null-space vector of (A - lam*I) for an eigenvalue lam: the row-pair cross
// product with the largest magnitude (most numerically reliable). Returned
// normalised; falls back to +x for a fully degenerate matrix.
inline Vec3 eigvecFor(const Sym3& A, float lam) {
    const Vec3 r0(A.xx - lam, A.xy, A.xz);
    const Vec3 r1(A.xy, A.yy - lam, A.yz);
    const Vec3 r2(A.xz, A.yz, A.zz - lam);
    const Vec3 c01 = cross3(r0, r1);
    const Vec3 c02 = cross3(r0, r2);
    const Vec3 c12 = cross3(r1, r2);
    Vec3 best = c01; float bn = lengthSq(c01);
    const float n02 = lengthSq(c02); if (n02 > bn) { best = c02; bn = n02; }
    const float n12 = lengthSq(c12); if (n12 > bn) { best = c12; bn = n12; }
    if (bn < 1e-20f) return Vec3(1, 0, 0);
    const float inv = 1.0f / std::sqrt(bn);
    return Vec3(best.x * inv, best.y * inv, best.z * inv);
}

// Closed-form eigendecomposition of a symmetric 3x3 (Smith/Cardano for the
// eigenvalues, null-space cross products + Gram-Schmidt for the orthonormal
// frame). Replaces the 24-sweep cyclic Jacobi: same shaping accuracy for the
// kernel covariance at a fraction of the per-particle cost. eval ascending,
// evec[] paired columns (orthonormal, right-handed).
inline void analyticEigen3(const Sym3& A, float eval[3], Vec3 evec[3]) {
    const double p1 = static_cast<double>(A.xy) * A.xy +
                      static_cast<double>(A.xz) * A.xz +
                      static_cast<double>(A.yz) * A.yz;
    const double tr = static_cast<double>(A.xx) + A.yy + A.zz;
    if (p1 < 1e-20) {                 // already diagonal
        eval[0] = A.xx; eval[1] = A.yy; eval[2] = A.zz;
        evec[0] = Vec3(1, 0, 0); evec[1] = Vec3(0, 1, 0); evec[2] = Vec3(0, 0, 1);
        return;
    }
    const double q  = tr / 3.0;
    const double p2 = (A.xx - q) * (A.xx - q) + (A.yy - q) * (A.yy - q) +
                      (A.zz - q) * (A.zz - q) + 2.0 * p1;
    const double p  = std::sqrt(p2 / 6.0);
    // det(B)/2 with B = (A - qI)/p, expanded directly.
    const double bxx = (A.xx - q) / p, byy = (A.yy - q) / p, bzz = (A.zz - q) / p;
    const double bxy = A.xy / p, bxz = A.xz / p, byz = A.yz / p;
    double r = (bxx * (byy * bzz - byz * byz) -
                bxy * (bxy * bzz - byz * bxz) +
                bxz * (bxy * byz - byy * bxz)) * 0.5;
    r = std::clamp(r, -1.0, 1.0);
    constexpr double kTwoPiThird = 2.0943951023931953; // 2*pi/3
    const double phi = std::acos(r) / 3.0;
    const double e2 = q + 2.0 * p * std::cos(phi);                // largest
    const double e0 = q + 2.0 * p * std::cos(phi + kTwoPiThird);  // smallest
    const double e1 = tr - e0 - e2;                               // middle
    eval[0] = static_cast<float>(e0);
    eval[1] = static_cast<float>(e1);
    eval[2] = static_cast<float>(e2);

    // Vectors for the two extreme eigenvalues are best separated; derive the
    // middle by cross product so the frame stays orthonormal even when two
    // eigenvalues are close.
    Vec3 v0 = eigvecFor(A, eval[0]);
    Vec3 v2 = eigvecFor(A, eval[2]);
    const float d = v0.x * v2.x + v0.y * v2.y + v0.z * v2.z;
    v2 = Vec3(v2.x - d * v0.x, v2.y - d * v0.y, v2.z - d * v0.z);
    float n2 = std::sqrt(lengthSq(v2));
    if (n2 < 1e-8f) {                 // degenerate: pick any vector perp to v0
        const Vec3 t = (std::fabs(v0.x) < 0.9f) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
        const float dd = v0.x * t.x + v0.y * t.y + v0.z * t.z;
        v2 = Vec3(t.x - dd * v0.x, t.y - dd * v0.y, t.z - dd * v0.z);
        n2 = std::sqrt(lengthSq(v2));
    }
    v2 = Vec3(v2.x / n2, v2.y / n2, v2.z / n2);
    const Vec3 v1 = cross3(v2, v0);
    evec[0] = v0; evec[1] = v1; evec[2] = v2;
}

} // namespace

bool buildLevelSet(const FluidParticles& particles,
                   const FluidSim::FluidGrid& grid,
                   const LevelSetParams& params,
                   std::vector<float>& sdf_out,
                   LevelSetStats* stats,
                   const FluidViewSelection* selection)
{
    using clock = std::chrono::steady_clock;
    RTPERF_FRAME_SCOPE("sim.render.surface.level_set");
    const auto t_start = clock::now();

    // The surface grid may be refined relative to the simulation grid: the SDF
    // is the rendered surface and does not have to share the sim voxel size.
    // Refining gives sub-voxel detail without paying the N^3 simulation cost.
    const int   m         = std::clamp(params.surface_resolution_multiplier, 1, 4);
    const float sim_voxel = grid.voxel_size;
    const int   nx    = grid.nx * m;
    const int   ny    = grid.ny * m;
    const int   nz    = grid.nz * m;
    const float voxel = (m > 1) ? (sim_voxel / static_cast<float>(m)) : sim_voxel;
    const Vec3  origin = grid.origin;            // unchanged — same physical extent
    const std::size_t cell_count = static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny) *
                                   static_cast<std::size_t>(nz);
    const std::size_t particle_count = particles.size();

    // Flat index for the refined grid (grid.cellIndex assumes the sim dims).
    auto cellIndex = [nx, ny](int i, int j, int k) -> std::size_t {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
               static_cast<std::size_t>(ny);
    };

    // Radii stay PHYSICAL (expressed in SIM voxels) so the surface SHAPE is
    // invariant to the multiplier — only the sampling fineness changes.
    const float kernel_R = std::max(1e-4f, params.kernel_radius_voxels * sim_voxel);
    const float kernel_R_sq = kernel_R * kernel_R;
    const float particle_r = std::max(0.0f, params.particle_radius_voxels * sim_voxel);
    // This is a zero-level-set displacement, not another reconstruction
    // radius. Keeping it separate from particle_r makes "fullness" independent
    // of particle mass and gives the future fragment-aware builder one explicit
    // dilation to limit at an opened crack. Bound it so it cannot outrun the
    // finite kernel support or turn a cheap look control into a hidden topology
    // change across several cells.
    const float surface_offset =
        std::clamp(params.surface_offset_voxels, -0.75f, 1.25f) * sim_voxel;
    const float narrow_band = std::max(kernel_R, params.narrow_band_voxels * sim_voxel);
    const float far_value = narrow_band;

    if (stats) {
        *stats = {};
        stats->work_cells = 0;
        stats->bin_cells = 0;
        stats->active_cells = 0;
        stats->surface_cells = 0;
        stats->particle_count = particle_count;
        stats->grid_cell_count = cell_count;
        stats->build_ms = 0.0f;
        stats->eff_nx = nx;
        stats->eff_ny = ny;
        stats->eff_nz = nz;
        stats->eff_voxel = voxel;
    }

    sdf_out.assign(cell_count, far_value);
    if (cell_count == 0 || particle_count == 0 || voxel <= 0.0f) {
        if (stats) {
            stats->build_ms = std::chrono::duration<float, std::milli>(
                                  clock::now() - t_start).count();
        }
        return false;
    }

    auto stage_start = clock::now();
    const auto elapsed = [&]() {
        return std::chrono::duration<float, std::milli>(clock::now() - stage_start).count();
    };

    // ── 1. CSR particle->cell bin --------------------------------------------
    // particle_cell[p] = flat cell index, or kInvalid if the particle is out
    // of grid bounds (silently dropped from the SDF).
    constexpr std::size_t kInvalid = static_cast<std::size_t>(-1);
    std::vector<std::size_t> particle_cell(particle_count, kInvalid);
    // Particle bins are independent of display refinement. Physical support
    // stays fixed when quality grows, so neighbour search need not grow m^3.
    const float bin_voxel = std::max(kernel_R, sim_voxel);
    const float inv_bin = 1.0f / bin_voxel;
    const int bnx = std::max(1, static_cast<int>(std::ceil(grid.nx * sim_voxel * inv_bin)));
    const int bny = std::max(1, static_cast<int>(std::ceil(grid.ny * sim_voxel * inv_bin)));
    const int bnz = std::max(1, static_cast<int>(std::ceil(grid.nz * sim_voxel * inv_bin)));
    const std::size_t bin_count = static_cast<std::size_t>(bnx) * bny * bnz;
    const auto binIndex = [bnx, bny](int i, int j, int k) {
        return static_cast<std::size_t>(i) + static_cast<std::size_t>(bnx) *
            (j + static_cast<std::size_t>(bny) * k);
    };
    std::vector<int> cell_count_per(bin_count, 0);

    const float inv_h = 1.0f / voxel;
    for (std::size_t p = 0; p < particle_count; ++p) {
        if (selection && !selection->keeps(particles, p)) continue;
        // Depleted fuel particles remain in the stable APIC pool until the
        // lifecycle compaction pass. Do not let an already evaporated slot
        // keep the rendered SurfaceSDF artificially thick.
        if (p < particles.mass_fraction.size() &&
            !(particles.mass_fraction[p] > 0.02f)) continue;
        const Vec3& wp = particles.position[p];
        if (!std::isfinite(wp.x) || !std::isfinite(wp.y) || !std::isfinite(wp.z)) continue;
        const Vec3 local = (wp - origin) * inv_h;
        if (local.x < 0.0f || local.x >= nx || local.y < 0.0f || local.y >= ny ||
            local.z < 0.0f || local.z >= nz) {
            continue;
        }
        const int i = static_cast<int>(std::floor(local.x));
        const int j = static_cast<int>(std::floor(local.y));
        const int k = static_cast<int>(std::floor(local.z));
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
        const Vec3 bin_local = (wp - origin) * inv_bin;
        const int bi = std::clamp(static_cast<int>(std::floor(bin_local.x)), 0, bnx - 1);
        const int bj = std::clamp(static_cast<int>(std::floor(bin_local.y)), 0, bny - 1);
        const int bk = std::clamp(static_cast<int>(std::floor(bin_local.z)), 0, bnz - 1);
        const std::size_t ci = binIndex(bi, bj, bk);
        particle_cell[p] = ci;
        ++cell_count_per[ci];
    }

    // Prefix-sum -> CSR offsets.
    std::vector<std::size_t> cell_offset(bin_count + 1, 0);
    for (std::size_t c = 0; c < bin_count; ++c) {
        cell_offset[c + 1] = cell_offset[c] + static_cast<std::size_t>(cell_count_per[c]);
    }
    std::vector<std::size_t> cell_csr(cell_offset.back());
    std::vector<std::size_t> cursor(bin_count, 0);
    for (std::size_t p = 0; p < particle_count; ++p) {
        const std::size_t ci = particle_cell[p];
        if (ci == kInvalid) continue;
        const std::size_t pos = cell_offset[ci] + cursor[ci]++;
        cell_csr[pos] = p;
    }

    int thread_cap = params.threads;
#ifdef _OPENMP
    if (thread_cap <= 0) thread_cap = omp_get_max_threads();
    thread_cap = std::max(1, thread_cap);
#else
    (void)thread_cap;
#endif

    if (stats) {
        stats->bin_ms = elapsed();
    }
    stage_start = clock::now();

    // ── 1b. Per-particle anisotropy (Yu & Turk 2013) -------------------------
    // For each particle compute a smoothed position x~ and an anisotropy matrix
    // G (symmetric) from the weighted covariance of its neighbours, so the
    // output pass can splat ellipsoidal kernels (|G*(x - x~)| < 1) instead of
    // isotropic spheres. Function-static scratch reused across calls — NEVER
    // thread_local (OMP workers read these main-populated buffers) and NEVER a
    // per-call vector (heap-lock stall). Populated in a parallel pre-pass; the
    // implicit barrier at its end orders the writes before the output reads.
    static std::vector<Vec3> s_xtilde;
    static std::vector<Sym3> s_aniso;
    const bool use_aniso = params.anisotropy_enabled;
    const bool prepare_centres = use_aniso || params.position_smoothing > 0.0f;
    if (prepare_centres) {
        if (s_xtilde.size() < particle_count) s_xtilde.resize(particle_count);
        if (s_aniso.size()  < particle_count) s_aniso.resize(particle_count);

        const float rs     = std::max(1e-4f, params.anisotropy_radius_voxels * sim_voxel);
        const float rs_sq  = rs * rs;
        const int   reach_a = std::max(1, static_cast<int>(
            std::ceil(rs * inv_bin)));
        const float kr     = std::max(1.0f, params.anisotropy_max_stretch);
        const int   nmin   = std::max(1, params.anisotropy_neighbor_min);
        const float lambda = std::clamp(params.position_smoothing, 0.0f, 1.0f);
        const float inv_kR = 1.0f / kernel_R;
        const Sym3  isoG   = isoSym(inv_kR);

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 256) num_threads(thread_cap)
#endif
        for (int64_t pp = 0; pp < static_cast<int64_t>(particle_count); ++pp) {
            const std::size_t pidx = static_cast<std::size_t>(pp);
            const Vec3 xp = particles.position[pidx];
            if (particle_cell[pidx] == kInvalid) { s_xtilde[pidx] = xp; s_aniso[pidx] = isoG; continue; }

            const Vec3 lp = (xp - origin) * inv_bin;
            const int ci = static_cast<int>(std::floor(lp.x));
            const int cj = static_cast<int>(std::floor(lp.y));
            const int ck = static_cast<int>(std::floor(lp.z));
            const int ai0 = std::max(0, ci - reach_a), ai1 = std::min(bnx - 1, ci + reach_a);
            const int aj0 = std::max(0, cj - reach_a), aj1 = std::min(bny - 1, cj + reach_a);
            const int ak0 = std::max(0, ck - reach_a), ak1 = std::min(bnz - 1, ck + reach_a);

            // Single-pass weighted mean + covariance. Accumulate the moments
            // relative to xp (rel = xj - xp, magnitudes < rs) so the float
            // products stay small and precise, then C = E[rr] - mean_rel(x)mean_rel
            // — one stencil traversal instead of two.
            float wsum = 0.0f; int ncount = 0;
            Vec3 mom1(0, 0, 0);                                // sum w*rel
            Sym3 mom2;                                         // sum w*(rel (x) rel)
            for (int kk = ak0; kk <= ak1; ++kk) {
                for (int jj = aj0; jj <= aj1; ++jj) {
                    for (int ii = ai0; ii <= ai1; ++ii) {
                        const std::size_t nci = binIndex(ii, jj, kk);
                        for (std::size_t a = cell_offset[nci]; a < cell_offset[nci + 1]; ++a) {
                            const Vec3& xj = particles.position[cell_csr[a]];
                            const float rx = xj.x - xp.x, ry = xj.y - xp.y, rz = xj.z - xp.z;
                            const float d2 = rx * rx + ry * ry + rz * rz;
                            if (d2 >= rs_sq) continue;
                            const float tt = 1.0f - d2 / rs_sq;
                            const float w = tt * tt * tt;
                            wsum += w; ++ncount;
                            mom1.x += w * rx; mom1.y += w * ry; mom1.z += w * rz;
                            if (use_aniso) {
                                mom2.xx += w * rx * rx;
                                mom2.yy += w * ry * ry;
                                mom2.zz += w * rz * rz;
                                mom2.xy += w * rx * ry;
                                mom2.xz += w * rx * rz;
                                mom2.yz += w * ry * rz;
                            }
                        }
            
                    }
                }
            }
            if (wsum < 1e-12f) { s_xtilde[pidx] = xp; s_aniso[pidx] = isoG; continue; }
            if (ncount < nmin) {
                s_xtilde[pidx] = xp;
                s_aniso[pidx] = isoG;
                continue;
            }
            const float invw = 1.0f / wsum;
            const Vec3 mr(mom1.x * invw, mom1.y * invw, mom1.z * invw); // mean relative to xp
            s_xtilde[pidx] = Vec3(xp.x + mr.x * lambda,
                                  xp.y + mr.y * lambda,
                                  xp.z + mr.z * lambda);
            if (!use_aniso) {
                s_aniso[pidx] = isoG;
                continue;
            }

            Sym3 C;
            C.xx = mom2.xx * invw - mr.x * mr.x; C.yy = mom2.yy * invw - mr.y * mr.y; C.zz = mom2.zz * invw - mr.z * mr.z;
            C.xy = mom2.xy * invw - mr.x * mr.y; C.xz = mom2.xz * invw - mr.x * mr.z; C.yz = mom2.yz * invw - mr.y * mr.z;

            float ev[3]; Vec3 R[3];
            analyticEigen3(C, ev, R);
            const float emax = std::max(ev[0], std::max(ev[1], ev[2]));
            if (emax < 1e-12f) { s_aniso[pidx] = isoG; continue; }
            // Convert variance to axis length, clamp aspect ratio, then normalise so
            // det == 1 (every particle's kernel covers ~the same volume — no blob
            // dominates). Low-variance axes stay thin; high-variance axes follow
            // the local sheet or stream direction.
            const float floorv = emax / (kr * kr);
            float s0 = std::sqrt(std::max(ev[0], floorv));
            float s1 = std::sqrt(std::max(ev[1], floorv));
            float s2 = std::sqrt(std::max(ev[2], floorv));
            const float vol = std::cbrt(std::max(1e-20f, s0 * s1 * s2));
            s0 /= vol; s1 /= vol; s2 /= vol;
            s_aniso[pidx] = symFromEigen(R, inv_kR / s0, inv_kR / s1, inv_kR / s2);
        }
    }

    if (stats) {
        stats->anisotropy_ms = elapsed();
    }
    stage_start = clock::now();

    // Sparse work tiles cover every ellipsoid and smoothing dependency.
    // Rebin temporary centres after filtering. Neighbour queries then need
    // only physical support, rather than worst-case support + centre shift.
    const float kr_stretch = std::max(1.0f, params.anisotropy_max_stretch);
    const float stretch_reach = use_aniso ? std::cbrt(kr_stretch * kr_stretch) : 1.0f;
    const float support = kernel_R * stretch_reach;
    const auto& centres = prepare_centres ? s_xtilde : particles.position;
    SurfaceParticleBins surface_bins(origin, grid.nx, grid.ny, grid.nz,
                                     sim_voxel, support);
    surface_bins.build(centres, particle_cell);
    const int reach = std::max(1, static_cast<int>(
        std::ceil(support * surface_bins.inverse_width)));
    const int smoothing = std::max(0, params.smoothing_iterations);
    SurfaceWorkTiles work_tiles(nx, ny, nz, origin, voxel);
    for (std::size_t p = 0; p < particle_count; ++p) {
        if (particle_cell[p] != kInvalid) {
            work_tiles.addSupport(centres[p],
                                  support + smoothing * sim_voxel);
        }
    }
    work_tiles.finish();
    const int64_t active_n = static_cast<int64_t>(work_tiles.workSize());
    // Transient compact reconstruction evidence; never stored in simulation cache.
    std::vector<float> support_confidence(
        smoothing > 0 ? work_tiles.workSize() : 0, 0.0f);
    if (stats) {
        stats->work_cells = work_tiles.workSize();
        stats->bin_cells = surface_bins.offsets.size() - 1;
    }

    if (stats) {
        stats->schedule_ms = elapsed();
    }
    stage_start = clock::now();

    std::size_t active_cells = 0;
    std::size_t surface_cells = 0;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 256) num_threads(thread_cap) \
        reduction(+:active_cells, surface_cells)
#endif
    for (int64_t at = 0; at < active_n; ++at) {
        int i, j, k;
        if (!work_tiles.cellAt(static_cast<std::size_t>(at), i, j, k)) {
            continue;
        }
        const Vec3 p_c = origin + Vec3(
            (static_cast<float>(i) + 0.5f) * voxel,
            (static_cast<float>(j) + 0.5f) * voxel,
            (static_cast<float>(k) + 0.5f) * voxel);
        int bi, bj, bk;
        surface_bins.coordinate(p_c, bi, bj, bk);
        const int i0 = std::max(0, bi - reach), i1 = std::min(surface_bins.nx - 1, bi + reach);
        const int j0 = std::max(0, bj - reach), j1 = std::min(surface_bins.ny - 1, bj + reach);
        const int k0 = std::max(0, bk - reach), k1 = std::min(surface_bins.nz - 1, bk + reach);

        float acc_w = 0.0f;
        float acc_mass = 0.0f;
        float acc_w_squared = 0.0f;
        Sym3 acc_metric;
        Vec3  acc_p(0.0f, 0.0f, 0.0f);

        for (int kk = k0; kk <= k1; ++kk) {
            for (int jj = j0; jj <= j1; ++jj) {
                for (int ii = i0; ii <= i1; ++ii) {
                    const std::size_t nci = surface_bins.index(ii, jj, kk);
                    const std::size_t beg = surface_bins.offsets[nci];
                    const std::size_t end = surface_bins.offsets[nci + 1];
                    for (std::size_t a = beg; a < end; ++a) {
                        const std::size_t pa = surface_bins.particles[a];
                        if (use_aniso) {
                            // Ellipsoidal kernel: support is |G*(x - x~)| < 1.
                            const Vec3& xt = s_xtilde[pa];
                            const Vec3 dd(p_c.x - xt.x, p_c.y - xt.y, p_c.z - xt.z);
                            const Vec3 q = symMul(s_aniso[pa], dd);
                            const float q2 = lengthSq(q);
                            if (q2 >= 1.0f) continue;
                            const float t = 1.0f - q2;
                            const float w = t * t * t;
                            const float mass = pa < particles.mass_fraction.size()
                                ? std::clamp(particles.mass_fraction[pa], 0.0f, 1.0f)
                                : 1.0f;
                            acc_metric.xx += w * s_aniso[pa].xx;
                            acc_metric.yy += w * s_aniso[pa].yy;
                            acc_metric.zz += w * s_aniso[pa].zz;
                            acc_metric.xy += w * s_aniso[pa].xy;
                            acc_metric.xz += w * s_aniso[pa].xz;
                            acc_metric.yz += w * s_aniso[pa].yz;
                            acc_w += w;
                            acc_w_squared += w * w;
                            acc_mass += w * mass;
                            acc_p.x += w * xt.x;
                            acc_p.y += w * xt.y;
                            acc_p.z += w * xt.z;
                        } else {
                            const Vec3& pp = centres[pa];
                            const Vec3 d = p_c - pp;
                            const float d2 = lengthSq(d);
                            if (d2 >= kernel_R_sq) continue;
                            // Wendland-like cubic falloff: smooth, finite
                            // support, derivative is well-behaved.
                            const float t = 1.0f - d2 / kernel_R_sq;
                            const float w = t * t * t;
                            const float mass = pa < particles.mass_fraction.size()
                                ? std::clamp(particles.mass_fraction[pa], 0.0f, 1.0f)
                                : 1.0f;
                            acc_w += w;
                            acc_w_squared += w * w;
                            acc_mass += w * mass;
                            acc_p.x += w * pp.x;
                            acc_p.y += w * pp.y;
                            acc_p.z += w * pp.z;
                        }
                    }
                }
            }
        }

        const std::size_t out_ci = cellIndex(i, j, k);
        if (acc_w > 1e-12f) {
            const float inv_w = 1.0f / acc_w;
            if (smoothing > 0) {
                const float effective_neighbours = acc_w * acc_w /
                    std::max(acc_w_squared, 1.0e-12f);
                const float t = std::clamp((effective_neighbours - 4.0f) / 12.0f,
                                           0.0f, 1.0f);
                support_confidence[static_cast<std::size_t>(at)] = t * t * (3.0f - 2.0f * t);
            }
            const Vec3 x_bar(acc_p.x * inv_w, acc_p.y * inv_w, acc_p.z * inv_w);
            const float dlen = std::sqrt(lengthSq(p_c - x_bar));
            const float mean_mass = std::clamp(acc_mass / std::max(acc_w, 1.0e-12f), 0.0f, 1.0f);
            const float mass_radius = particle_r * std::cbrt(std::max(mean_mass, 0.02f));
            float phi = dlen - mass_radius - surface_offset;
            if (use_aniso) {
                // Apply the same ellipsoid metric to the reconstructed surface,
                // not just its neighbourhood weights. Convert the implicit
                // residual to world units using its local gradient magnitude.
                const float scale = kernel_R * inv_w;
                acc_metric.xx *= scale;
                acc_metric.yy *= scale;
                acc_metric.zz *= scale;
                acc_metric.xy *= scale;
                acc_metric.xz *= scale;
                acc_metric.yz *= scale;
                const Vec3 shaped = symMul(acc_metric, p_c - x_bar);
                const float shaped_length = std::sqrt(lengthSq(shaped));
                float gradient_length = 1.0f;
                if (shaped_length > 1.0e-8f) {
                    const Vec3 gradient = symMul(acc_metric, shaped / shaped_length);
                    gradient_length = std::max(1.0e-6f, std::sqrt(lengthSq(gradient)));
                }
                phi = (shaped_length - mass_radius - surface_offset) / gradient_length;
            }
            // Clamp to narrow band so far interior cells still report
            // a finite distance the iso-walker can step through.
            const float phi_clamped = std::min(narrow_band, std::max(-narrow_band, phi));
            sdf_out[out_ci] = phi_clamped;
            ++active_cells;
            if (std::fabs(phi_clamped) < voxel) {
                ++surface_cells;
            }
        } else {
            sdf_out[out_ci] = far_value;
        }
    }


    if (stats) {
        stats->gather_ms = elapsed();
    }
    stage_start = clock::now();

    // ── 3. Optional fast Laplacian smoothing sweeps --------------------------
    if (params.smoothing_iterations > 0) {
        smoothSurfaceTiles(sdf_out, work_tiles, nx, ny, nz, far_value, smoothing,
                           thread_cap, &support_confidence,
                           std::max(1, static_cast<int>(std::lround(sim_voxel / voxel))),
                           0.2f * sim_voxel);

        // Recalculate active and surface cell counts on the smoothed field
        active_cells = 0;
        surface_cells = 0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:active_cells, surface_cells) num_threads(thread_cap)
#endif
        for (int64_t lane = 0; lane < active_n; ++lane) {
            int i, j, k;
            if (!work_tiles.cellAt(static_cast<std::size_t>(lane), i, j, k)) {
                continue;
            }
            const float val = sdf_out[cellIndex(i, j, k)];
            if (val < far_value - 1e-5f) {
                ++active_cells;
                if (std::abs(val) < voxel) {
                    ++surface_cells;
                }
            }
        }
    }

    if (stats) {
        stats->smooth_ms = elapsed();
        stats->active_cells = active_cells;
        stats->surface_cells = surface_cells;
        stats->build_ms = std::chrono::duration<float, std::milli>(
                              clock::now() - t_start).count();
    }
    return active_cells > 0;
}

bool buildCompositionGrid(const FluidParticles& particles,
                          const FluidSim::FluidGrid& grid,
                          const LevelSetParams& params,
                          const SubstanceMaterialEntry* substance_materials,
                          std::size_t substance_material_count,
                          int fallback_material,
                          std::vector<float>& composition_out,
                          const FluidViewSelection* selection)
{
    RTPERF_FRAME_SCOPE("sim.render.surface.composition");
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float voxel = grid.voxel_size;
    const std::size_t cell_count = static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny) *
                                   static_cast<std::size_t>(nz);
    const std::size_t particle_count = particles.size();

    if (cell_count == 0 || particle_count == 0 || voxel <= 0.0f ||
        substance_material_count == 0 || substance_materials == nullptr ||
        particles.substance_tag.size() < particle_count) {
        composition_out.clear();
        return false;
    }

    // ── Distinct material set ────────────────────────────────────────────────
    // The per-cell accumulator is indexed by POSITION IN THIS LIST, not by
    // material id, so its width is the number of materials actually in play
    // rather than the size of the scene's material table.
    //
    // ★ Slot 0 is always the fallback, so a cell that gathered nothing but
    // untagged liquid resolves to the domain material without a special case.
    int slot_material[kMaxFluidSubstanceMaterials + 1];
    std::size_t slot_count = 1;
    slot_material[0] = fallback_material;
    for (std::size_t i = 0; i < substance_material_count &&
                            slot_count <= kMaxFluidSubstanceMaterials; ++i) {
        bool seen = false;
        for (std::size_t s = 0; s < slot_count; ++s)
            if (slot_material[s] == substance_materials[i].material_id) { seen = true; break; }
        if (!seen) slot_material[slot_count++] = substance_materials[i].material_id;
    }
    // ★★★ NOTHING TO DESCRIBE. Every substance resolves to the material the
    // domain would have used anyway, so a composition field would be uniform:
    // megabytes uploaded per rebuild to drive a blend that cannot change a
    // pixel. Refusing here is what keeps single-material domains — which is
    // most of them — exactly as cheap as they were.
    if (slot_count <= 1) {
        composition_out.clear();
        return false;
    }

    // Tag -> slot. Linear scans over at most kMax entries; both loops are tiny
    // and a map would cost more than it saves at this size.
    auto slotForTag = [&](uint32_t tag) -> std::size_t {
        if (tag == kSubstanceUntagged) return 0;
        for (std::size_t i = 0; i < substance_material_count; ++i) {
            if (substance_materials[i].tag != tag) continue;
            for (std::size_t s = 0; s < slot_count; ++s)
                if (slot_material[s] == substance_materials[i].material_id) return s;
            return 0;
        }
        // ★ A tag with no binding is NOT an error and must not be dropped: it is
        // liquid poured by a source that has since been renamed, or handed over
        // by a mass transfer. It takes the domain material, exactly like
        // untagged liquid, rather than vanishing from the mixture and letting
        // the other substances renormalise to more than they really are.
        return 0;
    };

    composition_out.assign(cell_count * 3u, 0.0f);

    auto cellIndex = [nx, ny](int i, int j, int k) -> std::size_t {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
               static_cast<std::size_t>(ny);
    };

    // Same kernel as the level set and the coordinate gather, for the same
    // reason: the composition must be supported exactly where the surface is.
    const float kernel_R    = std::max(1e-4f, params.kernel_radius_voxels * voxel);
    const float kernel_R_sq = kernel_R * kernel_R;
    const int   reach       = std::max(1, static_cast<int>(
        std::ceil(params.kernel_radius_voxels)));

    int thread_cap = params.threads;
#ifdef _OPENMP
    if (thread_cap <= 0) thread_cap = omp_get_max_threads();
    thread_cap = std::max(1, thread_cap);
#else
    (void)thread_cap;
#endif

    constexpr std::size_t kInvalid = static_cast<std::size_t>(-1);
    static std::vector<std::size_t> s_particle_cell;
    static std::vector<int>         s_count_per;
    static std::vector<std::size_t> s_offset, s_csr, s_cursor;
    s_particle_cell.assign(particle_count, kInvalid);
    s_count_per.assign(cell_count, 0);

    const float inv_h = 1.0f / voxel;
    for (std::size_t p = 0; p < particle_count; ++p) {
        if (p < particles.mass_fraction.size() &&
            particles.mass_fraction[p] <= 0.02f) continue;
        // ★ Same exclusion the level set applied. A substance routed to splat
        // has no isosurface, so it must not contribute to a field that describes
        // one — the three gathers have to agree about which particles the
        // surface is made of, or the surface takes on the look of liquid that
        // is not there. That reads as plausible wetness, not as a bug.
        if (selection && !selection->keeps(particles, p)) continue;
        const Vec3& wp = particles.position[p];
        if (!std::isfinite(wp.x) || !std::isfinite(wp.y) || !std::isfinite(wp.z)) continue;
        const Vec3 local = (wp - grid.origin) * inv_h;
        const int i = static_cast<int>(std::floor(local.x));
        const int j = static_cast<int>(std::floor(local.y));
        const int k = static_cast<int>(std::floor(local.z));
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
        const std::size_t ci = cellIndex(i, j, k);
        s_particle_cell[p] = ci;
        ++s_count_per[ci];
    }

    s_offset.assign(cell_count + 1, 0);
    for (std::size_t c = 0; c < cell_count; ++c)
        s_offset[c + 1] = s_offset[c] + static_cast<std::size_t>(s_count_per[c]);
    if (s_offset.back() == 0) {
        composition_out.clear();
        return false;
    }
    s_csr.assign(s_offset.back(), 0);
    s_cursor.assign(cell_count, 0);
    for (std::size_t p = 0; p < particle_count; ++p) {
        const std::size_t ci = s_particle_cell[p];
        if (ci == kInvalid) continue;
        s_csr[s_offset[ci] + s_cursor[ci]++] = p;
    }

    // Per-slot miscibility, resolved once alongside the slot table. A slot can
    // be reached by several tags (two substances sharing a material); the
    // STRICTEST wins, because a slot that any immiscible substance feeds must
    // not be softened by a miscible sibling that happens to shade the same.
    float slot_miscibility[kMaxFluidSubstanceMaterials + 1];
    for (std::size_t s = 0; s < slot_count; ++s) slot_miscibility[s] = 1.0f;
    for (std::size_t i = 0; i < substance_material_count; ++i) {
        for (std::size_t s = 0; s < slot_count; ++s) {
            if (slot_material[s] != substance_materials[i].material_id) continue;
            slot_miscibility[s] = std::min(slot_miscibility[s],
                std::max(0.0f, std::min(1.0f, substance_materials[i].miscibility)));
            break;
        }
    }

    // Per-particle slot, resolved once. The gather below touches each particle
    // up to (2*reach+1)^3 times, and re-resolving the tag inside that loop would
    // multiply a linear scan by the stencil volume.
    static std::vector<uint8_t> s_particle_slot;
    s_particle_slot.assign(particle_count, 0u);
    for (std::size_t p = 0; p < particle_count; ++p) {
        s_particle_slot[p] = static_cast<uint8_t>(
            slotForTag(particles.substance_tag[p]));
    }

#ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(dynamic, 8) num_threads(thread_cap)
#endif
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const Vec3 p_c = grid.origin + Vec3(
                    (static_cast<float>(i) + 0.5f) * voxel,
                    (static_cast<float>(j) + 0.5f) * voxel,
                    (static_cast<float>(k) + 0.5f) * voxel);

                const int i0 = std::max(0, i - reach), i1 = std::min(nx - 1, i + reach);
                const int j0 = std::max(0, j - reach), j1 = std::min(ny - 1, j + reach);
                const int k0 = std::max(0, k - reach), k1 = std::min(nz - 1, k + reach);

                float acc[kMaxFluidSubstanceMaterials + 1] = {0.0f};
                float acc_total = 0.0f;
                for (int kk = k0; kk <= k1; ++kk)
                for (int jj = j0; jj <= j1; ++jj)
                for (int ii = i0; ii <= i1; ++ii) {
                    const std::size_t nci = cellIndex(ii, jj, kk);
                    for (std::size_t a = s_offset[nci]; a < s_offset[nci + 1]; ++a) {
                        const std::size_t pa = s_csr[a];
                        const Vec3 d = p_c - particles.position[pa];
                        const float d2 = lengthSq(d);
                        if (d2 >= kernel_R_sq) continue;
                        const float t = 1.0f - d2 / kernel_R_sq;
                        const float w = t * t * t;
                        acc[s_particle_slot[pa]] += w;
                        acc_total += w;
                    }
                }

                const std::size_t o = cellIndex(i, j, k) * 3u;
                if (acc_total <= 1e-12f) {
                    // Unsupported cell: the fallback, at full weight. NOT left
                    // at zero-with-zero-weight, which the consumer would read as
                    // "material 0 mixed with material 0" — true by accident and
                    // wrong the moment slot 0 stops being the fallback.
                    composition_out[o + 0] = static_cast<float>(fallback_material + 1);
                    composition_out[o + 1] = static_cast<float>(fallback_material + 1);
                    composition_out[o + 2] = 0.0f;
                    continue;
                }

                // Top two by weight.
                std::size_t best = 0, second = 0;
                float best_w = -1.0f, second_w = -1.0f;
                for (std::size_t s = 0; s < slot_count; ++s) {
                    if (acc[s] > best_w) {
                        second = best; second_w = best_w;
                        best = s;       best_w = acc[s];
                    } else if (acc[s] > second_w) {
                        second = s; second_w = acc[s];
                    }
                }
                if (second_w < 0.0f) { second = best; second_w = 0.0f; }

                // ★ Renormalised over the KEPT PAIR, not over everything
                // gathered. Dividing by acc_total instead would let a discarded
                // third substance quietly reduce both weights, so the blend
                // would drift toward the dominant material by an amount that
                // depends on what else happened to be nearby.
                const float pair_total = std::max(best_w + second_w, 1e-12f);
                float w = std::min(1.0f, std::max(0.0f, second_w / pair_total));

                // ── Miscibility: sharpen the transition, do not collapse it ──
                // Gain about the 0.5 crossing. m = 1 leaves the gathered
                // gradient untouched; smaller m compresses the whole ramp into
                // a proportionally thinner band around the front.
                //
                // ★ The pair takes the MINIMUM: one immiscible member is enough
                // to make the boundary sharp, and requiring both to agree would
                // mean pouring water into oil behaved differently from pouring
                // oil into water.
                //
                // ★★ Clamped at 64 rather than allowed to reach infinity. An
                // exact step would put the entire transition inside one cell,
                // where the consumer's trilinear filter cannot resolve it and
                // the front snaps to axis-aligned cell faces — visible as a
                // staircase of colour cubes, which is precisely the artefact
                // the old "Dominant Cell Material" mode produced.
                const float m = std::min(slot_miscibility[best], slot_miscibility[second]);
                if (m < 0.999f) {
                    const float gain = 1.0f / std::max(m, 1.0f / 64.0f);
                    w = std::min(1.0f, std::max(0.0f, 0.5f + (w - 0.5f) * gain));
                }

                composition_out[o + 0] = static_cast<float>(slot_material[best] + 1);
                composition_out[o + 1] = static_cast<float>(slot_material[second] + 1);
                composition_out[o + 2] = w;
            }
        }
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBSTANCE VISCOSITY FIELD
// ═══════════════════════════════════════════════════════════════════════════
// See the header for why this gather takes NO exclusion list: its neighbours
// describe one isosurface, this one describes the liquid.
bool buildSubstanceViscosityField(const FluidParticles& particles,
                                  const FluidSim::FluidGrid& grid,
                                  const SubstanceViscosityEntry* entries,
                                  std::size_t entry_count,
                                  float fallback_viscosity,
                                  std::vector<float>& viscosity_out)
{
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float voxel = grid.voxel_size;
    const std::size_t cell_count = static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny) *
                                   static_cast<std::size_t>(nz);
    const std::size_t particle_count = particles.size();
    const float fallback = std::max(0.0f, fallback_viscosity);

    if (cell_count == 0 || particle_count == 0 || voxel <= 0.0f ||
        entries == nullptr || entry_count == 0 ||
        particles.substance_tag.size() < particle_count) {
        viscosity_out.clear();
        return false;
    }

    // ★ NOTHING TO DESCRIBE. Every entry either inherits or asks for exactly the
    // domain value, so the field would be uniform — and a uniform field is what
    // the scalar `params.kinematic_viscosity` path already delivers without an
    // allocation, an upload, or a per-face lookup. Refusing here is what keeps
    // single-substance domains exactly as fast as they were.
    bool any_override = false;
    for (std::size_t i = 0; i < entry_count; ++i) {
        const float nu = entries[i].kinematic_viscosity;
        if (nu >= 0.0f && std::fabs(nu - fallback) > 1e-9f) { any_override = true; break; }
    }
    if (!any_override) {
        viscosity_out.clear();
        return false;
    }

    auto viscosityForTag = [&](uint32_t tag) -> float {
        if (tag == kSubstanceUntagged) return fallback;
        for (std::size_t i = 0; i < entry_count; ++i) {
            if (entries[i].tag != tag) continue;
            // ★ Negative is INHERIT, not "zero viscosity". Reading it as 0 would
            // turn "I did not author this" into "this substance is inviscid",
            // which renders as a plausible thin liquid rather than as an error.
            return entries[i].kinematic_viscosity < 0.0f
                 ? fallback : entries[i].kinematic_viscosity;
        }
        // A tag with no entry is liquid from a renamed source or a mass
        // transfer. It takes the domain viscosity, exactly like untagged liquid.
        return fallback;
    };

    // Per-particle viscosity, resolved once — the splat below touches each
    // particle 8 times and the lookup is a linear scan.
    static std::vector<float> s_particle_nu;
    s_particle_nu.assign(particle_count, fallback);
    for (std::size_t p = 0; p < particle_count; ++p)
        s_particle_nu[p] = viscosityForTag(particles.substance_tag[p]);

    static std::vector<float> s_weight;
    s_weight.assign(cell_count, 0.0f);
    viscosity_out.assign(cell_count, 0.0f);

    auto cellIndex = [nx, ny](int i, int j, int k) -> std::size_t {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
               static_cast<std::size_t>(ny);
    };

    // Trilinear splat onto CELL CENTRES — the same weights the transfer uses, so
    // a cell the fluid mask calls fluid is a cell this field has support in.
    // Serial: this is one pass over particles with scattered writes, and the
    // atomics or per-thread grids needed to parallelise it cost more than the
    // pass at the sizes this runs at.
    const float inv_h = 1.0f / voxel;
    for (std::size_t p = 0; p < particle_count; ++p) {
        if (p < particles.mass_fraction.size() &&
            particles.mass_fraction[p] <= 0.02f) continue;
        const Vec3& wp = particles.position[p];
        if (!std::isfinite(wp.x) || !std::isfinite(wp.y) || !std::isfinite(wp.z)) continue;
        // Cell-centre space: centre of cell (i,j,k) sits at integer (i,j,k).
        const Vec3 c = (wp - grid.origin) * inv_h - Vec3(0.5f, 0.5f, 0.5f);
        const int i0 = static_cast<int>(std::floor(c.x));
        const int j0 = static_cast<int>(std::floor(c.y));
        const int k0 = static_cast<int>(std::floor(c.z));
        const float fx = c.x - static_cast<float>(i0);
        const float fy = c.y - static_cast<float>(j0);
        const float fz = c.z - static_cast<float>(k0);
        const float nu = s_particle_nu[p];
        for (int dk = 0; dk <= 1; ++dk)
        for (int dj = 0; dj <= 1; ++dj)
        for (int di = 0; di <= 1; ++di) {
            const int i = i0 + di, j = j0 + dj, k = k0 + dk;
            if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
            const float w = (di ? fx : 1.0f - fx) *
                            (dj ? fy : 1.0f - fy) *
                            (dk ? fz : 1.0f - fz);
            if (w <= 0.0f) continue;
            const std::size_t ci = cellIndex(i, j, k);
            viscosity_out[ci] += w * nu;
            s_weight[ci]      += w;
        }
    }

    for (std::size_t ci = 0; ci < cell_count; ++ci) {
        // ★ An unsupported cell gets the FALLBACK, not zero. Zero is a real
        // viscosity — the inviscid one — so leaving it there would carve
        // frictionless pockets into a honey domain wherever the splat happened
        // not to reach, and they would read as "the viscosity is uneven" rather
        // than as missing data.
        viscosity_out[ci] = (s_weight[ci] > 1e-8f)
                          ? viscosity_out[ci] / s_weight[ci]
                          : fallback;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// SOLID-PHASE CELLS
// ═══════════════════════════════════════════════════════════════════════════
// See the header for why this rasterizes to the NEAREST cell while its three
// neighbours splat trilinearly: this one answers "does matter block flow
// here", and a smeared obstacle is a bigger obstacle, never a softer one.
bool buildSubstanceSolidCells(const FluidParticles& particles,
                              const FluidSim::FluidGrid& grid,
                              const uint32_t* solid_tags,
                              std::size_t solid_tag_count,
                              float fill_threshold,
                              std::vector<uint32_t>& cells_out,
                              std::vector<Vec3>& cell_velocity_out,
                              bool include_frozen)
{
    cells_out.clear();
    cell_velocity_out.clear();

    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float voxel = grid.voxel_size;
    const std::size_t cell_count = static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny) *
                                   static_cast<std::size_t>(nz);
    const std::size_t particle_count = particles.size();
    const bool use_tags = solid_tags != nullptr && solid_tag_count > 0 &&
                          particles.substance_tag.size() >= particle_count;
    const bool use_frozen = include_frozen && particles.flags.size() >= particle_count;
    if (cell_count == 0 || particle_count == 0 || voxel <= 0.0f ||
        (!use_tags && !use_frozen)) {
        return false;
    }

    // Accumulate solid mass and momentum per cell. Sparse in spirit but dense
    // in storage: one float + one Vec3 per cell of function-static scratch is
    // cheaper than hashing, and this runs once per domain per step.
    static std::vector<float> s_mass;
    static std::vector<Vec3>  s_momentum;
    static std::vector<float> s_liquid_mass;
    s_mass.assign(cell_count, 0.0f);
    s_momentum.assign(cell_count, Vec3(0.0f, 0.0f, 0.0f));
    s_liquid_mass.assign(cell_count, 0.0f);

    const float inv_h = 1.0f / voxel;
    bool any_solid_parcel = false;
    for (std::size_t p = 0; p < particle_count; ++p) {
        const bool solid_parcel = isMatterObstacle(
            particles, p, use_tags ? solid_tags : nullptr, solid_tag_count, use_frozen);
        const float mass = (p < particles.mass_fraction.size())
            ? std::max(0.0f, std::min(1.0f, particles.mass_fraction[p]))
            : 1.0f;
        // ★ A parcel that has burned or evaporated away still exists in the
        // array until the lifecycle pass compacts it. Letting it hold a solid
        // cell open would leave ghost obstacles standing where matter is gone.
        if (mass <= 0.02f) continue;
        const Vec3& wp = particles.position[p];
        if (!std::isfinite(wp.x) || !std::isfinite(wp.y) || !std::isfinite(wp.z)) continue;
        const Vec3 g = (wp - grid.origin) * inv_h;
        const int i = static_cast<int>(std::floor(g.x));
        const int j = static_cast<int>(std::floor(g.y));
        const int k = static_cast<int>(std::floor(g.z));
        if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) continue;
        const std::size_t ci = static_cast<std::size_t>(i) +
                               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
                               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                               static_cast<std::size_t>(ny);
        // ★★★ LIQUID IS COUNTED TOO, and that second accumulator is the whole
        // fix for "liquid approaching the solid suddenly explodes outward".
        //
        // A cell that flips to solid while liquid parcels are still standing in
        // it does two violent things at once: the pressure solve stops seeing
        // that liquid's volume (the fluid mask calls the cell a wall, so its
        // divergence contribution vanishes in one step), and the particle-level
        // solid resolution then EJECTS every one of those parcels to the nearest
        // free cell. Hundreds of parcels teleporting outward on the same frame
        // is exactly the reported burst -- and it reads as an instability in the
        // solver rather than as a mask that changed under the liquid's feet.
        if (solid_parcel) {
            s_mass[ci] += mass;
            s_momentum[ci] = s_momentum[ci] +
                (p < particles.velocity.size() ? particles.velocity[p] * mass
                                               : Vec3(0.0f, 0.0f, 0.0f));
            any_solid_parcel = true;
        } else {
            s_liquid_mass[ci] += mass;
        }
    }
    if (!any_solid_parcel) return false;

    // ★ At least one parcel's worth, whatever the caller asked for. A threshold
    // below one mass unit cannot be met by fewer parcels than one, so clamping
    // here keeps a mis-set (or zero) argument from turning every isolated
    // droplet of a solid substance into a full cell of wall.
    const float threshold = std::max(1.0f, fill_threshold);

    // -- Hysteresis: what was solid last step is harder to give up ------------
    // ★★★ THE CURE FOR "INTERMITTENT HARD KICKS". Rebuilt from scratch each
    // step, a cell sitting near the threshold flips solid/fluid/solid across
    // consecutive steps -- parcels drift by a fraction of a cell and the count
    // crosses back and forth. Every flip is a step change in the PRESSURE
    // OPERATOR: a cell that carried fluid divergence becomes a wall in one
    // step, the projection answers with an impulse, and the FLIP transfer hands
    // that impulse to the particles. It reads as the chunk periodically
    // punching the liquid, which is exactly what a user reports as "sert bir
    // kuvvet / darbe gibi".
    //
    // ★★ Two thresholds, not one: a cell must reach `threshold` to BECOME
    // solid and fall below `release` to stop being solid. Between them the
    // previous answer stands. A single threshold with any smoothing on top
    // would still chatter -- what removes chatter is a band, not a filter.
    //
    // ★ Release at 60%: low enough that a chunk which genuinely left a cell
    // frees it within a step or two (so the mask never lags visibly behind the
    // matter), high enough that ordinary sub-cell jitter cannot cross it.
    static std::vector<uint8_t> s_was_solid;
    static std::vector<Vec3>    s_prev_vel;
    const bool has_history = !grid.substance_solid_prev_cells.empty();
    if (has_history) {
        s_was_solid.assign(cell_count, 0u);
        s_prev_vel.assign(cell_count, Vec3(0.0f, 0.0f, 0.0f));
        const auto& prev_cells = grid.substance_solid_prev_cells;
        const auto& prev_vel = grid.substance_solid_prev_vel;
        for (std::size_t n = 0; n < prev_cells.size(); ++n) {
            const uint32_t c = prev_cells[n];
            if (c >= cell_count) continue;
            s_was_solid[c] = 1u;
            if (n < prev_vel.size()) s_prev_vel[c] = prev_vel[n];
        }
    }
    const float release = threshold * 0.6f;

    for (std::size_t ci = 0; ci < cell_count; ++ci) {
        const bool was_solid = has_history && s_was_solid[ci] != 0u;
        const float enter_or_stay = was_solid ? release : threshold;
        if (s_mass[ci] < enter_or_stay) continue;
        // ★★★ THE SOLID MUST DOMINATE THE CELL, not merely reach the threshold.
        // An interface cell holding both phases stays FLUID: the liquid in it
        // keeps its volume in the pressure solve and is never ejected, so the
        // boundary forms one cell further into the solid instead of forming
        // underneath the liquid. The cost is exactly that -- a chunk wets one
        // cell deep, which looks like surface tension and is invisible next to
        // the burst it replaces.
        //
        // ★ Strictly greater, and against the liquid mass rather than a tuned
        // ratio: at equal mass the honest answer is "this cell is a mixture",
        // and a mixture is not a wall. A ratio would be one more number to
        // calibrate per scene for a decision that has a correct default.
        if (s_mass[ci] <= s_liquid_mass[ci]) continue;
        cells_out.push_back(static_cast<uint32_t>(ci));
        Vec3 cell_vel = s_momentum[ci] * (1.0f / s_mass[ci]);
        // ★★ A cell that was ALREADY solid low-passes its wall velocity instead
        // of adopting this step's parcel average outright. That average is taken
        // over a handful of parcels, so it is noisy by construction -- and this
        // value is imposed on the liquid as a moving wall by both the viscous
        // stencil and the variational projection. Feeding raw per-step noise
        // into a boundary condition is the second way this coupling manufactures
        // impulses, and unlike the mask flip it survives any amount of
        // hysteresis. A freshly solid cell takes its value as measured: there is
        // no history to blend with, and inventing one would make a chunk arrive
        // slower than it does.
        if (was_solid) cell_vel = (cell_vel + s_prev_vel[ci]) * 0.5f;
        cell_velocity_out.push_back(cell_vel);
    }
    // ★ Parcels present but no cell full enough is a REAL and reportable state,
    // not an error: the chunk is thinner than the voxel size can express. The
    // caller gets false and leaves the mask alone, and the diagnostic that says
    // "solid parcels: N, solid cells: 0" is what tells a user to raise the
    // resolution instead of doubting the phase control.
    return !cells_out.empty();
}

} // namespace Fluid
} // namespace RayTrophiSim
