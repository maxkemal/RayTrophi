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
#include "Fluid/SubstanceTag.h"

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

void smoothSDF(std::vector<float>& sdf, int nx, int ny, int nz, float far_value, int iterations, int thread_cap) {
    if (iterations <= 0 || sdf.size() != static_cast<size_t>(nx) * ny * nz) return;

    const float alpha = 0.5f;

    auto cellIndex = [nx, ny](int i, int j, int k) {
        return static_cast<size_t>(i) + static_cast<size_t>(j) * nx + static_cast<size_t>(k) * nx * ny;
    };

    // Ping-pong between sdf and one scratch buffer instead of copying the whole
    // grid back each iteration (the old `temp = sdf` was O(cell_count) per
    // sweep). src holds the current field, dst receives the smoothed one; every
    // cell writes dst so no stale values survive the swap. Scratch is function-
    // static (reused across calls — never a per-call vector, per the heap-lock
    // rule); callers are sequential on the main thread.
    static std::vector<float> s_scratch;
    if (s_scratch.size() < sdf.size()) s_scratch.resize(sdf.size());
    std::vector<float>* src = &sdf;
    std::vector<float>* dst = &s_scratch;

    for (int iter = 0; iter < iterations; ++iter) {
        const std::vector<float>& in = *src;
        std::vector<float>& outv = *dst;
#ifdef _OPENMP
        #pragma omp parallel for collapse(3) schedule(static) num_threads(thread_cap)
#endif
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    size_t c = cellIndex(i, j, k);
                    float val = in[c];

                    // Skip cells that are completely outside the narrow band on all sides to avoid work
                    if (val >= far_value - 1e-5f) {
                        bool neighbor_active = false;
                        if (i > 0      && in[c - 1] < far_value - 1e-5f) neighbor_active = true;
                        if (i + 1 < nx && in[c + 1] < far_value - 1e-5f) neighbor_active = true;
                        if (j > 0      && in[c - nx] < far_value - 1e-5f) neighbor_active = true;
                        if (j + 1 < ny && in[c + nx] < far_value - 1e-5f) neighbor_active = true;
                        if (k > 0      && in[c - nx * ny] < far_value - 1e-5f) neighbor_active = true;
                        if (k + 1 < nz && in[c + nx * ny] < far_value - 1e-5f) neighbor_active = true;

                        if (!neighbor_active) {
                            outv[c] = far_value;
                            continue;
                        }
                    }

                    float sum = 0.0f;
                    float count = 0.0f;

                    if (i > 0)      { sum += in[c - 1]; count += 1.0f; }
                    if (i + 1 < nx) { sum += in[c + 1]; count += 1.0f; }
                    if (j > 0)      { sum += in[c - nx]; count += 1.0f; }
                    if (j + 1 < ny) { sum += in[c + nx]; count += 1.0f; }
                    if (k > 0)      { sum += in[c - nx * ny]; count += 1.0f; }
                    if (k + 1 < nz) { sum += in[c + nx * ny]; count += 1.0f; }

                    outv[c] = (count > 0.0f) ? ((1.0f - alpha) * val + alpha * (sum / count)) : val;
                }
            }
        }
        std::swap(src, dst);
    }

    // Final field lives in *src; copy back only when it ended in scratch. Copy
    // exactly sdf.size() elements — s_scratch may be larger (reused from a
    // bigger grid), so a whole-vector assign would wrongly resize sdf.
    if (src != &sdf) std::copy(src->begin(), src->begin() + sdf.size(), sdf.begin());
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
                   const std::vector<uint32_t>* excluded_substance_tags)
{
    using clock = std::chrono::steady_clock;
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

    // ── 1. CSR particle->cell bin --------------------------------------------
    // particle_cell[p] = flat cell index, or kInvalid if the particle is out
    // of grid bounds (silently dropped from the SDF).
    constexpr std::size_t kInvalid = static_cast<std::size_t>(-1);
    std::vector<std::size_t> particle_cell(particle_count, kInvalid);
    std::vector<int> cell_count_per(cell_count, 0);

    const float inv_h = 1.0f / voxel;
    for (std::size_t p = 0; p < particle_count; ++p) {
        if (excluded_substance_tags && p < particles.substance_tag.size() &&
            std::find(excluded_substance_tags->begin(), excluded_substance_tags->end(),
                      particles.substance_tag[p]) != excluded_substance_tags->end()) continue;
        // Depleted fuel particles remain in the stable APIC pool until the
        // lifecycle compaction pass. Do not let an already evaporated slot
        // keep the rendered SurfaceSDF artificially thick.
        if (p < particles.mass_fraction.size() &&
            particles.mass_fraction[p] <= 0.02f) continue;
        const Vec3& wp = particles.position[p];
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

    // Prefix-sum -> CSR offsets.
    std::vector<std::size_t> cell_offset(cell_count + 1, 0);
    for (std::size_t c = 0; c < cell_count; ++c) {
        cell_offset[c + 1] = cell_offset[c] + static_cast<std::size_t>(cell_count_per[c]);
    }
    std::vector<std::size_t> cell_csr(cell_offset.back());
    std::vector<std::size_t> cursor(cell_count, 0);
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
    if (use_aniso) {
        if (s_xtilde.size() < particle_count) s_xtilde.resize(particle_count);
        if (s_aniso.size()  < particle_count) s_aniso.resize(particle_count);

        const float rs     = std::max(1e-4f, params.anisotropy_radius_voxels * sim_voxel);
        const float rs_sq  = rs * rs;
        const int   reach_a = std::max(1, static_cast<int>(
            std::ceil(params.anisotropy_radius_voxels * static_cast<float>(m))));
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

            const Vec3 lp = (xp - origin) * inv_h;
            const int ci = static_cast<int>(std::floor(lp.x));
            const int cj = static_cast<int>(std::floor(lp.y));
            const int ck = static_cast<int>(std::floor(lp.z));
            const int ai0 = std::max(0, ci - reach_a), ai1 = std::min(nx - 1, ci + reach_a);
            const int aj0 = std::max(0, cj - reach_a), aj1 = std::min(ny - 1, cj + reach_a);
            const int ak0 = std::max(0, ck - reach_a), ak1 = std::min(nz - 1, ck + reach_a);

            // Single-pass weighted mean + covariance. Accumulate the moments
            // relative to xp (rel = xj - xp, magnitudes < rs) so the float
            // products stay small and precise, then C = E[rr] - mean_rel(x)mean_rel
            // — one stencil traversal instead of two.
            float wsum = 0.0f; int ncount = 0;
            Vec3 mom1(0, 0, 0);                                // sum w*rel
            Sym3 mom2;                                         // sum w*(rel (x) rel)
            for (int kk = ak0; kk <= ak1; ++kk)
            for (int jj = aj0; jj <= aj1; ++jj)
            for (int ii = ai0; ii <= ai1; ++ii) {
                const std::size_t nci = cellIndex(ii, jj, kk);
                for (std::size_t a = cell_offset[nci]; a < cell_offset[nci + 1]; ++a) {
                    const Vec3& xj = particles.position[cell_csr[a]];
                    const float rx = xj.x - xp.x, ry = xj.y - xp.y, rz = xj.z - xp.z;
                    const float d2 = rx * rx + ry * ry + rz * rz;
                    if (d2 >= rs_sq) continue;
                    const float tt = 1.0f - d2 / rs_sq;
                    const float w = tt * tt * tt;
                    wsum += w; ++ncount;
                    mom1.x += w * rx; mom1.y += w * ry; mom1.z += w * rz;
                    mom2.xx += w * rx * rx; mom2.yy += w * ry * ry; mom2.zz += w * rz * rz;
                    mom2.xy += w * rx * ry; mom2.xz += w * rx * rz; mom2.yz += w * ry * rz;
                }
            }
            if (wsum < 1e-12f) { s_xtilde[pidx] = xp; s_aniso[pidx] = isoG; continue; }
            const float invw = 1.0f / wsum;
            const Vec3 mr(mom1.x * invw, mom1.y * invw, mom1.z * invw); // mean relative to xp
            s_xtilde[pidx] = Vec3(xp.x + mr.x * lambda,
                                  xp.y + mr.y * lambda,
                                  xp.z + mr.z * lambda);
            if (ncount < nmin) { s_aniso[pidx] = isoG; continue; }

            Sym3 C;
            C.xx = mom2.xx * invw - mr.x * mr.x; C.yy = mom2.yy * invw - mr.y * mr.y; C.zz = mom2.zz * invw - mr.z * mr.z;
            C.xy = mom2.xy * invw - mr.x * mr.y; C.xz = mom2.xz * invw - mr.x * mr.z; C.yz = mom2.yz * invw - mr.y * mr.z;

            float ev[3]; Vec3 R[3];
            analyticEigen3(C, ev, R);
            const float emax = std::max(ev[0], std::max(ev[1], ev[2]));
            if (emax < 1e-12f) { s_aniso[pidx] = isoG; continue; }
            // Clamp each axis variance to [emax/kr, emax] then volume-normalise so
            // det == 1 (every particle's kernel covers ~the same volume — no blob
            // dominates). Stretch the kernel along low-variance (thin) axes by
            // using 1/sigma so a sheet's surface stays flat instead of bulging.
            const float floorv = emax / kr;
            float s0 = std::max(ev[0], floorv);
            float s1 = std::max(ev[1], floorv);
            float s2 = std::max(ev[2], floorv);
            const float vol = std::cbrt(std::max(1e-20f, s0 * s1 * s2));
            s0 /= vol; s1 /= vol; s2 /= vol;
            s_aniso[pidx] = symFromEigen(R, inv_kR / s0, inv_kR / s1, inv_kR / s2);
        }
    }

    // ── 2. Output pass --------------------------------------------------------
    // Stencil half-width in REFINED cells. kernel_R is physical (reach grows with
    // the multiplier). The anisotropic kernel stretches along its major axis; the
    // volume-normalised covariance bounds the worst-case major-axis stretch at
    // kr^(2/3) (one axis at the clamp ceiling, two at emax/kr), so size the reach
    // to exactly that — no clipping of elongated kernels, and reach shrinks with
    // smaller anisotropy_max_stretch instead of paying a fixed 2.5x.
    const float kr_stretch = std::max(1.0f, params.anisotropy_max_stretch);
    const float stretch_reach = use_aniso
        ? std::clamp(std::cbrt(kr_stretch * kr_stretch), 1.0f, 4.0f) : 1.0f;
    const int reach = std::max(1, static_cast<int>(
        std::ceil(params.kernel_radius_voxels * static_cast<float>(m) * stretch_reach)));

    // ── 2a. Active-cell band -------------------------------------------------
    // Only refined cells within `reach` of an occupied cell can pick up a
    // contribution; everything else keeps the far_value already assigned. A
    // full-grid sweep wastes a stencil traversal on the empty air around the
    // surface (the bulk of a typical domain). Dilate the occupancy by `reach`
    // (three separable 1-D OR passes) and gather only the survivors. Function-
    // static scratch — reused across calls, never per-call alloc.
    static std::vector<uint8_t> s_occ_a, s_occ_b;
    static std::vector<int64_t> s_active;
    if (s_occ_a.size() < cell_count) s_occ_a.resize(cell_count);
    if (s_occ_b.size() < cell_count) s_occ_b.resize(cell_count);
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) num_threads(thread_cap)
#endif
    for (int64_t c = 0; c < static_cast<int64_t>(cell_count); ++c)
        s_occ_a[static_cast<std::size_t>(c)] = (cell_count_per[static_cast<std::size_t>(c)] > 0) ? 1 : 0;

    // X dilation: s_occ_a -> s_occ_b
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static) num_threads(thread_cap)
#endif
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j) {
        const std::size_t base = static_cast<std::size_t>(k) * plane + static_cast<std::size_t>(j) * nx;
        for (int i = 0; i < nx; ++i) {
            const int lo = std::max(0, i - reach), hi = std::min(nx - 1, i + reach);
            uint8_t v = 0;
            for (int ii = lo; ii <= hi; ++ii) if (s_occ_a[base + ii]) { v = 1; break; }
            s_occ_b[base + i] = v;
        }
    }
    // Y dilation: s_occ_b -> s_occ_a
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static) num_threads(thread_cap)
#endif
    for (int k = 0; k < nz; ++k)
    for (int i = 0; i < nx; ++i) {
        const std::size_t base_k = static_cast<std::size_t>(k) * plane;
        for (int j = 0; j < ny; ++j) {
            const int lo = std::max(0, j - reach), hi = std::min(ny - 1, j + reach);
            uint8_t v = 0;
            for (int jj = lo; jj <= hi; ++jj) if (s_occ_b[base_k + static_cast<std::size_t>(jj) * nx + i]) { v = 1; break; }
            s_occ_a[base_k + static_cast<std::size_t>(j) * nx + i] = v;
        }
    }
    // Z dilation: s_occ_a -> s_occ_b
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static) num_threads(thread_cap)
#endif
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        const std::size_t col = static_cast<std::size_t>(j) * nx + i;
        for (int k = 0; k < nz; ++k) {
            const int lo = std::max(0, k - reach), hi = std::min(nz - 1, k + reach);
            uint8_t v = 0;
            for (int kk = lo; kk <= hi; ++kk) if (s_occ_a[static_cast<std::size_t>(kk) * plane + col]) { v = 1; break; }
            s_occ_b[static_cast<std::size_t>(k) * plane + col] = v;
        }
    }

    // Compact active cells into an index list (serial, O(cell_count)).
    s_active.clear();
    for (std::size_t c = 0; c < cell_count; ++c)
        if (s_occ_b[c]) s_active.push_back(static_cast<int64_t>(c));
    const int64_t active_n = static_cast<int64_t>(s_active.size());

    std::size_t active_cells = 0;
    std::size_t surface_cells = 0;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 256) num_threads(thread_cap) \
        reduction(+:active_cells, surface_cells)
#endif
    for (int64_t at = 0; at < active_n; ++at) {
            const std::size_t c_flat = static_cast<std::size_t>(s_active[static_cast<std::size_t>(at)]);
            const int k = static_cast<int>(c_flat / plane);
            const std::size_t rem = c_flat - static_cast<std::size_t>(k) * plane;
            const int j = static_cast<int>(rem / nx);
            const int i = static_cast<int>(rem % nx);
                const Vec3 p_c = origin + Vec3(
                    (static_cast<float>(i) + 0.5f) * voxel,
                    (static_cast<float>(j) + 0.5f) * voxel,
                    (static_cast<float>(k) + 0.5f) * voxel);

                const int i0 = std::max(0, i - reach);
                const int i1 = std::min(nx - 1, i + reach);
                const int j0 = std::max(0, j - reach);
                const int j1 = std::min(ny - 1, j + reach);
                const int k0 = std::max(0, k - reach);
                const int k1 = std::min(nz - 1, k + reach);

                float acc_w = 0.0f;
                float acc_mass = 0.0f;
                Vec3  acc_p(0.0f, 0.0f, 0.0f);

                for (int kk = k0; kk <= k1; ++kk) {
                    for (int jj = j0; jj <= j1; ++jj) {
                        for (int ii = i0; ii <= i1; ++ii) {
                            const std::size_t nci = cellIndex(ii, jj, kk);
                            const std::size_t beg = cell_offset[nci];
                            const std::size_t end = cell_offset[nci + 1];
                            for (std::size_t a = beg; a < end; ++a) {
                                const std::size_t pa = cell_csr[a];
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
                                    acc_w += w;
                                    acc_mass += w * mass;
                                    acc_p.x += w * xt.x;
                                    acc_p.y += w * xt.y;
                                    acc_p.z += w * xt.z;
                                } else {
                                    const Vec3& pp = particles.position[pa];
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
                    const Vec3 x_bar(acc_p.x * inv_w, acc_p.y * inv_w, acc_p.z * inv_w);
                    const float dlen = std::sqrt(lengthSq(p_c - x_bar));
                    const float mean_mass = std::clamp(acc_mass / std::max(acc_w, 1.0e-12f), 0.0f, 1.0f);
                    const float mass_radius = particle_r * std::cbrt(std::max(mean_mass, 0.02f));
                    const float phi = dlen - mass_radius - surface_offset;
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

    // ── 3. Optional fast Laplacian smoothing sweeps --------------------------
    if (params.smoothing_iterations > 0) {
        smoothSDF(sdf_out, nx, ny, nz, far_value, params.smoothing_iterations, thread_cap);

        // Recalculate active and surface cell counts on the smoothed field
        active_cells = 0;
        surface_cells = 0;
#ifdef _OPENMP
        #pragma omp parallel for reduction(+:active_cells, surface_cells) num_threads(thread_cap)
#endif
        for (int64_t ci = 0; ci < static_cast<int64_t>(cell_count); ++ci) {
            float val = sdf_out[static_cast<size_t>(ci)];
            if (val < far_value - 1e-5f) {
                ++active_cells;
                if (std::abs(val) < voxel) {
                    ++surface_cells;
                }
            }
        }
    }

    if (stats) {
        stats->active_cells = active_cells;
        stats->surface_cells = surface_cells;
        stats->build_ms = std::chrono::duration<float, std::milli>(
                              clock::now() - t_start).count();
    }
    return active_cells > 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// THIS GRID HOLDS A DISPLACEMENT, NOT A COORDINATE.
// ═══════════════════════════════════════════════════════════════════════════
// Cell c stores  d = mean over nearby particles of (uvw - position), and the
// shader reconstructs the material coordinate as  worldPos + trilinear(d).
//
// ★★★ The reason is a resolution argument, and it is the entire quality story
// of this feature. Split the coordinate into its two natural parts:
//
//     uvw(x) = x + d(x)
//              ^   ^
//              |   deformation: genuinely LOW frequency, belongs on a grid
//              position: FULL resolution, continuous, free in the shader
//
// Storing the sum condemns the full-resolution half to the sim voxel size
// (typically 5 cm), which is precisely the reported symptom: "material mode
// looks like one pixel per cell". Storing only d leaves the position term
// exact, so the coordinate is continuous everywhere and only the DEFORMATION is
// grid-limited — and deformation is smooth, so that limit costs nothing.
//
// ★ It also makes the feature falsifiable. Liquid that has not moved has
// uvw == position for EVERY particle, so d is identically zero — not "small",
// zero, and independently of how the particles are distributed, because the
// difference is taken per particle before the average. COORD_MATERIAL then
// reduces to worldPos exactly: a resting tank must be pixel-identical in
// Material and World mode. That turns "is the quality good enough?" from a
// judgement call into a diff, which is the only reason this note is worth
// writing down.
//
// Two consequences that are easy to get wrong and are handled below:
//   - extrapolation averages d directly, with NO world offsets (section 3)
//   - d may be filtered freely, which the absolute form could never allow (4)
bool buildMaterialCoordinateGrid(const FluidParticles& particles,
                                 const FluidSim::FluidGrid& grid,
                                 const LevelSetParams& params,
                                 std::vector<float>& uvw_out,
                                 const std::vector<uint32_t>* excluded_substance_tags)
{
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float voxel = grid.voxel_size;
    const std::size_t cell_count = static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny) *
                                   static_cast<std::size_t>(nz);
    const std::size_t particle_count = particles.size();

    // A uvw sidecar shorter than the particle array means some producer emitted
    // without one. Refuse rather than gathering a partial field: a coordinate
    // that is right for most of the body and origin-seeded for the rest reads as
    // a torn texture, which looks like a shader bug and is far harder to trace
    // back here than an outright "no coordinate, world-anchored as before".
    if (cell_count == 0 || particle_count == 0 || voxel <= 0.0f ||
        particles.uvw.size() < particle_count) {
        uvw_out.clear();
        return false;
    }

    uvw_out.assign(cell_count * 3u, 0.0f);

    auto cellIndex = [nx, ny](int i, int j, int k) -> std::size_t {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
               static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
               static_cast<std::size_t>(ny);
    };

    // Same kernel the level set uses, so the coordinate is supported wherever
    // the surface is. Deliberately NOT a wider one: a wider kernel would smear
    // uvw across a gap between two separate bodies of liquid and blend two
    // unrelated coordinates into the space between them.
    // Generation weights for this frame. Read once: they are a property of the
    // schedule, not of the cell being filled, and re-deriving them per particle
    // would let a mid-gather change tear the field.
    float w_a = 1.0f, w_b = 0.0f;
    particles.materialCoordWeights(w_a, w_b);
    // A short second generation means it was never built (an older cache, a
    // producer that emitted without it). Fall back to generation A alone rather
    // than mixing against zeros, which would drag the coordinate toward the
    // identity by w_b and read as the texture partly detaching from the liquid.
    const bool has_gen_b = particles.uvw_b.size() >= particle_count;
    if (!has_gen_b) { w_a = 1.0f; w_b = 0.0f; }

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

    // ── 1. CSR particle -> cell bin (sim grid) ───────────────────────────────
    constexpr std::size_t kInvalid = static_cast<std::size_t>(-1);
    static std::vector<std::size_t> s_particle_cell;
    static std::vector<int>         s_count_per;
    static std::vector<std::size_t> s_offset, s_csr, s_cursor;
    s_particle_cell.assign(particle_count, kInvalid);
    s_count_per.assign(cell_count, 0);

    const float inv_h = 1.0f / voxel;
    for (std::size_t p = 0; p < particle_count; ++p) {
        // Match the level set's depleted-particle rule exactly. If the two
        // disagreed, the coordinate field would be supported in cells the
        // surface does not occupy (or the reverse) — and the reverse is the
        // dangerous direction, because it puts an unsupported cell under a
        // surface that IS drawn.
        if (p < particles.mass_fraction.size() &&
            particles.mass_fraction[p] <= 0.02f) continue;
        // ★ Same exclusion the level set applied. A substance routed to splat
        // has no isosurface, so it must not contribute to a field that describes
        // one — the three gathers have to agree about which particles the
        // surface is made of, or the surface takes on the look of liquid that
        // is not there. That reads as plausible wetness, not as a bug.
        if (excluded_substance_tags && p < particles.substance_tag.size() &&
            std::find(excluded_substance_tags->begin(), excluded_substance_tags->end(),
                      particles.substance_tag[p]) != excluded_substance_tags->end()) continue;
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
    if (s_offset.back() == 0) {          // every particle fell outside the grid
        uvw_out.clear();
        return false;
    }
    s_csr.assign(s_offset.back(), 0);
    s_cursor.assign(cell_count, 0);
    for (std::size_t p = 0; p < particle_count; ++p) {
        const std::size_t ci = s_particle_cell[p];
        if (ci == kInvalid) continue;
        s_csr[s_offset[ci] + s_cursor[ci]++] = p;
    }

    // ── 2. Weighted gather ───────────────────────────────────────────────────
    // valid[c] marks cells that received real support; everything else is filled
    // by the extrapolation below. A separate mask rather than testing the value
    // against zero: (0,0,0) is a perfectly legal material coordinate for liquid
    // born at the world origin, and using the value as its own validity flag
    // would make that liquid's cells re-fill from their neighbours every frame.
    static std::vector<uint8_t> s_valid;
    s_valid.assign(cell_count, 0u);

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

                float acc_w = 0.0f;
                Vec3  acc_u(0.0f, 0.0f, 0.0f);
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
                        // ★★★ THE DIFFERENCE IS TAKEN PER PARTICLE, not by
                        // subtracting the cell centre from the averaged
                        // coordinate afterwards. The two are NOT the same:
                        //
                        //   mean(uvw_i) - c      carries (mean(p_i) - c), the
                        //                        centroid of whichever particles
                        //                        happened to sit near this cell
                        //   mean(uvw_i - p_i)    the average DISPLACEMENT, and
                        //                        the sampling bias cancels
                        //
                        // The first leaves millimetres of per-cell jitter that
                        // has no physical meaning and reads as cell-frequency
                        // noise. The second is exactly zero for liquid that has
                        // not moved, whatever the particle distribution does.
                        // ★ TWO GENERATIONS, BLENDED HERE — per particle, before
                        // the gather. Blending is linear and so is the gather,
                        // so this is identical to building two grids and mixing
                        // them, at half the memory and with NOTHING to change
                        // downstream: the buffer, the ABI and the shader stay
                        // exactly as they were.
                        //
                        // ★★ And it is only well-posed because these are
                        // DISPLACEMENTS. Mixing two absolute coordinates would
                        // average two positions and flatten the gradient — the
                        // same failure the extrapolation sweep once had, and the
                        // reason the residual work had to land first.
                        const Vec3 dp_a = particles.uvw[pa] - particles.position[pa];
                        const Vec3 dp_b = has_gen_b
                            ? (particles.uvw_b[pa] - particles.position[pa]) : dp_a;
                        acc_w += w;
                        acc_u.x += w * (w_a * dp_a.x + w_b * dp_b.x);
                        acc_u.y += w * (w_a * dp_a.y + w_b * dp_b.y);
                        acc_u.z += w * (w_a * dp_a.z + w_b * dp_b.z);
                    }
                }

                if (acc_w > 1e-12f) {
                    const float inv_w = 1.0f / acc_w;
                    const std::size_t o = cellIndex(i, j, k) * 3u;
                    // ★★★ RESIDUAL, not the absolute coordinate. The cell stores
                    // the mean displacement (uvw - position); the shader
                    // reconstructs uvw as worldPos + trilinear(residual). See the
                    // block comment above the function for why this is the whole
                    // quality story.
                    uvw_out[o + 0] = acc_u.x * inv_w;
                    uvw_out[o + 1] = acc_u.y * inv_w;
                    uvw_out[o + 2] = acc_u.z * inv_w;
                    s_valid[cellIndex(i, j, k)] = 1u;
                }
            }
        }
    }

    // ── 3. Extrapolation outward ─────────────────────────────────────────────
    // Flood the coordinate one voxel per sweep into unsupported cells, averaging
    // whichever face neighbours are already valid.
    //
    // ★ Sweep count is set by the CONSUMER's reach, not by taste: the shader
    // samples at the ISO crossing and trilinearly touches the 8 cells around it,
    // so the coordinate must be valid up to ~2 voxels past the last supported
    // cell. 3 sweeps covers that with a voxel to spare. Fewer, and the outermost
    // corner of the filter stencil is still zero — which does not blank the
    // texture, it drags it toward the origin by a fraction, i.e. it appears as a
    // subtle coordinate warp hugging the silhouette. That is the failure mode
    // nobody reports as a bug, so it is worth the extra sweep.
    //
    // ★★★ A PLAIN AVERAGE IS CORRECT HERE, AND ONLY BECAUSE THE FIELD IS A
    // RESIDUAL. When this grid held absolute coordinates, averaging the
    // neighbours FLATTENED the field as it extended — moving one voxel in world
    // must move one voxel in coordinate, and a mean destroys exactly that
    // gradient. It showed up as "a coarse vortex wrapped around every droplet":
    // a pool's surface sits in SUPPORTED cells so it looked right, while a
    // droplet spanning two or three cells crosses the ISO mostly in
    // EXTRAPOLATED ones, where the coordinate went nearly constant and the
    // texture blew up. The fix then was to carry the world offset on every tap.
    //
    // Storing the residual makes that offset arithmetic not just unnecessary but
    // WRONG: the identity part now lives in the shader (worldPos + d), so the
    // gradient is exact by construction, and re-adding ±voxel here would tilt
    // the displacement field by one voxel per sweep. Smoothness of d is all this
    // loop owes, and a mean is the right tool for that.
    //
    // ★ If you ever revert to absolute storage, the offsets must come back with
    // it. They are two halves of one decision, not an optimisation.
    constexpr int kExtrapolationSweeps = 3;
    static std::vector<uint8_t> s_next_valid;
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    for (int sweep = 0; sweep < kExtrapolationSweeps; ++sweep) {
        s_next_valid = s_valid;
        // int + bitwise-or rather than bool + `||`: the reduction operator set
        // that is portable across the OpenMP versions this project is built
        // with is the C one, and a bool reduction is the kind of thing that
        // compiles on one toolchain and not the next.
        int any_filled = 0;
#ifdef _OPENMP
        #pragma omp parallel for collapse(3) schedule(static) num_threads(thread_cap) \
            reduction(|:any_filled)
#endif
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const std::size_t c = cellIndex(i, j, k);
                    if (s_valid[c]) continue;
                    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
                    int n = 0;
                    // Reads s_valid (this sweep's INPUT) and writes s_next_valid,
                    // so a cell filled during this sweep cannot itself become a
                    // donor until the next one. Without that separation the
                    // result would depend on iteration order and the fill would
                    // race ahead along +x/+y/+z, dragging one cell's coordinate
                    // across the whole band in a single pass.
                    auto tap = [&](std::size_t nc) {
                        if (!s_valid[nc]) return;
                        sx += uvw_out[nc * 3u + 0];
                        sy += uvw_out[nc * 3u + 1];
                        sz += uvw_out[nc * 3u + 2];
                        ++n;
                    };
                    if (i > 0)      tap(c - 1);
                    if (i + 1 < nx) tap(c + 1);
                    if (j > 0)      tap(c - nx);
                    if (j + 1 < ny) tap(c + nx);
                    if (k > 0)      tap(c - plane);
                    if (k + 1 < nz) tap(c + plane);
                    if (n == 0) continue;
                    const float inv_n = 1.0f / static_cast<float>(n);
                    uvw_out[c * 3u + 0] = sx * inv_n;
                    uvw_out[c * 3u + 1] = sy * inv_n;
                    uvw_out[c * 3u + 2] = sz * inv_n;
                    s_next_valid[c] = 1u;
                    any_filled = 1;
                }
            }
        }
        s_valid.swap(s_next_valid);
        if (!any_filled) break;
    }

    // ── 4. Smooth the displacement ───────────────────────────────────────────
    // The per-particle difference above already cancels the sampling bias, so
    // what is left here is the SPREAD of displacement among the particles a cell
    // gathered — real signal in a shear zone, but at the band edge it is mostly
    // the seam between a cell with dozens of donors and one with two, plus the
    // step where gathered cells meet extrapolated ones.
    //
    // ★★★ THIS IS WHAT THE RESIDUAL BUYS. Smoothing d cannot damage the
    // coordinate's scale, because the identity part is not in this buffer at
    // all. Under absolute storage no filter was safe here: anything strong
    // enough to touch the noise also attacked the gradient carrying the
    // coordinate's world scale, which is why the field simply had to live with
    // whatever the gather produced.
    //
    // Two Jacobi passes, half centre / half neighbour mean. Deliberately weak:
    // real deformation is low-frequency and passes through untouched, while the
    // seams above sit at exactly the cell frequency this attenuates hardest.
    // More passes would start rounding off the sharp shear where two streams
    // meet, and that shear is signal.
    //
    // ★ Only valid cells donate and only valid cells are written, so the band
    // built above keeps its outward extension instead of being pulled back
    // toward zero by the untouched cells beyond it.
    {
        constexpr int kSmoothingPasses = 2;
        static std::vector<float> s_smooth_scratch;
        for (int pass = 0; pass < kSmoothingPasses; ++pass) {
            s_smooth_scratch = uvw_out;
#ifdef _OPENMP
            #pragma omp parallel for collapse(3) schedule(static) num_threads(thread_cap)
#endif
            for (int k = 0; k < nz; ++k) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        const std::size_t c = cellIndex(i, j, k);
                        if (!s_valid[c]) continue;
                        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
                        int n = 0;
                        auto tap = [&](std::size_t nc) {
                            if (!s_valid[nc]) return;
                            sx += s_smooth_scratch[nc * 3u + 0];
                            sy += s_smooth_scratch[nc * 3u + 1];
                            sz += s_smooth_scratch[nc * 3u + 2];
                            ++n;
                        };
                        if (i > 0)      tap(c - 1);
                        if (i + 1 < nx) tap(c + 1);
                        if (j > 0)      tap(c - nx);
                        if (j + 1 < ny) tap(c + nx);
                        if (k > 0)      tap(c - plane);
                        if (k + 1 < nz) tap(c + plane);
                        if (n == 0) continue;
                        const float inv_n = 0.5f / static_cast<float>(n);
                        uvw_out[c * 3u + 0] = 0.5f * s_smooth_scratch[c * 3u + 0] + sx * inv_n;
                        uvw_out[c * 3u + 1] = 0.5f * s_smooth_scratch[c * 3u + 1] + sy * inv_n;
                        uvw_out[c * 3u + 2] = 0.5f * s_smooth_scratch[c * 3u + 2] + sz * inv_n;
                    }
                }
            }
        }
    }

    return true;
}

bool buildCompositionGrid(const FluidParticles& particles,
                          const FluidSim::FluidGrid& grid,
                          const LevelSetParams& params,
                          const SubstanceMaterialEntry* substance_materials,
                          std::size_t substance_material_count,
                          int fallback_material,
                          std::vector<float>& composition_out,
                          const std::vector<uint32_t>* excluded_substance_tags)
{
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
        if (excluded_substance_tags && p < particles.substance_tag.size() &&
            std::find(excluded_substance_tags->begin(), excluded_substance_tags->end(),
                      particles.substance_tag[p]) != excluded_substance_tags->end()) continue;
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
                              std::vector<Vec3>& cell_velocity_out)
{
    cells_out.clear();
    cell_velocity_out.clear();

    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const float voxel = grid.voxel_size;
    const std::size_t cell_count = static_cast<std::size_t>(nx) *
                                   static_cast<std::size_t>(ny) *
                                   static_cast<std::size_t>(nz);
    const std::size_t particle_count = particles.size();
    if (cell_count == 0 || particle_count == 0 || voxel <= 0.0f ||
        solid_tags == nullptr || solid_tag_count == 0 ||
        particles.substance_tag.size() < particle_count) {
        return false;
    }

    auto isSolidTag = [&](uint32_t tag) -> bool {
        // ★ Untagged liquid is never solid. A domain fill, a scripted spawn and
        // everything authored before substances existed all carry tag 0, and
        // freezing them because a table row happens to exist would turn a
        // chocolate binding into a domain-wide block of ice.
        if (tag == kSubstanceUntagged) return false;
        for (std::size_t i = 0; i < solid_tag_count; ++i)
            if (solid_tags[i] == tag) return true;
        return false;
    };

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
        const bool solid_parcel = isSolidTag(particles.substance_tag[p]);
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
