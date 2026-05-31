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
                   LevelSetStats* stats)
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
                                    acc_w += w;
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
                                    acc_w += w;
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
                    const float phi = dlen - particle_r;
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

} // namespace Fluid
} // namespace RayTrophiSim
