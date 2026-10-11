#include "Fluid/FluidLevelSet.h"
#include "Fluid/SurfaceWorkTiles.h"
#include "Fluid/SurfaceParticleBins.h"
#include "PerfProfile.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace RayTrophiSim::Fluid {
namespace {
float lengthSq(const Vec3& v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}
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
                                 const FluidViewSelection* selection)
{
    RTPERF_FRAME_SCOPE("sim.render.surface.material_coordinates");
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
    if (!has_gen_b) {
        w_a = 1.0f;
        w_b = 0.0f;
    }

    const float kernel_R    = std::max(1e-4f, params.kernel_radius_voxels * voxel);
    const float kernel_R_sq = kernel_R * kernel_R;

    int thread_cap = params.threads;
#ifdef _OPENMP
    if (thread_cap <= 0) thread_cap = omp_get_max_threads();
    thread_cap = std::max(1, thread_cap);
#else
    (void)thread_cap;
#endif

    constexpr std::size_t kInvalid = static_cast<std::size_t>(-1);
    std::vector<std::size_t> eligible(particle_count, kInvalid);
    const float inv_h = 1.0f / voxel;
    // The coordinate consumer must cover the filtered/anisotropic surface and
    // its interpolation stencil. Three sweeps remain the minimum legacy halo.
    const float shift = std::clamp(params.position_smoothing, 0.0f, 1.0f) *
        std::max(1.0e-4f, params.anisotropy_radius_voxels * voxel);
    const float stretch = params.anisotropy_enabled
        ? std::cbrt(std::max(1.0f, params.anisotropy_max_stretch) *
                    std::max(1.0f, params.anisotropy_max_stretch)) : 1.0f;
    const int extension_sweeps = std::max(3, static_cast<int>(std::ceil(
        (kernel_R * (stretch - 1.0f) + shift) / voxel)) + 2);
    SurfaceWorkTiles work_tiles(nx, ny, nz, grid.origin, voxel);
    for (std::size_t p = 0; p < particle_count; ++p) {
        if ((p < particles.mass_fraction.size() &&
             !(particles.mass_fraction[p] > 0.02f)) ||
            (selection && !selection->keeps(particles, p))) {
            continue;
        }
        const Vec3& wp = particles.position[p];
        const Vec3 local = (wp - grid.origin) * inv_h;
        if (!std::isfinite(local.x) || !std::isfinite(local.y) ||
            !std::isfinite(local.z) || local.x < 0 || local.x >= nx ||
            local.y < 0 || local.y >= ny || local.z < 0 || local.z >= nz) {
            continue;
        }
        eligible[p] = 0;
        // Smoothing only writes valid cells, so it does not expand this band.
        work_tiles.addSupport(wp, kernel_R + extension_sweeps * voxel);
    }
    work_tiles.finish();
    const int64_t work_count = static_cast<int64_t>(work_tiles.workSize());
    SurfaceParticleBins bins(grid.origin, nx, ny, nz, voxel, kernel_R);
    bins.build(particles.position, eligible);
    if (bins.particles.empty()) {
        uvw_out.clear();
        return false;
    }
    const int bin_reach = std::max(1, static_cast<int>(
        std::ceil(kernel_R * bins.inverse_width)));

    // ── 2. Weighted gather ───────────────────────────────────────────────────
    // valid[c] marks cells that received real support; everything else is filled
    // by the extrapolation below. A separate mask rather than testing the value
    // against zero: (0,0,0) is a perfectly legal material coordinate for liquid
    // born at the world origin, and using the value as its own validity flag
    // would make that liquid's cells re-fill from their neighbours every frame.
    static std::vector<uint8_t> s_valid;
    s_valid.assign(cell_count, 0u);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 256) num_threads(thread_cap)
#endif
    for (int64_t lane = 0; lane < work_count; ++lane) {
        int i, j, k;
        if (!work_tiles.cellAt(static_cast<std::size_t>(lane), i, j, k)) {
            continue;
        }
        const Vec3 p_c = grid.origin + Vec3(
            (static_cast<float>(i) + 0.5f) * voxel,
            (static_cast<float>(j) + 0.5f) * voxel,
            (static_cast<float>(k) + 0.5f) * voxel);

        int bi, bj, bk;
        bins.coordinate(p_c, bi, bj, bk);
        const int i0 = std::max(0, bi - bin_reach), i1 = std::min(bins.nx - 1, bi + bin_reach);
        const int j0 = std::max(0, bj - bin_reach), j1 = std::min(bins.ny - 1, bj + bin_reach);
        const int k0 = std::max(0, bk - bin_reach), k1 = std::min(bins.nz - 1, bk + bin_reach);

        float acc_w = 0.0f;
        Vec3  acc_u(0.0f, 0.0f, 0.0f);
        for (int kk = k0; kk <= k1; ++kk) {
            for (int jj = j0; jj <= j1; ++jj) {
                for (int ii = i0; ii <= i1; ++ii) {
                    const std::size_t nci = bins.index(ii, jj, kk);
                    for (std::size_t a = bins.offsets[nci]; a < bins.offsets[nci + 1]; ++a) {
                        const std::size_t pa = bins.particles[a];
                        const Vec3 d = p_c - particles.position[pa];
                        const float d2 = lengthSq(d);
                        if (d2 >= kernel_R_sq) {
                            continue;
                        }
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
    const int kExtrapolationSweeps = extension_sweeps;
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
        #pragma omp parallel for schedule(static) num_threads(thread_cap) reduction(|:any_filled)
#endif
        for (int64_t lane = 0; lane < work_count; ++lane) {
            int i, j, k;
            if (!work_tiles.cellAt(static_cast<std::size_t>(lane), i, j, k)) {
                continue;
            }
            const std::size_t c = cellIndex(i, j, k);
            if (s_valid[c]) {
                continue;
            }
            float sx = 0.0f, sy = 0.0f, sz = 0.0f;
            int n = 0;
            // Reads s_valid (this sweep's INPUT) and writes s_next_valid,
            // so a cell filled during this sweep cannot itself become a
            // donor until the next one. Without that separation the
            // result would depend on iteration order and the fill would
            // race ahead along +x/+y/+z, dragging one cell's coordinate
            // across the whole band in a single pass.
            auto tap = [&](std::size_t nc) {
                if (!s_valid[nc]) {
                    return;
                }
                sx += uvw_out[nc * 3u + 0];
                sy += uvw_out[nc * 3u + 1];
                sz += uvw_out[nc * 3u + 2];
                ++n;
            };
            if (i > 0)      tap(c - 1);
            if (i + 1 < nx) {
                tap(c + 1);
            }
            if (j > 0)      tap(c - nx);
            if (j + 1 < ny) {
                tap(c + nx);
            }
            if (k > 0)      tap(c - plane);
            if (k + 1 < nz) {
                tap(c + plane);
            }
            if (n == 0) {
                continue;
            }
            const float inv_n = 1.0f / static_cast<float>(n);
            uvw_out[c * 3u + 0] = sx * inv_n;
            uvw_out[c * 3u + 1] = sy * inv_n;
            uvw_out[c * 3u + 2] = sz * inv_n;
            s_next_valid[c] = 1u;
            any_filled = 1;
        }
        s_valid.swap(s_next_valid);
        if (!any_filled) break;
    }

    smoothSurfaceCoordinateTiles(uvw_out, s_valid, work_tiles, nx, ny, nz, 2, thread_cap);
    return true;
}

}
