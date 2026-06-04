/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FluidLevelSet.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Narrow-band signed distance field reconstruction from APIC particles.
 *
 * Liquid simulations advect particles, not a surface. To render the result
 * as a refractive water surface the bridge needs an implicit surface
 * defined on the same MAC grid the solver uses. This file builds that
 * surface with the Zhu-Bridson "blobby" SDF:
 *
 *   For each grid cell c at world position p_c:
 *     w_i      = kernel(|p_c - p_i|, R)              over particles within R
 *     x_bar(c) = sum_i w_i p_i / sum_i w_i
 *     phi(c)   = |p_c - x_bar(c)| - r                              if has hits
 *     phi(c)   = +narrow_band_extent                               otherwise
 *
 * Cells outside the narrow band carry a sentinel positive distance so the
 * downstream isosurface walk can early-out. The grid layout matches
 * FluidGrid (origin/voxel_size/nx*ny*nz) so the SDF can ride the same
 * NanoVDB upload path as the density volume.
 *
 * Neighbour search uses a CSR cell-bin (particle -> cell index sort) so the
 * splat phase is race-free when parallelised: each output cell pulls
 * contributions from a fixed stencil of neighbour cells. No atomics.
 */

#pragma once

#include "../FluidGrid.h"
#include "FluidParticles.h"
#include <vector>
#include <cstddef>

namespace RayTrophiSim {
namespace Fluid {

struct LevelSetParams {
    // Kernel radius expressed in voxels. 2.0 gives a smooth water surface
    // without losing thin sheets; 3.0+ over-smooths small features.
    float kernel_radius_voxels = 2.0f;

    // Particle radius (subtracted from the cell-to-cluster distance to make
    // the surface bulge through the particle samples). 0.5 = touching spheres
    // at the seed density; lower = more concave between particles.
    float particle_radius_voxels = 0.55f;

    // Distance assigned to cells with no in-range particles. Must be > 0 and
    // typically larger than kernel_radius so the isosurface walk can early-
    // out. Expressed in voxels.
    float narrow_band_voxels = 3.0f;

    // OpenMP thread cap. 0 = auto.
    int threads = 0;

    // Fast Laplacian smoothing sweeps over the reconstructed SDF.
    // Highly efficient way to eliminate staircasing/blocking without grid cost.
    int smoothing_iterations = 2;

    // Surface-grid refinement vs. the simulation grid. The SDF is the rendered
    // surface, and it does NOT have to share the sim grid's voxel size — the
    // simulation can stay cheap (coarse) while the surface is reconstructed on
    // a finer grid (sub-voxel detail for wavy/rocky coastlines). 1 = same as
    // sim grid, 2 = half-voxel, 3/4 = finer still. Kernel/particle/band radii
    // stay PHYSICAL (sim-voxel) so the surface SHAPE is invariant to this — only
    // the sampling fineness changes. Cost (SDF build + NanoVDB upload) scales
    // with multiplier^3, so keep it modest on large domains.
    int surface_resolution_multiplier = 1;

    // ── Anisotropic kernel (Yu & Turk 2013) ──────────────────────────────────
    // Replaces the isotropic Zhu-Bridson splat with per-particle ELLIPSOIDAL
    // kernels, oriented + stretched by the weighted covariance of each
    // particle's neighbours. Flat sheets stay flat, thin films/necks are
    // cleaned, and nearby droplets merge smoothly instead of leaving bumpy
    // sphere-union artefacts. This is THE knob for "metaball-clean" liquid
    // surfaces; disabling it falls back to the plain isotropic splat.
    // OFF by default: it is the most expensive part of the SDF build (per-particle
    // neighbour covariance + eigensolve) and the isotropic splat is plenty for
    // real-time preview. Opt in for final/bake-quality "metaball-clean" surfaces.
    bool  anisotropy_enabled = false;
    // Neighbourhood radius (sim voxels) for the covariance estimate. ~2-3.
    float anisotropy_radius_voxels = 2.5f;
    // Max axis stretch ratio kr — clamps the ellipsoid so a single thin sheet
    // does not blow up to an infinite plane. 4 is the Yu-Turk default.
    float anisotropy_max_stretch = 4.0f;
    // Below this neighbour count a particle is isolated spray and is kept
    // spherical (no reliable covariance from too few samples).
    int   anisotropy_neighbor_min = 6;
    // Position smoothing lambda: x~ = (1-lambda)*x + lambda*neighbour_mean.
    // Removes per-particle jitter before surfacing. 0 = raw, 1 = fully smoothed.
    float position_smoothing = 0.9f;
};

struct LevelSetStats {
    std::size_t active_cells = 0;       // cells inside the narrow band
    std::size_t surface_cells = 0;      // cells whose |phi| < voxel
    std::size_t particle_count = 0;
    std::size_t grid_cell_count = 0;    // cell count of the (refined) SDF grid
    float       build_ms = 0.0f;

    // Effective grid the SDF was actually built on. Equals the sim grid when
    // surface_resolution_multiplier == 1, otherwise refined. Consumers must use
    // these (not the sim grid) to size the density-proxy loop and to upload the
    // NanoVDB volume at the correct resolution. Origin is unchanged from the
    // sim grid, so world bounds are identical (same physical extent, finer voxels).
    int   eff_nx = 0;
    int   eff_ny = 0;
    int   eff_nz = 0;
    float eff_voxel = 0.0f;
};

// Build an SDF grid aligned with `grid` from the current particle positions.
// `sdf_out` is resized to grid cell count; outside-band cells are set to
// +(narrow_band_voxels * voxel_size). Returns true if any active cell exists.
bool buildLevelSet(const FluidParticles& particles,
                   const FluidSim::FluidGrid& grid,
                   const LevelSetParams& params,
                   std::vector<float>& sdf_out,
                   LevelSetStats* stats = nullptr);

} // namespace Fluid
} // namespace RayTrophiSim
