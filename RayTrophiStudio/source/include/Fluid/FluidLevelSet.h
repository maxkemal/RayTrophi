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

    // Geometric dilation of the reconstructed zero level set, expressed in
    // SIMULATION voxels. Unlike render/volume density this changes only the
    // liquid silhouette; it neither raises the SDF resolution nor widens the
    // particle kernel, so its build cost is effectively constant. A modest
    // positive default restores the full water body formerly (and incorrectly)
    // obtained by driving the shader's optical density.
    float surface_offset_voxels = 0.65f;

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
                   LevelSetStats* stats = nullptr,
                   const std::vector<uint32_t>* excluded_substance_tags = nullptr);

// ═══════════════════════════════════════════════════════════════════════════
// MATERIAL COORDINATE (UVW) GRID
// ═══════════════════════════════════════════════════════════════════════════
// Gathers the particles' Lagrangian uvw (see FluidParticles::uvw) into a dense
// xyz-triple grid the isosurface shader can sample, so a texture on a liquid
// FLOWS WITH the liquid instead of the body sliding through a projection nailed
// to the world.
//
// ★ Built on the SIMULATION grid, NOT the refined surface grid, and that is a
// decision rather than an oversight. uvw is a smooth, slowly-varying field — a
// mildly deformed identity map — and it is trilinearly interpolated at sample
// time, so surface-level fineness buys it nothing. Riding the refinement
// multiplier instead would multiply this buffer by up to 64x for no visible
// gain, and would silently tie "I want a crisper surface" to "my texture
// coordinates now cost 800 MB". The SDF is the field that needs the detail.
//
// ★★ Cells with no particle support are EXTRAPOLATED from their valid
// neighbours, and this is not optional. The point being shaded is the ISO
// crossing, which sits at the edge of — and often just outside — the supported
// region. Leaving unsupported cells at zero would let the trilinear filter drag
// the coordinate toward the world origin exactly at the surface, i.e. garbage
// precisely where every sample is taken and nowhere else.
//
// `uvw_out` is resized to 3 * (nx*ny*nz), interleaved xyz. Returns false (and
// clears the buffer) when nothing was supported, which callers must treat as
// "no coordinate available" and fall back to the world-anchored projection —
// never as "the coordinate is zero".
// ★★★ `excluded_substance_tags` MUST be the SAME list buildLevelSet was given.
// A substance routed to splat has no isosurface, so letting its particles vote
// here drags the coordinate of a surface they are not part of. Symmetry between
// the three gathers is the invariant: they describe one surface, so they must
// agree about which particles that surface is made of.
bool buildMaterialCoordinateGrid(const FluidParticles& particles,
                                 const FluidSim::FluidGrid& grid,
                                 const LevelSetParams& params,
                                 std::vector<float>& uvw_out,
                                 const std::vector<uint32_t>* excluded_substance_tags = nullptr);

// ═══════════════════════════════════════════════════════════════════════════
// COMPOSITION FIELD — which materials the liquid is made of, per cell.
// ═══════════════════════════════════════════════════════════════════════════
// Interleaved xyz triples at SIM-grid resolution, one per cell:
//   [0] material index A  (dominant)      -- an INDEX, encoded as a float
//   [1] material index B  (runner-up)
//   [2] weight of B in [0,1]              -- A's weight is 1 - this
//
// Both indices are 1-based ids into the material table with 0 meaning "the
// built-in dielectric"; the consumer lerps the two materials' parameters and
// shades ONCE.
//
// ★★★ ONLY THE WEIGHT MAY BE INTERPOLATED. An index is a name, not a quantity:
// the value halfway between material 2 and material 4 is material 3, which is
// some unrelated material. So the consumer takes indices from the NEAREST cell
// and interpolates only [2]. In a binary mixture — the overwhelmingly common
// case — the pair is identical everywhere the mixture exists, so nothing is
// lost; with three or more, the pair changes at cell boundaries while the
// blend fraction stays smooth.
//
// ★★ TWO SLOTS, NOT N. At any one point a mixture is dominated by two things,
// and carrying N would mean an unbounded per-cell payload for a difference no
// shading model would show. A cell with three substances keeps the two
// strongest and renormalises; the third is not silently added to one of them,
// which would tint the result by an amount nobody could trace.
//
// `substance_materials` maps substance tag -> material id, at most
// kMaxFluidSubstanceMaterials entries. `fallback_material` is used for untagged
// liquid and for any tag not in the map.
//
// Returns false (and clears) when there is nothing to describe — no particles,
// or no substance in the domain differs from the fallback. ★ That second case
// matters: a domain with one material has NO composition to publish, and
// publishing a uniform field would cost megabytes and a blend that is a no-op.
struct SubstanceMaterialEntry {
    uint32_t tag = 0u;
    int      material_id = -1;
    // 1 = fully miscible (soft, kernel-wide gradient), 0 = immiscible (a front
    // one ramp wide). The PAIR uses the minimum of its two members: refusing to
    // mix is unilateral.
    //
    // ★★★ SHARPENING HAPPENS HERE, IN THE PRODUCER, AS A GAIN — never as a
    // collapse to 0/1. A hard 0/1 weight leaves the consumer's trilinear filter
    // nothing to filter, so the boundary lands exactly on cell faces and reads
    // as axis-aligned voxel cubes of colour. The gain keeps a sub-cell ramp, so
    // an immiscible front is sharp AND smooth-edged. That distinction is the
    // whole reason the retired `fluid_blend_substance_materials` flag looked
    // wrong: its "dominant material" mode was really "voxelised material".
    float    miscibility = 1.0f;
};

bool buildCompositionGrid(const FluidParticles& particles,
                          const FluidSim::FluidGrid& grid,
                          const LevelSetParams& params,
                          const SubstanceMaterialEntry* substance_materials,
                          std::size_t substance_material_count,
                          int fallback_material,
                          std::vector<float>& composition_out,
                          // ★★★ Same list buildLevelSet was given. Without it a
                          // substance rendered as SPLAT still votes in the
                          // mixture of the SDF surface, so the surface tints
                          // toward a material that has no surface there. That
                          // reads as plausible wetness rather than as a bug.
                          const std::vector<uint32_t>* excluded_substance_tags = nullptr);

// ═══════════════════════════════════════════════════════════════════════════
// SUBSTANCE VISCOSITY FIELD — a PHYSICS field, not a surface field.
// ═══════════════════════════════════════════════════════════════════════════
// Cell-centred kinematic viscosity in m^2/s, one float per sim cell, consumed
// by the implicit viscous solve (APICFluidSolver.cpp:applyViscosity and its
// device mirror sim_fluid_viscosity_rbgs.comp).
//
// ★★★ THIS GATHER TAKES NO EXCLUSION LIST, AND THAT IS THE POINT.
// Its three neighbours above (level set, material coordinate, composition) all
// describe ONE ISOSURFACE, so they must agree about which particles that
// surface is made of and each takes `excluded_substance_tags`. This one
// describes the LIQUID. A substance rendered as splat spheres still has mass,
// still occupies cells and still resists shear; dropping it here would make the
// flow change when someone edited a RENDER setting. If a later reader "fixes
// the inconsistency" by passing the exclusion list in, that is the bug.
//
// ★★ Trilinear P2G weights, not the level-set kernel. The viscous stencil
// relaxes faces the FLUID MASK calls fluid, and that mask is built with the
// transfer's weights; gathering viscosity with a wider surface kernel would
// leave fluid cells whose viscosity came mostly from liquid that is not in
// them.
//
// `fallback_viscosity` fills untagged liquid, unbound tags, and cells no
// particle supported. Returns false (and clears) when no entry actually
// overrides the fallback — a uniform field is exactly what the scalar path
// already does, for free.
struct SubstanceViscosityEntry {
    uint32_t tag = 0u;
    float    kinematic_viscosity = -1.0f;   // < 0 = inherit the fallback
};

bool buildSubstanceViscosityField(const FluidParticles& particles,
                                  const FluidSim::FluidGrid& grid,
                                  const SubstanceViscosityEntry* entries,
                                  std::size_t entry_count,
                                  float fallback_viscosity,
                                  std::vector<float>& viscosity_out);

// ═══════════════════════════════════════════════════════════════════════════
// SOLID-PHASE CELLS — where a SOLID substance blocks the flow, per step.
// ═══════════════════════════════════════════════════════════════════════════
// Compact list of cells occupied by parcels of a solid-phase substance, plus
// the mass-averaged parcel velocity in each of them. The caller ORs this into
// the grid's solid mask, where the existing no-slip / free-slip boundary and
// the pressure projection consume it without knowing it came from particles.
//
// ★★★ NEAREST CELL, NOT A TRILINEAR SPLAT — the opposite choice from the three
// gathers above, on purpose. Those reconstruct a smooth field, where spreading
// a parcel over eight cells is what makes the surface continuous. This one
// answers a BINARY question about where matter blocks flow, and smearing it
// would inflate every chunk by a full cell in each direction: a marble would
// dam a channel it should roll down, and the error would grow with voxel size
// rather than with anything the user set.
//
// ★★★ A CELL HOLDING MORE LIQUID THAN SOLID IS NEVER SOLID, whatever the
// threshold says. Flipping an interface cell to solid under the liquid standing
// in it drops that liquid's volume from the pressure solve and then has the
// particle stage eject every parcel in it — hundreds at once, outward, on one
// frame. That is a burst at the interface, and it reads as solver instability
// rather than as a mask that moved under the liquid's feet. Requiring dominance
// costs one cell of wetting at the boundary and buys the whole failure away.
//
// ★★ THE FILL THRESHOLD IS THE WHOLE CALIBRATION. A cell is solid once the
// solid mass in it reaches `fill_threshold` parcels. Too low and one stray
// parcel blocks a full h³ of flow (an under-resolved solid is a FATTER
// obstacle than it looks, never a thinner one); too high and a thin chunk
// never blocks at all, so the phase control reads as doing nothing. It is an
// explicit argument rather than a constant here so the caller can tie it to
// the domain's seed density — the only number that says what "full" means.
//
// ★ Velocity is carried so a MOVING chunk drags the liquid: it is written into
// grid.solid_vel, which both the viscous stencil (no-slip) and the variational
// pressure coupling already read for moving colliders. Dropping it would make
// a falling chunk punch a hole through the liquid without pushing it, which
// looks like weightlessness and reads as a tuning problem.
//
// Returns false (and clears both outputs) when no solid-phase parcel reaches
// the threshold anywhere — the caller must then leave the mask untouched
// rather than stamp an empty overlay.
bool buildSubstanceSolidCells(const FluidParticles& particles,
                              const FluidSim::FluidGrid& grid,
                              const uint32_t* solid_tags,
                              std::size_t solid_tag_count,
                              float fill_threshold,
                              std::vector<uint32_t>& cells_out,
                              std::vector<Vec3>& cell_velocity_out);

} // namespace Fluid
} // namespace RayTrophiSim
