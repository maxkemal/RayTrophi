/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FluidFogDensity.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Render-side shaping of a liquid's splatted density for the VolumeFog mode.
 * The solver's grid.density is a trilinear splat: each particle reaches only
 * the 8 surrounding cells, so sparse spray reads as isolated dots. The fog
 * route draws a Gaussian-spread copy instead; the solver grid is untouched.
 */

#pragma once

#include "../Vec3.h"
#include "FluidViewResolver.h"
#include <cstdint>
#include <memory>
#include <vector>

class VolumeShader;

namespace RayTrophiSim {
namespace Fluid {

// The fog look a liquid starts from (desc.fluid_fog_shader when unset). One
// maker for the fog sync and fluid.set_fog_shader, so a scripted edit on a
// domain that has not drawn yet starts from the same recipe the panel shows.
std::shared_ptr<VolumeShader> makeLiquidFogShader();

// Upper bound of the spread control (Gaussian sigma, in simulation voxels).
constexpr float kFogSpreadMaxVoxels = 6.0f;
// Fog grid resolution relative to the solver grid (fluid.set_fog
// resolution_multiplier). The blur cap below is in FOG voxels, so it scales
// with it: spread is authored in solver voxels.
constexpr int   kFogMaxResolutionMultiplier = 4;
constexpr float kFogBlurMaxFogVoxels =
    kFogSpreadMaxVoxels * static_cast<float>(kFogMaxResolutionMultiplier);

// Separable Gaussian blur of a dense nx*ny*nz field (x fastest) into `out`.
// sigma_voxels <= 0 copies the field. The kernel is renormalised over the
// in-bounds taps, so a field touching a domain wall keeps its level there
// instead of fading into the wall. Values below a small floor are zeroed so
// the tails do not turn the sparse NanoVDB conversion into a dense one.
void spreadFogDensity(const float* src, int nx, int ny, int nz,
                      float sigma_voxels, std::vector<float>& out);

// Fog erosion ranges (fluid.set_fog and the panel share them).
constexpr float kFogErosionMinSize = 0.001f;
constexpr float kFogErosionMaxSize = 100.0f;
constexpr int   kFogErosionMaxDetail = 6;
constexpr float kFogErosionMaxDepth = 1.0f;   // world units

// Cloud-style erosion of a dense fog field (x fastest), in place. A world-space
// fBm n in [0,1] (largest feature `size_world`, `detail` octaves) is sampled at
// each cell centre and the density remapped as
//     d' = max(0, (d - e) / (1 - e)),   e = strength * n,   for d < 1
// so thin cells are cut into clumps while a cell at full rest packing (1.0,
// the splat's unit) and anything denser is left alone: the edges break up,
// the body does not get holes. strength <= 0 is a no-op. Cells that drop
// below the fog floor are zeroed, like spreadFogDensity's tails.
//
// depth_world > 0 switches to SURFACE-BAND erosion for dense bodies, where the
// remap above has nothing to act on (a settled snow layer is d ~ 1 throughout,
// so it stays a flat slab). A chamfer distance transform gives every cell its
// depth below the d = 0.5 surface (s, voxels; fringe cells s = d - 0.5) and
// the noise pushes that surface in:
//     d' = d * smoothstep(-0.5, 0.5, s - D * n),   D = strength * depth / voxel
// so the surface breaks into clumps at most strength*depth deep, whatever the
// body's thickness; cells deeper than that are never touched. Sparse spray
// (d < 0.5 everywhere) is mostly removed in this mode: it is for bodies,
// depth 0 is for thin fog.
void erodeFogDensity(float* field, int nx, int ny, int nz,
                     const Vec3& origin, float voxel_size,
                     float strength, float size_world, int detail, int seed,
                     float depth_world);

class FluidParticles;

// Per-cell particle temperature in KELVIN for the fog's blackbody / channel
// emission, on the same grid and with the same spread as spreadFogDensity.
// Mass-weighted: sum(w*T) and sum(w) are splatted and spread SEPARATELY, then
// divided, so a cloud's thin edge keeps its parcels' temperature instead of
// reading as cooler (which blurring T directly would do). Cells no particle
// reaches are 0 and are not uploaded. Ambient liquid IS uploaded (~293 K); the
// shader's T^4 radiance, not a cutoff, is what keeps it dark.
// Returns false (out cleared) when the particles carry no temperature.
// `selection`: when set, only the parcels it keeps vote (the fog view of a
// domain that also draws a surface or splats).
bool splatFogTemperatureKelvin(const FluidParticles& particles,
                               int nx, int ny, int nz,
                               const Vec3& origin, float voxel_size,
                               float sigma_voxels, std::vector<float>& out,
                               const FluidViewSelection* selection = nullptr);

// Fog density of the parcels `selection` keeps (nullptr = every parcel), with
// the solver density splat's trilinear footprint. `parcel_density` is what one
// whole parcel adds: 1/particles_per_cell on the solver grid, so a subset fog
// matches, cell for cell, the fog the whole-domain splat would have given;
// res^3/particles_per_cell on a fog grid res times finer, which keeps "1 = a
// cell at rest packing" on every grid. Takes a WEIGHT, not a particle count --
// renamed from splatFogDensityForSelection(int particles_per_cell) so an old
// call cannot pass a count where a weight is meant. Returns false (out
// cleared) when no parcel matched.
bool splatFogDensityWeighted(const FluidParticles& particles,
                             int nx, int ny, int nz,
                             const Vec3& origin, float voxel_size,
                             float parcel_density,
                             const FluidViewSelection* selection,
                             std::vector<float>& out);

} // namespace Fluid
} // namespace RayTrophiSim
