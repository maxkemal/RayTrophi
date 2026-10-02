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

// Separable Gaussian blur of a dense nx*ny*nz field (x fastest) into `out`.
// sigma_voxels <= 0 copies the field. The kernel is renormalised over the
// in-bounds taps, so a field touching a domain wall keeps its level there
// instead of fading into the wall. Values below a small floor are zeroed so
// the tails do not turn the sparse NanoVDB conversion into a dense one.
void spreadFogDensity(const float* src, int nx, int ny, int nz,
                      float sigma_voxels, std::vector<float>& out);

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

// Density of only the parcels `selection` keeps (resolved view == Fog, by
// substance or by state label), with the SAME
// trilinear footprint and per-particle weight (1 / particles_per_cell) as the
// solver's density splat, so a fog drawn from a subset matches, cell for cell,
// the fog the whole-domain splat would have given. Used when a domain draws
// some parcels as fog and others as a surface or splats; grid.density
// stands in only when every live key is fog. Returns false (out cleared) when
// no parcel matched.
bool splatFogDensityForSelection(const FluidParticles& particles,
                                 int nx, int ny, int nz,
                                 const Vec3& origin, float voxel_size,
                                 int particles_per_cell,
                                 const FluidViewSelection& selection,
                                 std::vector<float>& out);

} // namespace Fluid
} // namespace RayTrophiSim
