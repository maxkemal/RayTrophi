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
#include <vector>

namespace RayTrophiSim {
namespace Fluid {

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
bool splatFogTemperatureKelvin(const FluidParticles& particles,
                               int nx, int ny, int nz,
                               const Vec3& origin, float voxel_size,
                               float sigma_voxels, std::vector<float>& out);

} // namespace Fluid
} // namespace RayTrophiSim
