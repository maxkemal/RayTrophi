/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FluidFogDensity.cpp
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 */

#include "Fluid/FluidFogDensity.h"
#include "Fluid/FluidParticles.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace RayTrophiSim {
namespace Fluid {

namespace {

// Tails below this are dropped after the last pass. A full cell splats to
// ~1.0, so this is four orders of magnitude under a filled voxel.
constexpr float kFogDensityFloor = 1e-4f;

// One 1D pass along an axis. `stride` steps between neighbours on that axis,
// `len` is the axis length, and `lines` enumerates every line start.
void blurAxis(const std::vector<float>& in, std::vector<float>& out,
              const std::vector<float>& kernel, int radius,
              int nx, int ny, int nz, int axis) {
    const std::size_t sx = 1;
    const std::size_t sy = static_cast<std::size_t>(nx);
    const std::size_t sz = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int len = axis == 0 ? nx : (axis == 1 ? ny : nz);
    const std::size_t stride = axis == 0 ? sx : (axis == 1 ? sy : sz);
    // Lines are indexed by the two other axes.
    const int a_len = axis == 0 ? ny : nx;
    const int b_len = axis == 2 ? ny : nz;
    const std::size_t a_stride = axis == 0 ? sy : sx;
    const std::size_t b_stride = axis == 2 ? sy : sz;
    const int line_count = a_len * b_len;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int line = 0; line < line_count; ++line) {
        const int a = line % a_len;
        const int b = line / a_len;
        const std::size_t base = static_cast<std::size_t>(a) * a_stride +
                                 static_cast<std::size_t>(b) * b_stride;
        for (int i = 0; i < len; ++i) {
            const int lo = std::max(0, i - radius);
            const int hi = std::min(len - 1, i + radius);
            float sum = 0.0f;
            float weight = 0.0f;
            for (int j = lo; j <= hi; ++j) {
                const float w = kernel[static_cast<std::size_t>(j - i + radius)];
                sum += w * in[base + static_cast<std::size_t>(j) * stride];
                weight += w;
            }
            out[base + static_cast<std::size_t>(i) * stride] =
                weight > 0.0f ? sum / weight : 0.0f;
        }
    }
}

// Separable Gaussian in place; no floor, so ratios of two blurred fields
// stay exact. sigma <= 0 leaves the field alone.
void gaussianBlur3D(std::vector<float>& field, int nx, int ny, int nz,
                    float sigma_voxels) {
    if (!(sigma_voxels > 0.0f)) return;
    const float sigma = std::min(sigma_voxels, kFogSpreadMaxVoxels);
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> kernel(static_cast<std::size_t>(2 * radius + 1));
    const float inv_two_sigma_sq = 1.0f / (2.0f * sigma * sigma);
    for (int k = -radius; k <= radius; ++k)
        kernel[static_cast<std::size_t>(k + radius)] =
            std::exp(-static_cast<float>(k * k) * inv_two_sigma_sq);

    std::vector<float> scratch(field.size());
    blurAxis(field, scratch, kernel, radius, nx, ny, nz, 0);
    blurAxis(scratch, field, kernel, radius, nx, ny, nz, 1);
    blurAxis(field, scratch, kernel, radius, nx, ny, nz, 2);
    field.swap(scratch);
}

} // namespace

void spreadFogDensity(const float* src, int nx, int ny, int nz,
                      float sigma_voxels, std::vector<float>& out) {
    if (!src || nx <= 0 || ny <= 0 || nz <= 0) {
        out.clear();
        return;
    }
    const std::size_t cells = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz);
    out.assign(src, src + cells);
    if (!(sigma_voxels > 0.0f)) return;
    gaussianBlur3D(out, nx, ny, nz, sigma_voxels);
    for (float& v : out)
        if (v < kFogDensityFloor) v = 0.0f;
}

namespace {
bool selected(const FluidParticles& particles, std::size_t p,
              const FluidViewSelection* selection) {
    return !selection || selection->keeps(particles, p);
}
} // namespace

bool splatFogDensityForSelection(const FluidParticles& particles,
                                 int nx, int ny, int nz,
                                 const Vec3& origin, float voxel_size,
                                 int particles_per_cell,
                                 const FluidViewSelection& selection,
                                 std::vector<float>& out) {
    const std::size_t n = particles.position.size();
    if (nx <= 0 || ny <= 0 || nz <= 0 || voxel_size <= 0.0f || n == 0) {
        out.clear();
        return false;
    }
    const std::size_t cells = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz);
    out.assign(cells, 0.0f);
    const float inv_h = 1.0f / voxel_size;
    const float particle_density = 1.0f / static_cast<float>((std::max)(1, particles_per_cell));
    bool any = false;
    for (std::size_t p = 0; p < n; ++p) {
        if (!selected(particles, p, &selection)) continue;
        const float mass = p < particles.mass_fraction.size()
            ? std::clamp(particles.mass_fraction[p], 0.0f, 1.0f)
            : 1.0f;
        if (!(mass > 0.0f)) continue;
        const Vec3& pos = particles.position[p];
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z)) continue;
        const Vec3 local = (pos - origin) * inv_h - Vec3(0.5f, 0.5f, 0.5f);
        const int i0 = static_cast<int>(std::floor(local.x));
        const int j0 = static_cast<int>(std::floor(local.y));
        const int k0 = static_cast<int>(std::floor(local.z));
        const float fx = local.x - static_cast<float>(i0);
        const float fy = local.y - static_cast<float>(j0);
        const float fz = local.z - static_cast<float>(k0);
        for (int dz = 0; dz <= 1; ++dz) {
            const int k = k0 + dz;
            if (k < 0 || k >= nz) continue;
            const float wz = dz ? fz : (1.0f - fz);
            for (int dy = 0; dy <= 1; ++dy) {
                const int j = j0 + dy;
                if (j < 0 || j >= ny) continue;
                const float wy = dy ? fy : (1.0f - fy);
                for (int dx = 0; dx <= 1; ++dx) {
                    const int i = i0 + dx;
                    if (i < 0 || i >= nx) continue;
                    const float w = (dx ? fx : (1.0f - fx)) * wy * wz;
                    const std::size_t c = static_cast<std::size_t>(i) +
                        static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
                        static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                            static_cast<std::size_t>(ny);
                    out[c] += particle_density * mass * w;
                    any = true;
                }
            }
        }
    }
    if (!any) out.clear();
    return any;
}

bool splatFogTemperatureKelvin(const FluidParticles& particles,
                               int nx, int ny, int nz,
                               const Vec3& origin, float voxel_size,
                               float sigma_voxels, std::vector<float>& out,
                               const FluidViewSelection* selection) {
    const std::size_t n = particles.position.size();
    if (nx <= 0 || ny <= 0 || nz <= 0 || voxel_size <= 0.0f || n == 0 ||
        particles.temperature.size() < n) {
        out.clear();
        return false;
    }
    const std::size_t cells = static_cast<std::size_t>(nx) *
                              static_cast<std::size_t>(ny) *
                              static_cast<std::size_t>(nz);
    std::vector<float> weighted_t(cells, 0.0f);
    std::vector<float> weight(cells, 0.0f);
    const float inv_h = 1.0f / voxel_size;
    // Same trilinear footprint as splatFluidDensityCPU / sim_fluid_density_splat,
    // so the temperature lands exactly where the density does.
    for (std::size_t p = 0; p < n; ++p) {
        if (!selected(particles, p, selection)) continue;
        const Vec3& pos = particles.position[p];
        const float t = particles.temperature[p];
        // 0 K is an unwritten parcel (see FluidParticles::temperature), not a
        // cold one; letting it vote would drag hot neighbours toward zero.
        if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
            !std::isfinite(t) || t <= 0.0f)
            continue;
        const Vec3 local = (pos - origin) * inv_h - Vec3(0.5f, 0.5f, 0.5f);
        const int i0 = static_cast<int>(std::floor(local.x));
        const int j0 = static_cast<int>(std::floor(local.y));
        const int k0 = static_cast<int>(std::floor(local.z));
        const float fx = local.x - static_cast<float>(i0);
        const float fy = local.y - static_cast<float>(j0);
        const float fz = local.z - static_cast<float>(k0);
        for (int dz = 0; dz <= 1; ++dz) {
            const int k = k0 + dz;
            if (k < 0 || k >= nz) continue;
            const float wz = dz ? fz : (1.0f - fz);
            for (int dy = 0; dy <= 1; ++dy) {
                const int j = j0 + dy;
                if (j < 0 || j >= ny) continue;
                const float wy = dy ? fy : (1.0f - fy);
                for (int dx = 0; dx <= 1; ++dx) {
                    const int i = i0 + dx;
                    if (i < 0 || i >= nx) continue;
                    const float w = (dx ? fx : (1.0f - fx)) * wy * wz;
                    const std::size_t c = static_cast<std::size_t>(i) +
                        static_cast<std::size_t>(j) * static_cast<std::size_t>(nx) +
                        static_cast<std::size_t>(k) * static_cast<std::size_t>(nx) *
                            static_cast<std::size_t>(ny);
                    weighted_t[c] += w * t;
                    weight[c] += w;
                }
            }
        }
    }
    gaussianBlur3D(weighted_t, nx, ny, nz, sigma_voxels);
    gaussianBlur3D(weight, nx, ny, nz, sigma_voxels);
    out.assign(cells, 0.0f);
    bool any = false;
    for (std::size_t c = 0; c < cells; ++c) {
        if (weight[c] > kFogDensityFloor) {
            out[c] = weighted_t[c] / weight[c];
            any = true;
        }
    }
    if (!any) out.clear();
    return any;
}

} // namespace Fluid
} // namespace RayTrophiSim
