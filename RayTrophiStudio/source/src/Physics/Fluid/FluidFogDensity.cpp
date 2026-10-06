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
    const float sigma = std::min(sigma_voxels, kFogBlurMaxFogVoxels);
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

// Lattice hash -> [0, 1). Integer-only so the field is identical on every
// machine and every frame (the erosion must not flicker between re-uploads).
float latticeValue(int x, int y, int z, uint32_t seed) {
    uint32_t h = seed * 0x9E3779B9u;
    h ^= static_cast<uint32_t>(x) * 0x85EBCA6Bu;
    h = (h ^ (h >> 13)) * 0xC2B2AE35u;
    h ^= static_cast<uint32_t>(y) * 0x27D4EB2Fu;
    h = (h ^ (h >> 15)) * 0x165667B1u;
    h ^= static_cast<uint32_t>(z) * 0x9E3779B1u;
    h = (h ^ (h >> 16)) * 0x85EBCA6Bu;
    h ^= h >> 13;
    return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
}

float quintic(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }

// Value noise in [0, 1], C2-smooth across lattice cells.
float valueNoise3(float x, float y, float z, uint32_t seed) {
    const float fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
    const int ix = static_cast<int>(fx), iy = static_cast<int>(fy), iz = static_cast<int>(fz);
    const float u = quintic(x - fx), v = quintic(y - fy), w = quintic(z - fz);
    auto lerp = [](float a, float b, float t) { return a + (b - a) * t; };
    const float x00 = lerp(latticeValue(ix, iy, iz, seed),         latticeValue(ix + 1, iy, iz, seed), u);
    const float x10 = lerp(latticeValue(ix, iy + 1, iz, seed),     latticeValue(ix + 1, iy + 1, iz, seed), u);
    const float x01 = lerp(latticeValue(ix, iy, iz + 1, seed),     latticeValue(ix + 1, iy, iz + 1, seed), u);
    const float x11 = lerp(latticeValue(ix, iy + 1, iz + 1, seed), latticeValue(ix + 1, iy + 1, iz + 1, seed), u);
    return lerp(lerp(x00, x10, v), lerp(x01, x11, v), w);
}

// fBm normalised back to [0, 1]; octave k is offset so lattices do not align.
float fbm3(float x, float y, float z, int octaves, uint32_t seed) {
    float sum = 0.0f, amp = 1.0f, norm = 0.0f, freq = 1.0f;
    for (int k = 0; k < octaves; ++k) {
        const float o = 17.31f * static_cast<float>(k);
        sum += amp * valueNoise3(x * freq + o, y * freq - o, z * freq + 0.5f * o,
                                 seed + static_cast<uint32_t>(k) * 101u);
        norm += amp;
        amp *= 0.5f;
        freq *= 2.0f;
    }
    return norm > 0.0f ? sum / norm : 0.5f;
}

// Surface-band erosion (see erodeFogDensity's header note).
//
// Distance-based, not blur-based. The first version blurred min(d,1) over the
// band and cut at its 0.5 level; a body THINNER than the blur radius never
// reaches 0.5 anywhere, so a thin snow layer with a deep band was erased
// completely (reported 2026-10-05 at depth 1.0). The distance to the surface
// has no such dependency: the noise can only cut `depth` deep, whatever the
// body's thickness.
void erodeFogSurfaceBand(float* field, int nx, int ny, int nz,
                         const Vec3& origin, float voxel_size,
                         float strength, float size_world, int detail, int seed,
                         float depth_world) {
    const std::size_t sx = static_cast<std::size_t>(nx);
    const std::size_t sxy = sx * static_cast<std::size_t>(ny);
    const std::size_t cells = sxy * static_cast<std::size_t>(nz);

    // Chamfer distance (voxels, 26-neighbourhood) from every INSIDE cell
    // (d >= 0.5, the fog's own surface level) to the nearest outside cell.
    // Two raster passes; outside cells hold 0. The domain wall counts as
    // inside, so a body touching it is not eroded from the wall.
    constexpr float kInf = 1e9f;
    std::vector<float> dist(cells);
    for (std::size_t c = 0; c < cells; ++c) dist[c] = field[c] >= 0.5f ? kInf : 0.0f;
    const float w1 = 1.0f, w2 = 1.41421356f, w3 = 1.73205081f;
    auto relax = [&](int i, int j, int k, int dir) {
        const std::size_t c = static_cast<std::size_t>(k) * sxy +
                              static_cast<std::size_t>(j) * sx + static_cast<std::size_t>(i);
        float best = dist[c];
        if (best == 0.0f) return;
        for (int dz = -1; dz <= 1; ++dz)
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    // Raster order: the forward pass reads the neighbours
                    // already visited (lexicographically before), the backward
                    // pass the ones after.
                    const int lex = dz * 9 + dy * 3 + dx;
                    if (lex == 0 || (dir > 0 ? lex > 0 : lex < 0)) continue;
                    const int ii = i + dx, jj = j + dy, kk = k + dz;
                    if (ii < 0 || jj < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz) continue;
                    const int axes = (dx != 0) + (dy != 0) + (dz != 0);
                    const float w = axes == 1 ? w1 : (axes == 2 ? w2 : w3);
                    const float v = dist[static_cast<std::size_t>(kk) * sxy +
                                         static_cast<std::size_t>(jj) * sx +
                                         static_cast<std::size_t>(ii)] + w;
                    if (v < best) best = v;
                }
        dist[c] = best;
    };
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) relax(i, j, k, +1);
    for (int k = nz - 1; k >= 0; --k)
        for (int j = ny - 1; j >= 0; --j)
            for (int i = nx - 1; i >= 0; --i) relax(i, j, k, -1);

    // Signed surface coordinate s (voxels): inside cells sit at distance-0.5
    // (the first inside layer is +0.5), fringe cells at d-0.5 in (-0.5, 0).
    // The noise pushes the surface in by up to `strength * depth`:
    //     d' = d * smoothstep(-0.5, 0.5, s - D * n),   D = strength*depth/voxel
    const float D = std::min(strength, 1.0f) * depth_world / voxel_size;
    const float inv_size = 1.0f / std::max(size_world, kFogErosionMinSize);
    const int octaves = std::clamp(detail, 1, kFogErosionMaxDetail);
    const uint32_t useed = static_cast<uint32_t>(seed);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int k = 0; k < nz; ++k) {
        const float wz = (origin.z + (static_cast<float>(k) + 0.5f) * voxel_size) * inv_size;
        for (int j = 0; j < ny; ++j) {
            const float wy = (origin.y + (static_cast<float>(j) + 0.5f) * voxel_size) * inv_size;
            const std::size_t row = static_cast<std::size_t>(k) * sxy +
                                    static_cast<std::size_t>(j) * sx;
            for (int i = 0; i < nx; ++i) {
                float& d = field[row + static_cast<std::size_t>(i)];
                if (!(d > 0.0f)) continue;
                const float di = dist[row + static_cast<std::size_t>(i)];
                const float s = di > 0.0f ? di - 0.5f : d - 0.5f;
                if (s - D >= 0.5f) continue;   // deeper than any cut can reach
                const float wx = (origin.x + (static_cast<float>(i) + 0.5f) * voxel_size) * inv_size;
                const float x = s - D * fbm3(wx, wy, wz, octaves, useed);
                const float t = std::clamp(x + 0.5f, 0.0f, 1.0f);
                d *= t * t * (3.0f - 2.0f * t);
                if (d < kFogDensityFloor) d = 0.0f;
            }
        }
    }
}

} // namespace

void erodeFogDensity(float* field, int nx, int ny, int nz,
                     const Vec3& origin, float voxel_size,
                     float strength, float size_world, int detail, int seed,
                     float depth_world) {
    if (!field || nx <= 0 || ny <= 0 || nz <= 0 || !(voxel_size > 0.0f)) return;
    if (!(strength > 0.0f)) return;
    if (depth_world > 0.0f) {
        erodeFogSurfaceBand(field, nx, ny, nz, origin, voxel_size, strength,
                            size_world, detail, seed, depth_world);
        return;
    }
    const float s = std::min(strength, 1.0f);
    const float inv_size = 1.0f / std::max(size_world, kFogErosionMinSize);
    const int octaves = std::clamp(detail, 1, kFogErosionMaxDetail);
    const uint32_t useed = static_cast<uint32_t>(seed);
    const std::size_t sx = static_cast<std::size_t>(nx);
    const std::size_t sxy = sx * static_cast<std::size_t>(ny);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int k = 0; k < nz; ++k) {
        const float wz = (origin.z + (static_cast<float>(k) + 0.5f) * voxel_size) * inv_size;
        for (int j = 0; j < ny; ++j) {
            const float wy = (origin.y + (static_cast<float>(j) + 0.5f) * voxel_size) * inv_size;
            float* row = field + static_cast<std::size_t>(k) * sxy + static_cast<std::size_t>(j) * sx;
            for (int i = 0; i < nx; ++i) {
                const float d = row[i];
                // Empty cells cost nothing; full cells are never eroded.
                if (!(d > 0.0f) || d >= 1.0f) continue;
                const float wx = (origin.x + (static_cast<float>(i) + 0.5f) * voxel_size) * inv_size;
                const float e = std::min(s * fbm3(wx, wy, wz, octaves, useed), 0.999f);
                const float eroded = (d - e) / (1.0f - e);
                row[i] = eroded < kFogDensityFloor ? 0.0f : eroded;
            }
        }
    }
}

namespace {
bool selected(const FluidParticles& particles, std::size_t p,
              const FluidViewSelection* selection) {
    return !selection || selection->keeps(particles, p);
}
} // namespace

bool splatFogDensityWeighted(const FluidParticles& particles,
                             int nx, int ny, int nz,
                             const Vec3& origin, float voxel_size,
                             float parcel_density,
                             const FluidViewSelection* selection,
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
    const float particle_density = (std::max)(0.0f, parcel_density);
    bool any = false;
    for (std::size_t p = 0; p < n; ++p) {
        if (!selected(particles, p, selection)) continue;
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
