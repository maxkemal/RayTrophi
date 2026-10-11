#include "Fluid/SurfaceWorkTiles.h"
#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid {
SurfaceWorkTiles::SurfaceWorkTiles(int nx, int ny, int nz,
                                 const Vec3& origin, float voxel)
    : nx_(nx), ny_(ny), nz_(nz), tx_((nx + edge - 1) / edge),
      ty_((ny + edge - 1) / edge), origin_(origin), voxel_(voxel) {}

void SurfaceWorkTiles::addSupport(const Vec3& center, float radius) {
    if (!(voxel_ > 0.0f) || !std::isfinite(radius) || radius < 0.0f) {
        return;
    }
    const Vec3 local = (center - origin_) / voxel_;
    const float coords[3] = {local.x, local.y, local.z};
    const int dims[3] = {nx_, ny_, nz_};
    int lo[3], hi[3];
    const float reach = radius / voxel_ + 1.0f;
    for (int a = 0; a < 3; ++a) {
        if (!std::isfinite(coords[a]) || dims[a] <= 0 ||
            coords[a] + reach < 0 || coords[a] - reach >= dims[a]) {
            return;
        }
        lo[a] = static_cast<int>(std::floor(std::max(0.0f, coords[a] - reach))) / edge;
        hi[a] = static_cast<int>(std::floor(std::min(
            static_cast<float>(dims[a] - 1), coords[a] + reach))) / edge;
    }
    for (int z = lo[2]; z <= hi[2]; ++z) {
        for (int y = lo[1]; y <= hi[1]; ++y) {
            for (int x = lo[0]; x <= hi[0]; ++x) {
                pending_.insert(static_cast<uint64_t>(x) +
                    static_cast<uint64_t>(tx_) * (y + static_cast<uint64_t>(ty_) * z));
            }
        }
    }
}

void SurfaceWorkTiles::finish() {
    tiles_.assign(pending_.begin(), pending_.end());
    std::sort(tiles_.begin(), tiles_.end());
    pending_.clear();
    pending_.rehash(0);
}

std::size_t SurfaceWorkTiles::workSize() const {
    return tiles_.size() * edge * edge * edge;
}

bool SurfaceWorkTiles::cellAt(std::size_t work, int& x, int& y, int& z) const {
    constexpr std::size_t cells = edge * edge * edge;
    const uint64_t tile = tiles_[work / cells];
    const std::size_t lane = work % cells;
    x = static_cast<int>(tile % tx_) * edge + static_cast<int>(lane % edge);
    y = static_cast<int>((tile / tx_) % ty_) * edge +
        static_cast<int>((lane / edge) % edge);
    z = static_cast<int>(tile / (static_cast<uint64_t>(tx_) * ty_)) * edge +
        static_cast<int>(lane / (edge * edge));
    return x < nx_ && y < ny_ && z < nz_;
}

void smoothSurfaceTiles(std::vector<float>& sdf, const SurfaceWorkTiles& tiles,
                        int nx, int ny, int nz, float far_value,
                        int iterations, int threads,
                        const std::vector<float>* support_confidence,
                        int sample_stride, float displacement_limit) {
#ifndef _OPENMP
    (void)threads;
#endif
    if (iterations <= 0 || tiles.workSize() == 0) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    std::vector<float> scratch(tiles.workSize(), far_value);
    const int64_t work = static_cast<int64_t>(tiles.workSize());
    const int stride = std::max(1, sample_stride);
    std::vector<float> original;
    if (support_confidence) {
        original.resize(tiles.workSize(), far_value);
        for (std::size_t lane = 0; lane < tiles.workSize(); ++lane) {
            int x, y, z;
            if (tiles.cellAt(lane, x, y, z)) {
                original[lane] = sdf[x + static_cast<std::size_t>(nx) * y + plane * z];
            }
        }
    }
    for (int iteration = 0; iteration < iterations; ++iteration) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int64_t lane = 0; lane < work; ++lane) {
            int x, y, z;
            if (!tiles.cellAt(static_cast<std::size_t>(lane), x, y, z)) {
                continue;
            }
            const std::size_t c = x + static_cast<std::size_t>(nx) * y + plane * z;
            const float value = sdf[c];
            float sum = 0.0f;
            int count = 0;
            bool active = value < far_value - 1.0e-5f;
            const auto add = [&](std::size_t neighbour) {
                const float v = sdf[neighbour];
                sum += v;
                ++count;
                active = active || v < far_value - 1.0e-5f;
            };
            if (!support_confidence) {
                if (x > 0) { add(c - 1); }
                if (x + 1 < nx) { add(c + 1); }
                if (y > 0) { add(c - nx); }
                if (y + 1 < ny) { add(c + nx); }
                if (z > 0) { add(c - plane); }
                if (z + 1 < nz) { add(c + plane); }
                scratch[static_cast<std::size_t>(lane)] = active
                    ? (count > 0 ? 0.5f * value + 0.5f * (sum / count) : value)
                    : far_value;
                continue;
            }
            // Frozen reconstruction evidence, not the progressively blurred field.
            float confidence = (*support_confidence)[static_cast<std::size_t>(lane)];
            bool thin = false;
            const auto pair = [&](std::size_t minus, std::size_t plus) {
                const float a = sdf[minus];
                const float b = sdf[plus];
                if (a >= far_value - 1.0e-5f || b >= far_value - 1.0e-5f) {
                    confidence = 0.0f;
                    return;
                }
                // Opposing exterior samples flag unresolved sheets/necks.
                thin = thin || (a > 0.0f && b > 0.0f);
                sum += a + b;
                count += 2;
            };
            if (x >= stride && x + stride < nx) { pair(c - stride, c + stride); }
            if (y >= stride && y + stride < ny) { pair(c - nx * stride, c + nx * stride); }
            if (z >= stride && z + stride < nz) { pair(c - plane * stride, c + plane * stride); }
            float next = value;
            if (active && count == 6 && !thin && confidence > 0.0f) {
                next += 0.65f * confidence * (sum / count - value);
                const float anchor = original[static_cast<std::size_t>(lane)];
                next = std::clamp(next, anchor - displacement_limit,
                                  anchor + displacement_limit);
            }
            scratch[static_cast<std::size_t>(lane)] = next;
        }
        // The first parallel region's barrier completes all reads before writes.
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int64_t lane = 0; lane < work; ++lane) {
            int x, y, z;
            if (tiles.cellAt(static_cast<std::size_t>(lane), x, y, z)) {
                const std::size_t c = x + static_cast<std::size_t>(nx) * y + plane * z;
                sdf[c] = scratch[static_cast<std::size_t>(lane)];
            }
        }
    }
}

void smoothSurfaceCoordinateTiles(std::vector<float>& coordinates,
                                  const std::vector<uint8_t>& valid,
                                  const SurfaceWorkTiles& tiles, int nx, int ny, int nz,
                                  int iterations, int threads) {
#ifndef _OPENMP
    (void)threads;
#endif
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    const int64_t work = static_cast<int64_t>(tiles.workSize());
    std::vector<Vec3> scratch(tiles.workSize());
    for (int iteration = 0; iteration < iterations; ++iteration) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int64_t lane = 0; lane < work; ++lane) {
            int x, y, z;
            if (!tiles.cellAt(static_cast<std::size_t>(lane), x, y, z)) {
                continue;
            }
            const std::size_t c = x + static_cast<std::size_t>(nx) * y + plane * z;
            if (!valid[c]) {
                continue;
            }
            Vec3 sum(0, 0, 0);
            int count = 0;
            const auto add = [&](std::size_t neighbour) {
                if (valid[neighbour]) {
                    sum.x += coordinates[neighbour * 3];
                    sum.y += coordinates[neighbour * 3 + 1];
                    sum.z += coordinates[neighbour * 3 + 2];
                    ++count;
                }
            };
            if (x > 0) { add(c - 1); }
            if (x + 1 < nx) { add(c + 1); }
            if (y > 0) { add(c - nx); }
            if (y + 1 < ny) { add(c + nx); }
            if (z > 0) { add(c - plane); }
            if (z + 1 < nz) { add(c + plane); }
            const Vec3 own(coordinates[c * 3], coordinates[c * 3 + 1],
                           coordinates[c * 3 + 2]);
            if (count > 0) {
                const float factor = 0.5f / static_cast<float>(count);
                scratch[static_cast<std::size_t>(lane)] = Vec3(
                    0.5f * own.x + factor * sum.x,
                    0.5f * own.y + factor * sum.y,
                    0.5f * own.z + factor * sum.z);
            } else {
                scratch[static_cast<std::size_t>(lane)] = own;
            }
        }
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (int64_t lane = 0; lane < work; ++lane) {
            int x, y, z;
            if (!tiles.cellAt(static_cast<std::size_t>(lane), x, y, z)) {
                continue;
            }
            const std::size_t c = x + static_cast<std::size_t>(nx) * y + plane * z;
            if (valid[c]) {
                const Vec3 value = scratch[static_cast<std::size_t>(lane)];
                coordinates[c * 3] = value.x;
                coordinates[c * 3 + 1] = value.y;
                coordinates[c * 3 + 2] = value.z;
            }
        }
    }
}
}
