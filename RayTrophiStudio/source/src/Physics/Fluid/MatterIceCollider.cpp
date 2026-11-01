#include "Fluid/MatterIceCollider.h"
#include "Fluid/FluidParticles.h"
#include "FluidGrid.h"
#include "SurfaceMeshCache.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace RayTrophiSim::Fluid {
std::size_t appendMatterIceColliderFaces(const FluidParticles& particles,
    const FluidSim::FluidGrid& grid, std::vector<SurfaceMeshTriangle>& faces) {
    const float h = grid.voxel_size;
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    if (!(h > 0.0f) || !std::isfinite(h) || nx <= 0 || ny <= 0 || nz <= 0 ||
        particles.flags.size() != particles.size()) {
        return 0;
    }
    const auto key = [&](int i, int j, int k) {
        return (static_cast<int64_t>(k) * ny + j) * nx + i;
    };
    std::vector<int64_t> cells;
    for (std::size_t p = 0; p < particles.size(); ++p) {
        if (!(particles.flags[p] & kParticleFlagFrozen)) {
            continue;
        }
        const Vec3 local = (particles.position[p] - grid.origin) * (1.0f / h);
        if (!std::isfinite(local.x) || !std::isfinite(local.y) || !std::isfinite(local.z) ||
            local.x < 0.0f || local.y < 0.0f || local.z < 0.0f ||
            local.x >= nx || local.y >= ny || local.z >= nz) {
            continue;
        }
        cells.push_back(key(static_cast<int>(std::floor(local.x)),
            static_cast<int>(std::floor(local.y)), static_cast<int>(std::floor(local.z))));
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    const auto frozen = [&](int i, int j, int k) {
        return i >= 0 && j >= 0 && k >= 0 && i < nx && j < ny && k < nz &&
            std::binary_search(cells.begin(), cells.end(), key(i, j, k));
    };
    const auto before = faces.size();
    for (const auto cell : cells) {
        const int i = static_cast<int>(cell % nx);
        const int j = static_cast<int>((cell / nx) % ny);
        const int k = static_cast<int>(cell / (static_cast<int64_t>(nx) * ny));
        for (int axis = 0; axis < 3; ++axis) {
            for (int side = 0; side < 2; ++side) {
                const int delta = side ? 1 : -1;
                if (frozen(i + (axis == 0 ? delta : 0), j + (axis == 1 ? delta : 0),
                        k + (axis == 2 ? delta : 0))) {
                    continue;
                }
                const int u = (axis + 1) % 3, v = (axis + 2) % 3;
                Vec3 corners[4];
                for (int q = 0; q < 4; ++q) {
                    int node[3] = {i, j, k};
                    node[axis] += side;
                    node[u] += q == 1 || q == 2 ? 1 : 0;
                    node[v] += q >= 2 ? 1 : 0;
                    corners[q] = grid.origin + Vec3(node[0] * h, node[1] * h, node[2] * h);
                }
                SurfaceMeshTriangle a, b;
                a.p0 = b.p0 = corners[0];
                a.p1 = corners[side ? 1 : 2];
                a.p2 = corners[side ? 2 : 1];
                b.p1 = corners[side ? 2 : 3];
                b.p2 = corners[side ? 3 : 2];
                faces.push_back(a);
                faces.push_back(b);
            }
        }
    }
    return faces.size() - before;
}
} // namespace RayTrophiSim::Fluid
