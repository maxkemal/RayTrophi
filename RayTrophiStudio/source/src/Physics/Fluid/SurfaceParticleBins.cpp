#include "Fluid/SurfaceParticleBins.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace RayTrophiSim::Fluid {
SurfaceParticleBins::SurfaceParticleBins(const Vec3& origin, int dx, int dy, int dz,
                                       float voxel, float width)
    : origin_(origin) {
    inverse_width = 1.0f / std::max(width, voxel);
    nx = std::max(1, static_cast<int>(std::ceil(dx * voxel * inverse_width)));
    ny = std::max(1, static_cast<int>(std::ceil(dy * voxel * inverse_width)));
    nz = std::max(1, static_cast<int>(std::ceil(dz * voxel * inverse_width)));
}

std::size_t SurfaceParticleBins::index(int x, int y, int z) const {
    return static_cast<std::size_t>(x) + static_cast<std::size_t>(nx) *
        (y + static_cast<std::size_t>(ny) * z);
}

void SurfaceParticleBins::coordinate(const Vec3& world, int& x, int& y, int& z) const {
    const Vec3 local = (world - origin_) * inverse_width;
    x = static_cast<int>(std::floor(local.x));
    y = static_cast<int>(std::floor(local.y));
    z = static_cast<int>(std::floor(local.z));
}

void SurfaceParticleBins::build(const std::vector<Vec3>& centres,
                               const std::vector<std::size_t>& eligible_cells) {
    const std::size_t invalid = std::numeric_limits<std::size_t>::max();
    const std::size_t count = static_cast<std::size_t>(nx) * ny * nz;
    offsets.assign(count + 1, 0);
    std::vector<std::size_t> cell(eligible_cells.size(), invalid);
    for (std::size_t p = 0; p < eligible_cells.size(); ++p) {
        if (eligible_cells[p] == invalid) {
            continue;
        }
        int x, y, z;
        coordinate(centres[p], x, y, z);
        x = std::clamp(x, 0, nx - 1);
        y = std::clamp(y, 0, ny - 1);
        z = std::clamp(z, 0, nz - 1);
        cell[p] = index(x, y, z);
        ++offsets[cell[p] + 1];
    }
    for (std::size_t c = 0; c < count; ++c) {
        offsets[c + 1] += offsets[c];
    }
    std::vector<std::size_t> cursor(offsets.begin(), offsets.end() - 1);
    particles.resize(offsets.back());
    for (std::size_t p = 0; p < cell.size(); ++p) {
        if (cell[p] != invalid) {
            particles[cursor[cell[p]]++] = p;
        }
    }
}
}
