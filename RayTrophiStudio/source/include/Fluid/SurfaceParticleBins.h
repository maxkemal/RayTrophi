#pragma once

#include "Vec3.h"
#include <cstddef>
#include <vector>

namespace RayTrophiSim::Fluid {
// CSR over temporary render centres. Particle indices still address canonical SoA.
class SurfaceParticleBins {
public:
    SurfaceParticleBins(const Vec3& origin, int nx, int ny, int nz,
                        float voxel, float width);
    void build(const std::vector<Vec3>& centres,
               const std::vector<std::size_t>& eligible_cells);
    void coordinate(const Vec3& world, int& x, int& y, int& z) const;
    std::size_t index(int x, int y, int z) const;
    int nx = 0, ny = 0, nz = 0;
    float inverse_width = 0.0f;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> particles;

private:
    Vec3 origin_;
};
}
