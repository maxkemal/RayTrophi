#pragma once

#include "Vec3.h"
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace RayTrophiSim::Fluid {
// Sparse scheduling on a fixed world-aligned grid. This owns no particle state.
class SurfaceWorkTiles {
public:
    static constexpr int edge = 8;
    SurfaceWorkTiles(int nx, int ny, int nz, const Vec3& origin, float voxel);
    void addSupport(const Vec3& center, float radius);
    void finish();
    std::size_t workSize() const;
    bool cellAt(std::size_t work, int& x, int& y, int& z) const;
    std::size_t tileCount() const { return tiles_.size(); }

private:
    int nx_, ny_, nz_, tx_, ty_;
    Vec3 origin_;
    float voxel_;
    std::unordered_set<uint64_t> pending_;
    std::vector<uint64_t> tiles_;
};

// Preserve the existing six-neighbour Jacobi result while evaluating only
// scheduled cells. The schedule must include support + iterations of halo.
void smoothSurfaceTiles(std::vector<float>& sdf, const SurfaceWorkTiles& tiles,
                        int nx, int ny, int nz, float far_value,
                        int iterations, int threads,
                        const std::vector<float>* support_confidence = nullptr,
                        int sample_stride = 1, float displacement_limit = 0.0f);
void smoothSurfaceCoordinateTiles(std::vector<float>& coordinates,
                                  const std::vector<uint8_t>& valid,
                                  const SurfaceWorkTiles& tiles, int nx, int ny, int nz,
                                  int iterations, int threads);
}
