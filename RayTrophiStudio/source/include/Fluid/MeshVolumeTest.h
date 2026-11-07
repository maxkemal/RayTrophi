#pragma once
// Inside/outside test for a scene mesh, used to fill its VOLUME with matter
// (Seed Mode "Mesh Volume": a sand castle is a hidden castle mesh seeded with
// sand). Same sign rule as the collider SDF cook (scene_data.h
// launchCpuSdfCook): a three-ray parity vote over the triangle BVH, so a seed
// and a collider agree about what "inside" means for the same mesh.
//
// ★ The mesh need not be perfectly closed: a majority of three non-axis-aligned
// rays tolerates a hole or a duplicated face that flips one ray. A mesh that is
// an open SHEET has no inside at all, and every point classifies as outside —
// the seed then reports zero particles instead of guessing a volume.

#include "../ColliderMeshBVH.h"
#include "../FluidGrid.h"
#include "../Vec3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace RayTrophiSim::Fluid {

class MeshVolumeTest {
public:
    // World-space triangles. Returns false (and stays empty) for no triangles.
    bool build(std::vector<ColliderMeshBVH::Triangle> triangles, uint64_t version) {
        version_ = version;
        bounds_min_ = Vec3(std::numeric_limits<float>::max());
        bounds_max_ = Vec3(-std::numeric_limits<float>::max());
        for (const auto& t : triangles) {
            bounds_min_ = Vec3::min(bounds_min_, Vec3::min(Vec3::min(t.a, t.b), t.c));
            bounds_max_ = Vec3::max(bounds_max_, Vec3::max(Vec3::max(t.a, t.b), t.c));
        }
        triangle_count_ = triangles.size();
        bvh_.build(std::move(triangles));
        return !bvh_.empty();
    }

    bool empty() const { return bvh_.empty(); }
    uint64_t version() const { return version_; }
    std::size_t triangleCount() const { return triangle_count_; }
    const Vec3& boundsMin() const { return bounds_min_; }
    const Vec3& boundsMax() const { return bounds_max_; }

    bool inside(const Vec3& p) const {
        if (bvh_.empty()) return false;
        if (p.x < bounds_min_.x || p.y < bounds_min_.y || p.z < bounds_min_.z ||
            p.x > bounds_max_.x || p.y > bounds_max_.y || p.z > bounds_max_.z) {
            return false;
        }
        // The three probe directions of the collider cook (and of
        // sim_collider_bvh_query.comp): not axis-aligned, so a ray does not run
        // along a grid-aligned edge or face of a modelled wall.
        static const Vec3 kProbe[3] = {
            Vec3(0.5060f, 0.7071f, 0.4943f),
            Vec3(-0.3651f, 0.5345f, 0.7625f),
            Vec3(0.8112f, -0.2701f, 0.5184f)};
        int votes = 0;
        for (int d = 0; d < 3; ++d) {
            if (bvh_.countRayHits(p, kProbe[d]) & 1) ++votes;
        }
        return votes >= 2;
    }

    // Unsigned distance to the surface (for the cell classifier).
    float distance(const Vec3& p) const {
        Vec3 closest;
        return std::sqrt(bvh_.closestDistanceSquared(p, closest));
    }

private:
    ColliderMeshBVH bvh_;
    Vec3 bounds_min_ = Vec3(0.0f);
    Vec3 bounds_max_ = Vec3(0.0f);
    std::size_t triangle_count_ = 0;
    uint64_t version_ = 0;
};

// Per-cell classification of a MeshVolumeTest on one simulation grid, over the
// cell range that overlaps the mesh bounds. Each cell costs one distance query
// and one parity vote at its centre; only cells the surface passes through
// (distance < half the cell diagonal) fall back to the per-point test. A seed
// that tested every particle would cost three BVH rays per parcel, which for a
// detailed castle is millions of rays for one click.
class MeshVolumeCellMask {
public:
    enum : uint8_t { kOutside = 0, kInside = 1, kBoundary = 2 };

    MeshVolumeCellMask(const MeshVolumeTest& test, const FluidSim::FluidGrid& grid)
        : test_(&test), origin_(grid.origin), h_(grid.voxel_size) {
        if (test.empty() || !(h_ > 0.0f)) return;
        auto cellLo = [&](float v, float o, int n) {
            return std::clamp(static_cast<int>(std::floor((v - o) / h_)), 0, n - 1);
        };
        i0_ = cellLo(test.boundsMin().x, origin_.x, grid.nx);
        j0_ = cellLo(test.boundsMin().y, origin_.y, grid.ny);
        k0_ = cellLo(test.boundsMin().z, origin_.z, grid.nz);
        const int i1 = cellLo(test.boundsMax().x, origin_.x, grid.nx);
        const int j1 = cellLo(test.boundsMax().y, origin_.y, grid.ny);
        const int k1 = cellLo(test.boundsMax().z, origin_.z, grid.nz);
        nx_ = i1 - i0_ + 1;
        ny_ = j1 - j0_ + 1;
        nz_ = k1 - k0_ + 1;
        if (nx_ <= 0 || ny_ <= 0 || nz_ <= 0) { nx_ = ny_ = nz_ = 0; return; }
        cls_.assign(static_cast<std::size_t>(nx_) * ny_ * nz_, kOutside);
        const float half_diagonal = 0.8661f * h_ * 1.02f;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int k = 0; k < nz_; ++k) {
            for (int j = 0; j < ny_; ++j) {
                for (int i = 0; i < nx_; ++i) {
                    const Vec3 c = origin_ + Vec3((i0_ + i + 0.5f) * h_,
                                                  (j0_ + j + 0.5f) * h_,
                                                  (k0_ + k + 0.5f) * h_);
                    uint8_t v;
                    if (test.distance(c) < half_diagonal) {
                        v = kBoundary;
                    } else {
                        v = test.inside(c) ? kInside : kOutside;
                    }
                    cls_[index(i, j, k)] = v;
                    }
            }
        }
        for (uint8_t v : cls_) {
            if (v != kOutside) ++occupied_cells_;
        }
    }

    // Grid-cell class (absolute grid indices). Outside the mesh bounds: outside.
    uint8_t cellClass(int i, int j, int k) const {
        const int a = i - i0_, b = j - j0_, c = k - k0_;
        if (a < 0 || b < 0 || c < 0 || a >= nx_ || b >= ny_ || c >= nz_) return kOutside;
        return cls_[index(a, b, c)];
    }

    // Exact answer for a point: the cell's class, refined on boundary cells.
    bool contains(const Vec3& p) const {
        const int i = static_cast<int>(std::floor((p.x - origin_.x) / h_));
        const int j = static_cast<int>(std::floor((p.y - origin_.y) / h_));
        const int k = static_cast<int>(std::floor((p.z - origin_.z) / h_));
        const uint8_t v = cellClass(i, j, k);
        if (v == kBoundary) return test_->inside(p);
        return v == kInside;
    }

    // Cells the volume touches (inside + boundary): the seed's request count.
    std::size_t occupiedCells() const { return occupied_cells_; }
    bool empty() const { return occupied_cells_ == 0; }

private:
    std::size_t index(int a, int b, int c) const {
        return static_cast<std::size_t>(a) +
               static_cast<std::size_t>(b) * nx_ +
               static_cast<std::size_t>(c) * nx_ * ny_;
    }
    const MeshVolumeTest* test_ = nullptr;
    Vec3 origin_;
    float h_ = 0.0f;
    int i0_ = 0, j0_ = 0, k0_ = 0;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    std::vector<uint8_t> cls_;
    std::size_t occupied_cells_ = 0;
};

} // namespace RayTrophiSim::Fluid
