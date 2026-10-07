#include "Fluid/MatterGrainColliderBvh.h"
#include "SurfaceMeshCache.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>

int main() {
    using namespace RayTrophiSim;
    using namespace RayTrophiSim::Fluid;
    SurfaceMeshTriangle a, b, c, disconnected;
    a.p0 = Vec3(0, 0, 0);
    a.p1 = Vec3(1, 0, 0);
    a.p2 = Vec3(1, 0, 1);
    b.p0 = a.p0;
    b.p1 = a.p2;
    b.p2 = Vec3(0, 0, 1);
    c.p0 = a.p0;
    c.p1 = a.p1;
    c.p2 = Vec3(0, 1, 0);
    disconnected.p0 = Vec3(2, 0, 0);
    disconnected.p1 = Vec3(3, 0, 0);
    disconnected.p2 = Vec3(3, 0, 1);
    MatterGrainColliderBvh result;
    std::string error;
    assert(buildMatterGrainColliderBvh({a, b, c, disconnected}, result, error));
    assert(result.surface_patches[0] == result.surface_patches[1]);
    assert(result.surface_patches[0] != result.surface_patches[2]);
    assert(result.surface_patches[0] != result.surface_patches[3]);
    const auto before = result.vertices.size();
    auto bad = a;
    bad.p0.x = std::numeric_limits<float>::quiet_NaN();
    assert(!buildMatterGrainColliderBvh({bad}, result, error));
    assert(result.vertices.size() == before);
    std::vector<SurfaceMeshTriangle> many;
    for (int i = 0; i < 100; ++i) {
        auto triangle = a;
        const Vec3 shift(float(i * 2), 0, 0);
        triangle.p0 = triangle.p0 + shift;
        triangle.p1 = triangle.p1 + shift;
        triangle.p2 = triangle.p2 + shift;
        many.push_back(triangle);
    }
    assert(buildMatterGrainColliderBvh(many, result, error));
    auto ids = result.source_faces;
    std::sort(ids.begin(), ids.end());
    for (uint32_t i = 0; i < ids.size(); ++i) {
        assert(ids[i] == i);
    }
    assert(result.nodes.size() < 2 * many.size());
    for (const auto& node : result.nodes) {
        if (node.first & 0x80000000u) {
            assert(node.second <= 4);
            assert((node.first & 0x7fffffffu) + node.second <= many.size());
        } else {
            assert(node.first < result.nodes.size() && node.second < result.nodes.size());
        }
    }
    // Moving collider: node bounds cover the sweep end - v * dt .. end, and
    // the velocities follow the reordered faces.
    std::vector<Vec3> velocity(3 * many.size(), Vec3(0.0f));
    for (std::size_t f = 0; f < many.size(); ++f) {
        for (int k = 0; k < 3; ++k) velocity[3 * f + k] = Vec3(0.0f, float(f), 0.0f);
    }
    assert(buildMatterGrainColliderBvh(many, result, error, &velocity, .5f));
    assert(result.velocities.size() == result.vertices.size());
    for (std::size_t slot = 0; slot < result.source_faces.size(); ++slot) {
        assert(result.velocities[3 * slot].y == float(result.source_faces[slot]));
    }
    // The root holds the deepest start: face 99 began at y = -49.5.
    assert(result.nodes[0].low[1] <= -49.5f + 1e-4f);
    assert(matterGrainColliderFingerprint(many, &velocity) != matterGrainColliderFingerprint(many));
    std::vector<Vec3> short_velocity(3);
    assert(!buildMatterGrainColliderBvh(many, result, error, &short_velocity, .5f));
    std::printf("PASS grain collider BVH (patches, transactional, sweep bounds, velocities)\n");
}
