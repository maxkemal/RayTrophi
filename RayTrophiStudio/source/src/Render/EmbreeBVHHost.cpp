// EmbreeBVH: Player (no-Embree) link alternative.
//
// PLAYER BUILD ONLY. Studio links EmbreeBVH.cpp (the real Embree scene); the
// two must never be linked into the same target (Studio lists this file as
// None).
//
// The Player selects the in-house CPU BVH instead: PlayerMain sets
// `use_embree = false`, so Renderer builds a ParallelBVHNode and CPU ray
// queries (picking, surface snap) keep working. This class therefore exists
// only so shared code that names EmbreeBVH links. If something does construct
// one (the particle bridge builds child BVHs for the CPU composite), it is an
// EMPTY acceleration structure: no hit, no occlusion, and one warning in the
// log so an empty CPU result is never mistaken for a measurement.
// Plan: docs/dev/RAYTROPHI_PLAYER_PLAN.md (Faz 3d).

#include "EmbreeBVH.h"
#include "globals.h"

#include <atomic>

namespace {
void warnNoEmbreeOnce() {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
        SCENE_LOG_WARN("[Player] EmbreeBVH requested but Embree is not part of RayTrophi Player; "
                       "this BVH is empty (CPU ray queries against it return no hit).");
    }
}
} // namespace

EmbreeBVH::EmbreeBVH() : scene(nullptr) {
}

EmbreeBVH::~EmbreeBVH() {
}

void EmbreeBVH::build(const std::vector<std::shared_ptr<Hittable>>& /*objects*/) {
    warnNoEmbreeOnce();
}

void EmbreeBVH::updateGeometryFromTrianglesFromSource(const std::vector<std::shared_ptr<Hittable>>& /*objects*/) {
    warnNoEmbreeOnce();
}

bool EmbreeBVH::occluded(const Ray& /*ray*/, float /*t_min*/, float /*t_max*/) const {
    return false;
}

bool EmbreeBVH::hit(const Ray& /*ray*/, float /*t_min*/, float /*t_max*/, HitRecord& /*rec*/,
                    bool /*ignore_volumes*/) const {
    return false;
}

int EmbreeBVH::remapMeshMaterialID(const TriangleMesh* /*mesh*/, uint16_t /*oldID*/, uint16_t /*newID*/) {
    return 0;
}
