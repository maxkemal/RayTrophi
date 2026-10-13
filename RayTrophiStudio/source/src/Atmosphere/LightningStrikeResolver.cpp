#include "Atmosphere/LightningStrikeResolver.h"

#include "Atmosphere/AtmosphereLocalWeather.h"
#include "Atmosphere/LightningStrike.h"
#include "Hittable.h"
#include "InstanceManager.h"
#include "Ray.h"
#include "TerrainManager.h"
#include "TriangleMesh.h"
#include "scene_data.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace atmosphere {
namespace {

// Rolling-sphere contact height of a surface point at horizontal distance r
// from the leader's vertical: the sphere centre's height when it first
// touches the point. Points outside the striking distance never attract.
bool contactHeight(float y, float r, float strikingDistance, float& out) {
    if (r > strikingDistance) return false;
    out = y + std::sqrt((std::max)(strikingDistance * strikingDistance - r * r, 0.0f));
    return true;
}

// First surface straight down from (x, topY, z): meshes through the scene BVH
// only, terrains through TerrainManager. Not MeshEdit::raycastSurface: built
// for picking, it falls back to a linear sweep of every object's triangles
// when the BVH misses -- and most strike columns miss a 1 km mountain mesh
// under a 30 km storm, so each new bolt swept 2 M triangles 25 times and the
// frame stalled for seconds (2026-10-14). Without a BVH meshes are skipped.
bool firstSurfaceDown(SceneData& scene, float x, float z, float topY, Vec3& position,
                      bool& mesh, std::string& object) {
    const Ray ray(Vec3(x, topY, z), Vec3(0.0f, -1.0f, 0.0f));
    float bestT = std::numeric_limits<float>::max();
    bool found = false;
    if (scene.bvh) {
        HitRecord rec;
        if (scene.bvh->hit(ray, 0.001f, bestT, rec)) {
            bestT = rec.t;
            found = true;
            mesh = true;
            object = rec.tri_mesh ? rec.tri_mesh->nodeName : std::string();
        }
    }
    TerrainManager& terrains = TerrainManager::getInstance();
    if (terrains.hasActiveTerrain()) {
        for (TerrainObject& terrain : terrains.getTerrains()) {
            float t = 0.0f;
            Vec3 normal;
            if (terrains.intersectRay(&terrain, ray, t, normal) && t < bestT) {
                bestT = t;
                found = true;
                mesh = false;
                object.clear();
            }
        }
    }
    if (found) position = Vec3(x, topY - bestT, z);
    return found;
}

bool queryStrike(SceneData& scene, float x, float z, float topY, float strikingDistance,
                 LightningStrikeTarget& out) {
    out.striking_distance_m = strikingDistance;
    float best = -std::numeric_limits<float>::max();
    bool found = false;

    // Terrain and meshes: the first surface down each vertical of a polar
    // pattern (centre + 3 rings x 8) inside the striking distance. With
    // nothing under the strike the authored ground altitude applies.
    const float ringFraction[3] = {0.35f, 0.7f, 1.0f};
    for (int ring = -1; ring < 3; ++ring) {
        const int spokes = ring < 0 ? 1 : 8;
        for (int s = 0; s < spokes; ++s) {
            const float r = ring < 0 ? 0.0f : ringFraction[ring] * strikingDistance;
            const float angle = (static_cast<float>(s) + 0.5f * static_cast<float>(ring & 1)) *
                                (6.28318530718f / 8.0f);
            const float px = x + r * std::cos(angle);
            const float pz = z + r * std::sin(angle);
            Vec3 hit;
            bool mesh = false;
            std::string object;
            ++out.candidates;
            if (!firstSurfaceDown(scene, px, pz, topY, hit, mesh, object)) continue;
            float h;
            if (!contactHeight(hit.y, r, strikingDistance, h) || h <= best) continue;
            best = h;
            found = true;
            out.x = hit.x;
            out.y = hit.y;
            out.z = hit.z;
            out.kind = mesh ? "mesh" : "terrain";
            out.object = object;
        }
    }

    // Scattered instances (trees): their top. On Vulkan they are not in the
    // CPU BVH, so the rays above do not see them.
    for (const InstanceGroup& group : InstanceManager::getInstance().getGroups()) {
        if (group.transient || group.point_sphere_mode || group.rendered_rt_excluded ||
            group.sources.empty()) continue;
        for (const InstanceTransform& inst : group.instances) {
            const float dx = inst.position.x - x, dz = inst.position.z - z;
            if (std::fabs(dx) > strikingDistance || std::fabs(dz) > strikingDistance) continue;
            int sourceIndex = inst.source_index;
            if (sourceIndex < 0 || sourceIndex >= static_cast<int>(group.sources.size()))
                sourceIndex = 0;
            const ScatterSource& source = group.sources[static_cast<size_t>(sourceIndex)];
            if (!source.has_local_bbox) continue;
            ++out.candidates;
            // Instance position = the source's base pivot (mesh_center).
            const float top = inst.position.y +
                (source.local_bbox.max.y - source.mesh_center.y) * inst.scale.y;
            float h;
            if (!contactHeight(top, std::sqrt(dx * dx + dz * dz), strikingDistance, h) ||
                h <= best) continue;
            best = h;
            found = true;
            out.x = inst.position.x;
            out.y = top;
            out.z = inst.position.z;
            out.kind = "scatter";
            out.object = group.name;
        }
    }
    out.found = found;
    out.contact_height_m = found ? best : 0.0f;
    return found;
}

uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    return h;
}

// Changes with anything a strike can attach to. Coarse on purpose: a stale
// attachment lasts one event (under a second), a missed change costs nothing.
uint64_t sceneGeometryGeneration(SceneData& scene) {
    uint64_t h = 0;
    h = mix(h, reinterpret_cast<uintptr_t>(scene.bvh.get()));
    h = mix(h, scene.world.objects.size());
    const auto env = localWeatherEnvironment();
    h = mix(h, reinterpret_cast<uintptr_t>(env ? env->terrain.get() : nullptr));
    for (const InstanceGroup& group : InstanceManager::getInstance().getGroups()) {
        if (group.transient) continue;
        h = mix(h, static_cast<uint64_t>(group.id));
        h = mix(h, group.instances.size());
    }
    return h;
}

} // namespace

void installLightningStrikeResolver(SceneData& scene) {
    LightningStrikeResolver resolver;
    resolver.query = [&scene](float x, float z, float topY, float strikingDistance,
                              LightningStrikeTarget& out) {
        return queryStrike(scene, x, z, topY, strikingDistance, out);
    };
    resolver.generation = [&scene]() { return sceneGeometryGeneration(scene); };
    setLightningStrikeResolver(std::move(resolver));
}

} // namespace atmosphere
