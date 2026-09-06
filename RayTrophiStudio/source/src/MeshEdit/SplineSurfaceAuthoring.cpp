#include "MeshEdit/SplineSurfaceAuthoring.h"

#include "scene_data.h"
#include "scene_ui.h"
#include "Camera.h"
#include "Hittable.h"
#include "TriangleMesh.h"
#include "TerrainManager.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>

namespace MeshEdit {

namespace {

// Nearest scene-geometry hit. The BVH is authoritative when it exists; the
// linear sweep is the same fallback the viewport selection tool uses while an
// async build is still in flight.
bool hitSceneGeometry(SceneData& scene, const Ray& ray, HitRecord& out) {
    float closestT = 1e9f;
    bool found = false;
    if (scene.bvh) {
        HitRecord temp;
        if (scene.bvh->hit(ray, 0.001f, closestT, temp)) {
            found = true;
            closestT = temp.t;
            out = temp;
        }
    }
    if (!found) {
        for (const auto& object : scene.world.objects) {
            if (!object) continue;
            HitRecord temp;
            if (object->hit(ray, 0.001f, closestT, temp)) {
                found = true;
                closestT = temp.t;
                out = temp;
            }
        }
    }
    return found;
}

bool hitTerrain(const Ray& ray, float& tOut, Vec3& normalOut) {
    auto& manager = TerrainManager::getInstance();
    if (!manager.hasActiveTerrain()) return false;
    float closestT = 1e20f;
    Vec3 closestNormal(0.0f, 1.0f, 0.0f);
    bool found = false;
    for (auto& terrain : manager.getTerrains()) {
        float t = 0.0f;
        Vec3 normal;
        if (manager.intersectRay(&terrain, ray, t, normal) && t < closestT) {
            closestT = t;
            closestNormal = normal;
            found = true;
        }
    }
    if (!found) return false;
    tOut = closestT;
    normalOut = closestNormal;
    return true;
}

bool hitGroundPlane(const Ray& ray, float& tOut) {
    if (std::fabs(ray.direction.y) <= 0.01f) return false;
    const float t = -ray.origin.y / ray.direction.y;
    if (t <= 0.0f) return false;
    tOut = t;
    return true;
}

} // namespace

bool raycastSurface(SceneData& scene,
                    const Ray& ray,
                    SurfaceFilter filter,
                    SurfaceSnapResult& out) {
    out = SurfaceSnapResult{};

    const bool allowMesh = (filter == SurfaceFilter::MeshAndTerrain);
    const bool allowTerrain = (filter == SurfaceFilter::MeshAndTerrain ||
                               filter == SurfaceFilter::TerrainOnly);

    HitRecord meshHit;
    const bool gotMesh = allowMesh && hitSceneGeometry(scene, ray, meshHit);

    float terrainT = 0.0f;
    Vec3 terrainNormal(0.0f, 1.0f, 0.0f);
    const bool gotTerrain = allowTerrain && hitTerrain(ray, terrainT, terrainNormal);

    // Compare distances instead of preferring whichever was tested first. The
    // previous helper returned the mesh hit without ever consulting the terrain,
    // so a mesh BEHIND the terrain won and the point landed underground.
    if (gotMesh && (!gotTerrain || meshHit.t <= terrainT)) {
        out.kind = SurfaceHitKind::Mesh;
        out.distance = meshHit.t;
        out.position = ray.origin + ray.direction * meshHit.t;
        out.normal = meshHit.normal;
        if (meshHit.tri_mesh) out.object_name = meshHit.tri_mesh->nodeName;
        return true;
    }
    if (gotTerrain) {
        out.kind = SurfaceHitKind::Terrain;
        out.distance = terrainT;
        out.position = ray.origin + ray.direction * terrainT;
        out.normal = terrainNormal;
        return true;
    }

    // Nothing under the cursor: the same Y=0 fallback the River tool has always
    // used, so clicking past the edge of the world still places a point where
    // the cursor is aiming rather than doing nothing.
    float groundT = 0.0f;
    if (hitGroundPlane(ray, groundT)) {
        out.kind = SurfaceHitKind::GroundPlane;
        out.distance = groundT;
        out.position = ray.origin + ray.direction * groundT;
        out.normal = Vec3(0.0f, 1.0f, 0.0f);
        return true;
    }
    return false;
}

bool snapToSurface(UIContext& ctx,
                   const ImVec2& mousePos,
                   SurfaceFilter filter,
                   SurfaceSnapResult& out) {
    out = SurfaceSnapResult{};
    if (!ctx.scene.camera) return false;
    ImGuiIO& io = ImGui::GetIO();
    const float viewportWidth = std::max(1.0f, io.DisplaySize.x);
    const float viewportHeight = std::max(1.0f, io.DisplaySize.y);
    const float u = mousePos.x / viewportWidth;
    const float v = 1.0f - (mousePos.y / viewportHeight);
    const Ray ray = ctx.scene.camera->get_ray(u, v);
    return raycastSurface(ctx.scene, ray, filter, out);
}

} // namespace MeshEdit
