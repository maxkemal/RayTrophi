/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          EmbreeBVH.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include <embree4/rtcore.h>
#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"
#include "Triangle.h"
#include "MaterialManager.h"
#include <memory>
class HittableInstance; // Forward decl
#include <vector>
#include "AABB.h"
#include <cstdint>
#include "OptixWrapper.h"

/**
 * @brief Optimized triangle data for Embree BVH
 * 
 * Stores materialID (2 bytes) instead of shared_ptr (16+ bytes).
 * Material access is always through MaterialManager for consistency.
 */
struct TriangleData {
    uint16_t materialID;      // 2 bytes
    int terrain_id = -1;      // Terrain ID if this triangle belongs to a terrain mesh
    const Triangle* original_ptr = nullptr; // Pointer to original object for identity/name checks
    // Facade-less path (flat/proxy migration): when original_ptr is null, this primitive came
    // from a TriangleMesh placed directly in world.objects (no per-face Triangle object). Normals
    // / UVs / identity are read from the mesh SoA via face_index instead of original_ptr.
    TriangleMesh* mesh_ptr = nullptr;
    uint32_t face_index = 0;
    
    // Helper methods for material access via MaterialManager
    Material* getMaterial() const {
        return MaterialManager::getInstance().getMaterial(materialID);
    }
    
    std::shared_ptr<Material> getMaterialShared() const {
        return MaterialManager::getInstance().getMaterialShared(materialID);
    }
};

class EmbreeBVH : public Hittable {
public:
    EmbreeBVH();
    ~EmbreeBVH();
    std::vector<TriangleData> triangle_data;
    void build(const std::vector<std::shared_ptr<Hittable>>& objects);
    void clearGeometry();
    void updateGeometryFromTriangles();
    void updateGeometryFromTrianglesFromSource(const std::vector<std::shared_ptr<Hittable>>& objects);
    bool occluded(const Ray& ray, float t_min, float t_max) const override;
    bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec, bool ignore_volumes = false) const override;
    void clearAndRebuild(const std::vector<std::shared_ptr<Hittable>>& objects);
    OptixGeometryData exportToOptixData() const;

    // Flat/proxy: remap the baked materialID of every facade-less face belonging to `mesh`
    // (matching oldID; pass oldID==INVALID_MATERIAL_ID to repaint all of the mesh's faces) to
    // newID, in place — avoids a full BVH rebuild after a material assignment on a dense mesh
    // that lives in world.objects as a single TriangleMesh. Returns faces remapped.
    int remapMeshMaterialID(const class TriangleMesh* mesh, uint16_t oldID, uint16_t newID);
    
    bool bounding_box(float time0, float time1, AABB& output_box) const override {
        if (triangle_data.empty()) return false;

        AABB bbox;
        for (const auto& tri : triangle_data) {
            if (!tri.original_ptr) continue;
            AABB tri_box;
            Vec3 v0 = tri.original_ptr->getVertexPosition(0);
            Vec3 v1 = tri.original_ptr->getVertexPosition(1);
            Vec3 v2 = tri.original_ptr->getVertexPosition(2);
            tri_box.min = Vec3::min(Vec3::min(v0, v1), v2);
            tri_box.max = Vec3::max(Vec3::max(v0, v1), v2);
            bbox = surrounding_box(bbox, tri_box);
        }

        output_box = bbox;
        return true;
    }

private:
    static RTCDevice device; // Shared device across all BVH instances (persistent)
    RTCScene scene;
  
    
    // Instance mapping: geometryID -> child BVH
   

    // Static callbacks for Embree User Geometry
    static void userBoundsFunc(const struct RTCBoundsFunctionArguments* args);
    static void userIntersectFunc(const struct RTCIntersectFunctionNArguments* args);
    static void userOccludedFunc(const struct RTCOccludedFunctionNArguments* args);

public:
   
    unsigned triangle_geom_id = 0xFFFFFFFF; // RTC_INVALID_GEOMETRY_ID
    std::vector<std::shared_ptr<HittableInstance>> instance_objects;
    std::vector<std::shared_ptr<Triangle>> cached_triangles; // [NEW] Önbelleklenmiş sahne üçgenleri
    // Keep facade-less meshes alive for as long as this BVH can return hits.
    // TriangleData keeps a compact raw mesh_ptr, while this vector owns its lifetime.
    std::vector<std::shared_ptr<TriangleMesh>> cached_direct_meshes;
    
    // Grouping structure to exploit contiguous flat buffers
    struct EmbreeMeshGroup {
        TriangleMesh* mesh = nullptr;
        size_t vertex_offset = 0;
        size_t vertex_count = 0;
        Matrix4x4 last_xform;
        uint64_t last_pose_hash = 0;
        // Facade-less (direct SoA TriangleMesh) group: the refit must bake world = getFinal()*P_orig
        // (like build does) instead of memcpy-ing the SoA "P" cache, which isn't re-baked on a
        // transform-only change. Lets a keyframed/physics flat mesh REFIT instead of full-rebuild.
        bool facadeless = false;
    };
    std::vector<EmbreeMeshGroup> active_mesh_groups;
    size_t standalone_vertex_offset = 0;
    size_t standalone_tri_offset = 0;
    // Flat/proxy: total facade-less faces emitted from direct TriangleMesh groups. These have no
    // entry in cached_triangles, so the refit size-equality guard must add this count or it would
    // forever mismatch (cached_triangles.size() != triangle_data.size()) and bail to full rebuild.
    size_t facadeless_tri_count = 0;

    // VDB Volume Support (User Geometry)
    unsigned vdb_geom_id = 0xFFFFFFFF;
    std::vector<const class VDBVolume*> vdb_objects;
    RTCScene getRTCScene() const { return scene; }
    static void shutdown(); // Call on app exit to release device
};

