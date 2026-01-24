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
    Vec3 v0, v1, v2;          // 36 bytes
    Vec3 n0, n1, n2;          // 36 bytes
    Vec2 t0, t1, t2;          // 24 bytes
    uint16_t materialID;      // 2 bytes
    const Triangle* original_ptr = nullptr; // Pointer to original object for identity/name checks
    
    // Total: 98 bytes (was 114 bytes with shared_ptr)
    // Savings: 16 bytes per triangle!
    
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

    void build(const std::vector<std::shared_ptr<Hittable>>& objects);
    void clearGeometry();
    void updateGeometryFromTriangles();
    void updateGeometryFromTrianglesFromSource(const std::vector<std::shared_ptr<Hittable>>& objects);
    bool occluded(const Ray& ray, float t_min, float t_max) const override;
    __m256 occluded_packet(const RayPacket& packet, float t_min, __m256 t_max) const override;
    void buildFromTriangleData(const std::vector<TriangleData>& triangles);
    bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec, bool ignore_volumes = false) const override;
    void hit_packet(const RayPacket& r, float t_min, float t_max, HitRecordPacket& rec, bool ignore_volumes = false) const override;
    void clearAndRebuild(const std::vector<std::shared_ptr<Hittable>>& objects);
    OptixGeometryData exportToOptixData() const;
    
    bool bounding_box(float time0, float time1, AABB& output_box) const override {
        if (triangle_data.empty()) return false;

        AABB bbox;
        for (const auto& tri : triangle_data) {
            AABB tri_box;
            tri_box.min = Vec3::min(Vec3::min(tri.v0, tri.v1), tri.v2);
            tri_box.max = Vec3::max(Vec3::max(tri.v0, tri.v1), tri.v2);
            bbox = surrounding_box(bbox, tri_box);
        }

        output_box = bbox;
        return true;
    }

private:
    static RTCDevice device; // Shared device across all BVH instances (persistent)
    RTCScene scene;
    std::vector<TriangleData> triangle_data;
    
    // Instance mapping: geometryID -> child BVH
    unsigned triangle_geom_id = 0xFFFFFFFF; // RTC_INVALID_GEOMETRY_ID
    std::vector<const HittableInstance*> instance_objects;

    // VDB Volume Support (User Geometry)
    unsigned vdb_geom_id = 0xFFFFFFFF;
    std::vector<const class VDBVolume*> vdb_objects;

    // Static callbacks for Embree User Geometry
    static void userBoundsFunc(const struct RTCBoundsFunctionArguments* args);
    static void userIntersectFunc(const struct RTCIntersectFunctionNArguments* args);
    static void userOccludedFunc(const struct RTCOccludedFunctionNArguments* args);

public:
    RTCScene getRTCScene() const { return scene; }
    static void shutdown(); // Call on app exit to release device
};

