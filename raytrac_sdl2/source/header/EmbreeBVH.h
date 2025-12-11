#pragma once
#include <embree4/rtcore.h>
#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"
#include "Triangle.h"
#include "MaterialManager.h"
#include <memory>
#include <vector>
#include "AABB.h"
#include <cstdint>
#include "OptixWrapper.h"

/**
 * @brief Optimized triangle data for Embree BVH
 * 
 * Now stores materialID (2 bytes) instead of shared_ptr (16+ bytes)
 * while maintaining legacy material pointer for backward compatibility.
 */
struct TriangleData {
    Vec3 v0, v1, v2;
    Vec3 n0, n1, n2;
    Vec2 t0, t1, t2;
    
    // Optimized: Material ID for MaterialManager lookup
    uint16_t materialID = 0xFFFF;
    
    // Legacy: Keep material pointer for backward compatibility
    std::shared_ptr<Material> material;
    
    // Helper to get material (prefers ID lookup, falls back to pointer)
    Material* getMaterial() const {
        if (materialID != 0xFFFF) {
            return MaterialManager::getInstance().getMaterial(materialID);
        }
        return material.get();
    }
    
    std::shared_ptr<Material> getMaterialShared() const {
        if (materialID != 0xFFFF) {
            return MaterialManager::getInstance().getMaterialShared(materialID);
        }
        return material;
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
    bool occluded(const Ray& ray, float t_min, float t_max) const;
    void buildFromTriangleData(const std::vector<TriangleData>& triangles);
    bool hit(const Ray& ray, float t_min, float t_max, HitRecord& rec) const override;
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
    RTCDevice device;
    RTCScene scene;
    std::vector<TriangleData> triangle_data;
};
