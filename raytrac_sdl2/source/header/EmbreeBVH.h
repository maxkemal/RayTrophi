#pragma once
#include <embree4/rtcore.h>
#include "Vec3.h"
#include "Ray.h"
#include "Hittable.h"
#include "Triangle.h"
#include <memory>
#include <vector>
#include "AABB.h"
#include <cstdint> // Add this include at the top of the file
#include "OptixWrapper.h"
struct TriangleData {
    Vec3 v0, v1, v2;
    Vec3 n0, n1, n2;
    Vec2 t0, t1, t2;
    std::shared_ptr<Material> material;
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
        // Embree kendi bounding box’ýný tuttuðu için dummy bir kutu döndürebilirsin.
        // Ya da sahnedeki tüm üçgenlerden büyük bir box hesaplayabilirsin.
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
