#pragma once

#include "Hittable.h"
#include "Vec3.h"
#include "Vec2.h"
#include "DNA/GeometryDetail.h"
#include <memory>
#include <string>
#include <vector>

class Transform;
class ParallelBVHNode;

/**
     * @brief Contiguous memory container for a triangle mesh.
     * 
     * Replaces the old system of having a separate `shared_ptr<Triangle>` for every single triangle.
     * Delegates all flat geometry storage (SoA, AVX aligned, delta-tracked) to DNA::GeometryDetail.
     */
class TriangleMesh : public Hittable {
public:
    // Core geometry database (contains flat, aligned attributes & deltas)
    std::shared_ptr<DNA::GeometryDetail> geometry;
    
    // Metadata
    std::string nodeName;
    std::shared_ptr<Transform> transform;

    // Terrain ID if this mesh IS a terrain's flat mesh (-1 otherwise). Mirrors the
    // old per-facade Triangle::terrain_id — propagated into HitRecord::terrain_id
    // by the CPU/Embree path (see EmbreeBVH.cpp's "facadeless" mesh group handling)
    // so terrain hits route through TerrainObject's layer/splat shading instead of
    // a plain PBR material lookup.
    int terrain_id = -1;

    // Optional Local BVH for CPU raytracing
    std::shared_ptr<ParallelBVHNode> local_bvh;

    TriangleMesh();
    virtual ~TriangleMesh() = default;

    // Hittable interface implementation
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec, bool ignore_volumes = false) const override;
    virtual bool bounding_box(float time0, float time1, AABB& output_box) const override;
    
    // Fast RTTI bypass
    virtual bool isTriangleMesh() const { return true; }

    // Helper methods
    void clear();
    void build_local_bvh();
    
    // Vertex and Triangle counts
    inline size_t num_vertices() const { 
        return geometry ? geometry->get_vertex_count() : 0; 
    }
    
    inline size_t num_triangles() const { 
        return geometry ? geometry->indices.size() / 3 : 0; 
    }
};
