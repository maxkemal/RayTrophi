/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          TriangleMesh.h
* Author:        Kemal Demirtas
* Date:          June 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef TRIANGLE_MESH_H
#define TRIANGLE_MESH_H

#include "Hittable.h"
#include "Vec3.h"
#include "Vec2.h"
#include <vector>
#include <string>
#include <memory>

class Transform;
class ParallelBVHNode;

/**
 * @brief Contiguous memory container for a triangle mesh.
 * 
 * Replaces the old system of having a separate `shared_ptr<Triangle>` for every single triangle.
 * Stores vertex data in SoA (Struct of Arrays) format for maximum cache coherency.
 */
class TriangleMesh : public Hittable {
public:
    // Core geometry arrays (Struct of Arrays)
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;
    
    // Topology (Indices are flat: 0,1,2 for first triangle, 3,4,5 for second...)
    std::vector<uint32_t> indices;
    
    // Material IDs (one per triangle)
    std::vector<uint16_t> materialIDs;
    
    // Metadata
    std::string nodeName;
    std::shared_ptr<Transform> transform;

    // Optional Local BVH for CPU raytracing
    std::shared_ptr<ParallelBVHNode> local_bvh;

    TriangleMesh() = default;
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
    inline size_t num_vertices() const { return positions.size(); }
    inline size_t num_triangles() const { return indices.size() / 3; }
};

#endif // TRIANGLE_MESH_H
