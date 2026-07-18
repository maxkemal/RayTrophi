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
class Matrix4x4;

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

    // Per-SoA-vertex pointiness (Geometry node's Pointiness output): 0.5 flat, >0.5 a
    // convex ridge, <0.5 a concave crease. Empty unless some live material graph reads
    // it — rebuilt from the live SoA by MeshAttr::computeMeshPointiness (MeshPointiness.h)
    // on every BVH rebuild, and interpolated barycentrically at the hit.
    std::vector<float> pointiness;

    // Attribute node: the named per-vertex float channels a material graph reads (sculpt /
    // Geo-DAG masks, paint layers, vertex groups), gathered out of GeometryDetail's custom
    // attribute map into ONE interleaved block — kMatAttribSlots floats per vertex,
    // material_attribs[vertexId * kMatAttribSlots + slot].
    //
    // The map is the authoring format and the wrong thing to read at a hit: it is keyed by
    // std::string, so sampling it per shading point would hash a string in the hot path.
    // Interleaving also happens to be exactly the layout the GPU block wants (one address,
    // constant stride), so CPU and Vulkan interpolate the identical bytes.
    //
    // Empty unless some live material graph reads an Attribute node — rebuilt by
    // MeshAttr::computeMeshMaterialAttributes on every BVH rebuild.
    std::vector<float> material_attribs;

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

    // Canonical flat-mesh skinning helpers. Skin weights live per SoA vertex in
    // GeometryDetail; no Triangle facade is required to query or deform them.
    bool hasSkinWeights() const;
    bool applySkinning(const std::vector<Matrix4x4>& finalBoneMatrices);
};
