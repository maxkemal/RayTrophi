/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          TriangleMesh.cpp
* Author:        Kemal Demirtas
* Date:          June 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#include "TriangleMesh.h"
#include "ParallelBVHNode.h" // Optional, can be stubbed out if disabled
#include "Transform.h"

void TriangleMesh::clear() {
    positions.clear();
    normals.clear();
    uvs.clear();
    indices.clear();
    materialIDs.clear();
    local_bvh = nullptr;
}

void TriangleMesh::build_local_bvh() {
    // For now, as per user request, we can skip BVH if not needed, 
    // or build a basic one. We'll leave it empty to fallback to naive loop
    // or implement a custom BVH tree later.
    local_bvh = nullptr; 
}

bool TriangleMesh::bounding_box(float time0, float time1, AABB& output_box) const {
    if (positions.empty()) return false;
    
    Vec3 min_pt = positions[0];
    Vec3 max_pt = positions[0];
    
    for (size_t i = 1; i < positions.size(); ++i) {
        min_pt = Vec3(
            (std::min)(min_pt.x, positions[i].x),
            (std::min)(min_pt.y, positions[i].y),
            (std::min)(min_pt.z, positions[i].z)
        );
        max_pt = Vec3(
            (std::max)(max_pt.x, positions[i].x),
            (std::max)(max_pt.y, positions[i].y),
            (std::max)(max_pt.z, positions[i].z)
        );
    }
    
    output_box = AABB(min_pt, max_pt);
    
    if (transform) {
        Matrix4x4 mat = transform->getMatrix();
        Vec3 corners[8] = {
            Vec3(min_pt.x, min_pt.y, min_pt.z),
            Vec3(max_pt.x, min_pt.y, min_pt.z),
            Vec3(min_pt.x, max_pt.y, min_pt.z),
            Vec3(max_pt.x, max_pt.y, min_pt.z),
            Vec3(min_pt.x, min_pt.y, max_pt.z),
            Vec3(max_pt.x, min_pt.y, max_pt.z),
            Vec3(min_pt.x, max_pt.y, max_pt.z),
            Vec3(max_pt.x, max_pt.y, max_pt.z)
        };
        Vec3 new_min = mat.transform_point(corners[0]);
        Vec3 new_max = new_min;
        for (int c = 1; c < 8; ++c) {
            Vec3 pt = mat.transform_point(corners[c]);
            new_min = Vec3((std::min)(new_min.x, pt.x), (std::min)(new_min.y, pt.y), (std::min)(new_min.z, pt.z));
            new_max = Vec3((std::max)(new_max.x, pt.x), (std::max)(new_max.y, pt.y), (std::max)(new_max.z, pt.z));
        }
        output_box = AABB(new_min, new_max);
    }
    
    return true;
}

bool TriangleMesh::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, bool ignore_volumes) const {
    if (!visible) return false;
    
    Ray local_r = r;
    if (transform) {
        Matrix4x4 inv = transform->getInverseMatrix();
        local_r.origin = inv.transform_point(r.origin);
        local_r.direction = inv.transform_vector(r.direction);
    }

    bool hit_anything = false;
    float closest_so_far = t_max;
    HitRecord temp_rec;

    if (local_bvh) {
        // If local BVH exists, intersect with it
        // hit_anything = local_bvh->hit(local_r, t_min, closest_so_far, temp_rec, ignore_volumes);
    } else {
        // Naive iteration (Warning: very slow for high poly without BVH!)
        size_t tri_count = indices.size() / 3;
        for (size_t i = 0; i < tri_count; ++i) {
            uint32_t i0 = indices[i * 3 + 0];
            uint32_t i1 = indices[i * 3 + 1];
            uint32_t i2 = indices[i * 3 + 2];
            
            const Vec3& v0 = positions[i0];
            const Vec3& v1 = positions[i1];
            const Vec3& v2 = positions[i2];
            
            Vec3 edge1 = v1 - v0;
            Vec3 edge2 = v2 - v0;
            Vec3 h = local_r.direction.cross(edge2);
            float a = edge1.dot(h);
            
            if (a > -1e-8f && a < 1e-8f) continue;
            
            float f = 1.0f / a;
            Vec3 s = local_r.origin - v0;
            float u = f * s.dot(h);
            
            if (u < 0.0f || u > 1.0f) continue;
            
            Vec3 q = s.cross(edge1);
            float v = f * local_r.direction.dot(q);
            
            if (v < 0.0f || u + v > 1.0f) continue;
            
            float t = f * edge2.dot(q);
            
            if (t > t_min && t < closest_so_far) {
                closest_so_far = t;
                hit_anything = true;
                
                temp_rec.t = t;
                temp_rec.u = u;
                temp_rec.v = v;
                temp_rec.point = local_r.at(t);
                
                // Normal interpolation
                if (!normals.empty()) {
                    const Vec3& n0 = normals[i0];
                    const Vec3& n1 = normals[i1];
                    const Vec3& n2 = normals[i2];
                    Vec3 normal = (1.0f - u - v) * n0 + u * n1 + v * n2;
                    temp_rec.interpolated_normal = normal.normalize();
                    temp_rec.set_face_normal(local_r, temp_rec.interpolated_normal);
                } else {
                    Vec3 normal = edge1.cross(edge2).normalize();
                    temp_rec.interpolated_normal = normal;
                    temp_rec.set_face_normal(local_r, normal);
                }
                
                // UV interpolation
                if (!uvs.empty()) {
                    const Vec2& uv0 = uvs[i0];
                    const Vec2& uv1 = uvs[i1];
                    const Vec2& uv2 = uvs[i2];
                    temp_rec.uv = (1.0f - u - v) * uv0 + u * uv1 + v * uv2;
                }
                
                if (!materialIDs.empty()) {
                    temp_rec.materialID = materialIDs[i];
                }
            }
        }
    }
    
    if (hit_anything) {
        rec = temp_rec;
        if (transform) {
            Matrix4x4 mat = transform->getMatrix();
            Matrix4x4 normMat = transform->getNormalTransform();
            rec.point = mat.transform_point(rec.point);
            rec.normal = normMat.transform_vector(rec.normal).normalize();
            rec.interpolated_normal = normMat.transform_vector(rec.interpolated_normal).normalize();
        }
    }
    
    return hit_anything;
}
