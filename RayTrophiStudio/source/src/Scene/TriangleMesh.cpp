#include "TriangleMesh.h"
#include "ParallelBVHNode.h"
#include "Transform.h"
#include <algorithm>

TriangleMesh::TriangleMesh() {
    geometry = std::make_shared<DNA::GeometryDetail>();
}

void TriangleMesh::clear() {
    if (geometry) {
        geometry->indices.clear();
        geometry->clear_deltas();
        geometry->resize_vertices(0);
    }
    local_bvh = nullptr;
}

void TriangleMesh::build_local_bvh() {
    local_bvh = nullptr; 
}

bool TriangleMesh::bounding_box(float time0, float time1, AABB& output_box) const {
    if (!geometry || geometry->get_vertex_count() == 0) return false;
    
    const Vec3* positions = geometry->get_attribute_data<Vec3>("P");
    if (!positions) return false;
    
    Vec3 min_pt = positions[0];
    Vec3 max_pt = positions[0];
    
    size_t count = geometry->get_vertex_count();
    for (size_t i = 1; i < count; ++i) {
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
    if (!visible || !geometry || geometry->indices.empty()) return false;
    
    Ray local_r = r;
    if (transform) {
        Matrix4x4 inv = transform->getInverseMatrix();
        local_r.origin = inv.transform_point(r.origin);
        local_r.direction = inv.transform_vector(r.direction);
    }

    bool hit_anything = false;
    float closest_so_far = t_max;
    HitRecord temp_rec;
    size_t hit_face = 0;

    if (local_bvh) {
        // Reserved for future local BVH integration
    } else {
        const Vec3* positions = geometry->get_attribute_data<Vec3>("P");
        const Vec3* normals = geometry->get_attribute_data<Vec3>("N");
        const Vec2* uvs = geometry->get_attribute_data<Vec2>("uv");
        const uint16_t* materialIDs = geometry->get_attribute_data<uint16_t>("materialID");
        
        if (!positions) return false;

        size_t tri_count = geometry->indices.size() / 3;
        for (size_t i = 0; i < tri_count; ++i) {
            uint32_t i0 = geometry->indices[i * 3 + 0];
            uint32_t i1 = geometry->indices[i * 3 + 1];
            uint32_t i2 = geometry->indices[i * 3 + 2];
            
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
                hit_face = i;

                temp_rec.t = t;
                temp_rec.u = u;
                temp_rec.v = v;
                temp_rec.point = local_r.at(t);
                
                // Normal interpolation
                if (normals) {
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
                if (uvs) {
                    const Vec2& uv0 = uvs[i0];
                    const Vec2& uv1 = uvs[i1];
                    const Vec2& uv2 = uvs[i2];
                    temp_rec.uv = (1.0f - u - v) * uv0 + u * uv1 + v * uv2;
                }
                
                if (materialIDs) {
                    // materialID is per-VERTEX (not per-face); index by a corner vertex id.
                    temp_rec.materialID = materialIDs[i0];
                }
            }
        }
    }
    
    if (hit_anything) {
        rec = temp_rec;
        rec.tri_mesh = const_cast<TriangleMesh*>(this);   // Faz 1: (mesh, faceIndex) handle
        rec.tri_face = static_cast<uint32_t>(hit_face);
        rec.terrain_id = terrain_id;
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
