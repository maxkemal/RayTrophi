#include "TriangleMesh.h"
#include "ParallelBVHNode.h"
#include "Transform.h"
#include "MeshPointiness.h"
#include "Matrix4x4.h"
#include "globals.h"
#include <algorithm>
#include <cstring>

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

bool TriangleMesh::hasSkinWeights() const {
    if (!geometry || geometry->skin_weights.empty()) return false;
    for (const auto& weights : geometry->skin_weights) {
        if (!weights.empty()) return true;
    }
    return false;
}

bool TriangleMesh::applySkinning(const std::vector<Matrix4x4>& finalBoneMatrices) {
    if (!geometry || finalBoneMatrices.empty() || !hasSkinWeights()) return false;

    // Pose hash prevents the same shared mesh from being deformed repeatedly when
    // several editor/runtime references point at it during the facade migration.
    uint64_t hash = 1469598103934665603ull;
    hash ^= static_cast<uint64_t>(finalBoneMatrices.size());
    hash *= 1099511628211ull;
    for (const Matrix4x4& matrix : finalBoneMatrices) {
        const float* values = &matrix.m[0][0];
        for (int i = 0; i < 16; ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, &values[i], sizeof(bits));
            hash ^= bits;
            hash *= 1099511628211ull;
        }
    }
    if (geometry->last_skinned_pose_hash == hash) return false;

    const size_t vertexCount = geometry->get_vertex_count();
    if (geometry->skin_weights.size() < vertexCount) return false;

    const Vec3* bindPositions = geometry->get_positions_orig();
    const Vec3* bindNormals = geometry->get_normals_orig();
    if (!bindPositions) bindPositions = geometry->get_positions();
    if (!bindNormals) bindNormals = geometry->get_normals();
    Vec3* positions = geometry->get_positions_mut();
    Vec3* normals = geometry->get_normals_mut();
    if (!bindPositions || !positions) return false;

    // P/N remain in mesh-local space. TriangleMesh::hit and GPU TLAS instances
    // apply the object transform exactly once; baking it here would double-transform
    // a skinned flat mesh.
    #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static)
    for (int vertex = 0; vertex < static_cast<int>(vertexCount); ++vertex) {
        const auto& weights = geometry->skin_weights[static_cast<size_t>(vertex)];
        if (weights.empty()) {
            positions[vertex] = bindPositions[vertex];
            if (normals && bindNormals) normals[vertex] = bindNormals[vertex];
            continue;
        }

        Matrix4x4 blended = Matrix4x4::zero();
        float totalWeight = 0.0f;
        for (const auto& [boneIndex, weight] : weights) {
            if (boneIndex >= 0 && boneIndex < static_cast<int>(finalBoneMatrices.size()) && weight > 1e-7f) {
                totalWeight += weight;
            }
        }
        if (totalWeight < 1e-5f) {
            positions[vertex] = bindPositions[vertex];
            if (normals && bindNormals) normals[vertex] = bindNormals[vertex];
            continue;
        }

        const float invWeight = 1.0f / totalWeight;
        for (const auto& [boneIndex, weight] : weights) {
            if (boneIndex < 0 || boneIndex >= static_cast<int>(finalBoneMatrices.size()) || weight <= 1e-7f) continue;
            const float normalizedWeight = weight * invWeight;
            const Matrix4x4& boneMatrix = finalBoneMatrices[static_cast<size_t>(boneIndex)];
            for (int row = 0; row < 4; ++row) {
                for (int column = 0; column < 4; ++column) {
                    blended.m[row][column] += boneMatrix.m[row][column] * normalizedWeight;
                }
            }
        }

        positions[vertex] = blended.transform_point(bindPositions[vertex]);
        if (normals && bindNormals) {
            normals[vertex] = blended.inverse().transpose().transform_vector(bindNormals[vertex]).normalize();
        }
    }

    geometry->last_skinned_pose_hash = hash;
    return true;
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
        // temp_rec.u/.v still hold the BARYCENTRIC pair here (the UV lands in temp_rec.uv).
        rec.pointiness = MeshAttr::samplePointiness(this, static_cast<uint32_t>(hit_face),
                                                    1.0f - temp_rec.u - temp_rec.v,
                                                    temp_rec.u, temp_rec.v);
        MeshAttr::sampleMaterialAttributes(this, static_cast<uint32_t>(hit_face),
                                           1.0f - temp_rec.u - temp_rec.v,
                                           temp_rec.u, temp_rec.v, rec.mat_attrib);
        rec.object_origin = MeshAttr::objectOrigin(transform.get());
        rec.object_position = rec.point;
        MeshAttr::sampleObjectPosition(this, static_cast<uint32_t>(hit_face),
                                       1.0f - temp_rec.u - temp_rec.v,
                                       temp_rec.u, temp_rec.v, rec.object_position);
        // Fresnel / Layer Weight. `r` is the WORLD ray (local_r is its object-space copy,
        // used for the intersection), so this vector is already in the space the VM wants.
        rec.view_dir = (-r.direction).normalize();
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
