#include "Triangle.h"
#include "Ray.h"
#include "AABB.h"
#include "RayPacket.h"
#include "HitRecordPacket.h"
#include "globals.h"
#include <cmath>

// ============================================================================
// Constructors
// ============================================================================

Triangle::Triangle()
    : materialID(MaterialManager::INVALID_MATERIAL_ID)
    , faceIndex(-1)
    , aabbDirty(true)
{
    for (int i = 0; i < 3; ++i) {
        vertices[i] = TriangleVertexData();
        vertices[i].position = Vec3(0.0f);
        vertices[i].original = Vec3(0.0f); 
        vertices[i].normal = Vec3(0.0f, 1.0f, 0.0f); // Default normal Y-up
        vertices[i].originalNormal = Vec3(0.0f, 1.0f, 0.0f);
        vertices[i].color = Vec3(0.0f);
    }
}

Triangle::Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
                   const Vec3& na, const Vec3& nb, const Vec3& nc,
                   const Vec2& ta, const Vec2& tb, const Vec2& tc,
                   uint16_t matID)
    : materialID(matID)
    , t0(ta), t1(tb), t2(tc)
    , faceIndex(-1)
    , aabbDirty(true)
{
    // Initialize vertex data
    vertices[0].position = a;
    vertices[0].original = a;
    vertices[0].normal = na.normalize();
    vertices[0].originalNormal = na.normalize();

    vertices[1].position = b;
    vertices[1].original = b;
    vertices[1].normal = nb.normalize();
    vertices[1].originalNormal = nb.normalize();

    vertices[2].position = c;
    vertices[2].original = c;
    vertices[2].normal = nc.normalize();
    vertices[2].originalNormal = nc.normalize();
    
    // Default color black (Flexible)
    vertices[0].color = Vec3(0.0f);
    vertices[1].color = Vec3(0.0f);
    vertices[2].color = Vec3(0.0f);

    update_bounding_box();
}

// Legacy constructor with shared_ptr (for backward compatibility)
Triangle::Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
                   const Vec3& na, const Vec3& nb, const Vec3& nc,
                   const Vec2& ta, const Vec2& tb, const Vec2& tc,
                   std::shared_ptr<Material> m)
    : t0(ta), t1(tb), t2(tc)
    , faceIndex(-1)
    , aabbDirty(true)
{
    // Register material with MaterialManager and get ID
    if (m) {
        materialID = MaterialManager::getInstance().getOrCreateMaterialID(
            m->materialName.empty() ? "Material_" + std::to_string(reinterpret_cast<uintptr_t>(m.get())) : m->materialName,
            m
        );
    } else {
        materialID = MaterialManager::INVALID_MATERIAL_ID;
    }

    // Initialize vertex data
    vertices[0].position = a;
    vertices[0].original = a;
    vertices[0].normal = na.normalize();
    vertices[0].originalNormal = na.normalize();

    vertices[1].position = b;
    vertices[1].original = b;
    vertices[1].normal = nb.normalize();
    vertices[1].originalNormal = nb.normalize();

    vertices[2].position = c;
    vertices[2].original = c;
    vertices[2].normal = nc.normalize();
    vertices[2].originalNormal = nc.normalize();

    // Default color black (Flexible)
    vertices[0].color = Vec3(0.0f);
    vertices[1].color = Vec3(0.0f);
    vertices[2].color = Vec3(0.0f);

    update_bounding_box();
}

// ============================================================================
// Material Access
// ============================================================================

std::shared_ptr<Material> Triangle::getMaterial() const {
    return MaterialManager::getInstance().getMaterialShared(materialID);
}

void Triangle::setMaterial(const std::shared_ptr<Material>& mat) {
    if (mat) {
        materialID = MaterialManager::getInstance().getOrCreateMaterialID(
            mat->materialName.empty() ? "Material_" + std::to_string(reinterpret_cast<uintptr_t>(mat.get())) : mat->materialName,
            mat
        );
    } else {
        materialID = MaterialManager::INVALID_MATERIAL_ID;
    }
}

// ============================================================================
// UV Coordinates
// ============================================================================

void Triangle::setUVCoordinates(const Vec2& uv0, const Vec2& uv1, const Vec2& uv2) {
    t0 = uv0;
    t1 = uv1;
    t2 = uv2;
}

std::tuple<Vec2, Vec2, Vec2> Triangle::getUVCoordinates() const {
    return std::make_tuple(t0, t1, t2);
}

// ============================================================================
// Normals
// ============================================================================

void Triangle::set_normals(const Vec3& normal0, const Vec3& normal1, const Vec3& normal2) {
    vertices[0].normal = normal0.normalize();
    vertices[1].normal = normal1.normalize();
    vertices[2].normal = normal2.normalize();
}

// ============================================================================
// Transform Management
// ============================================================================

void Triangle::set_transform(const Matrix4x4& t) {
    if (transformHandle) {
        transformHandle->setCurrent(t);
    }
    aabbDirty = true;
    update_bounding_box();
}

void Triangle::updateTriangleTransform(Triangle& triangle, const Matrix4x4& transform) {
    triangle.set_transform(transform);
}

Matrix4x4 Triangle::getTransformMatrix() const {
    if (transformHandle) {
        return transformHandle->getFinal();
    }
    return Matrix4x4::identity();
}

void Triangle::update_bounding_box() const {
    if (!aabbDirty) return;

    // Use current vertex positions
    const Vec3& tv0 = vertices[0].position;
    const Vec3& tv1 = vertices[1].position;
    const Vec3& tv2 = vertices[2].position;

    Vec3 minPoint(
        std::min({ tv0.x, tv1.x, tv2.x }),
        std::min({ tv0.y, tv1.y, tv2.y }),
        std::min({ tv0.z, tv1.z, tv2.z })
    );
    Vec3 maxPoint(
        std::max({ tv0.x, tv1.x, tv2.x }),
        std::max({ tv0.y, tv1.y, tv2.y }),
        std::max({ tv0.z, tv1.z, tv2.z })
    );

    constexpr float DELTA = 1e-5f;
    cachedAABB = AABB(minPoint - DELTA, maxPoint + DELTA);
    aabbDirty = false;
}

// ============================================================================
// Skinning
// ============================================================================

void Triangle::initializeSkinData() {
    if (!skinData.has_value()) {
        skinData = SkinnedTriangleData();
        skinData->vertexBoneWeights.resize(3);
        skinData->originalVertexPositions.resize(3);
        
        // Copy original positions
        skinData->originalVertexPositions[0] = vertices[0].original;
        skinData->originalVertexPositions[1] = vertices[1].original;
        skinData->originalVertexPositions[2] = vertices[2].original;
    }
}

void Triangle::setSkinBoneWeights(int vertexIndex, const std::vector<std::pair<int, float>>& weights) {
    initializeSkinData();
    if (vertexIndex >= 0 && vertexIndex < 3) {
        skinData->vertexBoneWeights[vertexIndex] = weights;
    }
}

const std::vector<std::pair<int, float>>& Triangle::getSkinBoneWeights(int vertexIndex) const {
    static const std::vector<std::pair<int, float>> empty;
    if (skinData.has_value() && vertexIndex >= 0 && vertexIndex < 3) {
        return skinData->vertexBoneWeights[vertexIndex];
    }
    return empty;
}

// Legacy access (for gradual migration - returns reference to temporary)
std::vector<std::vector<std::pair<int, float>>>& Triangle::getVertexBoneWeights() {
    initializeSkinData();
    return skinData->vertexBoneWeights;
}

std::vector<Vec3>& Triangle::getOriginalVertexPositions() {
    initializeSkinData();
    return skinData->originalVertexPositions;
}

Vec3 Triangle::apply_bone_to_vertex(int vi, const std::vector<Matrix4x4>& finalBoneMatrices) const {
    if (!hasSkinData()) return vertices[vi].position;
    
    const auto& boneWeights = skinData->vertexBoneWeights[vi];
    const auto& origPosition = skinData->originalVertexPositions[vi];
    
    Vec3 blended = Vec3(0);
    for (const auto& [boneIdx, weight] : boneWeights) {
        Vec3 transformed = finalBoneMatrices[boneIdx].transform_point(origPosition);
        blended += transformed * weight;
    }
    return blended;
}

void Triangle::apply_skinning(const std::vector<Matrix4x4>& finalBoneMatrices) {
    if (!hasSkinData()) return;

    const auto& boneWeights = skinData->vertexBoneWeights;
    const auto& origPositions = skinData->originalVertexPositions;

    // Validation check
    if (boneWeights.size() != 3 || origPositions.size() != 3 ||
        vertices[0].originalNormal.length_squared() < 1e-6 || 
        vertices[1].originalNormal.length_squared() < 1e-6 || 
        vertices[2].originalNormal.length_squared() < 1e-6) {

        for (int i = 0; i < 3; ++i) {
            vertices[i].position = vertices[i].original;
            vertices[i].normal = vertices[i].originalNormal;
        }
        return;
    }

    // Process each vertex
    for (int vi = 0; vi < 3; ++vi) {
        if (boneWeights[vi].empty()) {
            vertices[vi].position = vertices[vi].original;
            vertices[vi].normal = vertices[vi].originalNormal;
            continue;
        }

        // Linear Blend Skinning (standard approach)
        Matrix4x4 blendedBoneMatrix = Matrix4x4::zero();
        
        for (const auto& [boneIdx, weight] : boneWeights[vi]) {
            if (boneIdx >= static_cast<int>(finalBoneMatrices.size()) || weight < 1e-6f) {
                continue;
            }
            
            // Blend bone matrices
            const Matrix4x4& boneMatrix = finalBoneMatrices[boneIdx];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    blendedBoneMatrix.m[i][j] += boneMatrix.m[i][j] * weight;
                }
            }
        }

        // Apply to vertex
        vertices[vi].position = blendedBoneMatrix.transform_point(origPositions[vi]);
        
        // Apply to normal
        Matrix4x4 normalMatrix = blendedBoneMatrix.inverse().transpose();
        vertices[vi].normal = normalMatrix.transform_vector(vertices[vi].originalNormal).normalize();
    }

    aabbDirty = true;
}

Vec3 Triangle::apply_bone_to_normal(const Vec3& originalNormal,
                                     const std::vector<std::pair<int, float>>& boneWeights,
                                     const std::vector<Matrix4x4>& finalBoneMatrices) const {
    Vec3 blended = Vec3(0);
    for (const auto& [boneIdx, weight] : boneWeights) {
        Matrix4x4 normalMat = finalBoneMatrices[boneIdx].inverse().transpose();
        Vec3 transformed = normalMat.transform_vector(originalNormal);
        blended += transformed * weight;
    }
    return blended.normalize();
}

// ============================================================================
// Hit Detection
// ============================================================================

bool Triangle::hit(const Ray& r, float t_min, float t_max, HitRecord& rec, bool ignore_volumes) const {
    if (!visible) return false;

    // Use current vertex positions (already transformed)
    const Vec3& v0 = vertices[0].position;
    const Vec3& v1 = vertices[1].position;
    const Vec3& v2 = vertices[2].position;
    
    const Vec3 edge1 = v1 - v0;
    const Vec3 edge2 = v2 - v0;

    Vec3 h = Vec3::cross(r.direction, edge2);
    float a = Vec3::dot(edge1, h);

    // Parallelism check
    if (fabs(a) < EPSILON)
        return false;

    float f = 1.0f / a;
    Vec3 s = r.origin - v0;
    float u = f * Vec3::dot(s, h);

    // Tolerant boundaries
    if (u < -EPSILON || u > 1.0f + EPSILON)
        return false;

    Vec3 q = Vec3::cross(s, edge1);
    float v = f * Vec3::dot(r.direction, q);

    if (v < -EPSILON || (u + v) > 1.0f + EPSILON)
        return false;

    float t = f * Vec3::dot(edge2, q);

    // Check if in front of ray
    if (t < t_min || t > t_max)
        return false;

    // Fill hit record
    rec.triangle = this;
    rec.t = t;
    rec.point = r.at(t);

    // Barycentric weights (u, v, w)
    float w = 1.0f - u - v;

    // Interpolate normal directly with barycentric coordinates
    rec.interpolated_normal =
        (w * vertices[0].normal + u * vertices[1].normal + v * vertices[2].normal).normalize();

    rec.normal = rec.interpolated_normal;

    // Use u, v, w directly for UV interpolation
    Vec2 uv = w * t0 + u * t1 + v * t2;

    rec.uv = uv;
    rec.u = uv.u;
    rec.v = uv.v;
    
    // Set material from MaterialManager
    rec.material = MaterialManager::getInstance().getMaterialShared(materialID);
    rec.materialID = materialID;

    // --- ALPHA TESTING ---
    if (rec.material) {
        float opacity = rec.material->get_opacity(rec.uv);
        if (Vec3::random_float() > opacity) {
            return false; // Transparent hit, ignore and let BVH continue
        }
    }

    return true;
}

// ============================================================================
// Transform Initialization & Updates
// ============================================================================

void Triangle::updateTransformedVertices() {
    Matrix4x4 finalTransform = getTransformMatrix();
    Matrix4x4 normalTransform = finalTransform.inverse().transpose();

    for (int i = 0; i < 3; ++i) {
        vertices[i].position = finalTransform.transform_point(vertices[i].original);
        vertices[i].normal = normalTransform.transform_vector(vertices[i].originalNormal).normalize();
    }

    aabbDirty = true;
    update_bounding_box();
}

void Triangle::setNodeName(const std::string& name) {
    nodeName = name;
}

void Triangle::setBaseTransform(const Matrix4x4& t) {
    if (transformHandle) {
        transformHandle->setBase(t);
    }
    updateTransformedVertices();
}

void Triangle::initialize_transforms() {
    // Store original vertices
    for (int i = 0; i < 3; ++i) {
        vertices[i].original = vertices[i].position;
        vertices[i].originalNormal = vertices[i].normal.normalize();
    }
}

void Triangle::updateAnimationTransform(const Matrix4x4& animTransform) {
    if (transformHandle) {
        transformHandle->setCurrent(animTransform);
    }
    
    updateTransformedVertices();
}

bool Triangle::bounding_box(float time0, float time1, AABB& output_box) const {
    update_bounding_box();
    output_box = cachedAABB;
    return true;
}

// ============================================================================
// Packet Tracing Implementation (Phase 2)
// ============================================================================

void Triangle::hit_packet(const RayPacket& r, float t_min, float t_max, HitRecordPacket& rec, bool ignore_volumes) const {
    if (!visible) return;

    // 1. Load Triangle Vertices (Broadcast to all 8 lanes)
    Vec3SIMD v0_x(_mm256_set1_ps(vertices[0].position.x));
    Vec3SIMD v0_y(_mm256_set1_ps(vertices[0].position.y));
    Vec3SIMD v0_z(_mm256_set1_ps(vertices[0].position.z));

    Vec3SIMD v1_x(_mm256_set1_ps(vertices[1].position.x));
    Vec3SIMD v1_y(_mm256_set1_ps(vertices[1].position.y));
    Vec3SIMD v1_z(_mm256_set1_ps(vertices[1].position.z));

    Vec3SIMD v2_x(_mm256_set1_ps(vertices[2].position.x));
    Vec3SIMD v2_y(_mm256_set1_ps(vertices[2].position.y));
    Vec3SIMD v2_z(_mm256_set1_ps(vertices[2].position.z));

    // 2. Edge Vectors
    Vec3SIMD e1_x = v1_x - v0_x;
    Vec3SIMD e1_y = v1_y - v0_y;
    Vec3SIMD e1_z = v1_z - v0_z;

    Vec3SIMD e2_x = v2_x - v0_x;
    Vec3SIMD e2_y = v2_y - v0_y;
    Vec3SIMD e2_z = v2_z - v0_z;

    // 3. Möller–Trumbore
    Vec3SIMD h_x, h_y, h_z;
    Vec3SIMD::cross_8x(r.dir_x, r.dir_y, r.dir_z, e2_x, e2_y, e2_z, h_x, h_y, h_z);

    __m256 a = Vec3SIMD::dot_product_8x(e1_x, e1_y, e1_z, h_x, h_y, h_z);

    const float EPS = 1e-8f;
    __m256 is_parallel = _mm256_and_ps(
        _mm256_cmp_ps(a, _mm256_set1_ps(-EPS), _CMP_GT_OQ),
        _mm256_cmp_ps(a, _mm256_set1_ps(EPS), _CMP_LT_OQ)
    );

    __m256 f = _mm256_div_ps(_mm256_set1_ps(1.0f), a);
    
    Vec3SIMD s_x = r.orig_x - v0_x;
    Vec3SIMD s_y = r.orig_y - v0_y;
    Vec3SIMD s_z = r.orig_z - v0_z;

    __m256 u = _mm256_mul_ps(f, Vec3SIMD::dot_product_8x(s_x, s_y, s_z, h_x, h_y, h_z));
    
    __m256 bad_u = _mm256_or_ps(
        _mm256_cmp_ps(u, _mm256_setzero_ps(), _CMP_LT_OQ),
        _mm256_cmp_ps(u, _mm256_set1_ps(1.0f), _CMP_GT_OQ)
    );

    Vec3SIMD q_x, q_y, q_z;
    Vec3SIMD::cross_8x(s_x, s_y, s_z, e1_x, e1_y, e1_z, q_x, q_y, q_z);

    __m256 v = _mm256_mul_ps(f, Vec3SIMD::dot_product_8x(r.dir_x, r.dir_y, r.dir_z, q_x, q_y, q_z));
    
    __m256 bad_v = _mm256_or_ps(
        _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_LT_OQ),
        _mm256_cmp_ps(_mm256_add_ps(u, v), _mm256_set1_ps(1.0f), _CMP_GT_OQ)
    );

    __m256 t_val = _mm256_mul_ps(f, Vec3SIMD::dot_product_8x(e2_x, e2_y, e2_z, q_x, q_y, q_z));
    
    __m256 bad_t = _mm256_or_ps(
        _mm256_cmp_ps(t_val, _mm256_set1_ps(t_min), _CMP_LT_OQ),
        _mm256_cmp_ps(t_val, _mm256_set1_ps(t_max), _CMP_GT_OQ)
    );

    __m256 failed = _mm256_or_ps(is_parallel, _mm256_or_ps(bad_u, _mm256_or_ps(bad_v, bad_t)));
    __m256 success_mask = _mm256_andnot_ps(failed, _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ));

    // Result assembly
    HitRecordPacket tr;
    tr.t.data = t_val;
    tr.u.data = u;
    tr.v.data = v;
    
    r.point_at(tr.t, tr.p_x, tr.p_y, tr.p_z);

    __m256 w = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(u, v));
    
    auto splt = [&](int i, int axis) {
        if (axis == 0) return _mm256_set1_ps(vertices[i].normal.x);
        if (axis == 1) return _mm256_set1_ps(vertices[i].normal.y);
        return _mm256_set1_ps(vertices[i].normal.z);
    };

    tr.normal_x.data = _mm256_add_ps(_mm256_mul_ps(w, splt(0, 0)), _mm256_add_ps(_mm256_mul_ps(u, splt(1, 0)), _mm256_mul_ps(v, splt(2, 0))));
    tr.normal_y.data = _mm256_add_ps(_mm256_mul_ps(w, splt(0, 1)), _mm256_add_ps(_mm256_mul_ps(u, splt(1, 1)), _mm256_mul_ps(v, splt(2, 1))));
    tr.normal_z.data = _mm256_add_ps(_mm256_mul_ps(w, splt(0, 2)), _mm256_add_ps(_mm256_mul_ps(u, splt(1, 2)), _mm256_mul_ps(v, splt(2, 2))));

    Vec3SIMD::normalize_8x(tr.normal_x, tr.normal_y, tr.normal_z, tr.normal_x, tr.normal_y, tr.normal_z);

    // --- FRONT FACE & NORMAL FLIPPING ---
    __m256 n_dot_r = Vec3SIMD::dot_product_8x(tr.normal_x, tr.normal_y, tr.normal_z, r.dir_x, r.dir_y, r.dir_z);
    tr.front_face = _mm256_cmp_ps(n_dot_r, _mm256_setzero_ps(), _CMP_LT_OQ);
    
    // If not front face, flip normal (Double-sided support)
    tr.normal_x.data = _mm256_blendv_ps(_mm256_sub_ps(_mm256_setzero_ps(), tr.normal_x.data), tr.normal_x.data, tr.front_face);
    tr.normal_y.data = _mm256_blendv_ps(_mm256_sub_ps(_mm256_setzero_ps(), tr.normal_y.data), tr.normal_y.data, tr.front_face);
    tr.normal_z.data = _mm256_blendv_ps(_mm256_sub_ps(_mm256_setzero_ps(), tr.normal_z.data), tr.normal_z.data, tr.front_face);

    // --- ALPHA TESTING ---
    Material* mat = MaterialManager::getInstance().getMaterial(materialID);
    if (mat) {
        alignas(32) float opacity_vals[8];
        bool has_transparency = false;
        for (int i = 0; i < 8; i++) {
            if ((1 << i) & _mm256_movemask_ps(success_mask)) {
                float op = mat->get_opacity(Vec2(((float*)&tr.u.data)[i], ((float*)&tr.v.data)[i]));
                opacity_vals[i] = op;
                if (op < 0.99f) has_transparency = true;
            } else {
                opacity_vals[i] = 1.0f;
            }
        }

        if (has_transparency) {
            __m256 opacity_v = _mm256_load_ps(opacity_vals);
            __m256 rand_v = Vec3SIMD::random_float_8x();
            __m256 alpha_pass = _mm256_cmp_ps(rand_v, opacity_v, _CMP_LT_OQ);
            success_mask = _mm256_and_ps(success_mask, alpha_pass);
        }
    }

    tr.mat_id = _mm256_set1_epi32((int)materialID);
    rec.merge_if_closer(tr, success_mask);
}

bool Triangle::occluded(const Ray& ray, float t_min, float t_max) const {
    HitRecord dummy;
    // We already updated Triangle::hit to handle alpha test.
    // So current base implementation (which calls hit) is actually correct.
    // But for clarity and potentially skipping some HitRecord work:
    return hit(ray, t_min, t_max, dummy);
}

__m256 Triangle::occluded_packet(const RayPacket& r, float t_min, __m256 t_max) const {
    if (!visible) return _mm256_setzero_ps();

    // Re-use hit_packet logic basically, but just return success_mask
    // Load Triangle Vertices 
    Vec3SIMD v0_x(_mm256_set1_ps(vertices[0].position.x)), v0_y(_mm256_set1_ps(vertices[0].position.y)), v0_z(_mm256_set1_ps(vertices[0].position.z));
    Vec3SIMD v1_x(_mm256_set1_ps(vertices[1].position.x)), v1_y(_mm256_set1_ps(vertices[1].position.y)), v1_z(_mm256_set1_ps(vertices[1].position.z));
    Vec3SIMD v2_x(_mm256_set1_ps(vertices[2].position.x)), v2_y(_mm256_set1_ps(vertices[2].position.y)), v2_z(_mm256_set1_ps(vertices[2].position.z));

    Vec3SIMD e1_x = v1_x - v0_x, e1_y = v1_y - v0_y, e1_z = v1_z - v0_z;
    Vec3SIMD e2_x = v2_x - v0_x, e2_y = v2_y - v0_y, e2_z = v2_z - v0_z;

    Vec3SIMD h_x, h_y, h_z;
    Vec3SIMD::cross_8x(r.dir_x, r.dir_y, r.dir_z, e2_x, e2_y, e2_z, h_x, h_y, h_z);
    __m256 a = Vec3SIMD::dot_product_8x(e1_x, e1_y, e1_z, h_x, h_y, h_z);
    const float EPS = 1e-8f;
    __m256 is_parallel = _mm256_and_ps(_mm256_cmp_ps(a, _mm256_set1_ps(-EPS), _CMP_GT_OQ), _mm256_cmp_ps(a, _mm256_set1_ps(EPS), _CMP_LT_OQ));
    __m256 f = _mm256_div_ps(_mm256_set1_ps(1.0f), a);
    
    Vec3SIMD s_x = r.orig_x - v0_x, s_y = r.orig_y - v0_y, s_z = r.orig_z - v0_z;
    __m256 u = _mm256_mul_ps(f, Vec3SIMD::dot_product_8x(s_x, s_y, s_z, h_x, h_y, h_z));
    __m256 bad_u = _mm256_or_ps(_mm256_cmp_ps(u, _mm256_setzero_ps(), _CMP_LT_OQ), _mm256_cmp_ps(u, _mm256_set1_ps(1.0f), _CMP_GT_OQ));

    Vec3SIMD q_x, q_y, q_z;
    Vec3SIMD::cross_8x(s_x, s_y, s_z, e1_x, e1_y, e1_z, q_x, q_y, q_z);
    __m256 v = _mm256_mul_ps(f, Vec3SIMD::dot_product_8x(r.dir_x, r.dir_y, r.dir_z, q_x, q_y, q_z));
    __m256 bad_v = _mm256_or_ps(_mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_LT_OQ), _mm256_cmp_ps(_mm256_add_ps(u, v), _mm256_set1_ps(1.0f), _CMP_GT_OQ));
    __m256 t_val = _mm256_mul_ps(f, Vec3SIMD::dot_product_8x(e2_x, e2_y, e2_z, q_x, q_y, q_z));
    
    __m256 bad_t = _mm256_or_ps(_mm256_cmp_ps(t_val, _mm256_set1_ps(t_min), _CMP_LT_OQ), _mm256_cmp_ps(t_val, t_max, _CMP_GT_OQ));
    __m256 success_mask = _mm256_andnot_ps(_mm256_or_ps(is_parallel, _mm256_or_ps(bad_u, _mm256_or_ps(bad_v, bad_t))), _mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ));

    // ALPHA TEST
    if (_mm256_movemask_ps(success_mask)) {
        Material* mat = MaterialManager::getInstance().getMaterial(materialID);
        if (mat) {
            alignas(32) float opacity_vals[8];
            for (int i = 0; i < 8; i++) {
                if ((1 << i) & _mm256_movemask_ps(success_mask)) {
                    opacity_vals[i] = mat->get_opacity(Vec2(((float*)&u)[i], ((float*)&v)[i]));
                } else opacity_vals[i] = 1.0f;
            }
            __m256 alpha_pass = _mm256_cmp_ps(Vec3SIMD::random_float_8x(), _mm256_load_ps(opacity_vals), _CMP_LT_OQ);
            success_mask = _mm256_and_ps(success_mask, alpha_pass);
        }
    }

    return success_mask;
}
