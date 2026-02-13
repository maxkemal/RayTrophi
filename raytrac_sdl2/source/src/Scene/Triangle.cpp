#include "Triangle.h"
#include "Ray.h"
#include "AABB.h"
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

    // Get root transform (Gizmo placement in scene)
    Matrix4x4 rootTransform = getTransformMatrix();
    Matrix4x4 rootNormalTransform = rootTransform.inverse().transpose();

    // Process each vertex
    for (int vi = 0; vi < 3; ++vi) {
        if (boneWeights[vi].empty()) {
            vertices[vi].position = rootTransform.transform_point(vertices[vi].original);
            vertices[vi].normal = rootNormalTransform.transform_vector(vertices[vi].originalNormal).normalize();
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

        // Apply Bone Skinning -> Then Apply Root Transform 
        Vec3 localSkinnedPos = blendedBoneMatrix.transform_point(origPositions[vi]);
        vertices[vi].position = rootTransform.transform_point(localSkinnedPos);
        
        // Apply to normal
        Matrix4x4 nodeNormalMatrix = blendedBoneMatrix.inverse().transpose();
        Vec3 localSkinnedNormal = nodeNormalMatrix.transform_vector(vertices[vi].originalNormal).normalize();
        vertices[vi].normal = rootNormalTransform.transform_vector(localSkinnedNormal).normalize();
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

    // Ensure normal always points against ray and set front_face flag
    rec.set_face_normal(r, rec.interpolated_normal);

    // Use u, v, w directly for UV interpolation
    Vec2 uv = w * t0 + u * t1 + v * t2;

    rec.uv = uv;
    rec.u = uv.u;
    rec.v = uv.v;
    
    // --- HIGH PERFORMANCE MATERIAL ACCESS & ALPHA TESTING ---
    if (!cachedMaterial) {
        cachedMaterial = MaterialManager::getInstance().getMaterial(materialID);
    }

    // --- ALPHA TESTING ---
    if (cachedMaterial && cachedMaterial->isTransparent()) {
        float opacity = cachedMaterial->get_opacity(rec.uv);
        if (Vec3::random_float() > opacity) {
            return false; // Transparent hit, ignore and let BVH continue
        }
    }

    rec.materialPtr = cachedMaterial;
    rec.material = MaterialManager::getInstance().getMaterialShared(materialID);
    rec.materialID = materialID;
    rec.terrain_id = terrain_id;

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


bool Triangle::occluded(const Ray& ray, float t_min, float t_max) const {
    HitRecord dummy;
    // We already updated Triangle::hit to handle alpha test.
    // So current base implementation (which calls hit) is actually correct.
    // But for clarity and potentially skipping some HitRecord work:
    return hit(ray, t_min, t_max, dummy);
}


