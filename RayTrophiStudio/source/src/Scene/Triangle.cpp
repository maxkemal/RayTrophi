#include "Triangle.h"
#include "Ray.h"
#include "AABB.h"
#include "globals.h"
#include <cmath>
#include <mutex>

// File-scope mutexes to protect lazy initialization of P_orig and N_orig attributes.
// Declared at file-scope to avoid MSVC thread-safe local static (magic static) deadlocks
// inside OpenMP parallel regions.
static std::mutex g_original_position_init_mutex;
static std::mutex g_original_normal_init_mutex;

// ============================================================================
// Constructors
// ============================================================================

Triangle::Triangle()
    : materialID(MaterialManager::INVALID_MATERIAL_ID), parentMesh(nullptr), faceIndex(0)
{
    t0 = Vec2(0.0f, 0.0f);
    t1 = Vec2(0.0f, 0.0f);
    t2 = Vec2(0.0f, 0.0f);
    auto* sv = svw();
    for (int i = 0; i < 3; ++i) {
        sv[i].position = Vec3(0.0f);
        sv[i].normal = Vec3(0.0f, 1.0f, 0.0f); // Default normal Y-up
    }
}

Triangle::Triangle(std::shared_ptr<TriangleMesh> parent, uint32_t faceIdx)
    : parentMesh(parent), faceIndex(faceIdx)
{
    update_bounding_box();
}

Triangle::Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
                   const Vec3& na, const Vec3& nb, const Vec3& nc,
                   const Vec2& ta, const Vec2& tb, const Vec2& tc,
                   uint16_t matID)
    : materialID(matID)
{
    t0 = ta;
    t1 = tb;
    t2 = tc;
    
    // Initialize vertex data
    auto* sv = svw();
    sv[0].position = a;
    sv[0].normal = na.normalize();

    sv[1].position = b;
    sv[1].normal = nb.normalize();

    sv[2].position = c;
    sv[2].normal = nc.normalize();

    update_bounding_box();
}



// Legacy constructor with shared_ptr (for backward compatibility)
Triangle::Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
                   const Vec3& na, const Vec3& nb, const Vec3& nc,
                   const Vec2& ta, const Vec2& tb, const Vec2& tc,
                   std::shared_ptr<Material> m)
{
    t0 = ta;
    t1 = tb;
    t2 = tc;

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
    auto* sv = svw();
    sv[0].position = a;
    sv[0].normal = na.normalize();

    sv[1].position = b;
    sv[1].normal = nb.normalize();

    sv[2].position = c;
    sv[2].normal = nc.normalize();

    // UV set 0 is t0/t1/t2; uv_sets stays empty until a SECOND set is added (no per-tri heap alloc).

    update_bounding_box();
}

// ============================================================================
// Material Access
// ============================================================================

std::shared_ptr<Material> Triangle::getMaterial() const {
    return MaterialManager::getInstance().getMaterialShared(getMaterialID());
}

void Triangle::setMaterial(const std::shared_ptr<Material>& mat) {
    uint16_t id = MaterialManager::INVALID_MATERIAL_ID;
    if (mat) {
        id = MaterialManager::getInstance().getOrCreateMaterialID(
            mat->materialName.empty() ? "Material_" + std::to_string(reinterpret_cast<uintptr_t>(mat.get())) : mat->materialName,
            mat
        );
    }
    setMaterialID(id);
}

// ============================================================================
// UV Coordinates
// ============================================================================

// UV-set storage model (memory slim):
//   * Set 0 always lives in t0/t1/t2.
//   * legacyData->uv_sets is EMPTY for single-set triangles (the overwhelming majority, incl. every
//     procedurally subdivided mesh) — no per-triangle heap allocation.
//   * The first time an EXTRA set (index >= 1) is authored, legacyData->uv_sets is materialized as the
//     FULL list (legacyData->uv_sets[0] = current t0/t1/t2 backup), after which it mirrors the legacy
//     layout so active-set switching (applyUVSet) round-trips losslessly.
void Triangle::setUVCoordinates(const Vec2& uv0, const Vec2& uv1, const Vec2& uv2) {
    if (parentMesh && parentMesh->geometry) {
        Vec2* uvs = parentMesh->geometry->get_attribute_data_mut<Vec2>("uv");
        if (uvs) {
            uint32_t i0 = parentMesh->geometry->indices[faceIndex * 3 + 0];
            uint32_t i1 = parentMesh->geometry->indices[faceIndex * 3 + 1];
            uint32_t i2 = parentMesh->geometry->indices[faceIndex * 3 + 2];
            uvs[i0] = uv0;
            uvs[i1] = uv1;
            uvs[i2] = uv2;
            return;
        }
    }
    t0 = uv0;
    t1 = uv1;
    t2 = uv2;
    if (uv_sets && !uv_sets->empty()) {
        (*uv_sets)[0] = {uv0, uv1, uv2};
    }
}

std::tuple<Vec2, Vec2, Vec2> Triangle::getUVCoordinates() const {
    if (parentMesh && parentMesh->geometry) {
        const Vec2* uvs = parentMesh->geometry->get_attribute_data<Vec2>("uv");
        if (uvs) {
            uint32_t i0 = parentMesh->geometry->indices[faceIndex * 3 + 0];
            uint32_t i1 = parentMesh->geometry->indices[faceIndex * 3 + 1];
            uint32_t i2 = parentMesh->geometry->indices[faceIndex * 3 + 2];
            return std::make_tuple(uvs[i0], uvs[i1], uvs[i2]);
        }
    }
    return std::make_tuple(t0, t1, t2);
}

void Triangle::setUVSetCoordinates(size_t set_index, const Vec2& uv0, const Vec2& uv1, const Vec2& uv2) {
    if (parentMesh && parentMesh->geometry) {
        std::string attrName = (set_index == 0) ? "uv" : "uv" + std::to_string(set_index);
        if (!parentMesh->geometry->has_attribute(attrName)) {
            parentMesh->geometry->add_attribute<Vec2>(attrName);
        }
        Vec2* uvs = parentMesh->geometry->get_attribute_data_mut<Vec2>(attrName);
        if (uvs) {
            uint32_t i0 = parentMesh->geometry->indices[faceIndex * 3 + 0];
            uint32_t i1 = parentMesh->geometry->indices[faceIndex * 3 + 1];
            uint32_t i2 = parentMesh->geometry->indices[faceIndex * 3 + 2];
            uvs[i0] = uv0;
            uvs[i1] = uv1;
            uvs[i2] = uv2;
            return;
        }
    }
    if (set_index == 0 && (!uv_sets || uv_sets->empty())) {
        // Single-set fast path: set 0 is t0/t1/t2, no allocation.
        t0 = uv0; t1 = uv1; t2 = uv2;
        return;
    }
    if (!uv_sets) {
        uv_sets.reset(new std::vector<std::array<Vec2, 3>>());
    }
    if (uv_sets->empty()) {
        // Promote to multi-set: materialize the existing set 0 from t0/t1/t2 first.
        uv_sets->push_back({t0, t1, t2});
    }
    if (uv_sets->size() <= set_index) {
        uv_sets->resize(set_index + 1, {Vec2(0.0f, 0.0f), Vec2(0.0f, 0.0f), Vec2(0.0f, 0.0f)});
    }
    (*uv_sets)[set_index] = {uv0, uv1, uv2};
    if (set_index == 0) {
        t0 = uv0; t1 = uv1; t2 = uv2;   // keep t0/t1/t2 mirroring set 0
    }
}

std::tuple<Vec2, Vec2, Vec2> Triangle::getUVSetCoordinates(size_t set_index) const {
    if (parentMesh && parentMesh->geometry) {
        std::string attrName = (set_index == 0) ? "uv" : "uv" + std::to_string(set_index);
        const Vec2* uvs = parentMesh->geometry->get_attribute_data<Vec2>(attrName);
        if (uvs) {
            uint32_t i0 = parentMesh->geometry->indices[faceIndex * 3 + 0];
            uint32_t i1 = parentMesh->geometry->indices[faceIndex * 3 + 1];
            uint32_t i2 = parentMesh->geometry->indices[faceIndex * 3 + 2];
            return std::make_tuple(uvs[i0], uvs[i1], uvs[i2]);
        }
    }
    if (!uv_sets || uv_sets->empty()) {
        return std::make_tuple(t0, t1, t2);   // only set 0 exists
    }
    if (set_index < uv_sets->size()) {
        const auto& uv_set = (*uv_sets)[set_index];
        return std::make_tuple(uv_set[0], uv_set[1], uv_set[2]);
    }
    return std::make_tuple(t0, t1, t2);
}

size_t Triangle::getUVSetCount() const {
    if (parentMesh && parentMesh->geometry) {
        size_t count = 0;
        while (true) {
            std::string attrName = (count == 0) ? "uv" : "uv" + std::to_string(count);
            if (parentMesh->geometry->has_attribute(attrName)) {
                count++;
            } else {
                break;
            }
        }
        return count > 0 ? count : 1;
    }
    if (!uv_sets || uv_sets->empty()) return 1;
    return uv_sets->size();
}

void Triangle::applyUVSet(size_t set_index) {
    if (parentMesh && parentMesh->geometry) {
        std::string attrName = (set_index == 0) ? "uv" : "uv" + std::to_string(set_index);
        const Vec2* srcUvs = parentMesh->geometry->get_attribute_data<Vec2>(attrName);
        Vec2* dstUvs = parentMesh->geometry->get_attribute_data_mut<Vec2>("uv");
        if (srcUvs && dstUvs) {
            uint32_t i0 = parentMesh->geometry->indices[faceIndex * 3 + 0];
            uint32_t i1 = parentMesh->geometry->indices[faceIndex * 3 + 1];
            uint32_t i2 = parentMesh->geometry->indices[faceIndex * 3 + 2];
            dstUvs[i0] = srcUvs[i0];
            dstUvs[i1] = srcUvs[i1];
            dstUvs[i2] = srcUvs[i2];
            return;
        }
    }
    if (!uv_sets || uv_sets->empty()) {
        return;   // only set 0 (already active in t0/t1/t2 or parentMesh)
    }
    const size_t resolved_index = (set_index < uv_sets->size()) ? set_index : 0;
    const auto& uv_set = (*uv_sets)[resolved_index];
    t0 = uv_set[0];
    t1 = uv_set[1];
    t2 = uv_set[2];
}

// ============================================================================
// Normals
// ============================================================================

void Triangle::set_normals(const Vec3& normal0, const Vec3& normal1, const Vec3& normal2) {
    setVertexNormal(0, normal0.normalize());
    setVertexNormal(1, normal1.normalize());
    setVertexNormal(2, normal2.normalize());
    vertexPositionsDirty = true;
}

// ============================================================================
// Transform Management
// ============================================================================

void Triangle::set_transform(const Matrix4x4& t) {
    auto handle = getTransformHandle();
    if (handle) {
        handle->setCurrent(t);
    }
    update_bounding_box();
}

void Triangle::updateTriangleTransform(Triangle& triangle, const Matrix4x4& transform) {
    triangle.set_transform(transform);
}

Matrix4x4 Triangle::getTransformMatrix() const {
    auto handle = getTransformHandle();
    if (handle) {
        return handle->getFinal();
    }
    return Matrix4x4::identity();
}

void Triangle::update_bounding_box() const {
    // No-op: AABB is calculated dynamically to save memory
}

bool Triangle::bounding_box(float time0, float time1, AABB& output_box) const {
    const Vec3& tv0 = getVertexPosition(0);
    const Vec3& tv1 = getVertexPosition(1);
    const Vec3& tv2 = getVertexPosition(2);

    Vec3 minPoint(
        (std::min)({ tv0.x, tv1.x, tv2.x }),
        (std::min)({ tv0.y, tv1.y, tv2.y }),
        (std::min)({ tv0.z, tv1.z, tv2.z })
    );
    Vec3 maxPoint(
        (std::max)({ tv0.x, tv1.x, tv2.x }),
        (std::max)({ tv0.y, tv1.y, tv2.y }),
        (std::max)({ tv0.z, tv1.z, tv2.z })
    );

    constexpr float DELTA = 1e-5f;
    output_box = AABB(minPoint - DELTA, maxPoint + DELTA);
    return true;
}

// ============================================================================
// Skinning
// ============================================================================

void Triangle::initializeSkinData() {
    if (!skinData) {
        // Query original positions first, before skinData is allocated!
        Vec3 origPos0 = getOriginalVertexPosition(0);
        Vec3 origPos1 = getOriginalVertexPosition(1);
        Vec3 origPos2 = getOriginalVertexPosition(2);

        skinData.reset(new SkinnedTriangleData());
        skinData->vertexBoneWeights.resize(3);
        skinData->originalVertexPositions.resize(3);
        
        skinData->originalVertexPositions[0] = origPos0;
        skinData->originalVertexPositions[1] = origPos1;
        skinData->originalVertexPositions[2] = origPos2;
    }
}

bool Triangle::hasAnySkinWeights() const {
    if (parentMesh && parentMesh->geometry && !parentMesh->geometry->skin_weights.empty()) {
        size_t vCount = parentMesh->geometry->get_vertex_count();
        uint32_t i0 = parentMesh->geometry->indices[faceIndex * 3 + 0];
        uint32_t i1 = parentMesh->geometry->indices[faceIndex * 3 + 1];
        uint32_t i2 = parentMesh->geometry->indices[faceIndex * 3 + 2];
        if (i0 < vCount && i1 < vCount && i2 < vCount) {
            if (!parentMesh->geometry->skin_weights[i0].empty() ||
                !parentMesh->geometry->skin_weights[i1].empty() ||
                !parentMesh->geometry->skin_weights[i2].empty()) {
                return true;
            }
        }
    }
    if (!skinData) return false;
    for (const auto& weights : skinData->vertexBoneWeights) {
        if (!weights.empty()) return true;
    }
    return false;
}

void Triangle::setSkinBoneWeights(int vertexIndex, const std::vector<std::pair<int, float>>& weights) {
    initializeSkinData();
    if (vertexIndex >= 0 && vertexIndex < 3) {
        std::vector<std::pair<int, float>> sortedWeights = weights;
        // Sort descending by weight (the second element of the pair)
        std::sort(sortedWeights.begin(), sortedWeights.end(), 
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                return a.second > b.second;
            });
        skinData->vertexBoneWeights[vertexIndex] = sortedWeights;
    }
}

const std::vector<std::pair<int, float>>& Triangle::getSkinBoneWeights(int vertexIndex) const {
    static const std::vector<std::pair<int, float>> empty;
    if (parentMesh && parentMesh->geometry && !parentMesh->geometry->skin_weights.empty()) {
        uint32_t globalIndex = parentMesh->geometry->indices[faceIndex * 3 + vertexIndex];
        if (globalIndex < parentMesh->geometry->skin_weights.size()) {
            return parentMesh->geometry->skin_weights[globalIndex];
        }
    }
    if (skinData && vertexIndex >= 0 && vertexIndex < 3) {
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
    if (parentMesh && parentMesh->geometry && !parentMesh->geometry->skin_weights.empty()) {
        uint32_t globalIndex = parentMesh->geometry->indices[faceIndex * 3 + vi];
        if (globalIndex < parentMesh->geometry->skin_weights.size()) {
            const auto& boneWeights = parentMesh->geometry->skin_weights[globalIndex];
            if (boneWeights.empty()) return getVertexPosition(vi);
            
            const Vec3* origP = parentMesh->geometry->get_positions_orig();
            if (!origP) origP = parentMesh->geometry->get_positions();
            if (!origP) return getVertexPosition(vi);
            
            const Vec3& origPosition = origP[globalIndex];
            
            Vec3 blended = Vec3(0);
            float totalWeight = 0.0f;
            for (const auto& [boneIdx, weight] : boneWeights) {
                if (boneIdx >= 0 && boneIdx < (int)finalBoneMatrices.size()) {
                    Vec3 transformed = finalBoneMatrices[boneIdx].transform_point(origPosition);
                    blended += transformed * weight;
                    totalWeight += weight;
                }
            }
            if (totalWeight < 1e-4f) return getVertexPosition(vi);
            return blended;
        }
    }

    if (!hasSkinData() || vi >= skinData->vertexBoneWeights.size()) return getVertexPosition(vi);
    
    const auto& boneWeights = skinData->vertexBoneWeights[vi];
    if (boneWeights.empty()) return getVertexPosition(vi); // Fallback to current position (usually original if not yet moved)
    
    const auto& origPosition = skinData->originalVertexPositions[vi];
    
    Vec3 blended = Vec3(0);
    float totalWeight = 0.0f;
    for (const auto& [boneIdx, weight] : boneWeights) {
        if (boneIdx >= 0 && boneIdx < (int)finalBoneMatrices.size()) {
            Vec3 transformed = finalBoneMatrices[boneIdx].transform_point(origPosition);
            blended += transformed * weight;
            totalWeight += weight;
        }
    }
    
    if (totalWeight < 1e-4f) return getVertexPosition(vi);
    return blended;
}

void Triangle::apply_skinning(const std::vector<Matrix4x4>& finalBoneMatrices) {
    if (parentMesh && parentMesh->geometry && !parentMesh->geometry->skin_weights.empty()) {
        // Flat DNA mesh path: run mesh-level skinning ONCE per pose change
        
        // 1. Compute bone hash
        uint64_t hash = 1469598103934665603ull;
        hash ^= finalBoneMatrices.size();
        hash *= 1099511628211ull;
        if (!finalBoneMatrices.empty()) {
            size_t step = (finalBoneMatrices.size() > 16) ? (finalBoneMatrices.size() / 16) : 1;
            for (size_t i = 0; i < finalBoneMatrices.size(); i += step) {
                const float* m = &finalBoneMatrices[i].m[0][0];
                for (int j = 0; j < 16; ++j) {
                    uint32_t b;
                    std::memcpy(&b, &m[j], 4);
                    hash ^= b;
                    hash *= 1099511628211ull;
                }
            }
        }
        
        Matrix4x4 rootTransform = Matrix4x4::identity();
        Matrix4x4 rootNormalTransform = Matrix4x4::identity();
        if (parentMesh->transform) {
            rootTransform = parentMesh->transform->getFinal();
            rootNormalTransform = parentMesh->transform->getNormalTransform();
        }
        
        const float* rm = &rootTransform.m[0][0];
        for (int j = 0; j < 16; ++j) {
            uint32_t b;
            std::memcpy(&b, &rm[j], 4);
            hash ^= b;
            hash *= 1099511628211ull;
        }
        
        // 2. Check if already skinned for this pose
        if (parentMesh->geometry->last_skinned_pose_hash == hash) {
            vertexPositionsDirty = true;
            return;
        }
        
        // 3. Perform fast mesh-level parallel skinning
        size_t vCount = parentMesh->geometry->get_vertex_count();
        const Vec3* origP = parentMesh->geometry->get_positions_orig();
        if (!origP) origP = parentMesh->geometry->get_positions();
        const Vec3* origN = parentMesh->geometry->get_normals_orig();
        if (!origN) origN = parentMesh->geometry->get_normals();
        
        Vec3* positions = parentMesh->geometry->get_positions_mut();
        Vec3* normals = parentMesh->geometry->get_normals_mut();
        
        if (origP && origN && positions && normals) {
            #pragma omp parallel for num_threads(get_omp_threads_limit()) schedule(static)
            for (int v = 0; v < (int)vCount; ++v) {
                const auto& weights = parentMesh->geometry->skin_weights[v];
                if (weights.empty()) {
                    positions[v] = rootTransform.transform_point(origP[v]);
                    normals[v] = rootNormalTransform.transform_vector(origN[v]).normalize();
                    continue;
                }
                
                Matrix4x4 blendedBoneMatrix = Matrix4x4::zero();
                float totalWeight = 0.0f;
                for (const auto& [boneIdx, weight] : weights) {
                    if (boneIdx >= 0 && boneIdx < static_cast<int>(finalBoneMatrices.size())) {
                        totalWeight += weight;
                    }
                }
                
                if (totalWeight < 1e-5f) {
                    positions[v] = rootTransform.transform_point(origP[v]);
                    normals[v] = rootNormalTransform.transform_vector(origN[v]).normalize();
                    continue;
                }
                
                float invWeight = 1.0f / totalWeight;
                for (const auto& [boneIdx, weight] : weights) {
                    if (boneIdx < 0 || boneIdx >= static_cast<int>(finalBoneMatrices.size()) || weight < 1e-7f) {
                        continue;
                    }
                    float normalizedWeight = weight * invWeight;
                    const Matrix4x4& boneMatrix = finalBoneMatrices[boneIdx];
                    for (int r = 0; r < 4; ++r) {
                        for (int c = 0; c < 4; ++c) {
                            blendedBoneMatrix.m[r][c] += boneMatrix.m[r][c] * normalizedWeight;
                        }
                    }
                }
                
                Vec3 localSkinnedPos = blendedBoneMatrix.transform_point(origP[v]);
                positions[v] = rootTransform.transform_point(localSkinnedPos);
                
                Matrix4x4 nodeNormalMatrix = blendedBoneMatrix.inverse().transpose();
                Vec3 localSkinnedNormal = nodeNormalMatrix.transform_vector(origN[v]).normalize();
                normals[v] = rootNormalTransform.transform_vector(localSkinnedNormal).normalize();
            }
            
            parentMesh->geometry->last_skinned_pose_hash = hash;
        }
        
        vertexPositionsDirty = true;
        return;
    }

    if (!hasSkinData()) return;

    const auto& boneWeights = skinData->vertexBoneWeights;
    const auto& origPositions = skinData->originalVertexPositions;

    // Validation check
    if (boneWeights.size() != 3 || origPositions.size() != 3 ||
        getOriginalVertexNormal(0).length_squared() < 1e-6 || 
        getOriginalVertexNormal(1).length_squared() < 1e-6 || 
        getOriginalVertexNormal(2).length_squared() < 1e-6) {

        for (int i = 0; i < 3; ++i) {
            setVertexPosition(i, getOriginalVertexPosition(i));
            setVertexNormal(i, getOriginalVertexNormal(i));
        }
        return;
    }

    // Get root transform (Gizmo placement in scene)
    Matrix4x4 rootTransform;
    Matrix4x4 rootNormalTransform;
    auto handle = getTransformHandle();
    if (handle) {
        rootTransform = handle->getFinal();
        rootNormalTransform = handle->getNormalTransform();
    } else {
        rootTransform = Matrix4x4::identity();
        rootNormalTransform = Matrix4x4::identity();
    }

    // Process each vertex
    for (int vi = 0; vi < 3; ++vi) {
        if (boneWeights[vi].empty()) {
            setVertexPosition(vi, rootTransform.transform_point(getOriginalVertexPosition(vi)));
            setVertexNormal(vi, rootNormalTransform.transform_vector(getOriginalVertexNormal(vi)).normalize());
            continue;
        }

        // Linear Blend Skinning (standard approach)
        Matrix4x4 blendedBoneMatrix = Matrix4x4::zero();
        float totalWeight = 0.0f;
        
        // Calculate total weight for normalization
        for (const auto& [boneIdx, weight] : boneWeights[vi]) {
            if (boneIdx >= 0 && boneIdx < static_cast<int>(finalBoneMatrices.size())) {
                totalWeight += weight;
            }
        }

        if (totalWeight < 1e-5f) {
            // Fallback: If no valid weights, use root transform as if unweighted
            setVertexPosition(vi, rootTransform.transform_point(getOriginalVertexPosition(vi)));
            setVertexNormal(vi, rootNormalTransform.transform_vector(getOriginalVertexNormal(vi)).normalize());
            continue;
        }

        float invWeight = 1.0f / totalWeight;

        for (const auto& [boneIdx, weight] : boneWeights[vi]) {
            if (boneIdx < 0 || boneIdx >= static_cast<int>(finalBoneMatrices.size()) || weight < 1e-7f) {
                continue;
            }
            
            float normalizedWeight = weight * invWeight;

            // Blend bone matrices
            const Matrix4x4& boneMatrix = finalBoneMatrices[boneIdx];
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    blendedBoneMatrix.m[i][j] += boneMatrix.m[i][j] * normalizedWeight;
                }
            }
        }

        // Apply Bone Skinning -> Then Apply Root Transform 
        Vec3 localSkinnedPos = blendedBoneMatrix.transform_point(origPositions[vi]);
        setVertexPosition(vi, rootTransform.transform_point(localSkinnedPos));
        
        // Apply to normal
        Matrix4x4 nodeNormalMatrix = blendedBoneMatrix.inverse().transpose();
        Vec3 localSkinnedNormal = nodeNormalMatrix.transform_vector(getOriginalVertexNormal(vi)).normalize();
        setVertexNormal(vi, rootNormalTransform.transform_vector(localSkinnedNormal).normalize());
    }

    vertexPositionsDirty = true;
}

Vec3 Triangle::apply_bone_to_normal(const Vec3& originalNormal,
                                     const std::vector<std::pair<int, float>>& boneWeights,
                                     const std::vector<Matrix4x4>& finalBoneMatrices) const {
    Vec3 blended = Vec3(0);
    for (const auto& [boneIdx, weight] : boneWeights) {
        if (boneIdx < 0 || boneIdx >= static_cast<int>(finalBoneMatrices.size())) {
            continue;
        }

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
    const Vec3& v0 = getVertexPosition(0);
    const Vec3& v1 = getVertexPosition(1);
    const Vec3& v2 = getVertexPosition(2);
    
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
    rec.tri_mesh = parentMesh.get();   // Faz 1: (mesh, faceIndex) handle alongside Triangle*
    rec.tri_face = faceIndex;
    rec.t = t;
    rec.point = r.at(t);

    // Barycentric weights (u, v, w)
    float w = 1.0f - u - v;

    // Interpolate normal directly with barycentric coordinates
    rec.interpolated_normal =
        (w * getVertexNormal(0) + u * getVertexNormal(1) + v * getVertexNormal(2)).normalize();

    // Ensure normal always points against ray and set front_face flag
    rec.set_face_normal(r, rec.interpolated_normal);

    // Use u, v, w directly for UV interpolation
    auto [uv0, uv1, uv2] = getUVCoordinates();
    Vec2 uv = w * uv0 + u * uv1 + v * uv2;

    rec.uv = uv;
    rec.u = uv.u;
    rec.v = uv.v;
    
    // --- HIGH PERFORMANCE MATERIAL ACCESS & ALPHA TESTING ---
    Material* currentMat = MaterialManager::getInstance().getMaterial(getMaterialID());

    // --- ALPHA TESTING ---
    if (currentMat && currentMat->isTransparent()) {
        float opacity = currentMat->get_opacity(rec.uv);
        // Opacity textures on foliage/cutouts need deterministic masking on CPU.
        // Stochastic rejection here causes hit popping and broken aerial blending.
        if (opacity <= 0.5f) {
            return false; // Transparent hit, ignore and let BVH continue
        }
    }

    rec.materialPtr = currentMat;
    rec.materialID = getMaterialID();
    rec.terrain_id = terrain_id;

    return true;
}

// ============================================================================
// Transform Initialization & Updates
// ============================================================================

void Triangle::updateTransformedVertices() {
    auto handle = getTransformHandle();
    if (handle) {
        updateTransformedVerticesWith(handle->getFinal(), handle->getNormalTransform());
    } else {
        updateTransformedVerticesWith(Matrix4x4::identity(), Matrix4x4::identity());
    }
}

void Triangle::updateTransformedVerticesWith(const Matrix4x4& finalTransform,
                                             const Matrix4x4& normalTransform) {
    if (parentMesh && parentMesh->geometry) {
        uint32_t i0 = parentMesh->geometry->indices[faceIndex * 3 + 0];
        uint32_t i1 = parentMesh->geometry->indices[faceIndex * 3 + 1];
        uint32_t i2 = parentMesh->geometry->indices[faceIndex * 3 + 2];

        Vec3* positions = parentMesh->geometry->get_positions_mut();
        Vec3* normals = parentMesh->geometry->get_normals_mut();

        if (positions && normals) {
            positions[i0] = finalTransform.transform_point(getOriginalVertexPosition(0));
            positions[i1] = finalTransform.transform_point(getOriginalVertexPosition(1));
            positions[i2] = finalTransform.transform_point(getOriginalVertexPosition(2));

            normals[i0] = normalTransform.transform_vector(getOriginalVertexNormal(0)).normalize();
            normals[i1] = normalTransform.transform_vector(getOriginalVertexNormal(1)).normalize();
            normals[i2] = normalTransform.transform_vector(getOriginalVertexNormal(2)).normalize();

            vertexPositionsDirty = true;
            update_bounding_box();
            return;
        }
    }

    if (!getOriginalGeometry()) {
        if (finalTransform.isIdentity()) {
            return;
        }
        auto sharedGeom = std::make_shared<OriginalMeshGeometry>();
        const TriangleVertexData* sv = svr();
        sharedGeom->positions = { sv[0].position, sv[1].position, sv[2].position };
        sharedGeom->normals = { sv[0].normal, sv[1].normal, sv[2].normal };
        setOriginalGeometry(sharedGeom);
        setAssimpVertexIndices(0, 1, 2);
    }
    for (int i = 0; i < 3; ++i) {
        setVertexPosition(i, finalTransform.transform_point(getOriginalVertexPosition(i)));
        setVertexNormal(i, normalTransform.transform_vector(getOriginalVertexNormal(i)).normalize());
    }

    vertexPositionsDirty = true;
    update_bounding_box();
}

#include <unordered_set>
#include <mutex>

class StringInterner {
    std::unordered_set<std::string> strings;
    std::mutex mtx;
public:
    static StringInterner& getInstance() {
        static StringInterner instance;
        return instance;
    }
    const std::string* intern(const std::string& str) {
        if (str.empty()) return nullptr;
        std::lock_guard<std::mutex> lock(mtx);
        auto it = strings.insert(str);
        return &(*it.first);
    }
};

void Triangle::setNodeName(const std::string& name) {
    if (parentMesh) {
        parentMesh->nodeName = name;
    }
    nodeNamePtr = StringInterner::getInstance().intern(name);
}

void Triangle::setBaseTransform(const Matrix4x4& t) {
    auto handle = getTransformHandle();
    if (handle) {
        handle->setBase(t);
    }
    updateTransformedVertices();
}

void Triangle::initialize_transforms() {
    // Store original vertices
    for (int i = 0; i < 3; ++i) {
        setOriginalVertexPosition(i, getVertexPosition(i));
        setOriginalVertexNormal(i, getVertexNormal(i).normalize());
    }
}

void Triangle::updateAnimationTransform(const Matrix4x4& animTransform) {
    auto handle = getTransformHandle();
    if (handle) {
        handle->setCurrent(animTransform);
    }
    
    updateTransformedVertices();
}



// ============================================================================
// Packet Tracing Implementation (Phase 2)
// ============================================================================


bool Triangle::occluded(const Ray& ray, float t_min, float t_max) const {
    if (!visible) return false;

    const Vec3& v0 = getVertexPosition(0);
    const Vec3& v1 = getVertexPosition(1);
    const Vec3& v2 = getVertexPosition(2);

    const Vec3 edge1 = v1 - v0;
    const Vec3 edge2 = v2 - v0;

    Vec3 h = Vec3::cross(ray.direction, edge2);
    float a = Vec3::dot(edge1, h);

    if (fabs(a) < EPSILON)
        return false;

    float f = 1.0f / a;
    Vec3 s = ray.origin - v0;
    float u = f * Vec3::dot(s, h);

    if (u < -EPSILON || u > 1.0f + EPSILON)
        return false;

    Vec3 q = Vec3::cross(s, edge1);
    float v = f * Vec3::dot(ray.direction, q);

    if (v < -EPSILON || (u + v) > 1.0f + EPSILON)
        return false;

    float t = f * Vec3::dot(edge2, q);
    if (t < t_min || t > t_max)
        return false;

    Material* currentMat = MaterialManager::getInstance().getMaterial(getMaterialID());
    if (currentMat && currentMat->isTransparent()) {
        float w = 1.0f - u - v;
        auto [uv0, uv1, uv2] = getUVCoordinates();
        Vec2 uv = w * uv0 + u * uv1 + v * uv2;
        float opacity = currentMat->get_opacity(uv);
        if (opacity <= 0.5f) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// COW and Shared Original Geometry Accessors
// ============================================================================

const Vec3& Triangle::getOriginalVertexPosition(int i) const {
    if (skinData && i >= 0 && i < 3) {
        if (i < (int)skinData->originalVertexPositions.size()) {
            return skinData->originalVertexPositions[i];
        }
    }
    if (parentMesh && parentMesh->geometry) {
        // Lazy initialization of P_orig from P if it doesn't exist yet
        if (!parentMesh->geometry->has_attribute("P_orig")) {
            std::lock_guard<std::mutex> lock(g_original_position_init_mutex);
            if (!parentMesh->geometry->has_attribute("P_orig")) {
                auto* geom = const_cast<DNA::GeometryDetail*>(parentMesh->geometry.get());
                geom->add_attribute<Vec3>("P_orig");
                const Vec3* currentP = geom->get_attribute_data<Vec3>("P");
                Vec3* origP = geom->get_attribute_data_mut<Vec3>("P_orig");
                if (currentP && origP) {
                    size_t vCount = geom->get_vertex_count();
                    for (size_t v = 0; v < vCount; ++v) {
                        origP[v] = currentP[v];
                    }
                }
            }
        }
        const Vec3* origPositions = parentMesh->geometry->get_positions_orig();
        if (origPositions) {
            uint32_t globalIndex = parentMesh->geometry->indices[faceIndex * 3 + i];
            return origPositions[globalIndex];
        }
    }
    auto origGeom = getOriginalGeometry();
    if (origGeom && i >= 0 && i < 3) {
        unsigned int idx = getAssimpVertexIndices()[i];
        if (idx < origGeom->positions.size()) {
            return origGeom->positions[idx];
        }
    }
    return getVertexPosition(i);
}

void Triangle::setOriginalVertexPosition(int i, const Vec3& pos) {
    if (skinData && i >= 0 && i < 3) {
        if (i >= (int)skinData->originalVertexPositions.size()) {
            skinData->originalVertexPositions.resize(3, Vec3(0.0f));
        }
        skinData->originalVertexPositions[i] = pos;
        return;
    }
    if (parentMesh && parentMesh->geometry) {
        if (!parentMesh->geometry->has_attribute("P_orig")) {
            parentMesh->geometry->add_attribute<Vec3>("P_orig");
            const Vec3* currentP = parentMesh->geometry->get_attribute_data<Vec3>("P");
            Vec3* origP = parentMesh->geometry->get_attribute_data_mut<Vec3>("P_orig");
            if (currentP && origP) {
                size_t vCount = parentMesh->geometry->get_vertex_count();
                for (size_t v = 0; v < vCount; ++v) {
                    origP[v] = currentP[v];
                }
            }
        }
        Vec3* origPositions = parentMesh->geometry->get_attribute_data_mut<Vec3>("P_orig");
        if (origPositions) {
            uint32_t globalIndex = parentMesh->geometry->indices[faceIndex * 3 + i];
            origPositions[globalIndex] = pos;
            return;
        }
    }
    auto origGeom = getOriginalGeometry();
    if (!origGeom) {
        origGeom = std::make_shared<OriginalMeshGeometry>();
        origGeom->positions.resize(3, Vec3(0.0f));
        origGeom->normals.resize(3, Vec3(0.0f, 1.0f, 0.0f));
        setOriginalGeometry(origGeom);
        setAssimpVertexIndices(0, 1, 2);
    }
    unsigned int idx = getAssimpVertexIndices()[i];
    if (idx >= origGeom->positions.size()) {
        origGeom->positions.resize(idx + 1, Vec3(0.0f));
    }
    origGeom->positions[idx] = pos;
}

const Vec3& Triangle::getOriginalVertexNormal(int i) const {
    if (parentMesh && parentMesh->geometry) {
        // Lazy initialization of N_orig from N if it doesn't exist yet
        if (!parentMesh->geometry->has_attribute("N_orig")) {
            std::lock_guard<std::mutex> lock(g_original_normal_init_mutex);
            if (!parentMesh->geometry->has_attribute("N_orig")) {
                auto* geom = const_cast<DNA::GeometryDetail*>(parentMesh->geometry.get());
                geom->add_attribute<Vec3>("N_orig");
                const Vec3* currentN = geom->get_attribute_data<Vec3>("N");
                Vec3* origN = geom->get_attribute_data_mut<Vec3>("N_orig");
                if (currentN && origN) {
                    size_t vCount = geom->get_vertex_count();
                    for (size_t v = 0; v < vCount; ++v) {
                        origN[v] = currentN[v];
                    }
                }
            }
        }
        const Vec3* origNormals = parentMesh->geometry->get_normals_orig();
        if (origNormals) {
            uint32_t globalIndex = parentMesh->geometry->indices[faceIndex * 3 + i];
            return origNormals[globalIndex];
        }
    }
    auto origGeom = getOriginalGeometry();
    if (origGeom && i >= 0 && i < 3) {
        unsigned int idx = getAssimpVertexIndices()[i];
        if (idx < origGeom->normals.size()) {
            return origGeom->normals[idx];
        }
    }
    return getVertexNormal(i);
}

void Triangle::setOriginalVertexNormal(int i, const Vec3& normal) {
    if (parentMesh && parentMesh->geometry) {
        if (!parentMesh->geometry->has_attribute("N_orig")) {
            parentMesh->geometry->add_attribute<Vec3>("N_orig");
            const Vec3* currentN = parentMesh->geometry->get_attribute_data<Vec3>("N");
            Vec3* origN = parentMesh->geometry->get_attribute_data_mut<Vec3>("N_orig");
            if (currentN && origN) {
                size_t vCount = parentMesh->geometry->get_vertex_count();
                for (size_t v = 0; v < vCount; ++v) {
                    origN[v] = currentN[v];
                }
            }
        }
        Vec3* origNormals = parentMesh->geometry->get_attribute_data_mut<Vec3>("N_orig");
        if (origNormals) {
            uint32_t globalIndex = parentMesh->geometry->indices[faceIndex * 3 + i];
            origNormals[globalIndex] = normal;
            return;
        }
    }
    auto origGeom = getOriginalGeometry();
    if (!origGeom) {
        origGeom = std::make_shared<OriginalMeshGeometry>();
        origGeom->positions.resize(3, Vec3(0.0f));
        origGeom->normals.resize(3, Vec3(0.0f, 1.0f, 0.0f));
        setOriginalGeometry(origGeom);
        setAssimpVertexIndices(0, 1, 2);
    }
    unsigned int idx = getAssimpVertexIndices()[i];
    if (idx >= origGeom->normals.size()) {
        origGeom->normals.resize(idx + 1, Vec3(0.0f, 1.0f, 0.0f));
    }
    origGeom->normals[idx] = normal;
}

Vec3& Triangle::v_orig(int i) {
    if (skinData && i >= 0 && i < 3) {
        if (i >= (int)skinData->originalVertexPositions.size()) {
            skinData->originalVertexPositions.resize(3, Vec3(0.0f));
        }
        return skinData->originalVertexPositions[i];
    }
    if (parentMesh && parentMesh->geometry) {
        if (!parentMesh->geometry->has_attribute("P_orig")) {
            parentMesh->geometry->add_attribute<Vec3>("P_orig");
            const Vec3* currentP = parentMesh->geometry->get_attribute_data<Vec3>("P");
            Vec3* origP = parentMesh->geometry->get_attribute_data_mut<Vec3>("P_orig");
            if (currentP && origP) {
                size_t vCount = parentMesh->geometry->get_vertex_count();
                for (size_t v = 0; v < vCount; ++v) {
                    origP[v] = currentP[v];
                }
            }
        }
        Vec3* origPositions = parentMesh->geometry->get_attribute_data_mut<Vec3>("P_orig");
        if (origPositions) {
            uint32_t globalIndex = parentMesh->geometry->indices[faceIndex * 3 + i];
            return origPositions[globalIndex];
        }
    }
    auto origGeom = getOriginalGeometry();
    if (!origGeom) {
        origGeom = std::make_shared<OriginalMeshGeometry>();
        origGeom->positions.resize(3, Vec3(0.0f));
        origGeom->normals.resize(3, Vec3(0.0f, 1.0f, 0.0f));
        setOriginalGeometry(origGeom);
        setAssimpVertexIndices(0, 1, 2);
    }
    unsigned int idx = getAssimpVertexIndices()[i];
    if (idx >= origGeom->positions.size()) {
        origGeom->positions.resize(idx + 1, Vec3(0.0f));
    }
    return origGeom->positions[idx];
}

const Vec3& Triangle::v_orig(int i) const {
    return getOriginalVertexPosition(i);
}

Vec3& Triangle::v_orig_norm(int i) {
    if (parentMesh && parentMesh->geometry) {
        if (!parentMesh->geometry->has_attribute("N_orig")) {
            parentMesh->geometry->add_attribute<Vec3>("N_orig");
            const Vec3* currentN = parentMesh->geometry->get_attribute_data<Vec3>("N");
            Vec3* origN = parentMesh->geometry->get_attribute_data_mut<Vec3>("N_orig");
            if (currentN && origN) {
                size_t vCount = parentMesh->geometry->get_vertex_count();
                for (size_t v = 0; v < vCount; ++v) {
                    origN[v] = currentN[v];
                }
            }
        }
        Vec3* origNormals = parentMesh->geometry->get_attribute_data_mut<Vec3>("N_orig");
        if (origNormals) {
            uint32_t globalIndex = parentMesh->geometry->indices[faceIndex * 3 + i];
            return origNormals[globalIndex];
        }
    }
    auto origGeom = getOriginalGeometry();
    if (!origGeom) {
        origGeom = std::make_shared<OriginalMeshGeometry>();
        origGeom->positions.resize(3, Vec3(0.0f));
        origGeom->normals.resize(3, Vec3(0.0f, 1.0f, 0.0f));
        setOriginalGeometry(origGeom);
        setAssimpVertexIndices(0, 1, 2);
    }
    unsigned int idx = getAssimpVertexIndices()[i];
    if (idx >= origGeom->normals.size()) {
        origGeom->normals.resize(idx + 1, Vec3(0.0f, 1.0f, 0.0f));
    }
    return origGeom->normals[idx];
}

const Vec3& Triangle::v_orig_norm(int i) const {
    return getOriginalVertexNormal(i);
}


