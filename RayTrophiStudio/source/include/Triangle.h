/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Triangle.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <memory>
#include <string>
#include <algorithm>
#include <optional>
#include <array>
#include <vector>
#include "Hittable.h"
#include "Vec2.h"
#include "Matrix4x4.h"
#include "Vec3SIMD.h"
#include "Transform.h"
#include "MaterialManager.h"
#include "sbt_data.h"
#include <SDL.h>
#include <SDL_image.h>
#include <OptixWrapper.h>
#include "TriangleMesh.h"

// Forward declarations
class Material;
class Texture;

// Copyable owning pointer with value (deep-copy) semantics. Used so optional/rare-payload
// members (e.g. skinning data) cost only 8 bytes when absent — instead of reserving the full
// payload inline like std::optional — while keeping Triangle copyable (it is duplicated via
// make_shared<Triangle>(*src) in a few edit paths).
template <typename T>
struct deep_ptr {
    std::unique_ptr<T> p;
    deep_ptr() = default;
    deep_ptr(const deep_ptr& o) : p(o.p ? std::make_unique<T>(*o.p) : nullptr) {}
    deep_ptr(deep_ptr&&) noexcept = default;
    deep_ptr& operator=(const deep_ptr& o) { p = o.p ? std::make_unique<T>(*o.p) : nullptr; return *this; }
    deep_ptr& operator=(deep_ptr&&) noexcept = default;
    T* operator->() const { return p.get(); }
    T& operator*() const { return *p; }
    explicit operator bool() const { return static_cast<bool>(p); }
    T* get() const { return p.get(); }
    void reset(T* x = nullptr) { p.reset(x); }
};

struct OriginalMeshGeometry {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
};

/**
 * @brief Consolidated vertex data structure for memory optimization
 * Stores only world-space position and normal. Original bind-pose data is
 * stored in the shared OriginalMeshGeometry structure to save massive memory.
 */
struct TriangleVertexData {
    Vec3 position;       // Current/transformed position (world space)
    Vec3 normal;         // Current/transformed normal (world space)
};

/**
 * @brief Skinning data - allocated only for rigged triangles
 * Saves ~80+ bytes for non-rigged triangles
 */
struct SkinnedTriangleData {
    std::vector<std::vector<std::pair<int, float>>> vertexBoneWeights; // [vertex][(boneIndex, weight)]
    std::vector<Vec3> originalVertexPositions; // For skinning calculations
};

struct TriangleOptionalData {
    std::shared_ptr<OriginalMeshGeometry> originalGeometry;
    std::array<unsigned int, 3> assimpVertexIndices = { 0, 0, 0 };
    int faceIndex = -1;
    std::shared_ptr<Texture> texture;
    deep_ptr<OptixGeometryData::TextureBundle> textureBundle;
};

/**
 * @brief Optimized Triangle class
 * 
 * Memory optimizations applied:
 * - Vertex consolidation: 9 Vec3 -> TriangleVertexData[3] (108 -> 48 bytes saved)
 * - Material ID instead of shared_ptr (72 -> 2 bytes saved)
 * - Shared Transform (256 -> 8 bytes for shared case)
 * - Optional skinning data (saves ~80 bytes for non-rigged)
 * - Removed redundant members (smoothingGroup, duplicate UVs)
 * - Lazy AABB caching (removed min_point, max_point storage)
 * 
 * Total estimated savings: 612 -> ~146 bytes per triangle (75% reduction)
 */
// LegacyTriangleData removed for inline flat layout

class Triangle : public Hittable {
public:
    // Facade/View fields referencing a TriangleMesh
    std::shared_ptr<TriangleMesh> parentMesh = nullptr;
    uint32_t faceIndex = 0;

    // ========================================================================
    // Constructors
    // ========================================================================
    
    Triangle();
    
    // Facade constructor referencing a TriangleMesh
    Triangle(std::shared_ptr<TriangleMesh> parent, uint32_t faceIdx);

    // New optimized constructor with material ID
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
             const Vec3& na, const Vec3& nb, const Vec3& nc,
             const Vec2& ta, const Vec2& tb, const Vec2& tc,
             uint16_t matID);

    // Legacy constructor with shared_ptr (converts to ID internally)
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
             const Vec3& na, const Vec3& nb, const Vec3& nc,
             const Vec2& ta, const Vec2& tb, const Vec2& tc,
             std::shared_ptr<Material> m);

    inline uint16_t getMaterialID() const {
        if (parentMesh && parentMesh->geometry) {
            const auto& geom = *parentMesh->geometry;
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3;
            if (idxOffset < geom.indices.size()) {
                const uint16_t* matIDs = geom.get_attribute_data<uint16_t>("materialID");
                if (matIDs) {
                    uint32_t globalIndex = geom.indices[idxOffset];
                    if (globalIndex < geom.get_vertex_count()) return matIDs[globalIndex];
                }
            }
        }
        return materialID;
    }

    inline void setMaterialID(uint16_t id) {
        if (parentMesh && parentMesh->geometry) {
            const auto& geom = *parentMesh->geometry;
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3;
            if (idxOffset + 2 < geom.indices.size()) {
                uint16_t* matIDs = parentMesh->geometry->get_attribute_data_mut<uint16_t>("materialID");
                if (matIDs) {
                    const size_t vCount = geom.get_vertex_count();
                    uint32_t i0 = geom.indices[idxOffset + 0];
                    uint32_t i1 = geom.indices[idxOffset + 1];
                    uint32_t i2 = geom.indices[idxOffset + 2];
                    if (i0 < vCount && i1 < vCount && i2 < vCount) {
                        matIDs[i0] = id; matIDs[i1] = id; matIDs[i2] = id;
                        return;
                    }
                }
            }
        }
        materialID = id;
    }

    // ========================================================================
    // Vertex Access Methods (replace direct member access)
    // ========================================================================
    
    inline const Vec3& getVertexPosition(int i) const {
        if (parentMesh && parentMesh->geometry && i >= 0 && i < 3) {
            const auto& geom = *parentMesh->geometry;
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3 + static_cast<size_t>(i);
            if (idxOffset < geom.indices.size()) {
                const Vec3* positions = geom.get_positions();
                if (positions) {
                    uint32_t globalIndex = geom.indices[idxOffset];
                    if (globalIndex < geom.get_vertex_count()) {
                        return positions[globalIndex];
                    }
                }
            }
        }
        return svr()[i >= 0 && i < 3 ? i : 0].position;
    }
    
    inline void setVertexPosition(int i, const Vec3& pos) { 
        if (parentMesh && parentMesh->geometry && i >= 0 && i < 3) {
            const auto& geom = *parentMesh->geometry;
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3 + static_cast<size_t>(i);
            if (idxOffset < geom.indices.size()) {
                Vec3* positions = parentMesh->geometry->get_positions_mut();
                if (positions) {
                    uint32_t globalIndex = geom.indices[idxOffset];
                    if (globalIndex < geom.get_vertex_count()) {
                        positions[globalIndex] = pos;
                        vertexPositionsDirty = true;
                        return;
                    }
                }
            }
        }
        svw()[i >= 0 && i < 3 ? i : 0].position = pos;
        vertexPositionsDirty = true;
    }
    
    inline const Vec3& getVertexNormal(int i) const {
        if (parentMesh && parentMesh->geometry && i >= 0 && i < 3) {
            const auto& geom = *parentMesh->geometry;
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3 + static_cast<size_t>(i);
            if (idxOffset < geom.indices.size()) {
                const Vec3* normals = geom.get_normals();
                if (normals) {
                    uint32_t globalIndex = geom.indices[idxOffset];
                    if (globalIndex < geom.get_vertex_count()) {
                        return normals[globalIndex];
                    }
                }
            }
        }
        return svr()[i >= 0 && i < 3 ? i : 0].normal;
    }
    
    inline void setVertexNormal(int i, const Vec3& normal) { 
        if (parentMesh && parentMesh->geometry && i >= 0 && i < 3) {
            const auto& geom = *parentMesh->geometry;
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3 + static_cast<size_t>(i);
            if (idxOffset < geom.indices.size()) {
                Vec3* normals = parentMesh->geometry->get_normals_mut();
                if (normals) {
                    uint32_t globalIndex = geom.indices[idxOffset];
                    if (globalIndex < geom.get_vertex_count()) {
                        normals[globalIndex] = normal;
                        vertexPositionsDirty = true;
                        return;
                    }
                }
            }
        }
        svw()[i >= 0 && i < 3 ? i : 0].normal = normal;
        vertexPositionsDirty = true;
    }
    
    const Vec3& getOriginalVertexPosition(int i) const;
    void setOriginalVertexPosition(int i, const Vec3& pos);
    
    const Vec3& getOriginalVertexNormal(int i) const;
    void setOriginalVertexNormal(int i, const Vec3& normal);

    // Convenience accessors for backward compatibility
    inline Vec3 getV0() const { return getVertexPosition(0); }
    inline Vec3 getV1() const { return getVertexPosition(1); }
    inline Vec3 getV2() const { return getVertexPosition(2); }
    inline Vec3 getN0() const { return getVertexNormal(0); }
    inline Vec3 getN1() const { return getVertexNormal(1); }
    inline Vec3 getN2() const { return getVertexNormal(2); }

    // ========================================================================
    // Material Access (ID-based)
    // ========================================================================
    
    // Legacy compatibility - fetches from MaterialManager
    std::shared_ptr<Material> getMaterial() const;
    void setMaterial(const std::shared_ptr<Material>& mat);

    // ============================================================================
    // Direct Reference Accessors (Backward Compatibility)
    // ============================================================================
    inline Vec3& v_pos(int i) {
        if (parentMesh && parentMesh->geometry) {
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3 + static_cast<size_t>(i);
            if (idxOffset < parentMesh->geometry->indices.size()) {
                uint32_t globalIndex = parentMesh->geometry->indices[idxOffset];
                Vec3* posData = parentMesh->geometry->get_positions_mut();
                if (posData) return posData[globalIndex];
            }
        }
        return svw()[i].position;
    }

    inline const Vec3& v_pos(int i) const {
        if (parentMesh && parentMesh->geometry) {
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3 + static_cast<size_t>(i);
            if (idxOffset < parentMesh->geometry->indices.size()) {
                uint32_t globalIndex = parentMesh->geometry->indices[idxOffset];
                const Vec3* posData = parentMesh->geometry->get_positions();
                if (posData) return posData[globalIndex];
            }
        }
        return svr()[i].position;
    }

    Vec3& v_orig(int i);
    const Vec3& v_orig(int i) const;

    inline Vec3& v_norm(int i) {
        if (parentMesh && parentMesh->geometry) {
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3 + static_cast<size_t>(i);
            if (idxOffset < parentMesh->geometry->indices.size()) {
                uint32_t globalIndex = parentMesh->geometry->indices[idxOffset];
                Vec3* normData = parentMesh->geometry->get_normals_mut();
                if (normData) return normData[globalIndex];
            }
        }
        return svw()[i].normal;
    }

    inline const Vec3& v_norm(int i) const {
        if (parentMesh && parentMesh->geometry) {
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3 + static_cast<size_t>(i);
            if (idxOffset < parentMesh->geometry->indices.size()) {
                uint32_t globalIndex = parentMesh->geometry->indices[idxOffset];
                const Vec3* normData = parentMesh->geometry->get_normals();
                if (normData) return normData[globalIndex];
            }
        }
        return svr()[i].normal;
    }

    Vec3& v_orig_norm(int i);
    const Vec3& v_orig_norm(int i) const;

    inline Vec2& t_ref(int i) {
        if (parentMesh && parentMesh->geometry) {
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3 + static_cast<size_t>(i);
            if (idxOffset < parentMesh->geometry->indices.size()) {
                uint32_t globalIndex = parentMesh->geometry->indices[idxOffset];
                Vec2* uvData = parentMesh->geometry->get_uvs_mut();
                if (uvData) return uvData[globalIndex];
            }
        }
        if (i == 0) return t0;
        if (i == 1) return t1;
        return t2;
    }

    inline const Vec2& t_ref(int i) const {
        if (parentMesh && parentMesh->geometry) {
            const size_t idxOffset = static_cast<size_t>(faceIndex) * 3 + static_cast<size_t>(i);
            if (idxOffset < parentMesh->geometry->indices.size()) {
                uint32_t globalIndex = parentMesh->geometry->indices[idxOffset];
                const Vec2* uvData = parentMesh->geometry->get_uvs();
                if (uvData) return uvData[globalIndex];
            }
        }
        if (i == 0) return t0;
        if (i == 1) return t1;
        return t2;
    }

    // ============================================================================
    // Material Management
    // ============================================================================
    
    inline void setTransformHandle(std::shared_ptr<Transform> handle) {
        if (parentMesh) {
            parentMesh->transform = handle;
            return;
        }
        transformHandle = handle;
    }
    
    inline std::shared_ptr<Transform> getTransformHandle() const {
        if (parentMesh) {
            return parentMesh->transform;
        }
        return transformHandle;
    }
    
    // Fast non-owning accessor to avoid atomic refcount operations in hot paths.
    // Returns a raw pointer to the internal Transform (may be nullptr).
    inline Transform* getTransformPtr() const noexcept {
        if (parentMesh) {
            return parentMesh->transform.get();
        }
        return transformHandle.get();
    }
    
    void setBaseTransform(const Matrix4x4& transform);
    void updateAnimationTransform(const Matrix4x4& animTransform);
    void set_transform(const Matrix4x4& t);
    static void updateTriangleTransform(Triangle& triangle, const Matrix4x4& transform);
    
    Matrix4x4 getTransformMatrix() const;
    void initialize_transforms();
    void updateTransformedVertices();
    // Same as updateTransformedVertices but with the (node-shared) matrices
    // precomputed by the caller: skips the per-triangle getFinal() fetch and
    // the per-triangle 4x4 inverse, and — because it never touches the
    // lazily-cached TransformHandle — is safe to run on many triangles of
    // the same node in parallel.
    void updateTransformedVerticesWith(const Matrix4x4& finalTransform,
                                       const Matrix4x4& normalTransform);

    // ========================================================================
    // UV Coordinates
    // ========================================================================
    
    // UV coords now handled via legacyData or parentMesh

    // Transient index into the active editable-mesh cache's source_triangles (SoA migration;
    // -1 = not in the current edit/sculpt cache). Lets the cache map a Triangle* back to its
    // face index in O(1) without a per-triangle hash map (triangle_vertex_ids /
    // triangle_to_mesh_index). Runtime-only, never serialized; a consumer confirms validity
    // with source_triangles[editable_index].get() == this before trusting it.
    int editable_index = -1;

    void setUVCoordinates(const Vec2& uv0, const Vec2& uv1, const Vec2& uv2);
    std::tuple<Vec2, Vec2, Vec2> getUVCoordinates() const;
    void setUVSetCoordinates(size_t set_index, const Vec2& uv0, const Vec2& uv1, const Vec2& uv2);
    std::tuple<Vec2, Vec2, Vec2> getUVSetCoordinates(size_t set_index) const;
    // UV set 0 lives in t0/t1/t2 (always present); uv_sets holds ONLY the EXTRA sets and is
    // empty for the common single-set case — so a single-set triangle no longer pays a
    // per-triangle heap allocation (the 12.5M-allocation spike that dominated dense-subdivide
    // materialize). When uv_sets is non-empty it is the full set list (uv_sets[0] == set 0
    // backup), so the rare multi-UV workflow (active-set switching via applyUVSet) round-trips.
    size_t getUVSetCount() const;
    void applyUVSet(size_t set_index);

    // ========================================================================
    // Skinning (Optional - only allocated for rigged triangles)
    // ========================================================================
    
    void initializeSkinData();
    bool hasSkinData() const { 
        if (skinData) return true;
        if (parentMesh && parentMesh->geometry && !parentMesh->geometry->skin_weights.empty()) return true;
        return false;
    }
    bool hasAnySkinWeights() const;
    
    void setSkinBoneWeights(int vertexIndex, const std::vector<std::pair<int, float>>& weights);
    const std::vector<std::pair<int, float>>& getSkinBoneWeights(int vertexIndex) const;
    
    void apply_skinning(const std::vector<Matrix4x4>& finalBoneMatrices);
    Vec3 apply_bone_to_vertex(int vi, const std::vector<Matrix4x4>& finalBoneMatrices) const;
    Vec3 apply_bone_to_normal(const Vec3& originalNormal,
                              const std::vector<std::pair<int, float>>& boneWeights,
                              const std::vector<Matrix4x4>& finalBoneMatrices) const;

    // Legacy skinning access (for compatibility)
    std::vector<std::vector<std::pair<int, float>>>& getVertexBoneWeights();
    std::vector<Vec3>& getOriginalVertexPositions();

    // ========================================================================
    // Hit Detection
    // ========================================================================
    
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec, bool ignore_volumes = false) const override;
    virtual bool bounding_box(float time0, float time1, AABB& output_box) const override;
    
    virtual bool occluded(const Ray& ray, float t_min, float t_max) const override;

    inline bool has_tangent_basis() const { return false; }

    // ========================================================================
    // Normals
    // ========================================================================
    
    void set_normals(const Vec3& normal0, const Vec3& normal1, const Vec3& normal2);

    // ========================================================================
    // Mesh/Node Information
    // ========================================================================
   
    inline const std::string& getNodeName() const { 
        if (parentMesh && !parentMesh->nodeName.empty()) {
            return parentMesh->nodeName;
        }
        static const std::string emptyString = "";
        return nodeNamePtr ? *nodeNamePtr : emptyString; 
    }
    void setNodeName(const std::string& name);
    bool isTriangle() const override { return true; }

    // ========================================================================
    // AABB Caching (Lazy Evaluation)
    // ========================================================================
    
    void update_bounding_box() const;
    inline void markAABBDirty() { 
        vertexPositionsDirty = true;
    }

    // ========================================================================
    // Legacy Compatibility (Direct vertex access - use accessors instead)
    // Marked for potential deprecation
    // ========================================================================
    
    // These provide direct access but should be migrated to accessor methods
    Vec3& v0_ref() { return svw()[0].position; }
    Vec3& v1_ref() { return svw()[1].position; }
    Vec3& v2_ref() { return svw()[2].position; }
    Vec3& n0_ref() { return svw()[0].normal; }
    Vec3& n1_ref() { return svw()[1].normal; }
    Vec3& n2_ref() { return svw()[2].normal; }

    // Read-only direct access for performance-critical code
    const Vec3& v0_cref() const { return getVertexPosition(0); }
    const Vec3& v1_cref() const { return getVertexPosition(1); }
    const Vec3& v2_cref() const { return getVertexPosition(2); }
    const Vec3& n0_cref() const { return getVertexNormal(0); }
    const Vec3& n1_cref() const { return getVertexNormal(1); }
    const Vec3& n2_cref() const { return getVertexNormal(2); }
    const std::string* nodeNamePtr = nullptr;
    int terrain_id = -1;
    mutable bool vertexPositionsDirty = false;

    // ========================================================================
    // Inline Triangle Data (Replaced LegacyTriangleData)
    // ========================================================================
    // NOTE: per-corner positions/normals moved to a lazily-allocated standalone payload
    // (see svw()/svr() below) — facade triangles (parentMesh set) read geometry from
    // parentMesh->geometry (SoA) and never allocate it. t0/t1/t2 (UV set 0) stay inline.
    Vec2 t0, t1, t2;
    deep_ptr<std::vector<std::array<Vec2, 3>>> uv_sets;
    // Terrain ID if this is a terrain triangle

    // ========================================================================
    // Memory-Optimized Optional Data Accessors
    // ========================================================================
    inline TriangleOptionalData& ensureOptionalData() const {
        if (!opt) opt.reset(new TriangleOptionalData());
        return *opt;
    }

    inline std::shared_ptr<OriginalMeshGeometry> getOriginalGeometry() const {
        return opt ? opt->originalGeometry : nullptr;
    }
    inline void setOriginalGeometry(std::shared_ptr<OriginalMeshGeometry> geom) {
        if (!geom && !opt) return;
        ensureOptionalData().originalGeometry = geom;
    }

    inline int getFaceIndex() const {
        return opt ? opt->faceIndex : -1;
    }
    inline void setFaceIndex(int idx) {
        if (idx == -1 && !opt) return;
        ensureOptionalData().faceIndex = idx;
    }

    inline const std::array<unsigned int, 3>& getAssimpVertexIndices() const {
        static const std::array<unsigned int, 3> defaultIndices = { 0, 0, 0 };
        return opt ? opt->assimpVertexIndices : defaultIndices;
    }
    inline void setAssimpVertexIndices(unsigned int i0, unsigned int i1, unsigned int i2) {
        ensureOptionalData().assimpVertexIndices = { i0, i1, i2 };
    }

    inline std::shared_ptr<Texture> getTexture() const {
        return opt ? opt->texture : nullptr;
    }
    inline void setTexture(std::shared_ptr<Texture> tex) {
        if (!tex && !opt) return;
        ensureOptionalData().texture = tex;
    }

    inline OptixGeometryData::TextureBundle* getTextureBundle() const {
        return opt && opt->textureBundle ? opt->textureBundle.get() : nullptr;
    }
    inline void setTextureBundle(const OptixGeometryData::TextureBundle& bundle) {
        ensureOptionalData().textureBundle.reset(new OptixGeometryData::TextureBundle(bundle));
    }

private:
    // ========================================================================
    // Optimized Data Members
    // ========================================================================
    
   
    uint16_t materialID = 0xFFFF;                       // Material lookup ID (2 bytes)
   
  
    std::shared_ptr<Transform> transformHandle; // Shared transform (8 bytes)
    deep_ptr<SkinnedTriangleData> skinData;      // Skinning data — 8 bytes when absent (was
                                                 // std::optional, which reserved ~56 B inline)
    
                     // 32 bytes avg
    mutable deep_ptr<TriangleOptionalData> opt;  // Optional fields grouped on the heap (8 bytes when absent)

    // Standalone-only per-corner position/normal. NEVER allocated for facade triangles
    // (parentMesh set) — their geometry lives in parentMesh->geometry (SoA). Lazily
    // allocated for standalone triangles, so a facade object is ~96 B lighter.
    struct StandaloneVerts {
        TriangleVertexData v[3];
        StandaloneVerts() {
            for (auto& x : v) { x.position = Vec3(0.0f); x.normal = Vec3(0.0f, 1.0f, 0.0f); }
        }
    };
    mutable deep_ptr<StandaloneVerts> standalone_verts_;
    inline TriangleVertexData* svw() {
        if (!standalone_verts_) standalone_verts_.reset(new StandaloneVerts());
        return standalone_verts_->v;
    }
    inline const TriangleVertexData* svr() const {
        if (!standalone_verts_) standalone_verts_.reset(new StandaloneVerts());
        return standalone_verts_->v;
    }

    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    // syncLegacyMembers() - REMOVED (no longer needed)
};

#endif // TRIANGLE_H

