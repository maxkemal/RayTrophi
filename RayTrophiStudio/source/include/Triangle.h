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

/**
 * @brief Consolidated vertex data structure for memory optimization
 * Replaces 9 separate Vec3 members (original, transformed, current) with this
 * 48 bytes vs 108 bytes - 55% savings on vertex data alone
 */
struct TriangleVertexData {
    Vec3 position;       // Current/transformed position
    Vec3 original;       // Bind-pose original position
    Vec3 normal;         // Current/transformed normal
    Vec3 originalNormal; // Bind-pose normal
};

/**
 * @brief Skinning data - allocated only for rigged triangles
 * Saves ~80+ bytes for non-rigged triangles
 */
struct SkinnedTriangleData {
    std::vector<std::vector<std::pair<int, float>>> vertexBoneWeights; // [vertex][(boneIndex, weight)]
    std::vector<Vec3> originalVertexPositions; // For skinning calculations
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
class Triangle : public Hittable {
public:
    // ========================================================================
    // Constructors
    // ========================================================================
    
    Triangle();
    
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

    inline uint16_t getMaterialID() const { return materialID; }

    // ========================================================================
    // Vertex Access Methods (replace direct member access)
    // ========================================================================
    
    inline const Vec3& getVertexPosition(int i) const { return vertices[i].position; }
    inline void setVertexPosition(int i, const Vec3& pos) { 
        vertices[i].position = pos; 
        aabbDirty = true;
        vertexPositionsDirty = true;
    }
    
    inline const Vec3& getVertexNormal(int i) const { return vertices[i].normal; }
    inline void setVertexNormal(int i, const Vec3& normal) { 
        vertices[i].normal = normal; 
        vertexPositionsDirty = true;
    }
    
    inline const Vec3& getOriginalVertexPosition(int i) const { return vertices[i].original; }
    inline void setOriginalVertexPosition(int i, const Vec3& pos) { vertices[i].original = pos; }
    
    inline const Vec3& getOriginalVertexNormal(int i) const { return vertices[i].originalNormal; }
    inline void setOriginalVertexNormal(int i, const Vec3& normal) { vertices[i].originalNormal = normal; }

    // Source polygon (quad/ngon) id this triangle was split from, or -1 if none. Lets sculpt /
    // shading compute one normal per polygon (loop normals) instead of per split triangle.
    inline int getFaceIndex() const { return faceIndex; }
    inline void setFaceIndex(int idx) { faceIndex = idx; }

    // Convenience accessors for backward compatibility
    inline Vec3 getV0() const { return vertices[0].position; }
    inline Vec3 getV1() const { return vertices[1].position; }
    inline Vec3 getV2() const { return vertices[2].position; }
    inline Vec3 getN0() const { return vertices[0].normal; }
    inline Vec3 getN1() const { return vertices[1].normal; }
    inline Vec3 getN2() const { return vertices[2].normal; }

    // ========================================================================
    // Material Access (ID-based)
    // ========================================================================
    
  
    inline void setMaterialID(uint16_t id) { materialID = id; }
    
    // Legacy compatibility - fetches from MaterialManager
    std::shared_ptr<Material> getMaterial() const;
    void setMaterial(const std::shared_ptr<Material>& mat);

    // ========================================================================
    // Transform Management
    // ========================================================================
    
    void setTransformHandle(std::shared_ptr<Transform> handle) { transformHandle = handle; }
    std::shared_ptr<Transform> getTransformHandle() const { return transformHandle; }
    // Fast non-owning accessor to avoid atomic refcount operations in hot paths.
    // Returns a raw pointer to the internal Transform (may be nullptr).
    inline Transform* getTransformPtr() const noexcept { return transformHandle.get(); }
    
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
    
    Vec2 t0, t1, t2; // Texture coordinates (kept as-is, essential for rendering)

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
    size_t getUVSetCount() const { return uv_sets.empty() ? size_t{ 1 } : uv_sets.size(); }
    void applyUVSet(size_t set_index);

    // ========================================================================
    // Skinning (Optional - only allocated for rigged triangles)
    // ========================================================================
    
    void initializeSkinData();
    bool hasSkinData() const { return static_cast<bool>(skinData); }
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
        static const std::string emptyString = "";
        return nodeNamePtr ? *nodeNamePtr : emptyString; 
    }
    void setNodeName(const std::string& name);
    bool isTriangle() const override { return true; }
    
    void setAssimpVertexIndices(unsigned int i0, unsigned int i1, unsigned int i2) {
        assimpVertexIndices = { i0, i1, i2 };
    }
    inline const std::array<unsigned int, 3>& getAssimpVertexIndices() const {
        return assimpVertexIndices;
    }

    // ========================================================================
    // GPU/Texture Bundle (for OptiX integration)
    // ========================================================================
    
    // Heap-allocated ONLY when the mesh actually carries textures (set at import / material
    // assignment). 8 bytes when absent vs the 104-byte inline struct — procedural / dense
    // subdivided meshes (the 32M case) have no textures, so this is the single biggest soup
    // saving. The modern GPU path binds textures per-material (unified_converters), so nothing
    // reads this per-triangle; it only keeps imported CUDA texture handles alive.
    deep_ptr<OptixGeometryData::TextureBundle> textureBundle;
    std::shared_ptr<Texture> texture;

    // ========================================================================
    // AABB Caching (Lazy Evaluation)
    // ========================================================================
    
    void update_bounding_box() const;
    inline void markAABBDirty() { 
        aabbDirty = true; 
        vertexPositionsDirty = true;
    }

    // ========================================================================
    // Legacy Compatibility (Direct vertex access - use accessors instead)
    // Marked for potential deprecation
    // ========================================================================
    
    // These provide direct access but should be migrated to accessor methods
    Vec3& v0_ref() { return vertices[0].position; }
    Vec3& v1_ref() { return vertices[1].position; }
    Vec3& v2_ref() { return vertices[2].position; }
    Vec3& n0_ref() { return vertices[0].normal; }
    Vec3& n1_ref() { return vertices[1].normal; }
    Vec3& n2_ref() { return vertices[2].normal; }

    // Read-only direct access for performance-critical code
    const Vec3& v0_cref() const { return vertices[0].position; }
    const Vec3& v1_cref() const { return vertices[1].position; }
    const Vec3& v2_cref() const { return vertices[2].position; }
    const Vec3& n0_cref() const { return vertices[0].normal; }
    const Vec3& n1_cref() const { return vertices[1].normal; }
    const Vec3& n2_cref() const { return vertices[2].normal; }
    const std::string* nodeNamePtr = nullptr;
    TriangleVertexData vertices[3];           // Consolidated vertex data (144 bytes)
    int terrain_id = -1;
    mutable bool vertexPositionsDirty = false;
    std::vector<std::array<Vec2, 3>> uv_sets;
    // Terrain ID if this is a terrain triangle
private:
    // ========================================================================
    // Optimized Data Members
    // ========================================================================
    
   
    uint16_t materialID = 0xFFFF;                       // Material lookup ID (2 bytes)
   
  
    std::shared_ptr<Transform> transformHandle; // Shared transform (8 bytes)
    deep_ptr<SkinnedTriangleData> skinData;      // Skinning data — 8 bytes when absent (was
                                                 // std::optional, which reserved ~56 B inline)
    
                     // 32 bytes avg
    int faceIndex = -1;                        // 4 bytes
    std::array<unsigned int, 3> assimpVertexIndices = { 0, 0, 0 }; // 12 bytes

    // AABB Caching
    mutable AABB cachedAABB;                  // 24 bytes
    mutable bool aabbDirty = true;            // 1 byte
  
    mutable Material* cachedMaterial = nullptr; // Performance cache

    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    // syncLegacyMembers() - REMOVED (no longer needed)
};

#endif // TRIANGLE_H

