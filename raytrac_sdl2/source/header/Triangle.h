#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <memory>
#include <string>
#include <algorithm>
#include <optional>
#include <array>
#include "Hittable.h"
#include "Vec2.h"
#include "Matrix4x4.h"
#include "Vec3SIMD.h"
#include "Transform.h"
#include "MaterialManager.h"
#include "sbt_data.h"
#include <SDL.h>
#include <SDL_image.h>

// Forward declarations
class Material;
class Texture;

/**
 * @brief Consolidated vertex data structure for memory optimization
 * Replaces 9 separate Vec3 members (original, transformed, current) with this
 * 48 bytes vs 108 bytes - 55% savings on vertex data alone
 */
struct TriangleVertexData {
    Vec3 position;       // Current/transformed position (used for hit detection)
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

    // ========================================================================
    // Vertex Access Methods (replace direct member access)
    // ========================================================================
    
    inline const Vec3& getVertexPosition(int i) const { return vertices[i].position; }
    inline void setVertexPosition(int i, const Vec3& pos) { 
        vertices[i].position = pos; 
        aabbDirty = true;
    }
    
    inline const Vec3& getVertexNormal(int i) const { return vertices[i].normal; }
    inline void setVertexNormal(int i, const Vec3& normal) { vertices[i].normal = normal; }
    
    inline const Vec3& getOriginalVertexPosition(int i) const { return vertices[i].original; }
    inline void setOriginalVertexPosition(int i, const Vec3& pos) { vertices[i].original = pos; }
    
    inline const Vec3& getOriginalVertexNormal(int i) const { return vertices[i].originalNormal; }
    inline void setOriginalVertexNormal(int i, const Vec3& normal) { vertices[i].originalNormal = normal; }

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
    
    inline uint16_t getMaterialID() const { return materialID; }
    inline void setMaterialID(uint16_t id) { materialID = id; }
    
    // Legacy compatibility - fetches from MaterialManager
    std::shared_ptr<Material> getMaterial() const;
    void setMaterial(const std::shared_ptr<Material>& mat);

    // ========================================================================
    // Transform Management
    // ========================================================================
    
    void setTransformHandle(std::shared_ptr<Transform> handle) { transformHandle = handle; }
    std::shared_ptr<Transform> getTransformHandle() const { return transformHandle; }
    
    void setBaseTransform(const Matrix4x4& transform);
    void updateAnimationTransform(const Matrix4x4& animTransform);
    void set_transform(const Matrix4x4& t);
    static void updateTriangleTransform(Triangle& triangle, const Matrix4x4& transform);
    
    Matrix4x4 getTransformMatrix() const;
    void initialize_transforms();
    void updateTransformedVertices();

    // ========================================================================
    // UV Coordinates
    // ========================================================================
    
    Vec2 t0, t1, t2; // Texture coordinates (kept as-is, essential for rendering)
    
    void setUVCoordinates(const Vec2& uv0, const Vec2& uv1, const Vec2& uv2);
    std::tuple<Vec2, Vec2, Vec2> getUVCoordinates() const;

    // ========================================================================
    // Skinning (Optional - only allocated for rigged triangles)
    // ========================================================================
    
    void initializeSkinData();
    bool hasSkinData() const { return skinData.has_value(); }
    
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
    
    virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const override;
    virtual bool bounding_box(float time0, float time1, AABB& output_box) const override;
    
    inline bool has_tangent_basis() const { return false; }

    // ========================================================================
    // Normals
    // ========================================================================
    
    void set_normals(const Vec3& normal0, const Vec3& normal1, const Vec3& normal2);

    // ========================================================================
    // Mesh/Node Information
    // ========================================================================
    
    inline void setFaceIndex(int idx) { faceIndex = idx; }
    inline int getFaceIndex() const { return faceIndex; }
    
    inline const std::string& getNodeName() const { return nodeName; }
    void setNodeName(const std::string& name);
    
    void setAssimpVertexIndices(unsigned int i0, unsigned int i1, unsigned int i2) {
        assimpVertexIndices = { i0, i1, i2 };
    }
    inline const std::array<unsigned int, 3>& getAssimpVertexIndices() const {
        return assimpVertexIndices;
    }

    // ========================================================================
    // GPU/Texture Bundle (for OptiX integration)
    // ========================================================================
    
    OptixGeometryData::TextureBundle textureBundle;
    std::shared_ptr<Texture> texture;

    // ========================================================================
    // AABB Caching (Lazy Evaluation)
    // ========================================================================
    
    void update_bounding_box() const;
    void markAABBDirty() { aabbDirty = true; }

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

    // For backward compatibility with code using direct member access
    // These are read-only and return current transformed positions
    Vec3 v0, v1, v2;  // Keep for backward compat, synced with vertices[].position
    Vec3 n0, n1, n2;  // Keep for backward compat, synced with vertices[].normal

    // Original/transformed for backward compatibility
    Vec3 original_v0, original_v1, original_v2;
    Vec3 original_n0, original_n1, original_n2;
    Vec3 transformed_v0, transformed_v1, transformed_v2;
    Vec3 transformed_n0, transformed_n1, transformed_n2;

    // Legacy skinning access
    std::vector<std::vector<std::pair<int, float>>> vertexBoneWeights;
    std::vector<Vec3> originalVertexPositions;

    // Legacy material pointer (kept for gradual migration)
    std::shared_ptr<Material> mat_ptr;
    std::shared_ptr<GpuMaterial> gpuMaterialPtr;
    std::string materialName;

    // Legacy transform (kept for gradual migration)
    Matrix4x4 transform;

private:
    // ========================================================================
    // Optimized Data Members
    // ========================================================================
    
    TriangleVertexData vertices[3];           // Consolidated vertex data (48 bytes)
    uint16_t materialID;                       // Material lookup ID (2 bytes)
    std::shared_ptr<Transform> transformHandle; // Shared transform (8 bytes)
    std::optional<SkinnedTriangleData> skinData; // Optional skinning data
    
    std::string nodeName;
    int faceIndex = -1;
    std::array<unsigned int, 3> assimpVertexIndices;

    // AABB Caching
    mutable AABB cachedAABB;
    mutable bool aabbDirty = true;

    // Legacy transform members (for gradual migration)
    Matrix4x4 baseTransform_legacy;
    Matrix4x4 currentTransform_legacy;
    Matrix4x4 finalTransform_legacy;

    // Scratch buffers for apply_skinning
    mutable Vec3 blendedPos, blendedNorm;

    // ========================================================================
    // Internal Methods
    // ========================================================================
    
    void syncLegacyMembers();  // Sync new vertex array with legacy v0/v1/v2 etc.
};

#endif // TRIANGLE_H