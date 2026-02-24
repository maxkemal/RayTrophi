/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          HairSystem.h
 * Author:        Kemal Demirtaş
 * Description:   Main hair/fur management system
 *                Handles generation, grooming, and BVH building
 * =========================================================================
 */
#ifndef HAIR_SYSTEM_H
#define HAIR_SYSTEM_H

#include "Hair/HairStrand.h"
#include "Hair/HairBSDF.h"
#include "Vec3.h"
#include "Matrix4x4.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <embree4/rtcore.h>
#include "json.hpp"
#include "ForceField.h" // Add ForceField support
#include <mutex>

// Forward declarations
class Triangle;
class Mesh;
class Hittable;
class Renderer;

namespace Hair {

/**
 * @brief Hair generation parameters (like Blender's particle hair)
 */
struct HairGenerationParams {
    // Strand count & distribution
    uint32_t guideCount = 1000;         // Number of guide strands
    uint32_t interpolatedPerGuide = 4;  // Children per guide
    uint32_t pointsPerStrand = 8;       // Control points (4-16 typical)
    
    // Physical properties
    float length = 0.1f;                // Base length in world units
    float lengthVariation = 0.2f;       // Random length variation (0-1)
    float rootRadius = 0.001f;          // Radius at root (1mm default)
    float tipRadius = 0.0001f;          // Radius at tip (0.1mm default)
    
    // Styling
    float clumpiness = 0.5f;            // How much strands attract to guides
    float childRadius = 0.01f;          // Radius for spawning children around guide
    
    float curlFrequency = 0.0f;         // Helical curl waves per strand
    float curlRadius = 0.01f;           // Helical curl amplitude
    
    float waveFrequency = 0.0f;         // Sinusoidal wave frequency
    float waveAmplitude = 0.0f;         // Sinusoidal wave amplitude
    
    float frizz = 0.0f;                 // Random displacement (high frequency)
    float roughness = 0.0f;             // Random displacement (low frequency)
    
    float gravity = 0.0f;               // Downward bend (0-1)
    float forceInfluence = 1.0f;        // How much external force fields affect this groom

    // Physics / Dynamics
    bool useDynamics = false;           // Enable real-time Verlet physics
    float physicsDamping = 0.95f;       // Velocity damping (0-1)
    float physicsStiffness = 0.1f;      // Shape retention (0-1)
    float physicsMass = 1.0f;           // Particle mass

    
    // Rendering
    uint16_t defaultMaterialID = 0;     // Hair material
    bool useTangentShading = true;      // Anisotropic highlight
    
    // Quality & Curves
    bool useBSpline = true;            // Use B-Spline curves for smoothness (otherwise linear)
    uint32_t subdivisions = 2;          // Embree tessellation level (0-4)


    // Serialization
    void to_json(nlohmann::json& j, const HairGenerationParams& p);
    void from_json(const nlohmann::json& j, HairGenerationParams& p);
};


/**
 * @brief Hair groom/asset container
 */
struct HairGroom {
    std::string name;
    std::vector<HairStrand> guides;         // Master strands
    std::vector<HairStrand> interpolated;   // Generated children
    HairGenerationParams params;
    
    // Binding info
    std::string boundMeshName;              // Source scalp mesh
    std::vector<std::shared_ptr<Triangle>> boundTriangles; // All triangles of the bound mesh (for skinning)
    std::weak_ptr<Triangle> boundMeshPtr;   // Cached pointer to source scalp mesh (for fast transform updates - kept for legacy)
    Matrix4x4 transform;                    // Delta transform (from initial position)
    Matrix4x4 initialMeshTransform;         // Mesh transform when hair was generated
    
    std::string materialName;               // Shared material reference
    HairMaterialParams material;            // Per-groom material settings
    bool isDirty = true;                    // Needs rebuild
    bool isVisible = true;                  // Render toggle
};

/**
 * @brief Main Hair/Fur System
 * 
 * Usage:
 *   HairSystem hair;
 *   hair.generateOnMesh(scalpMesh, params);
 *   hair.buildBVH();
 *   // In render loop: hair.intersect(ray, hitInfo)
 */
class HairSystem {
public:
    HairSystem();
    ~HairSystem();
    
    // ========================================================================
    // Material Management
    // ========================================================================
    
    void addMaterial(const std::string& name, const HairMaterialParams& params);
    HairMaterialParams* getSharedMaterial(const std::string& name);
    const HairMaterialParams* getSharedMaterial(const std::string& name) const;
    void removeMaterial(const std::string& name);
    std::vector<std::string> getMaterialNames() const;
    void assignMaterialToGroom(const std::string& groomName, const std::string& materialName);

    
    // ========================================================================
    // Generation
    // ========================================================================
    
    /**
     * @brief Generate hair strands on a mesh surface
     * @param triangles Source mesh triangles (scalp/skin)
     * @param params Generation parameters
     * @param groomName Unique name for this groom
     */
    void generateOnMesh(
        const std::vector<std::shared_ptr<Triangle>>& triangles,
        const HairGenerationParams& params,
        const std::string& groomName = "default"
    );
    
    /**
     * @brief Generate fur with undercoat + guard hairs
     */
    void generateFur(
        const std::vector<std::shared_ptr<Triangle>>& triangles,
        const HairGenerationParams& undercoatParams,
        const HairGenerationParams& guardParams,
        const std::string& groomName = "default"
    );
    
    /**
     * @brief Import hair from Alembic (.abc) groom file
     */
    bool importAlembic(const std::string& filepath, const std::string& groomName);
    

    // ========================================================================
    // BVH & Intersection
    // ========================================================================
    
    /**
     * @brief Build acceleration structure for all grooms
     * Uses Embree's curve primitives (RTC_GEOMETRY_TYPE_ROUND_BEZIER_CURVE)
     */
    void buildBVH(bool includeInterpolated = true);
    
    /**
     * @brief Ray-hair intersection (CPU)
     * @return true if hit, fills hitInfo
     */
    bool intersect(
        const Vec3& rayOrigin,
        const Vec3& rayDir,
        float tMin,
        float tMax,
        struct HairHitInfo& hitInfo
    ) const;

    /**
     * @brief Volumetric intersection for magnetic grooming
     * Finds the closest hair point to the ray within searchRadius. 
     * Essential for styling where hair is thin.
     */
    bool intersectVolumetric(
        const Vec3& rayOrigin,
        const Vec3& rayDir,
        float tMin,
        float tMax,
        float searchRadius,
        struct HairHitInfo& hitInfo
    ) const;
    
    /**
     * @brief Fast shadow occlusion test (uses rtcOccluded1)
     * @return true if ray is blocked by any hair strand
     * @note Much faster than intersect() for shadow testing
     */
    bool occluded(
        const Vec3& rayOrigin,
        const Vec3& rayDir,
        float tMin,
        float tMax
    ) const;
    
    // Serialization
    // Serialization
    // Optimized to write geometry to binary stream if provided
    nlohmann::json serialize(std::ostream* binaryOut = nullptr) const;
    void deserialize(const nlohmann::json& j, std::istream* binaryIn = nullptr);

    
    // ========================================================================
    // GPU Upload
    // ========================================================================
    
    /**
     * @brief Prepare data for GPU rendering
     * @return GPU-ready flat buffers
     */
    HairGPUData prepareGPUData() const;
    
    /**
     * @brief Get OptiX-ready curve data for GPU rendering
     * @param outVertices Output: float4 array (x, y, z, radius)
     * @param outIndices Output: segment start indices
     * @param outTangents Output: tangent per segment
     * @param outVertexCount Output: number of vertices
     * @param outSegmentCount Output: number of segments
     */
    bool getOptiXCurveData(
        std::vector<float>& outVertices4,   // x,y,z,r packed
        std::vector<unsigned int>& outIndices,
        std::vector<uint32_t>& outStrandIDs,
        std::vector<float>& outTangents3,   // x,y,z packed
        std::vector<float>& outRootUVs2,    // u,v packed per-segment
        std::vector<float>& outStrandV,
        size_t& outVertexCount,
        size_t& outSegmentCount,
        bool includeInterpolated = true
    ) const;
    
    bool getOptiXCurveDataByGroom(
        const std::string& groomName,
        std::vector<float>& outVertices4,
        std::vector<unsigned int>& outIndices,
        std::vector<uint32_t>& outStrandIDs,
        std::vector<float>& outTangents3,
        std::vector<float>& outRootUVs2,
        std::vector<float>& outStrandV,
        size_t& outVertexCount,
        size_t& outSegmentCount,
        HairMaterialParams& outMatParams,
        int& outMatID,
        int& outMeshMatID,
        bool includeInterpolated = true
    ) const;
    
    // ========================================================================
    // Accessors
    // ========================================================================
    
    size_t getTotalStrandCount() const;
    size_t getTotalPointCount() const;
    size_t getGroomCount() const { return m_grooms.size(); }
    bool isBVHDirty() const { return m_bvhDirty; }
    
    HairGroom* getGroom(const std::string& name);
    const HairGroom* getGroom(const std::string& name) const;
    
    std::vector<std::string> getGroomNames() const;
    void clearAll();

    void removeGroom(const std::string& name);
    bool renameGroom(const std::string& oldName, const std::string& newName);
    
    // ========================================================================
    // Styling (real-time editing)
    // ========================================================================
    
    void setGravity(const std::string& groomName, float gravity);
    void setClumpiness(const std::string& groomName, float clump);
    void restyleGroom(const std::string& name, const Physics::ForceFieldManager* forceManager = nullptr, float time = 0.0f);
    void regenerateInterpolated(const std::string& groomName);
    
    /**
     * @brief Capture current groomedPositions as the new Rest/Bind Pose relative to the current mesh pose.
     * Essential for tools (Comb, Cut, etc.) to persist changes on Skinned Grooms.
     */
    void bakeGroomToRest(const std::string& groomName);


    
    // Check if groom name exists
    bool exists(const std::string& name) const { return m_grooms.find(name) != m_grooms.end(); }


    
    // ========================================================================
    // Transform & Binding
    // ========================================================================
    
    /**
     * @brief Update groom transform from bound mesh
     * Call this when the mesh moves/rotates/scales
     */
    void updateGroomTransform(const std::string& groomName, const Matrix4x4& meshTransform);
    
    /**
     * @brief Get transformed hair positions for rendering
     * Applies groom.transform to all points
     */
    Vec3 getTransformedPosition(const HairStrand& strand, size_t pointIndex, const Matrix4x4& transform) const;
    
    /**
     * @brief Mark groom as needing BVH rebuild (after transform change)
     */
    void markDirty(const std::string& groomName);
    
    /**
     * @brief Update all grooms from their bound meshes
     */
    void updateAllTransforms(const std::vector<std::shared_ptr<Hittable>>& sceneObjects, const std::vector<Matrix4x4>& boneMatrices);
    
    /**
     * @brief Update groom that is bound to a specific mesh by its boundMeshName
     * This is the correct way to update transforms when mesh moves
     */
    void updateFromMeshTransform(const std::string& meshName, const Matrix4x4& meshTransform);
    
    /**
     * @brief Update hair strands to follow skinned mesh deformation
     */
    void updateSkinnedGroom(const std::string& groomName, const std::vector<Matrix4x4>& boneMatrices);
    
    /**
     * @brief Find groom bound to a specific mesh
     */
    HairGroom* getGroomByMesh(const std::string& meshName);
    
    /**
     * @brief Add strands at position (for paint mode)
     */
    void addStrandsAtPosition(const std::string& groomName, const Vec3& position, 
                              const Vec3& normal, float radius, int count);
    
    /**
     * @brief Remove strands at position (for paint mode)
     */
    void removeStrandsAtPosition(const std::string& groomName, const Vec3& position, 
                                 float radius);
    
private:
    std::unordered_map<std::string, HairGroom> m_grooms;
    std::unordered_map<std::string, HairMaterialParams> m_materials; // Shared material pool
    std::unordered_map<unsigned int, std::string> m_geomToGroom; // Map Embree geomID to groom name
    std::unordered_map<unsigned int, size_t> m_geomToTangentOffset; // Map Embree geomID to tangent buffer offset
    
    mutable std::recursive_mutex m_mutex;
    
    // Embree scene for hair BVH

    RTCScene m_embreeScene = nullptr;
    bool m_bvhDirty = true;

    // Cached stats for O(1) retrieval
    mutable size_t m_totalStrandCount = 0;
    mutable size_t m_totalPointCount = 0;
    mutable bool m_statsDirty = true;

    void refreshStats() const;

    
    // For mapping primID to actual strand/segment info
    struct SegmentMap {
        uint32_t globalStrandID; // Used for random variations (stable hash)
        uint32_t localStrandIdx; // Used for retrieving guide/interpolated data
        float vStart;
        float vStep;
    };
    std::vector<SegmentMap> m_segMap;
    
    // High-performance tangent buffer for smooth shading
    // Stores [T_start, T_mid, T_end] triplets for each segment
    std::vector<Vec3> m_smoothTangents; 
    

    // Internal helpers
    void generateGuideStrands(
        HairGroom& groom,
        const std::vector<std::shared_ptr<Triangle>>& triangles
    );
    
    void interpolateChildren(HairGroom& groom);
    
    Vec3 sampleTriangleSurface(
        const Triangle& tri,
        float u, float v,
        Vec3& outNormal,
        Vec2& outUV
    ) const;
    void applyGravityToStrand(HairStrand& strand, float gravity);
    void applyCurlToStrand(HairStrand& strand, float frequency, float radius);
};

/**
 * @brief Hair intersection result
 */
struct HairHitInfo {
    float t;                    // Ray parameter
    Vec3 position;              // World hit position
    Vec3 tangent;               // Hair direction at hit
    Vec3 normal;                // Shading normal (perpendicular to tangent)
    float v;                    // Parametric position along strand (0=root, 1=tip)
    float u;                    // Parametric position around strand
    uint32_t strandID;          // Which strand was hit
    uint16_t materialID;        // Material for shading
    HairMaterialParams material; // Full material parameters for the hit groom
    std::string groomName;      // Name of the hit groom
    Vec2 rootUV;                // UV on scalp for texture variation
    uint16_t meshMaterialID;    // Inherited material ID from scalp mesh
};


} // namespace Hair

#endif // HAIR_SYSTEM_H
