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
#include <functional>
#include <embree4/rtcore.h>
#include "json.hpp"
#include "ForceField.h" // Add ForceField support
#include <mutex>

// Forward declarations
class Triangle;
class TriangleMesh;
class Mesh;
class Hittable;
class Renderer;

namespace RayTrophiSim {
class SimulationForceFieldSnapshot;
}

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
 * @brief A groom's scalp binding — (flat mesh, face index) pairs, NOT facade objects.
 *
 * A groom has to remember which face each guide sits on so skinning and brush-added
 * strands can rebuild the surface frame every frame. It used to keep a
 * `std::vector<std::shared_ptr<Triangle>>` holding a materialized facade for EVERY face
 * of the scalp mesh. After the DNA/SoA migration that is pure overhead: a facade is
 * ~150 B (object + make_shared control block), so a 1M-face scalp cost ~166 MB of
 * objects whose only real content is `parentMesh + faceIndex`. Worse, each facade owns a
 * `shared_ptr<TriangleMesh>`, so a live groom pinned the whole scalp geometry alive even
 * after the object was deleted from the scene.
 *
 * We store the two numbers instead and rebuild a facade on the STACK on demand
 * (`resolve()`, zero allocation). The mesh handles are weak, so a deleted scalp frees
 * its geometry; `HairSystem::rebindGroomsToScene` re-attaches them by node name (which
 * is also what restores the binding after a project reload — it is not serialized).
 */
struct HairBoundSurface {
    // Flat parent meshes, deduped, in scene order. A multi-material import splits one
    // node into several sibling TriangleMeshes sharing a nodeName, so this is a list.
    std::vector<std::weak_ptr<TriangleMesh>> meshes;
    std::vector<uint32_t> faceMesh;   // slot -> index into `meshes`
    std::vector<uint32_t> faceIndex;  // slot -> face inside that mesh
    // ONLY populated for pre-flat standalone Triangle soup (legacy projects / non-flat
    // sources). When non-empty it is the authority and the flat arrays are unused.
    std::vector<std::shared_ptr<Triangle>> legacy;

    size_t size() const { return legacy.empty() ? faceIndex.size() : legacy.size(); }
    bool empty() const { return size() == 0; }
    void clear();

    // Build from a transient facade list (what generation is handed by the UI/scripting).
    // Facades carrying a parentMesh collapse to (mesh, face); anything else falls back to
    // `legacy` so the old standalone-soup path keeps working unchanged.
    void build(const std::vector<std::shared_ptr<Triangle>>& tris);

    // Lock every parent mesh ONCE — call outside a per-strand loop, then pass the result
    // to resolve() so the hot loop pays one atomic per strand instead of a weak_ptr lock.
    std::vector<std::shared_ptr<TriangleMesh>> lockMeshes() const;

    // Bind `scratch` to slot `i` and return a Triangle view of it, or nullptr when the
    // slot is out of range or its mesh has been destroyed. The legacy path returns the
    // stored object and leaves `scratch` alone.
    const Triangle* resolve(const std::vector<std::shared_ptr<TriangleMesh>>& locked,
                            size_t i, Triangle& scratch) const;

    // Materialize every slot as a real facade. ONLY for generation, which needs a
    // std::vector<std::shared_ptr<Triangle>>; the result is transient and must not be stored.
    std::vector<std::shared_ptr<Triangle>> materialize() const;

    // True when at least one parent mesh is still alive (or this is a legacy binding).
    bool isAlive() const;

    // True when the bound scalp carries skin weights — the test that routes a groom to
    // the skinned update instead of the rigid-transform one.
    //
    // Cached: TriangleMesh::hasSkinWeights() walks the whole per-vertex weight array, and
    // this is asked once per groom per frame from updateAllTransforms. Whether a mesh is
    // rigged does not change without a rebind, which resets the cache.
    bool hasSkinData() const;
    void invalidateSkinCache() { m_skinCached = false; }

private:
    mutable bool m_skinCached = false;
    mutable bool m_skinCachedValue = false;
public:
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
    HairBoundSurface bound;                 // (flat mesh, face) binding — see HairBoundSurface
    Matrix4x4 transform;                    // Delta transform (from initial position)
    Matrix4x4 initialMeshTransform;         // Mesh transform when hair was generated
    
    std::string materialName;               // Shared material reference
    HairMaterialParams material;            // Per-groom material settings
    bool isDirty = true;                    // Needs rebuild
    bool isVisible = true;                  // Render toggle
};

// ── Procedural interpolated-child generation ────────────────────────────────────────
// Single source of truth for interpolated child geometry. Deterministic per
// (guideIndex, childIndex, baseSeed). Fills `child` with control points in the guide's
// LOCAL space (same space as guide.points) plus per-strand metadata — the caller applies
// any groom transform afterwards. Called by BOTH the materialized path
// (interpolateChildren → CPU/OptiX) and the procedural Vulkan upload, so every backend
// that uses it produces identical children. baseSeed convention: 54321 + groomName.length().
void generateChildStrand(HairStrand& child, const HairStrand& guide,
                         uint32_t guideIndex, uint32_t childIndex,
                         const HairGenerationParams& params, uint32_t baseSeed);

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
     * @brief Re-run generation for a groom using its STORED scalp binding.
     * Replaces the old "copy groom.boundTriangles out, hand it back to generateOnMesh"
     * dance, which only worked while the groom kept a facade per face alive.
     * @return false when the groom is unknown or its scalp mesh is gone.
     */
    bool regenerateGroom(const std::string& groomName, const HairGenerationParams& params);

    /**
     * @brief (Re)attach every groom whose scalp binding is empty or dead to a scene mesh
     *        matching its boundMeshName.
     *
     * The binding is NOT serialized (it is derived from the scene), so after a project
     * reload every groom starts unbound and, before this existed, silently stopped
     * following its mesh — the old fallback scanned world.objects for `Triangle`, which
     * a flat scene no longer contains. Cheap no-op once everything is bound.
     */
    void rebindGroomsToScene(const std::vector<std::shared_ptr<Hittable>>& sceneObjects);

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
     * @brief Mark the CPU (Embree) acceleration structure as needing a rebuild.
     *
     * DEFERRED: this no longer builds anything. The Embree scene is only ever read by the
     * CPU/Embree renderer and by brush picking, but it was being rebuilt from scratch on
     * every groom change — including every frame of a skinned animation under Vulkan RT
     * or OptiX, where nothing reads it at all. The build now happens on first use (see
     * ensureBVH), so an animating groom pays for it exactly zero times.
     */
    void buildBVH(bool includeInterpolated = true);

    /** Force any pending CPU BVH build to happen now. Rarely needed — the intersection
     *  entry points call it themselves. */
    void ensureBVH() const;
    
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

    // Advance the dynamics frame token. Call ONCE per viewport frame (top of the main loop).
    // updateAllTransforms is invoked more than once per frame (animation update + UI draw),
    // so updateRigidDynamicGroom uses this token to integrate the Verlet at most once per
    // frame. Left at 0 in headless/offline paths that never call it → per-frame dedup is
    // simply disabled there (they already invoke updateAllTransforms once per frame).
    void beginDynamicsFrame() { ++m_dynamicsFrame; }
    // Current viewport frame token (0 = headless/offline, dedup disabled). Also used by
    // Renderer::uploadHairToGPU to coalesce the several hair uploads that can otherwise
    // fire in a single frame (see the per-frame guard there).
    uint64_t dynamicsFrame() const { return m_dynamicsFrame; }

    // Procedural interpolated children (#1, Vulkan-first). When true (default) child
    // geometry is NOT materialized into groom.interpolated — the Vulkan upload generates
    // it on the fly, eliminating the persistent millions-of-HairStrand RAM cost. CPU/OptiX
    // read groom.interpolated, so they render children only when this is false (until they
    // are ported to the procedural generator).
    void setProceduralChildren(bool enabled) { m_proceduralChildren = enabled; }
    bool proceduralChildren() const { return m_proceduralChildren; }
    
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
    void restyleGroom(const std::string& name, const RayTrophiSim::SimulationForceFieldSnapshot* forceSnapshot, float time);
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
    // Dynamic hair on a RIGID (non-skinned) scalp. Bakes the moved rest shape into world-space
    // groomedPositions, pins the groom transform to identity, and runs the Verlet pass so the
    // strands lag/swing with inertia when the object is moved by a node/timeline anim or the
    // gizmo — the plain updateGroomTransform only rigid-follows and looks frozen.
    void updateRigidDynamicGroom(const std::string& groomName, const Matrix4x4& currentMeshTransform);

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

    // Per-frame dynamics environment (force fields). Set ONCE per frame before updateAllTransforms.
    // The dynamic-hair tick applies this force in its Verlet AND uses `time` as the "field is
    // live" signal: while time changes (sim/animated field) the strands keep responding; when it
    // stops changing they damp to the field's deflected equilibrium and settle (no perpetual
    // re-render). A null snapshot means no field this frame.
    void setDynamicsEnvironment(const RayTrophiSim::SimulationForceFieldSnapshot* force, float time) {
        m_envForce = force; m_envTime = time;
    }
    
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

    // Mutable: the CPU BVH is a lazily materialized cache, so the const intersection
    // entry points must be able to build it on demand.
    mutable RTCScene m_embreeScene = nullptr;
    // "Geometry changed, consumers should resync" latch. Cleared by buildBVH() exactly as
    // before, so isBVHDirty() keeps driving the GPU re-upload at the same cadence.
    bool m_bvhDirty = true;
    uint64_t m_dynamicsFrame = 0;   // viewport frame token for rigid-dynamic per-frame dedup (see beginDynamicsFrame)
    const RayTrophiSim::SimulationForceFieldSnapshot* m_envForce = nullptr;  // per-frame force field (setDynamicsEnvironment)
    float m_envTime = 0.0f;                                                   // force-field time; a change == field is live
    // Separate: is the Embree scene itself stale. Set by buildBVH(), cleared by the lazy
    // ensureBVH(). Splitting the two is what lets an animating groom re-upload to the GPU
    // every frame without also rebuilding a CPU BVH nobody reads.
    mutable bool m_cpuBvhDirty = true;
    mutable bool m_bvhIncludeInterpolated = true;  // argument of the pending buildBVH()
    bool m_proceduralChildren = true;   // see setProceduralChildren()

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
    
    // Does the actual Embree scene build. Non-const (it clears per-groom dirty flags);
    // ensureBVH() is the const entry point that drives it.
    void buildBVHImpl(bool includeInterpolated);

    void interpolateChildren(HairGroom& groom);
    void restyleGroomImpl(const std::string& name,
                          const std::function<Vec3(const Vec3&, const Vec3&)>& forceSampler,
                          float time);
    
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
