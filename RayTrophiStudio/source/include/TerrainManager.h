/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          TerrainManager.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

#include "TerrainSystem.h"
#include "json.hpp"
#include <vector>
#include <string>
#include <functional>
#include "FoliageFwd.h"
#include "Transform.h"

struct SceneData; // Forward decl
class Material;
class Texture;

// Serialization version - increment when format changes
static constexpr int TERRAIN_SERIALIZATION_VERSION = 3;

// ===========================================================================
// EROSION PARAMETERS
// ===========================================================================

enum class ErosionBoundaryMode : int {
    Preserve = 0, // Blend the simulation back to the authored border heights.
    Open = 1,     // Water/sediment may leave the domain; do not reshape geometry.
    SeaLevel = 2  // Blend border geometry toward an explicit normalized level.
};

struct HydraulicErosionParams {
    int iterations = 50000;        // Number of 'hits' (Determines passes in GPU)
    int dropletLifetime = 64;      // Max steps per droplet
    float inertia = 0.05f;         // direction momentum (0-1)
    float sedimentCapacity = 2.0f; // Sediment carrying capacity
    float minSlope = 0.005f;       // Minimal slope for flow
    float erodeSpeed = 0.05f;      // Erosion rate
    float depositSpeed = 0.1f;     // Deposit rate
    float evaporateSpeed = 0.01f;  // Evaporation rate
    float gravity = 9.8f;          // Gravitational acceleration
    int erosionRadius = 2;         // Default channel width
    float initialWater = 1.0f;     // Water carried by a newly spawned droplet
    float initialSpeed = 1.0f;     // Initial droplet velocity
    float uphillErosion = 0.3f;    // Momentum-driven erosion while climbing a cell
    float flatSettling = 1.0f;     // Multiplier for low-slope sediment settling
    float velocitySettling = 1.0f; // Multiplier for deceleration-driven settling
    float minWater = 0.01f;        // Droplet termination water threshold
    float minSpeed = 0.01f;        // Droplet termination velocity threshold
    bool removeSpikes = true;
    bool fillPits = true;
    bool smoothSurface = true;
    unsigned int seed = 1337u;     // Deterministic CPU/GPU droplet distribution
    ErosionBoundaryMode boundaryMode = ErosionBoundaryMode::Preserve;
    int boundaryWidth = 0;         // Cells; 0 selects a resolution-aware width
    float boundaryLevel = 0.0f;    // Normalized height used by SeaLevel mode
    bool channelEvolution = true;  // Mature accumulated runoff without a second Fluvial solve
    int channelIterations = 12;    // Race-free grid transport/evolution passes
    float channelErosion = 0.18f;  // Bed incision response to accumulated discharge
    float channelDeposition = 0.22f; // Low-energy sediment settling
    float channelWidthScale = 1.0f;  // Hydraulic geometry width multiplier
    float channelDepthScale = 1.0f;  // Hydraulic geometry depth multiplier
    bool macroDrainage = true;
    float macroValleyScaleMeters = 140.0f;
    float macroHeadwaterAreaKm2 = 0.012f;
    float macroValleyDepthMeters = 10.0f;
    float macroValleyFloor = 0.35f;
};

// Optional transient products emitted by the hydraulic solver. These are
// accumulated by the simulation itself (not reconstructed from height deltas).
struct HydraulicErosionFields {
    int width = 0;
    int height = 0;
    std::vector<float> erosion;
    std::vector<float> deposition;
    std::vector<float> discharge;
    std::vector<float> sediment;
    std::vector<float> directionX;
    std::vector<float> directionY;
    std::vector<float> channelWidth;
    std::vector<float> waterDepth;
    std::vector<float> waterLevel;

    void reset(int w, int h) {
        width = w; height = h;
        const size_t n = static_cast<size_t>(w) * static_cast<size_t>(h);
        erosion.assign(n, 0.0f); deposition.assign(n, 0.0f);
        discharge.assign(n, 0.0f); sediment.assign(n, 0.0f);
        directionX.assign(n, 0.0f); directionY.assign(n, 0.0f);
        channelWidth.assign(n, 0.0f); waterDepth.assign(n, 0.0f);
        waterLevel.assign(n, 0.0f);
    }
};

struct ThermalErosionParams {
    int iterations = 50;          // Moderate default
    float talusAngle = 0.5f;       // ~27 degrees
    float erosionAmount = 0.3f;    // Less aggressive
    float anisotropy = 0.0f;       // Directional thermal stress (0 isotropic)
    float anisotropyDirection = 0.0f; // Degrees in terrain XZ
    float talusSettling = 1.0f;    // Mobility of unstable debris
    float sedimentRemoval = 0.0f;  // Fraction of transported mass leaving domain
    bool fineDetail = false;
    float debrisSizeMeters = 1.0f;
};

class TerrainManager {
public:
    static TerrainManager& getInstance() {
        static TerrainManager instance;
        return instance;
    }
    // Keyframe Animation
    // We pass the track directly to avoid Scene dependency in this header
    void captureKeyframeToTrack(TerrainObject* terrain, struct ObjectAnimationTrack& track, int frame);
    void applyKeyframe(TerrainObject* terrain, const struct TerrainKeyframe& keyframe);
    
    // Helper to interpolate between two keyframes manually if needed (usually KeyframeSystem handles this)
    void updateFromTrack(TerrainObject* terrain, const struct ObjectAnimationTrack& track, int currentFrame);

    // Create a flat terrain grid
    // ★ height_scale is a CREATION parameter, not a post-edit. Callers used to
    // create the terrain, then assign heightmap.scale_y and call
    // updateTerrainMesh() a second time -- a full extra vertex+normal pass
    // (156 ms measured at 4096^2) whose only job was to re-apply a scalar the
    // first pass could have used.
    // ★ mesh_resolution is a CREATION parameter too, for the same reason
    // height_scale is: creating at the field resolution and decimating after
    // means paying the full acceleration-structure build once (5.6 s at 4096^2)
    // before saving anything. 0 = vertex grid follows the field.
    TerrainObject* createTerrain(SceneData& scene, int resolution, float size,
                                 float height_scale = 10.0f, int mesh_resolution = 0);
    
    // Create terrain from heightmap image (using stb_image)
    TerrainObject* createTerrainFromHeightmap(SceneData& scene, const std::string& filepath, float size, float maxHeight, int max_resolution = 1024);
    
    // Update mesh vertices based on heightmap (Call after sculpting)
    void updateTerrainMesh(TerrainObject* terrain, bool signalRebuild = true);

    // Rebuild mesh topology (Call when resolution changes)
    void rebuildTerrainMesh(SceneData& scene, TerrainObject* terrain);
    
    // Update only dirty sectors (incremental update for performance)
    void updateDirtySectors(TerrainObject* terrain, bool clearRegion = true);
    
    // ===========================================================================
    // NORMAL CALCULATION
    // ===========================================================================
    Vec3 calculateNormal(TerrainObject* terrain, int x, int y);  // Uses terrain->normal_quality
    Vec3 calculateSobelNormal(TerrainObject* terrain, int x, int y);  // 8-neighbor Sobel filter
    Vec3 calculateFastNormal(TerrainObject* terrain, int x, int y);   // 4-neighbor central difference
    
   
    // Sculpting
    // mode: 0=Raise, 1=Lower, 2=Flatten, 3=Smooth, 4=Stamp
    void sculpt(TerrainObject* terrain, const Vec3& hitPoint, int mode, float radius, float strength, float dt,
                float curve = 2.0f, float targetHeight = 0.0f,
                std::shared_ptr<class Texture> stampTexture = nullptr, float rotation = 0.0f,
                bool signalHeavyRebuild = true);
    
    void smoothTerrain(TerrainObject* terrain, int iterations);

    // Layer System & Painting
    void initLayers(TerrainObject* terrain);
    // channel: 0=R, 1=G, 2=B, 3=A
    void paintSplatMap(TerrainObject* terrain, const Vec3& hitPoint, int channel, float radius, float strength, float dt);
    // Auto-generate mask based on slope and height
    void autoMask(TerrainObject* terrain, float slopeWeight, float heightWeight, float heightMin, float heightMax, float slopeSteepness);
    
    // Internal helper to sync CPU splat data to GPU texture
    void updateSplatMapTexture(TerrainObject* terrain);
    // resizePaintMaps resizes both splatMap and macroColorMap to paintGridWidth/Height.
    // This is the canonical name; resizeSplatMap below is a compatibility forwarder.
    void resizePaintMaps(TerrainObject* terrain);
    inline void resizeSplatMap(TerrainObject* terrain) { resizePaintMaps(terrain); }
    void exportSplatMap(TerrainObject* terrain, const std::string& filepath);
    void importSplatMap(TerrainObject* terrain, const std::string& filepath);
    void exportHeightmap(TerrainObject* terrain, const std::string& filepath);

    // Flow Analysis (New)
    void calculateFlowMap(TerrainObject* terrain);

    // ===========================================================================
    // FOLIAGE SYSTEM
    // ===========================================================================    // Foliage
    void updateFoliage(TerrainObject* terrain, OptixWrapper* optix);
    void clearFoliage(TerrainObject* terrain, OptixWrapper* optix);
    void reapplyAllFoliage(OptixWrapper* optix); // Re-adds persistence after rebuild
    int migrateLegacyFoliageToInstanceGroups(SceneData& scene, bool clearLegacy = true);
    bool hasLegacyFoliage() const;
    
    // Serialization===========================================================================
    // EROSION SYSTEM
    // ===========================================================================
    // progressCallback (optional): invoked periodically with a 0..1 fraction of
    // iterations completed. These CPU loops are single-threaded and can run for
    // tens of seconds at default iteration counts on a background evaluate
    // thread — without this, the node-editor progress bar shows no movement for
    // the whole duration a single erosion node is running.
    void hydraulicErosion(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask = {}, const std::function<void(float)>& progressCallback = nullptr, HydraulicErosionFields* fields = nullptr);
    void hydraulicErosionAdvanced(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask = {});
    void fluvialErosion(TerrainObject* terrain, const HydraulicErosionParams& params,
                        const std::vector<float>& mask = {},
                        const std::function<void(float)>& progressCallback = nullptr,
                        const std::vector<float>& flowGuide = {});
    void fluvialErosionGPU(TerrainObject* terrain, const HydraulicErosionParams& params,
                           const std::vector<float>& mask = {},
                           const std::vector<float>& flowGuide = {});
    void hydraulicErosionGPU(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask = {}, HydraulicErosionFields* fields = nullptr);
    void hydraulicErosionMultiPass(TerrainObject* terrain,
                                   const std::vector<HydraulicErosionParams>& stages,
                                   bool useGPU, const std::vector<float>& mask = {},
                                   const std::function<void(float)>& progressCallback = nullptr,
                                   HydraulicErosionFields* fields = nullptr);
    void thermalErosionGPU(TerrainObject* terrain, const ThermalErosionParams& params, const std::vector<float>& mask = {});
    void thermalErosion(TerrainObject* terrain, const ThermalErosionParams& params, const std::vector<float>& mask = {}, const std::function<void(float)>& progressCallback = nullptr);

    void windErosion(TerrainObject* terrain, float strength, float direction, int iterations, const std::vector<float>& mask = {}, const std::function<void(float)>& progressCallback = nullptr);
    void windErosionGPU(TerrainObject* terrain, float strength, float direction, int iterations, const std::vector<float>& mask = {});
    
    // Edge preservation helpers (prevents cliffs/walls at terrain boundaries)
    void preserveEdges(TerrainObject* terrain, const std::vector<float>& originalHeights, int fadeWidth);
    int getEdgeFadeWidth(TerrainObject* terrain);
    
    // ===========================================================================
    // HARDNESS SYSTEM (for realistic erosion)
    // ===========================================================================
    void initHardnessMap(TerrainObject* terrain, float defaultHardness = 0.3f);
    void autoGenerateHardness(TerrainObject* terrain, float slopeWeight = 0.7f, float noiseAmount = 0.2f);
    void paintHardness(TerrainObject* terrain, const Vec3& hitPoint, float radius, float strength, float dt, bool increase);
    
    // Combined Wizard Process (Thermal -> Fluvial -> Wind)
    void applyCombinedErosion(TerrainObject* terrain, int iterations, float strength, bool useGPU = false);
    
    // Progress callback type for long-running operations
    using ProgressCallback = std::function<void(float progress, const std::string& stage)>;
    
    // Erosion with progress reporting
    void applyCombinedErosionWithProgress(TerrainObject* terrain, int iterations, float strength, ProgressCallback callback);
    
    // ===========================================================================
    // HEIGHTMAP EXPORT/IMPORT
    // ===========================================================================
   
    void importMaskChannel(TerrainObject* terrain, const std::string& filepath, int channel);
    
    // ===========================================================================
    // SERIALIZATION
    // ===========================================================================
    /**
     * @brief Serialize all terrains to JSON + binary files
     * @param terrainDir Directory to save terrain data
     * @return JSON object containing terrain metadata
     */
    nlohmann::json serialize(const std::string& terrainDir) const;
    
    /**
     * @brief Deserialize terrains from JSON + binary files
     * @param data JSON object containing terrain metadata
     * @param terrainDir Directory where terrain data is stored
     * @param scene SceneData reference to add triangles
     */
    void deserialize(const nlohmann::json& data, const std::string& terrainDir, SceneData& scene);
    
    /**
     * @brief Save heightmap to binary file (float32 raw format)
     */
    void saveHeightmapBinary(const TerrainObject* terrain, const std::string& filepath) const;
    
    /**
     * @brief Load heightmap from binary file
     */
    void loadHeightmapBinary(TerrainObject* terrain, const std::string& filepath);
    
    // Getters
    std::vector<TerrainObject>& getTerrains() { return terrains; }

    TerrainObject* getTerrain(int id);
    TerrainObject* getTerrainByName(const std::string& name);

    // ===========================================================================
    // HEIGHT SAMPLING & RAYCAST
    // ===========================================================================
    
    // Ray-Terrain Intersection (Ignores all other objects, perfect for sculpting/painting)
    // Returns true if hit, populates t_out and normal_out
    bool intersectRay(TerrainObject* terrain, const Ray& r, float& t_out, Vec3& normal_out, float t_min = 0.001f, float t_max = 1e9f);
    
    // Check if any terrain exists
    bool hasActiveTerrain() const { return !terrains.empty(); }
    
    // Sample height at world XZ coordinate
    float sampleHeight(float worldX, float worldZ) const {
        if (terrains.empty()) return 0.0f;
        
        // Find terrain containing this point
        for (const auto& terrain : terrains) {
            const Heightmap& hm = terrain.heightmap;
            if (hm.data.empty() || hm.width <= 0 || hm.height <= 0) continue;
            
            // 1. Transform World position to Local terrain space
            Vec3 localPos(worldX, 0, worldZ);
            if (terrain.transform) {
                Matrix4x4 inv = terrain.transform->getFinal().inverse();
                localPos = inv.multiplyVector(Vec4(worldX, 0, worldZ, 1.0f)).xyz();
            }

            // 2. Check if local position is within terrain bounds [0, scale_xz]
            if (localPos.x < 0 || localPos.x > hm.scale_xz || localPos.z < 0 || localPos.z > hm.scale_xz) {
                continue; // Not this terrain
            }

            // 3. Convert local position to heightmap grid coordinates
            float normalizedX = localPos.x / hm.scale_xz;
            float normalizedZ = localPos.z / hm.scale_xz;
            
            // Clamp to valid range (redundant due to bounds check but safer)
            normalizedX = std::clamp(normalizedX, 0.0f, 1.0f);
            normalizedZ = std::clamp(normalizedZ, 0.0f, 1.0f);
            
            // Get grid coordinates
            float gx = normalizedX * (hm.width - 1);
            float gz = normalizedZ * (hm.height - 1);
            
            int x0 = (int)std::floor(gx);
            int z0 = (int)std::floor(gz);
            int x1 = (std::min)(x0 + 1, hm.width - 1);
            int z1 = (std::min)(z0 + 1, hm.height - 1);
            
            float fx = gx - x0;
            float fz = gz - z0;
            
            // Bilinear interpolation
            float h00 = hm.data[z0 * hm.width + x0];
            float h10 = hm.data[z0 * hm.width + x1];
            float h01 = hm.data[z1 * hm.width + x0];
            float h11 = hm.data[z1 * hm.width + x1];
            
            float h0 = h00 * (1.0f - fx) + h10 * fx;
            float h1 = h01 * (1.0f - fx) + h11 * fx;
            float local_height = (h0 * (1.0f - fz) + h1 * fz) * hm.scale_y;
            
            // 4. Transform local height back to world space
            if (terrain.transform) {
                 Vec3 worldPos = terrain.transform->getFinal().multiplyVector(Vec4(localPos.x, local_height, localPos.z, 1.0f)).xyz();
                 return worldPos.y;
            }
            
            return local_height;
        }
        
        return 0.0f;
    }

    // Sample normal at world XZ coordinate
    Vec3 sampleNormal(float worldX, float worldZ) const;

    // Sample splat map channel value (0..1) at world XZ coordinate for the terrain containing the point
    // channel: 0=R,1=G,2=B,3=A. Returns -1.0f if no splat data or out of bounds.
    float sampleSplatChannel(float worldX, float worldZ, int channel) const;
    // Samples a named, graph-published terrain field (terrain.slope,
    // terrain.valley, terrain.wetness, ...). Returns 1 when no field is
    // requested and -1 when the requested field cannot be sampled.
    float sampleAnalysisField(float worldX, float worldZ, const std::string& fieldName) const;
    
    // ===========================================================================
    // RIVER BED CARVING (for River System integration)
    // ===========================================================================
    
    /**
     * @brief Natural carve parameters for realistic river bed generation
     */
    struct NaturalCarveParams {
        bool enableNoise = true;           // Noise-based edge irregularity
        float noiseScale = 0.15f;          // Noise frequency
        float noiseStrength = 0.3f;        // Noise intensity
        
        bool enableDeepPools = true;       // Random deep pools
        float poolFrequency = 0.15f;       // Pool occurrence rate
        float poolDepthMult = 1.8f;        // Pool depth multiplier
        
        bool enableRiffles = true;         // Shallow riffle zones
        float riffleFrequency = 0.2f;      // Riffle occurrence rate
        float riffleDepthMult = 0.4f;      // Riffle depth multiplier
        
        bool enableAsymmetry = true;       // Asymmetric bank profiles
        float asymmetryStrength = 0.6f;    // Inner/outer bank difference
        
        bool enablePointBars = true;       // Point bar deposits on inner bends
        float pointBarStrength = 0.4f;     // Point bar elevation amount
    };
    
    /**
     * @brief Carve a river bed into the terrain along a path
     * @param terrainId Target terrain ID (-1 for first terrain)
     * @param points Vector of world-space points along the river center
     * @param widths Width at each point
     * @param depths Depth at each point (how deep to carve)
     * @param smoothness Edge smoothing factor (0-1)
     * @param scene SceneData for mesh update
     */
    void carveRiverBed(int terrainId, 
                       const std::vector<Vec3>& points,
                       const std::vector<float>& widths,
                       const std::vector<float>& depths,
                       float smoothness,
                       SceneData& scene);
    
    /**
     * @brief Carve a natural river bed with advanced features
     * @param terrainId Target terrain ID (-1 for first terrain)
     * @param points Vector of world-space points along the river center
     * @param widths Width at each point
     * @param depths Depth at each point
     * @param smoothness Edge smoothing factor
     * @param naturalParams Natural carve parameters (noise, pools, etc.)
     * @param scene SceneData for mesh update
     */
    void carveRiverBedNatural(int terrainId, 
                              const std::vector<Vec3>& points,
                              const std::vector<float>& widths,
                              const std::vector<float>& depths,
                              float smoothness,
                              const NaturalCarveParams& naturalParams,
                              SceneData& scene);
    
    /**
     * @brief Lower terrain height at a world position
     * @param worldX World X coordinate
     * @param worldZ World Z coordinate
     * @param amount Amount to lower (positive = deeper)
     * @param radius Falloff radius
     * @param terrainId Target terrain (-1 for first)
     */
    void lowerHeightAt(float worldX, float worldZ, float amount, float radius, int terrainId = -1);
    
    // Management
    void removeTerrain(SceneData& scene, int id);
    void removeAllTerrains(SceneData& scene);

private:
    TerrainManager() = default;
    
    std::vector<TerrainObject> terrains;

    int next_id = 1;

    // CUDA Driver API Handles
    void* cudaModule = nullptr;
    void* erosionKernelFunc = nullptr;
    void* smoothKernelFunc = nullptr;
    void* thermalKernelFunc = nullptr;
    // Fluvial Kernels
    void* fluvRainKernelFunc = nullptr;
    void* fluvFluxKernelFunc = nullptr;
    void* fluvWaterKernelFunc = nullptr;
    void* fluvErodeKernelFunc = nullptr;
    void* streamPowerKernelFunc = nullptr;
    void* applyStreamPowerKernelFunc = nullptr;
    void* windKernelFunc = nullptr;
    // Post-processing kernels (for CPU-GPU parity)
    void* pitFillKernelFunc = nullptr;
    void* spikeRemovalKernelFunc = nullptr;
    void* edgePreservationKernelFunc = nullptr;
    void* thermalWithHardnessKernelFunc = nullptr;
    bool cudaInitialized = false;
    void initCuda();
};
