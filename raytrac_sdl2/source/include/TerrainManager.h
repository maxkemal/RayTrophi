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
static constexpr int TERRAIN_SERIALIZATION_VERSION = 2;

// ===========================================================================
// EROSION PARAMETERS
// ===========================================================================

struct HydraulicErosionParams {
    int iterations = 50000;        // Number of 'hits' (Determines passes in GPU)
    int dropletLifetime = 128;     // Max steps per droplet (Legacy CPU)
    float inertia = 0.05f;         // direction momentum (0-1)
    float sedimentCapacity = 1.5f; // Tamed for Stream Power Law
    float minSlope = 0.01f;        // Minimal slope for flow
    float erodeSpeed = 0.1f;       // Tamed erosion rate
    float depositSpeed = 0.3f;     // Deposit rate
    float evaporateSpeed = 0.01f;  // Evaporation rate
    float gravity = 4.0f;          // Reset to moderate
    int erosionRadius = 2;         // Default channel width
};

struct ThermalErosionParams {
    int iterations = 50;          // Moderate default
    float talusAngle = 0.5f;       // ~27 degrees
    float erosionAmount = 0.3f;    // Less aggressive
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
    TerrainObject* createTerrain(SceneData& scene, int resolution, float size);
    
    // Create terrain from heightmap image (using stb_image)
    TerrainObject* createTerrainFromHeightmap(SceneData& scene, const std::string& filepath, float size, float maxHeight, int max_resolution = 1024);
    
    // Update mesh vertices based on heightmap (Call after sculpting)
    void updateTerrainMesh(TerrainObject* terrain);

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
                float targetHeight = 0.0f, std::shared_ptr<class Texture> stampTexture = nullptr, float rotation = 0.0f);
    
    // Smooth terrain (Box blur to reduce quantization noise)
    void smoothTerrain(TerrainObject* terrain, int iterations);

    // Layer System & Painting
    void initLayers(TerrainObject* terrain);
    // channel: 0=R, 1=G, 2=B, 3=A
    void paintSplatMap(TerrainObject* terrain, const Vec3& hitPoint, int channel, float radius, float strength, float dt);
    // Auto-generate mask based on slope and height
    void autoMask(TerrainObject* terrain, float slopeWeight, float heightWeight, float heightMin, float heightMax, float slopeSteepness);
    
    // Internal helper to sync CPU splat data to GPU texture
    void updateSplatMapTexture(TerrainObject* terrain);
    
    // Resize splatmap to match heightmap dimensions (bilinear interpolation)
    void resizeSplatMap(TerrainObject* terrain);
    
    // Export splat map to PNG file
    void exportSplatMap(TerrainObject* terrain, const std::string& filepath);

    // Import splat map from image file
    void importSplatMap(TerrainObject* terrain, const std::string& filepath);

    // ===========================================================================
    // FOLIAGE SYSTEM
    // ===========================================================================    // Foliage
    void updateFoliage(TerrainObject* terrain, OptixWrapper* optix);
    void clearFoliage(TerrainObject* terrain, OptixWrapper* optix);
    void reapplyAllFoliage(OptixWrapper* optix); // Re-adds persistence after rebuild
    
    // Serialization===========================================================================
    // EROSION SYSTEM
    // ===========================================================================
    void hydraulicErosion(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask = {});
    void hydraulicErosionAdvanced(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask = {});
    void fluvialErosion(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask = {});
    void fluvialErosionGPU(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask = {});
    void hydraulicErosionGPU(TerrainObject* terrain, const HydraulicErosionParams& params, const std::vector<float>& mask = {});
    void thermalErosionGPU(TerrainObject* terrain, const ThermalErosionParams& params, const std::vector<float>& mask = {});
    void thermalErosion(TerrainObject* terrain, const ThermalErosionParams& params, const std::vector<float>& mask = {});
   
    void windErosion(TerrainObject* terrain, float strength, float direction, int iterations, const std::vector<float>& mask = {});
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
    void exportHeightmap(TerrainObject* terrain, const std::string& filepath);
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
    void* windKernelFunc = nullptr;
    // Post-processing kernels (for CPU-GPU parity)
    void* pitFillKernelFunc = nullptr;
    void* spikeRemovalKernelFunc = nullptr;
    void* edgePreservationKernelFunc = nullptr;
    void* thermalWithHardnessKernelFunc = nullptr;
    bool cudaInitialized = false;
    void initCuda();
};

