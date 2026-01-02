#pragma once

#include "TerrainSystem.h"
#include "json.hpp"
#include <vector>
#include <string>
#include <functional>

struct SceneData; // Forward decl
class Material;
class Texture;

// Serialization version - increment when format changes
static constexpr int TERRAIN_SERIALIZATION_VERSION = 1;

// ===========================================================================
// EROSION PARAMETERS
// ===========================================================================

struct HydraulicErosionParams {
    int iterations = 50000;        // Number of droplets
    int dropletLifetime = 64;      // Max steps per droplet
    float inertia = 0.05f;         // Direction momentum (0-1)
    float sedimentCapacity = 4.0f; // Max sediment per speed unit
    float minSlope = 0.01f;        // Minimum slope
    float erodeSpeed = 0.3f;       // Erosion rate
    float depositSpeed = 0.3f;     // Deposit rate
    float evaporateSpeed = 0.01f;  // Evaporation rate
    float gravity = 4.0f;
    int erosionRadius = 3;         // Brush radius
};

struct ThermalErosionParams {
    int iterations = 50;
    float talusAngle = 0.5f;       // ~27 degrees
    float erosionAmount = 0.5f;
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
    void updateDirtySectors(TerrainObject* terrain);
    
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
    
    // Export splat map to PNG file
    void exportSplatMap(TerrainObject* terrain, const std::string& filepath);

    // ===========================================================================
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
    void* windKernelFunc = nullptr;
    // Post-processing kernels (for CPU-GPU parity)
    void* pitFillKernelFunc = nullptr;
    void* spikeRemovalKernelFunc = nullptr;
    void* edgePreservationKernelFunc = nullptr;
    void* thermalWithHardnessKernelFunc = nullptr;
    bool cudaInitialized = false;
    void initCuda();
};
