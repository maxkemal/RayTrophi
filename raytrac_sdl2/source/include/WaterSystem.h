#pragma once

#include "Vec3.h"
#include "Material.h"
#include "Triangle.h"
#include "json.hpp"
#include <vector>
#include <memory>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════════
// WATER SYSTEM DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

struct WaterWaveParams {
    float wave_speed = 1.0f;
    float wave_strength = 0.5f;
    float wave_frequency = 1.0f;
    
    // Appearance
    Vec3 deep_color = Vec3(0.0f, 0.1f, 0.4f);
    Vec3 shallow_color = Vec3(0.4f, 0.7f, 0.9f);
    float clarity = 0.8f;      // 0.0 = opaque, 1.0 = clear
    float foam_level = 0.1f;   // Foam at crests
    float ior = 1.333f;        // Index of Refraction
    float roughness = 0.05f;   // Surface roughness
};

struct WaterSurface {
    int id = -1;
    std::string name;
    WaterWaveParams params;
    
    // The physics mesh (usually a grid plane)
    std::shared_ptr<Triangle> reference_triangle; // To track position/transform
    std::vector<std::shared_ptr<Triangle>> mesh_triangles;
    
    // Material ID used for this water
    uint16_t material_id = 0;
};

// ═══════════════════════════════════════════════════════════════════════════════
// WATER MANAGER CLASS
// ═══════════════════════════════════════════════════════════════════════════════

struct SceneData;
class Renderer;
class OptixWrapper;

class WaterManager {
public:
    static WaterManager& getInstance() {
        static WaterManager instance;
        return instance;
    }
    
    // Create a new water plane at given position
    WaterSurface* createWaterPlane(SceneData& scene, const Vec3& pos, float size, float density);
    
    // Get all water surfaces
    std::vector<WaterSurface>& getWaterSurfaces() { return water_surfaces; }
    
    // Get water surface by ID
    WaterSurface* getWaterSurface(int id);
    
    // Update water animation (called per frame)
    void update(float dt);
    
    // Remove water surface
    void removeWaterSurface(SceneData& scene, int id);
    
    // Clear all water surfaces (for new/load project)
    void clear() { water_surfaces.clear(); next_id = 1; }
    
    // Serialization
    nlohmann::json serialize() const;
    void deserialize(const nlohmann::json& j, SceneData& scene);

private:
    WaterManager() = default;
    
    std::vector<WaterSurface> water_surfaces;
    int next_id = 1;
};
