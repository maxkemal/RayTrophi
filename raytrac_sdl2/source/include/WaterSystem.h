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
    // Wave dynamics
    float wave_speed = 1.0f;
    float wave_strength = 0.5f;
    float wave_frequency = 1.0f;
    
    // Base colors
    Vec3 deep_color = Vec3(0.02f, 0.08f, 0.2f);      // Dark blue for deep water
    Vec3 shallow_color = Vec3(0.3f, 0.6f, 0.8f);    // Light cyan for shallow
    
    // Physics
    float clarity = 0.8f;      // 0.0 = murky, 1.0 = crystal clear
    float foam_level = 0.2f;   // Foam at wave crests
    float ior = 1.333f;        // Index of Refraction (water = 1.333)
    float roughness = 0.02f;   // Surface micro-roughness
    
    // === ADVANCED: Depth-based rendering ===
    float depth_max = 15.0f;                         // Depth for full deep_color (meters)
    Vec3 absorption_color = Vec3(0.3f, 0.6f, 0.7f); // Absorption tint (what color is absorbed)
    float absorption_density = 0.5f;                 // How quickly light is absorbed
    
    // === ADVANCED: Shore foam ===
    float shore_foam_distance = 1.5f;   // Distance from shore for foam effect
    float shore_foam_intensity = 0.6f;  // Shore foam strength (0-1)
    
    // === ADVANCED: Caustics ===
    float caustic_intensity = 0.4f;     // Caustic brightness
    float caustic_scale = 2.0f;         // Caustic pattern size
    float caustic_speed = 1.0f;         // Caustic animation speed
    
    // === ADVANCED: Sub-surface scattering ===
    float sss_intensity = 0.15f;        // Light scattering inside water
    Vec3 sss_color = Vec3(0.1f, 0.4f, 0.5f);  // SSS tint color
    
    // === ADVANCED: FFT Ocean (Tessendorf) ===
    bool use_fft_ocean = false;         // Enable FFT ocean simulation
    int fft_resolution = 256;           // FFT grid size (64, 128, 256, 512)
    float fft_ocean_size = 100.0f;      // World space coverage
    float fft_wind_speed = 10.0f;       // Wind speed (m/s)
    float fft_wind_direction = 0.0f;    // Wind direction (degrees)
    float fft_choppiness = 1.0f;        // Horizontal displacement strength
    float fft_amplitude = 0.0002f;      // Wave amplitude scale (Phillips A)
    float fft_time_scale = 1.0f;        // Animation speed

    // === ADVANCED: Realistic Details ===
    float micro_detail_strength = 0.05f;// Strength of high-freq noise (ripples) - Default small
    float micro_detail_scale = 20.0f;   // Scale of noise (higher = smaller ripples)
    float foam_noise_scale = 4.0f;      // Scale of foam breakup noise
    float foam_threshold = 0.4f;        // Offset for foam appearance
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
    
    // Runtime FFT State (opaque handle to FFTOceanState)
    void* fft_state = nullptr;
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
    
    // Updates all water surfaces (FFT simulation, etc.)
    // Returns true if material parameters (like texture handles) changed and require GPU sync
    bool update(float dt);
    
    // Returns the height map texture of the first active FFT water surface (or 0)
    cudaTextureObject_t getFirstFFTHeightMap();
    
    // Remove water surface
    void removeWaterSurface(SceneData& scene, int id);
    
    // Clear all water surfaces (for new/load project)
    void clear();
    
    // Serialization
    nlohmann::json serialize() const;
    void deserialize(const nlohmann::json& j, SceneData& scene);

private:
    WaterManager() = default;
    
    std::vector<WaterSurface> water_surfaces;
    int next_id = 1;
};
