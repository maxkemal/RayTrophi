/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          WaterSystem.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
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

    // === GEOMETRIC DISPLACEMENT (Physical Mesh) ===
    enum class NoiseType { Perlin, FBM, Ridge, Voronoi, Billow, Gerstner, TessendorfSimple };
    NoiseType geo_noise_type = NoiseType::Ridge;

    bool use_geometric_waves = false;
    float geo_wave_height = 2.0f;       // Amplitude
    float geo_wave_scale = 50.0f;       // Global Scale
    float geo_wave_choppiness = 1.0f;   // Ridge Offset / Sharpness
    float geo_wave_speed = 0.5f;        // Animation Speed (Phase Shift)
    
    // Detailed Noise Params
    int geo_octaves = 4;
    float geo_persistence = 0.5f;
    float geo_lacunarity = 2.0f;
    float geo_ridge_offset = 1.0f;
    
    // ===  OCEAN PARAMS ===
    float geo_damping = 0.0f;           // Damping for wind perpendicular waves (0-1)
    float geo_alignment = 0.5f;         // Wave alignment to wind direction (0=omni, 1=aligned)
    float geo_depth = 200.0f;           // Ocean depth in meters (affects shallow water behavior)
    float geo_swell_direction = 0.0f;   // Swell direction offset (degrees)
    float geo_swell_amplitude = 0.2f;   // Swell (long-distance waves) contribution
    float geo_sharpening = 0.0f;        // Post-process sharpening (0=smooth, 1=peaked waves)
    float geo_detail_scale = 3.0f;      // Secondary detail noise scale multiplier
    float geo_detail_strength = 0.15f;  // Secondary detail noise strength
    
    // Smooth Normals
    bool geo_smooth_normals = true;     // Enable smooth shading (vertex normal averaging)
};

struct WaterSurface {
    enum class Type { Plane, River, Custom };
    Type type = Type::Plane;
    
    int id = -1;
    std::string name;
    WaterWaveParams params;
    
    // The physics mesh (usually a grid plane)
    std::shared_ptr<Triangle> reference_triangle; // To track position/transform
    std::vector<std::shared_ptr<Triangle>> mesh_triangles;
    
    // Original vertex positions (for animation - keeps base grid positions)
    std::vector<Vec3> original_positions;
    
    // Material ID used for this water
    uint16_t material_id = 0;
    
    // Runtime FFT State (opaque handle to FFTOceanState)
    void* fft_state = nullptr;
    
    // Animation state
    float animation_time = 0.0f;
    bool animate_mesh = false;  // Enable CPU mesh animation (performance intensive)
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
    
    // Create water from existing mesh triangles (e.g. from Terrain)
    WaterSurface* createWaterFromMesh(SceneData& scene, const std::string& name, const std::vector<std::shared_ptr<Triangle>>& triangles);
    
    // Update mesh geometry based on physics parameters (static, called once)
    void updateWaterMesh(WaterSurface* surf);
    
    // Update mesh with animation (called each frame for animated surfaces)
    void updateAnimatedWaterMesh(WaterSurface* surf, float time);
    
    // Store original positions for animation (call after initial mesh creation)
    void cacheOriginalPositions(WaterSurface* surf);
    
    // Get all water surfaces
    std::vector<WaterSurface>& getWaterSurfaces() { return water_surfaces; }
    
    // Get water surface by ID
    WaterSurface* getWaterSurface(int id);
    
    // Updates all water surfaces (FFT simulation + animated meshes)
    // Returns true if material parameters (like texture handles) changed and require GPU sync
    bool update(float dt);
    
    // Returns the height map texture of the first active FFT water surface (or 0)
    cudaTextureObject_t getFirstFFTHeightMap();
    
    // Remove water surface
    void removeWaterSurface(SceneData& scene, int id);
    
    // Clear all water surfaces (for new/load project)
    void clear();
    
    // Apply keyframe values to water surface and rebuild mesh
    void applyKeyframe(WaterSurface* surf, const struct WaterKeyframe& keyframe);
    
    // Capture current state to keyframe track (for recording)
    void captureKeyframeToTrack(WaterSurface* surf, struct ObjectAnimationTrack& track, int frame);
    
    // Apply interpolated keyframe from track (for playback)
    void updateFromTrack(WaterSurface* surf, const struct ObjectAnimationTrack& track, int currentFrame);
    
    // Serialization
    nlohmann::json serialize() const;
    void deserialize(const nlohmann::json& j, SceneData& scene);

private:
    WaterManager() = default;
    
    std::vector<WaterSurface> water_surfaces;
    int next_id = 1;
};

