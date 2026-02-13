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
    Vec3 deep_color = Vec3(0.01f, 0.02f, 0.05f);     // Very dark blue for deep water
    Vec3 shallow_color = Vec3(2.0f/255.0f, 3.0f/255.0f, 3.0f/255.0f); // Dark blue (2, 3, 3) for physical look
    
    // Physics
    float clarity = 0.8f;      // 0.0 = murky, 1.0 = crystal clear
    float foam_level = 0.01f;  // Lower default foam
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
    float fft_ocean_size = 100.0f;      // World space coverage (meters)
    float fft_wind_speed = 10.0f;       // Wind speed (m/s) - affects wave size
    float fft_wind_direction = 0.0f;    // Wind direction (degrees)
    float fft_choppiness = 1.0f;        // Horizontal displacement strength
    float fft_amplitude = 0.001f;       // Phillips spectrum amplitude (higher = bigger waves)
    float fft_time_scale = 1.0f;        // Animation speed

    // === ADVANCED: Realistic Details ===
    float micro_detail_strength = 0.05f;// Strength of high-freq noise (ripples) - Default small
    float micro_detail_scale = 20.0f;   // Scale of noise (higher = smaller ripples)
    float micro_anim_speed = 0.1f;      // Animation speed multiplier for micro details
    float micro_morph_speed = 1.0f;     // Morph/shape-change speed for micro details
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
    
    // === FFT-DRIVEN MESH DISPLACEMENT (Best Quality - Combines FFT + Mesh) ===
    bool use_fft_mesh_displacement = false;  // Use FFT data to displace mesh vertices
    float fft_mesh_height_scale = 50.0f;     // Amplification for FFT height (raw FFT values are small)
    float fft_mesh_choppiness = 1.0f;        // Scale factor for FFT horizontal displacement
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // WATER PRESETS - Quick setup for common water types
    // ═══════════════════════════════════════════════════════════════════════════════
    enum class WaterPreset { 
        Custom,         // User-defined settings
        CalmOcean,      // Calm open ocean - gentle swells
        StormyOcean,    // Stormy sea - high waves, lots of foam
        TropicalOcean,  // Crystal clear tropical water
        Lake,           // Still lake - minimal waves
        River,          // Flowing river with current
        Pool,           // Swimming pool - very calm
        Pond            // Small pond with subtle ripples
    };
    WaterPreset current_preset = WaterPreset::Custom;
    
    // Apply preset values
    void applyPreset(WaterPreset preset) {
        current_preset = preset;
        
        switch (preset) {
            case WaterPreset::CalmOcean:
                // FFT settings for calm ocean
                use_fft_ocean = true;
                fft_resolution = 256;
                fft_ocean_size = 100.0f;      // 100m tile
                fft_wind_speed = 10.0f;       // Moderate wind
                fft_wind_direction = 0.0f;
                fft_choppiness = 1.0f;
                fft_amplitude = 0.001f;       // Higher for visible waves
                fft_time_scale = 1.0f;
                // Appearance
                deep_color = Vec3(0.01f, 0.03f, 0.08f);
                shallow_color = Vec3(0.05f, 0.15f, 0.2f);
                roughness = 0.02f;
                foam_level = 0.05f;
                // Micro details
                micro_detail_strength = 0.03f;
                micro_detail_scale = 15.0f;
                micro_anim_speed = 0.05f;
                micro_morph_speed = 0.3f;
                // FFT Mesh - Direct height control
                use_fft_mesh_displacement = true;
                fft_mesh_height_scale = 50.0f;   // Amplify FFT output significantly
                fft_mesh_choppiness = 1.0f;
                break;
                
            case WaterPreset::StormyOcean:
                use_fft_ocean = true;
                fft_resolution = 256;
                fft_ocean_size = 150.0f;
                fft_wind_speed = 25.0f;       // Strong wind
                fft_wind_direction = 0.0f;
                fft_choppiness = 2.0f;
                fft_amplitude = 0.003f;       // High amplitude for storms
                fft_time_scale = 1.2f;
                // Dark stormy colors
                deep_color = Vec3(0.02f, 0.04f, 0.06f);
                shallow_color = Vec3(0.1f, 0.15f, 0.18f);
                roughness = 0.05f;
                foam_level = 0.4f;
                foam_threshold = 0.25f;
                // Aggressive micro details
                micro_detail_strength = 0.08f;
                micro_detail_scale = 25.0f;
                micro_anim_speed = 0.15f;
                micro_morph_speed = 0.8f;
                // FFT Mesh
                use_fft_mesh_displacement = true;
                fft_mesh_height_scale = 80.0f;   // High waves
                fft_mesh_choppiness = 2.5f;
                break;
                
            case WaterPreset::TropicalOcean:
                use_fft_ocean = true;
                fft_resolution = 256;
                fft_ocean_size = 80.0f;
                fft_wind_speed = 5.0f;
                fft_wind_direction = 0.0f;
                fft_choppiness = 0.5f;
                fft_amplitude = 0.0005f;
                fft_time_scale = 0.6f;
                // Crystal clear tropical
                deep_color = Vec3(0.0f, 0.05f, 0.1f);
                shallow_color = Vec3(0.1f, 0.4f, 0.5f);
                clarity = 0.95f;
                roughness = 0.01f;
                foam_level = 0.02f;
                absorption_density = 0.2f;
                // Very subtle micro details
                micro_detail_strength = 0.02f;
                micro_detail_scale = 10.0f;
                micro_anim_speed = 0.03f;
                micro_morph_speed = 0.2f;
                // FFT Mesh
                use_fft_mesh_displacement = true;
                fft_mesh_height_scale = 30.0f;
                fft_mesh_choppiness = 0.5f;
                break;
                
            case WaterPreset::Lake:
                use_fft_ocean = true;
                fft_resolution = 128;
                fft_ocean_size = 50.0f;
                fft_wind_speed = 3.0f;
                fft_wind_direction = 0.0f;
                fft_choppiness = 0.3f;
                fft_amplitude = 0.0003f;
                fft_time_scale = 0.4f;
                // Lake colors
                deep_color = Vec3(0.02f, 0.05f, 0.08f);
                shallow_color = Vec3(0.08f, 0.2f, 0.25f);
                clarity = 0.7f;
                roughness = 0.01f;
                foam_level = 0.0f;
                // Almost no micro details - glassy
                micro_detail_strength = 0.01f;
                micro_detail_scale = 8.0f;
                micro_anim_speed = 0.02f;
                micro_morph_speed = 0.1f;
                // Gentle mesh displacement
                use_fft_mesh_displacement = true;
                fft_mesh_height_scale = 15.0f;
                fft_mesh_choppiness = 0.3f;
                break;
                
            case WaterPreset::River:
                use_fft_ocean = true;
                fft_resolution = 128;
                fft_ocean_size = 30.0f;  // Smaller for river
                fft_wind_speed = 4.0f;
                fft_wind_direction = 0.0f;  // Fixed flow direction
                fft_choppiness = 0.5f;
                fft_amplitude = 0.0004f;
                fft_time_scale = 1.5f;    // Faster animation
                // River colors
                deep_color = Vec3(0.03f, 0.06f, 0.05f);
                shallow_color = Vec3(0.1f, 0.18f, 0.15f);
                clarity = 0.5f;
                roughness = 0.03f;
                foam_level = 0.1f;
                // Directional micro details (flow feeling)
                micro_detail_strength = 0.04f;
                micro_detail_scale = 12.0f;
                micro_anim_speed = 0.2f;  // Faster - flowing
                micro_morph_speed = 0.5f;
                // River mesh displacement
                use_fft_mesh_displacement = true;
                fft_mesh_height_scale = 20.0f;
                fft_mesh_choppiness = 0.5f;
                break;
                
            case WaterPreset::Pool:
                use_fft_ocean = false;  // No FFT - very calm
                use_fft_mesh_displacement = false;
                // Pool colors
                deep_color = Vec3(0.0f, 0.1f, 0.2f);
                shallow_color = Vec3(0.2f, 0.5f, 0.6f);
                clarity = 1.0f;
                roughness = 0.005f;
                foam_level = 0.0f;
                // Minimal ripples
                micro_detail_strength = 0.005f;
                micro_detail_scale = 5.0f;
                micro_anim_speed = 0.01f;
                micro_morph_speed = 0.05f;
                break;
                
            case WaterPreset::Pond:
                use_fft_ocean = true;
                fft_resolution = 64;
                fft_ocean_size = 20.0f;
                fft_wind_speed = 2.0f;
                fft_wind_direction = 0.0f;
                fft_choppiness = 0.2f;
                fft_amplitude = 0.0001f;
                fft_time_scale = 0.3f;
                // Murky pond
                deep_color = Vec3(0.02f, 0.04f, 0.03f);
                shallow_color = Vec3(0.08f, 0.12f, 0.08f);
                clarity = 0.4f;
                roughness = 0.02f;
                foam_level = 0.0f;
                // Very subtle
                micro_detail_strength = 0.01f;
                micro_detail_scale = 6.0f;
                micro_anim_speed = 0.015f;
                micro_morph_speed = 0.1f;
                // Light displacement
                use_fft_mesh_displacement = true;
                fft_mesh_height_scale = 10.0f;
                fft_mesh_choppiness = 0.2f;
                break;
                
            case WaterPreset::Custom:
            default:
                // Keep current values
                break;
        }
    }
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
    
    // Runtime GPU Geometric Wave State (opaque handle to GPUGeoWaveState)
    void* gpu_geo_state = nullptr;
    
    // Animation state
    float animation_time = 0.0f;
    bool animate_mesh = false;      // Enable mesh animation
    bool use_gpu_animation = true;  // Use GPU for geometric waves (faster)
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
    
    // Update mesh with animation - CPU path (called each frame for animated surfaces)
    void updateAnimatedWaterMesh(WaterSurface* surf, float time);
    
    // Update mesh with animation - GPU path (much faster for large meshes)
    void updateGPUAnimatedWaterMesh(WaterSurface* surf, float time);
    
    // Update mesh using FFT ocean data - highest quality (GPU accelerated)
    void updateFFTDrivenMesh(WaterSurface* surf, float time);
    
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

