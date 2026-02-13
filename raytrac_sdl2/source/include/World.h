/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          World.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>

#ifndef __CUDACC__
#include "Vec3.h"
#include "json.hpp"
#endif


// GPU-Compatible Enum
enum WorldMode {
    WORLD_MODE_COLOR = 0,
    WORLD_MODE_HDRI = 1,
    WORLD_MODE_NISHITA = 2
};

// GPU-Compatible Structs
struct AtmosphereLUTData {
    cudaTextureObject_t transmittance_lut;     // 2D (ViewAngle, Altitude)
    cudaTextureObject_t skyview_lut;           // 2D (ViewAngle, SunAltitude)
    cudaTextureObject_t multi_scattering_lut;  // 2D (SunAltitude, Altitude) - Energy compensation
    cudaTextureObject_t aerial_perspective_lut; // 3D (X, Y, Distance) - Per-pixel in-scattering
    
    // Multi-scattering factor (calculated during precomputation)
    float3 integrated_multi_scattering;
};

// GPU-Compatible Structs (Blender-compatible naming)
struct NishitaSkyParams {
    float3 sun_direction;
    float sun_elevation;     // Blender: Sun Elevation (degrees)
    float sun_azimuth;       // Blender: Sun Rotation (degrees)
    float sun_intensity;     // Sun brightness
    float sun_size;          // Blender: Sun Disc size (degrees, default 0.545)
    
    // Atmosphere parameters (Blender-style, multipliers default 1.0)
    float air_density;       // Blender: Air (Rayleigh scattering multiplier)
    float dust_density;      // Blender: Dust (Mie scattering/aerosols multiplier)
    float ozone_density;     // Blender: Ozone (affects blue saturation)
    float altitude;          // Blender: Altitude (camera height in meters, 0 = sea level)
    
    
    // Cloud Layer 1 (Primary) parameters
    int clouds_enabled;      // 1 = show clouds
    float cloud_coverage;    // 0.0 - 1.0 (how much sky is covered)
    float cloud_density;     // Opacity multiplier
    float cloud_scale;       // Noise frequency (larger = bigger clouds)
    float cloud_height_min;  // Bottom altitude (meters)
    float cloud_height_max;  // Top altitude (meters)
    float cloud_offset_x;    // Wind/Seed Offset X

    float cloud_offset_z;    // Wind/Seed Offset Z
    
    // FFT Cloud Modulation
    cudaTextureObject_t cloud_fft_map; // FFT Height Map for Coverage
    int cloud_use_fft;                 // 1 = Use FFT Map, 0 = Use Procedural
    
    // Cloud Layer 2 (Secondary) parameters - for multi-layer clouds
    int cloud_layer2_enabled;     // 1 = show second layer
    float cloud2_coverage;        // 0.0 - 1.0
    float cloud2_density;         // Opacity multiplier
    float cloud2_scale;           // Noise frequency
    float cloud2_height_min;      // Bottom altitude (meters)
    float cloud2_height_max;      // Top altitude (meters)
    
    // Quality and detail settings
    float cloud_quality;     // Quality multiplier for steps
    float cloud_detail;      // Detail level (0.5 = low, 1.0 = normal, 2.0 = high detail noise)
    int cloud_base_steps;    // Base number of ray marching steps (e.g., 48)
    
    // Cloud Lighting settings
    int cloud_light_steps;        // Number of light marching steps (0 = disabled, 4-8 recommended)
    float cloud_shadow_strength;  // Shadow darkness (0 = no shadows, 1 = normal, 2 = dark shadows)
    float cloud_ambient_strength; // Ambient light strength (0.5 = low, 1.0 = normal)
    float cloud_silver_intensity; // Silver lining intensity (0 = off, 1 = normal, 2 = strong)
    float cloud_absorption;       // Light absorption rate (0.5 = thin clouds, 1.0 = normal, 2.0 = thick)
    
    // Cloud Advanced Scattering (VDB-like)
    float cloud_anisotropy;       // Forward scattering g-factor (0.0 to 0.99)
    float cloud_anisotropy_back;  // Backward scattering g-factor (-0.99 to 0.0)
    float cloud_lobe_mix;         // Blend between forward and backward lobes (0 to 1)
    
    // Cloud Emissive (Experimental)
    float3 cloud_emissive_color;  // Color of cloud emission
    float cloud_emissive_intensity; // Intensity of emission
    
    // Physical constants (usually not exposed in UI)
    float planet_radius;
    float atmosphere_height;
    float3 rayleigh_scattering;
    float3 mie_scattering;
    float mie_anisotropy;    // g factor (0.8 = forward scattering)
    float rayleigh_density;  // Scale height for Rayleigh
    float mie_density;       // Scale height for Mie
    
    // ═══════════════════════════════════════════════════════════════
    // ATMOSPHERIC FOG (Height-based + Distance-based)
    // ═══════════════════════════════════════════════════════════════
    int fog_enabled;               // 1 = show fog
    float fog_density;             // Base fog density (0.0 - 0.1)
    float fog_height;              // Fog falloff height (meters, 0 = ground level)
    float fog_falloff;             // Exponential falloff rate (0.001 - 0.01)
    float fog_distance;            // Max fog distance (meters)
    float3 fog_color;              // Fog color (usually bluish-white)
    float fog_sun_scatter;         // How much fog scatters sunlight toward camera
    
    // ═══════════════════════════════════════════════════════════════
    // VOLUMETRIC LIGHT RAYS (God Rays / Light Shafts)
    // ═══════════════════════════════════════════════════════════════
    int godrays_enabled;           // 1 = show god rays
    float godrays_intensity;       // God ray brightness (0.0 - 2.0)
    float godrays_density;         // God ray density/thickness
    int godrays_samples;           // Quality (8-32 recommended)
    
    // Physical Parameters (Atmosphere Physics)
    float humidity;                // 0.0 (Dry) to 1.0 (Humid/Hazy)
    float temperature;             // Celsius (-50 to +50)
    float ozone_absorption_scale;  // Scales the "Blue Hour" intensity (0.0 to 10.0)
};

// ═══════════════════════════════════════════════════════════════
// ATMOSPHERE ADVANCED (Rendering Toggles)
// ═══════════════════════════════════════════════════════════════
struct AtmosphereAdvanced {
    int multi_scatter_enabled;     // 1 = enable multi-scattering
    float multi_scatter_factor;    // Multi-scatter intensity (0.0 - 1.0)
    int aerial_perspective;        // 1 = Physical haze based on distance
    
    // Aerial Perspective Distance Control (UI adjustable)
    float aerial_min_distance;     // No haze below this distance (meters, default: 1000)
    float aerial_max_distance;     // Full haze at this distance (meters, default: 5000)
    
    // Environment Texture Overlay (Moved here for better UI grouping)
    int env_overlay_enabled;       // 1 = blend environment texture with Nishita
    cudaTextureObject_t env_overlay_tex;  // HDR/EXR environment texture
    float env_overlay_intensity;   // Texture contribution (0.0 - 2.0)
    float env_overlay_rotation;    // Rotation in degrees (0 - 360)
    int env_overlay_blend_mode;    // 0 = Mix, 1 = Multiply, 2 = Screen, 3 = Replace
};

struct WorldData {
    int mode; // WorldMode
    
    // Solid Color Mode
    float3 color;
    float color_intensity;

    // HDRI Mode
    cudaTextureObject_t env_texture;
    float env_rotation; // Rotation in radians
    float env_intensity;
    int env_width;      // For importance sampling (future)
    int env_height;

    // Nishita Mode
    NishitaSkyParams nishita;
    AtmosphereAdvanced advanced;
    
    // Camera position for volumetric clouds (updated every frame)
    float camera_y;  // Camera Y position in world space
    int frame_count; // For stochastic dithering

    // Atmosphere LUT (GPU Textures)
    AtmosphereLUTData lut;

    // Volume (Placeholder for later)
    float volume_density;
    float volume_anisotropy;
};

#ifndef __CUDACC__
#include <optional>
#include <string>
#include <vector>
#include <Texture.h>
class AtmosphereLUT;
class World {
public:
    World();
    ~World();

    void initializeLUT(); // Explicit init if needed

    WorldData getGPUData() const;

    // Setters
    void setMode(WorldMode mode);
    WorldMode getMode() const;
    std::string getHDRIPath() const;
    void setNishitaParams(const NishitaSkyParams& params);
    
    // Color Mode
    void setColor(const Vec3& color);
    void setColorIntensity(float intensity);
    Vec3 getColor() const;
    float getColorIntensity() const;

    // HDRI Mode
    void setHDRI(const std::string& path);
    void setHDRIRotation(float rotation_degrees);
    void setHDRIIntensity(float intensity);
    float getHDRIRotation() const;
    float getHDRIIntensity() const;
    bool hasHDRI() const;

    // Nishita Mode
    void setSunDirection(const Vec3& direction);
    void setSunIntensity(float intensity);
    void setPlanetRadius(float radius);
    void setAtmosphereHeight(float height);
    void setRayleighScattering(const Vec3& scattering);
    void setMieScattering(const Vec3& scattering);
    void setMieAnisotropy(float g);
    void setDustDensity(float density);
    
    NishitaSkyParams getNishitaParams() const;
    AtmosphereAdvanced getAdvancedParams() const;
    void setAdvancedParams(const AtmosphereAdvanced& a);
    
    // Environment Texture Overlay for Nishita
    void setNishitaEnvOverlay(const std::string& path);
    std::string getNishitaEnvOverlayPath() const;
    
    // Camera position for volumetric clouds
    void setCameraY(float y) { data.camera_y = y; }
    float getCameraY() const { return data.camera_y; }

    AtmosphereLUT* getLUT() const { return atmosphere_lut; }

    // CPU Evaluation (for background missing)
    Vec3 evaluate(const Vec3& ray_dir, const Vec3& origin = Vec3(0,0,0));
    WorldData data;
private:
 
   std::string hdri_path; // Store path for getter
   std::string env_overlay_path; // Store env overlay path
   Texture* hdri_texture = nullptr;
   Texture* env_overlay_texture = nullptr;
   AtmosphereLUT* atmosphere_lut = nullptr;
   
   // Internal helper for Nishita
   Vec3 calculateNishitaSky(const Vec3& ray_dir, const Vec3& origin = Vec3(0,0,0));

public:
    // Reset to default settings
    void reset();

    // Serialization
    void serialize(nlohmann::json& j) const;
    void deserialize(const nlohmann::json& j);
};
#endif

