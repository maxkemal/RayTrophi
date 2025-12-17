#pragma once
#include "Vec3.h"
#include <cuda_runtime.h>


// GPU-Compatible Enum
enum WorldMode {
    WORLD_MODE_COLOR = 0,
    WORLD_MODE_HDRI = 1,
    WORLD_MODE_NISHITA = 2
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
    
    // Night sky parameters
    float stars_intensity;   // Brightness of stars (0 = off)
    float stars_density;     // How many stars (0-1)
    
    // Moon parameters
    int moon_enabled;        // 1 = show moon, 0 = hide
    float moon_elevation;    // Moon height in degrees
    float moon_azimuth;      // Moon horizontal angle
    float moon_intensity;    // Moon brightness
    float moon_size;         // Moon angular size in degrees
    float moon_phase;        // 0 = new moon, 0.5 = full moon, 1 = new moon
    
    // Cloud Layer 1 (Primary) parameters
    int clouds_enabled;      // 1 = show clouds
    float cloud_coverage;    // 0.0 - 1.0 (how much sky is covered)
    float cloud_density;     // Opacity multiplier
    float cloud_scale;       // Noise frequency (larger = bigger clouds)
    float cloud_height_min;  // Bottom altitude (meters)
    float cloud_height_max;  // Top altitude (meters)
    float cloud_offset_x;    // Wind/Seed Offset X
    float cloud_offset_z;    // Wind/Seed Offset Z
    
    // Cloud Layer 2 (Secondary) parameters - for multi-layer clouds
    int cloud_layer2_enabled;     // 1 = show second layer
    float cloud2_coverage;        // 0.0 - 1.0
    float cloud2_density;         // Opacity multiplier
    float cloud2_scale;           // Noise frequency
    float cloud2_height_min;      // Bottom altitude (meters)
    float cloud2_height_max;      // Top altitude (meters)
    
    // Quality and detail settings
    float cloud_quality;     // Ray marching quality (0.25 = fast preview, 1.0 = normal, 2.0 = high quality)
    float cloud_detail;      // Detail level (0.5 = low, 1.0 = normal, 2.0 = high detail noise)
    
    // Cloud Lighting settings
    int cloud_light_steps;        // Number of light marching steps (0 = disabled, 4-8 recommended)
    float cloud_shadow_strength;  // Shadow darkness (0 = no shadows, 1 = normal, 2 = dark shadows)
    float cloud_ambient_strength; // Ambient light strength (0.5 = low, 1.0 = normal)
    float cloud_silver_intensity; // Silver lining intensity (0 = off, 1 = normal, 2 = strong)
    float cloud_absorption;       // Light absorption rate (0.5 = thin clouds, 1.0 = normal, 2.0 = thick)
    
    // Physical constants (usually not exposed in UI)
    float planet_radius;
    float atmosphere_height;
    float3 rayleigh_scattering;
    float3 mie_scattering;
    float mie_anisotropy;    // g factor (0.8 = forward scattering)
    float rayleigh_density;  // Scale height for Rayleigh
    float mie_density;       // Scale height for Mie
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
    
    // Camera position for volumetric clouds (updated every frame)
    float camera_y;  // Camera Y position in world space

    // Volume (Placeholder for later)
    float volume_density;
    float volume_anisotropy;
};

#ifndef __CUDACC__
#include <optional>
#include <string>
#include <vector>
class Texture;

class World {
public:
    World();
    ~World();

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
    
    // Camera position for volumetric clouds
    void setCameraY(float y) { data.camera_y = y; }
    float getCameraY() const { return data.camera_y; }

    // CPU Evaluation (for background missing)
    Vec3 evaluate(const Vec3& ray_dir);

private:
   WorldData data;
   std::string hdri_path; // Store path for getter
   Texture* hdri_texture = nullptr;
   
   // Internal helper for Nishita
   Vec3 calculateNishitaSky(const Vec3& ray_dir);
};
#endif
