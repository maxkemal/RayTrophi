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

enum WeatherType {
    WEATHER_NONE = 0,
    WEATHER_RAIN = 1,
    WEATHER_SNOW = 2,
    WEATHER_DUST = 3,
    WEATHER_MIST = 4
};

enum WeatherVisualMode {
    WEATHER_VISUAL_OVERLAY = 0,
    WEATHER_VISUAL_SURFACE_ONLY = 1
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
    float sun_intensity;     // Sun brightness (direct sun disk + direct lighting)
    float atmosphere_intensity; // Atmospheric scattering brightness (sky, halo, ambient)
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
    int cloud_seed;          // Procedural cloud pattern seed
    
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
    int cloud_base_steps;    // Base number of ray marching steps (fast default: 8)
    
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
    // A participating medium integrated into the aerial froxel together with
    // the air (atmosphere_aerial_froxel.comp):
    //   sigma(y) = fog_density * exp(-fog_falloff * max(y - fog_height, 0))
    // Below fog_height the layer is uniform; above it thins out.
    int fog_enabled;               // 1 = height fog on
    float fog_density;             // Extinction per metre at and below fog_height (typical 1e-5 - 1e-3)
    float fog_height;              // Top of the uniform layer, scene y in metres
    float fog_falloff;             // Exponential thinning above fog_height, 1/m (0.001 - 0.01)
    float fog_distance;            // Fog exists only within this distance of the camera (metres)
    float3 fog_albedo;             // Single-scattering albedo (Nishita: lit by sun + sky; other modes: radiance)
    float fog_anisotropy;          // Henyey-Greenstein g of the fog droplets, -0.95..0.95
    
    // ═══════════════════════════════════════════════════════════════
    // VOLUMETRIC LIGHT RAYS (God Rays / Light Shafts)
    // ═══════════════════════════════════════════════════════════════
    int godrays_enabled;           // 1 = show god rays
    float godrays_intensity;       // God ray brightness (0.0 - 2.0)
    float godrays_density;         // God ray density/thickness
    int godrays_samples;           // Quality (8-32 recommended)
    
    // Climate PACKET — written ONLY by World::syncClimatePacket() from
    // atmosphere::ClimateState. Renderers read these; nothing else writes
    // them (a copied NishitaSkyParams carrying a stale value is overwritten on
    // setNishitaParams). The authority is World::getClimate().
    float derived_relative_humidity; // 0..1, surface
    float derived_temperature_c;     // Celsius, surface; scales both scale heights
    float ozone_absorption_scale;  // Scales the "Blue Hour" intensity (0.0 to 10.0)
};

// ═══════════════════════════════════════════════════════════════
// ATMOSPHERE ADVANCED (Rendering Toggles)
// ═══════════════════════════════════════════════════════════════
struct AtmosphereAdvanced {
    int multi_scatter_enabled;     // 1 = enable multi-scattering
    float multi_scatter_factor;    // Multi-scatter intensity (0.0 - 1.0)
    // 1 = air scattering between the camera and surfaces (aerial froxel). How
    // much haze there is comes from the atmosphere itself -- air/dust density,
    // climate humidity -- not from a separate strength or distance ramp.
    int aerial_perspective;
    
    // Environment Texture Overlay (Moved here for better UI grouping)
    int env_overlay_enabled;       // 1 = blend environment texture with Nishita
    cudaTextureObject_t env_overlay_tex;  // HDR/EXR environment texture
    float env_overlay_intensity;   // Texture contribution (0.0 - 2.0)
    float env_overlay_rotation;    // Rotation in degrees (0 - 360)
    int env_overlay_blend_mode;    // 0 = Mix, 1 = Multiply, 2 = Screen, 3 = Replace
};

struct WeatherParams {
    int enabled;                    // 1 = weather system active
    int type;                       // WeatherType
    float intensity;                // Artist-facing strength, 0..1
    float density;                  // Particle/volume density, 0..1

    // Climate PACKET (see NishitaSkyParams::derived_*): mirrored from
    // atmosphere::ClimateState by World::syncClimatePacket(), never authored here.
    float3 derived_wind_direction;  // Normalized horizontal world-space wind direction
    float derived_wind_speed_mps;   // m/s at the surface

    float precipitation_scale;      // Visual streak/flake scale
    float visibility;               // 1 = clear, lower values reduce distance contrast
    float surface_wetness_output;   // Output signal for future surface interaction
    float surface_accumulation_output; // Output signal for snow/dust accumulation
    float surface_settling_output;  // Extra buildup in cavities, pockets, and slope bases
    float surface_height_output;    // Additional shading height for deposited material
    int visual_mode;                // WeatherVisualMode
    int surface_response_enabled;   // 1 = allow wetness/accumulation on materials
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
    WeatherParams weather;
    
    // Camera position for volumetric clouds (updated every frame)
    float camera_y;  // Camera Y position in world space
    int frame_count; // For stochastic dithering

    // Atmosphere LUT (GPU Textures)
    AtmosphereLUTData lut;

    // Volume (Placeholder for later)
    float volume_density;
    float volume_anisotropy;

    // Gate the Nishita-sky AMBIENT contribution to volume render (OptiX). The
    // OptiX volume in-scatter ambient samples the raw Nishita sky, which reads
    // much brighter than the Vulkan LUT-based ambient — breaking backend parity
    // (the sky over-lights the volume). Default 0 = OFF (volumes lit by sun +
    // scene lights only, matching Vulkan more closely). 1 = re-enable. The
    // underlying base-radiance parity (raw sky vs LUT) is the deeper fix.
    int volume_atmosphere_ambient;
};

#ifndef __CUDACC__
#include <optional>
#include <string>
#include <vector>
#include <Texture.h>
#include "Atmosphere/AtmosphereClimate.h"
#include "Atmosphere/AtmosphereClouds.h"
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
    const Texture* getHDRITexture() const { return hdri_texture; }

    // Nishita Mode
    void setSunDirection(const Vec3& direction);
    void setSunIntensity(float intensity);
    void setAtmosphereIntensity(float intensity);
    void setPlanetRadius(float radius);
    void setAtmosphereHeight(float height);
    void setRayleighScattering(const Vec3& scattering);
    void setMieScattering(const Vec3& scattering);
    void setMieAnisotropy(float g);
    void setDustDensity(float density);
    
    NishitaSkyParams getNishitaParams() const;
    AtmosphereAdvanced getAdvancedParams() const;
    void setAdvancedParams(const AtmosphereAdvanced& a);
    WeatherParams getWeatherParams() const;
    void setWeatherParams(const WeatherParams& params);

    // Climate authority (docs/dev/ATMOSPHERE_SYSTEM.md §2). setClimate rejects
    // invalid input instead of clamping; on success it normalizes the wind
    // direction, mirrors the render packet and dirties the LUT when the
    // temperature or humidity changed.
    const atmosphere::ClimateState& getClimate() const { return climate; }
    bool setClimate(const atmosphere::ClimateState& c, std::string* error = nullptr);
    atmosphere::DerivedWeather derivedWeather() const {
        return atmosphere::deriveWeather(climate, data.nishita.altitude);
    }
    // Samples at a scene position (metres); altitude = pos.y + nishita.altitude.
    atmosphere::ClimateSample sampleClimate(const Vec3& scene_pos) const;

    // Cloud authority (docs/dev/ATMOSPHERE_CLOUDS.md). Rejects invalid input.
    // The legacy nishita.cloud_* fields are a packet written from this.
    const atmosphere::CloudState& getClouds() const { return clouds; }
    bool setClouds(const atmosphere::CloudState& c, std::string* error = nullptr);
    // Timeline time the clouds are evaluated at (weather drift + evolution).
    // Set by the timeline applier; a function of the frame, so a replayed
    // frame gets the same sky.
    void setCloudTime(float seconds);
    float getCloudTime() const { return cloud_time_seconds; }
    // Accumulated weather drift in metres (x, z): climate wind x time. Exact
    // for a constant wind; a keyed wind uses the current velocity (still
    // deterministic per frame).
    Vec3 cloudWindOffset() const;
    // Climate wind (m/s): slants the precipitation shafts.
    Vec3 cloudWindVelocity() const { return climate.wind_direction * climate.wind_speed_mps; }
    // Bumps on every change that alters the cloud field (params or time).
    uint64_t cloudRevision() const { return cloud_revision; }

    // Environment Texture Overlay for Nishita
    void setNishitaEnvOverlay(const std::string& path);
    std::string getNishitaEnvOverlayPath() const;
    const Texture* getNishitaEnvOverlayTexture() const { return env_overlay_texture; }
    
    // Camera position for volumetric clouds
    void setCameraY(float y) { data.camera_y = y; }
    float getCameraY() const { return data.camera_y; }

    // Nishita sky ambient -> volume render gate (OptiX). Default off for
    // Vulkan parity; see WorldData::volume_atmosphere_ambient.
    void setVolumeAtmosphereAmbient(bool on) { data.volume_atmosphere_ambient = on ? 1 : 0; }
    bool getVolumeAtmosphereAmbient() const { return data.volume_atmosphere_ambient != 0; }

    AtmosphereLUT* getLUT() const { return atmosphere_lut; }

    // Deferred LUT mechanism: avoids recomputing 50K-pixel LUT on every slider tick
    bool needsLUTUpdate() const { return lut_dirty; }
    void clearLUTDirty() { lut_dirty = false; }
    bool flushLUT();  // Returns true when the LUT was recomputed.
    bool rebuildLUT(); // Force a LUT rebuild from current parameters.

    // CPU Evaluation (for background missing)
    Vec3 evaluate(const Vec3& ray_dir, const Vec3& origin = Vec3(0,0,0));
    WorldData data;
private:
 
   std::string hdri_path; // Store path for getter
   std::string env_overlay_path; // Store env overlay path
   Texture* hdri_texture = nullptr;
   Texture* env_overlay_texture = nullptr;
   AtmosphereLUT* atmosphere_lut = nullptr;
   bool lut_dirty = false;  // Deferred LUT recomputation flag
   atmosphere::ClimateState climate;
   atmosphere::CloudState clouds;
   float cloud_time_seconds = 0.0f;
   uint64_t cloud_revision = 1;
   // Writes the legacy nishita.cloud_* packet from `clouds`. Called from
   // syncClimatePacket (wind feeds the drift) and after cloud/time writes.
   bool rederiveClouds();
   void syncCloudPacket();

   // Writes the derived_* packet fields from `climate`. Called after every
   // write that could carry a stale copy of them (setNishitaParams,
   // setWeatherParams) and after every climate change.
   void syncClimatePacket();
   
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

