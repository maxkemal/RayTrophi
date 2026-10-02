#include "globals.h"
#include "AtmosphereLUT.h"
#include "vec3_utils.cuh"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// Helper for float3
inline float3 to_float3(const Vec3& v) { return make_float3(v.x, v.y, v.z); }
inline Vec3 to_vec3(const float3& v) { return Vec3(v.x, v.y, v.z); }

static Vec3 blendEnvironmentOverlayCPU(const Vec3& base, const Vec3& sampled, float intensity, int blendMode) {
    const float strength = std::max(0.0f, intensity);
    const float amount = std::min(strength, 1.0f);
    const Vec3 overlay = sampled * strength;

    switch (blendMode) {
        case 1:
            return base * (Vec3(1.0f) * (1.0f - amount) + sampled * amount);
        case 2:
            return base + overlay;
        case 3:
            return overlay;
        case 0:
        default:
            return base * (1.0f - amount) + overlay * amount;
    }
}

World::World() {
    data.mode = WORLD_MODE_NISHITA; // Default to Nishita Sky
    data.color = make_float3(0.05f, 0.05f, 0.05f); 
    data.color_intensity = 1.0f;
    data.env_texture = 0;
    data.env_rotation = 0.0f;
    data.env_intensity = 1.0f;
    data.env_width = 0;
    data.env_height = 0;

    // Nishita Defaults (Blender-compatible Earth-like atmosphere)
    data.nishita.sun_elevation = 15.0f; // Golden hour default
    data.nishita.sun_azimuth = 170.0f;   // Matches typical camera setup
    
    // Calculate initial direction for 15/45 degrees
    float elevRad = data.nishita.sun_elevation * 3.14159265f / 180.0f;
    float azimRad = data.nishita.sun_azimuth * 3.14159265f / 180.0f;
    
    // Standard conversion (Y-up, assuming +Z is South/Forward)
    // Adjust signs to match Renderer coordinate system (Match UI Logic: Positive Z)
    data.nishita.sun_direction = normalize(make_float3(
        cosf(elevRad) * sinf(azimRad), 
        sinf(elevRad), 
        cosf(elevRad) * cosf(azimRad)
    ));
    data.nishita.sun_intensity = 10.0f;
    data.nishita.atmosphere_intensity = 10.0f;
    data.nishita.sun_size = 0.545f;        // Real sun angular size in degrees
    
    // Blender-style atmosphere multipliers (default 1.0)
    data.nishita.air_density = 1.0f;       // Air (Rayleigh) multiplier
    data.nishita.dust_density = 1.0f;      // Dust (Mie/aerosols) multiplier
    data.nishita.ozone_density = 1.0f;     // Ozone multiplier
    data.nishita.altitude = 0.0f;          // Sea level (meters)
    
    
    // Procedural VDB cloud defaults
    data.nishita.clouds_enabled = 0;       // Disabled by default
    data.nishita.cloud_coverage = 0.18f;
    data.nishita.cloud_density = 0.35f;
    data.nishita.cloud_scale = 0.45f;
    data.nishita.cloud_height_min = 2500.0f;
    data.nishita.cloud_height_max = 4200.0f;
    data.nishita.cloud_offset_x = 0.0f;
    data.nishita.cloud_offset_z = 0.0f;
    data.nishita.cloud_seed = 0;
    data.nishita.cloud_quality = 1.0f;     // Normal quality (1.0 = 64-128 steps)
    data.nishita.cloud_detail = 0.55f;     // Normal detail level
    data.nishita.cloud_base_steps = 8;     // Fast default for procedural sky volume
    
    // Cloud Layer 2 defaults (High altitude cirrus-like)
    data.nishita.cloud_layer2_enabled = 0;   // Disabled by default
    data.nishita.cloud2_coverage = 0.16f;
    data.nishita.cloud2_density = 0.18f;      // Thin/transparent
    data.nishita.cloud2_scale = 0.25f;        // High wispy layer
    data.nishita.cloud2_height_min = 7000.0f;
    data.nishita.cloud2_height_max = 8500.0f;
    
    // Cloud Lighting defaults
    data.nishita.cloud_light_steps = 0;      // Light marching steps (0 = disabled)
    data.nishita.cloud_shadow_strength = 0.35f;
    data.nishita.cloud_ambient_strength = 1.0f;
    data.nishita.cloud_silver_intensity = 0.25f;
    data.nishita.cloud_absorption = 1.0f;
    
    // Cloud Advanced Scattering (VDB-like) defaults
    data.nishita.cloud_anisotropy = 0.55f;
    data.nishita.cloud_anisotropy_back = -0.2f;
    data.nishita.cloud_lobe_mix = 0.75f;
    
    // Cloud Emissive (Experimental) defaults
    data.nishita.cloud_emissive_color = make_float3(1.0f, 1.0f, 1.0f);
    data.nishita.cloud_emissive_intensity = 0.0f; // Off by default
    
    // Camera position for volumetric clouds
    data.camera_y = 0.0f;  // Will be updated from scene camera
    
    // Physical constants (ALL IN METERS for consistency with scattering coefficients)
    data.nishita.planet_radius = 6360000.0f;     // Earth radius in meters
    data.nishita.atmosphere_height = 60000.0f;   // Atmosphere thickness in meters
    
    // Rayleigh coefficients for RGB (per meter, standard earth)
    data.nishita.rayleigh_scattering = make_float3(5.802e-6f, 13.558e-6f, 33.100e-6f);
    data.nishita.mie_scattering = make_float3(3.996e-6f, 3.996e-6f, 3.996e-6f);
    data.nishita.mie_anisotropy = 0.98f;
    data.nishita.rayleigh_density = 8000.0f;     // Scale height H_R in meters
    data.nishita.mie_density = 1200.0f;          // Scale height H_M in meters
    
    // ═══════════════════════════════════════════════════════════
    // ATMOSPHERIC FOG DEFAULTS
    // ═══════════════════════════════════════════════════════════
    data.nishita.fog_enabled = 0;                // Disabled by default
    data.nishita.fog_density = 0.0001f;         // Light fog, extinction per metre
    data.nishita.fog_height = 500.0f;            // Uniform layer up to 500 m
    data.nishita.fog_falloff = 0.003f;           // Gradual thinning above it
    data.nishita.fog_distance = 10000.0f;        // 10km max distance
    // Water droplets barely absorb; the blue/grey tint comes from the sky
    // that lights the fog, not from the albedo.
    data.nishita.fog_albedo = make_float3(0.95f, 0.95f, 0.95f);
    data.nishita.fog_anisotropy = 0.6f;          // Forward-scattering droplets
    
    // ═══════════════════════════════════════════════════════════
    // VOLUMETRIC LIGHT RAYS (GOD RAYS) DEFAULTS
    // ═══════════════════════════════════════════════════════════
    data.nishita.godrays_enabled = 0;            // Disabled by default
    data.nishita.godrays_intensity = 0.5f;       // Medium intensity
    data.nishita.godrays_density = 0.1f;         // Light density

    // Atmosphere Advanced Defaults
    data.advanced.multi_scatter_enabled = 1;
    data.advanced.multi_scatter_factor = 0.3f;
    data.advanced.aerial_perspective = 1;
    data.advanced.env_overlay_enabled = 0;
    data.advanced.env_overlay_intensity = 1.0f;
    data.advanced.env_overlay_rotation = 0.0f;
    data.advanced.env_overlay_blend_mode = 0;

    data.weather.enabled = 0;
    data.weather.type = WEATHER_NONE;
    data.weather.intensity = 0.0f;
    data.weather.density = 0.0f;
    data.weather.precipitation_scale = 1.0f;
    data.weather.visibility = 1.0f;
    data.weather.surface_wetness_output = 0.0f;
    data.weather.surface_accumulation_output = 0.0f;
    data.weather.surface_settling_output = 0.0f;
    data.weather.surface_height_output = 0.0f;
    data.weather.visual_mode = WEATHER_VISUAL_OVERLAY;
    data.weather.surface_response_enabled = 1;

    // Physical Defaults
    climate = atmosphere::ClimateState{};
    syncClimatePacket();
    data.nishita.ozone_absorption_scale = 1.0f;
    data.nishita.godrays_samples = 16;           // Balanced quality
    data.nishita.godrays_samples = 16;           // Balanced quality
    // Advanced Atmosphere Defaults (Already set above in data.advanced)
    // Removed redundant nishita access
    
    // Env Overlay (Now handled in advanced)
    
    // Future volume params
    data.volume_density = 0.0f;
    data.volume_anisotropy = 0.0f;
    // Off by default — Nishita sky ambient over-lights OptiX volumes vs Vulkan.
    data.volume_atmosphere_ambient = 0;

    // Initialize LUT handles to zero
    data.lut.transmittance_lut = 0;
    data.lut.skyview_lut = 0;
    data.lut.integrated_multi_scattering = make_float3(0, 0, 0);
}

void World::initializeLUT() {
    if (!atmosphere_lut) {
        atmosphere_lut = new AtmosphereLUT();
    }

    atmosphere_lut->initialize();
    atmosphere_lut->precompute(data.nishita);

    if (g_hasCUDA) {
        data.lut = atmosphere_lut->getGPUData();
    } else {
        // CPU rendering still needs the host LUTs even when no GPU textures exist.
        data.lut.transmittance_lut = 0;
        data.lut.skyview_lut = 0;
        data.lut.multi_scattering_lut = 0;
        data.lut.aerial_perspective_lut = 0;
    }
}

World::~World() {
    if (hdri_texture) {
        delete hdri_texture;
        hdri_texture = nullptr;
    }
    if (atmosphere_lut) {
        delete atmosphere_lut;
        atmosphere_lut = nullptr;
    }
    if (env_overlay_texture) {
        delete env_overlay_texture;
        env_overlay_texture = nullptr;
    }
}

WorldData World::getGPUData() const {
    WorldData gpuData = data;
    if (isCudaTextureUploadAllowed()) {
        if (hdri_texture && hdri_texture->is_loaded() && !hdri_texture->isUploaded()) {
            hdri_texture->upload_to_gpu();
        }
        if (hdri_texture && hdri_texture->isUploaded()) {
            gpuData.env_texture = hdri_texture->get_cuda_texture();
            gpuData.env_width = hdri_texture->width;
            gpuData.env_height = hdri_texture->height;
        }

        if (env_overlay_texture && env_overlay_texture->is_loaded() && !env_overlay_texture->isUploaded()) {
            env_overlay_texture->upload_to_gpu();
        }
        if (env_overlay_texture && env_overlay_texture->isUploaded()) {
            // Bind the handle, but DO NOT force-enable — respect the user's
            // env_overlay_enabled flag (already copied from `data`). Forcing it
            // on meant a loaded overlay showed even when the checkbox was off,
            // and leaked across project loads.
            gpuData.advanced.env_overlay_tex = env_overlay_texture->get_cuda_texture();
        }
    }
    if (atmosphere_lut) {
        gpuData.lut = atmosphere_lut->getGPUData();
    }
    return gpuData;
}

void World::setMode(WorldMode mode) {
    data.mode = mode;

    // Lazy-load HDRI only when HDRI mode is actually activated.
    // This avoids trying to open stale/missing paths during project load
    // when the current world mode is not HDRI.
    if (mode == WORLD_MODE_HDRI && !hdri_path.empty()) {
        const bool hdriReady = (hdri_texture && hdri_texture->is_loaded());
        if (!hdriReady) {
            setHDRI(hdri_path);
        }
    }
}

WorldMode World::getMode() const {
    return static_cast<WorldMode>(data.mode);
}

std::string World::getHDRIPath() const {
    return hdri_path;
}

void World::setNishitaParams(const NishitaSkyParams& params) {
    // Preserve the loaded env overlay texture - it's set by setNishitaEnvOverlay
    cudaTextureObject_t savedTex = data.advanced.env_overlay_tex;

    const NishitaSkyParams previous = data.nishita;
    
    data.nishita = params;
    // `params` is usually a copy taken earlier; its climate packet may be
    // stale. The climate authority wins, always.
    syncClimatePacket();

    // Restore the texture handle
    if (savedTex != 0) {
        data.advanced.env_overlay_tex = savedTex;
    }

    // Temperature/humidity are NOT compared here: they belong to the climate
    // and setClimate() dirties the LUT for them.
    const bool lutRelevantChanged =
        previous.atmosphere_intensity != params.atmosphere_intensity ||
        previous.sun_elevation != params.sun_elevation ||
        previous.sun_azimuth != params.sun_azimuth ||
        previous.sun_direction.x != params.sun_direction.x ||
        previous.sun_direction.y != params.sun_direction.y ||
        previous.sun_direction.z != params.sun_direction.z ||
        previous.air_density != params.air_density ||
        previous.dust_density != params.dust_density ||
        previous.ozone_density != params.ozone_density ||
        previous.ozone_absorption_scale != params.ozone_absorption_scale ||
        previous.altitude != params.altitude ||
        previous.mie_anisotropy != params.mie_anisotropy ||
        previous.planet_radius != params.planet_radius ||
        previous.atmosphere_height != params.atmosphere_height ||
        previous.rayleigh_scattering.x != params.rayleigh_scattering.x ||
        previous.rayleigh_scattering.y != params.rayleigh_scattering.y ||
        previous.rayleigh_scattering.z != params.rayleigh_scattering.z ||
        previous.mie_scattering.x != params.mie_scattering.x ||
        previous.mie_scattering.y != params.mie_scattering.y ||
        previous.mie_scattering.z != params.mie_scattering.z ||
        previous.rayleigh_density != params.rayleigh_density ||
        previous.mie_density != params.mie_density;

    // DEFERRED: skyview LUT bakes sun direction and Mie halo, so sun movement
    // must also dirty it. Runtime decides whether Vulkan compute or CPU fallback
    // performs the actual refresh.
    if (lutRelevantChanged) {
        lut_dirty = true;
    }
}

bool World::flushLUT() {
    if (!atmosphere_lut) {
        initializeLUT();
        lut_dirty = false;
        return true;
    }

    if (!lut_dirty) return false;

    atmosphere_lut->precompute(data.nishita);
    lut_dirty = false;
    return true;
}

bool World::rebuildLUT() {
    if (!atmosphere_lut) {
        initializeLUT();
        lut_dirty = false;
        return true;
    }

    atmosphere_lut->precompute(data.nishita);
    lut_dirty = false;
    return true;
}

void World::setColor(const Vec3& color) {
    data.color = to_float3(color);
}

void World::setColorIntensity(float intensity) {
    data.color_intensity = intensity;
}

Vec3 World::getColor() const {
    return to_vec3(data.color);
}

float World::getColorIntensity() const {
    return data.color_intensity;
}

void World::setHDRI(const std::string& path) {
    if (hdri_texture) {
        delete hdri_texture;
        hdri_texture = nullptr;
    }

    // We use Emission type for HDRI to preserve intensity as much as possible,
    // though Texture class sRGB logic mainly affects Albedo.
    // Ideally we want Linear loading check.
    hdri_path = path;
    hdri_texture = new Texture(path, TextureType::Emission);
    
    if (hdri_texture->is_loaded()) {
        // Log HDR status
        SCENE_LOG_INFO("HDRI loaded: " + path + " | is_hdr=" + (hdri_texture->is_hdr ? "TRUE (float)" : "FALSE (uchar)"));
        
        data.env_width = hdri_texture->width;
        data.env_height = hdri_texture->height;
        if (!isCudaTextureUploadAllowed()) {
            data.env_texture = 0;
            SCENE_LOG_INFO("HDRI loaded; GPU upload deferred until render backend sync.");
        } else {
            bool uploaded = hdri_texture->upload_to_gpu();
            if (uploaded && hdri_texture->get_cuda_texture()) {
                data.env_texture = hdri_texture->get_cuda_texture();
                SCENE_LOG_INFO("HDRI uploaded to GPU: " + std::to_string(data.env_width) + "x" + std::to_string(data.env_height));
            } else if (uploaded) {
                data.env_texture = 0;
                SCENE_LOG_INFO("HDRI loaded; CUDA upload deferred until OptiX render sync.");
            } else {
                 SCENE_LOG_ERROR("Failed to upload HDRI to GPU");
                 data.env_texture = 0;
            }
        }
    } else {
        SCENE_LOG_ERROR("Failed to load HDRI texture: " + path);
        delete hdri_texture;
        hdri_texture = nullptr;
        data.env_texture = 0;
    }
}

void World::setHDRIRotation(float rotation_degrees) {
    data.env_rotation = rotation_degrees * (M_PI / 180.0f);
}

void World::setHDRIIntensity(float intensity) {
    data.env_intensity = intensity;
}

float World::getHDRIRotation() const {
    return data.env_rotation * (180.0f / M_PI);
}

float World::getHDRIIntensity() const {
    return data.env_intensity;
}

bool World::hasHDRI() const {
    return hdri_texture != nullptr && hdri_texture->is_loaded();
}

void World::setNishitaEnvOverlay(const std::string& path) {
    if (env_overlay_texture) {
        delete env_overlay_texture;
        env_overlay_texture = nullptr;
    }

    env_overlay_path = path;
    env_overlay_texture = new Texture(path, TextureType::Emission);
    
    if (env_overlay_texture->is_loaded()) {
        SCENE_LOG_INFO("Nishita Env Overlay loaded: " + path + " | is_hdr=" + (env_overlay_texture->is_hdr ? "TRUE" : "FALSE"));

        if (!isCudaTextureUploadAllowed()) {
            data.advanced.env_overlay_tex = 0;
            SCENE_LOG_INFO("Env Overlay loaded; GPU upload deferred until render backend sync.");
        } else {
            bool uploaded = env_overlay_texture->upload_to_gpu();
            if (uploaded && env_overlay_texture->get_cuda_texture()) {
                data.advanced.env_overlay_tex = env_overlay_texture->get_cuda_texture();
                SCENE_LOG_INFO("Env Overlay uploaded: " + std::to_string(env_overlay_texture->width) + "x" + std::to_string(env_overlay_texture->height));
            } else if (uploaded) {
                data.advanced.env_overlay_tex = 0;
                SCENE_LOG_INFO("Env Overlay loaded; CUDA upload deferred until OptiX render sync.");
            } else {
                SCENE_LOG_ERROR("Failed to upload Env Overlay to GPU");
                data.advanced.env_overlay_tex = 0;
            }
        }
    } else {
        SCENE_LOG_ERROR("Failed to load Env Overlay texture: " + path);
        delete env_overlay_texture;
        env_overlay_texture = nullptr;
        data.advanced.env_overlay_tex = 0;
        data.advanced.env_overlay_enabled = 0;
    }
}

std::string World::getNishitaEnvOverlayPath() const {
    return env_overlay_path;
}

void World::setSunDirection(const Vec3& direction) {
    Vec3 dir = direction.normalize();
    data.nishita.sun_direction = to_float3(dir);
    
    // Back-calculate Elevation & Azimuth for UI consistency
    // Y is Up (sin(elevation))
    float asin_y = asinf(fmaxf(-1.0f, fminf(1.0f, dir.y)));
    data.nishita.sun_elevation = asin_y * (180.0f / 3.14159265f);
    
    // X and Z define azimuth
    // x = cos(elev) * sin(azim)
    // z = cos(elev) * cos(azim)
    // atan2(x, z) gives azim
    float azimRad = atan2f(dir.x, dir.z);
    
    // Convert to degrees
    float azimDeg = azimRad * (180.0f / 3.14159265f);
    
    // Normalize to 0-360 if needed, or keep as is.
    // UI usually expects positive 0-360 or similar, but atan2 returns -180 to 180.
    if (azimDeg < 0.0f) azimDeg += 360.0f;
    
    data.nishita.sun_azimuth = azimDeg;
    
    // DEFERRED: Mark LUT dirty instead of immediate precompute
    lut_dirty = true;
}

void World::setSunIntensity(float intensity) {
    data.nishita.sun_intensity = intensity;
    // sun_intensity no longer baked into LUT — atmosphere_intensity is used instead
}

void World::setAtmosphereIntensity(float intensity) {
    data.nishita.atmosphere_intensity = intensity;
    // atmosphere_intensity is baked into the skyview LUT
    lut_dirty = true;
}

void World::setPlanetRadius(float radius) {
    data.nishita.planet_radius = radius;
}

void World::setAtmosphereHeight(float height) {
    data.nishita.atmosphere_height = height;
}

void World::setRayleighScattering(const Vec3& scattering) {
    data.nishita.rayleigh_scattering = to_float3(scattering);
}

void World::setMieScattering(const Vec3& scattering) {
    data.nishita.mie_scattering = to_float3(scattering);
}

void World::setMieAnisotropy(float g) {
    data.nishita.mie_anisotropy = g;
}

void World::setDustDensity(float density) {
    data.nishita.dust_density = density;
}

NishitaSkyParams World::getNishitaParams() const {
    return data.nishita;
}

static Vec3 weatherTintColor(const WeatherParams& w) {
    switch (w.type) {
        case WEATHER_RAIN: return Vec3(0.50f, 0.56f, 0.62f);
        case WEATHER_SNOW: return Vec3(0.86f, 0.90f, 0.96f);
        case WEATHER_DUST: return Vec3(0.74f, 0.58f, 0.38f);
        case WEATHER_MIST: return Vec3(0.70f, 0.76f, 0.82f);
        default: return Vec3(0.0f);
    }
}

static Vec3 applyWeatherSkyCPU(const WeatherParams& w, const Vec3& sky, const Vec3& rayDir) {
    if (w.enabled == 0 || w.type == WEATHER_NONE || w.intensity <= 0.0f || w.density <= 0.0f) {
        return sky;
    }

    const float visibilityLoss = std::max(0.0f, std::min(1.0f, 1.0f - w.visibility));
    const float horizon = std::pow((std::max)(0.0f, 1.0f - std::abs(rayDir.y)), 0.65f);
    float amount = w.intensity * (0.25f + w.density * 0.75f + visibilityLoss * 0.65f);
    amount = std::max(0.0f, std::min(0.85f, amount * (0.35f + horizon * 0.65f)));

    Vec3 tint = weatherTintColor(w);
    Vec3 dimmed = sky;
    if (w.type == WEATHER_RAIN) {
        dimmed *= 0.72f;
    } else if (w.type == WEATHER_DUST) {
        dimmed *= 0.82f;
    } else if (w.type == WEATHER_SNOW || w.type == WEATHER_MIST) {
        dimmed = dimmed * 0.90f + tint * 0.10f;
    }
    return dimmed * (1.0f - amount) + tint * amount;
}

// Simplified Nishita for CPU Preview (Optional, for now returns simple gradient or black if unimplemented)
// Implementing full Nishita on CPU might be slow for real-time without optimization.
Vec3 World::evaluate(const Vec3& ray_dir, const Vec3& origin) {
    if (data.mode == WORLD_MODE_COLOR) {
        return applyWeatherSkyCPU(data.weather, to_vec3(data.color) * data.color_intensity, ray_dir);
    }
    else if (data.mode == WORLD_MODE_HDRI) {
        if (hdri_texture && hdri_texture->is_loaded()) {
            // Calculate UV - match GPU logic
            float theta = acosf(ray_dir.y);
            float phi = atan2f(-ray_dir.z, ray_dir.x) + M_PI;
            
            float u = phi / (2.0f * M_PI); // 0..1
            float v = theta / M_PI;        // 0..1 (GPU style, top=0)
            
            // Apply rotation (env_rotation is in radians)
            u -= data.env_rotation / (2.0f * M_PI);
            u = u - floorf(u);  // Wrap to 0..1
            
            // get_color internally does y = (1-v)*height, so pass 1-v to cancel the flip
            // This makes CPU match GPU texture sampling
            return applyWeatherSkyCPU(data.weather, hdri_texture->get_color(u, 1.0f - v) * data.env_intensity, ray_dir);
        }
        return applyWeatherSkyCPU(data.weather, to_vec3(data.color), ray_dir); // Fallback
    }
    else if (data.mode == WORLD_MODE_NISHITA) {
        return calculateNishitaSky(ray_dir, origin);
    }
    return Vec3(0);
}

// MATCHED TO GPU: High-Quality LUT based Sky radiance
Vec3 World::calculateNishitaSky(const Vec3& ray_dir, const Vec3& origin) {
    float3 dir = normalize(to_float3(ray_dir));
    float3 sunDir = normalize(data.nishita.sun_direction);
    
    Vec3 L(0.0f);

    if (atmosphere_lut && atmosphere_lut->is_initialized()) {
        // 1. Sample Background from SkyView LUT
        float3 radiance = atmosphere_lut->sampleSkyView(dir, sunDir, data.nishita.planet_radius, data.nishita.planet_radius + data.nishita.atmosphere_height);
        L = to_vec3(radiance);
    } else {
        // Fallback to simple gradient if LUT not ready
        float t = 0.5f * (dir.y + 1.0f);
        L = Vec3(0.5f, 0.7f, 1.0f) * (1.0f - t) + Vec3(0.1f, 0.2f, 0.5f) * t;
    }

    // --- PROCEDURAL SUN GLOW (Matches GPU Exactly) ---
    float mu = dot(dir, sunDir);
    float g_mie = data.nishita.mie_anisotropy;
    float phaseM = (1.0f - g_mie * g_mie) / (4.0f * 3.14159f * powf(std::max(1.0f + g_mie * g_mie - 2.0f * g_mie * mu, 0.0001f), 1.5f));
    
    // Soft halo contribution (excess above clamped LUT phase)
    float excessPhase = fmaxf(0.0f, phaseM - 2.0f); // Matched to 2.0f LUT clamp
    
    if (excessPhase > 0.0f && atmosphere_lut) {
        // COORDINATE SYNC: Camera Y=0 is planet surface. Center is at (0, -Rg, 0)
        float Rg = data.nishita.planet_radius;
        Vec3 p = origin + Vec3(0, Rg, 0);
        float current_altitude = std::max(0.0f, p.length() - Rg);
        
        float cosTheta = std::max(0.01f, sunDir.y);
        float3 transSun = atmosphere_lut->sampleTransmittance(cosTheta, current_altitude, data.nishita.atmosphere_height);
        
        Vec3 mieScat = to_vec3(data.nishita.mie_scattering) * (data.nishita.mie_density * 0.15f);
        L += to_vec3(transSun) * (mieScat * excessPhase * data.nishita.atmosphere_intensity);
    }
    
    // --- MULTI-SCATTERING (Matches GPU Exactly) ---
    if (data.advanced.multi_scatter_enabled) {
        Vec3 scatteringAlbedo(0.8f, 0.85f, 0.9f);
        float multiFactor = data.advanced.multi_scatter_factor;
        
        // Approximation logic matching GPU apply_multi_scattering
        Vec3 ms = L;
        Vec3 secondOrder = L * scatteringAlbedo * 0.5f * expf(-0.5f * 0.3f);
        ms = ms + secondOrder * multiFactor;
        Vec3 thirdOrder = secondOrder * scatteringAlbedo * 0.25f * expf(-0.5f * 0.1f);
        ms = ms + thirdOrder * multiFactor * 0.5f;
        L = ms;
    }

    // --- SUN DISK (Matches GPU Exactly) ---
    float sunSizeDeg = data.nishita.sun_size;
    float elevationFactor = 1.0f;
    if (data.nishita.sun_elevation < 15.0f) {
        elevationFactor = 1.0f + (15.0f - std::max(data.nishita.sun_elevation, -10.0f)) * 0.04f;
    }
    sunSizeDeg *= elevationFactor;
    
    float sun_radius = sunSizeDeg * (3.14159265f / 180.0f) * 0.5f;
    float sun_cos_threshold = cosf(sun_radius);
    
    if (mu > sun_cos_threshold) {
         float angular_dist = acosf(std::min(1.0f, mu));
         float radial_pos = angular_dist / sun_radius;
         
         float u_limb = 0.6f;
         float cosine_mu = sqrtf(std::max(0.0f, 1.0f - radial_pos * radial_pos));
         float limbDarkening = 1.0f - u_limb * (1.0f - cosine_mu);
         
         float edge_t = std::max(0.0f, std::min(1.0f, (radial_pos - 0.85f) / 0.15f));
         float edgeSoftness = 1.0f - edge_t * edge_t * (3.0f - 2.0f * edge_t);
         
         // Transmittance to sun (altitude-dependent)
         float Rg = data.nishita.planet_radius;
         Vec3 p = origin + Vec3(0, Rg, 0);
         float current_altitude = std::max(0.0f, p.length() - Rg);
         
         float3 trans = { 1.0f, 1.0f, 1.0f };
         if (atmosphere_lut) {
             trans = atmosphere_lut->sampleTransmittance(std::max(0.01f, sunDir.y), current_altitude, data.nishita.atmosphere_height);
         }
         // Multiplier matched to GPU (80000.0f)
         L += to_vec3(trans) * data.nishita.sun_intensity * 80000.0f * limbDarkening * edgeSoftness;
    }

    if (data.advanced.env_overlay_enabled &&
        env_overlay_texture &&
        env_overlay_texture->is_loaded()) {
        float theta = acosf(std::max(-1.0f, std::min(1.0f, dir.y)));
        float phi = atan2f(-dir.z, dir.x) + static_cast<float>(M_PI);
        float u = phi / (2.0f * static_cast<float>(M_PI));
        float v = theta / static_cast<float>(M_PI);
        u -= data.advanced.env_overlay_rotation / 360.0f;
        u = u - floorf(u);
        Vec3 overlay = env_overlay_texture->get_color_bilinear(u, 1.0f - v);
        L = blendEnvironmentOverlayCPU(
            L,
            overlay,
            data.advanced.env_overlay_intensity,
            data.advanced.env_overlay_blend_mode);
    }

    return applyWeatherSkyCPU(data.weather, L, ray_dir);
}
       

// Reset to defaults
void World::reset() {
    // Selective reset - preserve LUT object but reset data
    data.mode = WORLD_MODE_NISHITA;
    data.color = make_float3(0.05f, 0.05f, 0.05f);
    data.color_intensity = 1.0f;
    data.env_texture = 0;
    data.env_rotation = 0.0f;
    data.env_intensity = 1.0f;
    
    // Nishita Defaults
    NishitaSkyParams defaults = {};
    defaults.sun_elevation = 15.0f;
    defaults.sun_azimuth = 170.0f;
    defaults.sun_intensity = 10.0f;
    defaults.atmosphere_intensity = 10.0f;
    defaults.sun_size = 0.545f;
    defaults.air_density = 1.0f;
    defaults.dust_density = 1.0f;
    defaults.ozone_density = 1.0f;
    defaults.altitude = 0.0f;
    
    defaults.planet_radius = 6360000.0f;
    defaults.atmosphere_height = 60000.0f;
    defaults.rayleigh_scattering = make_float3(5.802e-6f, 13.558e-6f, 33.100e-6f);
    defaults.mie_scattering = make_float3(3.996e-6f, 3.996e-6f, 3.996e-6f);
    defaults.mie_anisotropy = 0.98f;
    defaults.rayleigh_density = 8000.0f;
    defaults.mie_density = 1200.0f;
    
    // Recalculate sun direction from default angles
    float elevRad = defaults.sun_elevation * 3.14159265f / 180.0f;
    float azimRad = defaults.sun_azimuth * 3.14159265f / 180.0f;
    defaults.sun_direction = normalize(make_float3(
        cosf(elevRad) * sinf(azimRad), 
        sinf(elevRad), 
        cosf(elevRad) * cosf(azimRad)
    ));
    
    defaults.ozone_absorption_scale = 1.0f;

    data.nishita = defaults;
    climate = atmosphere::ClimateState{};
    clouds = atmosphere::CloudState{};
    cloud_time_seconds = 0.0f;
    ++cloud_revision;

    // Clear the Nishita env-overlay HDR — without this it survived reset() and
    // leaked into the next project (and showed even with the overlay disabled,
    // due to the old force-enable in getGPUData).
    if (env_overlay_texture) {
        delete env_overlay_texture;
        env_overlay_texture = nullptr;
    }
    env_overlay_path.clear();
    data.advanced.env_overlay_enabled = 0;
    data.advanced.env_overlay_tex = 0;
    data.advanced.env_overlay_intensity = 1.0f;
    data.advanced.env_overlay_rotation = 0.0f;
    data.advanced.env_overlay_blend_mode = 0;

    data.weather.enabled = 0;
    data.weather.type = WEATHER_NONE;
    data.weather.intensity = 0.0f;
    data.weather.density = 0.0f;
    data.weather.precipitation_scale = 1.0f;
    data.weather.visibility = 1.0f;
    data.weather.surface_wetness_output = 0.0f;
    data.weather.surface_accumulation_output = 0.0f;
    syncClimatePacket();

    // Defer the LUT re-bake instead of precomputing synchronously: reset() runs inside
    // newProject/openProject where a ~1.6s CPU sky bake blocked every project transition —
    // even loads that immediately overwrite these defaults (deserialize) or never render
    // the atmosphere at all (HDRI/color scenes). The runtime flushes lazily when Nishita
    // is actually rendered: Vulkan generates the LUT on GPU, the OptiX/CPU world-sync
    // branch and the CPU ray path call flushLUT() (which also self-initializes a missing
    // atmosphere_lut).
    lut_dirty = true;
}

AtmosphereAdvanced World::getAdvancedParams() const {
    return data.advanced;
}

void World::setAdvancedParams(const AtmosphereAdvanced& a) {
    data.advanced = a;
}

WeatherParams World::getWeatherParams() const {
    return data.weather;
}

void World::setWeatherParams(const WeatherParams& params) {
    data.weather = params;
    // Same stale-copy hazard as setNishitaParams: the wind is climate-owned.
    syncClimatePacket();
}

void World::syncClimatePacket() {
    data.nishita.derived_temperature_c = climate.surface_temperature_k - atmosphere::kKelvinOffset;
    data.nishita.derived_relative_humidity = climate.surface_relative_humidity;
    data.weather.derived_wind_direction = make_float3(climate.wind_direction.x,
                                                      climate.wind_direction.y,
                                                      climate.wind_direction.z);
    data.weather.derived_wind_speed_mps = climate.wind_speed_mps;
    // The physics side's copy (Faz 2). Here because every climate AND altitude
    // write already lands in this function -- one publish site, no lag.
    atmosphere::publishAmbient(climate, data.nishita.altitude);
    // Wind drives the cloud drift, and every stale-copy site that needs the
    // climate packet re-synced needs the cloud packet too. Every climate and
    // altitude write lands here, so the derived cloud layer follows them too.
    rederiveClouds();
    syncCloudPacket();
}

// Layer 0 from the climate (ATMOSPHERE_WEATHER.md §2). Other enabled layers
// that would overlap it are switched off (validateClouds forbids overlap).
// Returns true when anything changed.
bool World::rederiveClouds() {
    if (!clouds.derive_from_climate) return false;
    const atmosphere::DerivedWeather w = atmosphere::deriveWeather(climate, data.nishita.altitude);
    atmosphere::CloudState c = clouds;
    atmosphere::CloudLayer& l0 = c.layers[0];
    l0.enabled = w.cloud_coverage > 0.01f;
    l0.base_altitude_m = w.cloud_base_m;
    l0.thickness_m = w.cloud_thickness_m;
    l0.type = w.cloud_type;
    l0.coverage = w.cloud_coverage;
    // Dry sub-cloud air evaporates the shaft before the ground (virga).
    const float rh = climate.surface_relative_humidity;
    const float wet = std::min(1.0f, std::max(0.0f, (rh - 0.45f) / 0.35f));
    c.precipitation.rate_mm_h = w.precipitation_mm_h;
    c.precipitation.snow = w.snow;
    c.precipitation.ground_fraction = 0.3f + 0.7f * wet * wet * (3.0f - 2.0f * wet);
    for (int i = 1; i < atmosphere::kMaxCloudLayers; ++i) {
        atmosphere::CloudLayer& l = c.layers[i];
        if (l.enabled && l0.enabled &&
            l.base_altitude_m < l0.base_altitude_m + l0.thickness_m &&
            l0.base_altitude_m < l.base_altitude_m + l.thickness_m)
            l.enabled = false;
    }
    const atmosphere::CloudLayer& o = clouds.layers[0];
    bool changed = o.enabled != l0.enabled || o.base_altitude_m != l0.base_altitude_m ||
                   o.thickness_m != l0.thickness_m || o.type != l0.type || o.coverage != l0.coverage ||
                   clouds.precipitation.rate_mm_h != c.precipitation.rate_mm_h ||
                   clouds.precipitation.snow != c.precipitation.snow ||
                   clouds.precipitation.ground_fraction != c.precipitation.ground_fraction;
    for (int i = 1; i < atmosphere::kMaxCloudLayers; ++i)
        changed = changed || clouds.layers[i].enabled != c.layers[i].enabled;
    if (!changed) return false;
    clouds = c;
    ++cloud_revision;
    return true;
}

void World::syncCloudPacket() {
    const Vec3 drift = cloudWindOffset();
    atmosphere::cloudsToLegacyPacket(clouds, drift.x, drift.z, data.nishita);
}

Vec3 World::cloudWindOffset() const {
    const Vec3 wind = climate.wind_direction * climate.wind_speed_mps;
    return Vec3(wind.x * cloud_time_seconds, 0.0f, wind.z * cloud_time_seconds);
}

bool World::setClouds(const atmosphere::CloudState& c, std::string* error) {
    if (!atmosphere::validateClouds(c, error)) return false;
    // Unchanged -> no revision bump: a held cloud key re-applied every frame
    // must not reset accumulation every frame. Compared through the JSON
    // form (field-exact, no struct padding in the comparison).
    nlohmann::json before, after;
    atmosphere::cloudsToJson(clouds, before);
    atmosphere::cloudsToJson(c, after);
    if (before == after) return true;
    clouds = c;
    ++cloud_revision;
    rederiveClouds();   // a derived layer 0 stays derived whatever the edit
    syncCloudPacket();
    return true;
}

void World::setCloudTime(float seconds) {
    if (!std::isfinite(seconds) || seconds == cloud_time_seconds) return;
    cloud_time_seconds = seconds;
    ++cloud_revision;
    syncCloudPacket();
}

bool World::setClimate(const atmosphere::ClimateState& c, std::string* error) {
    if (!atmosphere::validateClimate(c, error)) return false;
    atmosphere::ClimateState next = c;
    atmosphere::normalizeWindDirection(next);

    // Both feed the sky LUT: temperature scales the scale heights, humidity
    // grows the aerosol (Mie) extinction.
    const bool lutRelevantChanged =
        next.surface_temperature_k != climate.surface_temperature_k ||
        next.surface_relative_humidity != climate.surface_relative_humidity;

    climate = next;
    syncClimatePacket();
    if (lutRelevantChanged) lut_dirty = true;
    return true;
}

atmosphere::ClimateSample World::sampleClimate(const Vec3& scene_pos) const {
    return atmosphere::sampleClimate(climate, scene_pos.y + data.nishita.altitude);
}

// Serialization
void World::serialize(nlohmann::json& j) const {
    j["mode"] = data.mode;
    
    // Color Mode
    j["color"] = { data.color.x, data.color.y, data.color.z };
    j["color_intensity"] = data.color_intensity;
    
    // HDRI
    // Persist HDRI path only when HDRI mode is active to avoid hidden stale paths
    // being carried in projects where HDRI is not currently used.
    if (data.mode == WORLD_MODE_HDRI) {
        j["hdri_path"] = hdri_path;
        j["env_rotation"] = getHDRIRotation();
        j["env_intensity"] = data.env_intensity;
    } else {
        j["hdri_path"] = "";
        j["env_rotation"] = 0.0f;
        j["env_intensity"] = 1.0f;
    }
    
    // Nishita
    nlohmann::json n;
    n["sun_elevation"] = data.nishita.sun_elevation;
    n["sun_azimuth"] = data.nishita.sun_azimuth;
    n["sun_intensity"] = data.nishita.sun_intensity;
    n["atmosphere_intensity"] = data.nishita.atmosphere_intensity;
    n["sun_size"] = data.nishita.sun_size;
    
    n["air_density"] = data.nishita.air_density;
    n["dust_density"] = data.nishita.dust_density;
    n["ozone_density"] = data.nishita.ozone_density; 
    n["altitude"] = data.nishita.altitude;
    
    n["fog_enabled"] = data.nishita.fog_enabled;
    n["fog_density"] = data.nishita.fog_density;
    n["fog_height"] = data.nishita.fog_height;
    n["fog_falloff"] = data.nishita.fog_falloff;
    n["fog_distance"] = data.nishita.fog_distance;
    n["fog_albedo"] = { data.nishita.fog_albedo.x, data.nishita.fog_albedo.y, data.nishita.fog_albedo.z };
    n["fog_anisotropy"] = data.nishita.fog_anisotropy;
    
    n["godrays_enabled"] = data.nishita.godrays_enabled;
    n["godrays_intensity"] = data.nishita.godrays_intensity;
    n["godrays_density"] = data.nishita.godrays_density;
    n["godrays_samples"] = data.nishita.godrays_samples;
    n["ozone_absorption_scale"] = data.nishita.ozone_absorption_scale;
    
    // Clouds: atmosphere.clouds (CloudState). The nishita.cloud_* fields are
    // a packet written from it and are not persisted.
    
    // Physical Constants
    n["planet_radius"] = data.nishita.planet_radius;
    n["atmosphere_height"] = data.nishita.atmosphere_height;
    n["mie_anisotropy"] = data.nishita.mie_anisotropy;
    n["rayleigh_density"] = data.nishita.rayleigh_density;
    n["mie_density"] = data.nishita.mie_density;

    j["nishita"] = n;

    // Advanced Atmosphere
    nlohmann::json adv;
    adv["multi_scatter_enabled"] = data.advanced.multi_scatter_enabled;
    adv["multi_scatter_factor"] = data.advanced.multi_scatter_factor;
    adv["aerial_perspective"] = data.advanced.aerial_perspective;
    
    adv["env_overlay_path"] = env_overlay_path;
    adv["env_overlay_enabled"] = data.advanced.env_overlay_enabled;
    adv["env_overlay_intensity"] = data.advanced.env_overlay_intensity;
    adv["env_overlay_rotation"] = data.advanced.env_overlay_rotation;
    adv["env_overlay_blend_mode"] = data.advanced.env_overlay_blend_mode;
    j["advanced"] = adv;

    nlohmann::json weather;
    weather["enabled"] = data.weather.enabled;
    weather["type"] = data.weather.type;
    weather["intensity"] = data.weather.intensity;
    weather["density"] = data.weather.density;
    weather["precipitation_scale"] = data.weather.precipitation_scale;
    weather["visibility"] = data.weather.visibility;
    weather["surface_wetness_output"] = data.weather.surface_wetness_output;
    weather["surface_accumulation_output"] = data.weather.surface_accumulation_output;
    weather["surface_settling_output"] = data.weather.surface_settling_output;
    weather["surface_height_output"] = data.weather.surface_height_output;
    weather["visual_mode"] = data.weather.visual_mode;
    weather["surface_response_enabled"] = data.weather.surface_response_enabled;
    j["weather"] = weather;

    // Climate authority. Replaces nishita.humidity/temperature and
    // weather.wind_*; those old keys are NOT read (units changed: C -> K).
    nlohmann::json atmo;
    atmosphere::climateToJson(climate, atmo["climate"]);
    atmosphere::cloudsToJson(clouds, atmo["clouds"]);
    j["atmosphere"] = atmo;
}

void World::deserialize(const nlohmann::json& j) {
    if (j.contains("mode")) data.mode = j["mode"];

    // Climate first: the Nishita block below bakes the LUT from the packet.
    // Projects saved before the climate existed get defaults; their old
    // nishita.humidity/temperature and weather.wind_* keys are deliberately
    // ignored (no backward-compat path, and the temperature unit changed).
    climate = atmosphere::ClimateState{};
    if (j.contains("atmosphere") && j["atmosphere"].contains("climate")) {
        std::string error;
        if (!atmosphere::climateFromJson(j["atmosphere"]["climate"], climate, &error)) {
            SCENE_LOG_WARN("[World] Invalid atmosphere.climate in project (" + error + ") - using defaults.");
        }
    }
    syncClimatePacket();
    
    if (j.contains("color")) {
        auto c = j["color"];
        data.color = make_float3(c[0], c[1], c[2]);
    }
    if (j.contains("color_intensity")) data.color_intensity = j["color_intensity"];
    
    if (j.contains("hdri_path")) {
        std::string path = j["hdri_path"];
        hdri_path = path;
        if (!path.empty() && data.mode == WORLD_MODE_HDRI) {
            setHDRI(path);
        }
    }
    if (j.contains("env_rotation")) setHDRIRotation(j["env_rotation"]);
    if (j.contains("env_intensity")) data.env_intensity = j["env_intensity"];
    
    if (j.contains("nishita")) {
        auto n = j["nishita"];
        data.nishita.sun_elevation = n.value("sun_elevation", 15.0f);
        data.nishita.sun_azimuth = n.value("sun_azimuth", 170.0f);
        data.nishita.sun_intensity = n.value("sun_intensity", 10.0f);
        data.nishita.atmosphere_intensity = n.value("atmosphere_intensity", data.nishita.sun_intensity);
        data.nishita.sun_size = n.value("sun_size", 0.545f);
        
        data.nishita.air_density = n.value("air_density", 1.0f);
        data.nishita.dust_density = n.value("dust_density", 1.0f);
        data.nishita.ozone_density = n.value("ozone_density", 1.0f);
        data.nishita.altitude = n.value("altitude", 0.0f);
        
        data.nishita.fog_enabled = n.value("fog_enabled", 0);
        data.nishita.fog_density = n.value("fog_density", 0.0001f);
        // Older projects serialized 0.1 as the disabled default. In metre-based
        // extinction that becomes practically opaque on first enable; migrate
        // only the untouched disabled default and preserve authored dense fog.
        if (!data.nishita.fog_enabled &&
            std::fabs(data.nishita.fog_density - 0.1f) < 1e-6f) {
            data.nishita.fog_density = 0.0001f;
        }
        data.nishita.fog_height = n.value("fog_height", 500.0f);
        data.nishita.fog_falloff = n.value("fog_falloff", 0.003f);
        data.nishita.fog_distance = n.value("fog_distance", 10000.0f);
        // fog_color/fog_sun_scatter (a flat colour mix) are NOT read: the fog
        // is now a lit medium and those numbers meant something else.
        if (n.contains("fog_albedo")) {
             auto fa = n["fog_albedo"];
             data.nishita.fog_albedo = make_float3(fa[0], fa[1], fa[2]);
        }
        data.nishita.fog_anisotropy = n.value("fog_anisotropy", 0.6f);
        
        data.nishita.godrays_enabled = n.value("godrays_enabled", 0);
        data.nishita.godrays_intensity = n.value("godrays_intensity", 0.5f);
        data.nishita.godrays_density = n.value("godrays_density", 0.1f);
        data.nishita.godrays_samples = n.value("godrays_samples", 16);

        data.nishita.ozone_absorption_scale = n.value("ozone_absorption_scale", 1.0f);
        // Multi-scatter and other advanced params are now handled in the 'advanced' block below
        
        // Old flat cloud_* keys are not read: syncCloudPacket overwrites the packet;
        // they are no longer the authority.

        data.nishita.cloud_coverage = (std::max)(0.0f, (std::min)(0.6f, data.nishita.cloud_coverage));
        data.nishita.cloud_density = (std::max)(0.0f, (std::min)(1.5f, data.nishita.cloud_density));
        data.nishita.cloud_scale = (std::max)(0.1f, (std::min)(1.0f, data.nishita.cloud_scale));
        data.nishita.cloud_detail = (std::max)(0.0f, (std::min)(1.0f, data.nishita.cloud_detail));
        data.nishita.cloud_base_steps = (std::max)(4, (std::min)(96, data.nishita.cloud_base_steps));
        data.nishita.cloud2_coverage = (std::max)(0.0f, (std::min)(0.6f, data.nishita.cloud2_coverage));
        data.nishita.cloud2_density = (std::max)(0.0f, (std::min)(1.5f, data.nishita.cloud2_density));
        data.nishita.cloud2_scale = (std::max)(0.1f, (std::min)(1.0f, data.nishita.cloud2_scale));
        
        // Physical Constants
        data.nishita.planet_radius = n.value("planet_radius", 6360000.0f);
        data.nishita.atmosphere_height = n.value("atmosphere_height", 60000.0f);
        data.nishita.mie_anisotropy = n.value("mie_anisotropy", 0.98f);
        data.nishita.rayleigh_density = n.value("rayleigh_density", 8000.0f);
        data.nishita.mie_density = n.value("mie_density", 1200.0f);

        // if (n.contains("env_overlay_path")) { // Moved to advanced
        //     std::string path = n["env_overlay_path"];
        //     if (!path.empty()) setNishitaEnvOverlay(path);
        // }
        // data.nishita.env_overlay_enabled = n.value("env_overlay_enabled", 0); // Moved to advanced
        // data.nishita.env_overlay_intensity = n.value("env_overlay_intensity", 1.0f); // Moved to advanced
        // data.nishita.env_overlay_rotation = n.value("env_overlay_rotation", 0.0f); // Moved to advanced
        // data.nishita.env_overlay_blend_mode = n.value("env_overlay_blend_mode", 0); // Moved to advanced
        
        // Multi-scattering - These are now handled by the 'advanced' block below
        // data.nishita.multi_scatter_enabled = n.value("multi_scatter_enabled", 1);
        // data.nishita.multi_scatter_factor = n.value("multi_scatter_factor", 0.3f);

        if (j.contains("advanced")) {
            auto a = j["advanced"];
            data.advanced.multi_scatter_enabled = a.value("multi_scatter_enabled", 1);
            data.advanced.multi_scatter_factor = a.value("multi_scatter_factor", 0.3f);
            data.advanced.aerial_perspective = a.value("aerial_perspective", 1);
            
            // Env Overlay
            data.advanced.env_overlay_enabled = a.value("env_overlay_enabled", 0);
            data.advanced.env_overlay_intensity = a.value("env_overlay_intensity", 1.0f);
            data.advanced.env_overlay_rotation = a.value("env_overlay_rotation", 0.0f);
            data.advanced.env_overlay_blend_mode = a.value("env_overlay_blend_mode", 0);
            
            if (a.contains("env_overlay_path")) {
                std::string path = a["env_overlay_path"];
                if (!path.empty()) setNishitaEnvOverlay(path);
            }
        }

        // Update Sun Direction
        float elevRad = data.nishita.sun_elevation * 3.14159265f / 180.0f;
        float azimRad = data.nishita.sun_azimuth * 3.14159265f / 180.0f;
        data.nishita.sun_direction = normalize(make_float3(
            cosf(elevRad) * sinf(azimRad), 
            sinf(elevRad), 
            cosf(elevRad) * cosf(azimRad)
        ));

        // CRITICAL: Re-initialize LUT after loading parameters — but ONLY when the loaded
        // world actually renders the atmosphere. Every project carries a "nishita" block
        // (defaults included), so the unconditional precompute made HDRI/color/interior
        // scenes pay a multi-second CPU sky bake at load for a sky they never sample.
        // For non-Nishita modes just mark the LUT dirty: the runtime already flushes it
        // lazily on demand (Main world-sync gate checks needsLUTUpdate() when the mode is
        // Nishita; the CPU ray path flushes in ray_color; flushLUT() self-initializes a
        // missing LUT).
        if (data.mode == WORLD_MODE_NISHITA) {
            if (atmosphere_lut) {
                atmosphere_lut->precompute(data.nishita);
            } else {
                initializeLUT();
            }
        } else {
            lut_dirty = true;
        }
    }

    if (j.contains("weather")) {
        auto w = j["weather"];
        data.weather.enabled = w.value("enabled", 0);
        data.weather.type = w.value("type", WEATHER_NONE);
        data.weather.intensity = w.value("intensity", 0.0f);
        data.weather.density = w.value("density", 0.0f);
        data.weather.precipitation_scale = w.value("precipitation_scale", 1.0f);
        data.weather.visibility = w.value("visibility", 1.0f);
        data.weather.surface_wetness_output = w.value("surface_wetness_output", 0.0f);
        data.weather.surface_accumulation_output = w.value("surface_accumulation_output", 0.0f);
        data.weather.surface_settling_output = w.value("surface_settling_output", 0.0f);
        data.weather.surface_height_output = w.value("surface_height_output", 0.0f);
        data.weather.visual_mode = w.value("visual_mode", WEATHER_VISUAL_OVERLAY);
        data.weather.surface_response_enabled = w.value("surface_response_enabled", 1);
    }
    // Clouds: the authority if saved, else defaults (layer 0 derived from the
    // climate; the old flat nishita.cloud_* keys are not read).
    clouds = atmosphere::CloudState{};
    if (j.contains("atmosphere") && j["atmosphere"].contains("clouds")) {
        std::string error;
        if (!atmosphere::cloudsFromJson(j["atmosphere"]["clouds"], clouds, &error)) {
            SCENE_LOG_WARN("[World] Invalid atmosphere.clouds in project (" + error + ") - clouds off.");
        }
    }
    ++cloud_revision;

    // Again, now that nishita.altitude (read above, after the first sync) and
    // the weather block are final: the published ambient snapshot must not
    // carry the pre-load altitude. Also writes the cloud packet.
    syncClimatePacket();
}
