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

// Helper Noise class for CPU
class CPUCloudNoise {
public:
    static float hash(float n) {
        return fmodf(sinf(n) * 43758.5453f, 1.0f);
    }
    
    // Fractional part (x - floor(x))
    static float frac(float x) {
        return x - floorf(x);
    }

    static float hash33(Vec3 p) {
        // Explicit dot product if global dot() is not available for Vec3
        float d = p.x * 12.9898f + p.y * 78.233f + p.z * 53.539f;
        return fmodf(sinf(d) * 43758.5453f, 1.0f);
    }

    static float noise3D(Vec3 x) {
        Vec3 p(floorf(x.x), floorf(x.y), floorf(x.z));
        Vec3 f(x.x - p.x, x.y - p.y, x.z - p.z);
        
        // Smoothstep: f * f * (3.0 - 2.0 * f)
        // Manual component-wise math to avoid operator ambiguities
        f.x = f.x * f.x * (3.0f - 2.0f * f.x);
        f.y = f.y * f.y * (3.0f - 2.0f * f.y);
        f.z = f.z * f.z * (3.0f - 2.0f * f.z);

        float n = p.x + p.y * 57.0f + p.z * 113.0f;

        return lerp(lerp(lerp(hash(n + 0.0f), hash(n + 1.0f), f.x),
                         lerp(hash(n + 57.0f), hash(n + 58.0f), f.x), f.y),
                    lerp(lerp(hash(n + 113.0f), hash(n + 114.0f), f.x),
                         lerp(hash(n + 170.0f), hash(n + 171.0f), f.x), f.y), f.z);
    }
    
    static float lerp(float a, float b, float t) {
        return a + t * (b - a);
    }

    static float fbm(Vec3 p, int octaves) {
        float f = 0.0f;
        float w = 0.5f;
        for (int i = 0; i < octaves; i++) {
            f += w * noise3D(p);
            p = p * 2.0f;
            w *= 0.5f;
        }
        return f;
    }
    
    // 3D Hash for Worley noise
    static Vec3 hash3(Vec3 p) {
        float px = p.x * 127.1f + p.y * 311.7f + p.z * 74.7f;
        float py = p.x * 269.5f + p.y * 183.3f + p.z * 246.1f;
        float pz = p.x * 113.5f + p.y * 271.9f + p.z * 124.6f;
        return Vec3(
            frac(sinf(px) * 43758.5453f),
            frac(sinf(py) * 43758.5453f),
            frac(sinf(pz) * 43758.5453f)
        );
    }
    
    // Worley (Cellular) Noise - creates puffy cloud structures
    static float worley(Vec3 p) {
        Vec3 i = Vec3(floorf(p.x), floorf(p.y), floorf(p.z));
        Vec3 f = Vec3(frac(p.x), frac(p.y), frac(p.z));
        
        float minDist = 1.0f;
        
        // Check 3x3x3 neighborhood
        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    Vec3 neighbor((float)x, (float)y, (float)z);
                    Vec3 point = hash3(i + neighbor);
                    Vec3 diff = neighbor + point - f;
                    float dist = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                    minDist = fminf(minDist, dist);
                }
            }
        }
        
        return sqrtf(minDist);
    }
    
    // Worley FBM
    static float worleyFbm(Vec3 p, int octaves) {
        float f = 0.0f;
        float w = 0.5f;
        for (int i = 0; i < octaves; i++) {
            f += w * worley(p);
            p = p * 2.0f;
            w *= 0.5f;
        }
        return f;
    }
    
    // ═══════════════════════════════════════════════════════════
    // CINEMATIC CLOUD SHAPE - Matches GPU Exactly (CloudNoise.cuh)
    // ═══════════════════════════════════════════════════════════
    static float cloud_shape(Vec3 p, float coverage) {
        // === LAYER 1: Base Shape ===
        float baseShape = fbm(p * 0.8f, 4);
        
        // === LAYER 2: Worley Cellular Structure ===
        float worleyBase = worley(p * 1.2f);
        float worleyDetail = worleyFbm(p * 2.5f, 3);
        float worlyClouds = 1.0f - worleyBase * 0.6f - worleyDetail * 0.3f;
        
        // === LAYER 3: Fine Detail Erosion ===
        float detailNoise = fbm(p * 4.0f, 4);
        float microDetail = fbm(p * 12.0f, 2) * 0.15f; // Matched to 0.15
        
        // === COMBINE LAYERS ===
        float combined = baseShape * worlyClouds;
        combined = combined - detailNoise * 0.15f - microDetail; // Matched subtraction
        combined = fmaxf(0.0f, combined);
        
        // === COVERAGE REMAP ===
        float threshold = (1.0f - coverage) * 0.55f;
        float density = fmaxf(0.0f, combined - threshold);
        
        // === SOFT EDGE FALLOFF ===
        float edge = fminf(1.0f, density * 4.0f);
        edge = edge * edge;
        density *= edge;
        
        // === DENSITY BOOST ===
        density *= 5.0f; // Matched to 5.0 for volume appearance
        
        return density;
    }
    
    // Powder effect for silver lining
    static float powderEffect(float density, float cosTheta) {
        float beer = expf(-density * 2.0f);
        float powder = 1.0f - expf(-density * 4.0f);
        float sunFactor = (1.0f + cosTheta) * 0.5f;
        return beer + (powder * beer - beer) * sunFactor * 0.5f;
    }
};

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
    data.nishita.sun_size = 0.545f;        // Real sun angular size in degrees
    
    // Blender-style atmosphere multipliers (default 1.0)
    data.nishita.air_density = 1.0f;       // Air (Rayleigh) multiplier
    data.nishita.dust_density = 1.0f;      // Dust (Mie/aerosols) multiplier
    data.nishita.ozone_density = 1.0f;     // Ozone multiplier
    data.nishita.altitude = 0.0f;          // Sea level (meters)
    
    
    // Cloud Defaults - Lower heights for easier access
    data.nishita.clouds_enabled = 0;       // Disabled by default
    data.nishita.cloud_coverage = 0.5f;
    data.nishita.cloud_density = 1.0f;
    data.nishita.cloud_scale = 1.0f;
    data.nishita.cloud_height_min = 500.0f;  // 500m - more accessible
    data.nishita.cloud_height_max = 2000.0f; // 2km - gives 1.5km layer thickness
    data.nishita.cloud_offset_x = 0.0f;
    data.nishita.cloud_offset_z = 0.0f;
    data.nishita.cloud_quality = 1.0f;     // Normal quality (1.0 = 64-128 steps)
    data.nishita.cloud_detail = 1.0f;      // Normal detail level
    data.nishita.cloud_base_steps = 48;    // Default base resolution
    
    // Cloud Layer 2 defaults (High altitude cirrus-like)
    data.nishita.cloud_layer2_enabled = 0;   // Disabled by default
    data.nishita.cloud2_coverage = 0.3f;
    data.nishita.cloud2_density = 0.3f;      // Thin/transparent
    data.nishita.cloud2_scale = 8.0f;        // Large/wispy
    data.nishita.cloud2_height_min = 6000.0f; // 6km
    data.nishita.cloud2_height_max = 7000.0f; // 7km (thin layer)
    
    // Cloud Lighting defaults
    data.nishita.cloud_light_steps = 0;      // Light marching steps (0 = disabled)
    data.nishita.cloud_shadow_strength = 1.0f;
    data.nishita.cloud_ambient_strength = 1.0f;
    data.nishita.cloud_silver_intensity = 1.0f;
    data.nishita.cloud_absorption = 1.0f;
    
    // Cloud Advanced Scattering (VDB-like) defaults
    data.nishita.cloud_anisotropy = 0.85f;       // Modern clouds forward scatter strongly
    data.nishita.cloud_anisotropy_back = -0.3f;   // Standard back lobe
    data.nishita.cloud_lobe_mix = 0.5f;          // Even mix
    
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
    data.nishita.fog_density = 0.1f;            // Light fog
    data.nishita.fog_height = 500.0f;            // Fog concentrated below 500m
    data.nishita.fog_falloff = 0.003f;           // Gradual falloff
    data.nishita.fog_distance = 10000.0f;        // 10km max distance
    data.nishita.fog_color = make_float3(0.7f, 0.8f, 0.9f);  // Bluish-white
    data.nishita.fog_sun_scatter = 0.5f;         // Medium sun scattering
    
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
    data.advanced.aerial_min_distance = 10.0f;    // Minimal haze starts almost immediately (10m)
    data.advanced.aerial_max_distance = 5000.0f;   // Full haze at 5km
    data.advanced.env_overlay_enabled = 0;
    data.advanced.env_overlay_intensity = 1.0f;
    data.advanced.env_overlay_rotation = 0.0f;
    data.advanced.env_overlay_blend_mode = 0;

    // Physical Defaults
    data.nishita.humidity = 0.1f;
    data.nishita.temperature = 15.0f;
    data.nishita.ozone_absorption_scale = 1.0f;
    data.nishita.godrays_samples = 16;           // Balanced quality
    data.nishita.godrays_samples = 16;           // Balanced quality
    // Advanced Atmosphere Defaults (Already set above in data.advanced)
    // Removed redundant nishita access
    
    // Env Overlay (Now handled in advanced)
    
    // Future volume params
    data.volume_density = 0.0f;
    data.volume_anisotropy = 0.0f;

    // Initialize LUT handles to zero
    data.lut.transmittance_lut = 0;
    data.lut.skyview_lut = 0;
    data.lut.integrated_multi_scattering = make_float3(0, 0, 0);
}

void World::initializeLUT() {
    // GPU SAFETY CHECK: Skip LUT initialization if no CUDA device available
    // Transmittance/SkyView LUTs require CUDA for precomputation, but not necessarily OptiX.
    if (!g_hasCUDA) {
        SCENE_LOG_WARN("No CUDA device available - skipping LUT initialization (CPU fallback mode)");
        // Set LUT handles to 0 so GPU code knows to use fallback
        data.lut.transmittance_lut = 0;
        data.lut.skyview_lut = 0;
        data.lut.multi_scattering_lut = 0;
        data.lut.aerial_perspective_lut = 0;
        return;
    }
    
    if (!atmosphere_lut) {
        atmosphere_lut = new AtmosphereLUT();
        atmosphere_lut->initialize();
        atmosphere_lut->precompute(data.nishita);
        data.lut = atmosphere_lut->getGPUData();
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
}

WorldData World::getGPUData() const {
    WorldData gpuData = data;
    if (atmosphere_lut) {
        gpuData.lut = atmosphere_lut->getGPUData();
    }
    return gpuData;
}

void World::setMode(WorldMode mode) {
    data.mode = mode;
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
    
    data.nishita = params;
    
    // Restore the texture handle
    if (savedTex != 0) {
        data.advanced.env_overlay_tex = savedTex;
    }

    // DEFERRED: Mark LUT as needing update instead of computing immediately.
    // This prevents 50K-pixel ray march from running on every UI slider tick.
    // Main loop calls flushLUT() once per frame.
    if (atmosphere_lut) {
        lut_dirty = true;
    }
}

void World::flushLUT() {
    if (!lut_dirty || !atmosphere_lut) return;
    atmosphere_lut->precompute(data.nishita);
    lut_dirty = false;
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
        
        bool uploaded = hdri_texture->upload_to_gpu();
        if (uploaded) {
            data.env_texture = hdri_texture->get_cuda_texture();
            data.env_width = hdri_texture->width;
            data.env_height = hdri_texture->height;
            SCENE_LOG_INFO("HDRI uploaded to GPU: " + std::to_string(data.env_width) + "x" + std::to_string(data.env_height));
        } else {
             SCENE_LOG_ERROR("Failed to upload HDRI to GPU");
             data.env_texture = 0;
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
        
        bool uploaded = env_overlay_texture->upload_to_gpu();
        if (uploaded) {
            data.advanced.env_overlay_tex = env_overlay_texture->get_cuda_texture();
            data.advanced.env_overlay_enabled = 1;
            SCENE_LOG_INFO("Env Overlay uploaded: " + std::to_string(env_overlay_texture->width) + "x" + std::to_string(env_overlay_texture->height));
        } else {
            SCENE_LOG_ERROR("Failed to upload Env Overlay to GPU");
            data.advanced.env_overlay_tex = 0;
            data.advanced.env_overlay_enabled = 0;
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
    if (atmosphere_lut) {
        lut_dirty = true;
    }
}

void World::setSunIntensity(float intensity) {
    data.nishita.sun_intensity = intensity;
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

// Simplified Nishita for CPU Preview (Optional, for now returns simple gradient or black if unimplemented)
// Implementing full Nishita on CPU might be slow for real-time without optimization.
Vec3 World::evaluate(const Vec3& ray_dir, const Vec3& origin) {
    if (data.mode == WORLD_MODE_COLOR) {
        return to_vec3(data.color) * data.color_intensity;
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
            return hdri_texture->get_color(u, 1.0f - v) * data.env_intensity;
        }
        return to_vec3(data.color); // Fallback
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
        L += to_vec3(transSun) * (mieScat * excessPhase * data.nishita.sun_intensity);
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

    // ═══════════════════════════════════════════════════════════
    // VOLUMETRIC CLOUDS (CPU Optimized Path)
    // ═══════════════════════════════════════════════════════════
    if ((data.nishita.clouds_enabled || data.nishita.cloud_layer2_enabled) && dir.y > 0.001f) {
        float transmittance = 1.0f;
        Vec3 accumulatedCloudColor(0.0f);
        float camY = std::max(0.0f, data.camera_y != 0.0f ? data.camera_y : data.nishita.altitude);
        Vec3 camPos(0.0f, camY, 0.0f);
        Vec3 rayDir = to_vec3(dir);

        for (int layer = 0; layer < 2; ++layer) {
            bool enabled = (layer == 0) ? data.nishita.clouds_enabled : data.nishita.cloud_layer2_enabled;
            if (!enabled) continue;

            float minH = (layer == 0) ? data.nishita.cloud_height_min : data.nishita.cloud2_height_min;
            float maxH = (layer == 0) ? data.nishita.cloud_height_max : data.nishita.cloud2_height_max;
            float coverage = (layer == 0) ? data.nishita.cloud_coverage : data.nishita.cloud2_coverage;
            float densityMult = (layer == 0) ? data.nishita.cloud_density : data.nishita.cloud2_density;
            float scale = 0.003f / std::max(0.01f, (layer == 0) ? data.nishita.cloud_scale : data.nishita.cloud2_scale);
            float t_enter, t_exit;
            if (camPos.y < minH) {
                if (rayDir.y <= 0.0f) continue;
                t_enter = (minH - camPos.y) / rayDir.y;
                t_exit = (maxH - camPos.y) / rayDir.y;
            }
            else if (camPos.y > maxH) {
                if (rayDir.y >= 0.0f) continue;
                t_enter = (maxH - camPos.y) / rayDir.y;
                t_exit = (minH - camPos.y) / rayDir.y;
            }
            else {
                t_enter = 0.0f;
                t_exit = (rayDir.y > 0.0f) ? (maxH - camPos.y) / rayDir.y : (minH - camPos.y) / rayDir.y;
            }

            if (t_exit > 0.0f && t_exit > t_enter) {
                t_enter = std::max(t_enter, 0.0f);

                float h_t = std::max(0.0f, std::min(1.0f, std::abs(rayDir.y) / 0.008f));
                float horizonFade = h_t * h_t * (3.0f - 2.0f * h_t);

                float quality = std::max(0.1f, std::min(3.0f, data.nishita.cloud_quality));
                int numSteps = (int)(32.0f * quality);
                float stepSize = (t_exit - t_enter) / (float)numSteps;

                float layerTransmittance = 1.0f;
                Vec3 layerColor(0.0f);
                float sunMu = dot(sunDir, dir);

                for (int i = 0; i < numSteps; ++i) {
                    float t_sample = t_enter + stepSize * (i + 0.5f);
                    Vec3 p = camPos + rayDir * t_sample;

                    float h_frac = (p.y - minH) / (maxH - minH);
                    float h_grad = std::max(0.01f, std::min(1.0f, h_frac / 0.05f)) * std::max(0.01f, std::min(1.0f, (1.0f - h_frac) / 0.3f));

                    Vec3 noisePos = (p + Vec3(data.nishita.cloud_offset_x, 0, data.nishita.cloud_offset_z)) * scale;
                    float density = CPUCloudNoise::cloud_shape(noisePos, coverage) * h_grad;

                    if (density > 0.003f) {
                        density *= densityMult * horizonFade;
                        float sigma_t = density * 1.5f;
                        float stepTrans = expf(-sigma_t * stepSize * 0.02f);

                        // Improved scattering approximation
                        float phase = (1.0f - g_mie * g_mie) / (4.0f * 3.14159f * powf(1.0f + g_mie * g_mie - 2.0f * g_mie * sunMu, 1.5f));
                        Vec3 sunColor = (sunDir.y > 0.15f) ? Vec3(1.0f, 0.95f, 0.85f) : Vec3(1.0f, 0.6f, 0.3f);
                        Vec3 inScat = sunColor * phase * data.nishita.sun_intensity * 0.2f;

                        layerColor = layerColor + (inScat * (1.0f - stepTrans) * layerTransmittance);
                        layerTransmittance *= stepTrans;
                    }
                    if (layerTransmittance < 0.01f) break;
                }
                accumulatedCloudColor += layerColor * transmittance;
                transmittance *= layerTransmittance;
            }
            L = L * transmittance + accumulatedCloudColor;
        }
    }
    return L;
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
    
    defaults.humidity = 0.1f;
    defaults.temperature = 15.0f;
    defaults.ozone_absorption_scale = 1.0f;
    
    data.nishita = defaults;
    
    // Re-initialize LUT with defaults
    if (atmosphere_lut) {
        atmosphere_lut->precompute(data.nishita);
    } else {
        initializeLUT();
    }
}

AtmosphereAdvanced World::getAdvancedParams() const {
    return data.advanced;
}

void World::setAdvancedParams(const AtmosphereAdvanced& a) {
    data.advanced = a;
}

// Serialization
void World::serialize(nlohmann::json& j) const {
    j["mode"] = data.mode;
    
    // Color Mode
    j["color"] = { data.color.x, data.color.y, data.color.z };
    j["color_intensity"] = data.color_intensity;
    
    // HDRI
    j["hdri_path"] = hdri_path;
    j["env_rotation"] = getHDRIRotation();
    j["env_intensity"] = data.env_intensity;
    
    // Nishita
    nlohmann::json n;
    n["sun_elevation"] = data.nishita.sun_elevation;
    n["sun_azimuth"] = data.nishita.sun_azimuth;
    n["sun_intensity"] = data.nishita.sun_intensity;
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
    n["fog_color"] = { data.nishita.fog_color.x, data.nishita.fog_color.y, data.nishita.fog_color.z };
    n["fog_sun_scatter"] = data.nishita.fog_sun_scatter;
    
    n["godrays_enabled"] = data.nishita.godrays_enabled;
    n["godrays_intensity"] = data.nishita.godrays_intensity;
    n["godrays_density"] = data.nishita.godrays_density;
    n["godrays_samples"] = data.nishita.godrays_samples;
    n["godrays_samples"] = data.nishita.godrays_samples;
    n["humidity"] = data.nishita.humidity;
    n["temperature"] = data.nishita.temperature;
    n["ozone_absorption_scale"] = data.nishita.ozone_absorption_scale;
    
    // Cloud Layer 1
    n["clouds_enabled"] = data.nishita.clouds_enabled;
    n["cloud_coverage"] = data.nishita.cloud_coverage;
    n["cloud_density"] = data.nishita.cloud_density;
    n["cloud_scale"] = data.nishita.cloud_scale;
    n["cloud_height_min"] = data.nishita.cloud_height_min;
    n["cloud_height_max"] = data.nishita.cloud_height_max;
    n["cloud_offset_x"] = data.nishita.cloud_offset_x;
    n["cloud_offset_z"] = data.nishita.cloud_offset_z;
    n["cloud_quality"] = data.nishita.cloud_quality;
    n["cloud_detail"] = data.nishita.cloud_detail;
    n["cloud_base_steps"] = data.nishita.cloud_base_steps;
    
    // Cloud Layer 2
    n["cloud_layer2_enabled"] = data.nishita.cloud_layer2_enabled;
    n["cloud2_coverage"] = data.nishita.cloud2_coverage;
    n["cloud2_density"] = data.nishita.cloud2_density;
    n["cloud2_scale"] = data.nishita.cloud2_scale;
    n["cloud2_height_min"] = data.nishita.cloud2_height_min;
    n["cloud2_height_max"] = data.nishita.cloud2_height_max;
    
    // Cloud Lighting
    n["cloud_light_steps"] = data.nishita.cloud_light_steps;
    n["cloud_shadow_strength"] = data.nishita.cloud_shadow_strength;
    n["cloud_ambient_strength"] = data.nishita.cloud_ambient_strength;
    n["cloud_silver_intensity"] = data.nishita.cloud_silver_intensity;
    n["cloud_absorption"] = data.nishita.cloud_absorption;
    
    // Cloud Advanced Scattering
    n["cloud_anisotropy"] = data.nishita.cloud_anisotropy;
    n["cloud_anisotropy_back"] = data.nishita.cloud_anisotropy_back;
    n["cloud_lobe_mix"] = data.nishita.cloud_lobe_mix;
    
    // Cloud Emissive
    n["cloud_emissive_intensity"] = data.nishita.cloud_emissive_intensity;
    n["cloud_emissive_color"] = { data.nishita.cloud_emissive_color.x, data.nishita.cloud_emissive_color.y, data.nishita.cloud_emissive_color.z };
    
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
    adv["aerial_min_distance"] = data.advanced.aerial_min_distance;
    adv["aerial_max_distance"] = data.advanced.aerial_max_distance;
    
    adv["env_overlay_path"] = env_overlay_path;
    adv["env_overlay_enabled"] = data.advanced.env_overlay_enabled;
    adv["env_overlay_intensity"] = data.advanced.env_overlay_intensity;
    adv["env_overlay_rotation"] = data.advanced.env_overlay_rotation;
    adv["env_overlay_blend_mode"] = data.advanced.env_overlay_blend_mode;
    j["advanced"] = adv;
}

void World::deserialize(const nlohmann::json& j) {
    if (j.contains("mode")) data.mode = j["mode"];
    
    if (j.contains("color")) {
        auto c = j["color"];
        data.color = make_float3(c[0], c[1], c[2]);
    }
    if (j.contains("color_intensity")) data.color_intensity = j["color_intensity"];
    
    if (j.contains("hdri_path")) {
        std::string path = j["hdri_path"];
        if (!path.empty()) setHDRI(path);
    }
    if (j.contains("env_rotation")) setHDRIRotation(j["env_rotation"]);
    if (j.contains("env_intensity")) data.env_intensity = j["env_intensity"];
    
    if (j.contains("nishita")) {
        auto n = j["nishita"];
        data.nishita.sun_elevation = n.value("sun_elevation", 15.0f);
        data.nishita.sun_azimuth = n.value("sun_azimuth", 170.0f);
        data.nishita.sun_intensity = n.value("sun_intensity", 10.0f);
        data.nishita.sun_size = n.value("sun_size", 0.545f);
        
        data.nishita.air_density = n.value("air_density", 1.0f);
        data.nishita.dust_density = n.value("dust_density", 1.0f);
        data.nishita.ozone_density = n.value("ozone_density", 1.0f);
        data.nishita.altitude = n.value("altitude", 0.0f);
        
        data.nishita.fog_enabled = n.value("fog_enabled", 0);
        data.nishita.fog_density = n.value("fog_density", 0.01f);
        data.nishita.fog_height = n.value("fog_height", 500.0f);
        data.nishita.fog_falloff = n.value("fog_falloff", 0.003f);
        data.nishita.fog_distance = n.value("fog_distance", 10000.0f);
        if (n.contains("fog_color")) {
             auto fc = n["fog_color"];
             data.nishita.fog_color = make_float3(fc[0], fc[1], fc[2]);
        }
        data.nishita.fog_sun_scatter = n.value("fog_sun_scatter", 0.5f);
        
        data.nishita.godrays_enabled = n.value("godrays_enabled", 0);
        data.nishita.godrays_intensity = n.value("godrays_intensity", 0.5f);
        data.nishita.godrays_density = n.value("godrays_density", 0.1f);
        data.nishita.godrays_samples = n.value("godrays_samples", 16);
        data.nishita.godrays_samples = n.value("godrays_samples", 16);
        
        // Physical params (Now back in nishita struct)
        data.nishita.humidity = n.value("humidity", 0.1f);
        data.nishita.temperature = n.value("temperature", 15.0f);
        data.nishita.ozone_absorption_scale = n.value("ozone_absorption_scale", 1.0f);
        // Multi-scatter and other advanced params are now handled in the 'advanced' block below
        
        // Cloud Layer 1
        data.nishita.clouds_enabled = n.value("clouds_enabled", 0);
        data.nishita.cloud_coverage = n.value("cloud_coverage", 0.5f);
        data.nishita.cloud_density = n.value("cloud_density", 1.0f);
        data.nishita.cloud_scale = n.value("cloud_scale", 1.0f);
        data.nishita.cloud_height_min = n.value("cloud_height_min", 500.0f);
        data.nishita.cloud_height_max = n.value("cloud_height_max", 2000.0f);
        data.nishita.cloud_offset_x = n.value("cloud_offset_x", 0.0f);
        data.nishita.cloud_offset_z = n.value("cloud_offset_z", 0.0f);
        data.nishita.cloud_quality = n.value("cloud_quality", 1.0f);
        data.nishita.cloud_detail = n.value("cloud_detail", 1.0f);
        data.nishita.cloud_base_steps = n.value("cloud_base_steps", 48);
        
        // Cloud Layer 2
        data.nishita.cloud_layer2_enabled = n.value("cloud_layer2_enabled", 0);
        data.nishita.cloud2_coverage = n.value("cloud2_coverage", 0.3f);
        data.nishita.cloud2_density = n.value("cloud2_density", 0.3f);
        data.nishita.cloud2_scale = n.value("cloud2_scale", 8.0f);
        data.nishita.cloud2_height_min = n.value("cloud2_height_min", 6000.0f);
        data.nishita.cloud2_height_max = n.value("cloud2_height_max", 7000.0f);
        
        // Cloud Lighting
        data.nishita.cloud_light_steps = n.value("cloud_light_steps", 0);
        data.nishita.cloud_shadow_strength = n.value("cloud_shadow_strength", 1.0f);
        data.nishita.cloud_ambient_strength = n.value("cloud_ambient_strength", 1.0f);
        data.nishita.cloud_silver_intensity = n.value("cloud_silver_intensity", 1.0f);
        data.nishita.cloud_absorption = n.value("cloud_absorption", 1.0f);
        
        // Cloud Advanced Scattering
        data.nishita.cloud_anisotropy = n.value("cloud_anisotropy", 0.85f);
        data.nishita.cloud_anisotropy_back = n.value("cloud_anisotropy_back", -0.3f);
        data.nishita.cloud_lobe_mix = n.value("cloud_lobe_mix", 0.5f);
        
        // Cloud Emissive
        data.nishita.cloud_emissive_intensity = n.value("cloud_emissive_intensity", 0.0f);
        if (n.contains("cloud_emissive_color")) {
            auto ec = n["cloud_emissive_color"];
            data.nishita.cloud_emissive_color = make_float3(ec[0], ec[1], ec[2]);
        }
        
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
        
        // Advanced Physics - These are now handled by the 'advanced' block below
        // data.nishita.humidity = n.value("humidity", 0.1f);
        // data.nishita.temperature = n.value("temperature", 15.0f);
        // data.nishita.ozone_absorption_scale = n.value("ozone_absorption_scale", 1.0f);
        
        // Multi-scattering - These are now handled by the 'advanced' block below
        // data.nishita.multi_scatter_enabled = n.value("multi_scatter_enabled", 1);
        // data.nishita.multi_scatter_factor = n.value("multi_scatter_factor", 0.3f);

        if (j.contains("advanced")) {
            auto a = j["advanced"];
            data.advanced.multi_scatter_enabled = a.value("multi_scatter_enabled", 1);
            data.advanced.multi_scatter_factor = a.value("multi_scatter_factor", 0.3f);
            data.advanced.aerial_perspective = a.value("aerial_perspective", 1);
            data.advanced.aerial_min_distance = a.value("aerial_min_distance", 10.0f);
            data.advanced.aerial_max_distance = a.value("aerial_max_distance", 5000.0f);
            
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

        // CRITICAL: Re-initialize LUT after loading parameters
        if (atmosphere_lut) {
            atmosphere_lut->precompute(data.nishita); 
        } else {
            initializeLUT();
        }
    }
}
