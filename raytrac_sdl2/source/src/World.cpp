#include "World.h"
#include "Texture.h"
#include "globals.h"
#include "vec3_utils.cuh" // For math helpers if needed, or just standard math
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
    // CINEMATIC CLOUD SHAPE - Matches GPU (CloudNoise.cuh)
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
        float microDetail = fbm(p * 12.0f, 2) * 0.1f;
        
        // === COMBINE LAYERS ===
        float combined = baseShape * worlyClouds;
        combined = combined - detailNoise * 0.25f - microDetail;
        combined = fmaxf(0.0f, combined);
        
        // === COVERAGE REMAP ===
        float threshold = (1.0f - coverage) * 0.55f;
        float density = fmaxf(0.0f, combined - threshold);
        
        // === SOFT EDGE FALLOFF ===
        float edge = fminf(1.0f, density * 4.0f);
        edge = edge * edge;
        density *= edge;
        
        // === DENSITY BOOST ===
        density *= 1.5f;
        
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
    data.mode = WORLD_MODE_COLOR;
    data.color = make_float3(0.05f, 0.05f, 0.05f); // Dark grey default
    data.color_intensity = 1.0f;
    data.env_texture = 0;
    data.env_rotation = 0.0f;
    data.env_intensity = 1.0f;
    data.env_width = 0;
    data.env_height = 0;

    // Nishita Defaults (Blender-compatible Earth-like atmosphere)
    data.nishita.sun_elevation = 45.0f;
    data.nishita.sun_azimuth = 180.0f;
    data.nishita.sun_direction = normalize(make_float3(0.0f, 0.707f, -0.707f));
    data.nishita.sun_intensity = 50.0f;
    data.nishita.sun_size = 1.0f;        // Real sun angular size in degrees
    
    // Blender-style atmosphere multipliers (default 1.0)
    data.nishita.air_density = 1.0f;       // Air (Rayleigh) multiplier
    data.nishita.dust_density = 1.0f;      // Dust (Mie/aerosols) multiplier
    data.nishita.ozone_density = 1.0f;     // Ozone multiplier
    data.nishita.altitude = 0.0f;          // Sea level (meters)
    
    // Night sky defaults
    data.nishita.stars_intensity = 1.0f;   // Star brightness
    data.nishita.stars_density = 0.5f;     // Medium star density
    
    // Moon defaults
    data.nishita.moon_enabled = 1;         // Moon visible
    data.nishita.moon_elevation = 45.0f;   // 45 degrees up
    data.nishita.moon_azimuth = 0.0f;      // Opposite to sun
    data.nishita.moon_intensity = 0.5f;    // Dimmer than sun
    data.nishita.moon_size = 0.52f;        // Real moon angular size in degrees
    data.nishita.moon_phase = 0.5f;        // Full moon
    
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
    
    // Cloud Layer 2 defaults (High altitude cirrus-like)
    data.nishita.cloud_layer2_enabled = 0;   // Disabled by default
    data.nishita.cloud2_coverage = 0.3f;
    data.nishita.cloud2_density = 0.3f;      // Thin/transparent
    data.nishita.cloud2_scale = 8.0f;        // Large/wispy
    data.nishita.cloud2_height_min = 6000.0f; // 6km
    data.nishita.cloud2_height_max = 7000.0f; // 7km (thin layer)
    
    // Cloud Lighting defaults
    data.nishita.cloud_light_steps = 6;      // Light marching steps (0 = disabled)
    data.nishita.cloud_shadow_strength = 1.0f;
    data.nishita.cloud_ambient_strength = 1.0f;
    data.nishita.cloud_silver_intensity = 1.0f;
    data.nishita.cloud_absorption = 1.0f;
    
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
    
    // Future volume params
    data.volume_density = 0.0f;
    data.volume_anisotropy = 0.0f;
}

World::~World() {
    if (hdri_texture) {
        delete hdri_texture;
        hdri_texture = nullptr;
    }
}

WorldData World::getGPUData() const {
    return data;
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
    data.nishita = params;
    // Potentially re-upload if needed, primarily handled at getGPUData time
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

void World::setSunDirection(const Vec3& direction) {
    data.nishita.sun_direction = normalize(to_float3(direction));
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
    // Mapping density to mie scattering multiplier or similar if desired, 
    // or just direct scale height adjustment?
    // Blender's Dust Density usually scales the mie density.
    // For now we assume user sets scattering coefficients directly or we could scale them.
    // Let's keep it simple.
}

NishitaSkyParams World::getNishitaParams() const {
    return data.nishita;
}

// Simplified Nishita for CPU Preview (Optional, for now returns simple gradient or black if unimplemented)
// Implementing full Nishita on CPU might be slow for real-time without optimization.
Vec3 World::evaluate(const Vec3& ray_dir) {
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
        return calculateNishitaSky(ray_dir);
    }
    return Vec3(0);
}

// Basic Single Scattering Nishita Implementation
// Adapting from common shader implementations
Vec3 World::calculateNishitaSky(const Vec3& ray_dir) {
    // Setup vectors
    float3 dir = normalize(to_float3(ray_dir));
    float3 sunDir = normalize(data.nishita.sun_direction);
    
    // Planet/atmosphere dimensions (all in METERS)
    float Rg = data.nishita.planet_radius;                        // Ground radius (meters)
    float Rt = Rg + data.nishita.atmosphere_height;               // Top of atmosphere (meters)
    
    // Camera position with altitude (altitude now in meters too)
    float cameraAltitude = data.nishita.altitude;                 // Altitude in meters
    float3 camPos = make_float3(0, Rg + cameraAltitude, 0);
    
    // ... Ray Sphere Intersect ...
    // Calculate distance to atmosphere top
    // Ray: P = camPos + t * dir
    // Sphere: |P|^2 = Rt^2
    
    float3 p = camPos;
    // float b = dot(dir, p); // 2 * dot but simplified
    // float c = dot(p, p) - Rt*Rt;
    // float delta = b*b - c; 
    
    // Standard analytic ray-sphere intersection
    float a = dot(dir, dir);
    float b = 2.0f * dot(dir, p);
    float c = dot(p, p) - Rt * Rt;
    float delta = b * b - 4.0f * a * c;
    
    if (delta < 0.0f) return Vec3(0);
    
    float t1 = (-b - sqrt(delta)) / (2.0f * a);
    float t2 = (-b + sqrt(delta)) / (2.0f * a);
    float t = (t1 >= 0.0f) ? t1 : t2;
    if (t < 0.0f) return Vec3(0);
    
    int numSamples = 8; // Low samples for CPU preview
    float stepSize = t / (float)numSamples;
    
    float3 totalRayleigh = make_float3(0, 0, 0);
    float3 totalMie = make_float3(0, 0, 0);
    
    float opticalDepthRayleigh = 0.0f;
    float opticalDepthMie = 0.0f;
    
    // Phase functions
    float mu = dot(dir, sunDir);
    float phaseR = 3.0f / (16.0f * M_PI) * (1.0f + mu * mu);
    float g = data.nishita.mie_anisotropy;
    float phaseM = 3.0f / (8.0f * M_PI) * ((1.0f - g * g) * (1.0f + mu * mu)) / ((2.0f + g * g) * pow(1.0f + g * g - 2.0f * g * mu, 1.5f));
    
    float currentT = 0.0f;
    
    for (int i = 0; i < numSamples; ++i) {
        float3 samplePos = p + dir * (currentT + stepSize * 0.5f);
        float height = length(samplePos) - Rg;
        
        if (height < 0) height = 0; // clamp
        
        float hr = exp(-height / data.nishita.rayleigh_density);
        float hm = exp(-height / data.nishita.mie_density);
        
        opticalDepthRayleigh += hr * stepSize;
        opticalDepthMie += hm * stepSize;
        
        // Light path to sun (transmittance)
        // Ray intersect atmosphere from samplePos to sunDir
        float b_light = 2.0f * dot(sunDir, samplePos);
        float c_light = dot(samplePos, samplePos) - Rt * Rt;
        float delta_light = b_light * b_light - 4.0f * c_light;
        
        if (delta_light >= 0.0f) {
             float t_light = (-b_light + sqrt(delta_light)) / 2.0f;
             // Cheap approximation for optical depth to sun
             // Usually requires another loop (nested integration)
             // For simplify, we assume uniform or precomputed.
             // But for Nishita we usually need the loop.
             // We can use Chapman function or just 4 samples.
             
             int numLightSamples = 4;
             float lightStep = t_light / (float)numLightSamples;
             float lightOpticalRayleigh = 0.0f;
             float lightOpticalMie = 0.0f;
             
             for(int j=0; j<numLightSamples; ++j) {
                 float3 lightSamplePos = samplePos + sunDir * (lightStep * (j + 0.5f));
                 float lightHeight = length(lightSamplePos) - Rg;
                 if(lightHeight < 0) lightHeight = 0;
                 lightOpticalRayleigh += exp(-lightHeight / data.nishita.rayleigh_density) * lightStep;
                 lightOpticalMie += exp(-lightHeight / data.nishita.mie_density) * lightStep;
             }
             
             float3 tau = data.nishita.rayleigh_scattering * (opticalDepthRayleigh + lightOpticalRayleigh) + 
                          data.nishita.mie_scattering * 1.1f * (opticalDepthMie + lightOpticalMie);
                          
             float3 attenuation = make_float3(exp(-tau.x), exp(-tau.y), exp(-tau.z));
             
             totalRayleigh += attenuation * hr * stepSize;
             totalMie += attenuation * hm * stepSize;
        }
        
        currentT += stepSize;
    }
    // Apply air and dust density multipliers
    float3 rayleighScatter = data.nishita.rayleigh_scattering * data.nishita.air_density;
    float3 mieScatter = data.nishita.mie_scattering * data.nishita.dust_density;
    
    float3 L = (totalRayleigh * rayleighScatter * phaseR + 
                totalMie * mieScatter * phaseM) * data.nishita.sun_intensity;
    
    // Apply ozone (affects blue channel - simple approximation)
    float ozoneFactor = data.nishita.ozone_density;
    L.x *= (1.0f + 0.1f * ozoneFactor);  // Slightly reduce red
    L.z *= (1.0f + 0.3f * ozoneFactor);  // Boost blue

    // Add Sun Disk using sun_size (in degrees)
    // Apply horizon magnification effect
    float sunSizeDeg = data.nishita.sun_size;
    float elevationFactor = 1.0f;
    if (data.nishita.sun_elevation < 15.0f) {
        elevationFactor = 1.0f + (15.0f - fmaxf(data.nishita.sun_elevation, -10.0f)) * 0.04f;
    }
    sunSizeDeg *= elevationFactor;
    
    float sun_radius = sunSizeDeg * (M_PI / 180.0f) * 0.5f; // Half angle in radians
    if (dot(dir, sunDir) > cosf(sun_radius)) {
         float3 tau = rayleighScatter * opticalDepthRayleigh + 
                      mieScatter * 1.1f * opticalDepthMie;
         float3 transmittance = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
         L += transmittance * data.nishita.sun_intensity * 1000.0f; // Direct sun
    }
    
    // ═══════════════════════════════════════════════════════════
    // Night Sky: Stars and Moon (visible when sun is low)
    // ═══════════════════════════════════════════════════════════
    auto saturate = [](float x) { return fmaxf(0.0f, fminf(1.0f, x)); };
    float nightFactor = 1.0f - saturate((data.nishita.sun_elevation + 5.0f) / 20.0f);
    
    if (nightFactor > 0.01f && data.nishita.stars_intensity > 0.0f) {
        // Procedural stars using simple hash
        float starScale = 1000.0f;
        float3 gridPos = make_float3(
            floorf(dir.x * starScale),
            floorf(dir.y * starScale),
            floorf(dir.z * starScale)
        );
        
        // Simple hash function
        float hash = fmodf(sinf(gridPos.x * 12.9898f + gridPos.y * 78.233f + gridPos.z * 45.164f) * 43758.5453f, 1.0f);
        if (hash < 0.0f) hash = -hash;
        
        float starThreshold = 1.0f - data.nishita.stars_density * 0.1f;
        if (hash > starThreshold && dir.y > 0.0f) {
            float colorVar = fmodf(hash * 7.0f, 1.0f);
            float3 starColor;
            if (colorVar < 0.3f) {
                starColor = make_float3(1.0f, 0.9f, 0.8f);
            } else if (colorVar < 0.6f) {
                starColor = make_float3(0.9f, 0.95f, 1.0f);
            } else if (colorVar < 0.8f) {
                starColor = make_float3(1.0f, 0.7f, 0.5f);
            } else {
                starColor = make_float3(0.7f, 0.8f, 1.0f);
            }
            
            float twinkle = 0.5f + 0.5f * sinf(hash * 100.0f);
            float starBrightness = (hash - starThreshold) / (1.0f - starThreshold);
            starBrightness = powf(starBrightness, 2.0f) * twinkle;
            
            L += starColor * starBrightness * data.nishita.stars_intensity * nightFactor * 0.1f;
        }
    }
    
    // Moon rendering
    if (data.nishita.moon_enabled && nightFactor > 0.01f && data.nishita.moon_intensity > 0.0f) {
        float moonElevRad = data.nishita.moon_elevation * M_PI / 180.0f;
        float moonAzimRad = data.nishita.moon_azimuth * M_PI / 180.0f;
        float3 moonDir = make_float3(
            cosf(moonElevRad) * sinf(moonAzimRad),
            sinf(moonElevRad),
            cosf(moonElevRad) * cosf(moonAzimRad)
        );
        
        // Horizon magnification effect (like sun)
        float moonSizeDeg = data.nishita.moon_size;
        float moonElevFactor = 1.0f;
        if (data.nishita.moon_elevation < 15.0f) {
            moonElevFactor = 1.0f + (15.0f - fmaxf(data.nishita.moon_elevation, -10.0f)) * 0.04f;
        }
        moonSizeDeg *= moonElevFactor;
        
        float moon_radius = moonSizeDeg * (M_PI / 180.0f) * 0.5f;
        float moonDot = dot(dir, moonDir);
        
        if (moonDot > cosf(moon_radius)) {
            // Base moon color (slightly blue-white)
            float3 moonColor = make_float3(0.9f, 0.9f, 0.95f);
            
            // Horizon color shift (orange/red when low)
            if (data.nishita.moon_elevation < 20.0f) {
                float horizonBlend = (20.0f - fmaxf(data.nishita.moon_elevation, -5.0f)) / 25.0f;
                horizonBlend = fminf(1.0f, fmaxf(0.0f, horizonBlend));
                float3 horizonColor = make_float3(1.0f, 0.7f, 0.4f);
                moonColor.x = moonColor.x * (1.0f - horizonBlend) + horizonColor.x * horizonBlend;
                moonColor.y = moonColor.y * (1.0f - horizonBlend) + horizonColor.y * horizonBlend;
                moonColor.z = moonColor.z * (1.0f - horizonBlend) + horizonColor.z * horizonBlend;
            }
            
            // Atmospheric dimming near horizon
            float atmosphericDim = 1.0f;
            if (data.nishita.moon_elevation < 10.0f) {
                atmosphericDim = 0.3f + 0.7f * fmaxf(0.0f, data.nishita.moon_elevation / 10.0f);
            }
            
            // Phase shading
            float phase = data.nishita.moon_phase;
            float phaseFactor = fabsf(phase - 0.5f) * 2.0f;
            float brightness = 1.0f - phaseFactor * 0.9f;
            
            L += moonColor * brightness * atmosphericDim * data.nishita.moon_intensity * nightFactor * 10.0f;
        }
    }

        // ═══════════════════════════════════════════════════════════
        // Volumetric Clouds (Planar Ray Marching) - CPU Implementation
        // Matches GPU quality settings
        // ═══════════════════════════════════════════════════════════
        if (data.nishita.clouds_enabled && dir.y > 0.01f) {
            float cloudMinY = data.nishita.cloud_height_min;
            float cloudMaxY = data.nishita.cloud_height_max;
            
            // Use actual camera Y position for cloud parallax (matches GPU)
            float camY = (data.camera_y != 0.0f) ? data.camera_y : data.nishita.altitude;
            Vec3 camPos(0.0f, camY, 0.0f);
            Vec3 rayDir = to_vec3(dir);
            
            // Simple plane intersection for camera below clouds
            float t_enter = (cloudMinY - camPos.y) / rayDir.y;
            float t_exit = (cloudMaxY - camPos.y) / rayDir.y;
            
            // Only render if we can see clouds
            if (t_exit > 0.0f && t_exit > t_enter) {
                if (t_enter < 0.0f) t_enter = 0.0f;
                
                // Horizon Fade - smoother transition (matches GPU)
                float h_t = fmaxf(0.0f, fminf(1.0f, rayDir.y / 0.15f));
                float horizonFade = h_t * h_t * (3.0f - 2.0f * h_t);
                
                // Quality-based step count (matches GPU)
                // cloud_quality: 0.25 = fast preview, 0.5 = low, 1.0 = normal, 2.0 = high
                float quality = fmaxf(0.1f, fminf(3.0f, data.nishita.cloud_quality));
                int baseSteps = (int)(32.0f * quality);  // CPU uses fewer base steps
                int numSteps = baseSteps + (int)((float)baseSteps * (1.0f - h_t));
                
                float stepSize = (t_exit - t_enter) / (float)numSteps;
                Vec3 cloudColor(0.0f, 0.0f, 0.0f);
                float transmittance = 1.0f;
                float t = t_enter;
                
                // Cloud parameters
                float scale = 0.003f / fmaxf(0.1f, data.nishita.cloud_scale);
                float coverage = data.nishita.cloud_coverage;
                float densityMult = data.nishita.cloud_density * horizonFade;
                
                Vec3 sunDirVec = to_vec3(normalize(data.nishita.sun_direction));
                float g = fmaxf(0.0f, fminf(0.95f, data.nishita.mie_anisotropy));
                
                // Ambient sky color (30% of background for soft cloud lighting)
                Vec3 bgColor = to_vec3(L);
                Vec3 ambientSky = bgColor * 0.3f;

                for (int i = 0; i < numSteps; ++i) {
                    // Jitter
                    float jitterSeed = (float)i + (rayDir.x * 53.0f + rayDir.z * 91.0f) * 10.0f;
                    Vec3 pos = camPos + rayDir * (t + stepSize * CPUCloudNoise::hash(jitterSeed));
                    
                    // Height Gradient
                    float heightFraction = (pos.y - cloudMinY) / (cloudMaxY - cloudMinY);
                    float heightGradient = 4.0f * heightFraction * (1.0f - heightFraction);
                    heightGradient = fmaxf(0.0f, fminf(1.0f, heightGradient));
                    
                    // Wind offset
                    Vec3 offsetPos = pos + Vec3(data.nishita.cloud_offset_x, 0.0f, data.nishita.cloud_offset_z);
                    
                    // Noise position
                    Vec3 noisePos = offsetPos * scale;
                    
                    // GPU cloud_shape ile aynı noise
                    float rawDensity = CPUCloudNoise::cloud_shape(noisePos, coverage);
                    float density = rawDensity * heightGradient;
                    
                    if (density > 0.003f) {
                        density *= densityMult;
                        
                        // ═══════════════════════════════════════════════════════════
                        // LIGHT MARCHING (Self-Shadowing)
                        // ═══════════════════════════════════════════════════════════
                        float lightTransmittance = 1.0f;
                        int lightSteps = 4;  // Fewer steps for CPU performance
                        float lightStepSize = (cloudMaxY - pos.y) / fmaxf(0.01f, sunDirVec.y) / (float)lightSteps;
                        lightStepSize = fminf(lightStepSize, 500.0f);
                        
                        if (sunDirVec.y > 0.01f) {
                            for (int j = 1; j <= lightSteps; ++j) {
                                Vec3 lightPos = pos + sunDirVec * (lightStepSize * (float)j);
                                
                                if (lightPos.y > cloudMaxY || lightPos.y < cloudMinY) break;
                                
                                Vec3 lightNoisePos = (lightPos + Vec3(data.nishita.cloud_offset_x, 0.0f, data.nishita.cloud_offset_z)) * scale;
                                float lightDensity = CPUCloudNoise::cloud_shape(lightNoisePos, coverage);
                                
                                float lh = (lightPos.y - cloudMinY) / (cloudMaxY - cloudMinY);
                                float lightHeightGrad = 4.0f * lh * (1.0f - lh);
                                lightDensity *= lightHeightGrad * densityMult;
                                
                                lightTransmittance *= expf(-lightDensity * lightStepSize * 0.015f);
                                
                                if (lightTransmittance < 0.1f) break;
                            }
                        }
                        
                        // ═══════════════════════════════════════════════════════════
                        // ADVANCED COLOR CALCULATION
                        // ═══════════════════════════════════════════════════════════
                        float cosTheta = rayDir.x * sunDirVec.x + rayDir.y * sunDirVec.y + rayDir.z * sunDirVec.z;
                        float phase = (1.0f - g * g) / (4.0f * (float)M_PI * powf(1.0f + g * g - 2.0f * g * cosTheta, 1.5f));
                        float powder = CPUCloudNoise::powderEffect(density, cosTheta);
                        
                        // ═══════════════════════════════════════════════════════════
                        // SUN COLOR BASED ON ELEVATION (Sunset/Sunrise)
                        // ═══════════════════════════════════════════════════════════
                        float sunElevation = sunDirVec.y;
                        Vec3 sunColor(1.0f, 0.95f, 0.9f);
                        
                        if (sunElevation < 0.3f) {
                            float sunsetFactor = 1.0f - sunElevation / 0.3f;
                            sunsetFactor = fmaxf(0.0f, fminf(1.0f, sunsetFactor));
                            Vec3 sunsetColor(1.0f, 0.5f, 0.2f);
                            sunColor = sunColor * (1.0f - sunsetFactor * 0.7f) + sunsetColor * sunsetFactor * 0.7f;
                        }
                        
                        // ═══════════════════════════════════════════════════════════
                        // DIRECT LIGHTING (with self-shadowing)
                        // ═══════════════════════════════════════════════════════════
                        float directIntensity = data.nishita.sun_intensity * phase * lightTransmittance * 5.0f;
                        Vec3 directLight = sunColor * directIntensity;
                        
                        float silverLining = fmaxf(0.0f, cosTheta) * powder * lightTransmittance * 4.0f;
                        directLight = directLight + sunColor * silverLining * data.nishita.sun_intensity;
                        
                        // ═══════════════════════════════════════════════════════════
                        // AMBIENT / MULTI-SCATTERING
                        // ═══════════════════════════════════════════════════════════
                        float multiScatter = 0.2f * (1.0f - expf(-density * 4.0f));
                        
                        Vec3 shadowColor(0.15f, 0.2f, 0.35f);
                        float shadowAmount = 1.0f - lightTransmittance;
                        
                        Vec3 ambient = ambientSky * 0.3f * (1.0f - shadowAmount);
                        ambient = ambient + shadowColor * data.nishita.sun_intensity * 0.1f * shadowAmount;
                        ambient = ambient + sunColor * multiScatter * data.nishita.sun_intensity * 0.3f;
                        
                        Vec3 lightColor = directLight + ambient;
                        
                        // Energy-conserving absorption
                        Vec3 stepColor = lightColor * density;
                        float absorption = density * stepSize * 0.012f;
                        float stepTransmittance = expf(-absorption);
                        
                        cloudColor = cloudColor + stepColor * transmittance * (1.0f - stepTransmittance);
                        transmittance *= stepTransmittance;
                        
                        if (transmittance < 0.01f) break;
                    }
                    t += stepSize;
                }
                
                // Blend with atmosphere - no artificial minimum
                Vec3 currentL = bgColor * transmittance + cloudColor;
                L = make_float3(currentL.x, currentL.y, currentL.z);
            }
        }

    return to_vec3(L);
}

