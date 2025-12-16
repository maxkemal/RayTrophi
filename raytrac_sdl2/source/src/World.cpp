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

World::World() {
    data.mode = WORLD_MODE_COLOR;
    data.color = make_float3(0.05f, 0.05f, 0.05f); // Dark grey default
    data.color_intensity = 1.0f;
    data.env_texture = 0;
    data.env_rotation = 0.0f;
    data.env_intensity = 1.0f;
    data.env_width = 0;
    data.env_height = 0;

    // Nishita Defaults (Earth-like)
    data.nishita.sun_elevation = 45.0f;
    data.nishita.sun_azimuth = 180.0f;
    data.nishita.sun_direction = normalize(make_float3(0.0f, 0.707f, -0.707f));
    data.nishita.sun_intensity = 20.0f;
    data.nishita.sun_density = 1.0f;
    data.nishita.dust_density = 1.0f; 
    data.nishita.planet_radius = 6360000.0f;
    data.nishita.atmosphere_height = 6420000.0f;
    
    // Rayleight coefficients for RGB (standard earth)
    data.nishita.rayleigh_scattering = make_float3(5.802e-6f, 13.558e-6f, 33.100e-6f);
    data.nishita.mie_scattering = make_float3(3.996e-6f, 3.996e-6f, 3.996e-6f);
    data.nishita.mie_anisotropy = 0.8f;
    data.nishita.rayleigh_density = 8000.0f; // Scale height H_R
    data.nishita.mie_density = 1200.0f;     // Scale height H_M
    
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
    // Setup vectors
    float3 dir = normalize(to_float3(ray_dir));
    float3 sunDir = data.nishita.sun_direction;
    float planetRadius = data.nishita.planet_radius;
    float atmosphereRadius = data.nishita.atmosphere_height; // This should be total radius or height? User provided height usually.
    // Actually Blender interface usually takes "Air" and "Dust" densities.
    // Assuming atmosphere_height is Top Radius (Planet Radius + Atmosphere Thickness)
    // If atmosphere_height is just thickness, add planet radius. 
    // Usually it's ~6420km vs Earth 6360km, so 60km thickness.
    
    // Check if input is radius or thickness. 6420e3 is radius.
    float Rt = atmosphereRadius;
    float Rg = planetRadius;
    
    // Camera position (assume on surface at top calculation or 0,0,0 + small offset)
    // For sky at infinity, we assume camera is at (0,Rg,0).
    float3 camPos = make_float3(0, Rg + 10.0f, 0); // 10 meters above ground
    
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
    
    float3 L = (totalRayleigh * data.nishita.rayleigh_scattering * phaseR + 
                totalMie * data.nishita.mie_scattering * phaseM) * data.nishita.sun_intensity;

    // Add Sun Disk
    const float sun_radius = 0.02f; // Increased for visibility (~1.15 degrees)
    if (dot(dir, sunDir) > cos(sun_radius)) {
         float3 tau = data.nishita.rayleigh_scattering * opticalDepthRayleigh + 
                      data.nishita.mie_scattering * 1.1f * opticalDepthMie;
         float3 transmittance = make_float3(exp(-tau.x), exp(-tau.y), exp(-tau.z));
         L += transmittance * data.nishita.sun_intensity * 100.0f; // Direct sun (scaled for visibility)
    }

    return to_vec3(L);
                         

}
