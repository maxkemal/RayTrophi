#include "Volumetric.h"
#include "Vec3.h"
#include "Ray.h"
#include <cmath>
#include <cstdlib>
#include "unified_volume_sampling.h"
#include "unified_converters.h"
#include "Triangle.h"
#include "AABB.h"
#include "VDBVolumeManager.h"



Volumetric::Volumetric(const Vec3& a, float d, float ap, float sf, const Vec3& e, std::shared_ptr<Perlin> noiseGen)
    : albedo(a), density(d), absorption_probability(ap), scattering_factor(sf), emission(e), noise(noiseGen) 
{
    // Defaults suitable for dynamic volumes
    max_distance = 10.0f;
    step_size = 0.05f; 
    max_steps = 128;   
    g = 0.0f; 
}

float Volumetric::calculate_density(const Vec3& point) const {
    // Check if using VDB density source
    if (density_source == 1 && vdb_volume_id >= 0) {
        // Sample from VDB volume
        float vdb_density = VDBVolumeManager::getInstance().sampleDensityCPU(
            vdb_volume_id, 
            static_cast<float>(point.x), 
            static_cast<float>(point.y), 
            static_cast<float>(point.z)
        );
        return vdb_density * density;  // Apply density multiplier
    }
    
    // Fallback to procedural noise (existing behavior)
    UnifiedVolumeParams params;
    params.density_multiplier = density;
    params.noise_scale = noise_scale;
    params.void_threshold = void_threshold;
    // Fallback to world space (invalid AABB)
    params.aabb_min = Vec3f(0);
    params.aabb_max = Vec3f(0);
    
    return calculate_unified_density(toVec3f(point), params);
}

Vec3 Volumetric::march_volume(const Vec3& origin, const Vec3& dir, float& out_transmittance, const Vec3& aabb_min, const Vec3& aabb_max) const {
    // Map class members to unified struct
    UnifiedVolumeParams params;
    params.density_multiplier = density;
    params.absorption_prob = absorption_probability;
    params.scattering_factor = scattering_factor;
    params.albedo = toVec3f(albedo);
    params.emission_color = toVec3f(emission);
    params.step_size = step_size;
    params.max_steps = max_steps;
    params.noise_scale = noise_scale;
    params.void_threshold = void_threshold;
    params.aabb_min = toVec3f(aabb_min);
    params.aabb_max = toVec3f(aabb_max);
    
    // Multi-Scattering Parameters
    params.g = g;
    params.multi_scatter = multi_scatter;
    params.g_back = g_back;
    params.lobe_mix = lobe_mix;
    params.light_steps = light_steps;
    params.shadow_strength = shadow_strength;
    
    // Default sun direction (can be set from scene later)
    params.sun_direction = Vec3f(0.0f, 1.0f, 0.0f);
    params.sun_intensity = 1.0f;
    
    // Generate jitter for stochastic sampling (reduces banding)
    float jitter = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    
    float T = 1.0f;
    // Call the unified kernel function
    Vec3f vol_color = unified_march_volume(toVec3f(origin), toVec3f(dir), params, T, jitter);
    
    out_transmittance = T;
    return toVec3(vol_color);
}

bool Volumetric::scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    // Only transmittance behavior for surface-based volume
    
    // Fetch AABB from the hit object (Triangle)
    Vec3 aabb_min(0), aabb_max(0);
    if (rec.triangle) {
        AABB box;
        if (rec.triangle->bounding_box(0, 0, box)) {
            aabb_min = box.min;
            aabb_max = box.max;
            
            // Add padding (matches AssimpLoader logic)
            float padding = 0.001f;
            aabb_min = aabb_min - Vec3(padding);
            aabb_max = aabb_max + Vec3(padding);
        }
    }

    float T = 1.0f;
    // Use the march_volume helper (which uses unified logic)
    march_volume(rec.point, r_in.direction, T, aabb_min, aabb_max);
    
    attenuation = Vec3(T);
    scattered = Ray(rec.point + r_in.direction * 0.001f, r_in.direction);
    return true; 
}

Vec3 Volumetric::getVolumetricEmission(const Vec3& p, const Vec3& dir, const Vec3& aabb_min, const Vec3& aabb_max) const {
    float dummy_T = 1.0f;
    return march_volume(p, dir, dummy_T, aabb_min, aabb_max);
}

Vec3 Volumetric::getEmission(const Vec2& uv, const Vec3& p) const {
    return Vec3(0); 
}

Vec3 Volumetric::getEmission() const {
    return Vec3(0);
}

float Volumetric::get_opacity(const Vec2& uv) const {
    return 1.0f; 
}
