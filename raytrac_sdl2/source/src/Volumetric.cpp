#include "Volumetric.h"
#include "Vec3.h"
#include "Ray.h"
#include <cmath>
#include <cstdlib>
#include "unified_volume_sampling.h"
#include "unified_converters.h"

Volumetric::Volumetric(const Vec3& a, double d, double ap, double sf, const Vec3& e, std::shared_ptr<Perlin> noiseGen)
    : albedo(a), density(d), absorption_probability(ap), scattering_factor(sf), emission(e), noise(noiseGen) 
{
    // Defaults suitable for dynamic volumes
    max_distance = 10.0;
    step_size = 0.05f; 
    max_steps = 128;   
    g = 0.0f; 
}

double Volumetric::calculate_density(const Vec3& point) const {
    // Defines shared density logic (procedural noise) for parity
    return (double)calculate_unified_density(toVec3f(point), (float)density);
}

Vec3 Volumetric::march_volume(const Vec3& origin, const Vec3& dir, float& out_transmittance) const {
    // Map class members to unified struct
    UnifiedVolumeParams params;
    params.density_multiplier = (float)density;
    params.absorption_prob = (float)absorption_probability;
    params.scattering_factor = (float)scattering_factor;
    params.emission_color = toVec3f(emission);
    params.step_size = step_size;
    params.max_steps = max_steps;
    
    // Generate jitter for stochastic sampling (reduces banding)
    float jitter = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    
    float T = 1.0f;
    // Call the unified kernel function
    Vec3f vol_emit = unified_march_volume(toVec3f(origin), toVec3f(dir), params, T, jitter);
    
    out_transmittance = T;
    return toVec3(vol_emit);
}

bool Volumetric::scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered) const {
    // Only transmittance behavior for surface-based volume
    float T = 1.0f;
    // Use the march_volume helper (which uses unified logic)
    march_volume(rec.point, r_in.direction, T);
    
    attenuation = Vec3(T);
    scattered = Ray(rec.point + r_in.direction * 0.001f, r_in.direction);
    return true; 
}

Vec3 Volumetric::getVolumetricEmission(const Vec3& p, const Vec3& dir) const {
    float dummy_T;
    return march_volume(p, dir, dummy_T);
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
