/**
 * @file unified_cpu_adapter.h
 * @brief Adapter to use unified functions with existing CPU code
 * 
 * This header bridges the gap between the unified rendering functions
 * and the existing CPU renderer, allowing gradual migration.
 * 
 * Usage in Renderer.cpp:
 *   1. Include this header
 *   2. Convert materials/lights to unified types at render start
 *   3. Use unified BRDF/light functions in ray_color
 */
#pragma once

#include "unified_types.h"
#include "unified_brdf.h"
#include "unified_light_sampling.h"
#include "unified_converters.h"
#include "Vec3.h"
#include "Vec2.h"
#include "Hittable.h"
#include "Ray.h"
#include <vector>
#include <memory>

// Forward declarations
class Material;
class Light;
class PrincipledBSDF;
class Texture;

/**
 * @brief CPU-side texture sampler
 * 
 * Wraps texture sampling for CPU renderer.
 */
class CPUTextureSampler {
public:
    // Sample texture by ID (returns default if not found)
    static Vec3f sampleAlbedo(const std::shared_ptr<Texture>& tex, float u, float v, const Vec3f& default_val);
    static float sampleRoughness(const std::shared_ptr<Texture>& tex, float u, float v, float default_val);
    static float sampleMetallic(const std::shared_ptr<Texture>& tex, float u, float v, float default_val);
    static float sampleOpacity(const std::shared_ptr<Texture>& tex, float u, float v, float default_val);
};

/**
 * @brief Unified render context for CPU
 * 
 * Holds converted materials and lights for efficient rendering.
 */
struct UnifiedCPURenderContext {
    std::vector<UnifiedMaterial> materials;
    std::vector<UnifiedLight> lights;
    Vec3f background_color;
    int max_bounces;
    int max_samples;
    
    // Light selection cache
    int light_count;
    
    UnifiedCPURenderContext() : 
        background_color(0.3f, 0.3f, 0.35f),
        max_bounces(8),
        max_samples(64),
        light_count(0)
    {}
    
    /**
     * @brief Build context from scene data
     */
    void buildFromScene(
        const std::vector<std::shared_ptr<Light>>& scene_lights,
        const Vec3& bg_color,
        int bounces,
        int samples
    ) {
        // Convert lights
        lights.clear();
        lights.reserve(scene_lights.size());
        for (const auto& light : scene_lights) {
            lights.push_back(toUnifiedLight(light));
        }
        light_count = static_cast<int>(lights.size());
        
        background_color = toVec3f(bg_color);
        max_bounces = bounces;
        max_samples = samples;
    }
};

/**
 * @brief CPU shadow query function
 * 
 * Checks if a ray is occluded between origin and max_dist.
 */
inline bool cpu_shadow_test(
    const Hittable* bvh,
    const Vec3f& origin, 
    const Vec3f& direction, 
    float max_dist
) {
    Vec3 origin_v3(origin.x, origin.y, origin.z);
    Vec3 dir_v3(direction.x, direction.y, direction.z);
    
    // Offset origin slightly along normal to avoid self-intersection
    Vec3 shadow_origin = origin_v3 + dir_v3 * UnifiedConstants::SHADOW_BIAS;
    
    Ray shadow_ray(shadow_origin, dir_v3);
    return bvh->occluded(shadow_ray, UnifiedConstants::SHADOW_BIAS, max_dist);
}

/**
 * @brief Extract material parameters with texture sampling
 * 
 * This helper extracts all material parameters, applying texture
 * sampling where available, matching GPU behavior.
 */
inline void extract_material_params(
    const std::shared_ptr<Material>& material,
    float u, float v,
    Vec3f* albedo_out,
    float* roughness_out,
    float* metallic_out,
    float* opacity_out,
    float* transmission_out,
    Vec3f* emission_out
) {
    // Default values (matching GPU defaults)
    *albedo_out = Vec3f(0.8f);
    *roughness_out = 0.5f;
    *metallic_out = 0.0f;
    *opacity_out = 1.0f;
    *transmission_out = 0.0f;
    *emission_out = Vec3f(0.0f);
    
    if (!material) return;
    
    auto pbsdf = std::dynamic_pointer_cast<PrincipledBSDF>(material);
    if (!pbsdf) return;
    
    Vec2 uv(u, v);
    
    // Albedo
    Vec3 alb = pbsdf->getPropertyValue(pbsdf->albedoProperty, uv);
    *albedo_out = toVec3f(alb);
    
    // Roughness (Y channel)
    Vec3 rough = pbsdf->getPropertyValue(pbsdf->roughnessProperty, uv);
    *roughness_out = static_cast<float>(rough.y);
    
    // Metallic (Z channel)
    Vec3 metal = pbsdf->getPropertyValue(pbsdf->metallicProperty, uv);
    *metallic_out = static_cast<float>(metal.z);
    
    // Opacity
    *opacity_out = pbsdf->get_opacity(uv);
    
    // Transmission
    *transmission_out = pbsdf->getTransmission(uv);
    
    // Emission
    Vec3 em = pbsdf->getEmission(uv, Vec3(0.0f));
    *emission_out = toVec3f(em);
}

/**
 * @brief Compute direct lighting using unified functions
 * 
 * This is a drop-in replacement for calculate_direct_lighting_single_light()
 * that uses the unified BRDF calculations.
 */
inline Vec3f compute_direct_lighting_unified(
    const UnifiedLight& light,
    const Hittable* bvh,
    const Vec3f& hit_position,
    const Vec3f& normal,
    const Vec3f& wo,
    const Vec3f& albedo,
    float roughness,
    float metallic,
    float rand_u,
    float rand_v
) {
    // Lambda for shadow test
    auto shadow_test = [bvh](const Vec3f& origin, const Vec3f& dir, float dist) -> bool {
        return cpu_shadow_test(bvh, origin, dir, dist);
    };
    
    return calculate_light_contribution_unified(
        light,
        hit_position,
        normal,
        wo,
        albedo,
        roughness,
        metallic,
        rand_u,
        rand_v,
        shadow_test
    );
}

/**
 * @brief Full ray color calculation using unified functions
 * 
 * This is a simplified example showing how to use unified functions
 * in the main ray tracing loop. The actual implementation should
 * integrate with existing texture sampling and hit record structures.
 * 
 * @param ray Initial ray
 * @param bvh BVH for intersection
 * @param ctx Unified render context
 * @param base_sample_index Sample index for random number generation
 * @return Final color for this ray
 */
// Note: Implementation would go in Renderer.cpp using existing infrastructure
// This is just the interface specification

// Note: This header doesn't use include guards since it's meant to be 
// included once per translation unit and relies on #pragma once
