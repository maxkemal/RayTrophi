/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          unified_light_sampling.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
/**
 * @file unified_light_sampling.h
 * @brief Unified light sampling functions for CPU/GPU parity
 * 
 * These functions mirror the GPU's light sampling exactly.
 * The GPU implementation (ray_color.cuh) is the reference.
 */
#pragma once

#include "unified_types.h"
#include "unified_brdf.h"

// =============================================================================
// LIGHT PDF CALCULATION
// =============================================================================

/**
 * @brief Calculate PDF for sampling a point on a light
 * 
 * @param light Light to sample
 * @param distance Distance from hit point to light sample
 * @param pdf_select Probability of selecting this light (1/N for uniform)
 * @return Combined PDF (light area PDF * selection PDF)
 */
UNIFIED_FUNC float compute_light_pdf(
    const UnifiedLight& light,
    float distance,
    float pdf_select
) {
    float pdf = 1.0f;
    
    switch (light.type) {
        case 0: // Point Light
        {
            float area = 4.0f * UnifiedConstants::PI * light.radius * light.radius;
            pdf = (1.0f / fmaxf(area, 1e-4f)) * pdf_select;
            break;
        }
        case 1: // Directional Light
        {
            float apparent_angle = atan2f(light.radius, 1000.0f);
            float cos_epsilon = cosf(apparent_angle);
            float solid_angle = 2.0f * UnifiedConstants::PI * (1.0f - cos_epsilon);
            pdf = (1.0f / fmaxf(solid_angle, 1e-4f)) * pdf_select;
            break;
        }
        case 2: // Area Light
        {
            float area = light.area_width * light.area_height;
            pdf = (1.0f / fmaxf(area, 1e-4f)) * pdf_select;
            break;
        }
        case 3: // Spot Light
        {
            float solid_angle = 2.0f * UnifiedConstants::PI * (1.0f - light.outer_cone_cos);
            pdf = (1.0f / fmaxf(solid_angle, 1e-4f)) * pdf_select;
            break;
        }
    }
    
    return pdf;
}

// =============================================================================
// SPOT LIGHT FALLOFF
// =============================================================================

/**
 * @brief Calculate spot light cone falloff
 * 
 * Exactly matches GPU: spot_light_falloff() in ray_color.cuh
 * 
 * @param light Spot light
 * @param wi Direction from hit point to light (normalized)
 * @return Falloff factor [0-1]
 */
UNIFIED_FUNC float spot_light_falloff(const UnifiedLight& light, const Vec3f& wi) {
    float cos_theta = dot(-wi, normalize(light.direction));
    
    if (cos_theta < light.outer_cone_cos) return 0.0f;
    if (cos_theta > light.inner_cone_cos) return 1.0f;
    
    // Smooth quadratic falloff between inner and outer cone
    float t = (cos_theta - light.outer_cone_cos) / 
              (light.inner_cone_cos - light.outer_cone_cos + 1e-6f);
    return t * t;
}

// =============================================================================
// SAMPLE LIGHT DIRECTION
// =============================================================================

/**
 * @brief Sample direction toward a light source
 * 
 * @param light Light to sample
 * @param hit_pos Hit point position
 * @param rand_u Random value [0,1]
 * @param rand_v Random value [0,1]
 * @param wi_out Output: direction to light (normalized)
 * @param distance_out Output: distance to light sample
 * @param attenuation_out Output: light attenuation factor
 * @return True if valid sample (not blocked by cone, etc.)
 */
UNIFIED_FUNC bool sample_light_direction(
    const UnifiedLight& light,
    const Vec3f& hit_pos,
    float rand_u,
    float rand_v,
    Vec3f* wi_out,
    float* distance_out,
    float* attenuation_out
) {
    Vec3f wi;
    float distance = 1.0f;
    float attenuation = 1.0f;
    
    switch (light.type) {
        case 0: // Point Light
        {
            Vec3f L = light.position - hit_pos;
            distance = L.length();
            if (distance < 1e-3f) return false;
            
            Vec3f dir = L / distance;
            
            // Jitter for soft shadows (random point on sphere)
            // Note: For CPU we'll need to pass random samples
            Vec3f jitter = Vec3f(
                (rand_u - 0.5f) * 2.0f,
                (rand_v - 0.5f) * 2.0f,
                (rand_u * rand_v - 0.5f) * 2.0f
            ).normalize() * light.radius;
            
            wi = normalize(dir * distance + jitter);
            attenuation = 1.0f / (distance * distance);
            break;
        }
        
        case 1: // Directional Light
        {
            Vec3f L = normalize(light.direction);
            
            // Build tangent frame for disk sampling
            Vec3f tangent = normalize(cross(L, Vec3f(0.0f, 1.0f, 0.0f)));
            if (tangent.length_squared() < 1e-6f) {
                tangent = normalize(cross(L, Vec3f(1.0f, 0.0f, 0.0f)));
            }
            Vec3f bitangent = normalize(cross(L, tangent));
            
            // Disk sample
            float r = sqrtf(rand_u) * light.radius;
            float phi = 2.0f * UnifiedConstants::PI * rand_v;
            Vec2f disk_p(cosf(phi) * r, sinf(phi) * r);
            
            // Match GPU: treat radius as angular spread via 1000-unit distance offset
            // Reference: sample_directional_light in ray_color.cuh
            Vec3f light_pos = L * 1000.0f + tangent * disk_p.x + bitangent * disk_p.y;
            wi = normalize(light_pos);
            attenuation = 1.0f;  // No falloff for directional
            distance = 1e8f;
            break;
        }
        
        case 2: // Area Light
        {
            // Sample point on area light
            float u_offset = (rand_u - 0.5f) * light.area_width;
            float v_offset = (rand_v - 0.5f) * light.area_height;
            Vec3f light_sample = light.position + 
                                 light.area_u * u_offset + 
                                 light.area_v * v_offset;
            
            Vec3f L = light_sample - hit_pos;
            distance = L.length();
            if (distance < 1e-3f) return false;
            
            wi = L / distance;
            
            // Cosine falloff based on light normal
            Vec3f light_normal = normalize(cross(light.area_u, light.area_v));
            float cos_light = fmaxf(dot(-wi, light_normal), 0.0f);
            attenuation = cos_light / (distance * distance);
            break;
        }
        
        case 3: // Spot Light
        {
            Vec3f L = light.position - hit_pos;
            distance = L.length();
            if (distance < 1e-3f) return false;
            
            wi = L / distance;
            
            // Check cone and compute falloff
            float falloff = spot_light_falloff(light, wi);
            if (falloff < 1e-4f) return false;
            
            attenuation = falloff / (distance * distance);
            break;
        }
        
        default:
            return false;
    }
    
    *wi_out = wi;
    *distance_out = distance;
    *attenuation_out = attenuation;
    return true;
}

// =============================================================================
// CALCULATE DIRECT LIGHTING
// =============================================================================

/**
 * @brief Calculate direct lighting contribution from a single light
 * 
 * This mirrors calculate_light_contribution() from ray_color.cuh.
 * 
 * @param light Light source
 * @param hit Hit information
 * @param mat Material at hit point
 * @param wo View direction (normalized, pointing away from surface)
 * @param albedo_sampled Sampled albedo color
 * @param roughness_sampled Sampled roughness
 * @param metallic_sampled Sampled metallic
 * @param rand_u Random value for light sampling
 * @param rand_v Random value for light sampling
 * @param is_shadowed Function/lambda to check shadow (returns true if blocked)
 * @return Direct lighting contribution
 */
template<typename ShadowFunc>
UNIFIED_FUNC Vec3f calculate_light_contribution_unified(
    const UnifiedLight& light,
    const Vec3f& hit_position,
    const Vec3f& normal,
    const Vec3f& wo,
    const Vec3f& albedo_sampled,
    float roughness_sampled,
    float metallic_sampled,
    float rand_u,
    float rand_v,
    float pdf_select,
    ShadowFunc is_shadowed
) {
    Vec3f wi;
    float distance;
    float attenuation;
    
    // Sample direction to light
    bool valid = sample_light_direction(
        light, hit_position, rand_u, rand_v,
        &wi, &distance, &attenuation
    );
    
    if (!valid) return Vec3f(0.0f);
    
    // Check if light is in front of surface
    float NdotL = dot(normal, wi);
    if (NdotL <= 0.001f) return Vec3f(0.0f);
    
    // Shadow check
    if (is_shadowed(hit_position, wi, distance)) {
        return Vec3f(0.0f);
    }
    
    // Evaluate BRDF
    Vec3f f = evaluate_brdf_unified(
        normal, wo, wi,
        albedo_sampled, roughness_sampled, metallic_sampled
    );
    
    // Light PDF
    float pdf_light = compute_light_pdf(light, distance, 1.0f);
    
    // BRDF PDF for MIS
    float pdf_brdf = pdf_brdf_unified(normal, wo, wi, roughness_sampled);
    float pdf_brdf_clamped = fminf(fmaxf(pdf_brdf, 0.001f), 5000.0f);
    
    // MIS weight
    float mis_weight = power_heuristic(pdf_light, pdf_brdf_clamped);
    
    // Light radiance
    Vec3f Li = light.color * light.intensity * attenuation;

    // Final contribution (Corrected: Divide by pdf_light and pdf_select for standard PT)
    // Formula: (f * Li * NdotL / pdf_light) * weight / pdf_select
    // For analytic lights, Li is usually already irradiance, so we don't divide by pdf_light here.
    // However, we MUST handle pdf_select.
    return (f * Li * NdotL * mis_weight) / fmaxf(pdf_select, 1e-4f);
}

// =============================================================================
// SMART LIGHT PICKING (IMPORTANCE-BASED)
// =============================================================================

/**
 * @brief Pick a light using importance-based selection
 * 
 * Matches pick_smart_light() from ray_color.cuh.
 * 
 * @param lights Array of lights
 * @param light_count Number of lights
 * @param hit_position Current hit position
 * @param random_val Random value [0,1]
 * @return Index of selected light, or -1 if no lights
 */
UNIFIED_FUNC int pick_smart_light_unified(
    const UnifiedLight* lights,
    int light_count,
    const Vec3f& hit_position,
    float random_val,
    float* pdf_out = nullptr
) {
    if (light_count == 0) {
        if (pdf_out) *pdf_out = 0.0f;
        return -1;
    }
    
    // Create a mutable random value we can consume
    float rng_val = random_val;
    
    // --- Step 1: Directional Light Shortcut (Matches GPU) ---
    // GPU iterates all lights and gives each Directional light a 33% chance.
    // If multiple directional lights exist, they are checked sequentially.
    
    float prob_to_reach_weighted = 1.0f;
    
    for (int i = 0; i < light_count; i++) {
        if (lights[i].type == 1) {  // Directional
            // Check if we pick this light using the 33% shortcut
            // To simulate "fresh" random usage like GPU would with random_float(rng),
            // we slice the current random value. 
            // However, since we only have single 'random_val', we map it.
            // Simplified: If random_val is in [0, 0.33] of the CURRENT range.
            
            // For true parity with GPU which calls random_float() inside the loop:
            // We can't strictly emulate multiple Independent calls from one float without hashing.
            // But assuming 1 Directional Light (common case), we just need to check 0.33.
            
            // Let's use a hashed approach to simulate a new random number for the check if we have multiple
            // For single directional light (standard), this collapses to simple check.
            
            // Standard approach: Use the passed random_val for the DECISION.
            // If we have 1 Dir light, range [0, 0.33) picks it.
            // Range [0.33, 1.0) proceeds to weighted.
            
            if (rng_val < 0.33f) {
                if (pdf_out) *pdf_out = 0.33f * prob_to_reach_weighted; // GPU: 0.33 probability at this step
                return i;
            }
            
            // If not picked, we proceed.
            // Renormalize rng_val for next steps?
            // GPU generates NEW random number. 
            // We map [0.33, 1.0] -> [0.0, 1.0]
            rng_val = (rng_val - 0.33f) / 0.67f;
            prob_to_reach_weighted *= 0.67f;
        }
    }
    
    // --- Step 2: Weighted Selection (Matches GPU) ---
    // Directional lights get 0 weight here.
    
    float weights[128];
    float total_weight = 0.0f;
    int max_lights = (light_count < 128) ? light_count : 128;
    
    for (int i = 0; i < max_lights; i++) {
        const UnifiedLight& light = lights[i];
        
        float weight = 0.0f;
        
        if (light.type != 1) { // Skip Directional (Weight 0)
            Vec3f delta = light.position - hit_position;
            float dist_sq = delta.length_squared();
            float dist = sqrtf(dist_sq);
            dist = fmaxf(dist, 1.0f);
            
            float intensity = light.color.luminance() * light.intensity;
            
            if (light.type == 0) { // Point
                 float falloff = 1.0f / (dist * dist);
                 weight = falloff * intensity;
            }
            else if (light.type == 2) { // Area
                 float falloff = 1.0f / (dist * dist);
                 float area = light.area_width * light.area_height;
                 weight = falloff * intensity * fminf(area, 10.0f);
            }
            else if (light.type == 3) { // Spot
                 float falloff = 1.0f / (dist * dist);
                 weight = falloff * intensity * 0.8f;
            }
        }
        
        weights[i] = weight;
        total_weight += weight;
    }
    
    // Pick from weighted
    int selected_idx = max_lights - 1; // Default
    
    if (total_weight < 1e-6f) {
        // Uniform fallback
        selected_idx = static_cast<int>(rng_val * light_count) % light_count;
        if (pdf_out) *pdf_out = prob_to_reach_weighted * (1.0f / (float)light_count);
        return selected_idx;
    }
    
    float r = rng_val * total_weight;
    float accum = 0.0f;
    
    for (int i = 0; i < max_lights; i++) {
        accum += weights[i];
        if (r <= accum) {
            selected_idx = i;
            break;
        }
    }
    
    if (pdf_out) {
        *pdf_out = prob_to_reach_weighted * (weights[selected_idx] / total_weight);
    }
    
    return selected_idx;
}

