#pragma once
#include "unified_types.h"
#include "unified_volume.h"
#include "unified_noise.h"

struct UnifiedVolumeParams {
    float density_multiplier = 0.0f;
    float absorption_prob = 0.0f;
    float scattering_factor = 0.0f;
    Vec3f albedo = {1.0f, 1.0f, 1.0f};      // Scattering color
    Vec3f emission_color = {0,0,0};
    float step_size = 0.1f;
    int max_steps = 100;
    float noise_scale = 1.0f;
    float void_threshold = 0.0f; // Control for creating empty spaces (0.0 to 1.0)
    Vec3f aabb_min = {0,0,0};
    Vec3f aabb_max = {0,0,0};
    
    // Multi-Scattering Parameters (NEW)
    float g = 0.0f;              // Forward anisotropy
    float multi_scatter = 0.3f;  // Multi-scatter contribution (0-1)
    float g_back = -0.3f;        // Backward scattering anisotropy
    float lobe_mix = 0.7f;       // Forward/backward lobe mix
    int light_steps = 4;         // Light march steps (0=disabled)
    float shadow_strength = 0.8f; // Self-shadow intensity
    
    // Sun direction for light marching (set by caller)
    Vec3f sun_direction = {0.0f, 1.0f, 0.0f};
    float sun_intensity = 1.0f;
};

// Calculate density using unified noise (Object Space aware)
UNIFIED_FUNC float calculate_unified_density(const Vec3f& p, const UnifiedVolumeParams& params) {
    float base_density = params.density_multiplier;
    if (base_density <= 0.0f) return 0.0f;
    
    // Normalize coordinates to Object Space (0..1) based on AABB
    Vec3f size = params.aabb_max - params.aabb_min;
    
    // Fallback if AABB is invalid (size too small) -> Use World Space
    Vec3f pos_obj;
    if (size.x < 0.0001f || size.y < 0.0001f || size.z < 0.0001f) {
        pos_obj = p; // Fallback to world space
    } else {
        pos_obj = (p - params.aabb_min) / size;
    }

    // AABB Bounds Check (Crucial: Stop density outside object for correct transparency)
    // If we are outside the [0,1] range in any axis, we are outside the volume.
    if (pos_obj.x < 0.0f || pos_obj.x > 1.0f ||
        pos_obj.y < 0.0f || pos_obj.y > 1.0f ||
        pos_obj.z < 0.0f || pos_obj.z > 1.0f) {
        return 0.0f; 
    }

    // Apply scaling
    float current_noise_scale = (params.noise_scale > 0.0f) ? params.noise_scale : 1.0f;
    Vec3f noise_p = pos_obj * current_noise_scale;
    
    // 3 Octave fBm for detail
    float turbulence = fbm(noise_p, 3);
    
    // Simple density modulation (matches GPU)
    // turbulence is roughly 0.3-0.7 range, use directly as multiplier
    float density_result = base_density * turbulence;
    
    // ═══════════════════════════════════════════════════════════
    // EDGE FALLOFF - Smooth transition at AABB boundaries
    // ═══════════════════════════════════════════════════════════
    {
        // Falloff distance as percentage of normalized volume (15% each side)
        float falloff_dist = 0.15f;
        
        // Distance to each face in normalized [0,1] space
        float dx_min = pos_obj.x;
        float dx_max = 1.0f - pos_obj.x;
        float dy_min = pos_obj.y;
        float dy_max = 1.0f - pos_obj.y;
        float dz_min = pos_obj.z;
        float dz_max = 1.0f - pos_obj.z;
        
        // Combined edge falloff (minimum distance to any face)
        float d_edge = fminf(fminf(fminf(dx_min, dx_max), fminf(dy_min, dy_max)), fminf(dz_min, dz_max));
        
        // Smooth falloff using smoothstep
        float edge_factor = 1.0f;
        if (d_edge < falloff_dist && falloff_dist > 0.001f) {
            float t = d_edge / falloff_dist;
            // Smoothstep: 3t^2 - 2t^3
            edge_factor = t * t * (3.0f - 2.0f * t);
        }
        
        density_result *= edge_factor;
    }
    
    return density_result;
}

// Light Marching for Self-Shadowing
// Returns transmittance from current point towards sun
UNIFIED_FUNC float light_march_transmittance(
    const Vec3f& pos,
    const Vec3f& sun_dir,
    const UnifiedVolumeParams& params
) {
    if (params.light_steps <= 0) return 1.0f;
    
    // Calculate march distance based on AABB size
    Vec3f aabb_size = params.aabb_max - params.aabb_min;
    float volume_size = (aabb_size.x + aabb_size.y + aabb_size.z) / 3.0f;
    float light_step = volume_size / (float)params.light_steps;
    
    float light_transmittance = 1.0f;
    float density_accum = 0.0f;
    
    for (int j = 1; j <= params.light_steps; ++j) {
        Vec3f light_pos = pos + sun_dir * (light_step * (float)j);
        
        // Check if still in AABB
        if (light_pos.x < params.aabb_min.x || light_pos.x > params.aabb_max.x ||
            light_pos.y < params.aabb_min.y || light_pos.y > params.aabb_max.y ||
            light_pos.z < params.aabb_min.z || light_pos.z > params.aabb_max.z) {
            break;
        }
        
        float light_density = calculate_unified_density(light_pos, params);
        density_accum += light_density * params.absorption_prob * light_step;
        
        if (density_accum > 5.0f) break; // Early exit for dense regions
    }
    
    // Beer's Law for light absorption
    float beers = expf(-density_accum);
    
    // Secondary softer term for multi-scatter approximation
    float beers_soft = expf(-density_accum * 0.25f);
    
    // Blend based on multi-scatter setting
    float albedo_avg = (params.albedo.x + params.albedo.y + params.albedo.z) / 3.0f;
    light_transmittance = beers * (1.0f - params.multi_scatter * albedo_avg) + 
                          beers_soft * params.multi_scatter * albedo_avg;
    
    // Apply shadow strength
    light_transmittance = 1.0f - params.shadow_strength * (1.0f - light_transmittance);
    
    return light_transmittance;
}

// Unified Ray Marching Function with Multi-Scattering
// Returns: Accumulated Color (In-scattering + Emission)
// Out: Transmittance (T)
UNIFIED_FUNC Vec3f unified_march_volume(
    const Vec3f& origin, 
    const Vec3f& dir, 
    const UnifiedVolumeParams& params,
    float& out_transmittance,
    float rand_jitter
) {
    Vec3f accumulated_color(0.0f);
    float T = 1.0f;
    Vec3f p = origin;
    
    // Apply jitter to start position (anti-banding)
    p = p + dir * (rand_jitter * params.step_size);
    
    // Safety break for loop
    int max_steps = (params.max_steps > 0 && params.max_steps < 1024) ? params.max_steps : 128;
    
    // Normalize sun direction
    Vec3f sun_dir = params.sun_direction;
    float sun_len = sqrtf(sun_dir.x*sun_dir.x + sun_dir.y*sun_dir.y + sun_dir.z*sun_dir.z);
    if (sun_len > 0.001f) sun_dir = sun_dir / sun_len;
    
    for(int i = 0; i < max_steps; ++i) {
        float d = calculate_unified_density(p, params);
        
        if (d > 0.001f) {
            float sigma_a = params.absorption_prob * d;
            float sigma_s = params.scattering_factor * d;
            float sigma_t = sigma_a + sigma_s;
            
            // Albedo average for multi-scatter calculation
            float albedo_avg = (params.albedo.x + params.albedo.y + params.albedo.z) / 3.0f;
            
            // Multi-scatter transmittance
            float step_T = compute_multiscatter_transmittance(
                sigma_t, params.step_size, params.multi_scatter, albedo_avg
            );
            
            // ═══════════════════════════════════════════════════════════
            // IN-SCATTERING CALCULATION
            // ═══════════════════════════════════════════════════════════
            
            // Dual-lobe phase function
            float cos_theta = -(dir.x * sun_dir.x + dir.y * sun_dir.y + dir.z * sun_dir.z);
            float phase = phase_dual_henyey_greenstein(cos_theta, params.g, params.g_back, params.lobe_mix);
            
            // Light marching for self-shadowing
            float light_T = light_march_transmittance(p, sun_dir, params);
            
            // Powder effect
            float powder = powder_effect_volume(d, cos_theta);
            
            // In-scatter contribution
            Vec3f Li = Vec3f(params.sun_intensity);
            Vec3f inscatter = params.albedo * Li * phase * sigma_s * light_T;
            inscatter = inscatter * (1.0f + powder * 0.5f); // Powder boost
            
            // ═══════════════════════════════════════════════════════════
            // EMISSION
            // ═══════════════════════════════════════════════════════════
            Vec3f emit = params.emission_color * d;
            
            // ═══════════════════════════════════════════════════════════
            // ACCUMULATE (matches GPU formula)
            // ═══════════════════════════════════════════════════════════
            Vec3f step_color = (inscatter + emit) * T * (1.0f - step_T);
            accumulated_color = accumulated_color + step_color;
            
            T *= step_T;
            if (T < 0.01f) {
                T = 0.0f;
                break;
            }
        }
        
        p = p + dir * params.step_size;
    }
    
    out_transmittance = T;
    return accumulated_color;
}

