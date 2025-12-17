#pragma once
#include "unified_types.h"
#include "unified_volume.h"
#include "unified_noise.h"

struct UnifiedVolumeParams {
    float density_multiplier;
    float absorption_prob;
    float scattering_factor;
    Vec3f emission_color;
    float step_size;
    int max_steps;
};

// Calculate density using unified noise
UNIFIED_FUNC float calculate_unified_density(const Vec3f& p, float base_density) {
    if (base_density <= 0.0f) return 0.0f;
    
    // Scale for noise coordinates
    float scale = 0.5f; 
    Vec3f noise_p = p * scale;
    
    // 3 Octave fBm
    float turbulence = fbm(noise_p, 3);
    
    // Thresholding for sparse look
    float threshold = 0.1f;
    if (turbulence < threshold) return 0.0f;
    
    float smooth_d = (turbulence - threshold) / (1.0f - threshold);
    return base_density * smooth_d;
}

// Unified Ray Marching Function
// Returns: Accumulated Emission (Radiance)
// Out: Transmittance (T)
UNIFIED_FUNC Vec3f unified_march_volume(
    const Vec3f& origin, 
    const Vec3f& dir, 
    const UnifiedVolumeParams& params,
    float& out_transmittance,
    float rand_jitter
) {
    Vec3f accumulated_emission(0.0f);
    float T = 1.0f;
    Vec3f p = origin;
    
    // Apply jitter to start position
    p = p + dir * (rand_jitter * params.step_size);
    
    // Safety break for loop
    int max_steps = (params.max_steps > 0 && params.max_steps < 1024) ? params.max_steps : 128;

    for(int i = 0; i < max_steps; ++i) {
        float d = calculate_unified_density(p, params.density_multiplier);
        
        if (d > 0.001f) {
            float sigma_a = params.absorption_prob * d;
            float sigma_s = params.scattering_factor * d;
            float sigma_t = sigma_a + sigma_s;
            
            float step_T = expf(-sigma_t * params.step_size);
            
            // Emission integration (approx)
            Vec3f step_emission = params.emission_color * (d * params.step_size);
            accumulated_emission = accumulated_emission + step_emission * T;
            
            T *= step_T;
            if (T < 0.01f) {
                T = 0.0f;
                break;
            }
        }
        
        p = p + dir * params.step_size;
    }
    
    out_transmittance = T;
    return accumulated_emission;
}
