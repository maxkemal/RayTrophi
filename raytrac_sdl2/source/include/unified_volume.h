#pragma once
#include "unified_types.h"

// =============================================================================
// UNIFIED VOLUME RENDERING
// =============================================================================

// Density source types (VDB-ready)
enum class VolumeDensitySource {
    Constant = 0,    // Uniform density throughout volume
    Procedural = 1,  // Noise-based density (Perlin, etc.)
    VDB = 2          // OpenVDB grid (future)
};

// Unified Volume Data - same structure for CPU and GPU
struct UnifiedVolumeData {
    // Density source
    int density_source;  // VolumeDensitySource as int for GPU compatibility
    
    // Base parameters
    float density;       // Base density multiplier
    float absorption;    // Light absorption rate
    float scattering;    // Scattering coefficient
    
    // Color
    float albedo_r, albedo_g, albedo_b;      // Scattering color
    float emission_r, emission_g, emission_b; // Volume emission
    float emission_intensity;
    
    // Phase function - Primary anisotropy
    float g;  // Forward anisotropy (-1 backward, 0 isotropic, 1 forward)
    
    // Multi-Scattering Parameters (NEW)
    float multi_scatter;   // Multi-scatter contribution (0-1)
    float g_back;          // Backward scattering anisotropy (-1 to 0)
    float lobe_mix;        // Forward/backward lobe mix ratio (0-1, 1=all forward)
    int light_steps;       // Light march steps for self-shadowing (0=disabled)
    float shadow_strength; // Self-shadow intensity (0-1)
    
    // Quality settings
    float step_size;  // Ray march step size
    int max_steps;    // Maximum ray march steps
    
    // Object bounds (AABB)
    float bounds_min_x, bounds_min_y, bounds_min_z;
    float bounds_max_x, bounds_max_y, bounds_max_z;
    
    // Procedural noise parameters
    float noise_scale;
    float noise_detail;
    
    // Future VDB support
    // cudaTextureObject_t vdb_density_tex;
    // float vdb_world_to_index_scale;
    
    // Default constructor
    #ifndef __CUDA_ARCH__
    UnifiedVolumeData() :
        density_source(0),
        density(1.0f), absorption(0.1f), scattering(0.5f),
        albedo_r(1.0f), albedo_g(1.0f), albedo_b(1.0f),
        emission_r(0.0f), emission_g(0.0f), emission_b(0.0f),
        emission_intensity(0.0f),
        g(0.0f),
        multi_scatter(0.3f), g_back(-0.3f), lobe_mix(0.7f),
        light_steps(4), shadow_strength(0.8f),
        step_size(0.1f), max_steps(100),
        bounds_min_x(0.0f), bounds_min_y(0.0f), bounds_min_z(0.0f),
        bounds_max_x(1.0f), bounds_max_y(1.0f), bounds_max_z(1.0f),
        noise_scale(1.0f), noise_detail(1.0f)
    {}
    #endif
};

// Orthonormal Basis Construction (Duff et al.)
UNIFIED_FUNC void build_orthonormal_basis(const Vec3f& n, Vec3f& b1, Vec3f& b2) {
    float sign = copysignf(1.0f, n.z);
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;
    b1 = Vec3f(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = Vec3f(b, sign + n.y * n.y * a, -n.y);
}

// Henyey-Greenstein Phase Function Evaluation
// g: anisotropy parameter (-1 = back, 0 = isotropic, 1 = forward)
UNIFIED_FUNC float phase_henyey_greenstein(float cos_theta, float g) {
    float g2 = g * g;
    float denom = 1.0f + g2 - 2.0f * g * cos_theta;
    return (1.0f - g2) / (4.0f * UnifiedConstants::PI * powf(denom, 1.5f));
}

// Sample Henyey-Greenstein Phase Function
UNIFIED_FUNC Vec3f sample_henyey_greenstein(const Vec3f& wo, float g, float r1, float r2) {
    float cos_theta;
    if (fabsf(g) < 1e-3f) {
        cos_theta = 1.0f - 2.0f * r1;
    } else {
        float sqr_term = (1.0f - g * g) / (1.0f - g + 2.0f * g * r1);
        cos_theta = (1.0f + g * g - sqr_term * sqr_term) / (2.0f * g);
    }
    
    // Safety clamp
    if (cos_theta > 1.0f) cos_theta = 1.0f;
    if (cos_theta < -1.0f) cos_theta = -1.0f;

    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * UnifiedConstants::PI * r2;
    
    Vec3f v1, v2;
    build_orthonormal_basis(wo, v1, v2);

    return v1 * (sin_theta * cosf(phi)) + v2 * (sin_theta * sinf(phi)) + wo * cos_theta;
}

// Homogeneous Volume Transmittance (Beer-Lambert)
// Returns transmittance (0.0 to 1.0) for a given distance and density/sigma_t
UNIFIED_FUNC Vec3f transmittance_homogeneous(float distance, const Vec3f& sigma_t) {
    return Vec3f(
        expf(-sigma_t.x * distance),
        expf(-sigma_t.y * distance),
        expf(-sigma_t.z * distance)
    );
}

// Sample Distance in Homogeneous Medium (Delta Tracking simplifed for constant)
// Returns sampled distance. If infinite, interaction didn't happen.
UNIFIED_FUNC float sample_distance_homogeneous(float sigma_max, float rand_val) {
    return -logf(1.0f - rand_val) / sigma_max;
}

// =============================================================================
// MULTI-SCATTERING SUPPORT (NEW)
// =============================================================================

// Dual-Lobe Henyey-Greenstein Phase Function
// Combines forward and backward scattering lobes for more realistic appearance
// g_forward: primary forward scattering (0.7-0.95 typical)
// g_back: backward scattering for silver lining effect (-0.5 to 0 typical)
// lobe_mix: ratio of forward to backward (0.7 = 70% forward, 30% backward)
UNIFIED_FUNC float phase_dual_henyey_greenstein(float cos_theta, float g_forward, float g_back, float lobe_mix) {
    float phase_fwd = phase_henyey_greenstein(cos_theta, g_forward);
    float phase_bwd = phase_henyey_greenstein(cos_theta, g_back);
    return lobe_mix * phase_fwd + (1.0f - lobe_mix) * phase_bwd;
}

// Multi-Scattering Transmittance Approximation (Frostbite/Disney style)
// Approximates multiple scattering events with a secondary softer extinction
// This prevents volumetrics from looking too dark in dense regions
// multi_scatter_factor: blend between single and multi-scatter (0-1)
// albedo: scattering albedo (higher = more multi-scatter contribution)
UNIFIED_FUNC float compute_multiscatter_transmittance(
    float sigma_t, 
    float distance, 
    float multi_scatter_factor,
    float albedo_avg
) {
    // Primary (single scattering) Beer's law
    float T_single = expf(-sigma_t * distance);
    
    // Secondary (multi-scattering approximation) - softer falloff
    // Uses 0.25x extinction for scattered light that bounces multiple times
    float T_multi = expf(-sigma_t * distance * 0.25f);
    
    // Blend based on albedo and user control
    // High albedo materials scatter more, so they benefit more from multi-scatter
    float blend = multi_scatter_factor * albedo_avg;
    
    return T_single * (1.0f - blend) + T_multi * blend;
}

// Powder Effect for realistic cloud/volume appearance
// Creates brighter edges when light enters the volume (forward scattering peak)
UNIFIED_FUNC float powder_effect_volume(float density, float cos_theta) {
    float powder = 1.0f - expf(-density * 2.0f);
    // Stronger effect for forward scattering
    float forward_bias = 0.5f + 0.5f * fmaxf(0.0f, cos_theta);
    return powder * forward_bias;
}
