#pragma once
#include "unified_types.h"

// =============================================================================
// UNIFIED VOLUME RENDERING
// =============================================================================

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
