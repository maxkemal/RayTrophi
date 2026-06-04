/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          hair_bsdf.cuh
 * Description:   GPU Hair BSDF Implementation (CUDA)
 *                Marschner model optimized for real-time path tracing
 * =========================================================================
 */
#pragma once

#include "vec3_utils.cuh"
#include "random_utils.cuh"

namespace HairGPU {

// ============================================================================
// Constants
// ============================================================================

#define HAIR_PI 3.14159265358979323846f
#define HAIR_INV_PI 0.31830988618379067154f
#define HAIR_SQRT2_INV 0.70710678118654752440f

// Precomputed Gaussian integrals for energy conservation
__constant__ float HAIR_COS_THETA_TABLE[256];

// ============================================================================
// GPU Hair Material Structure
// ============================================================================

struct alignas(16) GpuHairMaterial {
    // Block 1: Color & Appearance (16 bytes)
    float3 sigma_a;               // Absorption coefficient (12 bytes)
    int colorMode;                // Color mode (4 bytes)

    // Block 2: Physically Based Model (16 bytes)
    float3 color;                 // Direct color (12 bytes)
    float roughness;              // Longitudinal roughness (4 bytes)

    // Block 3: Physical Properties (16 bytes)
    float melanin;                // Melanin amount (4 bytes)
    float melaninRedness;         // Melanin redness (4 bytes)
    float ior;                    // Index of refraction (4 bytes)
    float cuticleAngle;           // In radians (4 bytes)

    // Block 4: Styling & Tint (16 bytes)
    float3 tintColor;             // Tint color (12 bytes)
    float tint;                   // Tint strength (4 bytes)

    // Block 5: Azimuthal & Random (16 bytes)
    float radialRoughness;        // Azimuthal roughness (4 bytes)
    float randomHue;              // Variation (4 bytes)
    float randomValue;            // Variation (4 bytes)
    float emissionStrength;       // (4 bytes)

    // Block 6: Emission & Variances (16 bytes)
    float3 emission;              // (12 bytes)
    float v_R;                    // Variance for R lobe (4 bytes)

    // Block 7: More Variances & Misc (16 bytes)
    float v_TT;                   // Variance for TT lobe (4 bytes)
    float v_TRT;                  // Variance for TRT lobe (4 bytes)
    float s_R;                    // Logistic scale for R lobe (4 bytes)
    float s_TT;                   // Logistic scale for TT lobe (4 bytes)

    // Block 8: More Azimuthal & Textures (16 bytes)
    float s_TRT;                  // Logistic scale for TRT lobe (4 bytes)
    float s_MS;                   // Logistic scale for MS lobe (4 bytes)
    cudaTextureObject_t albedo_tex;    // 8 bytes
    
    // Block 9: Remaining Textures (16 bytes)
    cudaTextureObject_t roughness_tex; // 8 bytes
    float coat;                   // Coat strength for fur (4 bytes)
    float specularTint;           // Tint primary highlight by hair color (4 bytes)

    // Block 10: Coat & Gradient (16 bytes)
    float3 coatTint;              // Coat reflection tint (12 bytes)
    float diffuseSoftness;        // MS weight: 0=hard specular, 1=soft diffuse (4 bytes)

    // Block 11: Root-Tip Gradient (16 bytes)
    float3 tipSigma;              // Absorption at tip (12 bytes)
    float rootTipBalance;         // 0=root color, 1=tip color at tip (4 bytes)

    // Block 12: Flags & Padding (16 bytes)
    int enableRootTipGradient;    // 0 or 1 (4 bytes)
    float pad1;                   // (4 bytes)
    float pad2;                   // (4 bytes)
    float pad3;                   // (4 bytes)
};

// ============================================================================
// Helper Functions
// ============================================================================

__device__ __forceinline__ float sqr(float x) { return x * x; }

// Overload make_float3 for single scalar (broadcast)
__device__ __forceinline__ float3 make_float3_scalar(float v) {
    return make_float3(v, v, v);
}

__device__ __forceinline__ float safe_sqrt(float x) { 
    return sqrtf(fmaxf(0.0f, x)); 
}

__device__ __forceinline__ float safe_asin(float x) {
    return asinf(fminf(fmaxf(x, -1.0f), 1.0f));
}

// Schlick Fresnel approximation
__device__ __forceinline__ float fresnel_schlick(float cosTheta, float f0) {
    float x = 1.0f - cosTheta;
    float x2 = x * x;
    return f0 + (1.0f - f0) * x2 * x2 * x;
}

// Exact Fresnel for dielectric
__device__ __forceinline__ float fresnel_dielectric(float cosThetaI, float eta) {
    float sinThetaI = safe_sqrt(1.0f - cosThetaI * cosThetaI);
    float sinThetaT = sinThetaI / eta;
    
    if (sinThetaT >= 1.0f) return 1.0f; // Total internal reflection
    
    float cosThetaT = safe_sqrt(1.0f - sinThetaT * sinThetaT);
    
    float rs = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float rp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
    
    return 0.5f * (rs * rs + rp * rp);
}

// Melanin to Absorption Coefficient mapping (Synchronized with HairBSDF.cpp)
__device__ __forceinline__ float3 melanin_to_absorption(float melanin, float redness) {
    // Eumelanin is the primary darkness. Redness introduces pheomelanin.
    // Instead of (1 - redness), use a softer blend so high melanin stays dark.
    float eumelanin = melanin * (1.0f - redness * 0.5f); 
    float pheomelanin = melanin * redness; 
    
    // Base absorption coefficients per mm (based on physical measurements)
    // Eumelanin: Absorbs all wavelengths, slightly more in blue (brown/black)
    float3 eumelaninSigma = make_float3(0.419f, 0.697f, 1.37f);     
    
    // Pheomelanin: Absorbs mostly blue/green, reflects red (red/blonde)
    float3 pheomelaninSigma = make_float3(0.187f, 0.4f, 1.05f);    
    
    // Total absorption (scaled for fiber scale)
    // Adjusted multipliers to match visual expectations
    return eumelaninSigma * eumelanin * 8.0f + pheomelaninSigma * pheomelanin * 8.0f;
}

// Gaussian distribution
__device__ __forceinline__ float gaussian(float x, float variance) {
    return expf(-x * x / (2.0f * variance)) / sqrtf(2.0f * HAIR_PI * variance);
}

// Logistic distribution (faster, used in Disney 2016)
__device__ __forceinline__ float logistic_pdf(float x, float s) {
    float exp_x = expf(-fabsf(x) / s);
    return exp_x / (s * sqr(1.0f + exp_x));
}

__device__ __forceinline__ float sample_logistic(float s, float u) {
    return -s * logf(1.0f / u - 1.0f);
}

// ============================================================================
// Hair Coordinate System
// ============================================================================

// Convert world directions to hair-local (u, v) coordinates
// u: azimuthal angle around hair
// theta: longitudinal angle relative to perpendicular plane
__device__ __forceinline__ void world_to_hair_coords(
    const float3& wo, 
    const float3& wi,
    const float3& tangent,
    float& sinThetaO, float& cosThetaO,
    float& sinThetaI, float& cosThetaI,
    float& cosPhi, float& sinPhi
) {
    // Project onto tangent to get longitudinal component
    sinThetaO = dot(wo, tangent);
    sinThetaI = dot(wi, tangent);
    
    cosThetaO = safe_sqrt(1.0f - sinThetaO * sinThetaO);
    cosThetaI = safe_sqrt(1.0f - sinThetaI * sinThetaI);
    
    // Perpendicular plane projections
    float3 wo_perp = normalize(wo - sinThetaO * tangent);
    float3 wi_perp = normalize(wi - sinThetaI * tangent);
    
    cosPhi = dot(wo_perp, wi_perp);
    sinPhi = dot(cross(wo_perp, wi_perp), tangent); // Use cross product for robust signed sine
}

// ============================================================================
// Marschner Model Lobes
// ============================================================================

// R lobe (primary specular reflection)
__device__ __forceinline__ float eval_M_R(
    float sinThetaI, float sinThetaO,
    float cosThetaI, float cosThetaO,
    float alpha, float variance
) {
    float sinThetaSum = sinThetaI + sinThetaO - 2.0f * alpha;
    return gaussian(sinThetaSum, variance);
}

// TT lobe (transmission)  
__device__ __forceinline__ float eval_M_TT(
    float sinThetaI, float sinThetaO,
    float cosThetaI, float cosThetaO,
    float alpha, float variance
) {
    float sinThetaSum = sinThetaI + sinThetaO + alpha;
    return gaussian(sinThetaSum, 0.5f * variance);
}

// TRT lobe (internal reflection)
__device__ __forceinline__ float eval_M_TRT(
    float sinThetaI, float sinThetaO,
    float cosThetaI, float cosThetaO,
    float alpha, float variance
) {
    float sinThetaSum = sinThetaI + sinThetaO - 4.0f * alpha;
    return gaussian(sinThetaSum, 2.0f * variance);
}

// Azimuthal scattering (N term)
__device__ __forceinline__ float eval_N(float phi, float s, float phiTarget) {
    float diff = phi - phiTarget;
    // Faster wrap to [-PI, PI]
    diff -= 2.0f * HAIR_PI * floorf(diff * (0.5f * HAIR_INV_PI) + 0.5f);
    
    // Energy conservation: Normalize the logistic distribution over [-PI, PI]
    // The integral of logistic PDF from -PI to PI is (CDF(PI) - CDF(-PI))
    float norm = (1.0f / (1.0f + expf(-HAIR_PI / s))) - (1.0f / (1.0f + expf(HAIR_PI / s)));
    
    return logistic_pdf(diff, s) / fmaxf(norm, 0.1f);
}

// ============================================================================
// Main BSDF Evaluation
// ============================================================================

__device__ float3 hair_bsdf_eval(
    const float3& wo,           // Outgoing (toward camera)
    const float3& wi,           // Incoming (toward light)
    const float3& tangent,      // Hair direction
    const GpuHairMaterial& mat,
    float h,                    // Hit offset from center [-1, 1]
    float v = 0.5f              // Position along strand (0=root, 1=tip)
) {
    // 1. Convert to hair coordinates
    float sinThetaO = dot(wo, tangent);
    float sinThetaI = dot(wi, tangent);
    
    float cosThetaO = safe_sqrt(1.0f - sinThetaO * sinThetaO);
    float cosThetaI = safe_sqrt(1.0f - sinThetaI * sinThetaI);
    
    // Early exit for grazing angles
    if (cosThetaO < 1e-4f || cosThetaI < 1e-4f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    // Azimuthal planes
    float3 wo_perp = normalize(wo - sinThetaO * tangent);
    float3 wi_perp = normalize(wi - sinThetaI * tangent);
    
    float cosPhi = dot(wo_perp, wi_perp);
    float sinPhi = dot(cross(wo_perp, wi_perp), tangent);
    float phi = atan2f(sinPhi, cosPhi);
    
    // Material parameters
    float alpha = mat.cuticleAngle;
    float eta = mat.ior;
    float3 sigma_a = mat.sigma_a;
    
    // --- Root-to-Tip Gradient ---
    if (mat.enableRootTipGradient) {
        float t = v * mat.rootTipBalance;
        sigma_a = sigma_a * (1.0f - t) + mat.tipSigma * t;
    }
    
    // Compute gamma angles for absorption path length and azimuthal shifts
    float gammaO = safe_asin(h);
    float gammaT = safe_asin(h / eta);
    
    float cosGammaO = cosf(gammaO);
    float cosGammaT = cosf(gammaT);
    
    // ========================================================================
    // Fresnel
    // ========================================================================
    float F_R = fresnel_dielectric(cosThetaO, eta);
    float M_R = eval_M_R(sinThetaI, sinThetaO, cosThetaO, cosThetaI, alpha, mat.v_R); 
    
    float phi_R = -2.0f * gammaO;
    float N_R = eval_N(phi, mat.s_R, phi_R);
    
    // --- Specular Tint: blend white highlight with hair body color ---
    float3 hairBodyColor = make_float3(
        expf(-sigma_a.x * 0.5f),
        expf(-sigma_a.y * 0.5f),
        expf(-sigma_a.z * 0.5f)
    );
    float3 specColor = make_float3(1.0f, 1.0f, 1.0f) * (1.0f - mat.specularTint) 
                     + hairBodyColor * mat.specularTint;
    float3 R = specColor * (F_R * M_R * N_R);
    
    // ========================================================================
    // TT Lobe (Transmission - Back-lit highlight)
    // ========================================================================
    float cosThetaD = cosf((asinf(sinThetaO) - asinf(sinThetaI)) * 0.5f);
    float path_L = 2.0f * cosGammaT / fmaxf(cosThetaD, 0.1f);
    float3 A = make_float3(expf(-sigma_a.x * path_L),
                           expf(-sigma_a.y * path_L),
                           expf(-sigma_a.z * path_L));
    
    float F_TT = (1.0f - F_R) * (1.0f - fresnel_dielectric(cosGammaT, 1.0f / eta));
    float M_TT = eval_M_TT(sinThetaI, sinThetaO, cosThetaO, cosThetaI, alpha, mat.v_TT); 
    
    float phi_TT = HAIR_PI + 2.0f * gammaT - 2.0f * gammaO;
    float N_TT = eval_N(phi, mat.s_TT, phi_TT);
    float3 TT = A * (F_TT * M_TT * N_TT); 
    
    // ========================================================================
    // TRT Lobe (Internal Reflection - Front-lit colored specular)
    // ========================================================================
    float F_internal = fresnel_dielectric(cosGammaT, 1.0f / eta);
    float F_TRT = (1.0f - F_R) * F_internal * (1.0f - fresnel_dielectric(cosGammaT, 1.0f / eta));
    float M_TRT = eval_M_TRT(sinThetaI, sinThetaO, cosThetaO, cosThetaI, alpha, mat.v_TRT); 
    
    float phi_TRT = 4.0f * gammaT - 2.0f * gammaO; 
    float N_TRT = eval_N(phi, mat.s_TRT, phi_TRT); 
    float3 TRT = (A * A) * (F_TRT * M_TRT * N_TRT);
    
    // ========================================================================
    // Multiple Scattering / Cortex Diffusion (Bulk Body Color)
    // ========================================================================
    float N_MS = eval_N(phi, mat.s_MS, 0.0f); 
    // --- Diffuse Softness controls MS weight ---
    float msWeight = mat.diffuseSoftness * 1.2f;
    float3 MS = (A * A) * (msWeight * N_MS); 
    
    // Combine lobes
    float3 bsdf = R + TT + TRT + MS;
    
    // --- Apply Artistic Tint ---
    if (mat.tint > 0.0f) {
        float3 tinted = bsdf * mat.tintColor;
        bsdf = bsdf * (1.0f - mat.tint) + tinted * mat.tint;
    }
    
    // --- Apply Coat Layer (wet look / fur gloss) ---
    // Coat = additional specular lobe from water/gel film on hair surface
    if (mat.coat > 0.0f) {
        float coatIOR = 1.33f; // Water film IOR
        float coatFresnel = fresnel_dielectric(cosThetaO, coatIOR) * mat.coat;
        
        // Coat specular: narrow Gaussian lobe (smooth water surface)
        float coatRoughness = fmaxf(mat.roughness * 0.3f, 0.02f);
        float coatVariance = coatRoughness * coatRoughness;
        float sinThetaSum_coat = sinThetaI + sinThetaO; // No cuticle shift for coat
        float M_coat = gaussian(sinThetaSum_coat, coatVariance);
        
        float phi_coat = -2.0f * gammaO; // Same as R lobe reflection angle
        float s_coat = fmaxf(coatRoughness * 0.8f * 0.5f, 0.01f);
        float N_coat = eval_N(phi, s_coat, phi_coat);
        
        float3 coatSpec = mat.coatTint * (coatFresnel * M_coat * N_coat);
        
        // Energy conservation: dim base BSDF by coat reflection
        bsdf = bsdf * (1.0f - coatFresnel) + coatSpec;
    }
    
    // Add emission if present
    if (mat.emissionStrength > 0.0f) {
        bsdf = bsdf + mat.emission * mat.emissionStrength;
    }
    
    // Final normalization
    float denominator = fmaxf(cosThetaD * cosThetaD, 0.001f);
    bsdf = bsdf / denominator;
    
    // Firefly clamp (matching CPU)
    bsdf.x = fminf(bsdf.x, 100.0f);
    bsdf.y = fminf(bsdf.y, 100.0f);
    bsdf.z = fminf(bsdf.z, 100.0f);
    
    return bsdf; 
}

// ============================================================================
// Importance Sampling
// ============================================================================

__device__ float3 hair_bsdf_sample(
    const float3& wo,
    const float3& tangent,
    const GpuHairMaterial& mat,
    float u1, float u2, float u3,
    float3& out_wi,
    float& out_pdf
) {
    // Choose lobe based on energy distribution (Matching CPU sample weights)
    float lobe_weights[3] = {0.4f, 0.3f, 0.3f};  // R, TT, TRT
    
    int lobe;
    if (u1 < lobe_weights[0]) {
        lobe = 0;
        u1 /= lobe_weights[0];
    } else if (u1 < lobe_weights[0] + lobe_weights[1]) {
        lobe = 1;
        u1 = (u1 - lobe_weights[0]) / lobe_weights[1];
    } else {
        lobe = 2;
        u1 = (u1 - lobe_weights[0] - lobe_weights[1]) / lobe_weights[2];
    }
    
    float alpha = mat.cuticleAngle;
    
    // Sample longitudinal (M term)
    float sinThetaO = dot(wo, tangent);
    
    float variance;
    float alphaShift;
    switch (lobe) {
        case 0: variance = mat.v_R;        alphaShift = -2.0f * alpha; break;
        case 1: variance = 0.5f * mat.v_TT;  alphaShift = alpha; break; 
        case 2: variance = 2.0f * mat.v_TRT; alphaShift = -4.0f * alpha; break; 
        default: variance = mat.v_R;  alphaShift = 0.0f;
    }
    
    // Sample from Gaussian (Simplified for GPU)
    float sinThetaI = sinThetaO + alphaShift + sqrtf(variance) * (2.0f * u1 - 1.0f);
    sinThetaI = fminf(fmaxf(sinThetaI, -0.999f), 0.999f);
    float cosThetaI = safe_sqrt(1.0f - sinThetaI * sinThetaI);
    
    // Sample azimuthal (N term) - Matching CPU Lobe Targets
    float phiTarget = (lobe == 1) ? HAIR_PI : 0.0f; // TT is forward-scattering (PI), R/TRT are backward (0)
    float lobeS = (lobe == 0) ? mat.s_R : (lobe == 1 ? mat.s_TT : mat.s_TRT);
    float phi = sample_logistic(fmaxf(lobeS, 0.01f), u2) + phiTarget;
    
    // Reconstruct wi from spherical coords relative to tangent
    float3 wo_perp = normalize(wo - sinThetaO * tangent);
    float3 bitangent = normalize(cross(tangent, wo_perp));
    
    out_wi = sinThetaI * tangent + 
             cosThetaI * (cosf(phi) * wo_perp + sinf(phi) * bitangent);
    out_wi = normalize(out_wi);
    
    // Compute PDF matching sampling distribution
    float M = gaussian(sinThetaI - sinThetaO - alphaShift, variance);
    float N = logistic_pdf(fmodf(phi - phiTarget + HAIR_PI, 2.0f * HAIR_PI) - HAIR_PI, fmaxf(lobeS, 0.01f));
    float norm = (1.0f / (1.0f + expf(-HAIR_PI / fmaxf(lobeS, 0.01f)))) - (1.0f / (1.0f + expf(HAIR_PI / fmaxf(lobeS, 0.01f))));
    out_pdf = fmaxf(M * (N / fmaxf(norm, 0.1f)) * lobe_weights[lobe], 0.0001f);
    
    // Evaluate full BSDF with correct hit information (center hit assumed for sampling)
    float3 bsdf = hair_bsdf_eval(wo, out_wi, tangent, mat, 0.0f);
    
    return bsdf;
}

// ============================================================================
// Melanin to Absorption (Physical Hair Color)
// ============================================================================


} // namespace HairGPU
