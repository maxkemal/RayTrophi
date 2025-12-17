/**
 * @file unified_brdf.h
 * @brief Shared BRDF functions for CPU and GPU rendering parity
 * 
 * These functions implement the core BRDF calculations used by both
 * CPU and GPU renderers. The code is identical on both platforms,
 * ensuring consistent render results.
 * 
 * Reference: GPU implementation (OptiX) is considered the ground truth.
 */
#pragma once

#include "unified_types.h"

// =============================================================================
// GGX MICROFACET DISTRIBUTION
// =============================================================================

/**
 * @brief GGX/Trowbridge-Reitz Normal Distribution Function
 * @param NdotH Dot product of normal and half-vector
 * @param roughness Material roughness [0-1]
 * @return Distribution value
 */
UNIFIED_FUNC float D_GGX(float NdotH, float roughness) {
    float alpha = fmaxf(roughness * roughness, UnifiedConstants::MIN_ALPHA);
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (UnifiedConstants::PI * denom * denom);
}

// =============================================================================
// GEOMETRY FUNCTIONS (MASKING/SHADOWING)
// =============================================================================

/**
 * @brief Schlick-GGX Geometry function (single direction)
 * @param NdotX Dot product of normal and direction (V or L)
 * @param roughness Material roughness
 * @return Geometry term for one direction
 */
UNIFIED_FUNC float G_SchlickGGX(float NdotX, float roughness) {
    // Industry standard for Direct Lighting (Analytic Lights):
    // k = (roughness + 1)^2 / 8
    // This provides softer, more natural looking highlights than alpha/2 (which is for IBL)
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    return NdotX / (NdotX * (1.0f - k) + k);
}

/**
 * @brief Smith Geometry function (masking-shadowing for both directions)
 * @param NdotV Dot product of normal and view direction
 * @param NdotL Dot product of normal and light direction
 * @param roughness Material roughness
 * @return Combined geometry term
 */
UNIFIED_FUNC float G_Smith(float NdotV, float NdotL, float roughness) {
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

// =============================================================================
// FRESNEL FUNCTIONS
// =============================================================================

/**
 * @brief Schlick Fresnel approximation
 * @param cosTheta Cosine of angle between view and half-vector
 * @param F0 Reflectance at normal incidence
 * @return Fresnel term
 */
UNIFIED_FUNC Vec3f F_Schlick(float cosTheta, const Vec3f& F0) {
    float f = powf(1.0f - cosTheta, 5.0f);
    return F0 + (Vec3f(1.0f) - F0) * f;
}

/**
 * @brief Schlick Fresnel with roughness (for environment mapping)
 * @param cosTheta Cosine of angle
 * @param F0 Reflectance at normal incidence
 * @param roughness Material roughness
 * @return Fresnel term adjusted for roughness
 */
UNIFIED_FUNC Vec3f F_SchlickRoughness(float cosTheta, const Vec3f& F0, float roughness) {
    Vec3f max_val = fmaxf(Vec3f(1.0f - roughness), F0);
    float f = powf(1.0f - cosTheta, 5.0f);
    return F0 + (max_val - F0) * f;
}

/**
 * @brief Schlick reflectance for dielectrics
 * @param cos_theta Cosine of incident angle
 * @param eta Ratio of indices of refraction
 * @return Reflectance probability
 */
UNIFIED_FUNC float schlick_reflectance(float cos_theta, float eta) {
    float r0 = (1.0f - eta) / (1.0f + eta);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);
}

// =============================================================================
// BRDF EVALUATION
// =============================================================================

/**
 * @brief Evaluate full Cook-Torrance BRDF
 * 
 * This is the core BRDF function used by both CPU and GPU.
 * It computes both diffuse and specular components with energy conservation.
 * 
 * @param N Surface normal (normalized)
 * @param V View direction (normalized, pointing away from surface)
 * @param L Light direction (normalized, pointing toward light)
 * @param albedo_sampled Albedo color (possibly from texture)
 * @param roughness_sampled Roughness value (possibly from texture)
 * @param metallic_sampled Metallic value (possibly from texture)
 * @param transmission Material transmission (0 = opaque, 1 = fully transmissive)
 * @param rand_val Random value [0,1] for stochastic transmission selection
 * @return BRDF value (not multiplied by NdotL)
 */
UNIFIED_FUNC Vec3f evaluate_brdf_unified(
    const Vec3f& N,
    const Vec3f& V,
    const Vec3f& L,
    const Vec3f& albedo_sampled,
    float roughness_sampled,
    float metallic_sampled,
    float transmission = 0.0f,
    float rand_val = 0.5f
) {
    // Half vector
    Vec3f H = normalize(V + L);
    
    // Dot products (clamped to avoid divide by zero)
    float NdotV = fmaxf(dot(N, V), UnifiedConstants::MIN_DOT);
    float NdotL = fmaxf(dot(N, L), UnifiedConstants::MIN_DOT);
    float NdotH = fmaxf(dot(N, H), UnifiedConstants::MIN_DOT);
    float VdotH = fmaxf(dot(V, H), UnifiedConstants::MIN_DOT);
    
    // GPU logic matching: No early exit for grazing angles here.
    // GPU handles small NdotV/NdotL robustly with epsilons in the denominator.
    // Removing the early exit prevents black artifacts at grazing angles.
    
    // Clamp albedo to valid range
    Vec3f albedo = albedo_sampled.clamp(0.01f, 1.0f);
    
    // F0 - Reflectance at normal incidence
    // Dielectrics have ~0.04, metals use albedo
    Vec3f F0 = lerp(Vec3f(0.04f), albedo, metallic_sampled);
    
    // Fresnel term
    Vec3f F = F_Schlick(VdotH, F0);
    
    // Distribution term
    float D = D_GGX(NdotH, roughness_sampled);
    
    // Geometry term
    float G = G_Smith(NdotV, NdotL, roughness_sampled);
    
    // Specular BRDF (Cook-Torrance)
    Vec3f specular = (F * D * G) / (4.0f * NdotV * NdotL + 0.001f);
    
    // Energy conservation for diffuse
    // Average Fresnel approximation for hemisphere
    Vec3f F_avg = F0 + (Vec3f(1.0f) - F0) / 21.0f;
    
    // Diffuse coefficient (metals have no diffuse)
    Vec3f k_d = (Vec3f(1.0f) - F_avg) * (1.0f - metallic_sampled);
    
    // Lambertian diffuse
    Vec3f diffuse = k_d * albedo * UnifiedConstants::INV_PI;
    
    // GPU-matching transmission handling:
    // When transmission >= 0.01, stochastically choose diffuse or specular
    if (transmission >= 0.01f) {
        if (rand_val > transmission) {
            return diffuse;
        }
        return specular;
    }
    
    return diffuse + specular;
}

// =============================================================================
// PDF CALCULATION
// =============================================================================


UNIFIED_FUNC float pdf_brdf_unified(
    const Vec3f& N,
    const Vec3f& V,
    const Vec3f& L,
    float roughness
) {
    Vec3f H = normalize(V + L);
    float NdotH = fmaxf(dot(N, H), 0.0001f);
    float VdotH = fmaxf(dot(V, H), 0.0001f);
    
    float alpha = fmaxf(roughness * roughness, UnifiedConstants::MIN_ALPHA);
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0f) + 1.0f;
    float D = alpha2 / (UnifiedConstants::PI * denom * denom);
    
    return D * NdotH / (4.0f * VdotH + 1e-4f);
}

// =============================================================================
// MIS WEIGHT CALCULATION
// =============================================================================

/**
 * @brief Power heuristic for Multiple Importance Sampling
 * 
 * Uses power of 2 (beta=2) which is standard in path tracing.
 * 
 * @param pdf_a First PDF
 * @param pdf_b Second PDF
 * @return MIS weight for strategy A
 */
UNIFIED_FUNC float power_heuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-4f);
}

/**
 * @brief Balance heuristic for MIS
 * @param pdf_a First PDF
 * @param pdf_b Second PDF
 * @return MIS weight for strategy A
 */
UNIFIED_FUNC float balance_heuristic(float pdf_a, float pdf_b) {
    return pdf_a / (pdf_a + pdf_b + 1e-4f);
}

// =============================================================================
// GGX IMPORTANCE SAMPLING
// =============================================================================

/**
 * @brief Sample a microfacet normal from GGX distribution
 * 
 * @param u1 Random value [0,1]
 * @param u2 Random value [0,1]
 * @param roughness Material roughness
 * @param N Surface normal
 * @return Sampled half-vector in world space
 */
UNIFIED_FUNC Vec3f importance_sample_ggx(float u1, float u2, float roughness, const Vec3f& N) {
    float alpha = roughness * roughness;
    
    // Sample spherical coordinates
    float phi = 2.0f * UnifiedConstants::PI * u1;
    float cosTheta = sqrtf((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sinTheta = sqrtf(fmaxf(1.0f - cosTheta * cosTheta, 0.0f));
    
    // Half vector in tangent space
    Vec3f H_tangent(
        cosf(phi) * sinTheta,
        sinf(phi) * sinTheta,
        cosTheta
    );
    
    // Build tangent frame
    Vec3f up = fabsf(N.z) < 0.999f ? Vec3f(0.0f, 0.0f, 1.0f) : Vec3f(1.0f, 0.0f, 0.0f);
    Vec3f tangentX = normalize(cross(up, N));
    Vec3f tangentY = cross(N, tangentX);
    
    // Transform to world space
    return normalize(tangentX * H_tangent.x + tangentY * H_tangent.y + N * H_tangent.z);
}

// =============================================================================
// REFRACTION
// =============================================================================

/**
 * @brief Compute refracted direction using Snell's law
 * 
 * @param V Incident direction (pointing toward surface)
 * @param N Surface normal (pointing outward)
 * @param eta Ratio of indices of refraction (n1/n2)
 * @param refracted Output refracted direction
 * @return True if refraction is possible (no total internal reflection)
 */
UNIFIED_FUNC bool refract_vec(const Vec3f& V, const Vec3f& N, float eta, Vec3f* refracted) {
    float cos_theta = fminf(dot(-V, N), 1.0f);
    Vec3f r_out_perp = eta * (V + cos_theta * N);
    float k = 1.0f - dot(r_out_perp, r_out_perp);
    if (k < 0.0f) return false;
    Vec3f r_out_parallel = -sqrtf(k) * N;
    *refracted = r_out_perp + r_out_parallel;
    return true;
}

// =============================================================================
// LUMINANCE & COLOR UTILITIES
// =============================================================================

/**
 * @brief Calculate luminance of color
 * @param c Color
 * @return Luminance value
 */
UNIFIED_FUNC float luminance(const Vec3f& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

/**
 * @brief Clamp contribution to prevent fireflies
 * @param contribution Color contribution
 * @param max_value Maximum allowed luminance
 * @return Clamped contribution
 */
UNIFIED_FUNC Vec3f clamp_contribution(const Vec3f& contribution, float max_value) {
    float lum = contribution.luminance();
    if (lum > max_value) {
        return contribution * (max_value / lum);
    }
    return contribution;
}

// =============================================================================
// RUSSIAN ROULETTE
// =============================================================================

/**
 * @brief Calculate Russian Roulette survival probability
 * @param throughput Current path throughput
 * @return Survival probability (clamped to valid range)
 */
UNIFIED_FUNC float russian_roulette_probability(const Vec3f& throughput) {
    float p = throughput.max_component();
    return fminf(fmaxf(p, UnifiedConstants::RR_MIN_PROB), UnifiedConstants::RR_MAX_PROB);
}

// =============================================================================
// BACKGROUND CONTRIBUTION
// =============================================================================

/**
 * @brief Calculate background contribution factor based on bounce
 * 
 * First bounce gets full background, subsequent bounces are attenuated
 * to prevent background from "painting" surfaces through reflections.
 * 
 * @param bounce Current bounce number (0 = primary ray)
 * @return Background contribution multiplier
 */
UNIFIED_FUNC float background_factor(int bounce) {
    if (bounce == 0) return 1.0f;
    return fmaxf(
        UnifiedConstants::BG_FALLOFF_MIN,
        1.0f / (1.0f + bounce * UnifiedConstants::BG_FALLOFF_RATE)
    );
}
