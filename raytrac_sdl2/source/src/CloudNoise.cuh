#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

#include "vec3_utils.cuh" // For float3 operators

// Helper: fractional part
__device__ inline float frac(float x) {
    return x - floorf(x);
}
__device__ inline float3 frac(float3 x) {
    return make_float3(frac(x.x), frac(x.y), frac(x.z));
}
__device__ inline float3 floor_float3(float3 x) {
    return make_float3(floorf(x.x), floorf(x.y), floorf(x.z));
}
__device__ inline float lerp_custom(float a, float b, float t) {
    return a + t * (b - a);
}

// Simple hash function for noise
__device__ inline float hash(float n) {
    n = fmodf(n, 1234.5678f); 
    return frac(sinf(n) * 43758.5453f);
}

__device__ inline float hash33(float3 p) {
    p.x = fmodf(p.x, 50.0f);
    p.y = fmodf(p.y, 50.0f);
    p.z = fmodf(p.z, 50.0f);
    
    float d = dot(p, make_float3(12.9898f, 78.233f, 53.539f));
    return frac(sinf(d) * 43758.5453f);
}

// 3D hash for Worley noise
__device__ inline float3 hash3(float3 p) {
    p = make_float3(
        dot(p, make_float3(127.1f, 311.7f, 74.7f)),
        dot(p, make_float3(269.5f, 183.3f, 246.1f)),
        dot(p, make_float3(113.5f, 271.9f, 124.6f))
    );
    return frac(make_float3(
        sinf(p.x) * 43758.5453f,
        sinf(p.y) * 43758.5453f,
        sinf(p.z) * 43758.5453f
    ));
}

// Worley (Cellular) Noise - creates bubble/cell structures
__device__ inline float worley(float3 p) {
    float3 i = floor_float3(p);
    float3 f = frac(p);
    
    float minDist = 1.0f;
    
    // Check 3x3x3 neighborhood
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                float3 neighbor = make_float3((float)x, (float)y, (float)z);
                float3 point = hash3(i + neighbor);  // Random point in cell
                float3 diff = neighbor + point - f;
                float dist = dot(diff, diff);  // Squared distance
                minDist = fminf(minDist, dist);
            }
        }
    }
    
    return sqrtf(minDist);
}

// 3D Value Noise
__device__ inline float noise3D(float3 x) {
    float3 p = floor_float3(x);
    float3 f = frac(x);
    
    // Smoothstep
    f = f * f * (make_float3(3.0f, 3.0f, 3.0f) - 2.0f * f);

    float n = p.x + p.y * 57.0f + p.z * 113.0f;

    return lerp_custom(
        lerp_custom(
            lerp_custom(hash(n + 0.0f), hash(n + 1.0f), f.x),
            lerp_custom(hash(n + 57.0f), hash(n + 58.0f), f.x), 
            f.y
        ),
        lerp_custom(
            lerp_custom(hash(n + 113.0f), hash(n + 114.0f), f.x),
            lerp_custom(hash(n + 170.0f), hash(n + 171.0f), f.x), 
            f.y
        ), 
        f.z
    );
}

// Fractal Brownian Motion (FBM) for cloud detail
__device__ inline float fbm(float3 p, int octaves) {
    float f = 0.0f;
    float w = 0.5f;
    for (int i = 0; i < octaves; i++) {
        f += w * noise3D(p);
        p *= 2.0f;
        w *= 0.5f;
    }
    return f;
}

// Worley FBM - creates cellular cloud structures
__device__ inline float worleyFbm(float3 p, int octaves) {
    float f = 0.0f;
    float w = 0.5f;
    for (int i = 0; i < octaves; i++) {
        f += w * worley(p);
        p *= 2.0f;
        w *= 0.5f;
    }
    return f;
}

// ═══════════════════════════════════════════════════════════
// CINEMATIC CLOUD SHAPE - Maya/Arnold/Houdini quality
// Uses combination of Perlin FBM + Worley for realistic clouds
// ═══════════════════════════════════════════════════════════
__device__ inline float cloud_shape(float3 p, float coverage) {
    // === LAYER 1: Base Shape (Low frequency Perlin) ===
    // This defines the overall cloud mass
    float baseShape = fbm(p * 0.8f, 4);
    
    // === LAYER 2: Worley Cellular Structure ===
    // Creates the characteristic "puffy" cloud look
    float worleyBase = worley(p * 1.2f);
    float worleyDetail = worleyFbm(p * 2.5f, 3);
    
    // Invert worley (1 - worley) to get puffy centers instead of hollow cells
    float worlyClouds = 1.0f - worleyBase * 0.6f - worleyDetail * 0.3f;
    
    // === LAYER 3: Fine Detail Erosion ===
    // High frequency noise for edge detail
    float detailNoise = fbm(p * 4.0f, 4);
    float microDetail = fbm(p * 12.0f, 2) * 0.1f;
    
    // === COMBINE LAYERS ===
    // Base Perlin * Worley structure - Detail erosion
    float combined = baseShape * worlyClouds;
    combined = combined - detailNoise * 0.25f - microDetail;
    combined = fmaxf(0.0f, combined);
    
    // === COVERAGE REMAP ===
    // Smooth threshold based on coverage
    float threshold = (1.0f - coverage) * 0.55f;
    float density = fmaxf(0.0f, combined - threshold);
    
    // === SOFT EDGE FALLOFF ===
    // Prevents harsh transitions
    float edge = fminf(1.0f, density * 4.0f);
    edge = edge * edge;  // Quadratic falloff
    density *= edge;
    
    // === DENSITY BOOST ===
    // Make clouds more substantial
    density *= 1.5f;
    
    return density;
}

// ═══════════════════════════════════════════════════════════
// POWDER EFFECT - Light scattering at cloud edges
// Creates the characteristic bright rim effect
// ═══════════════════════════════════════════════════════════
__device__ inline float powderEffect(float density, float cosTheta) {
    // Beer-Powder approximation
    float beer = expf(-density * 2.0f);
    float powder = 1.0f - expf(-density * 4.0f);
    
    // More powder effect when looking towards the sun
    float sunFactor = (1.0f + cosTheta) * 0.5f;  // 0 to 1
    
    return lerp_custom(beer, powder * beer, sunFactor * 0.5f);
}
