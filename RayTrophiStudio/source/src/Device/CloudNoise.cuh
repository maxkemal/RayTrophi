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
// floor_float3 removed (in vec3_utils.cuh)
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

// Improved 3D hash - better distribution, no visible tiling
__device__ inline float3 hash3_improved(float3 p) {
    // Use larger prime multipliers and better mixing
    p = make_float3(
        dot(p, make_float3(127.1f, 311.7f, 74.7f)),
        dot(p, make_float3(269.5f, 183.3f, 246.1f)),
        dot(p, make_float3(113.5f, 271.9f, 124.6f))
    );
    return make_float3(
        -1.0f + 2.0f * frac(sinf(p.x) * 43758.5453f),
        -1.0f + 2.0f * frac(sinf(p.y) * 43758.5453f),
        -1.0f + 2.0f * frac(sinf(p.z) * 43758.5453f)
    );
}

// 3D Gradient Noise (Perlin-style) - NO VISIBLE TILING
__device__ inline float noise3D(float3 x) {
    float3 i = floor_float3(x);
    float3 f = frac(x);
    
    // Quintic interpolation (smoother than smoothstep)
    float3 u = f * f * f * (f * (f * 6.0f - make_float3(15.0f, 15.0f, 15.0f)) + make_float3(10.0f, 10.0f, 10.0f));
    
    // 8 corner gradients
    float3 ga = hash3_improved(i + make_float3(0.0f, 0.0f, 0.0f));
    float3 gb = hash3_improved(i + make_float3(1.0f, 0.0f, 0.0f));
    float3 gc = hash3_improved(i + make_float3(0.0f, 1.0f, 0.0f));
    float3 gd = hash3_improved(i + make_float3(1.0f, 1.0f, 0.0f));
    float3 ge = hash3_improved(i + make_float3(0.0f, 0.0f, 1.0f));
    float3 gf = hash3_improved(i + make_float3(1.0f, 0.0f, 1.0f));
    float3 gg = hash3_improved(i + make_float3(0.0f, 1.0f, 1.0f));
    float3 gh = hash3_improved(i + make_float3(1.0f, 1.0f, 1.0f));
    
    // Distance vectors to corners
    float3 pa = f - make_float3(0.0f, 0.0f, 0.0f);
    float3 pb = f - make_float3(1.0f, 0.0f, 0.0f);
    float3 pc = f - make_float3(0.0f, 1.0f, 0.0f);
    float3 pd = f - make_float3(1.0f, 1.0f, 0.0f);
    float3 pe = f - make_float3(0.0f, 0.0f, 1.0f);
    float3 pf = f - make_float3(1.0f, 0.0f, 1.0f);
    float3 pg = f - make_float3(0.0f, 1.0f, 1.0f);
    float3 ph = f - make_float3(1.0f, 1.0f, 1.0f);
    
    // Dot products
    float va = dot(ga, pa);
    float vb = dot(gb, pb);
    float vc = dot(gc, pc);
    float vd = dot(gd, pd);
    float ve = dot(ge, pe);
    float vf = dot(gf, pf);
    float vg = dot(gg, pg);
    float vh = dot(gh, ph);
    
    // Trilinear interpolation
    return lerp_custom(
        lerp_custom(
            lerp_custom(va, vb, u.x),
            lerp_custom(vc, vd, u.x),
            u.y
        ),
        lerp_custom(
            lerp_custom(ve, vf, u.x),
            lerp_custom(vg, vh, u.x),
            u.y
        ),
        u.z
    ) * 0.5f + 0.5f;  // Normalize to 0-1 range
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
__device__ inline float cloud_shape(float3 p, float coverage, float detail = 1.0f) {
    // === LAYER 1: Base Shape (Low frequency Perlin) ===
    float baseShape = fbm(p * 0.8f, 4);
    
    // === LAYER 2: Worley Cellular Structure ===
    float worleyBase = worley(p * 1.2f);
    float worleyDetail = worleyFbm(p * 2.5f, 3);
    
    float worlyClouds = 1.0f - worleyBase * 0.6f - worleyDetail * 0.3f;
    
    // === LAYER 3: Fine Detail Erosion ===
    // High frequency noise for edge detail - influenced by detail parameter
    float detailNoise = fbm(p * 4.0f, 4);
    float microDetail = fbm(p * 12.0f, 2) * 0.15f * detail;
    
    // === COMBINE LAYERS ===
    float combined = baseShape * worlyClouds;
    combined = combined - detailNoise * 0.15f - microDetail;
    combined = fmaxf(0.0f, combined);
    
    // === COVERAGE REMAP ===
    float threshold = (1.0f - coverage) * 0.55f;
    float density = fmaxf(0.0f, combined - threshold);
    
    // === SOFT EDGE FALLOFF ===
    float edge = fminf(1.0f, density * 4.0f);
    edge = edge * edge;
    density *= edge;
    
    // === DENSITY BOOST ===
    density *= 5.0f;
    
    return density;
}

// ═══════════════════════════════════════════════════════════
// FAST CLOUD SHAPE - Optimized for performance
// Uses fewer octaves for ~5x faster calculation
// ═══════════════════════════════════════════════════════════
__device__ inline float fast_cloud_shape(float3 p, float coverage) {
    // Simplified: Only 2 octave FBM + single Worley
    float baseShape = fbm(p * 0.8f, 2);  // 2 octaves instead of 4
    float worleyBase = 1.0f - worley(p * 1.2f) * 0.7f;
    
    // Simple detail (only 2 octaves)
    float detail = fbm(p * 4.0f, 2) * 0.2f;
    
    float combined = baseShape * worleyBase - detail;
    combined = fmaxf(0.0f, combined);
    
    // Coverage threshold
    float threshold = (1.0f - coverage) * 0.5f;
    float density = fmaxf(0.0f, combined - threshold);
    
    // Soft edge
    density *= fminf(1.0f, density * 4.0f);
    
    return density * 5.0f;
}

// ═══════════════════════════════════════════════════════════
// LOD CLOUD SHAPE - For light marching (lower quality OK)
// ═══════════════════════════════════════════════════════════
__device__ inline float cloud_shape_lod(float3 p, float coverage) {
    // Ultra-simplified: Just 1 octave FBM + hash
    float baseShape = noise3D(p * 0.8f);
    float detail = noise3D(p * 3.0f) * 0.3f;
    
    float combined = baseShape - detail;
    float threshold = (1.0f - coverage) * 0.4f;
    
    return fmaxf(0.0f, (combined - threshold) * 1.5f);
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
