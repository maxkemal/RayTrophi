/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          unified_noise.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include "unified_types.h"

// =============================================================================
// UNIFIED PROCEDURAL NOISE (CPU & GPU)
// =============================================================================

// Pseudo-random hash function
UNIFIED_FUNC float hash13(const Vec3f& p) {
    // Fract(Sin(Dot(...))) method - common in shaders
    float h = dot(p, Vec3f(127.1f, 311.7f, 74.7f));
    return sinf(h) * 43758.5453123f - floorf(sinf(h) * 43758.5453123f);
}

// 3D Value Noise for volumetric density
UNIFIED_FUNC float value_noise(const Vec3f& p) {
    Vec3f i = Vec3f(floorf(p.x), floorf(p.y), floorf(p.z));
    Vec3f f = Vec3f(p.x - i.x, p.y - i.y, p.z - i.z);
    
    // Cubic interpolation (smoothstep)
    Vec3f u = f * f * (Vec3f(3.0f) - f * 2.0f);

    float n000 = hash13(i);
    float n100 = hash13(i + Vec3f(1.0f, 0.0f, 0.0f));
    float n010 = hash13(i + Vec3f(0.0f, 1.0f, 0.0f));
    float n110 = hash13(i + Vec3f(1.0f, 1.0f, 0.0f));
    float n001 = hash13(i + Vec3f(0.0f, 0.0f, 1.0f));
    float n101 = hash13(i + Vec3f(1.0f, 0.0f, 1.0f));
    float n011 = hash13(i + Vec3f(0.0f, 1.0f, 1.0f));
    float n111 = hash13(i + Vec3f(1.0f, 1.0f, 1.0f));

    float lx0 = (1.0f - u.x) * n000 + u.x * n100;
    float lx1 = (1.0f - u.x) * n010 + u.x * n110;
    float lz0 = (1.0f - u.y) * lx0 + u.y * lx1;

    float ux0 = (1.0f - u.x) * n001 + u.x * n101;
    float ux1 = (1.0f - u.x) * n011 + u.x * n111;
    float lz1 = (1.0f - u.y) * ux0 + u.y * ux1;

    return (1.0f - u.z) * lz0 + u.z * lz1;
}

// Fractal Brownian Motion (fBm)
UNIFIED_FUNC float fbm(Vec3f p, int octaves) {
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f; // Could be passed as param
    
    for (int i = 0; i < octaves; i++) {
        value += amplitude * value_noise(p); // Use p directly which is scaled before loop or outside
        p = p * 2.0f; // Lacunarity
        amplitude *= 0.5f; // Gain
    }
    return value;
}

