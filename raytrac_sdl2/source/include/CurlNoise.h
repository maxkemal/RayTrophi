/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          CurlNoise.h
* Author:        Kemal Demirtaş
* Date:          January 2026
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once

/**
 * @file CurlNoise.h
 * @brief High-quality procedural noise functions for physics simulations
 * 
 * Provides:
 * - Perlin noise (2D/3D)
 * - Simplex noise (2D/3D/4D)
 * - Curl noise (divergence-free for fluids)
 * - FBM (Fractal Brownian Motion)
 * - Turbulence
 * 
 * Based on industry-standard implementations used in Houdini and Blender.
 */

#include "Vec3.h"
#include <cmath>
#include <cstdint>
#include <array>

namespace Physics {
namespace Noise {

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS & UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

constexpr float PI = 3.14159265358979323846f;
constexpr float TAU = 2.0f * PI;

// Permutation table is internal - use initializeWithSeed() to set
extern const std::array<Vec3, 16> gradients3D;

/**
 * @brief Initialize permutation table with seed
 */
void initializeWithSeed(int seed);

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Fast hash function (for noise generation)
 */
inline uint32_t hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}

/**
 * @brief 3D hash to float [0, 1]
 */
inline float hashFloat(int x, int y, int z, int seed = 0) {
    uint32_t h = hash(x + hash(y + hash(z + hash(seed))));
    return (h & 0xFFFFFF) / float(0xFFFFFF);
}

/**
 * @brief Smoothstep interpolation
 */
inline float smoothstep(float t) {
    return t * t * (3.0f - 2.0f * t);
}

/**
 * @brief Quintic smoothstep (smoother derivatives)
 */
inline float smootherstep(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

/**
 * @brief Linear interpolation
 */
inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

/**
 * @brief Gradient at integer lattice point
 */
Vec3 gradient3D(int ix, int iy, int iz, int seed = 0);

// ═══════════════════════════════════════════════════════════════════════════════
// PERLIN NOISE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief 2D Perlin noise
 * @param x X coordinate
 * @param y Y coordinate
 * @param seed Random seed
 * @return Noise value in range [-1, 1]
 */
float perlin2D(float x, float y, int seed = 0);

/**
 * @brief 3D Perlin noise
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @param seed Random seed
 * @return Noise value in range [-1, 1]
 */
float perlin3D(float x, float y, float z, int seed = 0);

/**
 * @brief 3D Perlin noise (Vec3 input)
 */
inline float perlin3D(const Vec3& p, int seed = 0) {
    return perlin3D(p.x, p.y, p.z, seed);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMPLEX NOISE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief 2D Simplex noise (faster than Perlin)
 * @return Noise value in range [-1, 1]
 */
float simplex2D(float x, float y, int seed = 0);

/**
 * @brief 3D Simplex noise
 * @return Noise value in range [-1, 1]
 */
float simplex3D(float x, float y, float z, int seed = 0);

/**
 * @brief 3D Simplex noise (Vec3 input)
 */
inline float simplex3D(const Vec3& p, int seed = 0) {
    return simplex3D(p.x, p.y, p.z, seed);
}

/**
 * @brief 4D Simplex noise (for time-varying 3D noise)
 * @return Noise value in range [-1, 1]
 */
float simplex4D(float x, float y, float z, float w, int seed = 0);

// ═══════════════════════════════════════════════════════════════════════════════
// FRACTAL NOISE (FBM)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Fractal Brownian Motion (layered noise)
 * @param p Position
 * @param octaves Number of noise layers (1-8)
 * @param frequency Base frequency
 * @param lacunarity Frequency multiplier per octave (typically 2.0)
 * @param persistence Amplitude multiplier per octave (typically 0.5)
 * @param seed Random seed
 * @return FBM value (approximately [-1, 1] but can exceed)
 */
float fbm3D(const Vec3& p, int octaves = 4, float frequency = 1.0f,
            float lacunarity = 2.0f, float persistence = 0.5f, int seed = 0);

/**
 * @brief 2D FBM
 */
float fbm2D(float x, float y, int octaves = 4, float frequency = 1.0f,
            float lacunarity = 2.0f, float persistence = 0.5f, int seed = 0);

/**
 * @brief Time-varying 3D FBM using 4D simplex
 */
float fbm3D_animated(const Vec3& p, float time, int octaves = 4, float frequency = 1.0f,
                     float lacunarity = 2.0f, float persistence = 0.5f, int seed = 0);

// ═══════════════════════════════════════════════════════════════════════════════
// TURBULENCE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Turbulence (absolute value FBM for smoke-like patterns)
 */
float turbulence3D(const Vec3& p, int octaves = 4, float frequency = 1.0f,
                   float lacunarity = 2.0f, float persistence = 0.5f, int seed = 0);

/**
 * @brief Time-varying turbulence
 */
float turbulence3D_animated(const Vec3& p, float time, int octaves = 4, float frequency = 1.0f,
                            float lacunarity = 2.0f, float persistence = 0.5f, int seed = 0);

// ═══════════════════════════════════════════════════════════════════════════════
// CURL NOISE (Divergence-Free)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief 3D Curl noise (divergence-free velocity field)
 * 
 * Curl noise produces smooth, swirling motion without compression or expansion.
 * Perfect for fluid simulations, smoke, and realistic particle flows.
 * 
 * @param p Position in world space
 * @param frequency Noise frequency (lower = larger swirls)
 * @param seed Random seed
 * @return Curl noise vector (divergence-free)
 */
Vec3 curl3D(const Vec3& p, float frequency = 1.0f, int seed = 0);

/**
 * @brief Time-varying 3D Curl noise
 * 
 * @param p Position
 * @param time Current time
 * @param frequency Base frequency
 * @param speed Animation speed
 * @param seed Random seed
 * @return Animated curl noise vector
 */
Vec3 curl3D_animated(const Vec3& p, float time, float frequency = 1.0f, 
                     float speed = 0.1f, int seed = 0);

/**
 * @brief FBM Curl noise (layered for more detail)
 * 
 * @param p Position
 * @param octaves Number of layers
 * @param frequency Base frequency
 * @param lacunarity Frequency multiplier
 * @param persistence Amplitude multiplier
 * @param seed Random seed
 * @return Multi-octave curl noise
 */
Vec3 curlFBM(const Vec3& p, int octaves = 4, float frequency = 1.0f,
             float lacunarity = 2.0f, float persistence = 0.5f, int seed = 0);

/**
 * @brief Animated FBM Curl noise
 */
Vec3 curlFBM_animated(const Vec3& p, float time, int octaves = 4, float frequency = 1.0f,
                      float lacunarity = 2.0f, float persistence = 0.5f, 
                      float speed = 0.1f, int seed = 0);

// ═══════════════════════════════════════════════════════════════════════════════
// SPECIALIZED NOISE PATTERNS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Voronoi/Cellular noise (F1)
 * @return Distance to nearest cell center
 */
float voronoi3D(const Vec3& p, int seed = 0);

/**
 * @brief Voronoi F2-F1 (Crackle pattern)
 */
float voronoiCrackle(const Vec3& p, int seed = 0);

/**
 * @brief Ridge noise (mountains, sharp features)
 */
float ridgeNoise3D(const Vec3& p, int octaves = 4, float frequency = 1.0f,
                   float lacunarity = 2.0f, float offset = 1.0f, int seed = 0);

/**
 * @brief Billow noise (fluffy clouds, soft features)
 */
float billowNoise3D(const Vec3& p, int octaves = 4, float frequency = 1.0f,
                    float lacunarity = 2.0f, float persistence = 0.5f, int seed = 0);

// ═══════════════════════════════════════════════════════════════════════════════
// DOMAIN WARPING
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @brief Warp position using noise (for more complex patterns)
 * @param p Original position
 * @param warp_strength How much to displace
 * @param frequency Noise frequency
 * @param seed Random seed
 * @return Warped position
 */
Vec3 domainWarp(const Vec3& p, float warp_strength = 1.0f, 
                float frequency = 1.0f, int seed = 0);

/**
 * @brief Animated domain warp
 */
Vec3 domainWarp_animated(const Vec3& p, float time, float warp_strength = 1.0f,
                         float frequency = 1.0f, float speed = 0.1f, int seed = 0);

} // namespace Noise
} // namespace Physics
