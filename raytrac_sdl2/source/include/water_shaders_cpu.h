/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          water_shaders_cpu.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// CPU WATER SHADER - Gerstner Waves, Foam, Depth Coloring (CPU Version)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// Mirrors water_shaders.cuh but for CPU rendering

#include "Vec3.h"
#include <cmath>

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// WATER RESULT STRUCTURE
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
struct WaterResultCPU {
    Vec3 normal;           // Perturbed surface normal
    float foam;            // Wave crest foam (0-1)
    float height;          // Wave displacement height
    
    // Advanced layered effects
    float depth;           // Distance to floor
    float shore_factor;    // Proximity to shore (0=deep, 1=shore)
    float caustic_intensity;
    
    Vec3 water_color;      // Final blended shallow/deep color
    Vec3 absorption;       // Beer's law absorption factor
};

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// HELPER FUNCTIONS
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

inline float hash12_cpu(float px, float py) {
    float p3x = fmodf(px * 0.1031f, 1.0f);
    float p3y = fmodf(py * 0.1031f, 1.0f);
    float p3z = fmodf(px * 0.1031f, 1.0f);
    if (p3x < 0) p3x += 1.0f;
    if (p3y < 0) p3y += 1.0f;
    if (p3z < 0) p3z += 1.0f;
    float d = p3x * (p3y + 33.33f) + p3y * (p3z + 33.33f) + p3z * (p3x + 33.33f);
    p3x += d; p3y += d; p3z += d;
    float result = fmodf((p3x + p3y) * p3z, 1.0f);
    return result < 0 ? result + 1.0f : result;
}

inline float noise_cpu(float px, float py) {
    float ix = floorf(px);
    float iy = floorf(py);
    float fx = px - ix;
    float fy = py - iy;
    
    // Smoothstep
    float ux = fx * fx * (3.0f - 2.0f * fx);
    float uy = fy * fy * (3.0f - 2.0f * fy);
    
    float a = hash12_cpu(ix, iy);
    float b = hash12_cpu(ix + 1, iy);
    float c = hash12_cpu(ix, iy + 1);
    float d = hash12_cpu(ix + 1, iy + 1);
    
    float res = a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy;
    return res * 2.0f - 1.0f;
}

inline float fbm_cpu(float px, float py) {
    float v = 0.0f;
    float a = 0.5f;
    float c = 0.866025f; // cos(30)
    float s = 0.5f;      // sin(30)
    
    for (int i = 0; i < 4; ++i) {
        v += a * noise_cpu(px, py);
        float npx = px * c - py * s;
        float npy = px * s + py * c;
        px = npx * 2.0f + 100.0f;
        py = npy * 2.0f + 100.0f;
        a *= 0.5f;
    }
    return v;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// VORONOI NOISE (for Caustics)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

inline float hash22_x(float px, float py) {
    float p3x = fmodf(px * 0.1031f, 1.0f);
    float p3y = fmodf(py * 0.1030f, 1.0f);
    float p3z = fmodf(px * 0.0973f, 1.0f);
    if (p3x < 0) p3x += 1.0f;
    if (p3y < 0) p3y += 1.0f;
    if (p3z < 0) p3z += 1.0f;
    float d = p3x * (p3y + 33.33f) + p3y * (p3z + 33.33f) + p3z * (p3x + 33.33f);
    p3x += d; p3y += d;
    float result = fmodf((p3x + p3y) * p3z, 1.0f);
    return result < 0 ? result + 1.0f : result;
}

inline float hash22_y(float px, float py) {
    float p3x = fmodf(px * 0.1031f, 1.0f);
    float p3y = fmodf(py * 0.1030f, 1.0f);
    float p3z = fmodf(px * 0.0973f, 1.0f);
    if (p3x < 0) p3x += 1.0f;
    if (p3y < 0) p3y += 1.0f;
    if (p3z < 0) p3z += 1.0f;
    float d = p3x * (p3y + 33.33f) + p3y * (p3z + 33.33f) + p3z * (p3x + 33.33f);
    p3x += d; p3y += d;
    float result = fmodf((p3x - p3y) * p3z + 0.5f, 1.0f);
    return result < 0 ? result + 1.0f : result;
}

inline float voronoi_cpu(float px, float py, float time) {
    float nx = floorf(px);
    float ny = floorf(py);
    float fx = px - nx;
    float fy = py - ny;
    
    float min_dist = 1.0f;
    
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float cell_x = nx + i;
            float cell_y = ny + j;
            
            float point_x = hash22_x(cell_x, cell_y);
            float point_y = hash22_y(cell_x, cell_y);
            
            point_x = 0.5f + 0.5f * sinf(time * 0.5f + 6.2831f * point_x);
            point_y = 0.5f + 0.5f * sinf(time * 0.7f + 6.2831f * point_y);
            
            float diff_x = i + point_x - fx;
            float diff_y = j + point_y - fy;
            float dist = diff_x * diff_x + diff_y * diff_y;
            min_dist = std::fmin(min_dist, dist);
        }
    }
    
    return sqrtf(min_dist);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// CAUSTIC CALCULATION
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

inline float calculateWaterCausticsCPU(
    const Vec3& floor_position,
    float time,
    float caustic_scale,
    float caustic_speed
) {
    float ux = floor_position.x * caustic_scale;
    float uy = floor_position.z * caustic_scale;
    float t = time * caustic_speed;
    
    float v1 = voronoi_cpu(ux, uy, t);
    float v2 = voronoi_cpu(ux * 1.5f + 50.0f, uy * 1.5f + 50.0f, t * 1.3f);
    
    float caustic = 1.0f - v1 * v2;
    caustic = powf(std::fmax(caustic, 0.0f), 2.0f);
    
    return caustic;
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// DEPTH-BASED COLOR GRADIENT
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

inline Vec3 calculateDepthColorCPU(
    float depth,
    float depth_max,
    const Vec3& shallow_color,
    const Vec3& deep_color
) {
    float t = std::fmin(depth / std::fmax(depth_max, 0.1f), 1.0f);
    t = t * t * (3.0f - 2.0f * t);  // Smoothstep
    return Vec3::lerp(shallow_color, deep_color, t);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// SHORE FOAM CALCULATION
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

inline float calculateShoreFoamCPU(
    float depth,
    float shore_distance,
    float shore_intensity,
    const Vec3& position,
    float time
) {
    if (depth > shore_distance) return 0.0f;
    
    float shore_t = 1.0f - (depth / shore_distance);
    shore_t = shore_t * shore_t;
    
    float foam_noise = (fbm_cpu(position.x * 2.0f + time * 0.5f, position.z * 2.0f) + 1.0f) * 0.5f;
    float edge_pattern = sinf(depth * 20.0f - time * 3.0f) * 0.5f + 0.5f;
    
    float shore_foam = shore_t * shore_intensity * foam_noise * (0.5f + edge_pattern * 0.5f);
    
    return std::fmin(shore_foam, 1.0f);
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// BEER'S LAW ABSORPTION
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

inline Vec3 calculateWaterAbsorptionCPU(
    float depth,
    const Vec3& absorption_color,
    float absorption_density
) {
    return Vec3(
        expf(-absorption_color.x * depth * absorption_density),
        expf(-absorption_color.y * depth * absorption_density),
        expf(-absorption_color.z * depth * absorption_density)
    );
}

// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
// GERSTNER WAVE EVALUATION (Main Function)
// â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

inline WaterResultCPU evaluateGerstnerWaveCPU(
    const Vec3& position,
    const Vec3& baseNormal,
    float time,
    float speed_mult,
    float strength_mult,
    float freq_mult
) {
    const int NUM_WAVES = 8;
    
    // Wave directions
    float waveDirs[8][2] = {
        {0.9806f, 0.1961f},   // normalize(1.0, 0.2)
        {0.7071f, 0.7071f},   // normalize(0.7, 0.7)
        {-0.1961f, 0.9806f},  // normalize(-0.2, 1.0)
        {-0.7682f, 0.6402f},  // normalize(-0.6, 0.5)
        {-0.9363f, -0.3511f}, // normalize(-0.8, -0.3)
        {0.0f, -1.0f},        // normalize(0, -1)
        {0.5300f, -0.8480f},  // normalize(0.5, -0.8)
        {0.9138f, -0.4061f}   // normalize(0.9, -0.4)
    };
    
    float dHx = 0.0f;
    float dHz = 0.0f;
    float accumulatedHeight = 0.0f;
    float jacobian = 1.0f;
    
    float frequency = 0.2f * freq_mult;
    float amplitude = 1.0f * strength_mult;
    float speed = 0.5f * speed_mult;
    
    for (int i = 0; i < NUM_WAVES; ++i) {
        float dx = waveDirs[i][0];
        float dy = waveDirs[i][1];
        
        float x = position.x * dx + position.z * dy;
        float phase = x * frequency + time * speed;
        
        float c_ph = cosf(phase);
        float s_ph = sinf(phase);
        
        float steepness = 0.8f / (frequency * amplitude * NUM_WAVES);
        steepness = std::fmin(steepness, 1.0f);
        
        float wa = frequency * amplitude;
        float deriv = wa * c_ph;
        
        dHx += dx * deriv;
        dHz += dy * deriv;
        
        float q_wa_s = steepness * wa * s_ph;
        jacobian -= q_wa_s;
        
        accumulatedHeight += amplitude * s_ph;
        
        frequency *= 1.8f;
        amplitude *= 0.55f;
        speed *= 1.1f;
    }
    
    // Add micro-noise
    float t_off = time * 0.2f;
    float microDetail = fbm_cpu(position.x * 4.0f + t_off, position.z * 4.0f + t_off) * 0.05f * strength_mult;
    dHx += microDetail * 2.0f;
    dHz += microDetail * 2.0f;
    
    // Construct normal
    Vec3 waveNormal = Vec3(-dHx, 1.0f, -dHz).normalize();
    
    // Foam calculation
    float foam = 0.0f;
    float j_foam = (0.5f - jacobian);
    if (j_foam > 0.0f) foam += j_foam * 2.0f;
    
    float h_foam = (accumulatedHeight - 0.5f * strength_mult);
    if (h_foam > 0.0f) foam += h_foam;
    
    foam = std::fmax(0.0f, std::fmin(1.0f, foam));
    
    // Smoothstep foam
    foam = foam * foam * (3.0f - 2.0f * foam);
    
    WaterResultCPU res;
    res.normal = waveNormal;
    res.foam = foam;
    res.height = accumulatedHeight;
    res.depth = 0.0f;
    res.shore_factor = 0.0f;
    res.caustic_intensity = 0.0f;
    res.water_color = Vec3(0.2f, 0.5f, 0.7f);
    res.absorption = Vec3(1.0f, 1.0f, 1.0f);
    
    return res;
}

