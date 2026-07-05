/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          water_shaders_cpu.h
* Mirror of water_shaders.cuh for CPU rendering.
* =========================================================================
*/
#pragma once

#include "Vec3.h"
#include "WaterShaderCommon.h"
#include <cmath>
#include <algorithm>
#include <immintrin.h>
#include "Vec3SIMD.h"

// ═══════════════════════════════════════════════════════════════════════════════
// WATER RESULT STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════
struct WaterResultCPU {
    Vec3 normal;           // Perturbed surface normal
    float foam;            // Wave crest foam (0-1)
    float height;          // Wave displacement height
    
    // Advanced layered effects
    float depth;           // Distance to floor (for color gradient)
    float shore_factor;    // Proximity to shore (0=deep, 1=shore)
    float caustic_intensity; // Calculated caustic brightness
    
    // Pre-computed colors (filled by evaluateWaterAppearance)
    Vec3 water_color;      // Final blended shallow/deep color
    Vec3 absorption;       // Beer's law absorption factor
};

// ═══════════════════════════════════════════════════════════════════════════════
// WATER PARAMETERS (Mirror of GPU WaterParams)
// ═══════════════════════════════════════════════════════════════════════════════
struct WaterParamsCPU {
    // Waves
    float wave_speed;
    float wave_strength;
    float wave_frequency;
    
    // Colors
    Vec3 shallow_color;
    Vec3 deep_color;
    Vec3 absorption_color;
    
    // Depth & Appearance
    float depth_max;
    float absorption_density;
    float clarity;
    
    // Foam
    float foam_level;
    float shore_foam_distance;
    float shore_foam_intensity;
    
    // Caustics
    float caustic_intensity_scale;
    float caustic_scale;
    float caustic_speed;
    
    // SSS
    float sss_intensity;
    Vec3 sss_color;
    
    // FFT Ocean (Tessendorf)
    bool use_fft_ocean;             
    float fft_ocean_size;           
    float fft_choppiness;           

    // Micro Details
    float micro_detail_strength;
    float micro_detail_scale;
    float micro_anim_speed;             // Animation speed multiplier
    float micro_morph_speed;            // Shape morphing speed
    float foam_noise_scale;
    float foam_threshold;
    
    // Wind Animation (for micro details)
    float wind_direction;           // Radians
    float wind_speed;               // m/s
    float time;                     // Current animation time
};

inline WaterShader::SurfaceParams toWaterSurfaceParamsCPU(const WaterParamsCPU& params) {
    WaterShader::SurfaceParams out;
    out.wave_speed = params.wave_speed;
    out.wave_strength = params.wave_strength;
    out.wave_frequency = params.wave_frequency;
    out.shallow_color = params.shallow_color;
    out.deep_color = params.deep_color;
    out.absorption_color = params.absorption_color;
    out.depth_max = params.depth_max;
    out.absorption_density = params.absorption_density;
    out.clarity = params.clarity;
    out.foam_level = params.foam_level;
    out.shore_foam_distance = params.shore_foam_distance;
    out.shore_foam_intensity = params.shore_foam_intensity;
    out.caustic_intensity = params.caustic_intensity_scale;
    out.caustic_scale = params.caustic_scale;
    out.caustic_speed = params.caustic_speed;
    out.sss_intensity = params.sss_intensity;
    out.sss_color = params.sss_color;
    out.use_fft_ocean = params.use_fft_ocean;
    out.fft_ocean_size = WaterShader::resolvedDomain(params.fft_ocean_size);
    out.fft_choppiness = params.fft_choppiness;
    out.micro_detail_strength = params.micro_detail_strength;
    out.micro_detail_scale = params.micro_detail_scale;
    out.micro_anim_speed = params.micro_anim_speed;
    out.micro_morph_speed = params.micro_morph_speed;
    out.foam_noise_scale = params.foam_noise_scale;
    out.foam_threshold = params.foam_threshold;
    out.wind_direction = params.wind_direction;
    out.wind_speed = params.wind_speed;
    out.time = params.time;
    return out;
}

inline WaterShader::SurfaceSample toWaterSurfaceSampleCPU(const WaterResultCPU& res) {
    WaterShader::SurfaceSample out;
    out.normal = res.normal;
    out.foam = res.foam;
    out.height = res.height;
    out.depth = res.depth;
    out.shore_factor = res.shore_factor;
    out.caustic_intensity = res.caustic_intensity;
    out.water_color = res.water_color;
    out.absorption = res.absorption;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

inline float fract_cpu(float x) { return x - floorf(x); }
inline float lerp_cpu(float a, float b, float t) { return a + t * (b - a); }
inline float smoothstep_cpu(float edge0, float edge1, float x) {
    float t = std::fmax(0.0f, std::fmin(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

inline float hash12_cpu(float px, float py) {
    float p3x = fract_cpu(px * 0.1031f);
    float p3y = fract_cpu(py * 0.1031f);
    float p3z = fract_cpu(px * 0.1031f);
    float d = p3x * (p3y + 33.33f) + p3y * (p3z + 33.33f) + p3z * (p3x + 33.33f);
    p3x += d; p3y += d; p3z += d;
    float result = fract_cpu((p3x + p3y) * p3z);
    return result < 0 ? result + 1.0f : result;
}

inline float noise_cpu(float px, float py) {
    float ix = floorf(px);
    float iy = floorf(py);
    float fx = px - ix;
    float fy = py - iy;
    
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

// ═══════════════════════════════════════════════════════════════════════════════
// VORONOI NOISE (for Caustics)
// ═══════════════════════════════════════════════════════════════════════════════

inline float hash22_x(float px, float py) {
    float p3x = fract_cpu(px * 0.1031f);
    float p3y = fract_cpu(py * 0.1030f);
    float p3z = fract_cpu(px * 0.0973f);
    float d = p3x * (p3y + 33.33f) + p3y * (p3z + 33.33f) + p3z * (p3x + 33.33f);
    p3x += d; p3y += d;
    return fract_cpu((p3x + p3y) * p3z);
}

inline float hash22_y(float px, float py) {
    float p3x = fract_cpu(px * 0.1031f);
    float p3y = fract_cpu(py * 0.1030f);
    float p3z = fract_cpu(px * 0.0973f);
    float d = p3x * (p3y + 33.33f) + p3y * (p3z + 33.33f) + p3z * (p3x + 33.33f);
    p3x += d; p3y += d;
    return fract_cpu((p3x - p3y) * p3z + 0.5f);
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
            float point_x = 0.5f + 0.5f * sinf(time * 0.5f + 6.2831f * hash22_x(cell_x, cell_y));
            float point_y = 0.5f + 0.5f * sinf(time * 0.7f + 6.2831f * hash22_y(cell_x, cell_y));
            float diff_x = i + point_x - fx;
            float diff_y = j + point_y - fy;
            float dist = diff_x * diff_x + diff_y * diff_y;
            min_dist = std::fmin(min_dist, dist);
        }
    }
    return sqrtf(min_dist);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSTIC CALCULATION
// ═══════════════════════════════════════════════════════════════════════════════

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
    return powf(std::fmax(caustic, 0.0f), 2.0f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEPTH-BASED COLOR GRADIENT
// ═══════════════════════════════════════════════════════════════════════════════

inline Vec3 calculateDepthColorCPU(
    float depth,
    float depth_max,
    const Vec3& shallow_color,
    const Vec3& deep_color
) {
    float t = std::fmin(depth / std::fmax(depth_max, 0.1f), 1.0f);
    t = t * t * (3.0f - 2.0f * t);  
    return Vec3::lerp(shallow_color, deep_color, t);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHORE FOAM CALCULATION
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// BEER'S LAW ABSORPTION
// ═══════════════════════════════════════════════════════════════════════════════

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

// ═══════════════════════════════════════════════════════════════════════════════
// GERSTNER WAVE EVALUATION
// ═══════════════════════════════════════════════════════════════════════════════

inline float hsum_256(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(vsum, shuf);
    shuf = _mm_shuffle_ps(sums, sums, _MM_SHUFFLE(1, 0, 3, 2));
    sums = _mm_add_ps(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline WaterResultCPU evaluateGerstnerWaveCPU(
    const Vec3& position,
    const Vec3& baseNormal,
    float time,
    float speed_mult,
    float strength_mult,
    float freq_mult
) {
    alignas(32) static const float kWaveDirsX[8] = { 0.9806f, 0.7071f, -0.1961f, -0.7682f, -0.9363f, 0.0f, 0.5300f, 0.9138f };
    alignas(32) static const float kWaveDirsY[8] = { 0.1961f, 0.7071f, 0.9806f, 0.6402f, -0.3511f, -1.0f, -0.8480f, -0.4061f };
    alignas(32) static const float kOctaveFreqs[8]  = { 1.0f, 1.8f, 3.24f, 5.832f, 10.4976f, 18.89568f, 34.012224f, 61.2220032f };
    alignas(32) static const float kOctaveAmps[8]   = { 1.0f, 0.55f, 0.3025f, 0.166375f, 0.09150625f, 0.05032844f, 0.02768064f, 0.01522435f };
    alignas(32) static const float kOctaveSpeeds[8] = { 1.0f, 1.1f, 1.21f, 1.331f, 1.4641f, 1.61051f, 1.771561f, 1.948717f };

    __m256 v_dx = _mm256_load_ps(kWaveDirsX);
    __m256 v_dy = _mm256_load_ps(kWaveDirsY);
    __m256 v_freq = _mm256_mul_ps(_mm256_set1_ps(0.2f * freq_mult), _mm256_load_ps(kOctaveFreqs));
    __m256 v_amp = _mm256_mul_ps(_mm256_set1_ps(0.5f * strength_mult), _mm256_load_ps(kOctaveAmps));
    __m256 v_speed = _mm256_mul_ps(_mm256_set1_ps(0.5f * speed_mult), _mm256_load_ps(kOctaveSpeeds));

    // x = position.x * dx + position.z * dy
    __m256 v_x = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(position.x), v_dx),
                               _mm256_mul_ps(_mm256_set1_ps(position.z), v_dy));

    // phase = x * frequency + time * speed
    __m256 v_phase = _mm256_add_ps(_mm256_mul_ps(v_x, v_freq),
                                   _mm256_mul_ps(_mm256_set1_ps(time), v_speed));

    __m256 v_s_ph, v_c_ph;
    Vec3SIMD::sincos_256(v_phase, &v_s_ph, &v_c_ph);

    // wa = frequency * amplitude
    __m256 v_wa = _mm256_mul_ps(v_freq, v_amp);

    // dHx += dx * wa * c_ph
    __m256 v_dHx = _mm256_mul_ps(_mm256_mul_ps(v_dx, v_wa), v_c_ph);

    // dHz += dy * wa * c_ph
    __m256 v_dHz = _mm256_mul_ps(_mm256_mul_ps(v_dy, v_wa), v_c_ph);

    // jacobian -= steepness * wa * s_ph   (steepness = 0.5f)
    __m256 v_jac = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), v_wa), v_s_ph);

    // accumulatedHeight += amplitude * s_ph
    __m256 v_height = _mm256_mul_ps(v_amp, v_s_ph);

    float dHx = hsum_256(v_dHx);
    float dHz = hsum_256(v_dHz);
    float jacobian = 1.0f - hsum_256(v_jac);
    float accumulatedHeight = hsum_256(v_height);

    // Constructed Normal
    Vec3 waveNormal = Vec3(-dHx, 1.0f, -dHz).normalize();
    
    // Foam calculation (Simplified Jacobian-based)
    float foam = 0.0f;
    float j_foam = (0.5f - jacobian);
    if (j_foam > 0.0f) foam += j_foam * 2.0f;
    
    float h_foam = (accumulatedHeight - 0.5f * strength_mult);
    if (h_foam > 0.0f) foam += h_foam;
    
    foam = std::fmax(0.0f, std::fmin(1.0f, foam));
    foam = smoothstep_cpu(0.3f, 0.7f, foam);

    WaterResultCPU res;
    res.normal = waveNormal;
    res.foam = foam;
    res.height = accumulatedHeight;
    res.depth = 0.0f;
    res.shore_factor = 0.0f;
    res.caustic_intensity = 0.0f;
    res.water_color = Vec3(2.0f/255.0f, 3.0f/255.0f, 3.0f/255.0f); // Dark blue (2, 3, 3) for physical look
    res.absorption = Vec3(1.0f, 1.0f, 1.0f);
    
    return res;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN EVALUATE WATER FUNCTION (CPU)
// ═══════════════════════════════════════════════════════════════════════════════

inline WaterResultCPU evaluateWaterCPU(
    const Vec3& position,
    const Vec3& baseNormal,
    float time,
    const WaterParamsCPU& params
) {
    WaterResultCPU res;
    
    // CPU Fallback: Gerstner Waves
    res = evaluateGerstnerWaveCPU(
        position, baseNormal, time,
        params.wave_speed,
        params.wave_strength,
        params.wave_frequency
    );
    float foam_signal = res.foam;

    // Apply Micro Details if strength > 0 (hybrid drift + morph, matching GPU/Vulkan)
    if (params.micro_detail_strength > 0.001f) {
        // Wind direction vectors (main and cross-wind)
        float wind_dx = cosf(params.wind_direction);
        float wind_dz = sinf(params.wind_direction);
        float cross_dx = -wind_dz;  // Perpendicular
        float cross_dz = wind_dx;
        
        // Base speed scales with wind and user-controlled anim speed
        float base_speed = sqrtf(std::fmax(1.0f, params.wind_speed)) * params.micro_anim_speed;
        float t = params.time;
        float scale = params.micro_detail_scale;
        float morph = params.micro_morph_speed;
        
        // === LAYER 1: Primary ripples (wind direction) ===
        float off1_x = wind_dx * t * base_speed + sinf(t * 0.3f * morph) * 0.5f;
        float off1_z = wind_dz * t * base_speed + cosf(t * 0.2f * morph) * 0.5f;
        float pos1_x = position.x * scale + off1_x;
        float pos1_z = position.z * scale + off1_z;
        
        // === LAYER 2: Secondary ripples (slower, larger, offset angle) ===
        float off2_x = (wind_dx * 0.7f + cross_dx * 0.3f) * t * base_speed * 0.6f + cosf(t * 0.15f * morph + 1.5f) * 0.8f;
        float off2_z = (wind_dz * 0.7f + cross_dz * 0.3f) * t * base_speed * 0.6f + sinf(t * 0.25f * morph + 2.0f) * 0.8f;
        float pos2_x = position.x * scale * 0.5f + off2_x;
        float pos2_z = position.z * scale * 0.5f + off2_z;
        
        // === LAYER 3: Fine detail (cross-wind, faster morph) ===
        float off3_x = cross_dx * t * base_speed * 0.4f + sinf(t * 0.5f * morph + 3.0f) * 0.3f;
        float off3_z = cross_dz * t * base_speed * 0.4f + cosf(t * 0.4f * morph + 1.0f) * 0.3f;
        float pos3_x = position.x * scale * 2.0f + off3_x;
        float pos3_z = position.z * scale * 2.0f + off3_z;
        
        float dx = 0.01f;
        
        // Layer 1 derivatives
        float h1_c = fbm_cpu(pos1_x, pos1_z);
        float h1_x = fbm_cpu(pos1_x + dx, pos1_z);
        float h1_z = fbm_cpu(pos1_x, pos1_z + dx);
        
        // Layer 2 derivatives
        float h2_c = fbm_cpu(pos2_x, pos2_z);
        float h2_x = fbm_cpu(pos2_x + dx, pos2_z);
        float h2_z = fbm_cpu(pos2_x, pos2_z + dx);
        
        // Layer 3 derivatives (single octave for speed)
        float h3_c = noise_cpu(pos3_x, pos3_z);
        float h3_x = noise_cpu(pos3_x + dx, pos3_z);
        float h3_z = noise_cpu(pos3_x, pos3_z + dx);
        
        // Combine layers with weights
        float h_c = h1_c * 0.5f + h2_c * 0.35f + h3_c * 0.15f;
        float h_x = h1_x * 0.5f + h2_x * 0.35f + h3_x * 0.15f;
        float h_z = h1_z * 0.5f + h2_z * 0.35f + h3_z * 0.15f;
        
        float dsdx = (h_x - h_c) / dx;
        float dsdz = (h_z - h_c) / dx;
        float micro_slope = std::fmax(0.0f, std::fmin(1.0f, h_c * 0.5f + 0.5f));
        
        Vec3 micro_n = Vec3(-dsdx * params.micro_detail_strength, 1.0f, -dsdz * params.micro_detail_strength).normalize();
        res.normal = (res.normal + micro_n).normalize();

        float foam_noise_scale = params.foam_noise_scale;
        float foam_noise = fbm_cpu(position.x * foam_noise_scale + off1_x * 0.5f,
                                   position.z * foam_noise_scale + off1_z * 0.5f);
        float foam_breakup = foam_noise * 0.5f + 0.5f;
        float micro_foam = std::fmax(0.0f,
            (micro_slope + (foam_breakup - 0.5f) * 0.35f - params.foam_threshold) * 5.0f);
        foam_signal = std::fmax(foam_signal, micro_foam);
    }

    float max_depth = params.depth_max > 0.1f ? params.depth_max : 10.0f;
    float depth_ratio = std::fmax(0.0f, std::fmin(1.0f, 0.55f + res.height / std::fmax(params.wave_strength * 4.0f, 0.001f)));
    float depth_proxy = lerp_cpu(std::fmin(params.shore_foam_distance * 0.5f, max_depth), max_depth, depth_ratio);
    depth_proxy = std::fmax(0.05f, std::fmin(depth_proxy, max_depth));
    res.depth = depth_proxy;

    float shore_depth_proxy = depth_proxy;
    res.shore_factor = 1.0f - std::fmax(0.0f, std::fmin(1.0f, shore_depth_proxy / std::fmax(params.shore_foam_distance, 0.001f)));

    if (params.shore_foam_intensity > 0.01f) {
        float shore_foam = calculateShoreFoamCPU(shore_depth_proxy, params.shore_foam_distance, params.shore_foam_intensity, position, time);
        res.foam = std::fmin(foam_signal * params.foam_level + shore_foam, 1.0f);
    } else {
        res.foam = std::fmin(foam_signal * params.foam_level, 1.0f);
    }

    res.water_color = calculateDepthColorCPU(depth_proxy, params.depth_max, params.shallow_color, params.deep_color);
    res.caustic_intensity = 0.0f;
    if (params.caustic_intensity_scale > 0.01f) {
        Vec3 floor_proxy(position.x, position.y - depth_proxy, position.z);
        float caustic = calculateWaterCausticsCPU(floor_proxy, time, params.caustic_scale, params.caustic_speed);
        float caustic_fade = std::exp(-depth_proxy * params.absorption_density * 0.5f);
        res.caustic_intensity = caustic * params.caustic_intensity_scale * caustic_fade;
        res.water_color = res.water_color + params.shallow_color * res.caustic_intensity;
    }

    res.absorption = calculateWaterAbsorptionCPU(depth_proxy, params.absorption_color, params.absorption_density);

    return res;
}

inline WaterShader::SurfaceSample evaluateWaterSurfaceCPU(
    const Vec3& position,
    const Vec3& baseNormal,
    const WaterShader::SurfaceParams& params
) {
    WaterParamsCPU legacy;
    legacy.wave_speed = params.wave_speed;
    legacy.wave_strength = params.wave_strength;
    legacy.wave_frequency = params.wave_frequency;
    legacy.shallow_color = params.shallow_color;
    legacy.deep_color = params.deep_color;
    legacy.absorption_color = params.absorption_color;
    legacy.depth_max = params.depth_max;
    legacy.absorption_density = params.absorption_density;
    legacy.clarity = params.clarity;
    legacy.foam_level = params.foam_level;
    legacy.shore_foam_distance = params.shore_foam_distance;
    legacy.shore_foam_intensity = params.shore_foam_intensity;
    legacy.caustic_intensity_scale = params.caustic_intensity;
    legacy.caustic_scale = params.caustic_scale;
    legacy.caustic_speed = params.caustic_speed;
    legacy.sss_intensity = params.sss_intensity;
    legacy.sss_color = params.sss_color;
    legacy.use_fft_ocean = params.use_fft_ocean;
    legacy.fft_ocean_size = WaterShader::resolvedDomain(params.fft_ocean_size);
    legacy.fft_choppiness = params.fft_choppiness;
    legacy.micro_detail_strength = params.micro_detail_strength;
    legacy.micro_detail_scale = params.micro_detail_scale;
    legacy.micro_anim_speed = params.micro_anim_speed;
    legacy.micro_morph_speed = params.micro_morph_speed;
    legacy.foam_noise_scale = params.foam_noise_scale;
    legacy.foam_threshold = params.foam_threshold;
    legacy.wind_direction = params.wind_direction;
    legacy.wind_speed = params.wind_speed;
    legacy.time = params.time;
    return toWaterSurfaceSampleCPU(evaluateWaterCPU(position, baseNormal, params.time, legacy));
}


