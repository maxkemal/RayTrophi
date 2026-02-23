#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include "vec3_utils.cuh"


// ═══════════════════════════════════════════════════════════════════════════════
// WATER SHADER RESULT STRUCTURE
// ═══════════════════════════════════════════════════════════════════════════════
struct WaterResult {
    float3 normal;           // Perturbed surface normal
    float foam;              // Wave crest foam (0-1)
    float height;            // Wave displacement height
    
    // Advanced layered effects
    float depth;             // Distance to floor (for color gradient)
    float shore_factor;      // Proximity to shore (0=deep, 1=shore)
    float caustic_intensity; // Calculated caustic brightness
    
    // Pre-computed colors (filled by evaluateWaterAppearance)
    float3 water_color;      // Final blended shallow/deep color
    float3 absorption;       // Beer's law absorption factor
};

// ═══════════════════════════════════════════════════════════════════════════════
// WATER PARAMETERS (packed from GpuMaterial)
// ═══════════════════════════════════════════════════════════════════════════════
struct WaterParams {
    // Waves
    float wave_speed;
    float wave_strength;
    float wave_frequency;
    
    // Colors
    float3 shallow_color;
    float3 deep_color;
    float3 absorption_color;
    
    // Depth & Appearance
    float depth_max;
    float absorption_density;
    float clarity;
    
    // Foam
    float foam_level;
    float shore_foam_distance;
    float shore_foam_intensity;
    
    // Caustics
    float caustic_intensity;
    float caustic_scale;
    float caustic_speed;
    
    // SSS
    float sss_intensity;
    float3 sss_color;
    
    // FFT Ocean (Tessendorf)
    bool use_fft_ocean;             // Use FFT instead of Gerstner
    float fft_ocean_size;           // World space coverage
    float fft_choppiness;           // Horizontal displacement strength
    cudaTextureObject_t fft_height_tex;  // Height texture from FFT
    cudaTextureObject_t fft_normal_tex;  // Normal texture from FFT

    // Micro Details
    float micro_detail_strength;
    float micro_detail_scale;
    float micro_anim_speed;             // Animation speed multiplier
    float micro_morph_speed;            // Shape morphing speed
    float foam_noise_scale;
    float foam_threshold;
    
    // Wind Animation (for micro details)
    float wind_direction;           // Radians, direction wind is blowing
    float wind_speed;               // m/s, controls micro detail animation speed
    float time;                     // Current time for animation
};

// --- Helper Functions ---
// fract, floor, smoothstep moved to vec3_utils.cuh

__device__ inline float2 normalize_f2(float2 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y);
    if (len < 1e-6f) return make_float2(0.0f, 1.0f);
    return make_float2(v.x / len, v.y / len);
}

// Simple pseudo-random hash for noise
__device__ inline float hash12(float2 p) {
    float3 p3 = make_float3(p.x, p.y, p.x);
    p3 = fract(p3 * 0.1031f);
    float d = dot(p3, make_float3(p3.y + 33.33f, p3.z + 33.33f, p3.x + 33.33f));
    p3.x += d; p3.y += d; p3.z += d;
    return fract((p3.x + p3.y) * p3.z);
}

// Hash for 2D -> 2D (for Voronoi)
__device__ inline float2 hash22(float2 p) {
    float3 p3 = fract(make_float3(p.x, p.y, p.x) * make_float3(0.1031f, 0.1030f, 0.0973f));
    float d = dot(p3, make_float3(p3.y + 33.33f, p3.z + 33.33f, p3.x + 33.33f));
    p3.x += d; p3.y += d;
    return fract(make_float2((p3.x + p3.y) * p3.z, (p3.x - p3.y) * p3.z + 0.5f));
}

__device__ inline float noise(float2 p) {
    float2 i = floor_float2(p);
    float2 f = fract(p);
    
    // u = f * f * (3.0 - 2.0 * f)
    float2 term = make_float2(3.0f - 2.0f * f.x, 3.0f - 2.0f * f.y);
    float2 u = make_float2(f.x * f.x * term.x, f.y * f.y * term.y);
    
    float res = lerp(lerp(hash12(i + make_float2(0.0f, 0.0f)), 
                          hash12(i + make_float2(1.0f, 0.0f)), u.x),
                     lerp(hash12(i + make_float2(0.0f, 1.0f)), 
                          hash12(i + make_float2(1.0f, 1.0f)), u.x), u.y);
    return res * 2.0f - 1.0f;
}

// Fractal Brownian Motion for micro-details
__device__ inline float fbm(float2 p) {
    float v = 0.0f;
    float a = 0.5f;
    float2 shift = make_float2(100.0f, 100.0f);
    // Rotate to reduce axial bias
    float c = 0.866025f; // cos(30)
    float s = 0.5f;      // sin(30)
    
    for (int i = 0; i < 4; ++i) {
        v += a * noise(p);
        // p = rot * p * 2.0 + shift
        float px = p.x * c - p.y * s;
        float py = p.x * s + p.y * c;
        p = make_float2(px * 2.0f + shift.x, py * 2.0f + shift.y);
        a *= 0.5f;
    }
    return v;
}

// ═══════════════════════════════════════════════════════════════════════════════
// VORONOI NOISE (for Caustics)
// ═══════════════════════════════════════════════════════════════════════════════
__device__ inline float voronoi(float2 p, float time) {
    float2 n = floor_float2(p);
    float2 f = fract(p);
    
    float min_dist = 1.0f;
    
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float2 neighbor = make_float2((float)i, (float)j);
            float2 cell = n + neighbor;
            
            // Animated cell center
            float2 point = hash22(cell);
            point = make_float2(
                0.5f + 0.5f * sinf(time * 0.5f + 6.2831f * point.x),
                0.5f + 0.5f * sinf(time * 0.7f + 6.2831f * point.y)
            );
            
            float2 diff = neighbor + point - f;
            float dist = diff.x * diff.x + diff.y * diff.y;  // float2 dot product
            min_dist = fminf(min_dist, dist);
        }
    }
    
    return sqrtf(min_dist);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSTIC CALCULATION
// ═══════════════════════════════════════════════════════════════════════════════
__device__ inline float calculateWaterCaustics(
    float3 floor_position,
    float time,
    float caustic_scale,
    float caustic_speed
) {
    float2 uv = make_float2(floor_position.x, floor_position.z) * caustic_scale;
    float t = time * caustic_speed;
    
    // Two layers of animated Voronoi for complex pattern
    float v1 = voronoi(uv * 1.0f, t);
    float v2 = voronoi(uv * 1.5f + make_float2(50.0f, 50.0f), t * 1.3f);
    
    // Combine and sharpen
    float caustic = 1.0f - v1 * v2;
    caustic = powf(fmaxf(caustic, 0.0f), 2.0f);  // Sharpen peaks
    
    return caustic;
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEPTH-BASED COLOR GRADIENT
// ═══════════════════════════════════════════════════════════════════════════════
__device__ inline float3 calculateDepthColor(
    float depth,
    float depth_max,
    float3 shallow_color,
    float3 deep_color
) {
    float t = fminf(depth / fmaxf(depth_max, 0.1f), 1.0f);
    // Smooth step for more natural transition
    t = t * t * (3.0f - 2.0f * t);
    return lerp(shallow_color, deep_color, t);
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHORE FOAM CALCULATION
// ═══════════════════════════════════════════════════════════════════════════════
__device__ inline float calculateShoreFoam(
    float depth,
    float shore_distance,
    float shore_intensity,
    float3 position,
    float time,
    float foam_scale = 1.0f
) {
    if (depth > shore_distance || shore_distance < 0.001f) return 0.0f;
    
    // Base shore factor
    float shore_t = 1.0f - (depth / shore_distance);
    shore_t = shore_t * shore_t;  // Quadratic falloff
    
    // Smooth the threshold so it doesn't abruptly clip
    shore_t = smoothstep(0.0f, 1.0f, shore_t);
    
    // Add animated noise for natural look
    float noise_val = foam_scale > 0.001f ? foam_scale : 1.0f;
    float2 noisePos = make_float2(position.x * noise_val + time * 0.5f, position.z * noise_val - time * 0.3f);
    float foam_noise = (fbm(noisePos) + 1.0f) * 0.5f;
    
    // Combine with edge detection pattern 
    float edge_pattern = sinf(depth * (10.0f / shore_distance) - time * 3.0f) * 0.5f + 0.5f;
    
    float shore_foam = shore_t * shore_intensity * ((foam_noise * 0.7f) + (edge_pattern * 0.3f));
    
    return fminf(shore_foam * 1.5f, 1.0f); // Boost slightly for clearer visibility
}

// ═══════════════════════════════════════════════════════════════════════════════
// BEER'S LAW ABSORPTION
// ═══════════════════════════════════════════════════════════════════════════════
__device__ inline float3 calculateWaterAbsorption(
    float depth,
    float3 absorption_color,
    float absorption_density
) {
    // Beer's Law: I = I0 * exp(-absorption * distance)
    float3 absorption = make_float3(
        expf(-absorption_color.x * depth * absorption_density),
        expf(-absorption_color.y * depth * absorption_density),
        expf(-absorption_color.z * depth * absorption_density)
    );
    return absorption;
}

// Advanced Multi-Octave Gerstner Wave
// Returns world space normal and foam amount
__device__ inline WaterResult evaluateGerstnerWave(
    float3 position, 
    float3 baseNormal, 
    float time,
    float speed_mult, 
    float strength_mult, 
    float freq_mult
) {
    // Parameters
    const int NUM_WAVES = 8;
    
    // Direction & frequency seeds
    float2 waveDirs[8] = {
        normalize_f2(make_float2(1.0f, 0.2f)),
        normalize_f2(make_float2(0.7f, 0.7f)),
        normalize_f2(make_float2(-0.2f, 1.0f)),
        normalize_f2(make_float2(-0.6f, 0.5f)),
        normalize_f2(make_float2(-0.8f, -0.3f)),
        normalize_f2(make_float2(0.0f, -1.0f)),
        normalize_f2(make_float2(0.5f, -0.8f)),
        normalize_f2(make_float2(0.9f, -0.4f))
    };
    
    float dHx = 0.0f;
    float dHz = 0.0f;
    float accumulatedHeight = 0.0f;
    float jacobian = 1.0f; 
    
    float frequency = 0.2f * freq_mult; 
    float amplitude = 1.0f * strength_mult;
    float speed = 0.5f * speed_mult;
    
    // Main Wave Loop
    for(int i=0; i<NUM_WAVES; ++i) {
        float2 d = waveDirs[i];
        
        float x = position.x * d.x + position.z * d.y;
        float phase = x * frequency + time * speed;
        
        float c_ph = cosf(phase);
        float s_ph = sinf(phase);
        
        // Steepness
        float steepness = 0.8f / (frequency * amplitude * NUM_WAVES); 
        steepness = fminf(steepness, 1.0f);
        
        float wa = frequency * amplitude;
        float deriv = wa * c_ph; 
        
        dHx += d.x * deriv;
        dHz += d.y * deriv;
        
        float q_wa_s = steepness * wa * s_ph;
        jacobian -= q_wa_s;
        
        accumulatedHeight += amplitude * s_ph;
        
        // Next Octave
        frequency *= 1.8f;   
        amplitude *= 0.55f;  
        speed *= 1.1f;
    }
    
    // Add Micro-Noise (Ripples)
    float t_off = time * 0.2f;
    float2 noisePos = make_float2(position.x * 4.0f + t_off, position.z * 4.0f + t_off);
    float microDetail = fbm(noisePos) * 0.05f * strength_mult;
    
    dHx += microDetail * 2.0f;
    dHz += microDetail * 2.0f;
    
    // Construct Normal
    float3 waveNormal = normalize(make_float3(-dHx, 1.0f, -dHz));
    
    // Foam calculation
    float foam = 0.0f;
    
    float j_foam = (0.5f - jacobian); 
    if (j_foam > 0.0f) foam += j_foam * 2.0f;
    
    float h_foam = (accumulatedHeight - 0.5f * strength_mult);
    if (h_foam > 0.0f) foam += h_foam;
    
    foam = fmaxf(0.0f, fminf(1.0f, foam));
    
    // Sharpen foam transition
    foam = smoothstep(0.3f, 0.7f, foam);

    WaterResult res;
    res.normal = waveNormal;
    res.foam = foam;
    res.height = accumulatedHeight;
    
    // Initialize advanced fields with defaults
    // (These will be computed separately in raygen when depth info is available)
    res.depth = 0.0f;
    res.shore_factor = 0.0f;
    res.caustic_intensity = 0.0f;
    res.water_color = make_float3(0.2f, 0.5f, 0.7f);
    res.absorption = make_float3(1.0f, 1.0f, 1.0f);
    
    return res;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FFT OCEAN SAMPLING
// ═══════════════════════════════════════════════════════════════════════════════
// Samples the pre-computed FFT ocean textures instead of procedural Gerstner waves

__device__ inline WaterResult sampleFFTOcean(
    const float3& position,
    float ocean_size,
    float choppiness,
    cudaTextureObject_t height_tex,
    cudaTextureObject_t normal_tex,
    float foam_level,
    float micro_detail_strength,
    float micro_detail_scale,
    float micro_anim_speed,
    float micro_morph_speed,
    float foam_noise_scale,
    float foam_threshold,
    float wind_direction,
    float wind_speed,
    float time
) {
    // World position to UV
    float u = position.x / ocean_size;
    float v = position.z / ocean_size;
    
    // Wrap to [0, 1] for tiling
    u = u - floorf(u);
    v = v - floorf(v);
    
    // Sample height from FFT texture
    float height = tex2D<float>(height_tex, u, v);
    
    // Sample normal components (packed as float2: nx, nz)
    float2 nxz = tex2D<float2>(normal_tex, u, v);
    float ny = sqrtf(fmaxf(0.0f, 1.0f - nxz.x * nxz.x - nxz.y * nxz.y));
    float3 normal = normalize(make_float3(nxz.x, ny, nxz.y));
    
    // variables for micro slope
    float micro_slope = 0.0f;
    
    // === MICRO DETAIL (CAPILLARY WAVES) ===
    // Multi-layer animated noise for realistic ripples
    // Uses STATIONARY animation pattern - no drift, only in-place morphing
    if (micro_detail_strength > 0.001f) {
        // Wind direction vectors (main and cross-wind)
        float wind_dx = cosf(wind_direction);
        float wind_dz = sinf(wind_direction);
        float cross_dx = -wind_dz;  // Perpendicular
        float cross_dz = wind_dx;
        
        // Morph speed controlled by user parameter (replaces drift with in-place animation)
        float morph = micro_morph_speed;
        
        // Animation intensity based on wind (subtle influence, no drift)
        float wind_factor = fminf(1.0f, sqrtf(fmaxf(0.1f, wind_speed)) * 0.2f);
        
        // === STATIONARY WAVE PATTERN ===
        // Instead of offsetting position (drift), we morph the noise pattern itself
        // This creates realistic wave motion without any directional movement
        
        // === LAYER 1: Primary ripples (stationary, morphing) ===
        // Position is FIXED in world space - no drift offset
        float2 pos1_base = make_float2(position.x, position.z) * micro_detail_scale;
        // Time-based morphing of the pattern itself (oscillating, not drifting)
        float morph_phase1 = time * 0.5f * morph;
        float2 pos1 = make_float2(
            pos1_base.x + sinf(morph_phase1) * wind_factor * 0.3f,
            pos1_base.y + cosf(morph_phase1 * 0.8f) * wind_factor * 0.3f
        );
        
        // === LAYER 2: Secondary ripples (larger, slower morph) ===
        float2 pos2_base = make_float2(position.x, position.z) * micro_detail_scale * 0.5f;
        float morph_phase2 = time * 0.25f * morph;
        // Slight directional bias but oscillating, not drifting
        float2 pos2 = make_float2(
            pos2_base.x + sinf(morph_phase2) * wind_dx * wind_factor * 0.4f 
                        + cosf(morph_phase2 * 1.3f) * 0.2f,
            pos2_base.y + sinf(morph_phase2 * 0.7f) * wind_dz * wind_factor * 0.4f 
                        + sinf(morph_phase2 * 1.1f) * 0.2f
        );
        
        // === LAYER 3: Fine detail (cross-wind bias, fast morph) ===
        float2 pos3_base = make_float2(position.x, position.z) * micro_detail_scale * 2.0f;
        float morph_phase3 = time * 0.8f * morph;
        // Cross-wind direction for variety
        float2 pos3 = make_float2(
            pos3_base.x + sinf(morph_phase3) * cross_dx * wind_factor * 0.15f,
            pos3_base.y + cosf(morph_phase3) * cross_dz * wind_factor * 0.15f
        );
        
        // Sample all layers
        float d = 0.01f;
        
        // Layer 1 derivatives (main ripples)
        float h1_c = fbm(pos1);
        float h1_x = fbm(pos1 + make_float2(d, 0.0f));
        float h1_z = fbm(pos1 + make_float2(0.0f, d));
        
        // Layer 2 derivatives (larger scale, directional bias)
        float h2_c = fbm(pos2);
        float h2_x = fbm(pos2 + make_float2(d, 0.0f));
        float h2_z = fbm(pos2 + make_float2(0.0f, d));
        
        // Layer 3 derivatives (fine detail)
        float h3_c = noise(pos3);  // Single octave for speed
        float h3_x = noise(pos3 + make_float2(d, 0.0f));
        float h3_z = noise(pos3 + make_float2(0.0f, d));
        
        // Combine layers with different weights
        float h_c = h1_c * 0.5f + h2_c * 0.35f + h3_c * 0.15f;
        float h_x = h1_x * 0.5f + h2_x * 0.35f + h3_x * 0.15f;
        float h_z = h1_z * 0.5f + h2_z * 0.35f + h3_z * 0.15f;
        
        float dsdx = (h_x - h_c) / d;
        float dsdz = (h_z - h_c) / d;
        
        // Extract a stable "peak" value for foam rather than the chaotic spatial derivative
        // h_c ranges approximately -1.0 to 1.0. We remap it to 0.0 - 1.0 for foam crest thresholding.
        // This completely eliminates the "accumulating noise" issue from sub-pixel jitter!
        micro_slope = (h_c * 0.5f + 0.5f);
        
        // Perturb normal
        float3 noise_n = normalize(make_float3(-dsdx * micro_detail_strength, 1.0f, -dsdz * micro_detail_strength));
        
        // Blend normals (whiteout blending or straightforward re-normalization)
        normal = normalize(make_float3(
            normal.x + noise_n.x, 
            normal.y + noise_n.y, // Dominant Y
            normal.z + noise_n.z
        ));
    }

    // === FOAM ===
    // Completely switch from Macro steepness to Micro peak height for foam!
    // This allows foam to beautifully trace the small turbulent ripples instead of big waves,
    // and correctly anti-aliases without accumulating firefly pixel noise over passes.
    float slope;
    if (micro_detail_strength > 0.001f) {
        slope = micro_slope;
    } else {
        // Fallback to macro waves if no micro details exist
        slope = sqrtf(nxz.x * nxz.x + nxz.y * nxz.y);
    }
    
    // Scale slope by threshold and intensity
    float base_foam = fmaxf(0.0f, (slope - foam_threshold) * foam_level * 5.0f);
    
    // Break up foam with animated noise based on time
    float2 f_pos = make_float2(position.x, position.z) * foam_noise_scale;
    float foam_noise = fbm(f_pos + make_float2(time * 0.1f, time * 0.15f));
    
    float foam = base_foam * (0.5f + 0.5f * foam_noise);
    
    foam = fminf(1.0f, foam);
    foam = smoothstep(0.2f, 0.8f, foam);
    
    WaterResult res;
    res.normal = normal;
    res.foam = foam;
    res.height = height;
    res.depth = 0.0f;
    res.shore_factor = 0.0f;
    res.caustic_intensity = 0.0f;
    res.water_color = make_float3(0.2f, 0.5f, 0.7f);
    res.absorption = make_float3(1.0f, 1.0f, 1.0f);
    
    return res;
}

// Combined function: chooses FFT or Gerstner based on params
__device__ inline WaterResult evaluateWater(
    const float3& position,
    const float3& baseNormal,
    float time,
    const WaterParams& params
) {
    WaterResult res;
    if (params.use_fft_ocean && params.fft_height_tex != 0) {
        res = sampleFFTOcean(
            position,
            params.fft_ocean_size,
            params.fft_choppiness,
            params.fft_height_tex,
            params.fft_normal_tex,
            params.foam_level,
            params.micro_detail_strength,
            params.micro_detail_scale,
            params.micro_anim_speed,
            params.micro_morph_speed,
            params.foam_noise_scale,
            params.foam_threshold,
            params.wind_direction,
            params.wind_speed,
            params.time
        );
    } else {
        res = evaluateGerstnerWave(
            position, baseNormal, time,
            params.wave_speed,
            params.wave_strength,
            params.wave_frequency
        );
    }

    res.water_color = params.shallow_color;
    res.absorption = calculateWaterAbsorption(0.0f, params.absorption_color, params.absorption_density);

    return res;
}
