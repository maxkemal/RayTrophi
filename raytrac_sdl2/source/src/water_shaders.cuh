#pragma once
#include <cuda_runtime.h>
#include <vector_types.h>
#include "vec3_utils.cuh"


// Returns normal and foam intensity
struct WaterResult {
    float3 normal;
    float foam;
    float height;
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
    return res;
}
