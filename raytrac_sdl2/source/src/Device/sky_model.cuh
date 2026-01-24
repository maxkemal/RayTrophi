#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include "vec3_utils.cuh"
#include "World.h"

// ═══════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════
// fract, floor_float3, smoothstep are now in vec3_utils.cuh


// Helper for Ozone Absorption (Chappuis band approximation)
// Returns absorption coefficient for RGB
__device__ inline float3 getOzoneAbsorption() {
    // Peak absorption in orange/red (600nm), less in blue
    // This value is approximate. Scale by ozone_density.
    return make_float3(0.0065f, 0.015f, 0.0002f) * 10.0f; 
}


// ═══════════════════════════════════════════════════════════
// NISHITA SKY MODEL - GPU IMPLEMENTATION
// ═══════════════════════════════════════════════════════════
__device__ inline float3 calculate_nishita_sky_gpu(const float3& ray_dir, const NishitaSkyParams& params) {
    // Setup vectors
    float3 dir = normalize(ray_dir);
    float3 sunDir = normalize(params.sun_direction);
    
    
    // Planet/atmosphere dimensions (all in METERS)
    float Rg = params.planet_radius;                        // Ground radius (meters)
    float Rt = Rg + params.atmosphere_height;               // Top of atmosphere (meters)
    
    // Camera position with altitude
    float cameraAltitude = params.altitude;                 // Altitude in meters
    float3 camPos = make_float3(0.0f, Rg + cameraAltitude, 0.0f);
    
    // Ray-Sphere Intersection
    float3 p = camPos;
    float a = dot(dir, dir);
    float b = 2.0f * dot(dir, p);
    float c = dot(p, p) - Rt * Rt;
    float delta = b * b - 4.0f * a * c;
    
    if (delta < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    
    float t1 = (-b - sqrtf(delta)) / (2.0f * a);
    float t2 = (-b + sqrtf(delta)) / (2.0f * a);
    float t = (t1 >= 0.0f) ? t1 : t2;
    if (t < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    
    // Ray Marching Settings
    int numSamples = 16; 
    float stepSize = t / (float)numSamples;
    
    float3 totalRayleigh = make_float3(0.0f, 0.0f, 0.0f);
    float3 totalMie = make_float3(0.0f, 0.0f, 0.0f);
    
    float opticalDepthRayleigh = 0.0f;
    float opticalDepthMie = 0.0f;
    float opticalDepthOzone = 0.0f;
    
    // Phase functions
    float mu = dot(dir, sunDir);
    float phaseR = 3.0f / (16.0f * CUDART_PI_F) * (1.0f + mu * mu);
    float g = params.mie_anisotropy;
    float phaseM = 3.0f / (8.0f * CUDART_PI_F) * ((1.0f - g * g) * (1.0f + mu * mu)) / ((2.0f + g * g) * powf(1.0f + g * g - 2.0f * g * mu, 1.5f));
    
    
    float currentT = 0.0f;

    float3 ozoneAbs = getOzoneAbsorption() * params.ozone_density;

    // DEFINE L HERE so it can be used for moon accumulation
    float3 L = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < numSamples; ++i) {
        float3 samplePos = p + dir * (currentT + stepSize * 0.5f);
        float height = length(samplePos) - Rg;
        
        if (height < 0.0f) height = 0.0f; 
        
        float hr = expf(-height / params.rayleigh_density);
        float hm = expf(-height / params.mie_density);
        
        // Ozone distribution (layer between 10km and 40km approx, peak at 25km)
        // Normalized height h: 25km is approx center.
        // Simple tent function or gaussian
        float ho = fmaxf(0.0f, 1.0f - fabsf(height - 25000.0f) / 15000.0f);
        
        opticalDepthRayleigh += hr * stepSize;
        opticalDepthMie += hm * stepSize;
        opticalDepthOzone += ho * stepSize;
        
        // --- SUN LIGHT INTEGRATION ---
        {
            float b_light = 2.0f * dot(sunDir, samplePos);
            float c_light = dot(samplePos, samplePos) - Rt * Rt;
            float delta_light = b_light * b_light - 4.0f * c_light;
            
            if (delta_light >= 0.0f) {
                 float t_light = (-b_light + sqrtf(delta_light)) / 2.0f;
                 float lightStep = t_light / 4.0f;
                 float lightODR = 0.0f, lightODM = 0.0f, lightODO = 0.0f;
                 
                 for(int j=0; j<4; ++j) {
                     float3 lsPos = samplePos + sunDir * (lightStep * (j + 0.5f));
                     float lh = length(lsPos) - Rg;
                     if(lh < 0.0f) lh = 0.0f;
                     lightODR += expf(-lh / params.rayleigh_density) * lightStep;
                     lightODM += expf(-lh / params.mie_density) * lightStep;
                     lightODO += fmaxf(0.0f, 1.0f - fabsf(lh - 25000.0f) / 15000.0f) * lightStep;
                 }
                 
                 float3 tau = params.rayleigh_scattering * (opticalDepthRayleigh + lightODR) + 
                              params.mie_scattering * 1.1f * (opticalDepthMie + lightODM) +
                              ozoneAbs * (opticalDepthOzone + lightODO);
                              
                 float3 att = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
                 
                 totalRayleigh += att * hr * stepSize;
                 totalMie += att * hm * stepSize;
            }
        }
        
        
        currentT += stepSize;
    }
    
    float3 rayleighS = params.rayleigh_scattering * params.air_density;
    float3 mieS = params.mie_scattering * params.dust_density;
    
    // Sun Contribution + Add to existing L (Moon)
    float3 L_Sun = (totalRayleigh * rayleighS * phaseR + totalMie * mieS * phaseM) * params.sun_intensity;
    
    L += L_Sun;
    
    // Apply ozone to total (Affects sunlight mostly)
    float ozoneFactor = params.ozone_density;
    L.x *= (1.0f + 0.1f * ozoneFactor);  
    L.z *= (1.0f + 0.3f * ozoneFactor);  
    
    // Sun Disk with Limb Darkening
    float sunSizeDeg = params.sun_size;
    float elevationFactor = 1.0f;
    if (params.sun_elevation < 15.0f) elevationFactor = 1.0f + (15.0f - fmaxf(params.sun_elevation, -10.0f)) * 0.04f;
    sunSizeDeg *= elevationFactor;
    float sun_radius = sunSizeDeg * (CUDART_PI_F / 180.0f) * 0.5f;
    float sun_cos = dot(dir, sunDir);
    float sun_cos_threshold = cosf(sun_radius);
    
    if (sun_cos > sun_cos_threshold) {
         // Calculate radial position on sun disk (0 = center, 1 = edge)
         float angular_dist = acosf(fminf(1.0f, sun_cos));
         float radial_pos = angular_dist / sun_radius;
         
         // LIMB DARKENING: Sun is brighter at center, darker at edges
         float u = 0.6f; // Limb darkening coefficient
         float cosine_mu = sqrtf(fmaxf(0.0f, 1.0f - radial_pos * radial_pos));
         float limbDarkening = 1.0f - u * (1.0f - cosine_mu);
         
         // Smooth edge transition
         float edge_t = fmaxf(0.0f, fminf(1.0f, (radial_pos - 0.85f) / 0.15f));
         float edgeSoftness = 1.0f - edge_t * edge_t * (3.0f - 2.0f * edge_t);
         
         float3 tau = rayleighS * opticalDepthRayleigh + 
                      mieS * 1.1f * opticalDepthMie + 
                      ozoneAbs * opticalDepthOzone;
         float3 transmittance = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
         L += transmittance * params.sun_intensity * 1000.0f * limbDarkening * edgeSoftness;
    }
    

    return L;
}
