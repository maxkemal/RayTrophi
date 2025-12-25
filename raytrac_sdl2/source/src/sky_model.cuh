#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include "vec3_utils.cuh"
#include "World.h"

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
    // Ray: P = camPos + t * dir
    // Sphere: |P|^2 = Rt^2
    
    float3 p = camPos;
    float a = dot(dir, dir);
    float b = 2.0f * dot(dir, p);
    float c = dot(p, p) - Rt * Rt;
    float delta = b * b - 4.0f * a * c;
    
    if (delta < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    
    float t1 = (-b - sqrtf(delta)) / (2.0f * a);
    float t2 = (-b + sqrtf(delta)) / (2.0f * a);
    float t = (t1 >= 0.0f) ? t1 : t2;
    // Note: Assuming starting from inside atmosphere usually.
    // If t < 0, we might be looking away from intersection behind us, but here we cover sky.
    if (t < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    
    // Ray Marching Settings
    int numSamples = 16; 
    float stepSize = t / (float)numSamples;
    
    float3 totalRayleigh = make_float3(0.0f, 0.0f, 0.0f);
    float3 totalMie = make_float3(0.0f, 0.0f, 0.0f);
    
    float opticalDepthRayleigh = 0.0f;
    float opticalDepthMie = 0.0f;
    
    // Phase functions
    float mu = dot(dir, sunDir);
    float phaseR = 3.0f / (16.0f * CUDART_PI_F) * (1.0f + mu * mu);
    float g = params.mie_anisotropy;
    float phaseM = 3.0f / (8.0f * CUDART_PI_F) * ((1.0f - g * g) * (1.0f + mu * mu)) / ((2.0f + g * g) * powf(1.0f + g * g - 2.0f * g * mu, 1.5f));
    
    float currentT = 0.0f;
    
    for (int i = 0; i < numSamples; ++i) {
        float3 samplePos = p + dir * (currentT + stepSize * 0.5f);
        float height = length(samplePos) - Rg;
        
        if (height < 0.0f) height = 0.0f; // clamp
        
        float hr = expf(-height / params.rayleigh_density);
        float hm = expf(-height / params.mie_density);
        
        opticalDepthRayleigh += hr * stepSize;
        opticalDepthMie += hm * stepSize;
        
        // Light path to sun (transmittance)
        // Ray intersect atmosphere from samplePos to sunDir
        float b_light = 2.0f * dot(sunDir, samplePos);
        float c_light = dot(samplePos, samplePos) - Rt * Rt;
        float delta_light = b_light * b_light - 4.0f * c_light;
        
        if (delta_light >= 0.0f) {
             float t_light = (-b_light + sqrtf(delta_light)) / 2.0f;
             
             // Reduced samples for secondary ray for performance
             int numLightSamples = 4;
             float lightStep = t_light / (float)numLightSamples;
             float lightOpticalRayleigh = 0.0f;
             float lightOpticalMie = 0.0f;
             
             for(int j=0; j<numLightSamples; ++j) {
                 float3 lightSamplePos = samplePos + sunDir * (lightStep * (j + 0.5f));
                 float lightHeight = length(lightSamplePos) - Rg;
                 if(lightHeight < 0.0f) lightHeight = 0.0f;
                 lightOpticalRayleigh += expf(-lightHeight / params.rayleigh_density) * lightStep;
                 lightOpticalMie += expf(-lightHeight / params.mie_density) * lightStep;
             }
             
             float3 tau = params.rayleigh_scattering * (opticalDepthRayleigh + lightOpticalRayleigh) + 
                          params.mie_scattering * 1.1f * (opticalDepthMie + lightOpticalMie);
                          
             float3 attenuation = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
             
             totalRayleigh += attenuation * hr * stepSize;
             totalMie += attenuation * hm * stepSize;
        }
        
        currentT += stepSize;
    }
    
    // Apply air and dust density multipliers
    float3 rayleighScatter = params.rayleigh_scattering * params.air_density;
    float3 mieScatter = params.mie_scattering * params.dust_density;
    
    float3 L = (totalRayleigh * rayleighScatter * phaseR + 
                totalMie * mieScatter * phaseM) * params.sun_intensity;
    
    // Apply ozone (affects blue channel - simple approximation)
    float ozoneFactor = params.ozone_density;
    L.x *= (1.0f + 0.1f * ozoneFactor);  // Slightly reduce red/enhance absorption
    L.z *= (1.0f + 0.3f * ozoneFactor);  // Boost blue
    
    // Add Sun Disk using sun_size (in degrees)
    float sunSizeDeg = params.sun_size;
    float elevationFactor = 1.0f;
    if (params.sun_elevation < 15.0f) {
        elevationFactor = 1.0f + (15.0f - fmaxf(params.sun_elevation, -10.0f)) * 0.04f;
    }
    sunSizeDeg *= elevationFactor;
    
    float sun_radius = sunSizeDeg * (CUDART_PI_F / 180.0f) * 0.5f; // Half angle in radians
    if (dot(dir, sunDir) > cosf(sun_radius)) {
         float3 tau = rayleighScatter * opticalDepthRayleigh + 
                      mieScatter * 1.1f * opticalDepthMie;
         float3 transmittance = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
         L += transmittance * params.sun_intensity * 1000.0f; // Direct sun
    }
    
    // Night Sky: Stars and Moon
    float nightFactor = 1.0f - fmaxf(0.0f, fminf(1.0f, (params.sun_elevation + 5.0f) / 20.0f));
    
    if (nightFactor > 0.01f && params.stars_intensity > 0.0f) {
        // Procedural stars logic here if needed, simplified for now
        // Or reuse CloudNoise hash if included
    }
    
    // Simplified Moon (Visual only, no lighting contribution here)
    if (params.moon_enabled && nightFactor > 0.01f && params.moon_intensity > 0.0f) {
         float moonElevRad = params.moon_elevation * CUDART_PI_F / 180.0f;
         float moonAzimRad = params.moon_azimuth * CUDART_PI_F / 180.0f;
         float3 moonDir = make_float3(
            cosf(moonElevRad) * sinf(moonAzimRad),
            sinf(moonElevRad),
            cosf(moonElevRad) * cosf(moonAzimRad)
         );
         
         float moon_radius = params.moon_size * (CUDART_PI_F / 180.0f) * 0.5f;
         if (dot(dir, moonDir) > cosf(moon_radius)) {
             float3 moonColor = make_float3(0.9f, 0.9f, 0.95f);
             // Basic phase calc could go here
             L += moonColor * params.moon_intensity * nightFactor * 10.0f;
         }
    }

    return L;
}
