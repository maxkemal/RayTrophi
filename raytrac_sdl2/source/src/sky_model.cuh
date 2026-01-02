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

// Hash for stars
__device__ inline float hash33_s(float3 p) {
    p.x = fmodf(p.x, 50.0f);
    p.y = fmodf(p.y, 50.0f);
    p.z = fmodf(p.z, 50.0f);
    float d = dot(p, make_float3(12.9898f, 78.233f, 53.539f));
    return fract(sinf(d) * 43758.5453f);
}

// Helper for Ozone Absorption (Chappuis band approximation)
// Returns absorption coefficient for RGB
__device__ inline float3 getOzoneAbsorption() {
    // Peak absorption in orange/red (600nm), less in blue
    // This value is approximate. Scale by ozone_density.
    return make_float3(0.0065f, 0.015f, 0.0002f) * 10.0f; 
}

// Procedural Stars
__device__ inline float stars(float3 dir, float density) {
    float3 p = dir * 200.0f; // Scale
    // If hash not available, use local simple hash
    float3 p3 = fract(p * 0.1031f);
    float d = dot(p3, make_float3(p3.y + 33.33f, p3.z + 33.33f, p3.x + 33.33f));
    p3.x += d; p3.y += d; p3.z += d;
    float starVal = fract((p3.x + p3.y) * p3.z);
    
    return smoothstep(1.0f - density, 1.0f, starVal);
}

// ═══════════════════════════════════════════════════════════
// NISHITA SKY MODEL - GPU IMPLEMENTATION
// ═══════════════════════════════════════════════════════════
__device__ inline float3 calculate_nishita_sky_gpu(const float3& ray_dir, const NishitaSkyParams& params) {
    // Setup vectors
    float3 dir = normalize(ray_dir);
    float3 sunDir = normalize(params.sun_direction);
    
    // Moon setup
    float moonElevRad = params.moon_elevation * CUDART_PI_F / 180.0f;
    float moonAzimRad = params.moon_azimuth * CUDART_PI_F / 180.0f;
    float3 moonDir = make_float3(
        cosf(moonElevRad) * sinf(moonAzimRad),
        sinf(moonElevRad),
        cosf(moonElevRad) * cosf(moonAzimRad)
    );
    
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
    
    // Moon Phase (simplified)
    float muMoon = dot(dir, moonDir);
    float phaseMMoon = 3.0f / (8.0f * CUDART_PI_F) * ((1.0f - g * g) * (1.0f + muMoon * muMoon)) / ((2.0f + g * g) * powf(1.0f + g * g - 2.0f * g * muMoon, 1.5f));
    
    float currentT = 0.0f;
    float nightFactor = 1.0f - fmaxf(0.0f, fminf(1.0f, (params.sun_elevation + 5.0f) / 10.0f));
    bool doMoonScattering = (params.moon_enabled && nightFactor > 0.01f && params.moon_intensity > 0.0f);

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
        
        // --- MOON LIGHT INTEGRATION ---
        if (doMoonScattering) {
            float b_m = 2.0f * dot(moonDir, samplePos);
            float c_m = dot(samplePos, samplePos) - Rt * Rt;
            float delta_m = b_m * b_m - 4.0f * c_m;
            
            if (delta_m >= 0.0f) {
                 float t_m = (-b_m + sqrtf(delta_m)) / 2.0f;
                 float lightStep = t_m / 2.0f; // Fewer samples for moon
                 float lightODR = 0.0f, lightODM = 0.0f, lightODO = 0.0f;
                 
                 for(int j=0; j<2; ++j) {
                     float3 lsPos = samplePos + moonDir * (lightStep * (j + 0.5f));
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
                 
                 // Add moon contribution (dimmer)
                 // Moon light is basically Sun light reflected, so simpler spectrum (white-ish)
                 float3 moonLightColor = make_float3(0.5f, 0.6f, 0.8f) * params.moon_intensity * 0.1f; // Scaled down
                 
                 // Accumulate into same totals but effectively it's a second light source
                 // We add it to the final color integration directly here or via separate accumulation
                 // Simpler to add to separate accumulator? No, `totalRayleigh` multiplies separate Phase.
                 // Actually `totalRayleigh` is used for both. We must separate them unless phase is same.
                 // Phase is different (Sun vs Moon).
                 // Let's modify the final sum line instead.
                 // Hack: Accumulate Moon *contribution* directly into a separate "MoonLight" variable?
                 // No, standard way is: L += (BetaR * PhaseR + BetaM * PhaseM) * LightIn * Attenuation * Step
                 
                 // Wait, Rayleigh phase depends on angle to light source. So we need `phaseRMoon`.
                 float muMoon = dot(dir, moonDir);
                 float phaseRMoon = 3.0f / (16.0f * CUDART_PI_F) * (1.0f + muMoon * muMoon);
                 
                 float3 scatter = (params.rayleigh_scattering * params.air_density * phaseRMoon * hr + 
                                   params.mie_scattering * params.dust_density * phaseMMoon * hm);
                                   
                 L += scatter * moonLightColor * att * stepSize; 
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
    
    // Sun Disk
    float sunSizeDeg = params.sun_size;
    float elevationFactor = 1.0f; // Defined here
    if (params.sun_elevation < 15.0f) elevationFactor = 1.0f + (15.0f - fmaxf(params.sun_elevation, -10.0f)) * 0.04f;
    sunSizeDeg *= elevationFactor;
    float sun_radius = sunSizeDeg * (CUDART_PI_F / 180.0f) * 0.5f;
    if (dot(dir, sunDir) > cosf(sun_radius)) {
         float3 tau = rayleighS * opticalDepthRayleigh + 
                      mieS * 1.1f * opticalDepthMie + 
                      ozoneAbs * opticalDepthOzone;
         float3 transmittance = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
         L += transmittance * params.sun_intensity * 1000.0f; 
    }
    
    // Moon Disk (Visual)
    if (params.moon_enabled && nightFactor > 0.01f && params.moon_intensity > 0.0f) {
         float moon_radius = params.moon_size * (CUDART_PI_F / 180.0f) * 0.5f;
         if (dot(dir, moonDir) > cosf(moon_radius)) {
             float3 moonColor = make_float3(0.9f, 0.9f, 0.95f);
             // Attenuation for moon itself
             float3 tau = rayleighS * opticalDepthRayleigh + 
                          mieS * 1.1f * opticalDepthMie + 
                          ozoneAbs * opticalDepthOzone;
             float3 transmittance = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
             
             L += moonColor * params.moon_intensity * nightFactor * 50.0f * transmittance;
         }
    }
    
    // Stars
    if (nightFactor > 0.1f && params.stars_intensity > 0.0f) {
        // Simple stars - only visible where alpha is low (optical depth high? no, low OD means clear sky)
        // Stars are "infinite distance". Transmittance is for whole atmosphere.
        float3 tau = rayleighS * opticalDepthRayleigh + mieS * opticalDepthMie; // + Ozone
        float3 transmittance = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
        float starVis = (transmittance.x + transmittance.y + transmittance.z) / 3.0f;
        
        if (starVis > 0.1f) {
            float s = stars(dir, 0.002f); // 0.002 density
            L += make_float3(s,s,s) * params.stars_intensity * nightFactor * starVis;
        }
    }

    return L;
}
