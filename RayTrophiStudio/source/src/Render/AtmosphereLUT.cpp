#include "AtmosphereLUT.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "globals.h"
#include "Atmosphere/AtmosphereClimate.h"

AtmosphereLUT::AtmosphereLUT() {
    memset(&data, 0, sizeof(data));
}

AtmosphereLUT::~AtmosphereLUT() {
    cleanup();
}

// Helper for float3 math in host code
inline float host_dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float3 host_mul(float3 a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
inline float3 host_add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

// Helper to compute transmittance analytically during precomputation
inline float3 compute_transmittance_internal(float3 pos, float3 sunDir, float Rg, float Rt, float rH, float mH, const NishitaSkyParams& params, float3 ozoneExt) {
    float b = 2.0f * host_dot(pos, sunDir);
    float c = host_dot(pos, pos) - Rt * Rt;
    float delta = b * b - 4.0f * c;
    if (delta < 0.0f) return make_float3(1.0f, 1.0f, 1.0f);
    
    float dist = (-b + sqrtf(delta)) / 2.0f;
    int numSamples = 40; 
    float stepSize = dist / (float)numSamples;
    float oDR = 0.0f;
    float oDM = 0.0f;
    for (int i = 0; i < numSamples; ++i) {
        float3 sP = host_add(pos, host_mul(sunDir, stepSize * (i + 0.5f)));
        float h = sqrtf(host_dot(sP, sP)) - Rg;
        if (h < 0.0f) h = 0.0f;
        oDR += expf(-h / rH) * stepSize;
        oDM += expf(-h / mH) * stepSize;
    }
    float3 tau = host_add(host_add(host_mul(params.rayleigh_scattering, params.air_density * oDR), 
                                    host_mul(params.mie_scattering, params.dust_density * 1.1f * oDM)),
                                    host_mul(ozoneExt, oDR));
    return make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
}

void AtmosphereLUT::precompute(const NishitaSkyParams& inputParams) {
    if (!initialized) initialize();

    // Effective medium: the climate's humidity grows the aerosol extinction
    // (same formula atmosphere_lut.comp receives as weather.x). Folding it
    // into dust_density ONCE keeps every phase below on the same Mie value.
    NishitaSkyParams params = inputParams;
    params.dust_density *= atmosphere::hygroscopicMieScale(inputParams.derived_relative_humidity);

    float Rg = params.planet_radius;
    float Rt = Rg + params.atmosphere_height;
    float t_kelvin = params.derived_temperature_c + 273.15f;
    float h_scale = t_kelvin / 288.15f;
    float rH = params.rayleigh_density * h_scale;
    float mH = params.mie_density * h_scale;
    float3 ozoneAbs = make_float3(0.000000650f, 0.000001881f, 0.000000085f);
    float3 ozoneExt = host_mul(ozoneAbs, params.ozone_density * params.ozone_absorption_scale);

    // Phase 1: Transmittance LUT (256x64)
    host_transmittance.resize(TRANSMITTANCE_LUT_W * TRANSMITTANCE_LUT_H);
    #pragma omp parallel for num_threads(get_omp_threads_limit())
    for (int y = 0; y < TRANSMITTANCE_LUT_H; ++y) {
        for (int x = 0; x < TRANSMITTANCE_LUT_W; ++x) {
            float u = (float)x / (TRANSMITTANCE_LUT_W - 1.0f);
            float v = (float)y / (TRANSMITTANCE_LUT_H - 1.0f);
            float cosTheta = -0.2f + u * 1.2f; 
            float altitude = v * params.atmosphere_height;
            float3 sunDir = make_float3(sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta)), cosTheta, 0.0f);
            float3 pos = make_float3(0.0f, Rg + altitude, 0.0f);

            float3 transmittance = compute_transmittance_internal(pos, sunDir, Rg, Rt, rH, mH, params, ozoneExt);
            host_transmittance[y * TRANSMITTANCE_LUT_W + x] = make_float4(transmittance.x, transmittance.y, transmittance.z, 1.0f);
        }
    }
    if (g_hasCUDA) {
        uploadCudaTransmittance();
    }

    // Phase 2: Multi-Scattering LUT (32x32)
    host_multi_scatter.resize(MULTI_SCATTER_LUT_RES * MULTI_SCATTER_LUT_RES);
    #pragma omp parallel for num_threads(get_omp_threads_limit())
    for (int y = 0; y < MULTI_SCATTER_LUT_RES; ++y) {
        for (int x = 0; x < MULTI_SCATTER_LUT_RES; ++x) {
            float u = (float)x / (MULTI_SCATTER_LUT_RES - 1.0f);
            float v = (float)y / (MULTI_SCATTER_LUT_RES - 1.0f);
            float altitude = v * params.atmosphere_height;
            
            // Simplified MS Energy Factor (Hillaire 2020 approximation)
            float3 pos = { 0.0f, Rg + altitude, 0.0f };
            // Integrate transmittance across hemisphere to find average energy loss
            float avgTrans = 0.0f;
            for(int i=0; i<8; ++i) {
                float cosTheta = (float)i / 7.0f;
                float3 dir = { sqrtf(1.0f - cosTheta*cosTheta), cosTheta, 0.0f };
                float3 t = compute_transmittance_internal(pos, dir, Rg, Rt, rH, mH, params, ozoneExt);
                avgTrans += (t.x + t.y + t.z) / 3.0f;
            }
            avgTrans /= 8.0f;
            float msFactor = 1.0f / (1.0f - avgTrans);
            host_multi_scatter[y * MULTI_SCATTER_LUT_RES + x] = make_float4(msFactor * 0.1f, msFactor * 0.12f, msFactor * 0.15f, 1.0f);
        }
    }
    if (g_hasCUDA) {
        uploadCudaMultiScatter();
    }

    // Phase 3: SkyView LUT (256x128)
    host_skyview.resize(SKYVIEW_LUT_W * SKYVIEW_LUT_H);
    #pragma omp parallel for num_threads(get_omp_threads_limit())
    for (int y = 0; y < SKYVIEW_LUT_H; ++y) {
        for (int x = 0; x < SKYVIEW_LUT_W; ++x) {
            float u = (float)x / (SKYVIEW_LUT_W - 1.0f);
            float v = (float)y / (SKYVIEW_LUT_H - 1.0f);
            
            float azimuth = u * 2.0f * 3.14159f;
            float cosTheta = 1.0f - v * 2.0f;
            float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
            float3 dir = make_float3(sinTheta * cosf(azimuth), cosTheta, sinTheta * sinf(azimuth));
            float3 sunDir = params.sun_direction;
            float3 camPos = make_float3(0.0f, Rg + params.altitude, 0.0f);

            float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
            float b = 2.0f * host_dot(camPos, dir);
            float c = host_dot(camPos, camPos) - Rt * Rt;
            float d = b * b - 4.0f * c;
            if (d >= 0) {
                float dist = (-b + sqrtf(d)) / 2.0f;
                if (dist > 0) {
                    int steps = 128; // Increased from 32 to reduce banding
                    float step = dist / steps;
                    float3 opticalDepth = make_float3(0, 0, 0);
                    for (int i = 0; i < steps; ++i) {
                        float3 p = host_add(camPos, host_mul(dir, step * (i + 0.5f)));
                        float h = sqrtf(host_dot(p, p)) - Rg;
                        if (h < 0) h = 0;
                        float hr = expf(-h / rH);
                        float hm = expf(-h / mH);
                        
                        // Transmittance to sun (Calculated!)
                        float3 transSun = compute_transmittance_internal(p, sunDir, Rg, Rt, rH, mH, params, ozoneExt);
                        
                        float3 scatR = host_mul(params.rayleigh_scattering, params.air_density * hr);
                        float3 scatM = host_mul(params.mie_scattering, params.dust_density * hm);
                        
                        float mu = host_dot(dir, sunDir);
                        float phaseR = 3.0f / (16.0f * 3.14159f) * (1.0f + mu * mu);
                        float g = params.mie_anisotropy;
                        float phaseM = (1.0f - g * g) / (4.0f * 3.14159f * powf(1.0f + g * g - 2.0f * g * mu, 1.5f));
                        // CLAMP MIE PHASE for LUT: Remove the sharp sun peak from the LUT to prevent "blocky" artifacts.
                        // The sharp sun disk/glow will be rendered procedurally in the shader.
                        phaseM = fminf(phaseM, 2.0f); 
                        
                        float3 inScat = host_add(host_mul(scatR, phaseR), host_mul(scatM, phaseM));
                        float3 stepL = make_float3(inScat.x * transSun.x, inScat.y * transSun.y, inScat.z * transSun.z);
                        
                        float3 currentTrans = make_float3(expf(-opticalDepth.x), expf(-opticalDepth.y), expf(-opticalDepth.z));
                        radiance = host_add(radiance, host_mul(stepL, step * currentTrans.x)); 
                        
                        opticalDepth.x += (scatR.x + scatM.x) * step;
                        opticalDepth.y += (scatR.y + scatM.y) * step;
                        opticalDepth.z += (scatM.z + scatR.z) * step; // Note: Reusing scats
                    }
                }
            }
            host_skyview[y * SKYVIEW_LUT_W + x] = make_float4(radiance.x * params.atmosphere_intensity, radiance.y * params.atmosphere_intensity, radiance.z * params.atmosphere_intensity, 1.0f);
        }
    }
    if (g_hasCUDA) {
        uploadCudaSkyView();
    }
}

// ═══════════════════════════════════════════════════════════
// BILINEAR SAMPLING FOR CPU (Matches GPU tex2D)
// ═══════════════════════════════════════════════════════════
float3 AtmosphereLUT::sampleTransmittance(float cosTheta, float altitude, float atmosphereHeight) const {
    if (host_transmittance.empty()) return make_float3(1.0f, 1.0f, 1.0f);
    
    // UV Mapping based on precompute
    float u = (cosTheta + 0.2f) / 1.2f;
    float v = altitude / std::max(1.0f, atmosphereHeight); 
    
    // Clamp
    u = std::max(0.0f, std::min(0.999f, u));
    v = std::max(0.0f, std::min(0.999f, v));
    
    float x = u * (TRANSMITTANCE_LUT_W - 1);
    float y = v * (TRANSMITTANCE_LUT_H - 1);
    int x0 = (int)x; int x1 = std::min(x0 + 1, TRANSMITTANCE_LUT_W - 1);
    int y0 = (int)y; int y1 = std::min(y0 + 1, TRANSMITTANCE_LUT_H - 1);
    float dx = x - x0; float dy = y - y0;
    
    auto get = [&](int ix, int iy) {
        float4 f = host_transmittance[iy * TRANSMITTANCE_LUT_W + ix];
        return make_float3(f.x, f.y, f.z);
    };
    
    float3 c00 = get(x0, y0); float3 c10 = get(x1, y0);
    float3 c01 = get(x0, y1); float3 c11 = get(x1, y1);
    
    float3 c0 = host_add(host_mul(c00, 1.0f - dx), host_mul(c10, dx));
    float3 c1 = host_add(host_mul(c01, 1.0f - dx), host_mul(c11, dx));
    return host_add(host_mul(c0, 1.0f - dy), host_mul(c1, dy));
}

float3 AtmosphereLUT::sampleSkyView(float3 rayDir, float3 sunDir, float Rg, float Rt) const {
    if (host_skyview.empty()) return make_float3(0.0f, 0.0f, 0.0f);
    
    // Map to Azure/Blender SkyView UV coords
    float azimuth = atan2f(rayDir.z, rayDir.x);
    if (azimuth < 0.0f) azimuth += 2.0f * 3.14159f;
    
    float u = azimuth / (2.0f * 3.14159f);
    float v = (1.0f - rayDir.y) * 0.5f; 
    
    u = std::max(0.0f, std::min(0.999f, u));
    v = std::max(0.0f, std::min(0.999f, v));
    
    float x = u * (SKYVIEW_LUT_W - 1);
    float y = v * (SKYVIEW_LUT_H - 1);
    int x0 = (int)x; int x1 = (x0 + 1) % SKYVIEW_LUT_W; // Azimuth wraps
    int y0 = (int)y; int y1 = std::min(y0 + 1, SKYVIEW_LUT_H - 1);
    float dx = x - x0; float dy = y - y0;
    
    auto get = [&](int ix, int iy) {
        float4 f = host_skyview[iy * SKYVIEW_LUT_W + ix];
        return make_float3(f.x, f.y, f.z);
    };
    
    float3 c00 = get(x0, y0); float3 c10 = get(x1, y0);
    float3 c01 = get(x0, y1); float3 c11 = get(x1, y1);
    
    float3 c0 = host_add(host_mul(c00, 1.0f - dx), host_mul(c10, dx));
    float3 c1 = host_add(host_mul(c01, 1.0f - dx), host_mul(c11, dx));
    return host_add(host_mul(c0, 1.0f - dy), host_mul(c1, dy));
}
