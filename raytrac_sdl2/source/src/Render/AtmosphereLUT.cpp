#include "AtmosphereLUT.h"
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "globals.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        } \
    } while (0)

AtmosphereLUT::AtmosphereLUT() {
    data.transmittance_lut = 0;
    data.skyview_lut = 0;
    data.integrated_multi_scattering = make_float3(0.0f, 0.0f, 0.0f);
}

AtmosphereLUT::~AtmosphereLUT() {
    cleanup();
}

void AtmosphereLUT::cleanup() {
    if (data.transmittance_lut) {
        cudaDestroyTextureObject(data.transmittance_lut);
        data.transmittance_lut = 0;
    }
    if (data.skyview_lut) {
        cudaDestroyTextureObject(data.skyview_lut);
        data.skyview_lut = 0;
    }
    if (transmittance_array) {
        cudaFreeArray(transmittance_array);
        transmittance_array = nullptr;
    }
    if (skyview_array) {
        cudaFreeArray(skyview_array);
        skyview_array = nullptr;
    }
    if (multi_scatter_array) {
        cudaFreeArray(multi_scatter_array);
        multi_scatter_array = nullptr;
    }
    if (aerial_perspective_array) {
        cudaFreeArray(aerial_perspective_array);
        aerial_perspective_array = nullptr;
    }
    if (data.multi_scattering_lut) {
        cudaDestroyTextureObject(data.multi_scattering_lut);
        data.multi_scattering_lut = 0;
    }
    if (data.aerial_perspective_lut) {
        cudaDestroyTextureObject(data.aerial_perspective_lut);
        data.aerial_perspective_lut = 0;
    }
    initialized = false;
}

void AtmosphereLUT::initialize() {
    cleanup();

    // 1. Allocate Transmittance LUT (256x64 RGBA32F)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&transmittance_array, &channelDesc, TRANSMITTANCE_LUT_W, TRANSMITTANCE_LUT_H));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = transmittance_array;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear; // Bilinear filtering for quality
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    CUDA_CHECK(cudaCreateTextureObject(&data.transmittance_lut, &resDesc, &texDesc, nullptr));

    // 2. Allocate SkyView LUT (256x128 RGBA32F)
    CUDA_CHECK(cudaMallocArray(&skyview_array, &channelDesc, SKYVIEW_LUT_W, SKYVIEW_LUT_H));
    resDesc.res.array.array = skyview_array;
    CUDA_CHECK(cudaCreateTextureObject(&data.skyview_lut, &resDesc, &texDesc, nullptr));
    
    // 3. Allocate Multi-Scattering LUT (32x32 RGBA32F)
    CUDA_CHECK(cudaMallocArray(&multi_scatter_array, &channelDesc, MULTI_SCATTER_LUT_RES, MULTI_SCATTER_LUT_RES));
    resDesc.res.array.array = multi_scatter_array;
    CUDA_CHECK(cudaCreateTextureObject(&data.multi_scattering_lut, &resDesc, &texDesc, nullptr));

    // 4. Allocate Aerial Perspective LUT (32x32x32 RGBA32F)
    cudaExtent extent = make_cudaExtent(AERIAL_PERSPECTIVE_RES, AERIAL_PERSPECTIVE_RES, AERIAL_PERSPECTIVE_RES);
    CUDA_CHECK(cudaMalloc3DArray(&aerial_perspective_array, &channelDesc, extent));
    resDesc.res.array.array = aerial_perspective_array;
    CUDA_CHECK(cudaCreateTextureObject(&data.aerial_perspective_lut, &resDesc, &texDesc, nullptr));

    if (data.transmittance_lut == 0 || data.skyview_lut == 0 || data.multi_scattering_lut == 0 || data.aerial_perspective_lut == 0) {
        SCENE_LOG_ERROR("AtmospehreLUT: Failed to create texture objects!");
        initialized = false;
        return;
    }

    initialized = true;
    SCENE_LOG_INFO("AtmosphereLUT: GPU Resources (2D & 3D) allocated successfully.");
}

// Helper for float3 math in host code
inline float host_dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float3 host_mul(float3 a, float b) { return { a.x * b, a.y * b, a.z * b }; }
inline float3 host_add(float3 a, float3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }

// Helper to compute transmittance analytically during precomputation
inline float3 compute_transmittance_internal(float3 pos, float3 sunDir, float Rg, float Rt, float rH, float mH, const NishitaSkyParams& params, float3 ozoneExt) {
    float b = 2.0f * host_dot(pos, sunDir);
    float c = host_dot(pos, pos) - Rt * Rt;
    float delta = b * b - 4.0f * c;
    if (delta < 0.0f) return { 1.0f, 1.0f, 1.0f };
    
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
    return { expf(-tau.x), expf(-tau.y), expf(-tau.z) };
}

void AtmosphereLUT::precompute(const NishitaSkyParams& params) {
    if (!initialized) initialize();
    
    float Rg = params.planet_radius;
    float Rt = Rg + params.atmosphere_height;
    float t_kelvin = params.temperature + 273.15f;
    float h_scale = t_kelvin / 288.15f;
    float rH = params.rayleigh_density * h_scale;
    float mH = params.mie_density * h_scale;
    float3 ozoneAbs = { 0.000000650f, 0.000001881f, 0.000000085f };
    float3 ozoneExt = host_mul(ozoneAbs, params.ozone_density * params.ozone_absorption_scale);

    // Phase 1: Transmittance LUT (256x64)
    std::vector<float4> host_trans_data(TRANSMITTANCE_LUT_W * TRANSMITTANCE_LUT_H);
    #pragma omp parallel for
    for (int y = 0; y < TRANSMITTANCE_LUT_H; ++y) {
        for (int x = 0; x < TRANSMITTANCE_LUT_W; ++x) {
            float u = (float)x / (TRANSMITTANCE_LUT_W - 1.0f);
            float v = (float)y / (TRANSMITTANCE_LUT_H - 1.0f);
            float cosTheta = -0.2f + u * 1.2f; 
            float altitude = v * params.atmosphere_height;
            float3 sunDir = { sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta)), cosTheta, 0.0f };
            float3 pos = { 0.0f, Rg + altitude, 0.0f };

            float3 transmittance = compute_transmittance_internal(pos, sunDir, Rg, Rt, rH, mH, params, ozoneExt);
            host_trans_data[y * TRANSMITTANCE_LUT_W + x] = { transmittance.x, transmittance.y, transmittance.z, 1.0f };
        }
    }
    cudaMemcpy2DToArray(transmittance_array, 0, 0, host_trans_data.data(), TRANSMITTANCE_LUT_W * sizeof(float4), TRANSMITTANCE_LUT_W * sizeof(float4), TRANSMITTANCE_LUT_H, cudaMemcpyHostToDevice);

    // Phase 2: Multi-Scattering LUT (32x32)
    std::vector<float4> host_ms_data(MULTI_SCATTER_LUT_RES * MULTI_SCATTER_LUT_RES);
    #pragma omp parallel for
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
            host_ms_data[y * MULTI_SCATTER_LUT_RES + x] = { msFactor * 0.1f, msFactor * 0.12f, msFactor * 0.15f, 1.0f };
        }
    }
    cudaMemcpy2DToArray(multi_scatter_array, 0, 0, host_ms_data.data(), MULTI_SCATTER_LUT_RES * sizeof(float4), MULTI_SCATTER_LUT_RES * sizeof(float4), MULTI_SCATTER_LUT_RES, cudaMemcpyHostToDevice);

    // Phase 3: SkyView LUT (256x128)
    std::vector<float4> host_sky_data(SKYVIEW_LUT_W * SKYVIEW_LUT_H);
    #pragma omp parallel for
    for (int y = 0; y < SKYVIEW_LUT_H; ++y) {
        for (int x = 0; x < SKYVIEW_LUT_W; ++x) {
            float u = (float)x / (SKYVIEW_LUT_W - 1.0f);
            float v = (float)y / (SKYVIEW_LUT_H - 1.0f);
            
            float azimuth = u * 2.0f * 3.14159f;
            float cosTheta = 1.0f - v * 2.0f;
            float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
            float3 dir = { sinTheta * cosf(azimuth), cosTheta, sinTheta * sinf(azimuth) };
            float3 sunDir = params.sun_direction;
            float3 camPos = { 0.0f, Rg + params.altitude, 0.0f };

            float3 radiance = { 0, 0, 0 };
            float b = 2.0f * host_dot(camPos, dir);
            float c = host_dot(camPos, camPos) - Rt * Rt;
            float d = b * b - 4.0f * c;
            if (d >= 0) {
                float dist = (-b + sqrtf(d)) / 2.0f;
                if (dist > 0) {
                    int steps = 128; // Increased from 32 to reduce banding
                    float step = dist / steps;
                    float3 opticalDepth = { 0, 0, 0 };
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
                        float3 stepL = { inScat.x * transSun.x, inScat.y * transSun.y, inScat.z * transSun.z };
                        
                        float3 currentTrans = { expf(-opticalDepth.x), expf(-opticalDepth.y), expf(-opticalDepth.z) };
                        radiance = host_add(radiance, host_mul(stepL, step * currentTrans.x)); 
                        
                        opticalDepth.x += (scatR.x + scatM.x) * step;
                        opticalDepth.y += (scatR.y + scatM.y) * step;
                        opticalDepth.z += (scatR.z + scatM.z) * step;
                    }
                }
            }
            host_sky_data[y * SKYVIEW_LUT_W + x] = { radiance.x * params.sun_intensity, radiance.y * params.sun_intensity, radiance.z * params.sun_intensity, 1.0f };
        }
    }
    cudaMemcpy2DToArray(skyview_array, 0, 0, host_sky_data.data(), SKYVIEW_LUT_W * sizeof(float4), SKYVIEW_LUT_W * sizeof(float4), SKYVIEW_LUT_H, cudaMemcpyHostToDevice);
}
