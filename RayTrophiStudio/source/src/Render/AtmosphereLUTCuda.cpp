// Studio/offline CUDA residency for the shared atmosphere LUT calculator.
// Select this module or AtmosphereLUTHost.cpp at link time, never both.
#include "AtmosphereLUT.h"
#include "globals.h"
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        } \
    } while (0)

void AtmosphereLUT::cleanup() {
    if (g_hasCUDA && data.transmittance_lut) {
        cudaDestroyTextureObject(data.transmittance_lut);
        data.transmittance_lut = 0;
    }
    if (g_hasCUDA && data.skyview_lut) {
        cudaDestroyTextureObject(data.skyview_lut);
        data.skyview_lut = 0;
    }
    if (g_hasCUDA && transmittance_array) {
        cudaFreeArray(transmittance_array);
        transmittance_array = nullptr;
    }
    if (g_hasCUDA && skyview_array) {
        cudaFreeArray(skyview_array);
        skyview_array = nullptr;
    }
    if (g_hasCUDA && multi_scatter_array) {
        cudaFreeArray(multi_scatter_array);
        multi_scatter_array = nullptr;
    }
    if (g_hasCUDA && aerial_perspective_array) {
        cudaFreeArray(aerial_perspective_array);
        aerial_perspective_array = nullptr;
    }
    if (g_hasCUDA && data.multi_scattering_lut) {
        cudaDestroyTextureObject(data.multi_scattering_lut);
        data.multi_scattering_lut = 0;
    }
    if (g_hasCUDA && data.aerial_perspective_lut) {
        cudaDestroyTextureObject(data.aerial_perspective_lut);
        data.aerial_perspective_lut = 0;
    }
    initialized = false;
}

void AtmosphereLUT::initialize() {
    cleanup();

    if (!g_hasCUDA) {
        initialized = true; // Still "initialized" for CPU use
        return;
    }

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

void AtmosphereLUT::uploadCudaTransmittance() {
    cudaMemcpy2DToArray(transmittance_array, 0, 0, host_transmittance.data(), TRANSMITTANCE_LUT_W * sizeof(float4), TRANSMITTANCE_LUT_W * sizeof(float4), TRANSMITTANCE_LUT_H, cudaMemcpyHostToDevice);
}

void AtmosphereLUT::uploadCudaMultiScatter() {
    cudaMemcpy2DToArray(multi_scatter_array, 0, 0, host_multi_scatter.data(), MULTI_SCATTER_LUT_RES * sizeof(float4), MULTI_SCATTER_LUT_RES * sizeof(float4), MULTI_SCATTER_LUT_RES, cudaMemcpyHostToDevice);
}

void AtmosphereLUT::uploadCudaSkyView() {
    cudaMemcpy2DToArray(skyview_array, 0, 0, host_skyview.data(), SKYVIEW_LUT_W * sizeof(float4), SKYVIEW_LUT_W * sizeof(float4), SKYVIEW_LUT_H, cudaMemcpyHostToDevice);
}
