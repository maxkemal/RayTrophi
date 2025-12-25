// ===== payload.h =====
#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include "vec3_utils.cuh"
#include <math_constants.h>
#include "ray.h"
#include "sbt_data.h"


struct OptixHitResult {
    float3 position;
    float3 normal;
    float3 emission;
    GpuMaterial material;
    float2 uv;
    float3 sampled_albedo;
    int material_id;
    float t;
    int hit;
    // Texture object'leri
    cudaTextureObject_t albedo_tex;
    cudaTextureObject_t roughness_tex;
    cudaTextureObject_t normal_tex;
    cudaTextureObject_t metallic_tex;
    cudaTextureObject_t transmission_tex;
    cudaTextureObject_t opacity_tex;
    cudaTextureObject_t emission_tex;
    // Texture var mı bilgisi
    int has_albedo_tex;
    int has_roughness_tex;
    int has_normal_tex;
    int has_metallic_tex;
    int has_transmission_tex;
    int has_opacity_tex;
    int has_emission_tex;
    int pad0; // 16 byte hizalama için
    
    // Volumetric material info
    int is_volumetric;        // 1 = volumetric, 0 = surface
    float vol_density;
    float vol_absorption;
    float vol_scattering;
    float3 vol_albedo;
    float3 vol_emission;
    float vol_g;
    float vol_step_size;
    int vol_max_steps;
    float vol_noise_scale;
    float3 aabb_min;
    float3 aabb_max;
    
    // Multi-Scattering parameters (NEW)
    float vol_multi_scatter;
    float vol_g_back;
    float vol_lobe_mix;
    int vol_light_steps;
    float vol_shadow_strength;

};


// Pointer packing helpers
template <typename T>
__device__ __forceinline__ void packPayload(const T* ptr, unsigned int& p0, unsigned int& p1) {
    const unsigned long long ptr_val = reinterpret_cast<unsigned long long>(ptr);
    p0 = static_cast<unsigned int>(ptr_val);
    p1 = static_cast<unsigned int>(ptr_val >> 32);
}

template <typename T>
__device__ __forceinline__ T* unpackPayload(unsigned int p0, unsigned int p1) {
    const unsigned long long ptr_val = (static_cast<unsigned long long>(p1) << 32) | p0;
    return reinterpret_cast<T*>(ptr_val);
}

extern "C" {
    __constant__ RayGenParams optixLaunchParams;
}
