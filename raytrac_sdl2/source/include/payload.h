/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          payload.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
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
    int opacity_has_alpha;
    // Removed pad0 as we added an int, keeping it 16-byte aligned? 
    // Wait, let's keep pad0 if needed but check alignment.
    // has_albedo_tex (4) + ... + has_emission_tex (4) = 7 * 4 = 28 bytes.
    // + opacity_has_alpha (4) = 32 bytes. No padding needed for 16-byte alignment.
    // But let's check the previous state. 
    // has_albedo_tex(4), has_roughness_tex(4), has_normal_tex(4), has_metallic_tex(4), 
    // has_transmission_tex(4), has_opacity_tex(4), has_emission_tex(4) = 28 bytes.
    // + pad0 (4) = 32 bytes.
    // If I add opacity_has_alpha, it's 32 bytes. Perfect.

    
    // Volumetric material info
    int is_volumetric;        // Volumetric stuff
    float vol_density;
    float vol_absorption;
    float vol_scattering;
    float3 vol_albedo;
    float3 vol_emission;
    float vol_g;
    float vol_step_size;
    
    // Multi-scattering
    float vol_multi_scatter;
    float vol_g_back;
    float vol_lobe_mix;
    int vol_light_steps;
    float vol_shadow_strength;
    float vol_noise_scale; // Added noise scale
    
    int vol_max_steps;
    float3 aabb_min;
    float3 aabb_max;
    
    // NanoVDB Grid (for VDB-based volumetrics)
    void* nanovdb_grid;           // Device pointer to NanoVDB grid
    int has_nanovdb;              // 1 = use NanoVDB, 0 = procedural
    
    // ═══════════════════════════════════════════════════════════════════════════
    // GPU PICKING (Object identification for viewport selection)
    // ═══════════════════════════════════════════════════════════════════════════
    int object_id;                // Object ID from HitGroupData (-1 = no object ID)
    
    // ═══════════════════════════════════════════════════════════════════════════
    // HAIR RENDERING
    // ═══════════════════════════════════════════════════════════════════════════
    int is_hair;                  // 1 = hair hit, 0 = regular geometry
    float3 color;                 // Pre-computed color (for hair direct lighting)
    float3 albedo;                // Surface albedo
    float roughness;              // Surface roughness
    float metallic;               // Surface metallic
    
    // Blended Material Data (For Terrain Layers)
    int use_blended_data;      // 1 = use baked values below instead of sampling textures again
    float3 blended_albedo;
    float blended_roughness;
    float3 blended_normal;     // Tangent space or World space? Let's assume standard logic handles Normal.
                               // Actually, normal is already computed and stored in 'normal' field above.
                               // So we don't need blended_normal.
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

