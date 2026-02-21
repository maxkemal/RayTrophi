/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          sbt_data.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include "material_gpu.h" // GpuMaterial zaten ayrı bir yapıda tanımlı
#include <cuda_runtime.h>

struct __align__(16) HitGroupData
{
    // Per-triangle interpolated attributes
    float3 n0 = {0,0,0}, n1 = {0,0,0}, n2 = {0,0,0};     // Vertex normals
    float2 uv0 = {0,0}, uv1 = {0,0}, uv2 = {0,0};        // UVs
    float3 t0 = {0,0,0}, t1 = {0,0,0}, t2 = {0,0,0};     // Tangents

    // Material info (4-byte align)
    int material_id = -1;
    int mesh_material_id = -1;    // [NEW] Material of the scalp mesh
    int is_hair = 0;              // 1 = hair curve, 0 = mesh/terrain

    int has_albedo_tex = 0;
    int has_roughness_tex = 0;
    int has_normal_tex = 0;
    int has_metallic_tex = 0;
    int has_transmission_tex = 0;
    int has_opacity_tex = 0;
    int has_emission_tex = 0;
    int opacity_has_alpha = 0;
    int pad0 = 0; // fill-up to keep next part aligned

    float3 emission = {0,0,0};
    int pad1 = 0; // float3 -> next 16-byte boundaryc

    // Geometry buffers (only device pointers!)
    const float3* vertices = nullptr;
    const uint3* indices = nullptr;
    const float3* normals = nullptr;
    const float3* tangents = nullptr;
    const float2* uvs = nullptr;
    const uint32_t* strand_ids = nullptr;
    const float2* root_uvs = nullptr;     // Per-segment root UV (hair only)
    const float* strand_v = nullptr;      // Per-segment strand position (0=root, 1=tip)

    // Existence flags (bool kırıldı, int ile sabitiz)
    int has_normals = 0;
    int has_uvs = 0;
    int has_tangents = 0;
    int has_root_uvs = 0; // 1 = root_uvs pointer is valid
    int has_strand_v = 0; // 1 = strand_v pointer is valid

    // Texture objects (OptiX bunları seviyor)
    cudaTextureObject_t albedo_tex = 0;
    cudaTextureObject_t roughness_tex = 0;
    cudaTextureObject_t normal_tex = 0;
    cudaTextureObject_t metallic_tex = 0;
    cudaTextureObject_t transmission_tex = 0;
    cudaTextureObject_t opacity_tex = 0;
    cudaTextureObject_t emission_tex = 0;
    
    // Hair Material Data
    GpuHairMaterial hair_material;
    
    // Volumetric material support
    int is_volumetric = 0;        // 1 = volumetric material, 0 = surface
    float vol_density = 0.0f;     // Base density
    float vol_absorption = 0.0f;  // Absorption coefficient
    float vol_scattering = 0.0f;  // Scattering coefficient
    float3 vol_albedo = {1,1,1};  // Scattering color
    float3 vol_emission = {0,0,0}; // Volume emission
    float vol_g = 0.0f;           // Phase anisotropy (forward)
    float vol_step_size = 0.1f;   // Ray march step size
    int vol_max_steps = 100;      // Max ray march steps
    float vol_noise_scale = 1.0f; // Noise frequency multiplier
    
    // Terrain Layer System
    int is_terrain = 0;           // 1 = use layer blending
    int pad3 = 0;                 // Alignment
    
    cudaTextureObject_t splat_map_tex = 0;
    
    // Arrays for 4 layers (Albedo, Normal, Roughness)
    // 4 * 8 bytes = 32 bytes (16-byte aligned)
    // Arrays for 4 layers (Material IDs)
    int layer_material_ids[4] = {-1, -1, -1, -1}; // 16 bytes

    
    // Tiling scales for each layer
    float layer_uv_scale[4] = {1.0f, 1.0f, 1.0f, 1.0f}; // 16 bytes
    
    // Multi-Scattering parameters (NEW)
    
    // Multi-Scattering Parameters (NEW)
    float vol_multi_scatter = 0.3f;   // Multi-scatter contribution (0-1)
    float vol_g_back = -0.3f;         // Backward scattering anisotropy
    float vol_lobe_mix = 0.7f;        // Forward/backward lobe mix
    int vol_light_steps = 4;          // Light march steps (0=disabled)
    float vol_shadow_strength = 0.8f; // Self-shadow intensity
    
    // Object AABB bounds (for volumetric ray marching)
    float3 aabb_min = {0,0,0};    // Bounding box minimum
    float3 aabb_max = {0,0,0};    // Bounding box maximum
    
    // NanoVDB GPU Grid pointer (for VDB-based volumetrics)
    // NanoVDB GPU Grid pointer (for VDB-based volumetrics)
    void* nanovdb_grid = nullptr; // Device pointer to NanoVDB grid
    int has_nanovdb = 0;          // 1 = use NanoVDB grid
    int has_vol_texture = 0;      // 1 = use 3D texture (Dense Float Grid)
    
    // Dense 3D Texture (GasVolume)
    // Replaces NanoVDB when has_vol_texture is 1
    cudaTextureObject_t vol_density_texture = 0; 
    
    // ═══════════════════════════════════════════════════════════════════════════
    // FOLIAGE WIND (Shader-based vertex displacement)
    // ═══════════════════════════════════════════════════════════════════════════
    int is_foliage = 0;           // 1 = Apply shader wind displacement, 0 = Static
    float foliage_height = 1.0f;  // Mesh bounding box height (for Y-based bending)
    float3 foliage_pivot = {0,0,0}; // Local pivot point (usually base of tree)
    int foliage_pad = 0;          // Alignment padding
    
    // ═══════════════════════════════════════════════════════════════════════════
    // GPU PICKING (Object identification for viewport selection)
    // ═══════════════════════════════════════════════════════════════════════════
    int object_id = -1;           // Unique object ID (SBT record index or mesh index)
    int object_id_pad = 0;        // Alignment padding
};

