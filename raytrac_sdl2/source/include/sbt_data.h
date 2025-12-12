#pragma once
#include "material_gpu.h" // GpuMaterial zaten ayrı bir yapıda tanımlı
#include <cuda_runtime.h>

struct __align__(16) HitGroupData
{
    // Per-triangle interpolated attributes
    float3 n0, n1, n2;     // Vertex normals
    float2 uv0, uv1, uv2;  // UVs
    float3 t0, t1, t2;     // Tangents

    // Material info (4-byte align)
    int material_id;

    int has_albedo_tex;
    int has_roughness_tex;
    int has_normal_tex;
    int has_metallic_tex;
    int has_transmission_tex;
    int has_opacity_tex;
    int has_emission_tex;
    int pad0; // fill-up to keep next part aligned

    float3 emission;
    int pad1; // float3 -> next 16-byte boundary

    // Geometry buffers (only device pointers!)
    const float3* vertices;
    const uint3* indices;
    const float3* normals;
    const float3* tangents;
    const float2* uvs;

    // Existence flags (bool kırıldı, int ile sabitiz)
    int has_normals;
    int has_uvs;
    int has_tangents;

    // Texture objects (OptiX bunları seviyor)
    cudaTextureObject_t albedo_tex;
    cudaTextureObject_t roughness_tex;
    cudaTextureObject_t normal_tex;
    cudaTextureObject_t metallic_tex;
    cudaTextureObject_t transmission_tex;
    cudaTextureObject_t opacity_tex;
    cudaTextureObject_t emission_tex;
};
