#pragma once
#include "material_gpu.h" // GpuMaterial zaten ayrı bir yapıda tanımlı
#include <cuda_runtime.h>

struct __align__(16) HitGroupData {
    float3 n0, n1, n2;     // Vertex normal
    float2 uv0, uv1, uv2;  // UV koordinatları
    float3 t0, t1, t2;     // Tangent vektörleri

    int material_id;       // GpuMaterial id'si
    int has_albedo_tex;
    int has_roughness_tex;
    int has_normal_tex;
    int has_metallic_tex;
    int has_transmission_tex;
    int has_opacity_tex;
    int has_emission_tex;
    int pad0;              // align için
    float3 emission;

    float3* vertices;
    uint3* indices;
    float3* normals;
    float3* tangents;
    float2* uvs;

    bool has_normals;
    bool has_uvs;
    bool has_tangents;

    // Texture data
    cudaTextureObject_t albedo_tex;
    cudaTextureObject_t roughness_tex;
    cudaTextureObject_t normal_tex;
    cudaTextureObject_t metallic_tex;
    cudaTextureObject_t transmission_tex;
    cudaTextureObject_t opacity_tex;
    cudaTextureObject_t emission_tex;
};

