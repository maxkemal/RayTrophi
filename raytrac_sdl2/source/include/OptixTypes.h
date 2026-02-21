#pragma once
#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime.h>
#include "Vec3.h"
#include "material_gpu.h"

// ═══════════════════════════════════════════════════════════════════════════
// SHARED GEOMETRY TYPES FOR OPTIX
// ═══════════════════════════════════════════════════════════════════════════

// Per-mesh geometry data for BLAS building
struct MeshGeometry {
    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<float3> normals;
    std::vector<float2> uvs;
    std::vector<float3> tangents;
    std::vector<float3> colors;
    int material_id = 0;
    std::string mesh_name;          // Unique name (e.g. "Car_mat_0")
    std::string original_name;      // Base node name (e.g. "Car") for GPU picking
    
    // Skinning Data (Optional)
    std::vector<int4> boneIndices;
    std::vector<float4> boneWeights;
};

// Per-mesh data for grouping triangles by nodeName
struct MeshData {
    std::string mesh_name;                  // Unique name (e.g. "Car_mat_0")
    std::string original_name;              // Original node name (e.g. "Car")
    std::vector<int> triangle_indices;      // Indices into global triangle list
    int material_id = 0;                    // Primary material ID for this mesh
    bool has_skinning = false;              // True if this mesh part has bone skinning
};

// Per-hair curve geometry data for BLAS building
struct CurveGeometry {
    std::vector<float4> vertices;   // x,y,z,radius
    std::vector<unsigned int> indices;
    std::vector<float3> tangents;
    std::vector<uint32_t> strand_ids;
    std::vector<float2> root_uvs;   // Per-segment root UV (from scalp mesh)
    std::vector<float> strand_v;    // Per-segment position along strand (0=root, 1=tip)
    size_t vertex_count = 0;
    size_t segment_count = 0;
    bool use_bspline = false;
    int material_id = 0;
    int mesh_material_id = -1;
    GpuHairMaterial hair_material;
    std::string name;
};

// Unified geometry data passed to build methods
struct OptixGeometryData {
    // Texture bundle for unified material handling
    struct TextureBundle {
        cudaTextureObject_t albedo_tex = 0;
        cudaTextureObject_t roughness_tex = 0;
        cudaTextureObject_t normal_tex = 0;
        cudaTextureObject_t metallic_tex = 0;
        cudaTextureObject_t transmission_tex = 0;
        cudaTextureObject_t opacity_tex = 0;
        cudaTextureObject_t emission_tex = 0;

        int has_albedo_tex = 0;
        int has_roughness_tex = 0;
        int has_normal_tex = 0;
        int has_metallic_tex = 0;
        int has_transmission_tex = 0;
        int has_opacity_tex = 0;
        int has_emission_tex = 0;
        int opacity_has_alpha = 0;
    };
    
    // Volumetric material info for GPU
    struct VolumetricInfo {
        int is_volumetric = 0;
        float density = 1.0f;
        float absorption = 0.1f;
        float scattering = 0.5f;
        float3 albedo = {1.0f, 1.0f, 1.0f};
        float3 emission = {0.0f, 0.0f, 0.0f};
        float g = 0.0f;
        float step_size = 0.1f;
        int max_steps = 100;
        float noise_scale = 1.0f;
        
        float multi_scatter = 0.3f;
        float g_back = -0.3f;
        float lobe_mix = 0.7f;
        int light_steps = 4;
        float shadow_strength = 0.8f;
        
        float3 aabb_min = {0.0f, 0.0f, 0.0f};
        float3 aabb_max = {1.0f, 1.0f, 1.0f};
        
        void* nanovdb_grid = nullptr;
        int has_nanovdb = 0;
        int has_vol_texture = 0;
        cudaTextureObject_t vol_density_texture = 0;
    };

    std::vector<float3> vertices;
    std::vector<uint3> indices;
    std::vector<float3> normals;
    std::vector<float3> tangents;
    std::vector<float2> uvs;
    std::vector<float3> colors;
    std::vector<GpuMaterial> materials;
    std::vector<int> material_indices;
    std::vector<CurveGeometry> curves;
    
    std::vector<int4> boneIndices;
    std::vector<float4> boneWeights;
    
    std::vector<TextureBundle> textures;
    std::vector<VolumetricInfo> volumetric_info;
};
