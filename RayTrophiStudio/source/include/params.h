/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          params.h
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
// RayGenParams owns OptixTraversableHandle fields; do not rely on include order.
#include <optix.h>
#include <vector_types.h>
#include <material_gpu.h>
#include "World.h"
#include "GpuVolumeTypes.h"

#ifndef START_PARAMS_H_OVERRIDES
#define START_PARAMS_H_OVERRIDES
#ifndef __CUDACC__
// If not CUDA compiler, define macros empty if missing (though likely defined by cuda_runtime)
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

inline __host__ __device__ float3 make_float3(float s) {
    return make_float3(s, s, s);
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
#endif
struct gpuCamera {
    float3 origin;
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
    float3 u, v, w;

    float lens_radius;
    float focus_dist;
    int blade_count;
	float aperture;  // lens aperture
    float exposure_factor; // Calculated exposure multiplier
    float distortion;      // Lens Distortion (-0.5 to 0.5)
    
    // Motion Blur Parameters
    float shutter_open_time; // In seconds (derived from shutter speed)
    float3 vel_origin;       // Change in origin per shutter interval
    float3 vel_corner;       // Change in lower_left_corner
    float3 vel_horizontal;   // Change in horizontal vector
    float3 vel_vertical;     // Change in vertical vector
    int motion_blur_enabled; // 0=Off, 1=On
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CINEMA MODE - Physical Lens Imperfections
    // ═══════════════════════════════════════════════════════════════════════════
    int camera_mode;                    // 0=Auto, 1=Pro, 2=Cinema
    
    // Chromatic Aberration (Renk Sapması)
    int chromatic_aberration_enabled;   // 0=Off, 1=On
    float chromatic_aberration;         // Amount (0-0.05)
    float chromatic_aberration_r;       // Red channel scale
    float chromatic_aberration_b;       // Blue channel scale
    
    // Vignetting (Köşe Kararması)
    int vignetting_enabled;             // 0=Off, 1=On
    float vignetting_amount;            // Strength (0-1)
    float vignetting_falloff;           // Falloff curve (1.5-4.0)
    
    // Camera Shake Offset (calculated on CPU, applied to rays)
    float3 shake_offset;                // Position shake (meters)
    float3 shake_rotation;              // Rotation shake (radians)
    int shake_enabled;                  // 0=Off, 1=On
};

struct LightGPU {
    float3 position;
    float3 direction;       // directional, spot, area için
    float3 color;           // normalize edilmiş renk [0-1]
    float intensity;        // toplam güç (lümen)
    float radius;           // yumuşak gölge için
    int type;               // 0 = point, 1 = directional, 2 = area, 3 = spot
    
    // SpotLight ek parametreleri
    float inner_cone_cos;   // iç cone açısının kosinüsü
    float outer_cone_cos;   // dış cone açısının kosinüsü
    
    // AreaLight ek parametreleri
    float area_width;       // AreaLight genişliği
    float area_height;      // AreaLight yüksekliği
    float3 area_u;          // AreaLight U vektörü
    float3 area_v;          // AreaLight V vektörü
};

struct GpuVolumetricInfo {
    int is_volumetric;
    float density;
    float absorption;
    float scattering;
    float3 albedo;
    float3 emission;
    float g;
    float step_size;
    int max_steps;
    float noise_scale;
    
    // Multi-Scattering
    float multi_scatter;
    float g_back;
    float lobe_mix;
    int light_steps;
    float shadow_strength;
    
    float3 aabb_min;
    float3 aabb_max;
    
    // NanoVDB Grid Pointer (void* cast to nanovdb::GridHandle* or accessor on device)
    void* nanovdb_grid;
    int has_nanovdb;
};

struct RayGenParams {
    uchar4* framebuffer;
    float4* denoiser_albedo;
    float4* denoiser_normal;
    float4* stylize_position;   // Stylize AOV: .xyz world pos, .w encoded material id (0=miss,1=unknown,>=2 matid+2)
    int image_width;
    int image_height;
    int launch_offset_x;
    int launch_offset_y;
    int samples_per_pixel;
    gpuCamera camera;
    int* launch_coords_x;
    int* launch_coords_y;
    int batch_pixel_count;
    float3 light_position;
    float3 light_intensity;
    LightGPU* lights;
    int light_count;
    float3 background_color;
    OptixTraversableHandle handle;
    GpuMaterial* materials;
    int material_count;
    GpuVolumetricInfo* volumetric_infos;
    
    // VDB Volume Objects (independent of mesh geometry)
    GpuVDBVolume* vdb_volumes;
    int vdb_volume_count;
    
    // Gas Volume Objects (Dense 3D Textures)
    GpuGasVolume* gas_volumes;
    int gas_volume_count;
    
    WorldData world;
    // Yeni parametreler
    int min_samples;             // Minimum örnek sayısı
	int max_samples;             // Maksimum örnek sayısı
    float variance_threshold;    // Baz varyans eşik değeri
    int frame_number;            // Mevcut frame numarası
	int max_depth; 		        // Maksimum derinlik
    int diffuse_depth;
    int transmission_depth;
	bool use_adaptive_sampling;   // Adaptif örnekleme kullanılıp kullanılmayacağı
    bool use_denoiser;            // OIDN denoiser aktif mi (adaptive sampling threshold'u etkiler)
    // Uzamsal tutarlılık ve temporal akümülasyon için
    float* variance_buffer;      // Piksellerin varyans değerlerini saklamak için
    float* accumulation_buffer;  // Temporal akümülasyon için önceki frame verisi
    int* sample_count_buffer;
    // DIAGNOSTIC: number of pixels where the kernel READ a sample count of 0
    // from the accumulation buffer. When the host counter and the buffer's .w
    // disagree, only the KERNEL can say which one is lying -- every host-side
    // read goes through the same suspect memory it is trying to measure.
    // Costs nothing while null.
    unsigned int* accum_prev_zero_count;    // Her piksel için kullanılan örnek sayısını saklamak için
    float temporal_blend;        // Temporal akümülasyon karışım faktörü (0 = sadece yeni, 1 = sadece eski)
    int tile_x, tile_y;
    int tile_width, tile_height;
    int current_pass;
    int is_final_render; // 1 = Final render, 0 = Viewport
    // 1 = DISPLAY RESOLVE ONLY: do not trace, do not touch the accumulation or
    // variance buffers, do not advance the sample count. Just re-encode the
    // existing accumulation into `framebuffer` with the CURRENT display values.
    // Exists so a post-processing edit made AFTER convergence can still reach
    // the presented frame — the OptiX twin of the Vulkan tonemap-only refresh.
    // ★ NO default member initializer here: RayGenParams is declared as a CUDA
    // __constant__ variable (payload.h), and any dynamic initialization in the
    // type makes nvcc reject it ("dynamic initialization is not supported for a
    // __constant__ variable"). Host code sets this on every launch instead.
    int display_resolve_only;
    int grid_enabled;    // 1 = Grid visible, 0 = Hidden
    float grid_fade_distance; // Distance where grid fades out
    float clip_near;
    float clip_far;
    float time;            // Global time for animation (updates every frame)
    float water_time;       // Water time - frozen during accumulation passes
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ADAPTIVE SAMPLING DEBUG
    // ═══════════════════════════════════════════════════════════════════════════
    int* converged_count;     // Atomic counter for converged pixels (debug)
    
    // ═══════════════════════════════════════════════════════════════════════════
    // WIND ANIMATION (Shader-based foliage bending)
    // ═══════════════════════════════════════════════════════════════════════════
    float3 wind_direction;    // Normalized wind direction (world space)
    float wind_strength;      // Wind strength (0-1 normalized)
    float wind_speed;         // Animation speed multiplier
    float wind_time;          // Wind animation time (separate from water_time)
    
    // ═══════════════════════════════════════════════════════════════════════════
    // GPU PICKING (Object ID buffer for viewport selection)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // ═══════════════════════════════════════════════════════════════════════════
    // HAIR RENDERING (OptiX Curve Primitives)
    // ═══════════════════════════════════════════════════════════════════════════
    OptixTraversableHandle hair_handle;   // Separate GAS for hair curves
    int hair_enabled;                     // 0 = no hair, 1 = hair rendering active
    
    // Hair geometry data (for closesthit program)
    float4* hair_vertices;                // x,y,z,radius per control point
    unsigned int* hair_indices;           // Segment start indices
    float3* hair_tangents;                // Pre-computed tangent per segment
    int hair_segment_count;
    int hair_vertex_count;
    
    // Hair material parameters (Marschner BSDF)
    float3 hair_color;                    // Base hair color
    float hair_melanin;                   // 0=blonde, 0.5=brown, 1=black
    float hair_melanin_redness;           // Red pigment ratio
    float hair_roughness;                 // Longitudinal roughness
    float hair_radial_roughness;          // Azimuthal roughness
    float hair_ior;                       // Index of refraction (1.55 human)
    float3 hair_absorption;               // Explicit absorption coefficient (for Mode 2)
    float hair_alpha;                     // Cuticle tilt in radians
    float hair_coat;                      // Outer coat layer (fur)
    float3 hair_coat_tint;                // Coat color
    float hair_random_hue;
    float hair_random_value;
    
    // Hair coloring mode (0=Direct, 1=Melanin, 2=Absorption, 3=Root UV Map)
    int hair_color_mode;
    
    // Custom hair textures (assigned via Hair Material Panel)
    cudaTextureObject_t hair_albedo_tex;        // Custom albedo texture for hair
    cudaTextureObject_t hair_roughness_tex;      // Custom roughness texture for hair
    int hair_has_albedo_tex;                     // 1 = custom albedo texture valid
    int hair_has_roughness_tex;                  // 1 = custom roughness texture valid
    
    // Scalp mesh texture (inherited from bound mesh material)
    cudaTextureObject_t hair_scalp_albedo_tex;   // Scalp mesh albedo texture
    int hair_has_scalp_albedo_tex;               // 1 = scalp texture valid
    float3 hair_scalp_base_color;                // Scalp material base color (fallback)
   
};

// ═══════════════════════════════════════════════════════════════════════════════
// HAIR CURVE HIT DATA (for closesthit program)
// ═══════════════════════════════════════════════════════════════════════════════
struct HairHitGroupData {
    float4* vertices;         // x,y,z,radius per control point
    unsigned int* indices;    // Segment start indices
    float3* tangents;         // Pre-computed tangent per segment
    uint32_t* strand_ids;     // Per-segment strand identifier
    int segment_count;
    int vertex_count;
};

#define SHADOW_RAY_TYPE 1
#define RAY_TYPE_COUNT 2
