#pragma once
#include <vector_types.h>
#include <material_gpu.h>
#include "World.h"

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

// ═══════════════════════════════════════════════════════════════════════════════
// GPU VDB VOLUME (Industry-Standard VDB Rendering)
// Independent volume objects with transform and multi-grid support
// ═══════════════════════════════════════════════════════════════════════════════
struct GpuVDBVolume {
    // ─────────────────────────────────────────────────────────────────────────
    // GRID POINTERS (NanoVDB device pointers)
    // ─────────────────────────────────────────────────────────────────────────
    void* density_grid;       // FloatGrid* - Required
    void* temperature_grid;   // FloatGrid* - Optional (fire)
    void* velocity_grid;      // Vec3fGrid* - Optional (motion blur)
    void* emission_grid;      // FloatGrid* - Optional (custom emission)
    
    // ─────────────────────────────────────────────────────────────────────────
    // TRANSFORM (Object → World, row-major 3x4)
    // ─────────────────────────────────────────────────────────────────────────
    float transform[12];      // Object to world
    float inv_transform[12];  // World to object (for ray transform)
    
    // ─────────────────────────────────────────────────────────────────────────
    // BOUNDS
    // ─────────────────────────────────────────────────────────────────────────
    float3 world_bbox_min;    // World-space bounds (after transform)
    float3 world_bbox_max;
    float3 local_bbox_min;    // Native VDB bounds (before transform)
    float3 local_bbox_max;
    
    // ─────────────────────────────────────────────────────────────────────────
    // DENSITY SHADER
    // ─────────────────────────────────────────────────────────────────────────
    float density_multiplier;
    float density_remap_low;
    float density_remap_high;
    float density_pad;
    
    // ─────────────────────────────────────────────────────────────────────────
    // SCATTERING SHADER
    // ─────────────────────────────────────────────────────────────────────────
    float3 scatter_color;
    float scatter_coefficient;
    float scatter_anisotropy;       // G forward (-1 to 1)
    float scatter_anisotropy_back;  // G backward
    float scatter_lobe_mix;         // Forward/back blend
    float scatter_multi;            // Multi-scatter approximation
    
    // ─────────────────────────────────────────────────────────────────────────
    // ABSORPTION SHADER
    // ─────────────────────────────────────────────────────────────────────────
    float3 absorption_color;
    float absorption_coefficient;
    
    // ─────────────────────────────────────────────────────────────────────────
    // EMISSION SHADER (Fire/Explosions)
    // ─────────────────────────────────────────────────────────────────────────
    int emission_mode;  // 0=None, 1=Constant, 2=Blackbody, 3=Channel
    float3 emission_color;
    float emission_intensity;
    float temperature_scale;
    float blackbody_intensity;
    float emission_pad;

    // ─────────────────────────────────────────────────────────────────────────
    // COLOR RAMP (Gradient)
    // ─────────────────────────────────────────────────────────────────────────
    int color_ramp_enabled;       // 0=Off, 1=On
    int ramp_stop_count;          // Number of active stops (max 8)
    float ramp_positions[8];      // Stop positions (0-1)
    float3 ramp_colors[8];        // Stop colors
    float ramp_pad;               // Padding for alignment
    
    // ─────────────────────────────────────────────────────────────────────────
    // RAY MARCHING QUALITY
    // ─────────────────────────────────────────────────────────────────────────
    float step_size;
    int max_steps;
    int shadow_steps;
    float shadow_strength;
    
    // ─────────────────────────────────────────────────────────────────────────
    // MOTION BLUR
    // ─────────────────────────────────────────────────────────────────────────
    int motion_blur_enabled;
    float velocity_scale;
    int pad1;
    int pad2;
};

struct RayGenParams {
    uchar4* framebuffer;
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
    GpuVolumetricInfo* volumetric_infos;
    
    // VDB Volume Objects (independent of mesh geometry)
    GpuVDBVolume* vdb_volumes;
    int vdb_volume_count;
    
    WorldData world;
    // Yeni parametreler
    int min_samples;             // Minimum örnek sayısı
	int max_samples;             // Maksimum örnek sayısı
    float variance_threshold;    // Baz varyans eşik değeri
    int frame_number;            // Mevcut frame numarası
	int max_depth; 		        // Maksimum derinlik
	bool use_adaptive_sampling;   // Adaptif örnekleme kullanılıp kullanılmayacağı
    // Uzamsal tutarlılık ve temporal akümülasyon için
    float* variance_buffer;      // Piksellerin varyans değerlerini saklamak için
    float* accumulation_buffer;  // Temporal akümülasyon için önceki frame verisi
    int* sample_count_buffer;    // Her piksel için kullanılan örnek sayısını saklamak için
    float temporal_blend;        // Temporal akümülasyon karışım faktörü (0 = sadece yeni, 1 = sadece eski)
    int tile_x, tile_y;
    int tile_width, tile_height;
    int current_pass;
    int is_final_render; // 1 = Final render, 0 = Viewport
    int grid_enabled;    // 1 = Grid visible, 0 = Hidden
    float grid_fade_distance; // Distance where grid fades out
    float clip_near;
    float clip_far;
    float time;            // Global time for animation (updates every frame)
    float water_time;       // Water time - frozen during accumulation passes
    //float4* accumulation_buffer launch_tile_based_progressive için gerektiğinde;
   
};
#define SHADOW_RAY_TYPE 1
#define RAY_TYPE_COUNT 2
