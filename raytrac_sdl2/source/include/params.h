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
    
    // Motion Blur Parameters
    float shutter_open_time; // In seconds (derived from shutter speed)
    float3 vel_origin;       // Change in origin per shutter interval
    float3 vel_corner;       // Change in lower_left_corner
    float3 vel_horizontal;   // Change in horizontal vector
    float3 vel_vertical;     // Change in vertical vector
    int motion_blur_enabled; // 0=Off, 1=On
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
    //float4* accumulation_buffer launch_tile_based_progressive için gerektiğinde;
   
};
#define SHADOW_RAY_TYPE 1
#define RAY_TYPE_COUNT 2
