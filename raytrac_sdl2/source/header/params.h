#pragma once
#include <vector_types.h>
#include <material_gpu.h>
#include "AtmosphereProperties.h"
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
};
struct LightGPU {
    float3 position;
    float3 direction;       // directional, spot, area için
    float3 color;           // normalize edilmiş renk [0-1]
    float intensity;        // toplam güç (lümen)
    float radius;           // yumuşak gölge için
    int type;               // 0 = point, 1 = directional, 2 = area, 3 = spot
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
    AtmosphereProperties atmosphere;
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
    //float4* accumulation_buffer launch_tile_based_progressive için gerektiğinde;
   
};
#define SHADOW_RAY_TYPE 1
#define RAY_TYPE_COUNT 2
