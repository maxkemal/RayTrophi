#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>
#include "material_gpu.h"
#include "params.h"
#include "payload.h"
#include "sky_model.cuh"
#include "vec3_utils.cuh"
#include "ray_color.cuh"
#include "water_shaders.cuh"
#include "ray.h"
#include "hair_bsdf.cuh"
#include "gpucamera.cuh"
#include "random_utils.cuh"
#include "trace_ray.cuh"

extern "C" __constant__ RayGenParams optixLaunchParams;

__device__ __forceinline__ float compute_luminance(float3 color) {
    return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}

extern "C" __global__ void __raygen__rg() {
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint3 launch_dims = optixGetLaunchDimensions();

    const int i = launch_idx.x; 
    const int j = launch_idx.y;

    if (i >= optixLaunchParams.image_width || j >= optixLaunchParams.image_height) return;

    const int pixel_index = j * optixLaunchParams.image_width + i;

    float4* accum_buffer = reinterpret_cast<float4*>(optixLaunchParams.accumulation_buffer);
    float* variance_buffer = optixLaunchParams.variance_buffer;
    
    if (optixLaunchParams.use_adaptive_sampling && accum_buffer != nullptr && variance_buffer != nullptr) {
        float4 prev = accum_buffer[pixel_index];
        float prev_samples = prev.w;
        float current_variance = variance_buffer[pixel_index];
        float mean_lum = compute_luminance(make_float3(prev.x, prev.y, prev.z));
        float cv = (mean_lum > 0.00001f) ? sqrtf(current_variance) / mean_lum : 1.0f;
        float effective_threshold = optixLaunchParams.variance_threshold;
        if (optixLaunchParams.use_denoiser) {
            effective_threshold *= 2.0f;
        }
        if (prev_samples >= optixLaunchParams.min_samples && 
            current_variance > 0.0f && 
            cv < effective_threshold) {
            float3 prev_color = make_float3(prev.x, prev.y, prev.z);
            // Apply exposure to converged pixels too
            optixLaunchParams.framebuffer[pixel_index] = make_color(prev_color * optixLaunchParams.camera.exposure_factor);
            if (optixLaunchParams.converged_count != nullptr) {
                atomicAdd(optixLaunchParams.converged_count, 1);
            }
            return;
        }
    }

    unsigned int seed = optixLaunchParams.frame_number * 719393 + pixel_index * 13731 + 1337;
    PCGState pcg_rng;
    pcg_init(&pcg_rng, seed);
    curandState rng;
    curand_init(seed, 0, 0, &rng);

    const int samples_this_pass = optixLaunchParams.samples_per_pixel;
    float3 color_sum = make_float3(0.0f, 0.0f, 0.0f);

    for (int s = 0; s < samples_this_pass; ++s) {
        float u = (i + pcg_float(&pcg_rng)) / float(optixLaunchParams.image_width);
        float v = (j + pcg_float(&pcg_rng)) / float(optixLaunchParams.image_height);

        if (optixLaunchParams.camera.chromatic_aberration_enabled && 
            optixLaunchParams.camera.chromatic_aberration > 0.0001f) {
            float ca_amount = optixLaunchParams.camera.chromatic_aberration;
            float r_scale = optixLaunchParams.camera.chromatic_aberration_r;
            float b_scale = optixLaunchParams.camera.chromatic_aberration_b;
            float2 uv_r = apply_chromatic_aberration(u, v, ca_amount, r_scale);
            Ray ray_r = get_ray_from_camera(optixLaunchParams.camera, uv_r.x, uv_r.y, &rng);
            float3 color_r = ray_color(ray_r, &rng);
            Ray ray_g = get_ray_from_camera(optixLaunchParams.camera, u, v, &rng);
            float3 color_g = ray_color(ray_g, &rng);
            float2 uv_b = apply_chromatic_aberration(u, v, ca_amount, b_scale);
            Ray ray_b = get_ray_from_camera(optixLaunchParams.camera, uv_b.x, uv_b.y, &rng);
            float3 color_b = ray_color(ray_b, &rng);
            color_sum += make_float3(color_r.x, color_g.y, color_b.z);
        } else {
            Ray ray = get_ray_from_camera(optixLaunchParams.camera, u, v, &rng);
            float3 sample = ray_color(ray, &rng);
            color_sum += sample;
        }
    }

    float3 new_color = color_sum / float(samples_this_pass);
    float3 blended_color = new_color;
    float new_total_samples = float(samples_this_pass);
    
    if (accum_buffer != nullptr) {
        float4 prev = accum_buffer[pixel_index];
        float prev_samples = prev.w;
        if (prev_samples > 0.0f) {
            new_total_samples = prev_samples + samples_this_pass;
            float3 prev_color = make_float3(prev.x, prev.y, prev.z);
            blended_color = (prev_color * prev_samples + new_color * samples_this_pass) / new_total_samples;
            accum_buffer[pixel_index] = make_float4(blended_color.x, blended_color.y, blended_color.z, new_total_samples);
        } else {
            accum_buffer[pixel_index] = make_float4(new_color.x, new_color.y, new_color.z, float(samples_this_pass));
        }
    }
    
    if (optixLaunchParams.use_adaptive_sampling && variance_buffer != nullptr) {
        float new_lum = compute_luminance(new_color);
        float mean_lum = compute_luminance(blended_color);
        float diff = new_lum - mean_lum;
        float prev_variance = variance_buffer[pixel_index];
        float alpha = 1.0f / fmaxf(new_total_samples, 2.0f);
        float updated_variance = prev_variance * (1.0f - alpha) + (diff * diff) * alpha;
        variance_buffer[pixel_index] = fminf(fmaxf(updated_variance, 0.0f), 100.0f);
    }
    
    // Apply exposure only for display
    float3 exposed_color = blended_color * optixLaunchParams.camera.exposure_factor;

    if (optixLaunchParams.camera.vignetting_enabled) {
        float norm_x = (float(i) / float(optixLaunchParams.image_width)) * 2.0f - 1.0f;
        float norm_y = (float(j) / float(optixLaunchParams.image_height)) * 2.0f - 1.0f;
        float dist_sq = norm_x * norm_x + norm_y * norm_y;
        float vignette_radius = 1.414f;
        float normalized_dist = sqrtf(dist_sq) / vignette_radius;
        float falloff = optixLaunchParams.camera.vignetting_falloff;
        float vignette_factor = 1.0f - optixLaunchParams.camera.vignetting_amount * powf(normalized_dist, falloff);
        vignette_factor = fmaxf(vignette_factor, 0.0f);
        exposed_color *= vignette_factor;
    }

    optixLaunchParams.framebuffer[pixel_index] = make_color(exposed_color);
}
