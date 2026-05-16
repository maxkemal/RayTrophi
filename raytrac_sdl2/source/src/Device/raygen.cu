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
        float M2 = variance_buffer[pixel_index];
        float mean_lum = compute_luminance(make_float3(prev.x, prev.y, prev.z));

        const int ADAPTIVE_WARMUP = 4;
        int effective_min = optixLaunchParams.min_samples > ADAPTIVE_WARMUP
            ? optixLaunchParams.min_samples : ADAPTIVE_WARMUP;

        float effective_threshold = optixLaunchParams.variance_threshold;
        if (optixLaunchParams.use_denoiser) {
            effective_threshold *= 2.0f;
        }

        float rel_stderr = 1.0f;
        if (prev_samples >= 2.0f && M2 > 0.0f && mean_lum > 1e-5f) {
            float variance = M2 / (prev_samples - 1.0f);
            rel_stderr = sqrtf(variance / prev_samples) / mean_lum;
        }

        if (prev_samples >= float(effective_min) &&
            M2 > 0.0f &&
            rel_stderr < effective_threshold) {
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
    float3 albedo_sum = make_float3(0.0f, 0.0f, 0.0f);
    float3 normal_sum = make_float3(0.0f, 0.0f, 0.0f);
    int primary_hits = 0;
    float batch_lum_sum = 0.0f;
    float batch_lum_sq_sum = 0.0f;

    for (int s = 0; s < samples_this_pass; ++s) {
        float3 prev_color_sum = color_sum;
        float u = (i + pcg_float(&pcg_rng)) / float(optixLaunchParams.image_width);
        float v = (j + pcg_float(&pcg_rng)) / float(optixLaunchParams.image_height);
        float3 primary_albedo = make_float3(0.0f);
        float3 primary_normal = make_float3(0.0f);
        int primary_hit = 0;

        if (optixLaunchParams.camera.chromatic_aberration_enabled && 
            optixLaunchParams.camera.chromatic_aberration > 0.0001f) {
            float ca_amount = optixLaunchParams.camera.chromatic_aberration;
            float r_scale = optixLaunchParams.camera.chromatic_aberration_r;
            float b_scale = optixLaunchParams.camera.chromatic_aberration_b;
            float2 uv_r = apply_chromatic_aberration(u, v, ca_amount, r_scale);
            Ray ray_r = get_ray_from_camera(optixLaunchParams.camera, uv_r.x, uv_r.y, &rng);
            float3 color_r = ray_color(ray_r, &rng);
            Ray ray_g = get_ray_from_camera(optixLaunchParams.camera, u, v, &rng);
            float3 color_g = ray_color(ray_g, &rng, &primary_albedo, &primary_normal, &primary_hit);
            float2 uv_b = apply_chromatic_aberration(u, v, ca_amount, b_scale);
            Ray ray_b = get_ray_from_camera(optixLaunchParams.camera, uv_b.x, uv_b.y, &rng);
            float3 color_b = ray_color(ray_b, &rng);
            color_sum += make_float3(color_r.x, color_g.y, color_b.z);
        } else {
            Ray ray = get_ray_from_camera(optixLaunchParams.camera, u, v, &rng);
            float3 sample = ray_color(ray, &rng, &primary_albedo, &primary_normal, &primary_hit);
            color_sum += sample;
        }

        if (primary_hit) {
            albedo_sum += primary_albedo;
            normal_sum += primary_normal * 0.5f + make_float3(0.5f, 0.5f, 0.5f);
            primary_hits++;
        }

        if (optixLaunchParams.use_adaptive_sampling) {
            float3 this_sample = color_sum - prev_color_sum;
            float sl = compute_luminance(this_sample);
            batch_lum_sum += sl;
            batch_lum_sq_sum += sl * sl;
        }
    }

    float3 new_color = color_sum / float(samples_this_pass);
    float3 blended_color = new_color;
    float new_total_samples = float(samples_this_pass);
    float prev_samples_for_variance = 0.0f;
    float prev_mean_lum_for_variance = 0.0f;

    if (accum_buffer != nullptr) {
        float4 prev = accum_buffer[pixel_index];
        float prev_samples = prev.w;
        prev_samples_for_variance = prev_samples;
        prev_mean_lum_for_variance = compute_luminance(make_float3(prev.x, prev.y, prev.z));
        if (prev_samples > 0.0f) {
            new_total_samples = prev_samples + samples_this_pass;
            float3 prev_color = make_float3(prev.x, prev.y, prev.z);
            blended_color = (prev_color * prev_samples + new_color * samples_this_pass) / new_total_samples;
            accum_buffer[pixel_index] = make_float4(blended_color.x, blended_color.y, blended_color.z, new_total_samples);
        } else {
            accum_buffer[pixel_index] = make_float4(new_color.x, new_color.y, new_color.z, float(samples_this_pass));
        }
    }

    if (optixLaunchParams.denoiser_albedo != nullptr && optixLaunchParams.denoiser_normal != nullptr) {
        float4 prevAlb = optixLaunchParams.denoiser_albedo[pixel_index];
        float4 prevNrm = optixLaunchParams.denoiser_normal[pixel_index];
        if (primary_hits > 0) {
            float hitWeight = 1.0f / float(primary_hits);
            float3 sample_albedo = albedo_sum * hitWeight;
            float3 sample_normal = normal_sum * hitWeight;
            float prev_samples = prevAlb.w;
            if (prev_samples > 0.0f) {
                float total_samples = prev_samples + samples_this_pass;
                sample_albedo = (make_float3(prevAlb.x, prevAlb.y, prevAlb.z) * prev_samples + sample_albedo * samples_this_pass) / total_samples;
                sample_normal = (make_float3(prevNrm.x, prevNrm.y, prevNrm.z) * prev_samples + sample_normal * samples_this_pass) / total_samples;
                optixLaunchParams.denoiser_albedo[pixel_index] = make_float4(sample_albedo.x, sample_albedo.y, sample_albedo.z, total_samples);
                optixLaunchParams.denoiser_normal[pixel_index] = make_float4(sample_normal.x, sample_normal.y, sample_normal.z, total_samples);
            } else {
                optixLaunchParams.denoiser_albedo[pixel_index] = make_float4(sample_albedo.x, sample_albedo.y, sample_albedo.z, float(samples_this_pass));
                optixLaunchParams.denoiser_normal[pixel_index] = make_float4(sample_normal.x, sample_normal.y, sample_normal.z, float(samples_this_pass));
            }
        } else if (optixLaunchParams.frame_number == 0) {
            optixLaunchParams.denoiser_albedo[pixel_index] = make_float4(0.0f, 0.0f, 0.0f, float(samples_this_pass));
            optixLaunchParams.denoiser_normal[pixel_index] = make_float4(0.5f, 0.5f, 0.5f, float(samples_this_pass));
        }
    }
    
    if (optixLaunchParams.use_adaptive_sampling && variance_buffer != nullptr) {
        float k = float(samples_this_pass);
        float batch_mean = batch_lum_sum / k;
        float batch_M2 = batch_lum_sq_sum - batch_lum_sum * batch_mean;

        float prev_M2 = variance_buffer[pixel_index];
        float delta = batch_mean - prev_mean_lum_for_variance;
        float combined_M2 = prev_M2 + batch_M2;
        if (prev_samples_for_variance > 0.0f) {
            combined_M2 += delta * delta * (prev_samples_for_variance * k) / new_total_samples;
        }
        variance_buffer[pixel_index] = fminf(fmaxf(combined_M2, 0.0f), 1.0e8f);
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
