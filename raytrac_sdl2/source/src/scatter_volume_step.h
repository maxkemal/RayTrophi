// scatter_volume_step.cuh
#pragma once
#include "vec3_utils.cuh"
#include "Ray.h"
#include "random_utils.cuh"
#include <curand_kernel.h>   
#include <math_constants.h>  

__device__ float3 sample_phase_function(const float3& wo, float g, curandState* rng) {
    float rand1 = random_float(rng);
    float rand2 = random_float(rng);

    float cos_theta;
    if (fabsf(g) < 1e-3f) {
        cos_theta = 1.0f - 2.0f * rand1;
    }
    else {
        float sqr_term = (1.0f - g * g) / (1.0f - g + 2.0f * g * rand1);
        cos_theta = (1.0f + g * g - sqr_term * sqr_term) / (2.0f * g);
    }

    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * CUDART_PI_F * rand2;

    float3 u, v, w;
    w = normalize(wo);
    build_coordinate_system(w, u, v);

    float3 new_dir = normalize(
        u * cosf(phi) * sin_theta +
        v * sinf(phi) * sin_theta +
        w * cos_theta
    );

    return new_dir;
}
__device__ float compute_phase_function(const float3& wo, const float3& wi, float g) {
    float cos_theta = dot(wo, wi);
    float denom = 1.0f + g * g - 2.0f * g * cos_theta;
    return (1.0f - g * g) / (4.0f * CUDART_PI_F * denom * sqrtf(denom));
}
__device__ float sample_density(const float3& pos) {
    float scale = 0.05f; // Frekans (daha düşük → daha geniş dalgalar)
    float noise = 0.5f * (sinf(pos.x * scale) + sinf(pos.y * scale) + sinf(pos.z * scale));
    noise = clamp(noise, -0.5f, 0.5f); // Sinüs aralığını güvene alalım
    return noise + 0.5f; // [0,1] aralığına taşır (0.0 minimum - 1.0 maksimum)
}
__device__ bool scatter_volume_step(Ray& ray, float3& throughput, float3& color, const WorldData& world, curandState* rng)
{
    // Placeholder mapping since we removed AtmosphereProperties
    float density = world.volume_density;
    float anisotropy = world.volume_anisotropy;
    // Assuming simple uniform fog for now if density > 0
    // But since we don't have sigma_s/a directly, let's use density as sigma_s and a small default for sigma_a
    
    if (density <= 0.0f) {
        return false;
    }
    float sigma_s = density;
    float sigma_a = 0.0f; // Pure scattering for now? or small abs.

    // 1. Scatter mesafesi örnekle
    float scatter_distance = -logf(random_float(rng)) / sigma_s;
    ray.origin += scatter_distance * ray.direction;

    // 2. Absorption uygula
    throughput *= expf(-sigma_a * scatter_distance);

    // 3. Işık katkısını hacimden hesapla
    float3 scatter_color = make_float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < optixLaunchParams.light_count; ++i) {
        const LightGPU& light = optixLaunchParams.lights[i];

        float3 light_dir;
        float distance = 1.0f;
        float light_intensity = 1.0f;

        if (light.type == 0) { // Point Light
            light_dir = normalize(light.position - ray.origin);
            distance = length(light.position - ray.origin);
            light_intensity = 1.0f / (distance * distance);
        }
        else if (light.type == 1) { // Directional Light
            light_dir = normalize(light.direction);
            distance = 1e8f;
            light_intensity = 1.0f;
        }
        else {
            continue; // Bilinmeyen ışık tipi
        }

        // Gölge kontrolü
        Ray shadow_ray(ray.origin, light_dir);
        OptixHitResult shadow_payload = {};
        trace_shadow_ray(shadow_ray, &shadow_payload, 0.01f, distance);

        if (!shadow_payload.hit) { // Eğer ışık görünüyorsa
            float phase = compute_phase_function(-ray.direction, light_dir, anisotropy);
            scatter_color +=  make_float3(light_intensity * phase, light_intensity * phase,light_intensity * phase); //  Işık rengini ve şiddetini phase ile ağırlıkla
        }
    }
    float local_density = sample_density(ray.origin);
    scatter_color *= local_density; // yoğunluğa göre azalt/arttır
    // 4. Color katkısı ekle
    color += throughput * scatter_color;

    // 5. Yeni scatter yönü örnekle
    float3 u, v, w;
    w = normalize(ray.direction);
    build_coordinate_system(w, u, v);

    float rand1 = random_float(rng);
    float rand2 = random_float(rng);

    float cos_theta;
    if (fabsf(anisotropy) < 1e-3f) {
        cos_theta = 1.0f - 2.0f * rand1;
    }
    else {
        float sqr_term = (1.0f - anisotropy * anisotropy) / (1.0f - anisotropy + 2.0f * anisotropy * rand1);
        cos_theta = (1.0f + anisotropy * anisotropy - sqr_term * sqr_term) / (2.0f * anisotropy);
    }

    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * CUDART_PI_F * rand2;

    ray.direction = normalize(
        u * cosf(phi) * sin_theta +
        v * sinf(phi) * sin_theta +
        w * cos_theta
    );

    return true;
}
