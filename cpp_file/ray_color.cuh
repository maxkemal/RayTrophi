﻿#pragma once
#include "trace_ray.cuh"
#include "material_scatter.cuh"
#include "random_utils.cuh"
#include "ray.h"
#include "scatter_volume_step.h"

extern AtmosphereProperties g_atmosphere;

__device__ float power_heuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-4f);
}
__device__ float balance_heuristic(float pdf_a, float pdf_b) {
    return pdf_a / (pdf_a + pdf_b + 1e-4f);
}
__device__ int pick_smart_light(const float3& hit_position, curandState* rng) {
    int light_count = optixLaunchParams.light_count;
    if (light_count == 0) return -1;

    // --- Öncelik: Directional light ---
    for (int i = 0; i < light_count; i++) {
        if (optixLaunchParams.lights[i].type == 1) {
            if (random_float(rng) < 0.33f)
                return i;
        }
    }

    // --- Point light seçim ---
    float weights[32];
    float total_weight = 0.0f;
    for (int i = 0; i < light_count; i++) {
        const LightGPU& light = optixLaunchParams.lights[i];
        if (light.type == 0) {
            float dist = length(light.position - hit_position);
            dist = fmaxf(dist, 1.0f);
           
        }
        else {
            weights[i] = 0.0f;
        }
    }

    if (total_weight < 1e-6f)
        return clamp(int(random_float(rng) * light_count), 0, light_count - 1);

    float r = random_float(rng) * total_weight;
    float accum = 0.0f;
    for (int i = 0; i < light_count; i++) {
        accum += weights[i];
        if (r <= accum)
            return i;
    }

    return clamp(int(random_float(rng) * light_count), 0, light_count - 1);
}
__device__ float3 sample_directional_light(const LightGPU& light, const float3& hit_pos, curandState* rng, float3& wi_out) {
    float3 L = normalize(light.direction);
    float3 tangent = normalize(cross(L, make_float3(0.0f, 1.0f, 0.0f)));
    if (length(tangent) < 1e-3f) tangent = normalize(cross(L, make_float3(1.0f, 0.0f, 0.0f)));
    float3 bitangent = normalize(cross(L, tangent));

    float2 disk_sample = random_in_unit_disk(rng);
    float3 offset = (tangent * disk_sample.x + bitangent * disk_sample.y) * light.radius;

    float3 light_pos = hit_pos + L * 1000.0f + offset;
    wi_out = normalize(light_pos - hit_pos);
    return wi_out;
}

__device__ float3 calculate_light_contribution(
    const LightGPU& light,
    const GpuMaterial& material,
    const OptixHitResult& payload,
    const float3& wo,
    curandState* rng
) {
    float3 wi;
    float distance = 1.0f;
    float attenuation = 1.0f;
    const float shadow_bias = 1e-2f;

    if (light.type == 0) { // Point Light
        float3 L = light.position - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return make_float3(0.0f, 0.0f, 0.0f);
        float3 dir = normalize(L);
        float3 jitter = light.radius * random_in_unit_sphere(rng);
        wi = normalize(dir * distance + jitter);
        attenuation = 1.0f / (distance * distance);
    }
    else if (light.type == 1) { // Directional Light
        wi = sample_directional_light(light, payload.position, rng, wi);
        attenuation = 1.0f;
        distance = 1e8f;
    }
    else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float NdotL = dot(payload.normal, wi);
    //if (NdotL <= 0.001f) return make_float3(0.0f, 0.0f, 0.0f);

    float3 origin = payload.position + payload.normal * shadow_bias;
    Ray shadow_ray(origin, wi);
    OptixHitResult shadow_payload = {};
    trace_shadow_ray(shadow_ray, &shadow_payload, 0.01f, distance);
    if (shadow_payload.hit) return make_float3(0.0f, 0.0f, 0.0f);

    float3 f = evaluate_brdf(material, payload, wo, wi);
    float pdf_brdf_val = pdf_brdf(material, wo, wi, payload.normal);
    float pdf_brdf_val_mis = clamp(pdf_brdf_val, 0.1f, 5000.0f);

    float pdf_light = 1.0f;
    if (light.type == 0) {
        float area = 4.0f * M_PIf * light.radius * light.radius;
        pdf_light = 1.0f / area;
    }
    else if (light.type == 1) {
        float apparent_angle = atan2(light.radius, 1000.0f);
        float cos_epsilon = cos(apparent_angle);
        float solid_angle = 2.0f * M_PIf * (1.0f - cos_epsilon);
        pdf_light = 1.0f / solid_angle;
    }

    float mis_weight = power_heuristic(pdf_light, pdf_brdf_val_mis);
    float3 Li = light.color * light.intensity * attenuation;
    return (f * Li * NdotL) * mis_weight;
}

__device__ float3 calculate_direct_lighting(
    const OptixHitResult& payload,
    const float3& wo,
    curandState* rng
) {
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    GpuMaterial mat = optixLaunchParams.materials[payload.material_id];

    int light_count = optixLaunchParams.light_count;
    if (light_count == 0) return result;

    // ------ YENİ: Rastgele bir ışık seç ------
    int light_index = clamp((int)(random_float(rng) * light_count), 0, light_count - 1);
    const LightGPU& light = optixLaunchParams.lights[light_index];

    float pdf_light_select = 1.0f / light_count;

    float3 wi;
    float distance = 1.0f;
    float attenuation = 1.0f;
    const float shadow_bias = 1e-2f;

    // ==== Light sampling ====
    if (light.type == 0) { // Point Light
        float3 L = light.position - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return result;

        float3 dir = normalize(L);
        float3 jitter = light.radius * random_in_unit_sphere(rng);
        wi = normalize(dir * distance + jitter);
        attenuation = 1.0f / (distance * distance);
    }
    else if (light.type == 1) { // Directional Light
        float3 jitter = light.radius * random_in_unit_sphere(rng);
        wi = normalize(light.direction + jitter);
        attenuation = 1.0f;
        distance = 1e8f;
    }
    else {
        return result;
    }

    float NdotL = dot(payload.normal, wi);
    if (NdotL <= 0.001f) return result;

    // ==== Shadow ray ====
    float3 origin = payload.position + payload.normal * shadow_bias;
    Ray shadow_ray(origin, wi);

    OptixHitResult shadow_payload = {};
    trace_shadow_ray(shadow_ray, &shadow_payload, 0.001f, distance);
    if (shadow_payload.hit) return result;

    // ==== BRDF & PDF ====
    float3 f = evaluate_brdf(mat, payload, wo, wi);
    float pdf_brdf_val = pdf_brdf(mat, wo, wi, payload.normal);
    float pdf_brdf_val_mis = clamp(pdf_brdf_val, 0.1f, 5000.0f);

    // ==== Light PDF ====
    float pdf_light = 1.0f;
    if (light.type == 0) {
        float area = 4.0f * M_PIf * light.radius * light.radius;
        pdf_light = (1.0f / area)* pdf_light_select;
    }
    else if (light.type == 1) {
        float apparent_angle = atan2(light.radius, 1000.0f);
        float cos_epsilon = cos(apparent_angle);
        float solid_angle = 2.0f * M_PIf * (1.0f - cos_epsilon);
        pdf_light = (1.0f / solid_angle)* pdf_light_select;
    }

    float mis_weight = power_heuristic(pdf_light, pdf_brdf_val_mis);

    float3 Li = light.color * light.intensity * attenuation;
    result += (f * Li * NdotL) * mis_weight * light_count;
    return result;
}

__device__ float3 calculate_brdf_mis(
    const OptixHitResult& payload,
    const float3& wo,
    const Ray& scattered,
    const GpuMaterial& mat,
    const float pdf,
    curandState* rng
)
{
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    float3 wi = normalize(scattered.direction);

    float pdf_brdf_val_mis = clamp(pdf, 0.1f, 5000.0f);

    // ------ YENİ: Rastgele bir ışık seç ------
    int light_count = optixLaunchParams.light_count;
    if (light_count == 0) return result;  // Işık yoksa katkı yok.

    int light_index = clamp((int)(random_float(rng) * light_count), 0, light_count - 1);
    const LightGPU& light = optixLaunchParams.lights[light_index];

    // --- PDF light seçim katsayısı ---
    float pdf_light_select = 1.0f / light_count;

    // ---------- ESKİ KODLAR --------------
    if (light.type == 1) { // Directional
        float3 L = normalize(light.direction);
        float alignment = dot(wi, L);
        if (alignment > 0.999f) {
            float apparent_angle = atan2(light.radius, 1000.0f);
            float cos_epsilon = cos(apparent_angle);
            float solid_angle = 2.0f * M_PIf * (1.0f - cos_epsilon);
            float pdf_light = (1.0f / solid_angle) * pdf_light_select;  // Dikkat: pdf ışık seçimiyle bölünüyor

            float mis_weight = power_heuristic(pdf_brdf_val_mis, pdf_light);

            float3 f = evaluate_brdf(mat, payload, wo, wi);
            float NdotL = fmaxf(dot(payload.normal, wi), 0.0f);
            float3 Li = light.color * light.intensity ;

            result += (f * Li * NdotL) * mis_weight * light_count; // Light_count çarpılıyor, çünkü sadece bir ışık örneklendi.
        }
    }

    if (light.type == 0) { // Point Light
        float3 delta = light.position - payload.position;
        float dist = length(delta);
        if (dist < light.radius * 1.05f) {
            float area = 4.0f * M_PIf * light.radius * light.radius;
            float pdf_light = (1.0f / area) * pdf_light_select;

            float mis_weight = power_heuristic(pdf_brdf_val_mis, pdf_light);

            float3 f = evaluate_brdf(mat, payload, wo, wi);
            float NdotL = fmaxf(dot(payload.normal, wi), 0.0f);
            float3 Li = light.color * light.intensity / (dist * dist);

            result += (f * Li * NdotL) * mis_weight * light_count; // Sadece seçilen ışık örneklendiği için çarpım.
        }
    }

    return result;
}

__device__ float3 ray_color(Ray ray, curandState* rng) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    bool specular_bounce = false;
    const int max_depth = optixLaunchParams.max_depth;
    int light_count = optixLaunchParams.light_count;
    int light_index = (light_count > 0) ? pick_smart_light(ray.origin, rng) : -1;

    for (int bounce = 0; bounce < max_depth; ++bounce) {

        OptixHitResult payload = {};
        trace_ray(ray, &payload);

        if (!payload.hit) {
            float falloff = (bounce == 0) ? 1.0f : __powf(0.1f, bounce); // Her bounce'ta azalır
            color += throughput * optixLaunchParams.background_color * falloff;
            break;
        }

        float3 wo = -normalize(ray.direction);

        Ray scattered;
        float3 attenuation;
        float pdf;
        bool is_specular;
        GpuMaterial mat = optixLaunchParams.materials[payload.material_id];

        // --- Scatter başarısızsa çık ---
        if (!scatter_material(mat, payload, ray, rng, &scattered, &attenuation, &pdf, &is_specular))
            break;
        throughput *= attenuation;
        // --- Russian roulette ---
       
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            p = clamp(p, 0.0f, 0.98f);
            if (random_float(rng) > p)
                break;
            throughput /= p;
       
        // --- Eğer hiç ışık yoksa sadece emissive katkı yap ---
        if (light_count == 0) {
            color += throughput * payload.emission;
            throughput *= attenuation;
            ray = scattered;
            continue;
        }

            light_index = pick_smart_light(payload.position, rng);
       
        // --- Direkt katkı ---
        float3 direct = make_float3(0.0f, 0.0f, 0.0f);
        if (light_index >= 0) {
            direct = calculate_light_contribution(
                optixLaunchParams.lights[light_index], mat, payload, wo, rng
            );
        }

        // --- BRDF yönünde MIS katkı ---
        float3 brdf_mis = make_float3(0.0f, 0.0f, 0.0f);
        if (light_index >= 0) {

            const LightGPU& light = optixLaunchParams.lights[light_index];
            float3 wi = normalize(scattered.direction);
            float pdf_brdf_val_mis = clamp(pdf, 0.1f, 5000.0f);
            float pdf_light = 1.0f;
            float NdotL = fmaxf(dot(payload.normal, wi), 0.0f);

            if (light.type == 1) {
                float3 L = normalize(light.direction);
                if (dot(wi, L) > 0.999f) {
                    float solid_angle = 2.0f * M_PIf * (1.0f - cos(atan2(light.radius, 1000.0f)));
                    pdf_light = 1.0f / solid_angle;
                    float mis_weight = power_heuristic(pdf_brdf_val_mis, pdf_light);
                    float3 f = evaluate_brdf(mat, payload, wo, wi);
                    brdf_mis += f * light.intensity*light.color * NdotL * mis_weight;
                }
            }
            if (light.type == 0) {
                float3 delta = light.position - payload.position;
                float dist = length(delta);
                if (dist < light.radius * 1.05f) {
                    float area = 4.0f * M_PIf * light.radius * light.radius;
                    pdf_light = 1.0f / area;
                    float mis_weight = power_heuristic(pdf_brdf_val_mis, pdf_light);
                    float3 f = evaluate_brdf(mat, payload, wo, wi);
                    brdf_mis += f * (light.intensity*light.color / (dist * dist)) * NdotL * mis_weight;
                }
            }
        }
       
        // --- Toplam katkı ---
        color += throughput * (payload.emission*0.5 + direct + brdf_mis);
       
        ray = scattered;
        specular_bounce = is_specular;
    }

    return color;
}
