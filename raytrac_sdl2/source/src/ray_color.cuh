#pragma once
#include "trace_ray.cuh"
#include <math_constants.h>

#ifndef M_1_PIf
#define M_1_PIf 0.318309886183790671538f
#endif
#include "material_scatter.cuh"
#include "random_utils.cuh"
#include "ray.h"
#include "scatter_volume_step.h"

__device__ float3 evaluate_background(const WorldData& world, const float3& dir) {
    if (world.mode == 0) { // WORLD_MODE_COLOR
        return world.color;
    }
    else if (world.mode == 1) { // WORLD_MODE_HDRI
        if (world.env_texture) {
            float theta = acosf(dir.y);
            float phi = atan2f(-dir.z, dir.x) + M_PIf;
            
            float u = phi * (0.5f * M_1_PIf); // 0..1
            float v = theta * M_1_PIf;        // 0..1
            
            u -= world.env_rotation / (2.0f * M_PIf);
            u -= floorf(u);
            
            float4 tex = tex2D<float4>(world.env_texture, u, v);
            return make_float3(tex.x, tex.y, tex.z) * world.env_intensity;
        }
        return world.color;
    }
    else if (world.mode == 2) { // WORLD_MODE_NISHITA
        // Nishita Sky Model (Single Scattering)
        float3 sunDir = normalize(world.nishita.sun_direction);
        float planetRadius = world.nishita.planet_radius;
        float atmosphereRadius = world.nishita.atmosphere_height;
        float Rt = atmosphereRadius;
        float Rg = planetRadius;
        
        // Assume camera is on ground at (0, Rg, 0)
        float3 camPos = make_float3(0.0f, Rg + 10.0f, 0.0f); 
        float3 rayDir = normalize(dir);
        
        // Ray-Sphere Intersection (Atmosphere)
        float a = dot(rayDir, rayDir);
        float b = 2.0f * dot(rayDir, camPos);
        float c = dot(camPos, camPos) - Rt * Rt;
        float delta = b * b - 4.0f * a * c;
        
        if (delta < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
        
        float t1 = (-b - sqrtf(delta)) / (2.0f * a);
        float t2 = (-b + sqrtf(delta)) / (2.0f * a);
        float t = (t1 >= 0.0f) ? t1 : t2;
        if (t < 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
        
        int numSamples = 8;
        float stepSize = t / (float)numSamples;
        
        float3 totalRayleigh = make_float3(0.0f, 0.0f, 0.0f);
        float3 totalMie = make_float3(0.0f, 0.0f, 0.0f);
        
        float opticalDepthRayleigh = 0.0f;
        float opticalDepthMie = 0.0f;
        
        // Phase Functions
        float mu = dot(rayDir, sunDir);
        float phaseR = 3.0f / (16.0f * M_PIf) * (1.0f + mu * mu);
        float g = world.nishita.mie_anisotropy;
        float phaseM = 3.0f / (8.0f * M_PIf) * ((1.0f - g * g) * (1.0f + mu * mu)) / ((2.0f + g * g) * powf(1.0f + g * g - 2.0f * g * mu, 1.5f));
        
        float currentT = 0.0f;
        
        for (int i = 0; i < numSamples; ++i) {
            float3 samplePos = camPos + rayDir * (currentT + stepSize * 0.5f);
            float height = length(samplePos) - Rg;
            if (height < 0.0f) height = 0.0f;
            
            float hr = expf(-height / world.nishita.rayleigh_density);
            float hm = expf(-height / world.nishita.mie_density);
            
            opticalDepthRayleigh += hr * stepSize;
            opticalDepthMie += hm * stepSize;
            
            // Optical depth to sun
            float b_light = 2.0f * dot(sunDir, samplePos);
            float c_light = dot(samplePos, samplePos) - Rt * Rt;
            float delta_light = b_light * b_light - 4.0f * c_light;
            
            if (delta_light >= 0.0f) {
                float t_light = (-b_light + sqrtf(delta_light)) / 2.0f;
                
                int numLightSamples = 4;
                float lightStep = t_light / (float)numLightSamples;
                float lightOpticalRayleigh = 0.0f;
                float lightOpticalMie = 0.0f;
                
                for(int j=0; j<numLightSamples; ++j) {
                    float3 lightSamplePos = samplePos + sunDir * (lightStep * (j + 0.5f));
                    float lightHeight = length(lightSamplePos) - Rg;
                    if(lightHeight < 0.0f) lightHeight = 0.0f;
                    
                    lightOpticalRayleigh += expf(-lightHeight / world.nishita.rayleigh_density) * lightStep;
                    lightOpticalMie += expf(-lightHeight / world.nishita.mie_density) * lightStep;
                }
                
                float3 tau = world.nishita.rayleigh_scattering * (opticalDepthRayleigh + lightOpticalRayleigh) + 
                             world.nishita.mie_scattering * 1.1f * (opticalDepthMie + lightOpticalMie);
                             
                float3 attenuation = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
                
                totalRayleigh += attenuation * hr * stepSize;
                totalMie += attenuation * hm * stepSize;
            }
            currentT += stepSize;
        }
        
        float3 L = (totalRayleigh * world.nishita.rayleigh_scattering * phaseR + 
                totalMie * world.nishita.mie_scattering * phaseM) * world.nishita.sun_intensity;

        // Add Sun Disk
        // Sun angular radius ~0.266 degrees (0.00465 rad)
        const float sun_radius = 0.02f; // Increased for visibility (~1.15 degrees)
        if (dot(rayDir, sunDir) > cosf(sun_radius)) {
            // Transmittance to space (accumulated optical depth)
            float3 tau = world.nishita.rayleigh_scattering * opticalDepthRayleigh + 
                         world.nishita.mie_scattering * 1.1f * opticalDepthMie;
            float3 transmittance = make_float3(expf(-tau.x), expf(-tau.y), expf(-tau.z));
            L += transmittance * world.nishita.sun_intensity * 100.0f; // Direct sun (scaled for visibility)
        }
        return L;
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

__device__ float power_heuristic(float pdf_a, float pdf_b) {
    float a2 = pdf_a * pdf_a;
    float b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-4f);
}
__device__ float balance_heuristic(float pdf_a, float pdf_b) {
    return pdf_a / (pdf_a + pdf_b + 1e-4f);
}
__device__ float luminance(const float3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

__device__ int pick_smart_light(const float3& hit_position, curandState* rng) {
    int light_count = optixLaunchParams.light_count;
    if (light_count == 0) return -1;

    // --- Öncelik: Directional light --- (örnekleme olasılığı %33)
    for (int i = 0; i < light_count; i++) {
        if (optixLaunchParams.lights[i].type == 1) {
            if (random_float(rng) < 0.33f)
                return i;
        }
    }

    // --- Tüm ışık türleri için akıllı seçim ---
    float weights[128];
    float total_weight = 0.0f;

    for (int i = 0; i < light_count; i++) {
        const LightGPU& light = optixLaunchParams.lights[i];
        float dist = length(light.position - hit_position);
        dist = fmaxf(dist, 1.0f);
        
        float falloff = 1.0f / (dist * dist);
        float intensity = luminance(light.color * light.intensity);
        
        if (light.type == 0) { // Point Light
            weights[i] = falloff * intensity;
        }
        else if (light.type == 2) { // Area Light
            float area = light.area_width * light.area_height;
            weights[i] = falloff * intensity * fminf(area, 10.0f);
        }
        else if (light.type == 3) { // Spot Light
            weights[i] = falloff * intensity * 0.8f;
        }
        else {
            weights[i] = 0.0f;
        }
        
        total_weight += weights[i];
    }

    // --- Eğer total_weight çok düşükse fallback ---
    if (total_weight < 1e-6f)
        return clamp(int(random_float(rng) * light_count), 0, light_count - 1);

    // --- Weighted seçim ---
    float r = random_float(rng) * total_weight;
    float accum = 0.0f;
    for (int i = 0; i < light_count; i++) {
        accum += weights[i];
        if (r <= accum)
            return i;
    }

    // --- Güvenli fallback ---
    return clamp(int(random_float(rng) * light_count), 0, light_count - 1);
}

__device__ float3 sample_directional_light(const LightGPU& light, const float3& hit_pos, curandState* rng, float3& wi_out) {
    float3 L = normalize(light.direction);
    float3 tangent = normalize(cross(L, make_float3(0.0f, 1.0f, 0.0f)));
    if (length(tangent) < 1e-3f) tangent = normalize(cross(L, make_float3(1.0f, 0.0f, 0.0f)));
    float3 bitangent = normalize(cross(L, tangent));

    float2 disk_sample = random_in_unit_disk(rng);
    float3 offset = (tangent * disk_sample.x + bitangent * disk_sample.y) * light.radius;

    float3 light_pos = hit_pos + L * 1000 + offset;
    wi_out = normalize(light_pos - hit_pos);
    return wi_out;
}

// AreaLight için rastgele nokta örnekleme
__device__ float3 sample_area_light(const LightGPU& light, curandState* rng) {
    float rand_u = random_float(rng) - 0.5f;
    float rand_v = random_float(rng) - 0.5f;
    return light.position 
        + light.area_u * rand_u * light.area_width 
        + light.area_v * rand_v * light.area_height;
}

// SpotLight için cone falloff hesabı
__device__ float spot_light_falloff(const LightGPU& light, const float3& wi) {
    float cos_theta = dot(-wi, normalize(light.direction));
    if (cos_theta < light.outer_cone_cos) return 0.0f;
    if (cos_theta > light.inner_cone_cos) return 1.0f;
    // Smooth falloff between inner and outer cone
    float t = (cos_theta - light.outer_cone_cos) / (light.inner_cone_cos - light.outer_cone_cos + 1e-6f);
    return t * t;  // Quadratic falloff
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
    else if (light.type == 2) { // Area Light
        float3 light_sample = sample_area_light(light, rng);
        float3 L = light_sample - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return make_float3(0.0f, 0.0f, 0.0f);
        wi = normalize(L);
        
        // Cosine falloff based on light normal
        float3 light_normal = normalize(cross(light.area_u, light.area_v));
        float cos_light = fmaxf(dot(-wi, light_normal), 0.0f);
        attenuation = cos_light / (distance * distance);
    }
    else if (light.type == 3) { // Spot Light
        float3 L = light.position - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return make_float3(0.0f, 0.0f, 0.0f);
        wi = normalize(L);
        
        // Spot cone falloff
        float falloff = spot_light_falloff(light, wi);
        if (falloff < 1e-4f) return make_float3(0.0f, 0.0f, 0.0f);
        attenuation = falloff / (distance * distance);
    }
    else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float NdotL = max(dot(payload.normal, wi),0.0001);
    //if (NdotL <= 0.001f) return make_float3(0.0f, 0.0f, 0.0f);

    float3 origin = payload.position + payload.normal * shadow_bias;
    Ray shadow_ray(origin, wi);
    OptixHitResult shadow_payload = {};
    trace_shadow_ray(shadow_ray, &shadow_payload, 0.01f, distance);
    if (shadow_payload.hit) return make_float3(0.0f, 0.0f, 0.0f);

    float3 f = evaluate_brdf(material, payload, wo, wi);
    float pdf_brdf_val = pdf_brdf(material, wo, wi, payload.normal);
    float pdf_brdf_val_mis = clamp(pdf_brdf_val, 0.001f, 5000.0f);

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
    else if (light.type == 2) { // Area Light
        float area = light.area_width * light.area_height;
        pdf_light = 1.0f / fmaxf(area, 1e-4f);
    }
    else if (light.type == 3) { // Spot Light
        float solid_angle = 2.0f * M_PIf * (1.0f - light.outer_cone_cos);
        pdf_light = 1.0f / fmaxf(solid_angle, 1e-4f);
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
    const float shadow_bias = 1e-3f; // Match CPU bias (was 1e-2f)


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
    else if (light.type == 2) { // Area Light
        float3 light_sample = sample_area_light(light, rng);
        float3 L = light_sample - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return result;
        wi = normalize(L);
        
        float3 light_normal = normalize(cross(light.area_u, light.area_v));
        float cos_light = fmaxf(dot(-wi, light_normal), 0.0f);
        attenuation = cos_light / (distance * distance);
    }
    else if (light.type == 3) { // Spot Light
        float3 L = light.position - payload.position;
        distance = length(L);
        if (distance < 1e-3f) return result;
        wi = normalize(L);
        
        float falloff = spot_light_falloff(light, wi);
        if (falloff < 1e-4f) return result;
        attenuation = falloff / (distance * distance);
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
    trace_shadow_ray(shadow_ray, &shadow_payload, shadow_bias, distance);
    if (shadow_payload.hit) return result;

    // ==== BRDF & PDF ====
    float3 f = evaluate_brdf(mat, payload, wo, wi);
    float pdf_brdf_val = pdf_brdf(mat, wo, wi, payload.normal);
    float pdf_brdf_val_mis = clamp(pdf_brdf_val, 0.01f, 5000.0f);

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
        pdf_light = (1.0f / solid_angle) * pdf_light_select;
    }
    else if (light.type == 2) { // Area Light
        float area = light.area_width * light.area_height;
        pdf_light = (1.0f / fmaxf(area, 1e-4f)) * pdf_light_select;
    }
    else if (light.type == 3) { // Spot Light
        float solid_angle = 2.0f * M_PIf * (1.0f - light.outer_cone_cos);
        pdf_light = (1.0f / fmaxf(solid_angle, 1e-4f)) * pdf_light_select;
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
    const int max_depth = optixLaunchParams.max_depth;
    int light_count = optixLaunchParams.light_count;
    int light_index = (light_count > 0) ? pick_smart_light(ray.origin, rng) : -1;
    
    // Firefly önleme için maksimum katkı limiti
    const float MAX_CONTRIBUTION = 100.0f;

    for (int bounce = 0; bounce < max_depth; ++bounce) {

        OptixHitResult payload = {};
        trace_ray(ray, &payload);

        if (!payload.hit) {
            // --- Arka plan rengi ---
            float3 bg_color = evaluate_background(optixLaunchParams.world, ray.direction);
           
            // Bounce bazlı arka plan azaltma - ilk bounce tam, sonrakiler azaltılmış
            // Bu, yansımalarda arka plan renginin yüzeyleri boyamasını önler
            float bg_factor = (bounce == 0) ? 1.0f : fmaxf(0.1f, 1.0f / (1.0f + bounce * 0.5f));
            float3 bg_contribution = bg_color * bg_factor;
            
            color += throughput * bg_contribution;
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

        // --- GPU VOLUMETRIC RENDERING (SMOKE) ---
        if (mat.anisotropic > 0.9f) { // Flagged as Volumetric
            // Ray Marching only if entering the volume
            if (dot(ray.direction, payload.normal) < 0.0f) {
                // Find exit point
                Ray march_ray(payload.position + ray.direction * 0.01f, ray.direction);
                OptixHitResult exit_payload = {};
                trace_ray(march_ray, &exit_payload);

                if (exit_payload.hit) {
                    float dist = length(exit_payload.position - payload.position);
                    // dist limiting to avoid infinite march
                    if(dist > 20.0f) dist = 20.0f; 

                    int steps = 12; // Low step count for performance
                    float step_size = dist / steps;
                    float3 current_pos = payload.position;
                    float total_density = 0.0f;

                    for (int i = 0; i < steps; i++) {
                        current_pos += ray.direction * step_size * (random_float(rng) * 0.5f + 0.75f); // Jittered step

                        // Simple procedural noise for smoke
                        float3 s = current_pos * 3.5f; 
                        float noise = fabsf(sinf(s.x) * sinf(s.y + s.z * 0.5f) * cosf(s.z));
                        float density = fmaxf(0.0f, noise - 0.2f) * 4.0f; // Hardcoded parameters matching "Test Smoke"
                        
                        // Fade edges (approximate based on assumptions, hard on pure ray march without SDF)
                        
                        total_density += density * step_size;
                    }
                    
                    // Beer's Law (Transmittance)
                    float3 volume_albedo = make_float3(0.8f, 0.8f, 0.8f); // Grey smoke
                    float absorption = 0.2f;
                    float3 transmittance = make_float3(
                        expf(-total_density * absorption * (1.0f - volume_albedo.x)),
                        expf(-total_density * absorption * (1.0f - volume_albedo.y)),
                        expf(-total_density * absorption * (1.0f - volume_albedo.z))
                    );

                    throughput *= transmittance;
                    
                    // Add some scattered light (ambient approximation)
                    // color += throughput * make_float3(0.05f) * total_density; 

                    // Skip the volume interior by moving ray to exit point
                    scattered = Ray(exit_payload.position + ray.direction * 0.001f, ray.direction);
                }
            }
        }
        
        // --- Throughput clamp - aşırı parlak yansımaları önle ---
        float max_throughput = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        if (max_throughput > MAX_CONTRIBUTION) {
            throughput *= (MAX_CONTRIBUTION / max_throughput);
        }
        
        // --- Russian roulette - bounce > 2'den sonra ---
        float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
        p = clamp(p, 0.05f, 0.95f);  // Daha sıkı sınırlar
        if (bounce > 2) {
            if (random_float(rng) > p)
                break;
            throughput /= p;
            
            // Russian roulette sonrası tekrar clamp
            max_throughput = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (max_throughput > MAX_CONTRIBUTION) {
                throughput *= (MAX_CONTRIBUTION / max_throughput);
            }
        }
        
        // --- Eğer hiç ışık yoksa sadece emissive katkı yap ---
        if (light_count == 0) {
            float3 emission = payload.emission;
            if (payload.has_emission_tex) {
                float4 tex = tex2D<float4>(payload.emission_tex, payload.uv.x, payload.uv.y);
                emission = make_float3(tex.x, tex.y, tex.z) * mat.emission;
            }
            color += throughput * emission;
            ray = scattered;
            continue;
        }

        light_index = pick_smart_light(payload.position, rng);
       
        // --- Direkt ışık katkısı ---
        float3 direct = make_float3(0.0f, 0.0f, 0.0f);
        if (!is_specular && light_index >= 0) {
            direct = calculate_light_contribution(
                optixLaunchParams.lights[light_index], mat, payload, wo, rng
            );
            // Firefly kontrolü - aşırı parlak direkt katkıları sınırla
            float direct_lum = luminance(direct);
            if (direct_lum > MAX_CONTRIBUTION) {
                direct *= (MAX_CONTRIBUTION / direct_lum);
            }
        }

        // --- BRDF yönünde MIS katkı ---
        float3 brdf_mis = make_float3(0.0f, 0.0f, 0.0f);
        if (!is_specular && light_index >= 0) {
            const LightGPU& light = optixLaunchParams.lights[light_index];
            float3 wi = normalize(scattered.direction);
            float pdf_brdf_val_mis = clamp(pdf, 0.1f, 5000.0f);
            float pdf_light = 1.0f;
            float NdotL = fmaxf(dot(payload.normal, wi), 0.0f);

            if (light.type == 1) { // Directional
                float3 L = normalize(light.direction);
                if (dot(wi, L) > 0.999f) {
                    float solid_angle = 2.0f * M_PIf * (1.0f - cos(atan2(light.radius, 1000.0f)));
                    pdf_light = 1.0f / solid_angle;
                    float mis_weight = power_heuristic(pdf_brdf_val_mis, pdf_light);
                    float3 f = evaluate_brdf(mat, payload, wo, wi);
                    brdf_mis += f * light.intensity * light.color * NdotL * mis_weight;
                }
            }
            if (light.type == 0) { // Point
                float3 delta = light.position - payload.position;
                float dist = length(delta);
                if (dist < light.radius * 1.05f) {
                    float area = 4.0f * M_PIf * light.radius * light.radius;
                    pdf_light = 1.0f / area;
                    float mis_weight = power_heuristic(pdf_brdf_val_mis, pdf_light);
                    float3 f = evaluate_brdf(mat, payload, wo, wi);
                    brdf_mis += f * (light.intensity * light.color / (dist * dist)) * NdotL * mis_weight;
                }
            }
            
            // Firefly kontrolü - aşırı parlak BRDF MIS katkılarını sınırla
            float brdf_lum = luminance(brdf_mis);
            if (brdf_lum > MAX_CONTRIBUTION) {
                brdf_mis *= (MAX_CONTRIBUTION / brdf_lum);
            }
        }
      
        float3 emission = payload.emission;
        
        // --- Toplam katkı ---
        float3 total_contribution = direct + brdf_mis + emission;
        
        // Son firefly kontrolü
        float total_lum = luminance(total_contribution);
        if (total_lum > MAX_CONTRIBUTION * 2.0f) {
            total_contribution *= (MAX_CONTRIBUTION * 2.0f / total_lum);
        }
        
        color += throughput * total_contribution;
       
        ray = scattered;
    }

    // Final clamp - NaN ve Inf kontrolü
    color.x = isfinite(color.x) ? fminf(fmaxf(color.x, 0.0f), 100.0f) : 0.0f;
    color.y = isfinite(color.y) ? fminf(fmaxf(color.y, 0.0f), 100.0f) : 0.0f;
    color.z = isfinite(color.z) ? fminf(fmaxf(color.z, 0.0f), 100.0f) : 0.0f;

    return color;
}
