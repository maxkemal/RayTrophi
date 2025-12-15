#pragma once
#include "trace_ray.cuh"
#include "material_scatter.cuh"
#include "random_utils.cuh"
#include "ray.h"
#include "scatter_volume_step.h"

extern AtmosphereProperties g_atmosphere;
__device__ void compute_atmosphere(
    const Ray& ray, float t_max, const AtmosphereProperties& params,
    float3* out_transmittance, float3* out_inscattered)
{
    // Ray marching (basit)
    const int steps = 32;
    float3 transmittance = make_float3(1.0f, 1.0f, 1.0f);
    float3 inscattered = make_float3(0.0f, 0.0f, 0.0f);
    float dt = t_max / steps;

    for (int i = 0; i < steps; ++i) {
        float t = dt * (i + 0.5f);
        float3 pos = ray.origin + t * ray.direction;

        float density = params.base_density * exp(-pos.y * 0.001f); // basit height decay
        float3 scatter = make_float3(params.sigma_s * density, params.sigma_s * density, params.sigma_s * density);
        float3 absorb = make_float3(params.sigma_a * density, params.sigma_a * density, params.sigma_a * density);

        float3 step_transmittance = exp_componentwise(-dt * absorb);
        transmittance *= step_transmittance;

        float3 incident = make_float3(1.0f, 1.0f, 1.0f); // Güneş ışığı vs.
        inscattered += transmittance * scatter * incident * dt;
    }

    *out_transmittance = transmittance;
    *out_inscattered = inscattered;
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
    trace_shadow_ray(shadow_ray, &shadow_payload, 0.01f, distance);
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
    //bool specular_bounce = false;
    const int max_depth = optixLaunchParams.max_depth;
    int light_count = optixLaunchParams.light_count;
    int light_index = (light_count > 0) ? pick_smart_light(ray.origin, rng) : -1;

    for (int bounce = 0; bounce < max_depth; ++bounce) {

        OptixHitResult payload = {};
        trace_ray(ray, &payload);

        if (!payload.hit) {
			// --- Eğer ışık yoksa arka plan rengi ve atmosfer katkısı ekle ---
            float3 transmittance = make_float3(1.0f, 1.0f, 1.0f);
            float3 inscattered = make_float3(0.0f, 0.0f, 0.0f);
            if (optixLaunchParams.atmosphere.active) {
                float t_far = 1000.0f; // max distance (gerekirse dinamik al)
                compute_atmosphere(ray, t_far, optixLaunchParams.atmosphere, &transmittance, &inscattered);
            }
           
            color += throughput * (optixLaunchParams.background_color  * transmittance + inscattered) ;
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
            if (bounce > 2) {
                if (random_float(rng) > p)
                    break;
                throughput /= p;
            }
        // --- Eğer hiç ışık yoksa sadece emissive katkı yap ---
        if (light_count == 0) {
            // Emission texture varsa kullan
            float3 emission = payload.emission;
            if (payload.has_emission_tex) {
                float4 tex = tex2D<float4>(payload.emission_tex, payload.uv.x, payload.uv.y);
                emission = make_float3(tex.x, tex.y, tex.z) * mat.emission;
            }
            color += throughput * emission;
            // throughput zaten satır 443'te attenuation ile çarpıldı, tekrar çarpma!
            ray = scattered;
            continue;
        }

            light_index = pick_smart_light(payload.position, rng);
       
        // --- Direkt katkı ---
        float3 direct = make_float3(0.0f, 0.0f, 0.0f);
        if (!is_specular && light_index >= 0) {
            direct = calculate_light_contribution(
                optixLaunchParams.lights[light_index], mat, payload, wo, rng
            );
			 // Direkt katkıyı throughput ile çarp
        }

        // --- BRDF yönünde MIS katkı ---
        // Specular (delta dağılım) yüzeylerde MIS yapma - transmission/opacity geçişleri dahil
        float3 brdf_mis = make_float3(0.0f, 0.0f, 0.0f);
        if (!is_specular && light_index >= 0) {

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
      
        float3 emission = payload.emission;
        // --- Toplam katkı ---
        color += throughput * (direct + brdf_mis + emission);
       
        ray = scattered;
		

    }

    return color;
}
