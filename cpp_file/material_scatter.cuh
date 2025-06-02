﻿// Safe and stable BRDF + scatter implementation
#pragma once
#include "vec3_utils.cuh"
#include "random_utils.cuh"
#include "ray.h"
#include "params.h"
#include "payload.h"
#include <material_gpu.h>


__device__ float G_SchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

__device__ float G_Smith(float NdotV, float NdotL, float roughness) {
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

__device__ float pdf_brdf(const GpuMaterial& mat, const float3& wo, const float3& wi, const float3& N) {
    float3 H = normalize(wo + wi);
    float NdotH = max(dot(N, H), 0.0001f);
    float VdotH = max(dot(wo, H), 0.0001f);

    float roughness = mat.roughness;
    float alpha = max(roughness * roughness, 0.001f);
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0f) + 1.0f;
    float D = alpha2 / (M_PIf * denom * denom);

    return D * NdotH / (4.0f * VdotH + 1e-4f);
}

__device__ float3 float3_min(const float3& a, const float3& b) {
    return make_float3(fminf(a.x, b.x),
        fminf(a.y, b.y),
        fminf(a.z, b.z));
}


__device__ float3 evaluate_brdf(
    const GpuMaterial& material,
    const OptixHitResult& payload,
    const float3& wo,
    const float3& wi
)
{
    const float3 N = payload.normal;
    float2 uv = payload.uv;
    uv = clamp(uv, 0.0f, 1.0f);

    float NdotL = max(dot(N, wi), 0.0f);
    float NdotV = max(dot(N, wo), 0.0f);
    if (NdotL == 0 || NdotV == 0) return make_float3(0.0f, 0.0f, 0.0f);

    float3 H = normalize(wi + wo);
    float NdotH = max(dot(N, H), 0.0001f);
    float VdotH = max(dot(wo, H), 0.0001f);
   
    float3 albedo = material.albedo;
    if (payload.has_albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, uv.x, uv.y);
        albedo = (tex.y == 0.0f && tex.z == 0.0f) ?
            make_float3(tex.x, tex.x, tex.x) :
            make_float3(tex.x, tex.y, tex.z);
    }


    float roughness = material.roughness;
    if (payload.has_roughness_tex) {
        roughness = tex2D<float4>(payload.roughness_tex, uv.x, uv.y).y;
       
    }
    
    float metallic = material.metallic;
    if (payload.has_metallic_tex) {
        metallic = tex2D<float4>(payload.metallic_tex, uv.x, uv.y).z;
       
    }

    float alpha = max(roughness * roughness, 0.001f);
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0f) + 1.0f;
    float D = alpha2 / (M_PIf * denom * denom);

    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float3 F = F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) * powf(1.0f - VdotH, 5.0f);

    float G = G_Smith(NdotV, NdotL, roughness);
    float3 spec = (F * D * G) / (4.0f * NdotV * NdotL + 0.001f);   
    float3 F_avg = F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) / 21.0f;

    float3 k_d = (make_float3(1.0f, 1.0f, 1.0f) - F_avg) * (1.0f - metallic);

    float3 diffuse = k_d * albedo/M_PIf;

    if (material.transmission >= 0.01f)
    {
        curandState rng;
        if (random_float(&rng) > material.transmission)
        {
            return diffuse;
        }
        return spec;
    }

    return (diffuse + spec);
}
__device__ float3 fresnel_schlick_roughness(float cosTheta, float3 F0, float roughness)
{
    // Bu versiyon daha yumuşak geçiş sağlar, Cycles benzeri
    return F0 + (max(make_float3(1.0f - roughness, 1.0f - roughness, 1.0f - roughness), F0) - F0) * powf(1.0f - cosTheta, 5.0f);
}

__device__ float3 importance_sample_ggx(float u1, float u2, float roughness, const float3& N) {
    float alpha = max(roughness * roughness, 0.001f);
    float phi = 2.0f * M_PIf * u1;
    float cosTheta = sqrtf((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sinTheta = sqrtf(max(1.0f - cosTheta * cosTheta, 0.0f));

    float3 H = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);

    float3 up = fabsf(N.z) < 0.999f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
    float3 tangentX = normalize(cross(up, N));
    float3 tangentY = cross(N, tangentX);

    return normalize(tangentX * H.x + tangentY * H.y + N * H.z);
}
__device__ float schlick(float cos_theta, float eta) {
    float r0 = (1.0f - eta) / (1.0f + eta);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);
}
__device__ bool refract(const float3& V, const float3& N, float eta, float3* out_refracted) {
    float cos_theta = fminf(dot(-V, N), 1.0f);
    float3 r_out_perp = eta * (V + cos_theta * N);
    float k = 1.0f - dot(r_out_perp, r_out_perp);
    if (k < 0.0f) return false;
    float3 r_out_parallel = -sqrtf(k) * N;
    *out_refracted = r_out_perp + r_out_parallel;
    return true;
}

__device__ float3 sample_sss_direction(const float3& normal, curandState* rng) {
    // Yüzeye yakın iç yönlere öncelik ver
    float3 dir = random_in_unit_sphere(rng);
    if (dot(dir, normal) < 0.0f)
        dir = -dir; // içeri yönlendir
    return normalize(dir);
}
__device__ float3 calculate_caustic_gpu(
    const float3& incident,
    const float3& normal,
    const float3& refracted,
    const float3& color,
    float caustic_intensity = 1.0f
) {
    float dot_product = dot(incident, normal);
    float refraction_angle = acosf(fminf(fmaxf(dot(refracted, -normal), -1.0f), 1.0f));
    float caustic_factor = powf(fmaxf(refraction_angle, 0.0f), 3.0f) * caustic_intensity;
    return color * caustic_factor * (1.0f - dot_product * dot_product);
}

__device__ bool transmission_scatter(
    const GpuMaterial& material,
    const OptixHitResult& payload,
    const Ray& ray_in,
    curandState* rng,
    Ray* scattered,
    float3* attenuation
)
{
    const float3 N = normalize(payload.normal);
    const float3 V = -normalize(ray_in.direction);
    float3 outward_normal = normalize(N);
    float3 unit_direction = normalize(ray_in.direction);
    float2 uv = payload.uv;

    float eta = dot(V, N) > 0.0f ? (1.0f / material.ior) : material.ior;

    float cos_theta = fminf(dot(-unit_direction, outward_normal), 1.0f);
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

    float reflect_prob = schlick(cos_theta, eta);
    bool cannot_refract = (eta * sin_theta > 1.0f);

    float direct_trans_prob = 0.05f;
    float refract_prob = cannot_refract ? 0.0f : (1.0f - reflect_prob) * (1.0f - direct_trans_prob);

    float total_prob = reflect_prob + refract_prob + direct_trans_prob;
    reflect_prob /= total_prob;
    refract_prob /= total_prob;
    direct_trans_prob /= total_prob;

    float random_val = random_float(rng);

    float3 direction;
    if (random_val < reflect_prob) {
        direction = reflect(unit_direction, outward_normal);
    }
    else if (random_val < reflect_prob + refract_prob) {
        bool refracted_success = refract(unit_direction, outward_normal, eta, &direction);
        if (!refracted_success) {
            direction = reflect(unit_direction, outward_normal);
        }
    }
    else {
        direction = unit_direction;
    }

    //  ROUGHNESS etkisi: Yönü hafif dağıt
    float roughness = material.roughness;
	if (payload.has_roughness_tex) {
		float4 tex = tex2D<float4>(payload.roughness_tex, uv.x, uv.y);
		roughness = tex.y;
	}
    if (roughness > 0.0f) {
        float3 random_dir = random_cosine_direction(rng);
        direction = normalize(lerp(direction, random_dir, roughness*0.1f));
    }

    //  Transmission rengi hesapla
    float thickness = 0.1f;

    float3 tint = material.albedo;   
    if (payload.has_albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, uv.x, uv.y);
        tint = (tex.y == 0.0f && tex.z == 0.0f) ?
            make_float3(tex.x, tex.x, tex.x) :
            make_float3(tex.x, tex.y, tex.z);
    }

    float3 transmission_color = make_float3(
        expf(-tint.x * thickness),
        expf(-tint.y * thickness),
        expf(-tint.z * thickness)
    );
    float transmission_weight = (1.0f - schlick(cos_theta, eta)* material.transmission);
    float3 result = tint;

    //  Caustic katkısı
    float3 caustic = calculate_caustic_gpu(unit_direction, outward_normal, direction, tint, 0.1f);
    result += caustic;

    *attenuation = result;
    *scattered = Ray(payload.position + outward_normal * 0.00001f, normalize(direction));
    return true;
}
__device__ float get_alpha_gpu(const OptixHitResult& payload, const float2& uv)
{
    // Öncelikle opacity texture varsa kullan
    if (payload.opacity_tex) {
        float4 tex = tex2D<float4>(payload.opacity_tex, uv.x, uv.y);
        return tex.w > 0.0f ? tex.w : (tex.x + tex.y + tex.z) / 3.0f;
    }

    // Opacity yoksa albedo'yu kontrol et
    if (payload.albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, uv.x, uv.y);
        return tex.w > 0.0f ? tex.w : (tex.x + tex.y + tex.z) / 3.0f;
    }

    // Ne opacity ne de albedo varsa → tamamen opak
    return 1.0f;
}

__device__ bool scatter_material(
    const GpuMaterial& material,         // dışarıdan gelen materyal
    OptixHitResult& payload,        // payload (normal, uv, textures)
    const Ray& ray_in,
    curandState* rng,
    Ray* scattered,
    float3* attenuation,
    float* pdf,
    bool* is_specular
)
{
    float2 uv = payload.uv;
    float3 N = payload.normal;
   
    uv = clamp(uv, 0.0f, 1.0f);
    const float3 V = -normalize(ray_in.direction);

    // BURADA DİKKAT:
    // DOĞRUDAN material albedo, roughness, metallic vs. kullanacağız
    float3 albedo = material.albedo;
    if (payload.has_albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, uv.x, uv.y);
        albedo = (tex.y == 0.0f && tex.z == 0.0f) ?
            make_float3(tex.x, tex.x, tex.x) :
            make_float3(tex.x, tex.y, tex.z);
    }


   /* float opacity = material.opacity;
    opacity *= get_alpha_gpu(payload, uv);

    if (opacity < 1.0f) {
       
            *attenuation = make_float3(1.0f, 1.0f, 1.0f);
            *scattered = Ray(payload.position + 1e-5f * ray_in.direction, ray_in.direction);
            *pdf = 1.0f;
           
            return true;
       
    }*/

    float roughness = material.roughness;
    if (payload.has_roughness_tex)
        roughness = tex2D<float4>(payload.roughness_tex, uv.x, uv.y).y;
    
  
    float metallic = material.metallic;
    if (payload.has_metallic_tex)
        metallic = tex2D<float4>(payload.metallic_tex, uv.x, uv.y).z;

    float transmission = material.transmission;
    if (payload.has_transmission_tex)
        transmission = tex2D<float4>(payload.transmission_tex, uv.x, uv.y).x;

    float3 emission = material.emission;
    if (payload.has_emission_tex) {
        float4 tex = tex2D<float4>(payload.emission_tex, uv.x, uv.y);
        emission = make_float3(tex.x, tex.y, tex.z);
    }
    // ✨ Fresnel-Schlick + GGX ile metalik / difüz karışımı
    float3 H = importance_sample_ggx(random_float(rng), random_float(rng), roughness, N);
    float3 L = normalize(reflect(-V, H));
    // if (dot(L, N) <= 0.0f) return false;

    float cos_theta = fmaxf(dot(V, H), 1e-4f);
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);   

    if (transmission > 0.0f && random_float(rng) < transmission) {
        return transmission_scatter(material, payload, ray_in, rng, scattered, attenuation); // transmission_scatter da material alacak
    }
    float3 ssscolor=make_float3(1,0.5,0.5) ;
    float sss = material.subsurface;
    if (sss > 0.01f && random_float(rng) < sss) {
        float3 sss_dir = sample_sss_direction(N, rng);
        float3 tint = ssscolor;
        if (length(tint) < 0.001f) tint = albedo;

        float thickness = 0.05f + 0.2f * random_float(rng); // yüzeye yakın saçılım
        float3 sss_color = make_float3(
            expf(-tint.x * thickness),
            expf(-tint.y * thickness),
            expf(-tint.z * thickness)
        );

        *scattered = Ray(payload.position - N * 0.0005f, sss_dir);
        *attenuation = tint * sss_color;
        *pdf = 1.0f;
        *is_specular = false;
        return true;
    }


    float3 F = fresnel_schlick_roughness(cos_theta, F0, roughness);
    float NdotL = max(dot(N, L), 0.01f);
    float3 wo = -normalize(ray_in.direction);
    float3 wi = normalize(scattered->direction);
    float lift = lerp(1.0f / M_PIf, 1.0f, material.artistic_albedo_response);
    float3 F_avg = F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) / 21.0f;

    float3 k_d = (make_float3(1.0f, 1.0f, 1.0f) - F_avg) * (1.0f - metallic);

    float3 diffuse = k_d * albedo/M_PIf ;
    float3 specular = F;
    *pdf = pdf_brdf(material, wo, wi, payload.normal);   
    *scattered = Ray(payload.position + L * 0.001f, L);
    *attenuation = (diffuse + specular + emission);
   
    return true;
}
