// Safe and stable BRDF + scatter implementation
#pragma once
#include "vec3_utils.cuh"
#include "random_utils.cuh"
#include "ray.h"
#include "params.h"
#include "payload.h"
#include <material_gpu.h>


// Industry standard minimum alpha value (matches Disney/Cycles/PBRT)
#define GPU_MIN_ALPHA 0.0001f
#define GPU_MIN_DOT 0.0001f

__device__ float G_SchlickGGX(float NdotV, float roughness) {
    // Industry standard for Direct Lighting: k = (r+1)^2 / 8
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

__device__ float G_Smith(float NdotV, float NdotL, float roughness) {
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

__device__ float pdf_brdf(const GpuMaterial& mat, const float3& wo, const float3& wi, const float3& N) {
    float3 H = normalize(wo + wi);
    float NdotH = max(dot(N, H), GPU_MIN_DOT);
    float VdotH = max(dot(wo, H), GPU_MIN_DOT);

    float roughness = mat.roughness;
    float alpha = max(roughness * roughness, GPU_MIN_ALPHA);
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
__device__ float get_alpha_gpu(const OptixHitResult& payload, const float2& uv)
{
    // 1. Explicit Opacity Map (Priority)
    // Assume standard Grayscale map (White=Opaque, Black=Transparent).
    // We use Red channel (tex.x) because texture loaders usually put grayscale into RGB.
    // We do NOT check tex.w (Alpha) here because many opacity maps are JPG/PNG with full Alpha (1.0).
    if (payload.opacity_tex) {
        float4 tex = tex2D<float4>(payload.opacity_tex, uv.x, uv.y);
        // Use average of RGB for robustness against colored masks, or just X. 
        // Let's use average to be safe if it's not pure grayscale.
        return (tex.x + tex.y + tex.z) / 3.0f; 
    }

    // 2. Albedo Map Alpha Channel (Fallback)
    // If no explicit opacity map, check if Albedo has alpha transparency.
    if (payload.albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, uv.x, uv.y);
        // Trust the Alpha channel. 
        // If loaded as RGB (no alpha), texture loader usually sets W=1.0.
        // If loaded as RGBA, W is the alpha.
        return tex.w; 
    }

    // 3. Default (Opaque)
    return 1.0f;
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

    float NdotL = max(dot(N, wi), GPU_MIN_DOT);
    float NdotV = max(dot(N, wo), GPU_MIN_DOT);
    if (NdotL < GPU_MIN_DOT || NdotV < GPU_MIN_DOT) return make_float3(0.0f, 0.0f, 0.0f);

    float3 H = normalize(wi + wo);
    float NdotH = max(dot(N, H), GPU_MIN_DOT);
    float VdotH = max(dot(wo, H), GPU_MIN_DOT);
    if (NdotH < GPU_MIN_DOT || VdotH < GPU_MIN_DOT) return make_float3(0.0f, 0.0f, 0.0f);
    float3 albedo = material.albedo;
    if (payload.use_blended_data) {
        albedo = payload.blended_albedo;
    }
    else if (payload.has_albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, uv.x, uv.y);
        albedo = (tex.y == 0.0f && tex.z == 0.0f) ?
            make_float3(tex.x, tex.x, tex.x) :
            make_float3(tex.x, tex.y, tex.z);
    }
	// albedo = clamp(albedo, 0.01f, 1.0f); // REMOVED: Clamping prevents true black and reduces contrast

   

    float roughness = material.roughness;
    if (payload.use_blended_data) {
        roughness = payload.blended_roughness;
    }
    else if (payload.has_roughness_tex) {
        roughness = tex2D<float4>(payload.roughness_tex, uv.x, uv.y).y;
    }
   /* float opacity = material.opacity;
    opacity *= get_alpha_gpu(payload, uv);

    if (opacity < 1.0f) {

       
        return make_float3(0.0f, 0.0f, 0.0f);;

    }*/
    float metallic = material.metallic;
    if (payload.has_metallic_tex) {
        metallic = tex2D<float4>(payload.metallic_tex, uv.x, uv.y).z;
       
    }
  

    float alpha = max(roughness * roughness, GPU_MIN_ALPHA);
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
    float alpha = (roughness * roughness);
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

// ═══════════════════════════════════════════════════════════════════════════════
// RANDOM WALK SUBSURFACE SCATTERING
// ═══════════════════════════════════════════════════════════════════════════════

// Henyey-Greenstein phase function for anisotropic scattering
__device__ float henyey_greenstein_sample(float g, float rand) {
    if (fabsf(g) < 0.001f) {
        return 1.0f - 2.0f * rand;  // Isotropic
    }
    float sqr_term = (1.0f - g * g) / (1.0f - g + 2.0f * g * rand);
    return (1.0f + g * g - sqr_term * sqr_term) / (2.0f * g);
}

// Sample direction using Henyey-Greenstein phase function
__device__ float3 sample_hg_direction(const float3& forward, curandState* rng, float g) {
    float cos_theta = henyey_greenstein_sample(g, random_float(rng));
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * M_PIf * random_float(rng);
    
    // Create local coordinate system
    float3 up = fabsf(forward.z) < 0.999f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
    float3 T = normalize(cross(up, forward));
    float3 B = cross(forward, T);
    
    // Transform to world space
    float3 local_dir = make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);
    return normalize(T * local_dir.x + B * local_dir.y + forward * local_dir.z);
}

// Compute extinction coefficient from subsurface radius
__device__ float3 compute_sss_sigma_t(const float3& sss_color, const float3& sss_radius, float scale) {
    // sigma_t = 1 / (radius * scale) - higher radius means farther scatter
    float3 scaled_radius = sss_radius * scale;
    return make_float3(
        scaled_radius.x > 0.0001f ? 1.0f / scaled_radius.x : 10000.0f,
        scaled_radius.y > 0.0001f ? 1.0f / scaled_radius.y : 10000.0f,
        scaled_radius.z > 0.0001f ? 1.0f / scaled_radius.z : 10000.0f
    );
}

// Random Walk SSS
__device__ bool sss_random_walk_scatter(
    const GpuMaterial& material,
    const OptixHitResult& payload,
    const Ray& ray_in,
    curandState* rng,
    Ray* scattered,
    float3* attenuation
) {
    float3 N = normalize(payload.normal);
    float3 V = -normalize(ray_in.direction);
    
    // Get SSS parameters
    float3 sss_color = material.subsurface_color;
    float3 sss_radius = material.subsurface_radius;
    float sss_scale = fmaxf(material.subsurface_scale, 0.001f);
    float sss_anisotropy = material.subsurface_anisotropy;
    
    // Compute extinction coefficients per RGB channel
    float3 sigma_t = compute_sss_sigma_t(sss_color, sss_radius, sss_scale);
    
    // Sample scatter distance using exponential distribution (mean free path)
    float rand_channel = random_float(rng);
    float sigma_sample;
    if (rand_channel < 0.333f) {
        sigma_sample = sigma_t.x;
    } else if (rand_channel < 0.666f) {
        sigma_sample = sigma_t.y;
    } else {
        sigma_sample = sigma_t.z;
    }
    
    // Exponential distance sampling
    float scatter_dist = -logf(fmaxf(random_float(rng), 0.0001f)) / sigma_sample;
    scatter_dist = fminf(scatter_dist, sss_scale * 10.0f);  // Clamp max distance
    
    // Sample scatter direction using Henyey-Greenstein
    float3 scatter_dir = sample_hg_direction(-N, rng, sss_anisotropy);
    
    // Compute Beer-Lambert attenuation per channel
    float3 absorption = make_float3(
        expf(-sigma_t.x * scatter_dist),
        expf(-sigma_t.y * scatter_dist),
        expf(-sigma_t.z * scatter_dist)
    );
    
    // Final attenuation: SSS color tinted by absorption
    *attenuation = sss_color * absorption;
    
    // Scatter from slightly inside the surface
    float3 scatter_origin = payload.position - N * 0.001f;
    *scattered = Ray(scatter_origin, scatter_dir);
    
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLEAR COAT SCATTER (Lacquer layer like car paint)
// ═══════════════════════════════════════════════════════════════════════════════
__device__ bool clearcoat_scatter(
    const GpuMaterial& material,
    const OptixHitResult& payload,
    const Ray& ray_in,
    curandState* rng,
    Ray* scattered,
    float3* attenuation,
    float* pdf
) {
    float3 N = normalize(payload.normal);
    float3 V = -normalize(ray_in.direction);
    
    // Clear coat uses fixed IOR ~1.5 (like lacquer/varnish)
    const float clearcoat_ior = 1.5f;
    float cc_f0 = ((clearcoat_ior - 1.0f) / (clearcoat_ior + 1.0f));
    cc_f0 *= cc_f0;  // ≈ 0.04
    
    // Sample microfacet normal using GGX
    float cc_roughness = fmaxf(material.clearcoat_roughness, 0.001f);
    float3 H = importance_sample_ggx(random_float(rng), random_float(rng), cc_roughness, N);
    float3 L = reflect(-V, H);
    
    float NdotL = dot(N, L);
    if (NdotL <= 0.0f) return false;
    
    // Fresnel-Schlick for clear coat
    float VdotH = fmaxf(dot(V, H), 0.001f);
    float fresnel = cc_f0 + (1.0f - cc_f0) * powf(1.0f - VdotH, 5.0f);
    
    // GGX distribution and geometry terms
    float NdotH = fmaxf(dot(N, H), 0.001f);
    float NdotV = fmaxf(dot(N, V), 0.001f);
    
    float alpha = cc_roughness * cc_roughness;
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0f) + 1.0f;
    float D = alpha2 / (M_PIf * denom * denom);
    float G = G_Smith(NdotV, NdotL, cc_roughness);
    
    // Clear coat BRDF (white specular reflection)
    float spec = fresnel * D * G / (4.0f * NdotV * NdotL + 0.001f);
    
    *attenuation = make_float3(spec, spec, spec) * material.clearcoat;
    *scattered = Ray(payload.position + N * 0.001f, L);
    *pdf = D * NdotH / (4.0f * VdotH + 0.001f);
    
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSLUCENT SCATTER (Thin surface light pass-through: leaves, paper, fabric)
// ═══════════════════════════════════════════════════════════════════════════════
__device__ bool translucent_scatter(
    const GpuMaterial& material,
    const OptixHitResult& payload,
    const Ray& ray_in,
    curandState* rng,
    Ray* scattered,
    float3* attenuation
) {
    float3 N = normalize(payload.normal);
    
    // Get albedo for tinting
    float3 albedo = material.albedo;
    if (payload.use_blended_data) {
        albedo = payload.blended_albedo;
    } else if (payload.has_albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, payload.uv.x, payload.uv.y);
        albedo = make_float3(tex.x, tex.y, tex.z);
    }
    
    // Diffuse transmission through the surface
    // Sample cosine-weighted direction in opposite hemisphere
    float3 trans_dir = random_cosine_direction(rng);
    
    // Transform to world space, pointing through the surface
    float3 up = fabsf(N.z) < 0.999f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
    float3 T = normalize(cross(up, N));
    float3 B = cross(N, T);
    float3 world_dir = normalize(T * trans_dir.x + B * trans_dir.y - N * trans_dir.z);
    
    // Tinted by albedo with some absorption
    *attenuation = albedo * 0.8f;  // Slight absorption
    *scattered = Ray(payload.position - N * 0.001f, world_dir);
    
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSTIC CALCULATION (for transmission effects)
// ═══════════════════════════════════════════════════════════════════════════════
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
    if (payload.use_blended_data) {
        roughness = payload.blended_roughness;
    }
    else if (payload.has_roughness_tex) {
		float4 tex = tex2D<float4>(payload.roughness_tex, uv.x, uv.y);
		roughness = tex.y;
	}
    if (roughness > 0.0f) {
        float3 random_dir = random_cosine_direction(rng);
        direction = normalize(lerp(direction, random_dir, roughness*0.1f));
    }

    //  Transmission rengi hesapla - Beer's Law
    float thickness = 0.1f;

    float3 tint = material.albedo;   
    if (payload.use_blended_data) {
        tint = payload.blended_albedo;
    }
    else if (payload.has_albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, uv.x, uv.y);
        tint = (tex.y == 0.0f && tex.z == 0.0f) ?
            make_float3(tex.x, tex.x, tex.x) :
            make_float3(tex.x, tex.y, tex.z);
    }

    // Beer's Law: absorbans için ters renk kullan (açık renkler daha az soğurur)
    float3 absorption = make_float3(
        (1.0f - tint.x) * thickness,
        (1.0f - tint.y) * thickness,
        (1.0f - tint.z) * thickness
    );
    float3 transmission_color = make_float3(
        expf(-absorption.x),
        expf(-absorption.y),
        expf(-absorption.z)
    );
    
    // Fresnel katkısını ekle - yüksek transmission'da daha fazla geçiş
    float fresnel = schlick(cos_theta, eta);
    float transmission_factor = 1.0f - fresnel * (1.0f - material.transmission);
    
    // Şeffaf camlar için tint yerine transmission_color kullan
    // Transmission = 1.0 ise neredeyse tam ışık geçisi (hafif tint ile)
    float3 result = lerp(tint, transmission_color, material.transmission) * transmission_factor;
    
    // Çok şeffaf materyaller için minimum attenuation garantisi
    if (material.transmission > 0.9f) {
        result = make_float3(
            fmaxf(result.x, 0.9f),
            fmaxf(result.y, 0.9f),
            fmaxf(result.z, 0.9f)
        );
    }

    //  Caustic katkısı
    float3 caustic = calculate_caustic_gpu(unit_direction, outward_normal, direction, tint, 0.1f);
    result += caustic * 0.05f;  // Caustic etkisini azalt

    *attenuation = result;
    *scattered = Ray(payload.position + outward_normal * 0.00001f, normalize(direction));
    return true;
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
	if (length(N) < 0.001f) {
		N = make_float3(0.0f, 0.0f, 1.0f); // Normal sıfırsa Z eksenine yönlendir
	}
   
    float3 albedo = material.albedo/M_PIf;
    if (payload.use_blended_data) {
        albedo = payload.blended_albedo / M_PIf; // Maintain division convention if needed, though check logic
        // Wait, blended_albedo comes from raygen which already did sRGB->Linear. 
        // material.albedo/M_PIf suggests albedo is used as diffuse reflectance rho.
        // Usually albedo is just color. Dividing by PI is strictly for Lambertian BRDF term.
        // Let's keep consistency.
    }
    else if (payload.has_albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, uv.x, uv.y);
        albedo = (tex.y == 0.0f && tex.z == 0.0f) ?
            make_float3(tex.x, tex.x, tex.x) :
            make_float3(tex.x, tex.y, tex.z);
    }
   
    float opacity = material.opacity;
    opacity *= get_alpha_gpu(payload, uv);
    
    // Tam şeffaf (opacity ≈ 0) durumunda direkt geç, hayalet görüntü bırakma
    if (opacity < 0.001f) {
        // Tamamen şeffaf - ışın hiç etkileşime girmeden geçer
        *attenuation = make_float3(1.0f, 1.0f, 1.0f);  // Hiç soğurma yok
        *scattered = Ray(payload.position + ray_in.direction * 0.001f, ray_in.direction);
        *pdf = 1.0f;
        *is_specular = true;  // Delta dağılım - MIS atla
        return true;
    }
    
    if (opacity < 1.0f) {
        // Yarı-şeffaf: stokastik olarak opak veya şeffaf yol seç
        if (random_float(rng) < opacity) {
            // Opak yol seçildi - normal BRDF değerlendirmesi devam edecek
            // Attenuation'ı opacity ile ÇARPMA - sadece BRDF sonucu kullanılacak
        }
        else {
            // Şeffaf yol seçildi - ışın geçer
            *attenuation = make_float3(1.0f, 1.0f, 1.0f);  // Tam geçiş
            *scattered = Ray(payload.position + ray_in.direction * 0.001f, ray_in.direction);
            *pdf = 1.0f;
            *is_specular = true;  // Delta dağılım
            return true;
        }
    }
   
   
    float roughness = material.roughness;
    if (payload.use_blended_data) {
        roughness = payload.blended_roughness;
    }
    else if (payload.has_roughness_tex)
        roughness = tex2D<float4>(payload.roughness_tex, uv.x, uv.y).y;
    
  
    float metallic = material.metallic;
    if (payload.has_metallic_tex)
        metallic = tex2D<float4>(payload.metallic_tex, uv.x, uv.y).z;

    float transmission = material.transmission;
    if (payload.has_transmission_tex)
        transmission = tex2D<float4>(payload.transmission_tex, uv.x, uv.y).x;

   
   
    //  Fresnel-Schlick + GGX ile metalik / difüz karışımı
    float3 H = importance_sample_ggx(random_float(rng), random_float(rng), roughness, N);
    float3 L = normalize(reflect(-V, H));

    float cos_theta = fmaxf(dot(V, H), 1e-4f);
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);   
    float3 F = fresnel_schlick_roughness(cos_theta, F0, roughness);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LOBE SELECTION (Energy-conserving order)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // 1. CLEAR COAT (Top layer - evaluated first)
    float clearcoat = material.clearcoat;
    if (clearcoat > 0.01f) {
        // Fresnel for clear coat decides reflection probability
        float cc_ior = 1.5f;
        float cc_f0 = ((cc_ior - 1.0f) / (cc_ior + 1.0f));
        cc_f0 *= cc_f0;
        float cc_fresnel = cc_f0 + (1.0f - cc_f0) * powf(1.0f - fmaxf(dot(V, N), 0.0f), 5.0f);
        float cc_prob = clearcoat * cc_fresnel;
        
        if (random_float(rng) < cc_prob) {
            *is_specular = true;
            return clearcoat_scatter(material, payload, ray_in, rng, scattered, attenuation, pdf);
        }
    }
    
    // 2. TRANSMISSION (Glass/Water)
    if (transmission > 0.01f && random_float(rng) < transmission) {
        *is_specular = true;
        return transmission_scatter(material, payload, ray_in, rng, scattered, attenuation);
    }
    
    // 3. SUBSURFACE SCATTERING (Random Walk)
    float sss = material.subsurface;
    if (sss > 0.01f && random_float(rng) < sss) {
        *is_specular = false;
        *pdf = 1.0f;
        return sss_random_walk_scatter(material, payload, ray_in, rng, scattered, attenuation);
    }
    
    // 4. TRANSLUCENT (Thin surface transmission: leaves, paper, fabric)
    float translucent = material.translucent;
    if (translucent > 0.01f && random_float(rng) < translucent) {
        *is_specular = false;
        *pdf = 1.0f / M_PIf;  // Cosine-weighted PDF
        return translucent_scatter(material, payload, ray_in, rng, scattered, attenuation);
    }
    
    // 5. STANDARD DIFFUSE + SPECULAR (Base layer)
    float NdotL = max(dot(N, L), 0.01f);
    float3 wo = -normalize(ray_in.direction);
    float3 wi = L;
 
    float3 F_avg = F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) / 21.0f;
    float3 k_d = (make_float3(1.0f, 1.0f, 1.0f) - F_avg) * (1.0f - metallic);

    float3 diffuse = k_d * albedo / M_PIf;
    float3 specular = F;
    *pdf = pdf_brdf(material, wo, wi, payload.normal);   
    *scattered = Ray(payload.position + L * 0.001f, L);
    *attenuation = (diffuse + specular);
    *is_specular = (roughness < 0.05f && metallic > 0.5f);  // Mirror-like surfaces

    return true;
}

