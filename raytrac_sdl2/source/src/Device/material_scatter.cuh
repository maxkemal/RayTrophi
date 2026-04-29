// Safe and stable BRDF + scatter implementation
#pragma once
#include "params.h"
#include "vec3_utils.cuh"
#include "random_utils.cuh"
#include "ray.h"
#include "payload.h"
#include <material_gpu.h>
#include "procedural_detail.cuh"


// Industry standard minimum alpha value (matches Disney/Cycles/PBRT)
#define GPU_MIN_ALPHA 0.0001f
#define GPU_MIN_DOT 0.0001f

__device__ __forceinline__ bool finite3(const float3& v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

__device__ __forceinline__ float3 clamp3f(const float3& v, float lo, float hi) {
    return make_float3(
        fminf(fmaxf(v.x, lo), hi),
        fminf(fmaxf(v.y, lo), hi),
        fminf(fmaxf(v.z, lo), hi)
    );
}

__device__ __forceinline__ float wrap_repeat_gpu(float x) {
    float r = fmodf(x, 1.0f);
    return (r < 0.0f) ? (r + 1.0f) : r;
}

__device__ __forceinline__ float wrap_mirror_gpu(float x) {
    float r = fmodf(x, 2.0f);
    if (r < 0.0f) r += 2.0f;
    return (r > 1.0f) ? (2.0f - r) : r;
}

__device__ __forceinline__ float wrap_clamp_gpu(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}

__device__ __forceinline__ float2 apply_material_uv_transform(const GpuMaterial& material, const float2& originalUv) {
    float u = originalUv.x - 0.5f;
    float v = originalUv.y - 0.5f;
    u *= material.uv_scale_x;
    v *= material.uv_scale_y;

    const float radians = material.uv_rotation_degrees * (M_PIf / 180.0f);
    const float cosTheta = cosf(radians);
    const float sinTheta = sinf(radians);
    const float rotatedU = u * cosTheta - v * sinTheta;
    const float rotatedV = u * sinTheta + v * cosTheta;

    float transformedU = (rotatedU + 0.5f + material.uv_offset_x) * material.uv_tiling_x;
    float transformedV = (rotatedV + 0.5f + material.uv_offset_y) * material.uv_tiling_y;

    switch (material.uv_wrap_mode) {
    case 0:
        transformedU = wrap_repeat_gpu(transformedU);
        transformedV = wrap_repeat_gpu(transformedV);
        break;
    case 1:
        transformedU = wrap_mirror_gpu(transformedU);
        transformedV = wrap_mirror_gpu(transformedV);
        break;
    case 2:
        transformedU = wrap_clamp_gpu(transformedU);
        transformedV = wrap_clamp_gpu(transformedV);
        break;
    case 3:
        transformedU = originalUv.x;
        transformedV = originalUv.y;
        break;
    case 4: {
        const float uScaled = transformedU * 3.0f;
        const float vScaled = transformedV * 3.0f;
        const int face = static_cast<int>(uScaled) + 3 * static_cast<int>(vScaled);
        float uLocal = fmodf(uScaled, 1.0f);
        float vLocal = fmodf(vScaled, 1.0f);
        if (uLocal < 0.0f) uLocal += 1.0f;
        if (vLocal < 0.0f) vLocal += 1.0f;
        switch (face % 6) {
        case 0: transformedU = uLocal; transformedV = vLocal; break;
        case 1: transformedU = vLocal; transformedV = 1.0f - uLocal; break;
        case 2: transformedU = 1.0f - uLocal; transformedV = vLocal; break;
        case 3: transformedU = 1.0f - vLocal; transformedV = 1.0f - uLocal; break;
        case 4: transformedU = uLocal; transformedV = 1.0f - vLocal; break;
        default: transformedU = vLocal; transformedV = uLocal; break;
        }
        break;
    }
    default:
        break;
    }

    return make_float2(transformedU, transformedV);
}

__device__ float G_SchlickGGX(float NdotV, float roughness) {
    // Match Vulkan/CPU direct-lighting parity: k = (roughness + 1)^2 / 8
    float r = roughness + 1.0f;
    float k = (r * r) / 8.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

__device__ float G_Smith(float NdotV, float NdotL, float roughness) {
    return G_SchlickGGX(NdotV, roughness) * G_SchlickGGX(NdotL, roughness);
}

__device__ float3 fresnel_schlick_roughness(float cosTheta, float3 F0, float roughness);

__device__ float pdf_brdf(const GpuMaterial& mat, const float3& wo, const float3& wi, const float3& N, int bounce_index = 0) {
    float3 sum = wo + wi;
    float sum_len2 = dot(sum, sum);
    if (sum_len2 < 1e-12f) return 1e-6f;
    float3 H = normalize(sum);
    float NdotH = max(dot(N, H), GPU_MIN_DOT);
    float VdotH = max(dot(wo, H), GPU_MIN_DOT);

    float roughness = fminf(fmaxf(mat.roughness, 0.02f), 1.0f);
    // Must mirror the regularization applied in evaluate_brdf / scatter_material
    // so MIS weights stay consistent with the actually-sampled BRDF.
    if (bounce_index > 0) roughness = fmaxf(roughness, 0.1f);
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
__device__ float get_alpha_gpu(const GpuMaterial& material, const OptixHitResult& payload, const float2& uv)
{
    // 1. Try Bindless Opacity from Material (Fast Path - No SBT sync needed)
    if (material.opacity_tex) {
        float4 tex = tex2D<float4>(material.opacity_tex, uv.x, uv.y);
        // Fallback: Use Alpha if available, otherwise Red
        float val = tex.w; // Default to W (Alpha)
        // If it's a dedicated opacity map (grayscale), it might be in R
        if (val < 0.001f) val = tex.x; 

        return (val < 0.1f) ? 0.0f : val;
    }

    // 2. Legacy SBT Path (Fallback)
    if (payload.opacity_tex) {
        float4 tex = tex2D<float4>(payload.opacity_tex, uv.x, uv.y);
        float val = (payload.opacity_has_alpha) ? tex.w : tex.x;
        return (val < 0.1f) ? 0.0f : val;
    }
    
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
    float2 uv = apply_material_uv_transform(material, payload.uv);

    // Procedural tile-break: independent slider — set to 0 to keep albedo maps clean.
    if (material.tile_break_strength > 0.0f &&
        (material.albedo_tex || material.roughness_tex || material.normal_tex)) {
        uv = pd_tileBreak(uv, payload.position, material.tile_break_strength);
    }

    float NdotL = max(dot(N, wi), GPU_MIN_DOT);
    float NdotV = max(dot(N, wo), GPU_MIN_DOT);
    if (NdotL < GPU_MIN_DOT || NdotV < GPU_MIN_DOT) return make_float3(0.0f, 0.0f, 0.0f);

    float3 sum = wi + wo;
    float sum_len2 = dot(sum, sum);
    if (sum_len2 < 1e-12f) return make_float3(0.0f, 0.0f, 0.0f);
    float3 H = normalize(sum);
    float NdotH = max(dot(N, H), GPU_MIN_DOT);
    float VdotH = max(dot(wo, H), GPU_MIN_DOT);
    if (NdotH < GPU_MIN_DOT || VdotH < GPU_MIN_DOT) return make_float3(0.0f, 0.0f, 0.0f);
    float3 albedo = material.albedo;
    if (payload.use_blended_data) {
        albedo = payload.blended_albedo;
    }
    else if (material.albedo_tex) {
        // Fast Bindless Path
        float4 tex = tex2D<float4>(material.albedo_tex, uv.x, uv.y);
        albedo = make_float3(tex.x, tex.y, tex.z);
    }
    else if (payload.has_albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, uv.x, uv.y);
        albedo = (fabsf(tex.x - tex.y) < 1e-3f && fabsf(tex.x - tex.z) < 1e-3f) ?
            make_float3(tex.x, tex.x, tex.x) :
            make_float3(tex.x, tex.y, tex.z);
    }
	// albedo = clamp(albedo, 0.01f, 1.0f); // REMOVED: Clamping prevents true black and reduces contrast

   

    float roughness = material.roughness;
    if (payload.use_blended_data) {
        roughness = payload.blended_roughness;
    }
    else if (material.roughness_tex) {
        // Fast Bindless Path
        roughness = tex2D<float4>(material.roughness_tex, uv.x, uv.y).y; // Green channel (Standard ORM)
    }
    else if (payload.has_roughness_tex) {
        roughness = tex2D<float4>(payload.roughness_tex, uv.x, uv.y).y;
    }
   /* float opacity = material.opacity;
    opacity *= get_alpha_gpu(payload, uv);

    if (opacity < 1.0f) {

       
        return make_float3(0.0f, 0.0f, 0.0f);;

    }*/
    roughness = fminf(fmaxf(roughness, 0.02f), 1.0f);

    // Path regularization (Müller 2018): on indirect bounces, clamp roughness
    // to a floor so secondary GGX lobes cannot produce near-mirror D-peaks.
    // Primary bounce (bounce_index == 0) is untouched — sharp direct
    // reflections remain correct. Kills metallic fireflies at the source
    // instead of masking them with per-path radiance clamps.
    if (payload.bounce_index > 0) {
        roughness = fmaxf(roughness, 0.1f);
    }

    float metallic = material.metallic;
    if (payload.use_blended_data) {
        metallic = payload.blended_metallic;
    }
    else if (material.metallic_tex) {
        // Fast Bindless Path
        metallic = tex2D<float4>(material.metallic_tex, uv.x, uv.y).z; // Blue channel (Standard ORM)
    }
    else if (payload.has_metallic_tex) {
        // Legacy SBT Fallback
        metallic = tex2D<float4>(payload.metallic_tex, uv.x, uv.y).z;
    }

    // ── Procedural detail: subtle color variation + dirt + roughness ──────────
    // micro_detail_strength drives all world-space effects without touching UVs.
    // tile_break_strength (above) is the separate UV-warp control.
    if (material.micro_detail_strength > 0.0f) {
        float sc  = fmaxf(material.micro_detail_scale, 0.5f);
        float str = material.micro_detail_strength;

        // Subtle world-space luminance variation — ±8% max, independent seed
        float3 colorSeed = make_float3(
            payload.position.x * sc * 0.7f + 31.4f,
            payload.position.y * sc * 0.7f + 17.2f,
            payload.position.z * sc * 0.7f + 42.9f);
        float colorVar   = pd_vnoise3(colorSeed);
        float colorDelta = (colorVar - 0.5f) * 0.16f * str;
        float cm = 1.0f + colorDelta;
        albedo = make_float3(
            fmaxf(0.0f, fminf(1.0f, albedo.x * cm)),
            fmaxf(0.0f, fminf(1.0f, albedo.y * cm)),
            fmaxf(0.0f, fminf(1.0f, albedo.z * cm)));

        // Dirt: fBm darkening (dust, grime, worn patches)
        float dirtFactor = pd_dirt(payload.position, sc) * str;
        float3 dirtColor = make_float3(0.14f, 0.10f, 0.08f);
        albedo = make_float3(
            albedo.x + (albedo.x * dirtColor.x - albedo.x) * dirtFactor,
            albedo.y + (albedo.y * dirtColor.y - albedo.y) * dirtFactor,
            albedo.z + (albedo.z * dirtColor.z - albedo.z) * dirtFactor);

        roughness = fmaxf(0.02f, fminf(1.0f,
            roughness + pd_roughnessVar(payload.position, sc) * str * 0.5f));
    }

    float alpha = max(roughness * roughness, GPU_MIN_ALPHA);
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0f) + 1.0f;
    float D = alpha2 / (M_PIf * denom * denom);

    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);
    float3 F = fresnel_schlick_roughness(VdotH, F0, roughness);
    float3 F_avg = F0 + (make_float3(1.0f) - F0) / 21.0f;

    float G = G_Smith(NdotV, NdotL, roughness);
    float3 spec = (F * D * G) / (4.0f * NdotV * NdotL + 0.001f);   
    
    // Diffuse weight (Energy Conservation): Use dynamic (1-F) like Vulkan
    float3 k_d = (make_float3(1.0f) - F_avg) * (1.0f - metallic);
    float3 diffuse = k_d * albedo / M_PIf;
  
   
    float transmission = material.transmission;
    if (payload.use_blended_data) {
        transmission = payload.blended_transmission;
    }
    // Bindless texture support for transmission can be added here if needed

    // Deterministic BRDF evaluation (no random branch here):
    // high transmission reduces diffuse lobe while keeping specular response.
    float surfaceWeight = 1.0f - fminf(fmaxf(transmission, 0.0f), 1.0f);
    float3 brdf = diffuse * surfaceWeight + spec;
    if (!finite3(brdf)) return make_float3(0.0f, 0.0f, 0.0f);
    return clamp3f(brdf, 0.0f, 1e4f);
}
__device__ float3 fresnel_schlick_roughness(float cosTheta, float3 F0, float roughness)
{
    // Bu versiyon daha yumuşak geçiş sağlar, Cycles benzeri
    return F0 + (max(make_float3(1.0f - roughness, 1.0f - roughness, 1.0f - roughness), F0) - F0) * powf(1.0f - cosTheta, 5.0f);
}

__device__ float3 importance_sample_ggx(float u1, float u2, float roughness, const float3& N) {
    float safeRoughness = fminf(fmaxf(roughness, 0.02f), 1.0f);
    float alpha = (safeRoughness * safeRoughness);
    float phi = 2.0f * M_PIf * u1;
    float cosTheta = sqrtf((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sinTheta = sqrtf(max(1.0f - cosTheta * cosTheta, 0.0f));

    float3 H = make_float3(cosf(phi) * sinTheta, sinf(phi) * sinTheta, cosTheta);

    float3 up = fabsf(N.z) < 0.999f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
    float3 tangentX = normalize(cross(up, N));
    float3 tangentY = cross(N, tangentX);

    return normalize(tangentX * H.x + tangentY * H.y + N * H.z);
}

__device__ float3 ggx_glass_effective_normal(float u1, float u2, float roughness, const float3& N, const float3& V) {
    float safeRoughness = fminf(fmaxf(roughness, 0.02f), 1.0f);
    float alpha = safeRoughness * safeRoughness;
    float phi = 2.0f * M_PIf * u1;
    float cosTheta = sqrtf((1.0f - u2) / fmaxf(1.0f + (alpha * alpha - 1.0f) * u2, 1e-7f));
    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

    float3 halfVecLocal = make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);

    float3 up = fabsf(N.z) < 0.999f ? make_float3(0, 0, 1) : make_float3(1, 0, 0);
    float3 tangentX = normalize(cross(up, N));
    float3 tangentY = cross(N, tangentX);
    float3 halfVec = normalize(tangentX * halfVecLocal.x + tangentY * halfVecLocal.y + N * halfVecLocal.z);

    return normalize(reflect(-V, halfVec));
}
__device__ float schlick(float cos_theta, float eta) {
    float r0 = (1.0f - eta) / (1.0f + eta);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);
}

// GGX VNDF sampling (Heitz 2018) — mirrors closesthit.rchit ggxSampleVNDF exactly
__device__ float3 ggxSampleVNDF(const float3& N, const float3& V, float alpha, float r1, float r2) {
    // 1. Build ONB from surface normal — Duff et al. 2017, matches Vulkan buildONB
    float sign_ = (N.z >= 0.0f) ? 1.0f : -1.0f;
    float a     = -1.0f / (sign_ + N.z);
    float b     = N.x * N.y * a;
    float3 tangent   = make_float3(1.0f + sign_ * N.x * N.x * a, sign_ * b, -sign_ * N.x);
    float3 bitangent = make_float3(b, sign_ + N.y * N.y * a, -N.y);

    // 2. Transform V to tangent space
    float3 Ve = make_float3(dot(V, tangent), dot(V, bitangent), dot(V, N));

    // 3. Scale by alpha → stretched hemisphere
    float3 Vh = normalize(make_float3(alpha * Ve.x, alpha * Ve.y, Ve.z));

    // 4. ONB perpendicular to Vh
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1;
    if (lensq > 1e-7f) {
        T1 = make_float3(-Vh.y, Vh.x, 0.0f) * (1.0f / sqrtf(lensq));
    } else {
        T1 = make_float3(1.0f, 0.0f, 0.0f);
    }
    float3 T2 = cross(Vh, T1);

    // 5. Sample unit disk
    float r = sqrtf(r1);
    float phi = 2.0f * M_PIf * r2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);

    // 6. Hemisphere reprojection (Heitz 2018 — critical for correct roughness spread)
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;

    // 7. Microfacet normal in local space
    float3 Nh = T1 * t1 + T2 * t2 + Vh * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));

    // 8. Unstretch and transform back to world space
    float3 Ne = normalize(make_float3(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
    float3 H = normalize(tangent * Ne.x + bitangent * Ne.y + N * Ne.z);

    // 9. Reflect view around half-vector
    return normalize(reflect(-V, H));
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
    float2 uv = apply_material_uv_transform(material, payload.uv);
    float3 V = -normalize(ray_in.direction);
    
    // Get SSS parameters
    float3 sss_color = material.subsurface_color;
    if (payload.use_blended_data) sss_color = payload.blended_subsurface_color;

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
    // If GPU material requests single-step behavior, return single scatter
    if (material.sss_use_random_walk == 0) {
        float3 out_pos = payload.position - N * 0.001f + scatter_dir * scatter_dist;
        *scattered = Ray(offset_ray(out_pos, N), normalize(scatter_dir));
        *attenuation = sss_color * absorption;
        return true;
    }
    
    // Multi-scatter random walk
    int maxSteps = material.sss_max_steps;
    if (maxSteps <= 0) maxSteps = 1;
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 pos = payload.position - N * 0.001f; // start slightly inside
    float3 dir = scatter_dir;

    for (int step = 0; step < maxSteps; ++step) {
        // Choose a channel to sample free-path (same strategy as single-step impl)
        float rand_channel = random_float(rng);
        float sigma_sample;
        if (rand_channel < 0.333f) sigma_sample = sigma_t.x;
        else if (rand_channel < 0.666f) sigma_sample = sigma_t.y;
        else sigma_sample = sigma_t.z;

        float dist = -logf(fmaxf(random_float(rng), 1e-6f)) / max(sigma_sample, 1e-6f);
        // Move inside
        pos = pos + dir * dist;

        // Absorption per channel
        float3 absorb = make_float3(
            expf(-sigma_t.x * dist),
            expf(-sigma_t.y * dist),
            expf(-sigma_t.z * dist)
        );
        throughput *= absorb;

        // Russian roulette: terminate internal walk probabilistically to save work
        float survive_prob = fmaxf(fminf((throughput.x + throughput.y + throughput.z) / 3.0f, 0.99f), 0.01f);
        if (random_float(rng) > survive_prob) {
            break;
        }

        // If direction is leaving towards the surface (dot(dir,N) > 0), exit
        if (dot(dir, N) > 0.0f) {
            // outgoing direction — slightly offset to avoid self-intersection
            *scattered = Ray(offset_ray(pos, N), normalize(dir));
            // Final attenuation: tint by subsurface color and accumulated throughput
            *attenuation = sss_color * throughput;
            return true;
        }

        // Otherwise, scatter internally and continue
        dir = sample_hg_direction(dir, rng, sss_anisotropy);
    }

    // Fallback exit: sample an outward cosine direction and return accumulated throughput
    float3 out_dir = random_cosine_direction(rng);
    float3 up = fabsf(N.z) < 0.999f ? make_float3(0,0,1) : make_float3(1,0,0);
    float3 TX = normalize(cross(up, N));
    float3 TY = cross(N, TX);
    float3 world_out = normalize(TX * out_dir.x + TY * out_dir.y + N * out_dir.z);
    *scattered = Ray(offset_ray(pos, N), world_out);
    *attenuation = sss_color * throughput;
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
    const float CC_F0 = 0.04f;

    float cc_roughness = material.clearcoat_roughness;
    if (payload.use_blended_data) cc_roughness = payload.blended_clearcoat_roughness;
    float clearcoat_strength = material.clearcoat;
    if (payload.use_blended_data) clearcoat_strength = payload.blended_clearcoat;
    cc_roughness = fmaxf(cc_roughness, 0.001f);
    float alpha = fmaxf(cc_roughness * cc_roughness, 1e-4f);

    // Sample outgoing direction using GGX VNDF (matches Vulkan GLSL implementation)
    float r1 = random_float(rng);
    float r2 = random_float(rng);
    float3 L = ggxSampleVNDF(N, V, alpha, r1, r2);

    if (dot(N, L) <= 0.0f) {
        // Fallback to perfect reflection
        L = reflect(-V, N);
        if (dot(N, L) <= 0.0f) return false;
    }

    float3 H = normalize(V + L);
    float VdotH = fmaxf(dot(V, H), 0.001f);
    float NdotL = fmaxf(dot(N, L), 1e-4f);
    float NdotV = fmaxf(dot(N, V), 1e-4f);

    // Schlick Fresnel for clearcoat
    float fresnel = CC_F0 + (1.0f - CC_F0) * powf(1.0f - VdotH, 5.0f);

    // Geometry G1 for outgoing direction (Schlick-GGX style)
    float k = alpha * 0.5f;
    float G1L = NdotL / (NdotL * (1.0f - k) + k);

    // Return white specular scaled by Fresnel*G1L (caller will compensate selection probability)
    *attenuation = make_float3(fresnel * G1L, fresnel * G1L, fresnel * G1L) * fmaxf(clearcoat_strength, 0.0f);
    *scattered = Ray(offset_ray(payload.position, N), L);

    // PDF approximation using microfacet half-vector formulation
    float NdotH = fmaxf(dot(N, H), 1e-6f);
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0f) + 1.0f;
    float D = alpha2 / (M_PIf * denom * denom);
    *pdf = D * NdotH / (4.0f * VdotH + 1e-6f);

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
    const float2 local_uv = apply_material_uv_transform(material, payload.uv);
    
    // Get albedo for tinting
    float3 albedo = material.albedo;
    if (payload.use_blended_data) {
        albedo = payload.blended_albedo;
    } else if (payload.has_albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, local_uv.x, local_uv.y);
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
    *scattered = Ray(offset_ray(payload.position, -N), world_dir);
    
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
    float3 unit_direction = normalize(ray_in.direction);
    float2 uv = apply_material_uv_transform(material, payload.uv);

    float ior = material.ior;
    if (payload.use_blended_data) ior = payload.blended_ior;
    bool front_face = payload.front_face != 0;
    float3 macro_normal = N;
    float eta = front_face ? (1.0f / ior) : ior;

    float roughness = material.roughness;
    if (payload.use_blended_data) {
        roughness = payload.blended_roughness;
    }
    else if (material.roughness_tex) {
        roughness = tex2D<float4>(material.roughness_tex, uv.x, uv.y).y;
    }
    else if (payload.has_roughness_tex) {
        float4 tex = tex2D<float4>(payload.roughness_tex, uv.x, uv.y);
        roughness = tex.y;
    }
    roughness = fminf(fmaxf(roughness, 0.0f), 1.0f);

    float3 micro_normal = macro_normal;
    if (roughness >= 0.01f) {
        micro_normal = ggx_glass_effective_normal(
            random_float(rng),
            random_float(rng),
            roughness,
            macro_normal,
            -unit_direction
        );
    }

    float cos_theta = fminf(dot(-unit_direction, macro_normal), 1.0f);
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    bool cannot_refract = (eta * sin_theta > 1.0f);
    float reflect_prob = schlick(cos_theta, ior);
    bool do_reflect = cannot_refract || (random_float(rng) < reflect_prob);

    float3 direction;
    float3 offset_n;
    if (do_reflect) {
        direction = reflect(unit_direction, micro_normal);
        offset_n = macro_normal;
    }
    else {
        bool refracted_success = refract(unit_direction, micro_normal, eta, &direction);
        if (!refracted_success || dot(direction, macro_normal) >= 0.0f) {
            direction = reflect(unit_direction, macro_normal);
            offset_n = macro_normal;
        }
        else {
            offset_n = -macro_normal;
        }
    }

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

    if (do_reflect) {
        *attenuation = make_float3(1.0f, 1.0f, 1.0f);
    }
    else {
        tint = clamp3f(tint, 0.0f, 1.0f);
        float cosInside = fmaxf(fabsf(dot(normalize(direction), -macro_normal)), 0.05f);
        float thickness = 0.65f / cosInside;
        float3 absorption = make_float3(
            (1.0f - tint.x) * thickness,
            (1.0f - tint.y) * thickness,
            (1.0f - tint.z) * thickness
        );
        *attenuation = make_float3(
            expf(-absorption.x),
            expf(-absorption.y),
            expf(-absorption.z)
        );
    }

    *scattered = Ray(offset_ray(payload.position, offset_n), normalize(direction));
    
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
    float2 uv = apply_material_uv_transform(material, payload.uv);
    float3 N = payload.normal;
 
    const float3 V = -normalize(ray_in.direction);
    if (length(N) < 0.001f) {
        N = make_float3(0.0f, 0.0f, 1.0f); // Normal is zero, orient to Z axis
    }
    float opacity = material.opacity;
    opacity *= get_alpha_gpu(material, payload, uv);


    if (opacity < 1.0f) {
        // Yarı-şeffaf: stokastik olarak opak veya şeffaf yol seç
        if (random_float(rng) < opacity) {
            // Opak yol seçildi - normal BRDF değerlendirmesi devam edecek
            // Attenuation'ı opacity ile ÇARPMA - sadece BRDF sonucu kullanılacak
        }
        else {
            // Şeffaf yol seçildi - ışın geçer
            *attenuation = make_float3(1.0f, 1.0f, 1.0f);  // Tam geçiş
            *scattered = Ray(offset_ray(payload.position, ray_in.direction), ray_in.direction);
            *pdf = 1.0f;
            *is_specular = true;  // Delta dağılım
            return true;
        }
    }
    float3 albedo = material.albedo;
    if (payload.use_blended_data) {
        albedo = payload.blended_albedo;
        // Wait, blended_albedo comes from raygen which already did sRGB->Linear. 
        // material.albedo/M_PIf suggests albedo is used as diffuse reflectance rho.
        // Usually albedo is just color. Dividing by PI is strictly for Lambertian BRDF term.
        // Let's keep consistency.
    }
    else if (payload.has_albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, uv.x, uv.y);
        albedo = (fabsf(tex.x - tex.y) < 1e-3f && fabsf(tex.x - tex.z) < 1e-3f) ?
            make_float3(tex.x, tex.x, tex.x) :
            make_float3(tex.x, tex.y, tex.z);
    }

   
    float roughness = material.roughness;
    if (payload.use_blended_data) {
        roughness = payload.blended_roughness;
    }
    else if (payload.has_roughness_tex)
        roughness = tex2D<float4>(payload.roughness_tex, uv.x, uv.y).y;
    
  
    float metallic = material.metallic;
    if (payload.use_blended_data) {
        metallic = payload.blended_metallic;
    }
    else if (payload.has_metallic_tex)
        metallic = tex2D<float4>(payload.metallic_tex, uv.x, uv.y).z;

    float transmission = material.transmission;
    if (payload.use_blended_data) {
        transmission = payload.blended_transmission;
    }
    else if (payload.has_transmission_tex)
        transmission = tex2D<float4>(payload.transmission_tex, uv.x, uv.y).x;

    roughness = fminf(fmaxf(roughness, 0.02f), 1.0f);
    metallic = fminf(fmaxf(metallic, 0.0f), 1.0f);
    transmission = fminf(fmaxf(transmission, 0.0f), 1.0f);

    // Path regularization (Müller 2018): indirect bounces use a roughness
    // floor so secondary GGX sampling cannot hit near-mirror D-peaks and
    // create metallic fireflies. Primary bounce is untouched so direct
    // specular highlights stay sharp. Must match the same clamp used in
    // evaluate_brdf() so scatter and NEE evaluate the same BRDF.
    if (payload.bounce_index > 0) {
        roughness = fmaxf(roughness, 0.1f);
    }

   
   
    float3 F0 = lerp(make_float3(0.04f, 0.04f, 0.04f), albedo, metallic);   
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LOBE SELECTION (Energy-conserving order)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // 1. CLEAR COAT (Top layer - evaluated first)
    float clearcoat = material.clearcoat;
    if (payload.use_blended_data) clearcoat = payload.blended_clearcoat;
    if (clearcoat > 0.01f) {
        // Fresnel for clear coat decides reflection probability
        float cc_ior = 1.5f;
        float cc_f0 = ((cc_ior - 1.0f) / (cc_ior + 1.0f));
        cc_f0 *= cc_f0;
        float cc_fresnel = cc_f0 + (1.0f - cc_f0) * powf(1.0f - fmaxf(dot(V, N), 0.0f), 5.0f);
        float cc_prob = clearcoat * cc_fresnel;

        float clearcoat_roughness = material.clearcoat_roughness;
        if (payload.use_blended_data) clearcoat_roughness = payload.blended_clearcoat_roughness;

        if (random_float(rng) < cc_prob) {
            *is_specular = (clearcoat_roughness < 0.02f);
            bool got = clearcoat_scatter(material, payload, ray_in, rng, scattered, attenuation, pdf);
            if (got) {
                // Compensate selection probability (match Vulkan behavior)
                *attenuation = (*attenuation) * (1.0f / fmaxf(cc_prob, 0.01f));
                return true;
            }
            return false;
        }
        // If not chosen, continue to base layer (no change here)
    }
    
    // 2. TRANSMISSION (Glass/Water)
    // Always mark transmission bounces as specular to skip NEE.
    // roughness is clamped >= 0.02f above, so (roughness < 0.02f) is always false —
    // meaning NEE ran for every refraction. evaluate_brdf gives (diffuse*0 + spec) for
    // high-transmission glass, causing bright specular fireflies on the refracted path.
    if (transmission > 0.01f) {
        if (random_float(rng) < transmission) {
            *is_specular = true;
            return transmission_scatter(material, payload, ray_in, rng, scattered, attenuation);
        }
        albedo *= 1.0f / fmaxf(1.0f - transmission, 0.01f);
        transmission = 0.0f;
    }
    
    // 3. SUBSURFACE SCATTERING (Random Walk)
    float sss = material.subsurface;
    if (payload.use_blended_data) sss = payload.blended_subsurface;
    if (sss > 0.01f && random_float(rng) < sss) {
        *is_specular = false;
        *pdf = 1.0f;
        return sss_random_walk_scatter(material, payload, ray_in, rng, scattered, attenuation);
    }
    
    // 4. TRANSLUCENT (Thin surface transmission: leaves, paper, fabric)
    float translucent = material.translucent;
    if (payload.use_blended_data) translucent = payload.blended_translucent;
    if (translucent > 0.01f && random_float(rng) < translucent) {
        *is_specular = false;
        *pdf = 1.0f / M_PIf;  // Cosine-weighted PDF
        return translucent_scatter(material, payload, ray_in, rng, scattered, attenuation);
    }
    
    // 5. STANDARD DIFFUSE + SPECULAR (Base layer)
    // Mirror Vulkan closesthit.rchit three-case structure exactly:
    //   metallic >= 0.999 → pure metal (always specular, no selection)
    //   metallic <= 0.001 → pure dielectric (Fresnel-driven, white F0 specular, NO compensation)
    //   otherwise         → stochastic metallic blend weighted by the selection probability
    float3 wo = -normalize(ray_in.direction);
    float cosTheta_N = fmaxf(dot(V, N), 0.0f);
    float3 F_avg = F0 + (make_float3(1.0f, 1.0f, 1.0f) - F0) / 21.0f;
    float3 k_d   = (make_float3(1.0f, 1.0f, 1.0f) - F_avg) * (1.0f - metallic);

    // Shared VNDF specular scatter helper (inline lambda replacement)
    auto scatter_specular = [&](float3 F0_lobe, float comp) -> bool {
        float alpha = fmaxf(roughness * roughness, GPU_MIN_ALPHA);
        float3 L = ggxSampleVNDF(N, V, alpha, random_float(rng), random_float(rng));
        if (dot(N, L) <= 0.0f) L = reflect(-V, N);
        float3 H        = normalize(V + L);
        float  cos_th   = fmaxf(dot(V, H), 1e-4f);
        float3 F        = fresnel_schlick_roughness(cos_th, F0_lobe, roughness);
        float  NdotL_s  = fmaxf(dot(N, L), GPU_MIN_DOT);
        float  k_g1     = alpha * 0.5f;
        float  G1L      = NdotL_s / (NdotL_s * (1.0f - k_g1) + k_g1);
        *scattered      = Ray(offset_ray(payload.position, N), L);
        *attenuation    = clamp3f(F * G1L * comp, 0.0f, 1e4f);
        *pdf            = pdf_brdf(material, wo, L, N, payload.bounce_index);
        *is_specular    = (roughness < 0.02f);
        return true;
    };

    if (metallic >= 0.999f) {
        // ── Pure metal: always specular, no selection probability ──
        // Matches Vulkan: if (metallic >= 0.999) scatterMetal(albedo, roughness)
        scatter_specular(F0, 1.0f);

    } else if (metallic <= 0.001f) {
        // ── Pure dielectric: Fresnel-driven selection, NO compensation ──
        // Matches Vulkan: fresnelBase-based rnd; specular uses vec3(1.0) as F0
        // so that Fresnel=1 and attenuation=G1L (≈1) — the 0.04 weight comes
        // from the selection probability itself, not from the F term.
        float fresnelBase = 0.04f + (1.0f - 0.04f) * powf(1.0f - cosTheta_N, 5.0f);
        if (random_float(rng) < fresnelBase) {
            // Specular: white F0 → Fresnel=1, weight = G1L (naturally small chance)
            scatter_specular(make_float3(1.0f, 1.0f, 1.0f), 1.0f);
        } else {
            // Diffuse: full albedo, no compensation (matches Vulkan scatterDiffuse)
            float3 diff_dir  = random_cosine_direction(rng);
            float3 up        = fabsf(N.z) < 0.999f ? make_float3(0,0,1) : make_float3(1,0,0);
            float3 tanX      = normalize(cross(up, N));
            float3 tanY      = cross(N, tanX);
            float3 world_dir = normalize(tanX*diff_dir.x + tanY*diff_dir.y + N*diff_dir.z);
            *scattered    = Ray(offset_ray(payload.position, N), world_dir);
            *attenuation  = clamp3f(albedo, 0.0f, 1e4f);
            *pdf          = fmaxf(dot(N, world_dir), GPU_MIN_DOT) / M_PIf;
            *is_specular  = false;
        }

    } else {
        // Metallic blend: stochastic selection.
        //
        // The selection probability already is the material weight. Applying
        // 1/p compensation without also multiplying the lobe by that weight
        // makes intermediate metallic values estimate full diffuse + full
        // specular energy, which is not energy conserving and causes bright
        // outlier paths on metal/rough-metal surfaces.
        float p_metal = metallic;
        if (random_float(rng) < p_metal) {
            scatter_specular(F0, 1.0f);
        } else {
            float3 diff_dir  = random_cosine_direction(rng);
            float3 up        = fabsf(N.z) < 0.999f ? make_float3(0,0,1) : make_float3(1,0,0);
            float3 tanX      = normalize(cross(up, N));
            float3 tanY      = cross(N, tanX);
            float3 world_dir = normalize(tanX*diff_dir.x + tanY*diff_dir.y + N*diff_dir.z);
            *scattered    = Ray(offset_ray(payload.position, N), world_dir);
            *attenuation  = clamp3f(albedo, 0.0f, 1e4f);
            *pdf          = fmaxf(dot(N, world_dir), GPU_MIN_DOT) / M_PIf;
            *is_specular  = false;
        }
    }

    if (!finite3(*attenuation)) {
        *attenuation = make_float3(1.0f, 1.0f, 1.0f);
    }
    return true;
}
