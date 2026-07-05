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

// --- Resin noise/hash helpers (GPU parity with Vulkan) ---
__device__ __forceinline__ float rh_hash13(float3 p) {
    float3 q = make_float3(p.x * 0.1031f, p.y * 0.1031f, p.z * 0.1031f);
    q.x = q.x - floorf(q.x);
    q.y = q.y - floorf(q.y);
    q.z = q.z - floorf(q.z);
    float dot_val = q.x * (q.y + 33.33f) + q.y * (q.z + 33.33f) + q.z * (q.x + 33.33f);
    q.x += dot_val;
    q.y += dot_val;
    q.z += dot_val;
    float r = (q.x + q.y) * q.z;
    return r - floorf(r);
}

__device__ __forceinline__ float3 rh_hash33(float3 p) {
    float x = p.x * 127.1f + p.y * 311.7f + p.z * 74.7f;
    float y = p.x * 269.5f + p.y * 183.3f + p.z * 246.1f;
    float z = p.x * 113.5f + p.y * 271.9f + p.z * 124.6f;
    float rx = sinf(x) * 43758.5453f;
    float ry = sinf(y) * 43758.5453f;
    float rz = sinf(z) * 43758.5453f;
    return make_float3(rx - floorf(rx), ry - floorf(ry), rz - floorf(rz));
}

__device__ __forceinline__ float rh_vnoise(float3 x) {
    float3 i = make_float3(floorf(x.x), floorf(x.y), floorf(x.z));
    float3 f = make_float3(x.x - i.x, x.y - i.y, x.z - i.z);
    f.x = f.x * f.x * f.x * (f.x * (f.x * 6.0f - 15.0f) + 10.0f);
    f.y = f.y * f.y * f.y * (f.y * (f.y * 6.0f - 15.0f) + 10.0f);
    f.z = f.z * f.z * f.z * (f.z * (f.z * 6.0f - 15.0f) + 10.0f);

    float n000 = rh_hash13(i);
    float n100 = rh_hash13(make_float3(i.x + 1.0f, i.y, i.z));
    float n010 = rh_hash13(make_float3(i.x, i.y + 1.0f, i.z));
    float n110 = rh_hash13(make_float3(i.x + 1.0f, i.y + 1.0f, i.z));
    float n001 = rh_hash13(make_float3(i.x, i.y, i.z + 1.0f));
    float n101 = rh_hash13(make_float3(i.x + 1.0f, i.y, i.z + 1.0f));
    float n011 = rh_hash13(make_float3(i.x, i.y + 1.0f, i.z + 1.0f));
    float n111 = rh_hash13(make_float3(i.x + 1.0f, i.y + 1.0f, i.z + 1.0f));

    float nx00 = n000 * (1.0f - f.x) + n100 * f.x;
    float nx10 = n010 * (1.0f - f.x) + n110 * f.x;
    float nx01 = n001 * (1.0f - f.x) + n101 * f.x;
    float nx11 = n011 * (1.0f - f.x) + n111 * f.x;
    float nxy0 = nx00 * (1.0f - f.y) + nx10 * f.y;
    float nxy1 = nx01 * (1.0f - f.y) + nx11 * f.y;
    return nxy0 * (1.0f - f.z) + nxy1 * f.z;
}

__device__ __forceinline__ float rh_fbm(float3 p) {
    float v = 0.0f, a = 0.5f, tot = 0.0f;
    for (int i = 0; i < 5; ++i) {
        v   += a * fabsf(2.0f * rh_vnoise(p) - 1.0f);
        tot += a;
        p = make_float3(p.x * 2.03f + 7.1f, p.y * 2.03f + 3.7f, p.z * 2.03f + 11.3f);
        a *= 0.5f;
    }
    return v / fmaxf(tot, 1e-4f);
}

__device__ __forceinline__ float rh_worley(float3 p) {
    float3 ip = make_float3(floorf(p.x), floorf(p.y), floorf(p.z));
    float3 fp = make_float3(p.x - ip.x, p.y - ip.y, p.z - ip.z);
    float d = 1.0f;
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                float3 g = make_float3((float)x, (float)y, (float)z);
                float3 o = rh_hash33(ip + g);
                float3 diff = g + o - fp;
                d = fminf(d, sqrtf(dot(diff, diff)));
            }
        }
    }
    return d;
}

__device__ inline bool ms_weather_active(const WeatherParams& weather) {
    return weather.enabled != 0 && weather.type != WEATHER_NONE &&
           weather.intensity > 0.0f && weather.density > 0.0f;
}

__device__ inline bool ms_weather_surface_active(const WeatherParams& weather) {
    if (weather.enabled == 0 || weather.type == WEATHER_NONE || weather.surface_response_enabled == 0) {
        return false;
    }

    float surfaceSignal = 0.0f;
    if (weather.type == WEATHER_RAIN) {
        surfaceSignal = weather.surface_wetness_output;
    } else if (weather.type == WEATHER_SNOW || weather.type == WEATHER_DUST) {
        surfaceSignal = weather.surface_accumulation_output;
    }

    return surfaceSignal > 0.001f || (weather.intensity > 0.0f && weather.density > 0.0f);
}

__device__ inline float3 ms_mix3(float3 a, float3 b, float t) {
    return a * (1.0f - t) + b * t;
}

__device__ inline float ms_saturate(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ inline float3 ms_safe_normalize(const float3& v, const float3& fallback) {
    float len2 = dot(v, v);
    if (len2 <= 1e-10f) return fallback;
    return v * rsqrtf(len2);
}

__device__ inline float ms_weather_surface_exposure(const WeatherParams& weather, const OptixHitResult& payload) {
    const float3 baseNormal = ms_safe_normalize(payload.normal, make_float3(0.0f, 1.0f, 0.0f));
    float upMask = ms_saturate((baseNormal.y - 0.12f) / 0.78f);
    upMask = upMask * upMask * (3.0f - 2.0f * upMask);

    float windAmount = ms_saturate(weather.wind_speed / 35.0f);
    float windLen2 = dot(weather.wind_direction, weather.wind_direction);
    float3 windDir = (windLen2 > 1e-8f) ? normalize(weather.wind_direction) : make_float3(1.0f, 0.0f, 0.0f);
    float3 incoming = normalize(make_float3(0.0f, 1.0f, 0.0f) - windDir * windAmount);
    float windFacing = ms_saturate(dot(baseNormal, incoming));
    float exposure = ms_saturate(upMask * (1.0f - windAmount * 0.78f) + windFacing * (0.12f + windAmount * 1.22f));
    float n = pd_vnoise3(payload.position * fmaxf(weather.precipitation_scale, 0.1f) * 0.18f +
                         make_float3(13.1f, 47.2f, 5.7f));
    float breakup = ms_saturate(n * 1.35f - 0.18f);
    return exposure * (0.45f + 0.55f * breakup);
}

__device__ inline float ms_weather_surface_geometric_support(const WeatherParams& weather, const float3& supportNormal) {
    const float3 macroNormal = ms_safe_normalize(supportNormal, make_float3(0.0f, 1.0f, 0.0f));
    float support = 0.0f;
    if (weather.type == WEATHER_SNOW) {
        support = ms_saturate((macroNormal.y - 0.02f) / 0.72f);
        support = support * support;
    } else {
        support = ms_saturate((macroNormal.y - 0.02f) / 0.78f);
    }
    return support * support * (3.0f - 2.0f * support);
}

__device__ inline float3 ms_weather_surface_support_normal(const OptixHitResult& payload) {
    const float3 baseNormal = ms_safe_normalize(payload.normal, make_float3(0.0f, 1.0f, 0.0f));
    const float3 macroNormal = ms_safe_normalize(payload.geom_normal, baseNormal);
    return ms_safe_normalize(macroNormal * 0.38f + baseNormal * 0.62f, baseNormal);
}

__device__ inline float ms_weather_surface_settling(const WeatherParams& weather, const OptixHitResult& payload) {
    if (weather.type != WEATHER_SNOW && weather.type != WEATHER_DUST) return 0.0f;

    const float settlingAmount = ms_saturate(weather.surface_settling_output);
    if (settlingAmount <= 1e-4f) return 0.0f;

    const float3 shadingNormal = ms_safe_normalize(payload.normal, make_float3(0.0f, 1.0f, 0.0f));
    const float3 macroNormal = ms_weather_surface_support_normal(payload);
    const float support = ms_weather_surface_geometric_support(weather, macroNormal);
    const float supportGate = ms_saturate((support - 0.02f) / 0.58f);
    if (supportGate <= 1e-4f) return 0.0f;
    const float exposure = ms_weather_surface_exposure(weather, payload);
    const float cavity = ms_saturate((1.0f - dot(shadingNormal, macroNormal)) * 3.8f + (1.0f - support) * 0.10f);
    float3 windFlat = make_float3(weather.wind_direction.x, 0.0f, weather.wind_direction.z);
    float3 leeDir = dot(windFlat, windFlat) > 1e-8f ? ms_safe_normalize(make_float3(-windFlat.x, 0.28f, -windFlat.z), make_float3(0.0f, 1.0f, 0.0f)) : make_float3(0.0f, 1.0f, 0.0f);
    const float lee = ms_saturate(dot(macroNormal, leeDir) * 0.85f + cavity * 0.35f);
    const float shelter = ms_saturate((1.0f - exposure) * 0.52f + cavity * 0.26f + (1.0f - support) * 0.22f + lee * 0.42f);
    const float pocketNoise = pd_vnoise3(payload.position * 0.085f + make_float3(31.4f, 9.7f, 54.2f));
    const float pocketMask = ms_saturate(cavity * 0.92f + pocketNoise * 0.26f);
    const float slopeBase = ms_saturate((support - 0.16f) / 0.54f);
    const float density = ms_saturate(weather.density);
    const float typeBoost = (weather.type == WEATHER_SNOW) ? 1.34f : 1.04f;
    const float anchor = fmaxf(pocketMask, slopeBase * 0.30f + cavity * 0.40f + lee * 0.30f);
    return ms_saturate(settlingAmount * supportGate * shelter * anchor * (0.76f + density * 1.10f) * typeBoost);
}

__device__ inline float ms_weather_surface_accumulation(const WeatherParams& weather, const OptixHitResult& payload) {
    if (weather.type != WEATHER_SNOW && weather.type != WEATHER_DUST) return 0.0f;

    const float baseAccum = ms_saturate(weather.surface_accumulation_output);
    const float intensity = ms_saturate(weather.intensity);
    const float density = ms_saturate(weather.density);
    const float geomSupport = ms_weather_surface_geometric_support(weather, ms_weather_surface_support_normal(payload));
    const float intensityResponse = 0.80f + intensity * 0.70f;
    const float densityResponse = 0.35f + density * 1.15f;
    const float typeBoost = (weather.type == WEATHER_SNOW) ? 1.10f : 0.90f;
    const float directAccum = baseAccum * intensityResponse * ms_weather_surface_exposure(weather, payload) * densityResponse * typeBoost * geomSupport;
    const float settling = ms_weather_surface_settling(weather, payload);
    return ms_saturate(directAccum + (1.0f - ms_saturate(directAccum)) * settling);
}

__device__ inline float ms_weather_surface_height(const WeatherParams& weather, const float3& pos) {
    const float scale = fmaxf(weather.precipitation_scale, 0.1f);
    const float heightBoost = 0.25f + ms_saturate(weather.surface_height_output) * 3.75f;
    if (weather.type == WEATHER_SNOW) {
        float2 windXZ = make_float2(weather.wind_direction.x, weather.wind_direction.z);
        float windLen2 = windXZ.x * windXZ.x + windXZ.y * windXZ.y;
        float invWindLen = windLen2 > 1e-8f ? rsqrtf(windLen2) : 0.0f;
        float2 along = windLen2 > 1e-8f ? make_float2(windXZ.x * invWindLen, windXZ.y * invWindLen) : make_float2(1.0f, 0.0f);
        float2 across = make_float2(-along.y, along.x);
        float2 uv = make_float2(pos.x * along.x + pos.z * along.y, pos.x * across.x + pos.z * across.y);
        float3 p = make_float3(uv.x * scale * 0.12f, pos.y * scale * 0.03f, uv.y * scale * 0.12f);
        float broad = pd_vnoise3(p * 0.55f + make_float3(17.3f, 9.1f, 41.7f));
        float drift = 1.0f - fabsf(pd_vnoise3(make_float3(p.x * 1.45f, p.y * 0.8f, p.z * 0.58f) + make_float3(3.7f, 29.4f, 11.8f)) * 2.0f - 1.0f);
        drift *= drift;
        float clumps = 1.0f - fabsf(pd_vnoise3(p * 2.90f + make_float3(61.2f, 7.5f, 18.9f)) * 2.0f - 1.0f);
        float micro = pd_vnoise3(p * 7.40f + make_float3(8.3f, 51.7f, 27.4f));
        return (broad * 0.22f + drift * 0.36f + clumps * 0.27f + micro * 0.15f) * heightBoost;
    }

    const float3 p = pos * (scale * 0.18f);
    float wisps = pd_vnoise3(p + make_float3(19.7f, 5.3f, 27.1f));
    float grain = pd_vnoise3(p * 2.85f + make_float3(4.1f, 37.8f, 12.4f));
    float streak = pd_vnoise3(p * 1.65f + make_float3(44.5f, 14.2f, 7.6f));
    return (wisps * 0.30f + grain * 0.45f + streak * 0.25f) * heightBoost;
}

__device__ inline float3 ms_weather_surface_normal(const WeatherParams& weather, const OptixHitResult& payload) {
    const float3 baseNormal = ms_safe_normalize(payload.normal, make_float3(0.0f, 1.0f, 0.0f));
    if (!ms_weather_surface_active(weather)) return baseNormal;
    if (weather.type != WEATHER_SNOW && weather.type != WEATHER_DUST) return baseNormal;

    const float accumulation = ms_weather_surface_accumulation(weather, payload);
    if (accumulation <= 1e-4f) return baseNormal;

    const float3 supportNormal = ms_weather_surface_support_normal(payload);
    const float geomSupport = ms_weather_surface_geometric_support(weather, supportNormal);
    if (geomSupport <= 1e-4f) return baseNormal;

    const float settling = ms_weather_surface_settling(weather, payload);
    float detailCapture = ms_saturate((baseNormal.y - 0.04f) / 0.82f);
    detailCapture = 0.45f + 0.55f * detailCapture;
    const float heightResponse = 0.12f + ms_saturate(weather.surface_height_output) * 0.95f;
    const float buildup = ms_saturate(accumulation + settling * 0.85f);
    float normalStrength = buildup * detailCapture * heightResponse * (weather.type == WEATHER_SNOW ? 0.42f : 0.15f);
    if (normalStrength <= 1e-4f) return baseNormal;

    float3 windDir = weather.wind_direction;
    float3 tangent = windDir - baseNormal * dot(windDir, baseNormal);
    if (dot(tangent, tangent) <= 1e-8f) {
        tangent = cross(fabsf(baseNormal.y) < 0.999f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f), baseNormal);
    }
    tangent = ms_safe_normalize(tangent, make_float3(1.0f, 0.0f, 0.0f));
    float3 bitangent = ms_safe_normalize(cross(baseNormal, tangent), make_float3(0.0f, 0.0f, 1.0f));
    tangent = ms_safe_normalize(cross(bitangent, baseNormal), tangent);

    float sampleStep = (weather.type == WEATHER_SNOW ? 0.62f : 0.90f) / fmaxf(weather.precipitation_scale, 0.35f);
    float heightCenter = ms_weather_surface_height(weather, payload.position);
    float heightT = ms_weather_surface_height(weather, payload.position + tangent * sampleStep);
    float heightB = ms_weather_surface_height(weather, payload.position + bitangent * sampleStep);
    float gradT = fminf(fmaxf((heightT - heightCenter) / sampleStep, -0.28f), 0.28f);
    float gradB = fminf(fmaxf((heightB - heightCenter) / sampleStep, -0.28f), 0.28f);

    float3 perturbed = baseNormal - tangent * (gradT * normalStrength) - bitangent * (gradB * normalStrength);
    perturbed = ms_safe_normalize(perturbed, baseNormal);

    const float3 geomNormal = ms_safe_normalize(payload.geom_normal, baseNormal);
    if (dot(perturbed, geomNormal) < 0.05f) {
        perturbed = ms_safe_normalize(baseNormal + (perturbed - geomNormal) * 0.35f, baseNormal);
    }
    if (dot(perturbed, supportNormal) < 0.55f) {
        perturbed = ms_safe_normalize(supportNormal + (perturbed - supportNormal) * 0.05f, supportNormal);
    }
    return perturbed;
}

__device__ inline void apply_weather_surface_gpu(
    const WeatherParams& weather,
    const OptixHitResult& payload,
    float3& albedo,
    float& roughness,
    float& metallic
) {
    if (!ms_weather_surface_active(weather)) return;

    float exposed = ms_weather_surface_exposure(weather, payload);

    if (weather.type == WEATHER_RAIN) {
        float wet = fminf(fmaxf(weather.surface_wetness_output, 0.0f), 1.0f) *
                    (0.35f + 0.65f * exposed);
        albedo = ms_mix3(albedo, albedo * 0.50f, wet * 0.62f);
        roughness = fmaxf(0.012f, roughness * (1.0f - wet * 0.78f));
        metallic = fmaxf(0.0f, metallic - wet * 0.05f);
    } else if (weather.type == WEATHER_SNOW) {
        float acc = ms_weather_surface_accumulation(weather, payload);
        float settling = ms_weather_surface_settling(weather, payload);
        float heightLift = ms_saturate(weather.surface_height_output);
        float cover = ms_saturate(acc + settling * 0.84f + heightLift * (acc * 0.08f + settling * 0.30f));
        albedo = ms_mix3(albedo, make_float3(0.88f, 0.91f, 0.96f), cover * 0.72f);
        roughness = fminf(1.0f, roughness + cover * (0.45f + heightLift * 0.10f));
        metallic *= (1.0f - cover * 0.8f);
    } else if (weather.type == WEATHER_DUST) {
        float acc = ms_weather_surface_accumulation(weather, payload);
        float settling = ms_weather_surface_settling(weather, payload);
        float heightLift = ms_saturate(weather.surface_height_output);
        float cover = ms_saturate(acc + settling * 0.90f + heightLift * settling * 0.22f);
        albedo = ms_mix3(albedo, make_float3(0.58f, 0.46f, 0.30f), cover * 0.55f);
        roughness = fminf(1.0f, roughness + cover * (0.35f + heightLift * 0.08f));
        metallic *= (1.0f - cover * 0.55f);
    }
}

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

__device__ __noinline__ float pdf_brdf(const GpuMaterial& mat, const float3& wo, const float3& wi, const float3& N, int bounce_index = 0) {
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


__device__ __noinline__ float3 evaluate_brdf(
    const GpuMaterial& material,
    const OptixHitResult& payload,
    const float3& wo,
    const float3& wi
)
{
    const float3 N = ms_weather_surface_normal(optixLaunchParams.world.weather, payload);
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

    float specular = material.specular;
    if (material.specular_tex) {
        specular = tex2D<float4>(material.specular_tex, uv.x, uv.y).x * material.specular;
    }
    else if (payload.has_specular_tex) {
        specular = tex2D<float4>(payload.specular_tex, uv.x, uv.y).x * material.specular;
    }
    specular = fminf(fmaxf(specular, 0.0f), 1.0f);

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

    apply_weather_surface_gpu(optixLaunchParams.world.weather, payload, albedo, roughness, metallic);
    roughness = fminf(fmaxf(roughness, 0.02f), 1.0f);
    metallic = fminf(fmaxf(metallic, 0.0f), 1.0f);

    float alpha = max(roughness * roughness, GPU_MIN_ALPHA);
    float alpha2 = alpha * alpha;
    float denom = (NdotH * NdotH) * (alpha2 - 1.0f) + 1.0f;
    float D = alpha2 / (M_PIf * denom * denom);

    float dielectricF0 = fminf(fmaxf(0.08f * specular, 0.0f), 0.08f);
    float3 F0 = lerp(make_float3(dielectricF0, dielectricF0, dielectricF0), albedo, metallic);
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
    if (ms_weather_surface_active(optixLaunchParams.world.weather) &&
        optixLaunchParams.world.weather.type == WEATHER_RAIN) {
        float wet = fminf(fmaxf(optixLaunchParams.world.weather.surface_wetness_output, 0.0f), 1.0f);
        clearcoat_strength = fmaxf(clearcoat_strength, wet * 0.72f);
        cc_roughness = fminf(cc_roughness, fmaxf(0.006f, 0.045f - wet * 0.030f));
    }
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

    // Iridescent thin-film tint (same OPD/cos model as the bubble path). iridescence=0 →
    // white (plain clearcoat, no change). At grazing the optical path difference grows,
    // cycling the interference hue (oil-slick / beetle-shell / candy paint).
    float3 cc_tint = make_float3(1.0f, 1.0f, 1.0f);
    float iridescence = fminf(fmaxf(material.clearcoat_iridescence, 0.0f), 1.0f);
    if (iridescence > 1e-3f) {
        float opd = material.clearcoat_film_thickness * (1.0f / fmaxf(VdotH, 0.15f));
        float3 film = make_float3(0.55f + 0.45f * cosf(opd * 6.2831853f),
                                  0.55f + 0.45f * cosf(opd * 6.2831853f + 2.0944f),
                                  0.55f + 0.45f * cosf(opd * 6.2831853f + 4.18879f));
        cc_tint = make_float3(1.0f - iridescence + iridescence * film.x,
                              1.0f - iridescence + iridescence * film.y,
                              1.0f - iridescence + iridescence * film.z);
    }

    // Return white specular scaled by Fresnel*G1L (caller will compensate selection probability)
    *attenuation = make_float3(fresnel * G1L, fresnel * G1L, fresnel * G1L) * fmaxf(clearcoat_strength, 0.0f) * cc_tint;
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
    // Resin needs real refraction to read as a solid volume — with IOR≈1 the ray
    // passes straight through and only darkens (no lensing / thickness cue).
    if (material.transmission_density > 1e-4f) ior = fmaxf(ior, 1.45f);
    // Spectral dispersion state (selection happens AFTER the lobe decision below —
    // only refracted paths disperse; the mirror lobe is wavelength-independent).
    int    disp_out_ch = ray_in.dispersion_channel;   // 0 = unset, 1/2/3 = R/G/B
    float3 disp_sel    = make_float3(1.0f, 1.0f, 1.0f);
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
    if (roughness > 0.0005f) {
        const float sample_roughness = fmaxf(roughness, 0.02f);
        const float blend = fminf(fmaxf(roughness / 0.02f, 0.0f), 1.0f);
        const float smooth_blend = blend * blend * (3.0f - 2.0f * blend);
        const float3 sampled_normal = ggx_glass_effective_normal(
            random_float(rng),
            random_float(rng),
            sample_roughness,
            macro_normal,
            -unit_direction
        );
        micro_normal = normalize(macro_normal * (1.0f - smooth_blend) + sampled_normal * smooth_blend);
    }

    float cos_theta = fminf(dot(-unit_direction, macro_normal), 1.0f);
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    bool cannot_refract = (eta * sin_theta > 1.0f);
    float reflect_prob = schlick(cos_theta, ior);
    bool do_reflect = cannot_refract || (random_float(rng) < reflect_prob);

    // ── Spectral dispersion: ONLY the refracted lobe disperses (parity with Vulkan
    // scatterGlass). Selecting before the lobe decision splashed ×3 mono-channel
    // noise onto reflection-lit surfaces. Channel is picked ONCE per path and
    // travels on the Ray so the exit interface refracts with the same channel IOR.
    // Resin (transmission_density) path skipped.
    if (!do_reflect && material.dispersion > 1e-3f && material.transmission_density <= 1e-4f) {
        int ch = disp_out_ch - 1;
        if (ch < 0) {
            ch = min((int)(random_float(rng) * 3.0f), 2);
            disp_sel = make_float3(ch == 0 ? 3.0f : 0.0f,
                                   ch == 1 ? 3.0f : 0.0f,
                                   ch == 2 ? 3.0f : 0.0f);
            disp_out_ch = ch + 1;
        }
        float spread = (ior - 1.0f) * material.dispersion * 0.06f;  // half of total F–C spread
        ior += (ch == 0) ? -spread : ((ch == 2) ? spread : 0.0f);
        eta = front_face ? (1.0f / ior) : ior;   // refraction uses the channel IOR
    }

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
    if (payload.use_blended_data && !payload.water_surface_active) {
        // Terrain (or other blended) surfaces: use the blended albedo as tint.
        // Water surfaces deliberately keep material.albedo (= deep_color) here
        // so the Beer-Lambert tint matches CPU PrincipledBSDF::scatter, which
        // passes albedoProperty.color (= constant deep_color) to Dielectric.
        // payload.blended_albedo carries the depth+foam-blended water_color
        // for direct-lighting BRDF (consumed in evaluate_brdf), not for tint.
        tint = payload.blended_albedo;
    }
    else if (payload.has_albedo_tex) {
        float4 tex = tex2D<float4>(payload.albedo_tex, uv.x, uv.y);
        tint = (tex.y == 0.0f && tex.z == 0.0f) ?
            make_float3(tex.x, tex.x, tex.x) :
            make_float3(tex.x, tex.y, tex.z);
    }

    const bool real_depth = (material.transmission_density > 1e-4f);

    // ── RESIN terminate-on-base (parity with Vulkan scatterGlass): the refraction
    // lobe travels the resin THICKNESS, hits the base albedo at that depth, and
    // scatters back out through the resin (absorb in + out). Opaque under a refractive
    // absorbing resin layer; the reflection lobe stays the glossy resin top. A small
    // base extinction (0.25) makes Resin Depth darken even for a white resin;
    // resin_color tints which channels survive.
    if (real_depth && !do_reflect) {
        float3 Tdir = normalize(direction);

        // Reached the base: parallax-offset the base lookup along the refracted
        // lateral travel. Always applied when resin layer is active.
        float3 inPlane = Tdir - macro_normal * dot(Tdir, macro_normal);
        float2 parUV = uv
                     + make_float2(dot(inPlane, payload.tangent), dot(inPlane, payload.bitangent))
                       * (material.transmission_density * 0.05f);

        float3 base_tint = material.albedo;
        if (payload.use_blended_data && !payload.water_surface_active) {
            base_tint = payload.blended_albedo;
        } else if (payload.has_albedo_tex) {
            float2 par_material_uv = apply_material_uv_transform(material, parUV);
            float4 tex = tex2D<float4>(payload.albedo_tex, par_material_uv.x, par_material_uv.y);
            base_tint = (tex.y == 0.0f && tex.z == 0.0f) ?
                make_float3(tex.x, tex.x, tex.x) :
                make_float3(tex.x, tex.y, tex.z);
        }

        bool resinHasInclusions = (material.resin_inclusion > 0.001f || material.resin_dirt > 0.001f);
        float3 ct      = clamp3f(material.resin_color, 0.0f, 1.0f);
        float  cosV    = fmaxf(fabsf(dot(-unit_direction, macro_normal)), 0.25f);
        float3 ext     = make_float3((1.0f - ct.x) + 0.25f, (1.0f - ct.y) + 0.25f, (1.0f - ct.z) + 0.25f);
        float3 absorb  = make_float3(1.0f, 1.0f, 1.0f);
        bool dirtHit   = false;

        if (resinHasInclusions) {
            const int RESIN_STEPS = 6;
            float dt  = fmaxf(material.transmission_density, 1e-3f) / float(RESIN_STEPS);
            float scl = fmaxf(material.resin_inclusion_scale, 0.01f);
            float3 P  = payload.position;
            for (int i = 0; i < RESIN_STEPS; ++i) {
                P = P + Tdir * dt;
                float dust = rh_fbm(P * scl);
                dust = dust * dust;
                float localExt = 1.0f + material.resin_inclusion * dust * 6.0f;
                absorb.x *= expf(-dt * ext.x * localExt);
                absorb.y *= expf(-dt * ext.y * localExt);
                absorb.z *= expf(-dt * ext.z * localExt);

                if (material.resin_dirt > 0.001f) {
                    float cell = rh_worley(P * scl * 3.0f);
                    auto smoothstep_lambda = [](float edge0, float edge1, float x) {
                        float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
                        return t * t * (3.0f - 2.0f * t);
                    };
                    float speck = 1.0f - smoothstep_lambda(0.0f, 0.18f, cell);
                    if (speck * material.resin_dirt > 0.5f) {
                        dirtHit = true;
                        break;
                    }
                }
            }
            if (dirtHit) {
                base_tint = material.resin_dirt_color * absorb;
            } else {
                base_tint = base_tint * absorb;
            }
        } else {
            float pathLen = 2.0f * material.transmission_density / cosV;
            absorb  = make_float3(expf(-pathLen * ext.x), expf(-pathLen * ext.y), expf(-pathLen * ext.z));
            base_tint = base_tint * absorb;
        }

        float3 baseDir = cosine_sample_hemisphere(rng, macro_normal);
        *attenuation   = clamp3f(base_tint, 0.0f, 1.0f) * disp_sel;
        *scattered     = Ray(offset_ray(payload.position, macro_normal), normalize(baseDir));
        scattered->dispersion_channel = disp_out_ch;
        return true;
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

    *attenuation = *attenuation * disp_sel;
    *scattered = Ray(offset_ray(payload.position, offset_n), normalize(direction));
    scattered->dispersion_channel = disp_out_ch;

    return true;
}

// __noinline__: keeps this large BSDF dispatch out of the raygen kernel's
// register budget. Inlined, it dominated raygen's frame and crushed
// occupancy on indirect bounces. As a separate device function it gets
// its own stack frame; the call overhead (~30 cycles) is dwarfed by the
// occupancy gain on a 1000+ line shader. Same rationale applies to
// evaluate_brdf / pdf_brdf below.
enum PathBounceType {
    PATH_BOUNCE_SPECULAR = 0,
    PATH_BOUNCE_DIFFUSE = 1,
    PATH_BOUNCE_TRANSMISSION = 2
};

__device__ __noinline__ bool scatter_material(
    const GpuMaterial& material,         // dışarıdan gelen materyal
    OptixHitResult& payload,        // payload (normal, uv, textures)
    const Ray& ray_in,
    curandState* rng,
    Ray* scattered,
    float3* attenuation,
    float* pdf,
    bool* is_specular,
    int* bounce_type
)
{
    *bounce_type = PATH_BOUNCE_SPECULAR;
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

    // ── Thin-shell BUBBLE (champagne / soda / soap-foam close-up) ──────────────
    // A bubble is a THIN dielectric film: light either Fresnel-reflects off the
    // shell — the bright silver rim, strong at grazing — or passes STRAIGHT through
    // (a thin shell enters and exits parallel, no net refraction bending). So it
    // reads as a bright-rimmed transparent sphere regardless of the medium around
    // it (no nested-dielectric needed). bubble_ior drives the rim Fresnel.
    if (material.flags & GPU_MAT_FLAG_BUBBLE) {
        const float3 Nb = (length(N) > 1e-4f) ? normalize(N) : make_float3(0.0f, 0.0f, 1.0f);
        const float3 wi = normalize(ray_in.direction);
        float cosT = fminf(fabsf(dot(wi, Nb)), 1.0f);
        float bio  = (material.bubble_ior > 1.0001f) ? material.bubble_ior : 1.33f;
        float r0   = (1.0f - bio) / (1.0f + bio);
        r0 = r0 * r0;
        float fres = r0 + (1.0f - r0) * powf(1.0f - cosT, 5.0f);
        float3 dir, att;
        if (random_float(rng) < fres) {
            dir = reflect(wi, Nb);                       // bright Fresnel rim
            // Thin-film interference (soap iridescence): optical path through the
            // film scales with view angle (~1/cosT), tinting the reflected rim with
            // a wavelength-dependent rainbow. bubble_film = thickness/cycle count.
            if (material.bubble_film > 1e-3f) {
                float opd = material.bubble_film * (1.0f / fmaxf(cosT, 0.15f));
                att = make_float3(0.55f + 0.45f * cosf(opd * 6.2831853f),
                                  0.55f + 0.45f * cosf(opd * 6.2831853f + 2.0944f),
                                  0.55f + 0.45f * cosf(opd * 6.2831853f + 4.1888f));
            } else {
                att = make_float3(1.0f, 1.0f, 1.0f);
            }
        } else {
            dir = wi;                                    // straight pass-through (thin shell)
            att = make_float3(0.85f + 0.15f * material.albedo.x,   // faint film tint, mostly clear
                              0.85f + 0.15f * material.albedo.y,
                              0.85f + 0.15f * material.albedo.z);
        }
        *scattered    = Ray(offset_ray(payload.position, dir), dir);
        *attenuation  = att;
        *pdf          = 1.0f;
        *is_specular  = true;
        *bounce_type  = PATH_BOUNCE_SPECULAR;
        return true;
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

    float specular = material.specular;
    if (material.specular_tex)
        specular = tex2D<float4>(material.specular_tex, uv.x, uv.y).x * material.specular;
    else if (payload.has_specular_tex)
        specular = tex2D<float4>(payload.specular_tex, uv.x, uv.y).x * material.specular;

    float transmission = material.transmission;
    if (payload.use_blended_data) {
        transmission = payload.blended_transmission;
    }
    else if (payload.has_transmission_tex)
        transmission = tex2D<float4>(payload.transmission_tex, uv.x, uv.y).x;

    roughness = fminf(fmaxf(roughness, 0.02f), 1.0f);
    metallic = fminf(fmaxf(metallic, 0.0f), 1.0f);
    specular = fminf(fmaxf(specular, 0.0f), 1.0f);
    transmission = fminf(fmaxf(transmission, 0.0f), 1.0f);

    // Path regularization (Müller 2018): indirect bounces use a roughness
    // floor so secondary GGX sampling cannot hit near-mirror D-peaks and
    // create metallic fireflies. Primary bounce is untouched so direct
    // specular highlights stay sharp. Must match the same clamp used in
    // evaluate_brdf() so scatter and NEE evaluate the same BRDF.
    if (payload.bounce_index > 0) {
        roughness = fmaxf(roughness, 0.1f);
    }

    apply_weather_surface_gpu(optixLaunchParams.world.weather, payload, albedo, roughness, metallic);
    roughness = fminf(fmaxf(roughness, 0.02f), 1.0f);
    metallic = fminf(fmaxf(metallic, 0.0f), 1.0f);
    payload.normal = ms_weather_surface_normal(optixLaunchParams.world.weather, payload);
    N = payload.normal;

   
   
    float dielectricF0 = fminf(fmaxf(0.08f * specular, 0.0f), 0.08f);
    float3 F0 = lerp(make_float3(dielectricF0, dielectricF0, dielectricF0), albedo, metallic);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LOBE SELECTION (Energy-conserving order)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // 1. CLEAR COAT (Top layer - evaluated first)
    float clearcoat = material.clearcoat;
    if (payload.use_blended_data) clearcoat = payload.blended_clearcoat;
    float clearcoat_roughness = material.clearcoat_roughness;
    if (payload.use_blended_data) clearcoat_roughness = payload.blended_clearcoat_roughness;
    if (ms_weather_surface_active(optixLaunchParams.world.weather) &&
        optixLaunchParams.world.weather.type == WEATHER_RAIN) {
        float wet = fminf(fmaxf(optixLaunchParams.world.weather.surface_wetness_output, 0.0f), 1.0f);
        clearcoat = fmaxf(clearcoat, wet * 0.72f);
        clearcoat_roughness = fminf(clearcoat_roughness, fmaxf(0.006f, 0.045f - wet * 0.030f));
    }
    if (clearcoat > 0.01f) {
        // Fresnel for clear coat decides reflection probability
        float cc_ior = 1.5f;
        float cc_f0 = ((cc_ior - 1.0f) / (cc_ior + 1.0f));
        cc_f0 *= cc_f0;
        float cc_fresnel = cc_f0 + (1.0f - cc_f0) * powf(1.0f - fmaxf(dot(V, N), 0.0f), 5.0f);
        float cc_prob = clearcoat * cc_fresnel;

        if (random_float(rng) < cc_prob) {
            *is_specular = (clearcoat_roughness < 0.02f);
            *bounce_type = PATH_BOUNCE_SPECULAR;
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
    //
    // Resin: a thick resin/glass IS a transmissive body — the depth-absorption look
    // only reads if (almost) every ray travels through it. So when transmission_density
    // is set, force the glass path regardless of the Transmission slider (self-contained).
    float eff_transmission = transmission;
    if (material.transmission_density > 1e-4f) eff_transmission = 1.0f;
    if (eff_transmission > 0.01f) {
        if (random_float(rng) < eff_transmission) {
            *is_specular = true;
            *bounce_type = PATH_BOUNCE_TRANSMISSION;
            return transmission_scatter(material, payload, ray_in, rng, scattered, attenuation);
        }
        albedo *= 1.0f / fmaxf(1.0f - eff_transmission, 0.01f);
        transmission = 0.0f;
    }
    
    // 3. SUBSURFACE SCATTERING (Random Walk)
    float sss = material.subsurface;
    if (payload.use_blended_data) sss = payload.blended_subsurface;
    if (sss > 0.01f && random_float(rng) < sss) {
        *is_specular = false;
        *bounce_type = PATH_BOUNCE_DIFFUSE;
        *pdf = 1.0f;
        return sss_random_walk_scatter(material, payload, ray_in, rng, scattered, attenuation);
    }
    
    // 4. TRANSLUCENT (Thin surface transmission: leaves, paper, fabric)
    float translucent = material.translucent;
    if (payload.use_blended_data) translucent = payload.blended_translucent;
    if (translucent > 0.01f && random_float(rng) < translucent) {
        *is_specular = false;
        *bounce_type = PATH_BOUNCE_DIFFUSE;
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
        *bounce_type    = PATH_BOUNCE_SPECULAR;
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
        float fresnelBase = dielectricF0 + (1.0f - dielectricF0) * powf(1.0f - cosTheta_N, 5.0f);
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
            *bounce_type  = PATH_BOUNCE_DIFFUSE;
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
            *bounce_type  = PATH_BOUNCE_DIFFUSE;
        }
    }

    if (!finite3(*attenuation)) {
        *attenuation = make_float3(1.0f, 1.0f, 1.0f);
    }
    return true;
}
