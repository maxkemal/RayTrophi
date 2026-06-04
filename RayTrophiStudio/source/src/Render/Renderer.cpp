#include "renderer.h"
#include <SDL_image.h>
#include <filesystem>
#include <chrono>      // For wall-clock deltaTime in animation fallback
#include <execution>
#include <algorithm>    // std::for_each_n
#include <cstring>      // std::memcpy for camera hash
#include <imgui.h>
#include <imgui_impl_sdlrenderer2.h>
#include <scene_ui.h>
#include "OptixWrapper.h"
#include "Backend/IBackend.h"
#include "Backend/OptixBackend.h"
#include <future>
#include <thread>
#include <functional>
#include <atomic>
#include <vector_types.h>  // CUDA float4, float3 types for hair GPU upload
#include <cuda_runtime.h>  // cudaGetDevice, cudaStreamSynchronize (OIDN GPU path)
#include "oidn_blend_cuda.h"
#include "Hair/HairBSDF.h"


// Includes moved from renderer.h
#include "Camera.h"
#include "light.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "AreaLight.h"
#include "SpotLight.h"
#include "Volumetric.h"
#include "PrincipledBSDF.h"
#include "Dielectric.h"
#include "Material.h"
#include "VDBVolume.h"
#include "VDBVolumeManager.h"
#include "VolumeShader.h"
#include "Triangle.h"
#include "Mesh.h"
#include "AABB.h"
#include "Ray.h"
#include "Hittable.h"
#include "HittableList.h"
#include "ParallelBVHNode.h"
#include "AnimatedObject.h"
#include "AnimationController.h"
#include "OzzRuntime.h"
#include "AnimationNodes.h"
#include "scene_data.h"

// Unified rendering system for CPU/GPU parity
#include "unified_types.h"
#include "unified_brdf.h"
#include "unified_light_sampling.h"
#include "unified_converters.h"
#include "MaterialManager.h"
#include "PBRMaterialSnapshot.h"
#include "CameraPresets.h"
#include "TerrainManager.h"
#include "WaterSystem.h"      // For water/FFT keyframe animation
#include "WaterMaterialSync.h"
#include "InstanceManager.h"  // For wind animation in render_Animation
#include "water_shaders_cpu.h"  // CPU water shader functions
#include "HittableInstance.h"
#include "VolumetricRenderer.h"
#include "HitMaterialResolver.h"
#include "AtmosphereLUT.h"       // Required for CPU transmittance sampling
#include "Hair/HairBSDF.h"       // Hair BSDF for shading
#include "FoliageWindSystem.h"
#include <Backend/VulkanBackend.h>
#include "Backend/IViewportBackend.h"

namespace {
float rt_cloud_lerp(float a, float b, float t) {
    return a + (b - a) * t;
}

float rt_cloud_smoothstep(float edge0, float edge1, float x) {
    float t = std::clamp((x - edge0) / std::max(edge1 - edge0, 1e-6f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float rt_cloud_hash(const Vec3& p) {
    const float d = p.x * 127.1f + p.y * 311.7f + p.z * 74.7f;
    const float v = std::sin(d) * 43758.5453f;
    return v - std::floor(v);
}

float rt_cloud_noise(Vec3 p) {
    Vec3 i(std::floor(p.x), std::floor(p.y), std::floor(p.z));
    Vec3 f(p.x - i.x, p.y - i.y, p.z - i.z);
    f.x = f.x * f.x * (3.0f - 2.0f * f.x);
    f.y = f.y * f.y * (3.0f - 2.0f * f.y);
    f.z = f.z * f.z * (3.0f - 2.0f * f.z);

    const float n000 = rt_cloud_hash(i + Vec3(0, 0, 0));
    const float n100 = rt_cloud_hash(i + Vec3(1, 0, 0));
    const float n010 = rt_cloud_hash(i + Vec3(0, 1, 0));
    const float n110 = rt_cloud_hash(i + Vec3(1, 1, 0));
    const float n001 = rt_cloud_hash(i + Vec3(0, 0, 1));
    const float n101 = rt_cloud_hash(i + Vec3(1, 0, 1));
    const float n011 = rt_cloud_hash(i + Vec3(0, 1, 1));
    const float n111 = rt_cloud_hash(i + Vec3(1, 1, 1));

    const float nx00 = rt_cloud_lerp(n000, n100, f.x);
    const float nx10 = rt_cloud_lerp(n010, n110, f.x);
    const float nx01 = rt_cloud_lerp(n001, n101, f.x);
    const float nx11 = rt_cloud_lerp(n011, n111, f.x);
    return rt_cloud_lerp(rt_cloud_lerp(nx00, nx10, f.y), rt_cloud_lerp(nx01, nx11, f.y), f.z);
}

float rt_cloud_fbm(Vec3 p, int octaves) {
    float value = 0.0f;
    float amp = 0.5f;
    for (int i = 0; i < octaves; ++i) {
        value += amp * rt_cloud_noise(p);
        p = p * 2.0f;
        amp *= 0.5f;
    }
    return value;
}

bool rt_weather_active(const WeatherParams& weather) {
    return weather.enabled != 0 && weather.type != WEATHER_NONE &&
           weather.intensity > 0.0f && weather.density > 0.0f;
}

bool rt_weather_visual_active(const WeatherParams& weather) {
    return rt_weather_active(weather) && weather.visual_mode != WEATHER_VISUAL_SURFACE_ONLY;
}

bool rt_weather_surface_active(const WeatherParams& weather) {
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

float rt_weather_surface_mask(const WeatherParams& weather, const Vec3& pos, const Vec3& normal);
float rt_weather_surface_geometric_support(const WeatherParams& weather, const Vec3& geomNormal);
float rt_weather_surface_settling(const WeatherParams& weather, const Vec3& pos, const Vec3& normal, const Vec3& geomNormal);

Vec3 rt_weather_surface_support_normal(const Vec3& normal, const Vec3& geomNormal) {
    const Vec3 baseNormal = normal.normalize();
    const Vec3 macroNormal = geomNormal.normalize();
    return Vec3::lerp(macroNormal, baseNormal, 0.62f).normalize();
}

float rt_weather_surface_accumulation(const WeatherParams& weather, const Vec3& pos, const Vec3& normal, const Vec3& geomNormal) {
    if (weather.type != WEATHER_SNOW && weather.type != WEATHER_DUST) {
        return 0.0f;
    }

    const float baseAccum = std::clamp(weather.surface_accumulation_output, 0.0f, 1.0f);
    const float intensity = std::clamp(weather.intensity, 0.0f, 1.0f);
    const float density = std::clamp(weather.density, 0.0f, 1.0f);
    const Vec3 supportNormal = rt_weather_surface_support_normal(normal, geomNormal);
    const float geomSupport = rt_weather_surface_geometric_support(weather, supportNormal);
    const float intensityResponse = 0.80f + intensity * 0.70f;
    const float densityResponse = 0.35f + density * 1.15f;
    const float typeBoost = (weather.type == WEATHER_SNOW) ? 1.10f : 0.90f;
    const float directAccum = baseAccum * intensityResponse * rt_weather_surface_mask(weather, pos, normal) * densityResponse * typeBoost * geomSupport;
    const float settling = rt_weather_surface_settling(weather, pos, normal, geomNormal);
    return std::clamp(directAccum + (1.0f - std::clamp(directAccum, 0.0f, 1.0f)) * settling, 0.0f, 1.0f);
}

float rt_weather_surface_geometric_support(const WeatherParams& weather, const Vec3& geomNormal) {
    const Vec3 macroNormal = geomNormal.normalize();
    float support = 0.0f;
    if (weather.type == WEATHER_SNOW) {
        support = std::clamp((static_cast<float>(macroNormal.y) - 0.02f) / 0.72f, 0.0f, 1.0f);
        support *= support;
    } else {
        support = std::clamp((static_cast<float>(macroNormal.y) - 0.02f) / 0.78f, 0.0f, 1.0f);
    }
    return support * support * (3.0f - 2.0f * support);
}

float rt_weather_surface_height(const WeatherParams& weather, const Vec3& pos) {
    const float scale = std::max(weather.precipitation_scale, 0.1f);
    const float heightBoost = 0.25f + std::clamp(weather.surface_height_output, 0.0f, 1.0f) * 3.75f;
    if (weather.type == WEATHER_SNOW) {
        Vec3 wind(weather.wind_direction.x, weather.wind_direction.y, weather.wind_direction.z);
        Vec3 windXZ(wind.x, 0.0f, wind.z);
        Vec3 along = windXZ.length_squared() > 1e-8f ? windXZ.normalize() : Vec3(1.0f, 0.0f, 0.0f);
        Vec3 across(-along.z, 0.0f, along.x);
        const float u = pos.x * along.x + pos.z * along.z;
        const float v = pos.x * across.x + pos.z * across.z;
        const Vec3 p(u * scale * 0.12f, pos.y * scale * 0.03f, v * scale * 0.12f);
        const float broad = rt_cloud_noise(p * 0.55f + Vec3(17.3f, 9.1f, 41.7f));
        float drift = 1.0f - std::abs(rt_cloud_noise(Vec3(p.x * 1.45f, p.y * 0.8f, p.z * 0.58f) + Vec3(3.7f, 29.4f, 11.8f)) * 2.0f - 1.0f);
        drift *= drift;
        const float clumps = 1.0f - std::abs(rt_cloud_noise(p * 2.90f + Vec3(61.2f, 7.5f, 18.9f)) * 2.0f - 1.0f);
        const float micro = rt_cloud_noise(p * 7.40f + Vec3(8.3f, 51.7f, 27.4f));
        return (broad * 0.22f + drift * 0.36f + clumps * 0.27f + micro * 0.15f) * heightBoost;
    }

    const Vec3 p = pos * (scale * 0.18f);
    const float wisps = rt_cloud_noise(p + Vec3(19.7f, 5.3f, 27.1f));
    const float grain = rt_cloud_noise(p * 2.85f + Vec3(4.1f, 37.8f, 12.4f));
    const float streak = rt_cloud_noise(p * 1.65f + Vec3(44.5f, 14.2f, 7.6f));
    return (wisps * 0.30f + grain * 0.45f + streak * 0.25f) * heightBoost;
}

float rt_weather_surface_settling(const WeatherParams& weather, const Vec3& pos, const Vec3& normal, const Vec3& geomNormal) {
    if (weather.type != WEATHER_SNOW && weather.type != WEATHER_DUST) {
        return 0.0f;
    }

    const float settlingAmount = std::clamp(weather.surface_settling_output, 0.0f, 1.0f);
    if (settlingAmount <= 1e-4f) {
        return 0.0f;
    }

    const Vec3 shadingNormal = normal.normalize();
    const Vec3 macroNormal = rt_weather_surface_support_normal(normal, geomNormal);
    const float support = rt_weather_surface_geometric_support(weather, macroNormal);
    const float supportGate = std::clamp((support - 0.02f) / 0.58f, 0.0f, 1.0f);
    if (supportGate <= 1e-4f) {
        return 0.0f;
    }
    const float exposure = rt_weather_surface_mask(weather, pos, shadingNormal);
    const float cavity = std::clamp((1.0f - static_cast<float>(Vec3::dot(shadingNormal, macroNormal))) * 3.8f + (1.0f - support) * 0.10f, 0.0f, 1.0f);
    const Vec3 wind = Vec3(weather.wind_direction.x, weather.wind_direction.y, weather.wind_direction.z);
    const Vec3 windFlat(wind.x, 0.0f, wind.z);
    const Vec3 leeDir = windFlat.length_squared() > 1e-8f ? Vec3(-windFlat.x, 0.28f, -windFlat.z).normalize() : Vec3(0.0f, 1.0f, 0.0f);
    const float lee = std::clamp(static_cast<float>(Vec3::dot(macroNormal, leeDir)) * 0.85f + cavity * 0.35f, 0.0f, 1.0f);
    const float shelter = std::clamp((1.0f - exposure) * 0.52f + cavity * 0.26f + (1.0f - support) * 0.22f + lee * 0.42f, 0.0f, 1.0f);
    const float pocketNoise = rt_cloud_noise(pos * 0.085f + Vec3(31.4f, 9.7f, 54.2f));
    const float pocketMask = std::clamp(cavity * 0.92f + pocketNoise * 0.26f, 0.0f, 1.0f);
    const float slopeBase = std::clamp((support - 0.16f) / 0.54f, 0.0f, 1.0f);
    const float density = std::clamp(weather.density, 0.0f, 1.0f);
    const float typeBoost = (weather.type == WEATHER_SNOW) ? 1.34f : 1.04f;
    const float anchor = std::max(pocketMask, slopeBase * 0.30f + cavity * 0.40f + lee * 0.30f);
    return std::clamp(settlingAmount * supportGate * shelter * anchor * (0.76f + density * 1.10f) * typeBoost, 0.0f, 1.0f);
}

Vec3 rt_weather_surface_normal(const WeatherParams& weather, const Vec3& pos, const Vec3& normal, const Vec3& geomNormal) {
    const Vec3 baseNormal = normal.normalize();
    if (!rt_weather_surface_active(weather)) return baseNormal;
    if (weather.type != WEATHER_SNOW && weather.type != WEATHER_DUST) return baseNormal;

    const float accumulation = rt_weather_surface_accumulation(weather, pos, baseNormal, geomNormal);
    if (accumulation <= 1e-4f) return baseNormal;

    const Vec3 supportNormal = rt_weather_surface_support_normal(baseNormal, geomNormal);
    const float geomSupport = rt_weather_surface_geometric_support(weather, supportNormal);
    if (geomSupport <= 1e-4f) return baseNormal;

    const float settling = rt_weather_surface_settling(weather, pos, baseNormal, geomNormal);
    const float detailCapture = 0.45f + 0.55f * std::clamp((static_cast<float>(baseNormal.y) - 0.04f) / 0.82f, 0.0f, 1.0f);
    const float heightResponse = 0.12f + std::clamp(weather.surface_height_output, 0.0f, 1.0f) * 0.95f;
    const float buildup = std::clamp(accumulation + settling * 0.85f, 0.0f, 1.0f);
    const float normalStrength = buildup * detailCapture * heightResponse * (weather.type == WEATHER_SNOW ? 0.42f : 0.15f);
    if (normalStrength <= 1e-4f) return baseNormal;

    const Vec3 wind = Vec3(weather.wind_direction.x, weather.wind_direction.y, weather.wind_direction.z);
    Vec3 tangent = wind - baseNormal * Vec3::dot(wind, baseNormal);
    if (tangent.length_squared() <= 1e-8f) {
        const Vec3 helper = std::abs(baseNormal.y) < 0.999f ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
        tangent = Vec3::cross(helper, baseNormal);
    }
    tangent = tangent.normalize();
    Vec3 bitangent = Vec3::cross(baseNormal, tangent).normalize();
    tangent = Vec3::cross(bitangent, baseNormal).normalize();

    const float sampleStep = (weather.type == WEATHER_SNOW ? 0.62f : 0.90f) / std::max(weather.precipitation_scale, 0.35f);
    const float heightCenter = rt_weather_surface_height(weather, pos);
    const float heightT = rt_weather_surface_height(weather, pos + tangent * sampleStep);
    const float heightB = rt_weather_surface_height(weather, pos + bitangent * sampleStep);
    const float gradT = std::clamp((heightT - heightCenter) / sampleStep, -0.28f, 0.28f);
    const float gradB = std::clamp((heightB - heightCenter) / sampleStep, -0.28f, 0.28f);

    Vec3 perturbed = (baseNormal - tangent * (gradT * normalStrength) - bitangent * (gradB * normalStrength)).normalize();
    if (Vec3::dot(perturbed, baseNormal) < 0.05f) {
        perturbed = Vec3::lerp(baseNormal, perturbed, 0.35f).normalize();
    }
    if (Vec3::dot(perturbed, supportNormal) < 0.55f) {
        perturbed = Vec3::lerp(supportNormal, perturbed, 0.05f).normalize();
    }
    return perturbed;
}

float rt_weather_surface_mask(const WeatherParams& weather, const Vec3& pos, const Vec3& normal) {
    float up = std::clamp((static_cast<float>(normal.y) - 0.08f) / 0.82f, 0.0f, 1.0f);
    up = up * up * (3.0f - 2.0f * up);
    const Vec3 wind = Vec3(weather.wind_direction.x, weather.wind_direction.y, weather.wind_direction.z);
    const Vec3 windDir = wind.length_squared() > 1e-8f ? wind.normalize() : Vec3(1.0f, 0.0f, 0.0f);
    const float windAmount = std::clamp(weather.wind_speed / 35.0f, 0.0f, 1.0f);
    const Vec3 incoming = (Vec3(0.0f, 1.0f, 0.0f) - windDir * windAmount).normalize();
    const float windFacing = std::clamp(static_cast<float>(Vec3::dot(normal.normalize(), incoming)), 0.0f, 1.0f);
    const float exposure = std::clamp(up * (1.0f - windAmount * 0.78f) + windFacing * (0.12f + windAmount * 1.22f), 0.0f, 1.0f);
    const float scale = std::max(weather.precipitation_scale, 0.1f);
    const Vec3 p = pos * (scale * 0.22f) + Vec3(13.1f, 47.2f, 5.7f);
    const float n = rt_cloud_fbm(p, 3);
    const float breakup = std::clamp(n * 1.45f - 0.15f, 0.0f, 1.0f);
    return exposure * (0.42f + 0.58f * breakup);
}

void rt_apply_weather_surface(
    const WeatherParams& weather,
    const Vec3& pos,
    const Vec3& normal,
    const Vec3& geomNormal,
    Vec3& albedo,
    float& roughness,
    float& metallic,
    float& clearcoat,
    float& clearcoatRoughness
) {
    if (!rt_weather_surface_active(weather)) return;

    const float exposed = rt_weather_surface_mask(weather, pos, normal);
    if (weather.type == WEATHER_RAIN) {
        const float wet = std::clamp(weather.surface_wetness_output, 0.0f, 1.0f) *
                          (0.35f + 0.65f * exposed);
        albedo = Vec3::lerp(albedo, albedo * 0.50f, wet * 0.62f);
        roughness = std::max(0.012f, roughness * (1.0f - wet * 0.78f));
        metallic = std::max(0.0f, metallic - wet * 0.05f);
        clearcoat = std::max(clearcoat, wet * 0.72f);
        clearcoatRoughness = std::min(clearcoatRoughness, std::max(0.006f, 0.045f - wet * 0.030f));
    } else if (weather.type == WEATHER_SNOW) {
        const float acc = rt_weather_surface_accumulation(weather, pos, normal, geomNormal);
        const float settling = rt_weather_surface_settling(weather, pos, normal, geomNormal);
        const float heightLift = std::clamp(weather.surface_height_output, 0.0f, 1.0f);
        const float cover = std::clamp(acc + settling * 0.84f + heightLift * (acc * 0.08f + settling * 0.30f), 0.0f, 1.0f);
        const float sparkle = std::clamp(rt_cloud_noise(pos * 6.5f + Vec3(19.0f, 3.0f, 41.0f)) * acc, 0.0f, 1.0f);
        albedo = Vec3::lerp(albedo, Vec3(0.88f, 0.91f, 0.96f) + Vec3(0.08f) * sparkle, cover * 0.74f);
        roughness = std::clamp(roughness + cover * (0.42f + heightLift * 0.10f) - sparkle * 0.10f, 0.02f, 1.0f);
        metallic *= (1.0f - cover * 0.8f);
    } else if (weather.type == WEATHER_DUST) {
        const float acc = rt_weather_surface_accumulation(weather, pos, normal, geomNormal);
        const float settling = rt_weather_surface_settling(weather, pos, normal, geomNormal);
        const float heightLift = std::clamp(weather.surface_height_output, 0.0f, 1.0f);
        const float cover = std::clamp(acc + settling * 0.90f + heightLift * settling * 0.22f, 0.0f, 1.0f);
        albedo = Vec3::lerp(albedo, Vec3(0.58f, 0.46f, 0.30f), cover * 0.58f);
        roughness = std::min(1.0f, roughness + cover * (0.38f + heightLift * 0.08f));
        metallic *= (1.0f - cover * 0.55f);
    }
}

Vec3 rt_weather_tint_color(const WeatherParams& weather) {
    switch (weather.type) {
        case WEATHER_RAIN: return Vec3(0.50f, 0.56f, 0.62f);
        case WEATHER_SNOW: return Vec3(0.86f, 0.90f, 0.96f);
        case WEATHER_DUST: return Vec3(0.74f, 0.58f, 0.38f);
        case WEATHER_MIST: return Vec3(0.70f, 0.76f, 0.82f);
        default: return Vec3(0.0f);
    }
}

Vec3 rt_apply_weather_atmosphere(const WeatherParams& weather, const Vec3& color, const Vec3& rayDir, float distance) {
    if (!rt_weather_visual_active(weather)) return color;

    const float vis = std::max(0.02f, std::clamp(weather.visibility, 0.0f, 1.0f));
    const float sigma = weather.intensity * weather.density * (0.00018f + (1.0f - vis) * 0.00042f);
    float amount = std::clamp(1.0f - std::exp(-std::max(distance, 0.0f) * sigma), 0.0f, 0.82f);

    Vec3 tint = rt_weather_tint_color(weather);
    if (weather.type == WEATHER_RAIN) tint *= 0.72f;
    if (weather.type == WEATHER_DUST) tint *= 1.12f;

    const Vec3 wind = Vec3(weather.wind_direction.x, weather.wind_direction.y, weather.wind_direction.z);
    const Vec3 windDir = wind.length_squared() > 1e-8 ? wind.normalize() : Vec3(1.0f, 0.0f, 0.0f);
    const float forward = std::pow(std::max(0.0f, Vec3::dot(rayDir.normalize(), windDir)), 4.0f);
    amount = std::min(0.90f, amount + forward * weather.intensity * weather.density * 0.08f);
    return Vec3::lerp(color, tint, amount);
}

float rt_precip_smoothstep(float edge0, float edge1, float x) {
    float t = 0.0f;
    if (edge1 >= edge0) {
        t = std::clamp((x - edge0) / std::max(edge1 - edge0, 1e-6f), 0.0f, 1.0f);
    } else {
        t = std::clamp((edge0 - x) / std::max(edge0 - edge1, 1e-6f), 0.0f, 1.0f);
    }
    return t * t * (3.0f - 2.0f * t);
}

float rt_precip_hash(float x, float y) {
    const float v = std::sin(x * 127.1f + y * 311.7f) * 43758.5453123f;
    return v - std::floor(v);
}

float rt_precip_noise(float x, float y) {
    const float cell_x = std::floor(x);
    const float cell_y = std::floor(y);
    const float frac_x = x - cell_x;
    const float frac_y = y - cell_y;
    const float smooth_x = frac_x * frac_x * (3.0f - 2.0f * frac_x);
    const float smooth_y = frac_y * frac_y * (3.0f - 2.0f * frac_y);
    const float n00 = rt_precip_hash(cell_x, cell_y);
    const float n10 = rt_precip_hash(cell_x + 1.0f, cell_y);
    const float n01 = rt_precip_hash(cell_x, cell_y + 1.0f);
    const float n11 = rt_precip_hash(cell_x + 1.0f, cell_y + 1.0f);
    const float nx0 = rt_cloud_lerp(n00, n10, smooth_x);
    const float nx1 = rt_cloud_lerp(n01, n11, smooth_x);
    return rt_cloud_lerp(nx0, nx1, smooth_y);
}

float rt_precip_line(float u, float v, float windX, float windY, float time, float density, float scale) {
    const float wind_drive = 1.0f + density * 0.85f;
    const float px = u * 74.0f / scale + windX * time * (6.5f + density * 4.0f);
    const float py = v * 22.0f / scale - time * (34.0f + density * 10.0f) + windY * time * (7.5f + density * 4.5f);
    const float cx = std::floor(px);
    const float cy = std::floor(py);
    const float fx = px - cx;
    const float fy = py - cy;
    const float rnd = rt_precip_hash(cx, cy);
    const float spawn = rt_precip_smoothstep(0.94f - density * 0.10f, 1.0f, rnd);
    const float x = std::abs(fx - 0.5f - (rnd - 0.5f) * (0.35f + 0.18f * wind_drive));
    return spawn * rt_precip_smoothstep(0.055f - density * 0.016f, 0.0f, x) * rt_precip_smoothstep(1.0f, 0.04f, fy);
}

float rt_precip_flake(float u, float v, float windX, float windY, float time, float density, float scale) {
    const float px = u * 46.0f / scale + windX * time * (2.6f + density * 2.5f);
    const float py = v * 30.0f / scale - time * (2.8f + density * 0.9f) + windY * time * (1.1f + density * 0.9f);
    const float cx = std::floor(px);
    const float cy = std::floor(py);
    const float fx = px - cx;
    const float fy = py - cy;
    const float rnd = rt_precip_hash(cx, cy);
    const float spawn = rt_precip_smoothstep(0.84f - density * 0.22f, 1.0f, rnd);
    const float ox = rt_precip_hash(cx + 13.7f, cy + 13.7f) + std::sin(time * 1.7f + rnd * 6.2831f) * (0.12f + density * 0.08f) + windX * 0.12f;
    const float oy = rt_precip_hash(cx + 41.3f, cy + 41.3f) + windY * 0.05f;
    const float radius = rt_cloud_lerp(0.035f, 0.120f + density * 0.025f, rt_precip_hash(cx + 7.1f, cy + 7.1f));
    const float dx = fx - ox;
    const float dy = fy - oy;
    return spawn * rt_precip_smoothstep(radius, 0.0f, std::sqrt(dx * dx + dy * dy));
}

float rt_precip_dust(float u, float v, float windX, float windY, float time, float density, float scale) {
    const float advected_u = u / scale + windX * time * (0.24f + density * 0.08f);
    const float advected_v = v / scale + windY * time * (0.18f + density * 0.10f);
    const float streaks = rt_precip_noise(advected_u * 18.0f, advected_v * 7.5f);
    const float wisps = rt_precip_noise(advected_u * 33.0f + 19.4f, advected_v * 12.0f + 7.1f);
    const float grain = rt_precip_noise(advected_u * 95.0f + 3.7f, advected_v * 42.0f + 17.3f);
    const float elongated = rt_precip_smoothstep(0.52f, 0.98f, streaks * 0.72f + wisps * 0.28f);
    const float soft_grain = rt_precip_smoothstep(0.38f, 0.88f, grain);
    return (elongated * (0.72f + density * 0.28f) + soft_grain * 0.18f) * density;
}

Vec3 rt_apply_weather_precipitation_overlay(
    const WeatherParams& weather,
    const Vec3& color,
    const Vec3& rayDirIn,
    float distance,
    float time
) {
    if (!rt_weather_visual_active(weather)) return color;

    const float density = std::clamp(std::pow(std::clamp(weather.intensity, 0.0f, 1.0f), 0.82f) *
        (0.28f + std::clamp(weather.density, 0.0f, 1.0f) * 1.22f), 0.0f, 1.0f);
    if (density <= 0.001f) return color;

    const Vec3 rayDir = rayDirIn.normalize();
    constexpr float kPi = 3.14159265358979323846f;
    const float u = std::atan2(static_cast<float>(rayDir.z), static_cast<float>(rayDir.x)) / (2.0f * kPi) + 0.5f;
    const float v = std::acos(std::clamp(static_cast<float>(rayDir.y), -1.0f, 1.0f)) / kPi;
    const float scale = std::max(weather.precipitation_scale, 0.25f);
    const float depthFade = rt_precip_smoothstep(0.6f, 18.0f, std::max(distance, 0.0f));
    const float horizonFade = std::clamp(1.0f - std::max(static_cast<float>(rayDir.y), 0.0f) * 0.35f, 0.35f, 1.0f);
    const Vec3 wind(weather.wind_direction.x, weather.wind_direction.y, weather.wind_direction.z);
    const float windLen2 = static_cast<float>(wind.x * wind.x + wind.z * wind.z);
    const float windAmount = std::clamp(weather.wind_speed / 35.0f, 0.0f, 1.0f);
    const float invWindLen = windLen2 > 1e-8f ? 1.0f / std::sqrt(windLen2) : 0.0f;
    const float windX = (windLen2 > 1e-8f ? static_cast<float>(wind.x) * invWindLen : 1.0f) * windAmount;
    const float windY = (windLen2 > 1e-8f ? static_cast<float>(wind.z) * invWindLen : 0.0f) * windAmount;
    const float windVisual = 0.65f + windAmount * 0.9f;
    Vec3 result = color;
    const Vec3 tint = rt_weather_tint_color(weather);

    if (weather.type == WEATHER_RAIN) {
        const float amount = std::min(1.0f, rt_precip_line(u, v, windX, windY, time, density, scale) * (0.85f + density * 0.85f + windAmount * 0.35f));
        result = Vec3::lerp(result, result * 0.84f, amount * density * 0.24f * depthFade);
        result += Vec3(0.45f, 0.55f, 0.66f) * amount * density * 0.28f * depthFade * horizonFade * windVisual;
    } else if (weather.type == WEATHER_SNOW) {
        const float amount = std::min(1.0f, rt_precip_flake(u, v, windX, windY, time, density, scale) * (0.80f + density * 0.95f + windAmount * 0.22f));
        result = Vec3::lerp(result, tint, amount * density * 0.34f * depthFade * horizonFade);
        result += Vec3(0.85f, 0.92f, 1.0f) * amount * density * 0.16f * depthFade * windVisual;
    } else if (weather.type == WEATHER_DUST) {
        const float amount = std::min(1.0f, rt_precip_dust(u, v, windX, windY, time, density, scale) * (0.75f + density * 1.05f + windAmount * 0.45f));
        result = Vec3::lerp(result, tint, amount * 0.16f * depthFade * windVisual);
        result += tint * amount * 0.075f * depthFade * windVisual;
    } else if (weather.type == WEATHER_MIST) {
        const float amount = std::min(1.0f, rt_precip_dust(u, v, windX * 0.25f, windY * 0.25f, time * 0.35f, density, scale * 1.4f) * (0.65f + density * 0.85f));
        result = Vec3::lerp(result, tint, amount * 0.11f * depthFade);
    }

    return result;
}

float rt_sample_procedural_cloud_cpu(const Vec3& local_p, const VDBVolume* vdb, const NishitaSkyParams& n) {
    if (!vdb) return 0.0f;
    const Vec3 bmin = vdb->getLocalBoundsMin();
    const Vec3 bmax = vdb->getLocalBoundsMax();
    const Vec3 span(std::max(1e-5f, bmax.x - bmin.x), std::max(1e-5f, bmax.y - bmin.y), std::max(1e-5f, bmax.z - bmin.z));
    Vec3 norm((local_p.x - bmin.x) / span.x, (local_p.y - bmin.y) / span.y, (local_p.z - bmin.z) / span.z);
    if (norm.x < 0.0f || norm.x > 1.0f || norm.y < 0.0f || norm.y > 1.0f || norm.z < 0.0f || norm.z > 1.0f) return 0.0f;

    float coverage = n.cloud_coverage;
    float density_ref = n.cloud_density;
    float scale = n.cloud_scale;
    if (n.cloud_layer2_enabled && (!n.clouds_enabled || n.cloud2_density > density_ref)) {
        coverage = n.cloud2_coverage;
        density_ref = n.cloud2_density;
        scale = n.cloud2_scale;
    }

    const float base_scale = std::max(8.0f, std::min(72.0f, 10.0f / std::max(0.1f, scale)));
    Vec3 cloud_pos(norm.x * base_scale + n.cloud_offset_x * 0.00002f, norm.y * 1.35f, norm.z * base_scale + n.cloud_offset_z * 0.00002f);
    cloud_pos = cloud_pos + Vec3(n.cloud_seed * 0.137f, n.cloud_seed * 0.317f, n.cloud_seed * 0.719f);
    const float detail = std::clamp(n.cloud_detail, 0.0f, 1.0f);
    const float erosion = std::clamp(1.0f - coverage, 0.0f, 1.0f);

    const float warp_x = rt_cloud_fbm(Vec3(cloud_pos.x * 0.38f, cloud_pos.y * 0.16f, cloud_pos.z * 0.38f) + Vec3(11, 0, 7), 2) - 0.5f;
    const float warp_z = rt_cloud_fbm(Vec3(cloud_pos.x * 0.38f, cloud_pos.y * 0.16f, cloud_pos.z * 0.38f) + Vec3(41, 3, 23), 2) - 0.5f;
    const Vec3 warped = cloud_pos + Vec3(warp_x * 1.35f, 0.0f, warp_z * 1.35f);

    const float base = rt_cloud_fbm(Vec3(warped.x * 0.52f, warped.y * 0.28f, warped.z * 0.52f), 4);
    const float billow = 1.0f - std::abs(rt_cloud_fbm(Vec3(warped.x * 1.15f, warped.y * 0.5f, warped.z * 1.15f) + Vec3(17, 3, 11), 4) * 2.0f - 1.0f);
    const float detail_noise = rt_cloud_fbm(warped * rt_cloud_lerp(2.8f, 7.0f, detail) + Vec3(31, 7, 19), 2);
    const float puffy = rt_cloud_smoothstep(0.32f, 0.88f, billow);
    const float cumulus = std::clamp((density_ref - 0.38f) / 0.85f, 0.0f, 1.0f);
    float shape = rt_cloud_lerp(base, base * 0.45f + puffy * 0.75f, rt_cloud_lerp(0.55f, 0.9f, cumulus));
    shape -= detail_noise * rt_cloud_lerp(0.06f, 0.28f, erosion);

    const float threshold = rt_cloud_lerp(0.80f, 0.26f, std::clamp(coverage, 0.0f, 1.0f)) - cumulus * 0.08f;
    float d = std::max((shape - threshold) / std::max(1.0f - threshold, 1e-4f), 0.0f);
    const float bottom = rt_cloud_smoothstep(rt_cloud_lerp(0.12f, 0.04f, cumulus), rt_cloud_lerp(0.42f, 0.24f, cumulus), norm.y);
    const float top = 1.0f - rt_cloud_smoothstep(rt_cloud_lerp(0.72f, 0.82f, cumulus), 1.02f, norm.y);
    const float dome = rt_cloud_lerp(1.0f, rt_cloud_smoothstep(0.08f, 0.58f, norm.y) * (1.0f - rt_cloud_smoothstep(0.88f, 1.04f, norm.y)) + 0.25f, cumulus);
    const Vec3 edge(0.5f - std::abs(norm.x - 0.5f), 0.5f - std::abs(norm.y - 0.5f), 0.5f - std::abs(norm.z - 0.5f));
    const float edge_falloff = rt_cloud_smoothstep(0.0f, 0.08f, std::min(edge.x, edge.z));
    return d * rt_cloud_lerp(d, std::sqrt(std::max(d, 0.0f)), cumulus * 0.55f) * bottom * top * dome * edge_falloff * rt_cloud_lerp(4.6f, 3.4f, cumulus);
}
}

bool Renderer::isCudaAvailable() {
    try {
        oidn::DeviceRef testDevice = oidn::newDevice(oidn::DeviceType::CUDA);
        testDevice.commit();
        return true; // CUDA destekleniyor
    }
    catch (const std::exception& e) {
        return false; // CUDA desteklenmiyor
    }
}
// Helper to initialize OIDN device once
void Renderer::initOIDN() {
    if (oidnInitialized) return;

    // Device-selection cascade for the host (CPU-visible buffer) denoise path.
    // The denoise itself runs on whichever device we bind here; for a GPU device
    // OIDN handles the host<->device buffer copies internally, so AMD/Intel cards
    // with Vulkan-RT but no CUDA still get a GPU-accelerated denoise instead of
    // the much slower CPU path. Priority:
    //   1. CUDA — NVIDIA (the only backend that also feeds the zero-copy interop)
    //   2. HIP  — AMD GPUs
    //   3. SYCL — Intel Arc / Xe GPUs
    //   4. CPU  — universal fallback
    // OIDN ships a separate device DLL per backend; a missing DLL (or absent
    // runtime, e.g. cudart on an AMD box) just makes newDevice()/commit() fail,
    // so each attempt is wrapped and we fall through to the next.
    auto tryDevice = [this](oidn::DeviceType type, const char* label) -> bool {
        try {
            oidn::DeviceRef dev = oidn::newDevice(type);
            dev.commit();
            const char* errMsg = nullptr;
            if (dev.getError(errMsg) != oidn::Error::None) {
                SCENE_LOG_WARN(std::string("[OIDN] ") + label + " commit failed: "
                               + (errMsg ? errMsg : "unknown error"));
                return false;
            }
            oidnDevice = dev;
            oidnInitialized = true;
            SCENE_LOG_INFO(std::string("[OIDN] Initialized with ") + label + ".");
            return true;
        }
        catch (const std::exception& e) {
            SCENE_LOG_WARN(std::string("[OIDN] ") + label + " unavailable: " + e.what());
            return false;
        }
    };

    if (tryDevice(oidn::DeviceType::CUDA, "CUDA")) return;
    if (tryDevice(oidn::DeviceType::HIP,  "HIP (AMD)")) return;
    if (tryDevice(oidn::DeviceType::SYCL, "SYCL (Intel)")) return;
    if (tryDevice(oidn::DeviceType::CPU,  "CPU")) return;

    SCENE_LOG_ERROR("[OIDN] No usable denoiser device (CUDA/HIP/SYCL/CPU all failed).");
}

void Renderer::applyOIDNDenoising(SDL_Surface* surface, int numThreads, bool denoise, float blend) {
    if (!surface) return;
    const int w = surface->w;
    const int h = surface->h;
    const size_t pixelCount = (size_t)w * (size_t)h;
    std::vector<float> linearColor(pixelCount * 3);

    const SDL_PixelFormat* fmt = surface->format;
    Uint32* pixels = static_cast<Uint32*>(surface->pixels);
    const Uint32 rMask = fmt->Rmask, gMask = fmt->Gmask, bMask = fmt->Bmask, aMask = fmt->Amask;
    const Uint8 rShift = fmt->Rshift, gShift = fmt->Gshift, bShift = fmt->Bshift;
    const float inv255 = 1.0f / 255.0f;

    // Parallel sRGB→linear decode (gamma-2 approximation: x*x)
    {
        float* __restrict dst = linearColor.data();
        Uint32* __restrict pxBase = pixels;
        std::for_each_n(std::execution::par_unseq,
            pxBase, pixelCount,
            [=](Uint32& px) {
                const size_t i = static_cast<size_t>(&px - pxBase);
                float r = ((px & rMask) >> rShift) * inv255;
                float g = ((px & gMask) >> gShift) * inv255;
                float b = ((px & bMask) >> bShift) * inv255;
                const size_t idx = i * 3;
                dst[idx]     = r * r;
                dst[idx + 1] = g * g;
                dst[idx + 2] = b * b;
            });
    }

    OIDNFrameData frame;
    frame.width = w;
    frame.height = h;
    frame.color = linearColor.data();

    std::vector<float> denoised;
    if (!applyOIDNDenoising(frame, blend, denoised)) {
        return;
    }

    // Parallel linear→sRGB encode + pack back into SDL surface
    //
    // [GAMMA FIX] The decode block above used r = (x/255)^2 (gamma-2
    // approximation of sRGB→linear). The original encode here clamped+
    // packed the *linear* value straight into the 8-bit pixel WITHOUT
    // applying the inverse — so the saved PNG (and viewport display) was
    // treated as sRGB while actually carrying linear values, causing the
    // entire image to read significantly darker than the pre-denoise
    // surface. Apply the matching inverse sqrt so the roundtrip is
    // gamma-neutral.
    {
        const float* __restrict src = denoised.data();
        Uint32* __restrict pxBase = pixels;
        std::for_each_n(std::execution::par_unseq,
            pxBase, pixelCount,
            [=](Uint32& px) {
                const size_t i = static_cast<size_t>(&px - pxBase);
                const size_t idx = i * 3;
                float r = std::max(src[idx],     0.0f);
                float g = std::max(src[idx + 1], 0.0f);
                float b = std::max(src[idx + 2], 0.0f);

                // Inverse of the gamma-2 decode (r = x*x). Matches the
                // approximation used at line 607 so denoiser-on output
                // visually matches denoiser-off.
                r = std::sqrt(std::min(r, 1.0f));
                g = std::sqrt(std::min(g, 1.0f));
                b = std::sqrt(std::min(b, 1.0f));

                Uint8 ri = static_cast<Uint8>(r * 255.0f + 0.5f);
                Uint8 gi = static_cast<Uint8>(g * 255.0f + 0.5f);
                Uint8 bi = static_cast<Uint8>(b * 255.0f + 0.5f);

                Uint32 alpha = px & aMask;
                px = alpha
                    | ((Uint32)ri << rShift)
                    | ((Uint32)gi << gShift)
                    | ((Uint32)bi << bShift);
            });
    }
}

bool Renderer::applyOIDNDenoising(const OIDNFrameData& frame, float blend, std::vector<float>& output,
                                  int quality) {
    if (!frame.color || frame.width <= 0 || frame.height <= 0) return false;
    std::lock_guard<std::mutex> lock(oidnMutex);

    // If we were previously bound to an external CUDA stream (GPU zero-copy path on OptiX),
    // that stream may now be destroyed (e.g., after switching OptiX→Vulkan). Using the stale
    // device from the host path crashes inside nvcuda64.dll during execute/write. Drop the
    // bound device and let initOIDN() create a fresh, self-managed CUDA or CPU device.
    if (oidnCudaInitialized) {
        oidnColorBuffer  = oidn::BufferRef();
        oidnAlbedoBuffer = oidn::BufferRef();
        oidnNormalBuffer = oidn::BufferRef();
        oidnOutputBuffer = oidn::BufferRef();
        oidnFilter       = oidn::FilterRef();
        oidnDevice       = oidn::DeviceRef();
        oidnInitialized     = false;
        oidnCudaInitialized = false;
        oidnCudaOrdinal = -1;
        oidnCudaStream  = nullptr;
        if (oidnGpuOutputDevPtr) {
            cudaFree(oidnGpuOutputDevPtr);
            oidnGpuOutputDevPtr = nullptr;
        }
        oidnGpuOutputBytes = 0;
        if (oidnGpuPackedDevPtr) {
            cudaFree(oidnGpuPackedDevPtr);
            oidnGpuPackedDevPtr = nullptr;
        }
        oidnGpuPackedBytes = 0;
        oidnCachedWidth = 0;
        oidnCachedHeight = 0;
        oidnUsingAlbedo = false;
        oidnUsingNormal = false;
        oidnGpuCachedColor  = nullptr;
        oidnGpuCachedAlbedo = nullptr;
        oidnGpuCachedNormal = nullptr;
    }

    if (!oidnInitialized) {
        initOIDN();
        if (!oidnInitialized) return false;
    }

    const int width = frame.width;
    const int height = frame.height;
    size_t pixelCount = (size_t)width * height;
    size_t bufferSize = pixelCount * 3;  // Float3
    const bool useAlbedo = frame.albedo != nullptr;
    const bool useNormal = frame.normal != nullptr;
    const bool filterLayoutChanged =
        (useAlbedo != oidnUsingAlbedo) ||
        (useNormal != oidnUsingNormal);
    bool sizeChanged = (width != oidnCachedWidth || height != oidnCachedHeight);
    const bool qualityChanged = (quality != oidnGpuCachedQuality);
    if (sizeChanged || filterLayoutChanged || qualityChanged) {
        try {
            oidnColorBuffer = oidnDevice.newBuffer(bufferSize * sizeof(float));
            oidnAlbedoBuffer = useAlbedo ? oidnDevice.newBuffer(bufferSize * sizeof(float)) : oidn::BufferRef();
            oidnNormalBuffer = useNormal ? oidnDevice.newBuffer(bufferSize * sizeof(float)) : oidn::BufferRef();
            oidnOutputBuffer = oidnDevice.newBuffer(bufferSize * sizeof(float));

            oidnFilter = oidnDevice.newFilter("RT");
            oidnFilter.setImage("color", oidnColorBuffer,
                oidn::Format::Float3, width, height);
            if (useAlbedo) {
                oidnFilter.setImage("albedo", oidnAlbedoBuffer,
                    oidn::Format::Float3, width, height);
            }
            if (useNormal) {
                oidnFilter.setImage("normal", oidnNormalBuffer,
                    oidn::Format::Float3, width, height);
            }
            oidnFilter.setImage("output", oidnOutputBuffer,
                oidn::Format::Float3, width, height);
            oidnFilter.set("hdr", true);
            oidnFilter.set("srgb", false);
            oidnFilter.set("quality", quality);
            oidnFilter.commit();

            oidnGpuCachedQuality = quality;
            oidnCachedWidth = width;
            oidnCachedHeight = height;
            oidnUsingAlbedo = useAlbedo;
            oidnUsingNormal = useNormal;
        }
        catch (const std::exception& e) {
            SCENE_LOG_ERROR(std::string("[OIDN] Buffer creation failed: ") + e.what());
            return false;
        }
    }

    output.resize(bufferSize);

    try {
        // Direct upload from caller's contiguous float3 buffer — no staging copy
        oidnColorBuffer.write(0, bufferSize * sizeof(float), frame.color);
        if (useAlbedo) {
            oidnAlbedoBuffer.write(0, bufferSize * sizeof(float), frame.albedo);
        }
        if (useNormal) {
            oidnNormalBuffer.write(0, bufferSize * sizeof(float), frame.normal);
        }
        oidnFilter.execute();

        const char* errMsg;
        if (oidnDevice.getError(errMsg) != oidn::Error::None) {
            SCENE_LOG_ERROR(std::string("[OIDN] ") + errMsg);
            return false;
        }

        // Download straight into output — no intermediate vector
        oidnOutputBuffer.read(0, bufferSize * sizeof(float), output.data());
    }
    catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("[OIDN] Execution failed: ") + e.what());
        return false;
    }

    // Blend in-place with caller's color as the "original" reference
    if (blend < 0.999f) {
        const float blend_inv = 1.0f - blend;
        const float* __restrict src = frame.color;
        float* __restrict dst = output.data();
        std::for_each_n(std::execution::par_unseq,
            dst, bufferSize,
            [=](float& d) {
                const size_t i = static_cast<size_t>(&d - dst);
                d = d * blend + src[i] * blend_inv;
            });
    }

    return true;
}

bool Renderer::applyOIDNDenoisingToCPUAccumulation(float blend, bool useAuxiliary) {
    const size_t pixelCount = (size_t)image_width * (size_t)image_height;
    if (!cpu_accumulation_valid || cpu_accumulation_buffer.size() != pixelCount || pixelCount == 0) {
        cpu_denoised_valid = false;
        return false;
    }

    std::vector<float> linearColor(pixelCount * 3);
    std::vector<float> linearAlbedo;
    std::vector<float> linearNormal;
    const bool hasAlbedo = useAuxiliary && cpu_albedo_accumulation_buffer.size() == pixelCount;
    const bool hasNormal = useAuxiliary && cpu_normal_accumulation_buffer.size() == pixelCount;
    if (hasAlbedo) linearAlbedo.resize(pixelCount * 3);
    if (hasNormal) linearNormal.resize(pixelCount * 3);

    {
        const int W = image_width;
        const int H = image_height;
        const Vec4* __restrict accSrc = cpu_accumulation_buffer.data();
        const Vec4* __restrict albSrc = hasAlbedo ? cpu_albedo_accumulation_buffer.data() : nullptr;
        const Vec4* __restrict nrmSrc = hasNormal ? cpu_normal_accumulation_buffer.data() : nullptr;
        float* __restrict colDst = linearColor.data();
        float* __restrict albDst = hasAlbedo ? linearAlbedo.data() : nullptr;
        float* __restrict nrmDst = hasNormal ? linearNormal.data() : nullptr;
        // Iterate over source pixels — pointer range is a valid parallel iterator.
        // Y-flip is computed per element from index arithmetic.
        std::for_each_n(std::execution::par_unseq,
            accSrc, pixelCount,
            [=](const Vec4& sRef) {
                const size_t srcIdx = static_cast<size_t>(&sRef - accSrc);
                const int y_from_bottom = static_cast<int>(srcIdx / (size_t)W);
                const int x = static_cast<int>(srcIdx % (size_t)W);
                const int dstY = H - 1 - y_from_bottom;
                const size_t dstIdx = ((size_t)dstY * (size_t)W + (size_t)x) * 3;
                colDst[dstIdx]     = sRef.x;
                colDst[dstIdx + 1] = sRef.y;
                colDst[dstIdx + 2] = sRef.z;
                if (albDst) {
                    const Vec4& a = albSrc[srcIdx];
                    albDst[dstIdx]     = a.x;
                    albDst[dstIdx + 1] = a.y;
                    albDst[dstIdx + 2] = a.z;
                }
                if (nrmDst) {
                    const Vec4& n = nrmSrc[srcIdx];
                    nrmDst[dstIdx]     = n.x;
                    nrmDst[dstIdx + 1] = n.y;
                    nrmDst[dstIdx + 2] = n.z;
                }
            });
    }

    OIDNFrameData frame;
    frame.width = image_width;
    frame.height = image_height;
    frame.color = linearColor.data();
    frame.albedo = hasAlbedo ? linearAlbedo.data() : nullptr;
    frame.normal = hasNormal ? linearNormal.data() : nullptr;

    cpu_denoised_valid = applyOIDNDenoising(frame, blend, cpu_denoised_buffer);
    return cpu_denoised_valid;
}

bool Renderer::fillStylizeAOVFromBackend(Backend::IBackend* backend, const Camera& camera, bool forceRefresh) {
    if (!backend || !stylizeMode.enabled) {
        // Invalidate so a later re-enable re-pulls — the scene/camera may have changed
        // (and reset accumulation) while stylize was off and we weren't tracking it.
        m_stylizeAOVValid = false;
        return false;
    }

    const size_t pixel_count = static_cast<size_t>(image_width) * static_cast<size_t>(image_height);
    const Vec3 cameraOrigin = camera.lookfrom;   // GPU ray origin (see syncCameraToBackend)

    // Cache invalidation keyed on a camera hash. Camera motion changes the hash every
    // frame → AOVs re-pull every frame → no lag/ghost trails. When the view is static the
    // hash is stable and the cached cpu_* buffers are reused (skipping the readback). The
    // sample-reset check is a secondary trigger for scene edits with a static camera.
    auto hashFloat = [](uint64_t h, float f) -> uint64_t {
        uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        return (h ^ static_cast<uint64_t>(bits)) * 1099511628211ull;
    };
    uint64_t cam_hash = 1469598103934665603ull;
    cam_hash = hashFloat(cam_hash, camera.lookfrom.x); cam_hash = hashFloat(cam_hash, camera.lookfrom.y); cam_hash = hashFloat(cam_hash, camera.lookfrom.z);
    cam_hash = hashFloat(cam_hash, camera.lookat.x);   cam_hash = hashFloat(cam_hash, camera.lookat.y);   cam_hash = hashFloat(cam_hash, camera.lookat.z);
    cam_hash = hashFloat(cam_hash, camera.vup.x);      cam_hash = hashFloat(cam_hash, camera.vup.y);      cam_hash = hashFloat(cam_hash, camera.vup.z);
    cam_hash = hashFloat(cam_hash, camera.vfov);

    const int cur_samples = backend->getCurrentSampleCount();
    const bool size_changed =
        cpu_albedo_accumulation_buffer.size() != pixel_count ||
        cpu_world_position_accumulation_buffer.size() != pixel_count ||
        cpu_material_id_buffer.size() != pixel_count;
    if (forceRefresh || size_changed || cam_hash != m_stylizeAOVCamHash ||
        (cur_samples >= 0 && cur_samples < m_stylizeAOVLastSamples)) {
        m_stylizeAOVValid = false;
    }
    m_stylizeAOVCamHash = cam_hash;
    m_stylizeAOVLastSamples = cur_samples;

    if (m_stylizeAOVValid) return true;          // view unchanged → cached AOVs still valid
    if (cur_samples < 1) return false;           // images cleared at sample 0 — nothing to read yet

    // Pull the backend's accumulated primary-hit AOVs to the host. useAuxiliary=true is
    // required (albedo/normal/position only exist in that mode); includeColor=false skips
    // the wasted full-res color copy — stylize reads color from the display surface.
    Backend::DenoiserFrameData frame;
    if (!backend->getDenoiserFrame(frame, /*useAuxiliary=*/true, /*includeColor=*/false)) return false;
    if (frame.width != image_width || frame.height != image_height) return false;
    if (!frame.albedo || !frame.normal || !frame.position) return false;  // position AOV absent → skip

    if (cpu_albedo_accumulation_buffer.size() != pixel_count)        cpu_albedo_accumulation_buffer.resize(pixel_count);
    if (cpu_normal_accumulation_buffer.size() != pixel_count)        cpu_normal_accumulation_buffer.resize(pixel_count);
    if (cpu_world_position_accumulation_buffer.size() != pixel_count) cpu_world_position_accumulation_buffer.resize(pixel_count);
    if (cpu_depth_accumulation_buffer.size() != pixel_count)         cpu_depth_accumulation_buffer.resize(pixel_count);
    if (cpu_material_id_buffer.size() != pixel_count)                cpu_material_id_buffer.resize(pixel_count);

    const float* __restrict alb = frame.albedo;   // stride 3, bottom-up (matches CPU AOV layout)
    const float* __restrict nrm = frame.normal;   // stride 3, bottom-up (already decoded to [-1,1])
    const float* __restrict pos = frame.position; // stride 4: x,y,z = world pos, w = encoded material

    // Per-pixel decode is independent → run it parallel/vectorized. Iterate over the depth
    // buffer and recover the linear index from the element address (same trick as the
    // OIDN blend loop above), so no index vector is needed.
    float* __restrict depthBase = cpu_depth_accumulation_buffer.data();
    std::for_each_n(std::execution::par_unseq, depthBase, pixel_count,
        [this, pos, alb, nrm, cameraOrigin, depthBase](float& depthRef) {
            const size_t i = static_cast<size_t>(&depthRef - depthBase);
            const float wx = pos[i*4+0], wy = pos[i*4+1], wz = pos[i*4+2];
            const float matEncoded = pos[i*4+3];   // 0 = miss, 1 = hit/unknown mat, >=2 → matid = w-2
            const bool hit = matEncoded >= 0.5f;

            // Reconstruct linear depth from world position (the shader stores material id in
            // the .w channel instead of depth, so outlines key off real material boundaries).
            float depth = 0.0f;
            if (hit) {
                const float dx = wx - cameraOrigin.x, dy = wy - cameraOrigin.y, dz = wz - cameraOrigin.z;
                depth = std::sqrt(dx*dx + dy*dy + dz*dz);
            }

            uint32_t material_id = 0xFFFFFFFFu;
            if (matEncoded >= 1.5f) {
                material_id = static_cast<uint32_t>(matEncoded + 0.5f) - 2u;
            }

            cpu_albedo_accumulation_buffer[i] = Vec4{ alb[i*3+0], alb[i*3+1], alb[i*3+2], hit ? 1.0f : 0.0f };
            cpu_normal_accumulation_buffer[i] = Vec4{ nrm[i*3+0], nrm[i*3+1], nrm[i*3+2], 0.0f };
            cpu_world_position_accumulation_buffer[i] = Vec4{ wx, wy, wz, 0.0f };
            depthRef = depth;
            cpu_material_id_buffer[i] = material_id;
        });
    m_stylizeAOVValid = true;
    return true;
}

bool Renderer::applyOIDNDenoisingGPU(const Backend::DenoiserFrameDataGPU& frame,
                                     float blend,
                                     const OIDNTonemapParams& tm,
                                     std::vector<uint32_t>& packedOutput) {
    if (!frame.colorDevPtr || frame.width <= 0 || frame.height <= 0) return false;

    // Skip while the viewport backend is tearing down/rebuilding its CUDA-imported
    // textures or while OptiX TLAS rebuild is running. Touching frame.colorDevPtr
    // during either window hits cudaErrorIllegalAddress (700), poisons the CUDA
    // context, and trashes the subsequent OptiX BLAS build.
    extern std::atomic<bool> g_viewport_rebuild_in_progress;
    extern std::atomic<bool> g_optix_rebuild_in_progress;
    if (g_viewport_rebuild_in_progress.load(std::memory_order_acquire) ||
        g_optix_rebuild_in_progress.load(std::memory_order_acquire)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(oidnMutex);

    // Resolve CUDA device ordinal
    int desiredOrd = frame.cudaDeviceOrdinal;
    if (desiredOrd < 0) {
        if (cudaGetDevice(&desiredOrd) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
    }
    cudaStream_t desiredStream = static_cast<cudaStream_t>(frame.cudaStream);

    const bool deviceChanged =
        !oidnCudaInitialized ||
        oidnCudaOrdinal != desiredOrd ||
        oidnCudaStream != frame.cudaStream;

    if (deviceChanged) {
        try {
            // CRITICAL: drain in-flight work on the OLD device/stream BEFORE we
            // reassign oidnDevice. FilterRef/BufferRef dtors release handles
            // owned by the old device; if the old stream still has pending
            // OIDN kernels in flight, dtor races against running work and we
            // get an access violation in oidn::FilterRef::~FilterRef
            // (oidn.hpp:384). Sync first, then drop refs while the old
            // device is still the live oidnDevice.
            if (oidnInitialized) {
                if (oidnCudaStream) {
                    cudaStreamSynchronize(static_cast<cudaStream_t>(oidnCudaStream));
                    cudaGetLastError();
                }
                try { oidnDevice.sync(); } catch (...) {}
                oidnFilter       = oidn::FilterRef();
                oidnColorBuffer  = oidn::BufferRef();
                oidnAlbedoBuffer = oidn::BufferRef();
                oidnNormalBuffer = oidn::BufferRef();
                oidnOutputBuffer = oidn::BufferRef();
                if (oidnGpuOutputDevPtr) {
                    cudaFree(oidnGpuOutputDevPtr);
                    oidnGpuOutputDevPtr = nullptr;
                }
                oidnGpuOutputBytes = 0;
                if (oidnGpuPackedDevPtr) {
                    cudaFree(oidnGpuPackedDevPtr);
                    oidnGpuPackedDevPtr = nullptr;
                }
                oidnGpuPackedBytes = 0;
                oidnCachedWidth = 0;
                oidnCachedHeight = 0;
                oidnUsingAlbedo = false;
                oidnUsingNormal = false;
                oidnGpuCachedColor = nullptr;
                oidnGpuCachedAlbedo = nullptr;
                oidnGpuCachedNormal = nullptr;
            }

            oidnDevice = oidn::newCUDADevice(desiredOrd, desiredStream);
            oidnDevice.commit();
            const char* errMsg = nullptr;
            if (oidnDevice.getError(errMsg) != oidn::Error::None) {
                SCENE_LOG_WARN(std::string("[OIDN GPU] CUDA device commit failed: ")
                               + (errMsg ? errMsg : "unknown"));
                oidnCudaInitialized = false;
                return false;
            }
            oidnCudaOrdinal = desiredOrd;
            oidnCudaStream = frame.cudaStream;
            oidnCudaInitialized = true;
            oidnInitialized = true;
        } catch (const std::exception& e) {
            SCENE_LOG_WARN(std::string("[OIDN GPU] CUDA device bind failed: ") + e.what());
            oidnCudaInitialized = false;
            return false;
        }
    }

    const int width = frame.width;
    const int height = frame.height;
    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t outBufferSize = pixelCount * 3;

    const bool useAlbedo = frame.albedoDevPtr != nullptr;
    const bool useNormal = frame.normalDevPtr != nullptr;

    // Basic validation of device-resident layout to avoid illegal accesses
    if (frame.pixelByteStride < sizeof(float) * 3) {
        SCENE_LOG_ERROR("[OIDN GPU] invalid pixelByteStride (too small)");
        return false;
    }
    if (frame.rowByteStride < frame.pixelByteStride * static_cast<size_t>(width)) {
        SCENE_LOG_ERROR("[OIDN GPU] invalid rowByteStride (too small for width)");
        return false;
    }

    const bool sizeChanged   = (width != oidnCachedWidth) || (height != oidnCachedHeight);
    const bool layoutChanged = (useAlbedo != oidnUsingAlbedo) || (useNormal != oidnUsingNormal);
    const bool ptrsChanged   =
        (frame.colorDevPtr  != oidnGpuCachedColor)  ||
        (frame.albedoDevPtr != oidnGpuCachedAlbedo) ||
        (frame.normalDevPtr != oidnGpuCachedNormal);
    const bool outputSizeChanged = oidnGpuOutputBytes != outBufferSize * sizeof(float);
    const bool qualityChanged = (tm.quality != oidnGpuCachedQuality);
    const bool rebuildFilter = sizeChanged || layoutChanged || ptrsChanged || outputSizeChanged || qualityChanged || !oidnFilter;

    if (rebuildFilter) {
        try {
            // Ensure we're on the device expected by the frame to avoid stale-context issues
            int currentDev = -1;
            if (cudaGetDevice(&currentDev) != cudaSuccess) {
                cudaGetLastError();
                SCENE_LOG_WARN("[OIDN GPU] cudaGetDevice failed before setup");
                return false;
            }
            if (desiredOrd >= 0 && desiredOrd != currentDev) {
                if (cudaSetDevice(desiredOrd) != cudaSuccess) {
                    cudaGetLastError();
                    SCENE_LOG_WARN("[OIDN GPU] cudaSetDevice failed when binding desired device");
                    return false;
                }
            }

            // Drain any in-flight OIDN/CUDA work on this stream before tearing
            // down the existing filter and (optionally) freeing the output
            // buffer. Reassigning oidnFilter destroys the old FilterRef; if its
            // device still has pending async work, the dtor races and we crash
            // in oidn::FilterRef::~FilterRef (oidn.hpp:384).
            if (oidnFilter) {
                if (desiredStream) {
                    cudaStreamSynchronize(desiredStream);
                    cudaGetLastError();
                }
                try { oidnDevice.sync(); } catch (...) {}
                oidnFilter = oidn::FilterRef();
            }

            if (outputSizeChanged) {
                if (oidnGpuOutputDevPtr) {
                    cudaFree(oidnGpuOutputDevPtr);
                    oidnGpuOutputDevPtr = nullptr;
                }
                if (cudaMalloc(&oidnGpuOutputDevPtr, outBufferSize * sizeof(float)) != cudaSuccess) {
                    cudaGetLastError();
                    oidnGpuOutputDevPtr = nullptr;
                    oidnGpuOutputBytes = 0;
                    return false;
                }
                oidnGpuOutputBytes = outBufferSize * sizeof(float);

                // Packed RGBA8 output buffer (one uint32 per pixel). Tracks the
                // same lifetime as the float3 OIDN output so a single size gate
                // governs both.
                if (oidnGpuPackedDevPtr) {
                    cudaFree(oidnGpuPackedDevPtr);
                    oidnGpuPackedDevPtr = nullptr;
                }
                const size_t packedBytes = pixelCount * sizeof(uint32_t);
                if (cudaMalloc(&oidnGpuPackedDevPtr, packedBytes) != cudaSuccess) {
                    cudaGetLastError();
                    oidnGpuPackedDevPtr = nullptr;
                    oidnGpuPackedBytes = 0;
                    // Free the float3 buffer we just allocated to keep state consistent.
                    cudaFree(oidnGpuOutputDevPtr);
                    oidnGpuOutputDevPtr = nullptr;
                    oidnGpuOutputBytes = 0;
                    return false;
                }
                oidnGpuPackedBytes = packedBytes;
            }

            oidnFilter = oidnDevice.newFilter("RT");
            // Device-resident inputs with explicit stride.
            oidnFilter.setImage("color", frame.colorDevPtr, oidn::Format::Float3,
                                static_cast<size_t>(width), static_cast<size_t>(height),
                                /*byteOffset=*/0,
                                frame.pixelByteStride, frame.rowByteStride);
            if (useAlbedo) {
                oidnFilter.setImage("albedo", frame.albedoDevPtr, oidn::Format::Float3,
                                    static_cast<size_t>(width), static_cast<size_t>(height),
                                    0, frame.pixelByteStride, frame.rowByteStride);
            }
            if (useNormal) {
                oidnFilter.setImage("normal", frame.normalDevPtr, oidn::Format::Float3,
                                    static_cast<size_t>(width), static_cast<size_t>(height),
                                    0, frame.pixelByteStride, frame.rowByteStride);
            }
            // Output: packed float3 CUDA buffer owned by Renderer so we can optionally
            // blend in-place on device before the final readback.
            oidnFilter.setImage("output", oidnGpuOutputDevPtr, oidn::Format::Float3,
                                static_cast<size_t>(width), static_cast<size_t>(height),
                                /*byteOffset=*/0,
                                sizeof(float) * 3,
                                static_cast<size_t>(width) * sizeof(float) * 3);
            oidnFilter.set("hdr", true);
            oidnFilter.set("srgb", false);
            // OIDN model tier, user-selectable (RenderSettings::denoiser_quality).
            // Default Fast is the cheapest model — on a mid-range GPU OIDN execute
            // dominates the denoise cost (~14-30ms profiled at 720p+aux), so the
            // tier is the single highest-leverage knob. Final renders pass High
            // here via OIDNTonemapParams::quality.
            oidnFilter.set("quality", tm.quality);
            oidnFilter.commit();

            oidnCachedWidth  = width;
            oidnCachedHeight = height;
            oidnUsingAlbedo  = useAlbedo;
            oidnUsingNormal  = useNormal;
            oidnGpuCachedColor  = frame.colorDevPtr;
            oidnGpuCachedAlbedo = frame.albedoDevPtr;
            oidnGpuCachedNormal = frame.normalDevPtr;
            oidnGpuCachedQuality = tm.quality;
        } catch (const std::exception& e) {
            SCENE_LOG_ERROR(std::string("[OIDN GPU] filter setup failed: ") + e.what());
            return false;
        }
    }

    packedOutput.resize(pixelCount);

#if RT_OIDN_PROFILING
    // ── [PROFILE] CUDA-event timestamps around each GPU stage. Events live on
    // the desiredStream; elapsed time is in float ms (cudaEventElapsedTime).
    // Lazily created and reused across calls; recreated if the stream changes.
    static thread_local cudaEvent_t profEvBeforeExec = nullptr;
    static thread_local cudaEvent_t profEvAfterExec  = nullptr;
    static thread_local cudaEvent_t profEvAfterTm    = nullptr;
    static thread_local cudaEvent_t profEvAfterCopy  = nullptr;
    auto ensureProfEvents = [&]() {
        if (!profEvBeforeExec) cudaEventCreate(&profEvBeforeExec);
        if (!profEvAfterExec)  cudaEventCreate(&profEvAfterExec);
        if (!profEvAfterTm)    cudaEventCreate(&profEvAfterTm);
        if (!profEvAfterCopy)  cudaEventCreate(&profEvAfterCopy);
    };
    ensureProfEvents();
#endif

    try {
        // Ensure OptiX writes on its stream are visible before OIDN reads shared memory.
        if (desiredStream) {
            if (cudaStreamSynchronize(desiredStream) != cudaSuccess) {
                cudaGetLastError();
                SCENE_LOG_WARN("[OIDN GPU] cudaStreamSynchronize failed before execute");
                // proceed — OIDN may still report an error
            }
        }

#if RT_OIDN_PROFILING
        cudaEventRecord(profEvBeforeExec, desiredStream);
#endif
        oidnFilter.execute();
#if RT_OIDN_PROFILING
        cudaEventRecord(profEvAfterExec, desiredStream);
#endif

        const char* errMsg = nullptr;
        if (oidnDevice.getError(errMsg) != oidn::Error::None) {
            SCENE_LOG_ERROR(std::string("[OIDN GPU] ") + (errMsg ? errMsg : "execute error"));
            return false;
        }

        if (blend < 0.999f) {
            if (!launchOidnBlendKernel(static_cast<float*>(oidnGpuOutputDevPtr),
                                       frame.colorDevPtr,
                                       pixelCount,
                                       blend,
                                       frame.pixelByteStride,
                                       desiredStream)) {
                cudaError_t le = cudaGetLastError();
                SCENE_LOG_ERROR(std::string("[OIDN GPU] blend kernel launch failed: ") + std::to_string(static_cast<int>(le)));
                // Drain sticky CUDA error and force OIDN device rebind on next frame
                // so we don't keep launching kernels into a poisoned context.
                if (le == cudaErrorIllegalAddress) {
                    cudaGetLastError();
                    oidnCudaInitialized = false;
                }
                return false;
            }
        }

        // Fused tonemap + sRGB + pack-to-uint32. Replaces the per-pixel CPU
        // std::pow loop and shrinks the D2H transfer by 3×.
        if (!launchOidnTonemapKernel(static_cast<const float*>(oidnGpuOutputDevPtr),
                                     oidnGpuPackedDevPtr,
                                     width, height,
                                     tm.exposure,
                                     tm.aMaskOr,
                                     tm.rShift, tm.gShift, tm.bShift,
                                     tm.flipY,
                                     desiredStream)) {
            cudaError_t le = cudaGetLastError();
            SCENE_LOG_ERROR(std::string("[OIDN GPU] tonemap kernel launch failed: ") + std::to_string(static_cast<int>(le)));
            if (le == cudaErrorIllegalAddress) {
                cudaGetLastError();
                oidnCudaInitialized = false;
            }
            return false;
        }
#if RT_OIDN_PROFILING
        cudaEventRecord(profEvAfterTm, desiredStream);
#endif

        if (desiredStream) {
            if (cudaStreamSynchronize(desiredStream) != cudaSuccess) {
                cudaError_t se = cudaGetLastError();
                SCENE_LOG_ERROR(std::string("[OIDN GPU] cudaStreamSynchronize failed after tonemap: ") + std::to_string(static_cast<int>(se)));
                // Same recovery: drain sticky error and force device rebind.
                if (se == cudaErrorIllegalAddress) {
                    cudaGetLastError();
                    oidnCudaInitialized = false;
                }
                return false;
            }
        }

        // Final transfer: tonemapped RGBA8 → host. 4 B/px vs the original 12 B/px.
        if (cudaMemcpy(packedOutput.data(), oidnGpuPackedDevPtr,
                       pixelCount * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaError_t me = cudaGetLastError();
            SCENE_LOG_ERROR(std::string("[OIDN GPU] final cudaMemcpy failed: ") + std::to_string(static_cast<int>(me)));
            return false;
        }
#if RT_OIDN_PROFILING
        cudaEventRecord(profEvAfterCopy, desiredStream);

        // ── [PROFILE] Elapsed times between events. cudaEventSynchronize on the
        // last event ensures all the intermediate events have valid timing.
        cudaEventSynchronize(profEvAfterCopy);
        float msExec = 0.0f, msTm = 0.0f, msCopy = 0.0f;
        cudaEventElapsedTime(&msExec, profEvBeforeExec, profEvAfterExec);
        cudaEventElapsedTime(&msTm,   profEvAfterExec,  profEvAfterTm);
        cudaEventElapsedTime(&msCopy, profEvAfterTm,    profEvAfterCopy);
        static thread_local int   profCounter = 0;
        static thread_local float profExecAvg = 0.0f, profTmAvg = 0.0f, profCopyAvg = 0.0f;
        profExecAvg = profExecAvg * 0.95f + msExec * 0.05f;
        profTmAvg   = profTmAvg   * 0.95f + msTm   * 0.05f;
        profCopyAvg = profCopyAvg * 0.95f + msCopy * 0.05f;
        if (++profCounter % 300 == 0) {
            SCENE_LOG_INFO(std::string("[OIDN][CUDA] oidn=")
                           + std::to_string(profExecAvg)
                           + "ms tonemap+blend=" + std::to_string(profTmAvg)
                           + "ms d2h=" + std::to_string(profCopyAvg) + "ms");
        }
#endif
    } catch (const std::exception& e) {
        SCENE_LOG_ERROR(std::string("[OIDN GPU] execute failed: ") + e.what());
        return false;
    }

    return true;
}

bool Renderer::hasCPUDenoisedBuffer() const {
    return cpu_denoised_valid &&
        cpu_denoised_buffer.size() == (size_t)image_width * (size_t)image_height * 3;
}

void Renderer::invalidateCPUDenoisedBuffer() {
    cpu_denoised_valid = false;
}
Renderer::Renderer(int image_width, int image_height, int samples_per_pixel, int max_depth)
    : image_width(image_width), image_height(image_height), aspect_ratio(static_cast<double>(image_width) / image_height), halton_cache(new float[MAX_DIMENSIONS * MAX_SAMPLES_HALTON]), color_processor(image_width, image_height)
{
    initialize_halton_cache();

    frame_buffer.resize(image_width * image_height);
    sample_counts.resize(image_width * image_height, 0);
    max_halton_index = MAX_SAMPLES_HALTON - 1; // Halton dizisi i�in maksimum indeks

    // Adaptive sampling i�in bufferlar
    variance_buffer.resize(image_width * image_height, 0.0f);

    rendering_complete = false;

    variance_map.resize(image_width * image_height, 0.0f);


}
void Renderer::resetResolution(int w, int h) {
    image_width = w;
    image_height = h;
    aspect_ratio = static_cast<double>(image_width) / image_height;

    const size_t pixel_count = w * h;

    // Buffers resize
    frame_buffer.resize(pixel_count);
    variance_buffer.resize(pixel_count, 0.0f);  // reset variance
    sample_counts.resize(pixel_count, 0);       // reset counts
    variance_map.resize(pixel_count, 0.0f);     // optional if used in display

    // Optional: zero the actual frame buffer content
    std::fill(frame_buffer.begin(), frame_buffer.end(), Vec3(0.0f));

    // OIDN cache invalidate - buffer'lar bir sonraki denoise'da yeniden olu�turulacak
    oidnCachedWidth = 0;
    oidnCachedHeight = 0;
    invalidateCPUDenoisedBuffer();

    // Pixel list cache invalidate - resolution changed
    cpu_pixel_list_valid = false;
}


Renderer::~Renderer()
{
    if (oidnGpuOutputDevPtr) {
        cudaFree(oidnGpuOutputDevPtr);
        oidnGpuOutputDevPtr = nullptr;
    }
    oidnGpuOutputBytes = 0;
    if (oidnGpuPackedDevPtr) {
        cudaFree(oidnGpuPackedDevPtr);
        oidnGpuPackedDevPtr = nullptr;
    }
    oidnGpuPackedBytes = 0;
    frame_buffer.clear();
    sample_counts.clear();
    variance_map.clear();
}


bool Renderer::SaveSurface(SDL_Surface* surface, const char* file_path)
{

    // Ayn� isimde dosya varsa silmeye �al�� (zorla yazma)
    if (std::filesystem::exists(file_path)) {
        std::error_code ec;
        std::filesystem::remove(file_path, ec);
        if (ec) {
            SDL_Log("Dosya silinemiyor. Ba�ka bir i�lem taraf�ndan kullan�l�yor olabilir.");
            return false;
        }
    }

    SDL_Surface* surface_to_save =
        SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGB24, 0);

    if (!surface_to_save) {
        SDL_Log("Surface format conversion failed: %s", SDL_GetError());
        return false;
    }


    int result = IMG_SavePNG(surface_to_save, file_path);
    SDL_FreeSurface(surface_to_save);

    if (result != 0) {
        SDL_Log("Failed to save image: %s", IMG_GetError());
        return false;
    }

    return true;
}

Vec3 Renderer::getColorFromSurface(SDL_Surface* surface, int i, int j) {
    Uint32* pixels = static_cast<Uint32*>(surface->pixels);
    Uint32 pixel = pixels[(surface->h - 1 - j) * surface->pitch / 4 + i];

    Uint8 r, g, b;
    SDL_GetRGB(pixel, surface->format, &r, &g, &b);

    // sRGB to linear d�n���m� istersen buraya koy
    return Vec3(r / 255.0f, g / 255.0f, b / 255.0f);
}

static int point_light_pick_count = 0;
static int directional_pick_count = 0;

// ============================================================================
// NEW ANIMATION SYSTEM INTEGRATION
// ============================================================================

void Renderer::initializeAnimationSystem(SceneData& scene) {
    // Initialize per-model animators
    for (auto& ctx : scene.importedModelContexts) {
        if (!ctx.hasAnimation) {
            continue;
        }

        if (!ctx.animator) {
            ctx.animator = std::make_shared<AnimationController>();
        }

        // Filter clips for this model
        std::vector<std::shared_ptr<AnimationData>> modelClips;
        for (auto& anim : scene.animationDataList) {
            if (anim && (anim->modelName == ctx.importName || (anim->modelName.empty() && scene.importedModelContexts.size() == 1))) {
                modelClips.push_back(anim);
            }
        }

        ctx.animator->registerClips(modelClips);
        ctx.ozzAnimationSet = OzzRuntime::buildStubAnimationSet(ctx.importName, scene.boneData, modelClips);

        if (!modelClips.empty()) {
            // [FIX] Do NOT auto-play on import.
            // Keep character in Bind Pose (T-Pose) until an animation node is added to AnimGraph.
            // ctx.animator->play(modelClips[0]->name, 0.0f);
            SCENE_LOG_INFO("[Renderer] Created animator for model: " + ctx.importName + " (Clips: " + std::to_string(modelClips.size()) + ")");
            if (ctx.ozzAnimationSet && ctx.ozzAnimationSet->hasScaffoldData()) {
                size_t trackCount = 0;
                if (!ctx.ozzAnimationSet->clipRuntimes.empty()) {
                    trackCount = ctx.ozzAnimationSet->clipRuntimes.front().jointTracks.size();
                }
                SCENE_LOG_INFO("[Renderer] Prepared " + std::string(OzzRuntime::backendLabel()) +
                    " animation scaffold for model: " + ctx.importName +
                    " (joints: " + std::to_string(ctx.ozzAnimationSet->skeleton.jointCount) +
                    ", clips: " + std::to_string(ctx.ozzAnimationSet->clips.size()) +
                    ", track-joints: " + std::to_string(trackCount) + ")");

                if (ctx.ozzAnimationSet->isUsable() && !ctx.ozzAnimationSet->runtimeAnimations.empty()) {
                    std::vector<Matrix4x4> sampledMatrices;
                    if (OzzRuntime::sampleAnimationToModelMatrices(*ctx.ozzAnimationSet, 0, 0.0f, &sampledMatrices)) {
                        SCENE_LOG_INFO("[Renderer] Ozz runtime sampling ready for model: " + ctx.importName +
                            " (sampled matrices: " + std::to_string(sampledMatrices.size()) + ")");
                    }
                }
            }
        }
    }
}

bool Renderer::updateAnimationWithGraph(SceneData& scene, float deltaTime, bool apply_cpu_skinning) {
    bool anyChanged = false;

    // Resize internal matrix buffer to match scene total bone count
    // (kept for global access, e.g. GPU upload)
    // IMPORTANT: Use resize(), NOT assign()! When a second model is imported,
    // totalBones increases. assign() would destroy Model A's existing bone matrices,
    // causing its skinned mesh to collapse to origin on the next GPU skinning pass
    // (before Model A's animator re-computes). resize() preserves existing entries.
    size_t totalBones = scene.boneData.boneNameToIndex.size();
    if (this->finalBoneMatrices.size() < totalBones) {
        this->finalBoneMatrices.resize(totalBones, Matrix4x4::identity());
    }
    // Optimization: avoid resizing every frame, but ensure identity for non-animated models
    // Actually, AnimationController::update already fills with identity.

    // Update each model context independently
    for (auto& ctx : scene.importedModelContexts) {
        // Per-model bone matrices for isolated skinning
        std::vector<Matrix4x4> modelBoneMatrices;
        bool modelChanged = false;
        auto applyModelRestPose = [&]() {
            bool applied = false;
            for (const auto& [boneName, boneIndex] : scene.boneData.boneNameToIndex) {
                if (boneName.find(ctx.importName + "_") != 0) {
                    continue;
                }

                const size_t globalIdx = static_cast<size_t>(boneIndex);
                if (globalIdx >= this->finalBoneMatrices.size()) {
                    continue;
                }
                this->finalBoneMatrices[globalIdx] = Matrix4x4::identity();
                applied = true;
            }
            return applied;
        };

        auto activeGraph = ctx.runtimeGraph ? ctx.runtimeGraph : ctx.graph;
        if (ctx.useAnimGraph && activeGraph) {
            float graphDeltaTime = deltaTime;
            if (!ctx.animGraphFollowTimeline && graphDeltaTime == 0.0f) {
                graphDeltaTime = 1.0f / 60.0f;
            }
            if (ctx.animator && ctx.animator->isPaused()) {
                graphDeltaTime = 0.0f;
            }

            // EVALUATE NODE GRAPH (Unity/Unreal style)
            if (ctx.animator) {
                activeGraph->evalContext.clipsPtr = &ctx.animator->getAllClips();
            }

            // Timeline-driven anim graph overrides for cinematic shots.
            activeGraph->evalContext.triggerParams.clear();
            auto trackIt = scene.timeline.tracks.find(ctx.importName);
            if (trackIt != scene.timeline.tracks.end()) {
                Keyframe animKf = trackIt->second.evaluate(scene.timeline.current_frame);
                if (animKf.has_anim_graph) {
                    for (auto& node : activeGraph->nodes) {
                        auto* clipNode = dynamic_cast<AnimationGraph::AnimClipNode*>(node.get());
                        if (!clipNode) continue;

                        auto clipIt = animKf.anim_graph.clip_overrides.find(clipNode->id);
                        if (clipIt != animKf.anim_graph.clip_overrides.end() && !clipIt->second.empty()) {
                            if (clipNode->clipName != clipIt->second) {
                                clipNode->clipName = clipIt->second;
                                clipNode->reset();
                            }
                        }

                        auto speedIt = animKf.anim_graph.clip_speed_overrides.find(clipNode->id);
                        if (speedIt != animKf.anim_graph.clip_speed_overrides.end()) {
                            clipNode->playbackSpeed = speedIt->second;
                        }
                    }
                    for (const auto& [name, value] : animKf.anim_graph.float_params) {
                        activeGraph->evalContext.floatParams[name] = value;
                    }
                    for (const auto& [name, value] : animKf.anim_graph.bool_params) {
                        activeGraph->evalContext.boolParams[name] = value;
                    }
                    for (const auto& [name, value] : animKf.anim_graph.int_params) {
                        activeGraph->evalContext.intParams[name] = value;
                    }
                    for (const auto& triggerName : animKf.anim_graph.triggers) {
                        activeGraph->evalContext.triggerParams[triggerName] = triggerName;
                    }
                    if (!animKf.anim_graph.force_state.empty()) {
                        for (auto& node : activeGraph->nodes) {
                            auto* smNode = dynamic_cast<AnimationGraph::StateMachineNode*>(node.get());
                            if (smNode) {
                                smNode->forceState(animKf.anim_graph.force_state);
                                break;
                            }
                        }
                    }
                }
            }

            // CRITICAL FIX: Fetch robust global inverse transform from boneData
            // ImportedModelContext's copy might be uninitialized identity
            Matrix4x4 finalInv = scene.boneData.globalInverseTransform;
            if (!ctx.importName.empty()) {
                 auto invIt = scene.boneData.perModelInverses.find(ctx.importName);
                 if (invIt != scene.boneData.perModelInverses.end()) finalInv = invIt->second;
            }
            ctx.globalInverseTransform = finalInv; // Update cache
            activeGraph->evalContext.globalInverseTransform = finalInv;
            
            // --- ROOT MOTION PREPARATION FOR ANIM GRAPH ---
            activeGraph->evalContext.useRootMotion = ctx.useRootMotion;
            if (ctx.useRootMotion && ctx.animator) {
                auto clips = ctx.animator->getAllClips();
                if (!clips.empty() && clips[0].sourceData) {
                    activeGraph->evalContext.rootMotionBone = !ctx.rootMotionBone.empty()
                        ? ctx.rootMotionBone
                        : ctx.animator->findBestRootMotionBone(clips[0].name);
                }
            }
            activeGraph->evalContext.rootMotion = RootMotionDelta(); // reset

            AnimationGraph::PoseData pose = activeGraph->evaluate(graphDeltaTime, scene.boneData);

            if (pose.isValid()) {
                if (pose.wasUpdated) {
                    modelChanged = true;
                    anyChanged = true; 
                    
                    // --- APPLY ROOT MOTION FOR ANIM GRAPH ---
                    if (ctx.useRootMotion && pose.rootMotion.hasPosition && !ctx.members.empty()) {
                        Vec3 horizontalDelta = pose.rootMotion.positionDelta;
                        horizontalDelta.y = 0.0f;
                        std::vector<Transform*> processed;
                        for (auto& member : ctx.members) {
                            if (auto tri = std::dynamic_pointer_cast<Triangle>(member)) {
                                Transform* h = tri->getTransformPtr();
                                if (h && std::find(processed.begin(), processed.end(), h) == processed.end()) {
                                    h->position = h->position + horizontalDelta;
                                    h->updateMatrix();
                                    h->markDirty();
                                    processed.push_back(h);
                                }
                            }
                        }
                    }
                }
                modelBoneMatrices = pose.boneTransforms;

                // ===========================================================================
                // CRITICAL FIX: Direct bone-to-index merging for AnimGraph
                // Graph pose.boneTransforms order matches the order in boneData.boneIndexToName
                // but we must ONLY update the indices that belong to THIS model.
                // CRITICAL FIX: Always copy the matrix.
                // If it's identity, it means the bone IS at origin/bind pose.
                // Skipping identity used to cause "stuck" bones from previous poses.
                // ===========================================================================
                for (size_t localIdx = 0; localIdx < modelBoneMatrices.size(); ++localIdx) {
                    const std::string& boneName = scene.boneData.getBoneNameByIndex(localIdx);
                    
                    // Only update if this bone belongs to this model prefix
                    if (boneName.find(ctx.importName + "_") == 0) {
                        // Find the global index for this bone (it should match localIdx here 
                        // IF scene.boneData was built in the same order, but let's be safe)
                        auto it = scene.boneData.boneNameToIndex.find(boneName);
                        if (it != scene.boneData.boneNameToIndex.end()) {
                            unsigned int globalIdx = it->second;
                            if (globalIdx < this->finalBoneMatrices.size()) {
                                this->finalBoneMatrices[globalIdx] = modelBoneMatrices[localIdx];
                            }
                        }
                    }
                }
            }
        }
        else if (ctx.animator) {
            const std::string activeClipName = ctx.animator->getCurrentClipName();
            const bool hasActiveClip = !activeClipName.empty();

            // Sync UI toggle to animator state
            if (ctx.useRootMotion) {
                std::string bestRoot = !ctx.rootMotionBone.empty()
                    ? ctx.rootMotionBone
                    : ctx.animator->findBestRootMotionBone(activeClipName);
                ctx.animator->setRootMotionEnabled(true, bestRoot);
            }
            else {
                ctx.animator->setRootMotionEnabled(false);
            }

            bool usedOzzRuntime = false;
            const auto& allClips = ctx.animator->getAllClips();
            if (!hasActiveClip) {
                // Apply rest pose ONCE when entering the no-clip state, not every frame.
                // Without this guard, applyModelRestPose() returned true on every frame
                // (bone matrices are always set to identity) which triggered resetCPUAccumulation()
                // continuously even though nothing was actually changing.
                if (!ctx.restPoseApplied) {
                    ctx.restPoseApplied = applyModelRestPose();
                    if (ctx.restPoseApplied) {
                        modelChanged = true;
                        if (!ctx.loggedOzzRuntimeUsage) {
                            SCENE_LOG_INFO("[Renderer] Animation runtime for model '" + ctx.importName + "': Rest Pose");
                            ctx.loggedOzzRuntimeUsage = true;
                        }
                    }
                }
            }
            else if (!ctx.useRootMotion && ctx.preferOzzRuntime && ctx.ozzAnimationSet && ctx.ozzAnimationSet->isUsable() && !allClips.empty()) {
                // A clip is now active — allow rest pose to be reapplied if it goes inactive again.
                ctx.restPoseApplied = false;
                auto findOzzClipIndex = [&](const std::string& clipName) -> int {
                    for (size_t i = 0; i < allClips.size(); ++i) {
                        if (allClips[i].name == clipName) {
                            return static_cast<int>(i);
                        }
                    }
                    return -1;
                };

                bool supportsOzzBlendPath = true;
                std::vector<OzzRuntime::BlendLayerInput> ozzBlendLayers;
                for (const auto& layer : ctx.animator->getLayers()) {
                    if (layer.weight <= 0.0f || !layer.blendState.clipA) {
                        continue;
                    }
                    if (layer.blendMode == BlendMode::Override) {
                        supportsOzzBlendPath = false;
                        break;
                    }

                    OzzRuntime::BlendLayerInput input;
                    input.clipIndexA = findOzzClipIndex(layer.blendState.clipA->name);
                    input.timeA = layer.blendState.timeA;
                    input.layerWeight = layer.weight;
                    input.blendMode = layer.blendMode;
                    input.affectedBones = layer.affectedBones;

                    if (layer.blendState.clipB) {
                        input.clipIndexB = findOzzClipIndex(layer.blendState.clipB->name);
                        input.timeB = layer.blendState.timeB;
                        input.blendWeight = layer.blendState.blendWeight;
                    }

                    if (input.clipIndexA < 0) {
                        supportsOzzBlendPath = false;
                        break;
                    }
                    if (layer.blendState.clipB && input.blendWeight > 0.0f && input.clipIndexB < 0) {
                        supportsOzzBlendPath = false;
                        break;
                    }
                    ozzBlendLayers.push_back(input);
                }

                if (supportsOzzBlendPath && !ozzBlendLayers.empty()) {
                    if (OzzRuntime::sampleBlendedAnimationToModelMatrices(*ctx.ozzAnimationSet, ozzBlendLayers, &modelBoneMatrices)) {
                        usedOzzRuntime = true;
                        if (!ctx.loggedOzzRuntimeUsage) {
                            SCENE_LOG_INFO("[Renderer] Animation runtime for model '" + ctx.importName + "': Ozz");
                            ctx.loggedOzzRuntimeUsage = true;
                        }
                        for (const auto& [boneName, globalIdx] : scene.boneData.boneNameToIndex) {
                            if (boneName.find(ctx.importName + "_") != 0) {
                                continue;
                            }
                            if (globalIdx >= this->finalBoneMatrices.size() ||
                                globalIdx >= ctx.ozzAnimationSet->sceneBoneToRuntimeJoint.size()) {
                                continue;
                            }

                            const int runtimeJointIndex = ctx.ozzAnimationSet->sceneBoneToRuntimeJoint[globalIdx];
                            if (runtimeJointIndex < 0 || static_cast<size_t>(runtimeJointIndex) >= modelBoneMatrices.size()) {
                                continue;
                            }
                            this->finalBoneMatrices[globalIdx] = modelBoneMatrices[runtimeJointIndex];
                        }
                    }
                } else {
                    size_t ozzClipIndex = 0;
                    bool foundClip = false;
                    for (size_t i = 0; i < allClips.size(); ++i) {
                        if (allClips[i].name == activeClipName) {
                            ozzClipIndex = i;
                            foundClip = true;
                            break;
                        }
                    }

                    if (foundClip) {
                        const float sampleTime = ctx.animator->getCurrentTime();
                        if (OzzRuntime::sampleAnimationToModelMatrices(*ctx.ozzAnimationSet, ozzClipIndex, sampleTime, &modelBoneMatrices)) {
                            usedOzzRuntime = true;
                            if (!ctx.loggedOzzRuntimeUsage) {
                                SCENE_LOG_INFO("[Renderer] Animation runtime for model '" + ctx.importName + "': Ozz");
                                ctx.loggedOzzRuntimeUsage = true;
                            }
                            for (const auto& [boneName, globalIdx] : scene.boneData.boneNameToIndex) {
                                if (boneName.find(ctx.importName + "_") != 0) {
                                    continue;
                                }
                                if (globalIdx >= this->finalBoneMatrices.size() ||
                                    globalIdx >= ctx.ozzAnimationSet->sceneBoneToRuntimeJoint.size()) {
                                    continue;
                                }

                                const int runtimeJointIndex = ctx.ozzAnimationSet->sceneBoneToRuntimeJoint[globalIdx];
                                if (runtimeJointIndex < 0 || static_cast<size_t>(runtimeJointIndex) >= modelBoneMatrices.size()) {
                                    continue;
                                }
                                this->finalBoneMatrices[globalIdx] = modelBoneMatrices[runtimeJointIndex];
                            }
                        }
                    }
                }
            }

            bool changed = ctx.animator->update(deltaTime, scene.boneData);

            if (!usedOzzRuntime) {
                if (!ctx.loggedOzzRuntimeUsage) {
                    SCENE_LOG_INFO("[Renderer] Animation runtime for model '" + ctx.importName + "': Legacy");
                    ctx.loggedOzzRuntimeUsage = true;
                }
                // Get this model's bone matrices (current state)
                modelBoneMatrices = ctx.animator->getFinalBoneMatrices();

                // ===========================================================================
                // CRITICAL FIX: Map Animator Local Indices to Global Indices
                // The animator's matrices are per-model. We must map them to global slots.
                // ===========================================================================
                if (!allClips.empty()) {
                    // We can use the bone mapping from the first clip as a reference for node names
                    const auto& source = allClips[0].sourceData;
                    if (source) {
                        // This is more complex because Animator doesn't easily expose local-to-global mapping
                        // But we can iterate over ALL bones in the scene and see which ones belong to this model
                        // MODEL ISOLATION FIX:
                        // In the AnimationController, cachedFinalBoneMatrices is already sized for the global bone count.
                        // However, it only contains valid data for bones that belong to the model it's controlling.
                        // We need to copy ONLY the bones that this model 'owns' to avoid overwriting
                        // other models' poses with the fallback Identity pose from this animator's cache.
                        for (const auto& [boneName, globalIdx] : scene.boneData.boneNameToIndex) {
                            if (boneName.find(ctx.importName + "_") == 0) {
                                if (globalIdx < this->finalBoneMatrices.size() && globalIdx < modelBoneMatrices.size()) {
                                    this->finalBoneMatrices[globalIdx] = modelBoneMatrices[globalIdx];
                                }
                            }
                        }
                    }
                }
            }

            // usedOzzRuntime=true simply means Ozz filled bone matrices this frame.
            // When the animator is paused the pose does NOT change, so treat it
            // the same as changed=false to avoid resetting accumulation every frame.
            const bool ozzPoseAdvanced = usedOzzRuntime &&
                !(ctx.animator && ctx.animator->isPaused()) &&
                std::abs(deltaTime) > 1e-6f;

            if (changed || ozzPoseAdvanced) {
                modelChanged = true;


                // --- ROOT MOTION (Pivot movement) ---
                if (ctx.useRootMotion && !usedOzzRuntime) {
                    RootMotionDelta delta = ctx.animator->consumeRootMotion();
                    if (delta.hasPosition && !ctx.members.empty()) {
                        Vec3 horizontalDelta = delta.positionDelta;
                        horizontalDelta.y = 0.0f;
                        std::vector<Transform*> processed;
                        for (auto& member : ctx.members) {
                            if (auto tri = std::dynamic_pointer_cast<Triangle>(member)) {
                                Transform* h = tri->getTransformPtr();
                                if (h && std::find(processed.begin(), processed.end(), h) == processed.end()) {
                                    h->position = h->position + horizontalDelta;
                                    h->updateMatrix(); h->markDirty();
                                    processed.push_back(h);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if (modelChanged) {
            anyChanged = true;
            
            // ============================================================
            // PER-MODEL SKINNING: Apply ONLY to this model's own members
            // If ctx.members is empty (e.g. after project load), lazy-init
            // by matching triangle nodeName prefix to ctx.importName.
            // ============================================================
            if (apply_cpu_skinning && !modelBoneMatrices.empty()) {
                // Lazy populate members if empty (project load doesn't serialize them)
                if (ctx.members.empty() && !ctx.importName.empty()) {
                    std::string prefix = ctx.importName + "_";
                    for (auto& obj : scene.world.objects) {
                        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                        if (tri && tri->nodeName.find(prefix) == 0) {
                            ctx.members.push_back(tri);
                        }
                    }
                }
                
                for (auto& member : ctx.members) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(member);
                    if (tri && tri->hasSkinData()) {
                        tri->apply_skinning(modelBoneMatrices);
                    }
                }
            }
        }
    }

    if (!anyChanged && !this->finalBoneMatrices.empty()) {
        // Even if no clip changed, we might need initial matrices
    }

    // ============================================================
    // POST-SKINNING: Update hair system to follow deformed mesh
    // Hair must be updated AFTER skinning so it reads fresh vertex positions.
    // ============================================================
    // NOTE: hairSystem.updateAllTransforms now called at the end of updateAnimationState
    // to ensure it runs for both Graph and Legacy/Manual animations.

    // Only reset CPU accumulation when geometry actually changed
    // Otherwise sample counter keeps resetting to 0 every frame
    if (anyChanged) {
        resetCPUAccumulation();
    }

    // Clear dirty flags for all animators
    for (auto& ctx : scene.importedModelContexts) {
        if (ctx.animator) {
            ctx.animator->clearDirtyFlag();
        }
    }

    return anyChanged;  // Only report geometry change when it actually happened
}


bool Renderer::updateAnimationState(SceneData& scene, float current_time, bool apply_cpu_skinning, bool force_bind_pose) {
    // ===========================================================================
    // GEOMETRY CHANGE TRACKING (Animation Performance Optimization)
    // ===========================================================================
    // Track if actual geometry (vertex positions) changed.
    // Return false for camera-only or material-only animations to avoid unnecessary BVH rebuilds.
    bool geometry_changed = false;

    // Reset the per-frame "what moved this frame" list. Filled below by the
    // group update loop; consumed by render_Animation to drive a targeted
    // backend update path on small change sets.
    pending_anim_transform_updates.clear();

    static bool was_in_bind_pose = false;
    if (force_bind_pose) {
        if (!was_in_bind_pose) {
            was_in_bind_pose = true;
            // [HAIR PANEL FIX] Do NOT mark geometry_changed=true for the GPU path
            // (apply_cpu_skinning=false).  Hair is re-uploaded to GPU inside
            // uploadHairToGPU() below; triggering updateGeometry afterwards would
            // overwrite m_meshBlasCount (including the newly-uploaded hair BLAS) and
            // cause stale/orphaned BLASes to accumulate on every Hair-tab switch.
            // CPU-only / skinning paths still need geometry_changed = apply_cpu_skinning.
            geometry_changed = apply_cpu_skinning;
            
            if (!scene.boneData.boneNameToIndex.empty()) {
                this->finalBoneMatrices.assign(scene.boneData.boneNameToIndex.size(), Matrix4x4::identity());
            }

            // Ensure animation groups are built
            if (animation_groups_dirty || animation_groups.empty()) {
                animation_groups.clear();
                std::unordered_map<void*, size_t> transformToGroup;
                for (auto& obj : scene.world.objects) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                    if (!tri) continue;
                    void* transformKey = tri->getTransformPtr();
                    if (transformToGroup.find(transformKey) == transformToGroup.end()) {
                        transformToGroup[transformKey] = animation_groups.size();
                        AnimatableGroup newGroup;
                        newGroup.nodeName = tri->getNodeName();
                        newGroup.isSkinned = tri->hasSkinData();
                        newGroup.transformHandle = tri->getTransformHandle();
                        animation_groups.push_back(newGroup);
                    }
                    animation_groups[transformToGroup[transformKey]].triangles.push_back(tri);
                }
                animation_groups_dirty = false;
            }

            for (auto& group : animation_groups) {
                if (group.isSkinned) {
                    if (apply_cpu_skinning) {
                        for (auto& tri : group.triangles) {
                            tri->apply_skinning(static_cast<const std::vector<Matrix4x4>&>(this->finalBoneMatrices));
                        }
                    }
                }
            }
            
            // Rebuild BVHs
            auto embree_ptr = std::dynamic_pointer_cast<EmbreeBVH>(scene.bvh);
            if (embree_ptr) {
                embree_ptr->updateGeometryFromTrianglesFromSource(scene.world.objects);
            }
            
            if (hairSystem.getTotalStrandCount() > 0) {
                hairSystem.updateAllTransforms(scene.world.objects, this->finalBoneMatrices);
                if (hairSystem.isBVHDirty()) {
                    hairSystem.buildBVH(true);
                    uploadHairToGPU();
                }
            }
        }
        return geometry_changed;
    } else {
        if (was_in_bind_pose) {
            was_in_bind_pose = false;
            // Force animation resync
        }
    }


    // Unified Animation Check: If we have clips and bones, use the Controller

    // NOTE: resetCPUAccumulation is now called inside updateAnimationWithGraph
    // only when geometry actually changes. Calling it here unconditionally
    // was preventing sample accumulation beyond 1.

    lastAnimationUpdateTime = current_time;

    // Unified Animation Check: If we have clips and bones, use the Controller
    bool useAnimationController = !scene.animationDataList.empty() && !scene.boneData.boneNameToIndex.empty();

    if (useAnimationController) {
        static float last_sim_time = -1.0f;
        static int last_timeline_frame = -1;
        float deltaTime = (last_sim_time >= 0.0f) ? (current_time - last_sim_time) : (1.0f / 60.0f);
        if (deltaTime < -0.5f || deltaTime > 0.5f) deltaTime = 0.0f;

        // SCRUBBING FIX: Absolute time seek support
        // If current_time changed drastically or timeline was scrubbed, sync animators
        bool timelineScrubbed = false;
        if (last_timeline_frame >= 0) {
            const int frameDelta = scene.timeline.current_frame - last_timeline_frame;
            timelineScrubbed = std::abs(frameDelta) > 1;
        }
        
        if (!timelineScrubbed && std::abs(deltaTime) < 0.0001f) {
            deltaTime = 0.0f;
        }

        if (timelineScrubbed) {
            for (auto& modelCtx : scene.importedModelContexts) {
                if (modelCtx.animator) {
                    modelCtx.animator->setTime(current_time, 0); // Seek to current simulation time
                }
            }
            // Use zero delta for seek frames to avoid double-advancing
            deltaTime = 0.0f; 
        }

        // Drive the animation
        bool changed = updateAnimationWithGraph(scene, deltaTime, apply_cpu_skinning);

        geometry_changed = changed || timelineScrubbed;

        last_sim_time = current_time;
        last_timeline_frame = scene.timeline.current_frame;

        // Rigid (non-skinned) node animation: updateAnimationWithGraph only updates bone matrices.
        // Meshes animated via node-level TRS keys (e.g. props, cubes) need their transformHandle
        // updated from the Assimp node hierarchy every frame — do that here.
        {
            std::unordered_map<std::string, Matrix4x4> animatedGlobalNodeTransforms;

            for (const auto& modelCtx : scene.importedModelContexts) {
                if (!modelCtx.loader || !modelCtx.loader->getScene() || !modelCtx.loader->getScene()->mRootNode) continue;
                if (!modelCtx.hasAnimation) continue;

                std::map<std::string, std::shared_ptr<AnimationData>> animLookup;
                for (const auto& anim : scene.animationDataList) {
                    if (!anim) continue;
                    for (const auto& p : anim->positionKeys) animLookup[p.first] = anim;
                    for (const auto& p : anim->rotationKeys) animLookup[p.first] = anim;
                    for (const auto& p : anim->scalingKeys)  animLookup[p.first] = anim;
                }

                std::unordered_map<std::string, Matrix4x4> modelNodeTransforms;
                modelCtx.loader->calculateAnimatedNodeTransformsRecursive(
                    modelCtx.loader->getScene()->mRootNode,
                    Matrix4x4::identity(),
                    animLookup,
                    current_time,
                    modelNodeTransforms);

                for (const auto& pair : modelNodeTransforms)
                    animatedGlobalNodeTransforms[pair.first] = pair.second;
            }

            if (!animatedGlobalNodeTransforms.empty()) {
                // Ensure animation_groups are populated
                if (animation_groups_dirty || animation_groups.empty()) {
                    animation_groups.clear();
                    std::unordered_map<void*, size_t> transformToGroup;
                    for (auto& obj : scene.world.objects) {
                        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                        if (!tri) continue;
                        void* key = tri->getTransformPtr();
                        if (transformToGroup.find(key) == transformToGroup.end()) {
                            transformToGroup[key] = animation_groups.size();
                            AnimatableGroup ng;
                            ng.nodeName       = tri->getNodeName();
                            ng.isSkinned      = tri->hasSkinData();
                            ng.transformHandle = tri->getTransformHandle();
                            animation_groups.push_back(ng);
                        }
                        animation_groups[transformToGroup[key]].triangles.push_back(tri);
                    }
                    animation_groups_dirty = false;
                }

                for (auto& group : animation_groups) {
                    if (group.isSkinned) continue;

                    bool nodeHasAnim = false;
                    for (const auto& anim : scene.animationDataList) {
                        if (!anim) continue;
                        if (anim->positionKeys.count(group.nodeName) ||
                            anim->rotationKeys.count(group.nodeName) ||
                            anim->scalingKeys.count(group.nodeName)) {
                            nodeHasAnim = true;
                            break;
                        }
                    }

                    if (nodeHasAnim && animatedGlobalNodeTransforms.count(group.nodeName)) {
                        Matrix4x4 animTransform = animatedGlobalNodeTransforms[group.nodeName];
                        if (group.transformHandle) {
                            group.transformHandle->setBase(animTransform);
                            group.transformHandle->setCurrent(Matrix4x4::identity());
                            if (apply_cpu_skinning) {
                                for (auto& tri : group.triangles) tri->updateTransformedVertices();
                            }
                            geometry_changed = true;
                        }
                    }
                }
            }
        }
    }
    else {
        // --- LEGACY FALLBACK PATH ---
        // This part is only reached if useAnimationController is false.


    // --- 1. Ad�m: Animasyonlu Node Hiyerar�isini G�ncelle ---
    std::unordered_map<std::string, Matrix4x4> animatedGlobalNodeTransforms;

    // Ensure bone matrices buffer is large enough for all bones in the scene
    if (!scene.boneData.boneNameToIndex.empty()) {
        if (this->finalBoneMatrices.size() < scene.boneData.boneNameToIndex.size()) {
            this->finalBoneMatrices.resize(scene.boneData.boneNameToIndex.size(), Matrix4x4::identity());
        }
    }

    // Iterate over ALL imported models to update their respective hierarchies
    for (const auto& modelCtx : scene.importedModelContexts) {
        if (!modelCtx.loader || !modelCtx.loader->getScene() || !modelCtx.loader->getScene()->mRootNode) continue;

        // ===========================================================================
        // CRITICAL FIX: Skip models without animation data
        // ===========================================================================
        if (!modelCtx.hasAnimation) {
            continue; // Skip non-animated models - preserve their transforms
        }

        // Build lookups for THIS model's animations
        std::map<std::string, std::shared_ptr<AnimationData>> animationLookupMap;
        for (const auto& anim : scene.animationDataList) {
            if (!anim) continue;
            for (const auto& pair : anim->positionKeys) animationLookupMap[pair.first] = anim;
            for (const auto& pair : anim->rotationKeys) animationLookupMap[pair.first] = anim;
            for (const auto& pair : anim->scalingKeys) animationLookupMap[pair.first] = anim;
        }

        // Temporary map for THIS model's node transforms
        std::unordered_map<std::string, Matrix4x4> modelNodeTransforms;

        modelCtx.loader->calculateAnimatedNodeTransformsRecursive(
            modelCtx.loader->getScene()->mRootNode,
            Matrix4x4::identity(),
            animationLookupMap,
            current_time,
            modelNodeTransforms
        );

        // Merge into global map (for later use by non-bone animated objects)
        for (const auto& pair : modelNodeTransforms) {
            animatedGlobalNodeTransforms[pair.first] = pair.second;
        }

        // --- PRE-CALCULATE GLOBAL BONE MATRICES for THIS model ---
        for (const auto& [boneName, boneIndex] : scene.boneData.boneNameToIndex) {
            // Only process bones that belong to THIS model context
            if (boneName.find(modelCtx.importName + "_") == 0) {
                if (modelNodeTransforms.count(boneName) > 0 && scene.boneData.boneOffsetMatrices.count(boneName) > 0) {
                    Matrix4x4 animatedBoneGlobal = modelNodeTransforms[boneName];
                    Matrix4x4 offsetMatrix = scene.boneData.boneOffsetMatrices[boneName];

                    // P_world = model_globalInv * animGlobal * offset
                    finalBoneMatrices[boneIndex] = modelCtx.globalInverseTransform * animatedBoneGlobal * offsetMatrix;
                }
                else {
                    // Fallback to identity for missing keys in this model's rigged hierarchy
                    if (boneIndex < finalBoneMatrices.size()) {
                        finalBoneMatrices[boneIndex] = Matrix4x4::identity();
                    }
                }
            }
        }
    }


    // Ensure finalBoneMatrices is sized correctly for any remaining bones
    if (finalBoneMatrices.size() < scene.boneData.boneNameToIndex.size()) {
        finalBoneMatrices.resize(scene.boneData.boneNameToIndex.size(), Matrix4x4::identity());
    }

    // --- 0. Ad�m: Performans �nbelle�i Haz�rl��� ---
    if (animation_groups_dirty || animation_groups.empty()) {
        animation_groups.clear();
        std::unordered_map<void*, size_t> transformToGroup;

        // [PERF] Pre-flight filter: only add a triangle to animation_groups
        // if the object it represents actually has keyframe data (timeline
        // track OR file animation track). Without this, scatter-dense scenes
        // with thousands of static HittableInstance wrappers turn the
        // per-frame group loop into O(scene-size). With it, the loop only
        // visits truly animated nodes — which matches what processAnimations
        // does in Play mode (iterates timeline.tracks, not the full scene).
        auto isNodeAnimated = [&](const std::string& nodeName) -> bool {
            if (nodeName.empty()) return false;
            if (!scene.timeline.tracks.empty() &&
                scene.timeline.tracks.find(nodeName) != scene.timeline.tracks.end()) {
                return true;
            }
            for (const auto& anim : scene.animationDataList) {
                if (!anim) continue;
                if (anim->positionKeys.count(nodeName) > 0 ||
                    anim->rotationKeys.count(nodeName) > 0 ||
                    anim->scalingKeys.count(nodeName) > 0) {
                    return true;
                }
            }
            return false;
        };

        // Local helper: add a triangle to its appropriate animation group.
        // Returns the group index (or SIZE_MAX if rejected). Skinned meshes
        // are always added (bone transforms drive vertex updates regardless
        // of keyframe presence).
        auto addTriangleToGroups = [&](const std::shared_ptr<Triangle>& tri,
                                       const std::string& overrideName = std::string()) -> size_t {
            if (!tri) return SIZE_MAX;
            void* transformKey = tri->getTransformPtr();
            if (!transformKey) return SIZE_MAX;

            const bool isSkinned = tri->hasSkinData();
            const std::string& nameForKeyCheck = overrideName.empty() ? tri->getNodeName() : overrideName;
            if (!isSkinned && !isNodeAnimated(nameForKeyCheck)) {
                return SIZE_MAX; // static — no group needed
            }

            auto it = transformToGroup.find(transformKey);
            size_t idx;
            if (it == transformToGroup.end()) {
                idx = animation_groups.size();
                transformToGroup[transformKey] = idx;
                AnimatableGroup newGroup;
                newGroup.nodeName = nameForKeyCheck;
                newGroup.isSkinned = isSkinned;
                newGroup.transformHandle = tri->getTransformHandle();
                animation_groups.push_back(newGroup);
            } else {
                idx = it->second;
            }
            animation_groups[idx].triangles.push_back(tri);
            return idx;
        };

        for (auto& obj : scene.world.objects) {
            if (!obj) continue;

            if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                addTriangleToGroups(tri);
                continue;
            }

            // [HITTABLEINSTANCE FIX] HittableInstance wrappers hide their
            // triangles inside source_triangles. We still need to drive
            // their TransformHandle from keyframes, but match the track by
            // the wrapper's node_name (which is what the user sees / keys
            // in the timeline), not the inner triangle's name — those can
            // differ for instanced meshes.
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                if (inst->source_triangles && !inst->source_triangles->empty()) {
                    const std::string& instName = inst->node_name;
                    size_t group_idx = SIZE_MAX;
                    for (auto& src_tri : *inst->source_triangles) {
                        size_t idx = addTriangleToGroups(src_tri, instName);
                        if (idx != SIZE_MAX && group_idx == SIZE_MAX) {
                            group_idx = idx;
                        }
                    }
                    if (group_idx != SIZE_MAX) {
                        animation_groups[group_idx].instances.push_back(inst);
                    }
                }
            }
        }
        animation_groups_dirty = false;
        //SCENE_LOG_INFO("Animation groups rebuilt: " + std::to_string(animation_groups.size()) + " groups.");
    }

    // --- 2. Ad�m: Gruplar� Animasyon T�r�ne G�re G�ncelle ---
    for (auto& group : animation_groups) {
        if (group.triangles.empty()) continue;

        if (group.isSkinned) {
            // Skeleton animation modifies geometry
            geometry_changed = true;
            if (apply_cpu_skinning) {
                for (auto& tri : group.triangles) {
                    tri->apply_skinning(static_cast<const std::vector<Matrix4x4>&>(finalBoneMatrices));
                }
            }
        }
        else {
            // --- RIGID ANIMATION ---
            bool nodeHasAnimation = false;
            for (const auto& anim : scene.animationDataList) {
                if (!anim) continue;
                if (anim->positionKeys.count(group.nodeName) > 0 ||
                    anim->rotationKeys.count(group.nodeName) > 0 ||
                    anim->scalingKeys.count(group.nodeName) > 0) {
                    nodeHasAnimation = true;
                    break;
                }
            }

            if (nodeHasAnimation && animatedGlobalNodeTransforms.count(group.nodeName) > 0) {
                Matrix4x4 animTransform = animatedGlobalNodeTransforms[group.nodeName];
                if (group.transformHandle) {
                    group.transformHandle->setBase(animTransform);
                    group.transformHandle->setCurrent(Matrix4x4::identity());
                    if (apply_cpu_skinning) {
                        for (auto& tri : group.triangles) tri->updateTransformedVertices();
                    }
                    geometry_changed = true;
                }
            }
            else {
                if (animatedGlobalNodeTransforms.count(group.nodeName) > 0) {
                    Matrix4x4 staticTransform = animatedGlobalNodeTransforms[group.nodeName];
                    if (group.transformHandle) {
                        group.transformHandle->setBase(staticTransform);
                        group.transformHandle->setCurrent(Matrix4x4::identity());
                        if (apply_cpu_skinning) {
                            for (auto& tri : group.triangles) tri->updateTransformedVertices();
                        }
                        geometry_changed = true;
                    }
                }

                // --- MANUAL KEYFRAME SUPPORT ---
                if (!nodeHasAnimation && !scene.timeline.tracks.empty()) {
                    extern RenderSettings render_settings;
                    int current_frame = static_cast<int>(current_time * render_settings.animation_fps);
                    auto track_it = scene.timeline.tracks.find(group.nodeName);
                    if (track_it != scene.timeline.tracks.end() && !track_it->second.keyframes.empty()) {
                        Keyframe kf = track_it->second.evaluate(current_frame);
                        if (kf.has_transform) {
                            Matrix4x4 translation = Matrix4x4::translation(kf.transform.position);
                            float rx = kf.transform.rotation.x * (3.14159265f / 180.0f);
                            float ry = kf.transform.rotation.y * (3.14159265f / 180.0f);
                            float rz = kf.transform.rotation.z * (3.14159265f / 180.0f);
                            Matrix4x4 rotation = Matrix4x4::rotationZ(rz) * Matrix4x4::rotationY(ry) * Matrix4x4::rotationX(rx);
                            Matrix4x4 scale = Matrix4x4::scaling(kf.transform.scale);
                            Matrix4x4 final_transform = translation * rotation * scale;

                            if (group.transformHandle) {
                                group.transformHandle->setBase(final_transform);
                                group.transformHandle->setCurrent(Matrix4x4::identity());
                                if (apply_cpu_skinning) {
                                    for (auto& tri : group.triangles) tri->updateTransformedVertices();
                                }
                                geometry_changed = true;
                            }
                        }
                    }
                }
            }
        }

        // [HITTABLEINSTANCE SYNC] If this group's transform handle was just
        // updated AND it is shared with one or more HittableInstance wrappers
        // (cached at build time), propagate the new world matrix into each
        // wrapper's own `transform` field. Vulkan's updateInstanceTransforms
        // reads inst->transform directly when refitting the TLAS, so without
        // this propagation HittableInstance-wrapped meshes stay frozen at
        // their pre-render pose.
        if (group.transformHandle &&
            (!group.instances.empty() || !group.isSkinned)) {
            const Matrix4x4& worldMatrix = group.transformHandle->getMatrix();
            for (auto& inst : group.instances) {
                if (inst) inst->setTransform(worldMatrix);
            }
            // [PERF] Record this node's new transform so the worker thread
            // can drive a targeted backend update for the few nodes that
            // actually moved this frame, instead of scanning every instance
            // in the scene. Skinned groups update via dispatchSkinning, so
            // their transform isn't pushed here.
            if (!group.isSkinned && !group.nodeName.empty()) {
                pending_anim_transform_updates.emplace_back(group.nodeName, worldMatrix);
            }
        }
    }

    // --- 3. Ad�m: I��k ve Kamera Animasyonlar� (from files AND manual keyframes) ---

    // FILE-BASED Light Animation (existing code)
    for (auto& light : scene.lights) {
        bool lightHasAnimation = false;
        for (const auto& anim : scene.animationDataList) {
            if (!anim) continue;
            if (anim->positionKeys.count(light->nodeName) > 0 ||
                anim->rotationKeys.count(light->nodeName) > 0 ||
                anim->scalingKeys.count(light->nodeName) > 0) {
                lightHasAnimation = true;
                break;
            }
        }

        if (lightHasAnimation && animatedGlobalNodeTransforms.count(light->nodeName) > 0) {
            Matrix4x4 finalTransform = animatedGlobalNodeTransforms[light->nodeName];
            light->position = finalTransform.transform_point(Vec3(0, 0, 0));
            if (light->type() == LightType::Directional || light->type() == LightType::Spot) {
                light->direction = finalTransform.transform_vector(Vec3(0, 0, -1)).normalize();
            }

            // Link Nishita Sun to first Directional Light
            // Link Nishita Sun to first Directional Light
            if (light->type() == LightType::Directional) {
                world.setSunDirection(-light->direction);
                world.setSunIntensity(light->intensity);
            }
        }

        // MANUAL KEYFRAME Light Animation (NEW!)
        if (!lightHasAnimation && !scene.timeline.tracks.empty()) {
            extern RenderSettings render_settings;
            int current_frame = static_cast<int>(current_time * render_settings.animation_fps);

            auto track_it = scene.timeline.tracks.find(light->nodeName);
            if (track_it != scene.timeline.tracks.end() && !track_it->second.keyframes.empty()) {
                Keyframe kf = track_it->second.evaluate(current_frame);

                if (kf.has_light) {
                    // Only apply properties that were explicitly keyed
                    if (kf.light.has_position) {
                        light->position = kf.light.position;
                    }
                    if (kf.light.has_color) {
                        light->color = kf.light.color;
                    }
                    if (kf.light.has_intensity) {
                        light->intensity = kf.light.intensity;
                    }

                    if (kf.light.has_direction) {
                        if (light->type() == LightType::Directional || light->type() == LightType::Spot) {
                            light->direction = kf.light.direction.normalize();

                            if (light->type() == LightType::Directional) {
                                world.setSunDirection(-light->direction);
                                world.setSunIntensity(light->intensity);
                            }
                        }
                    }
                }
            }
        }
    }

    // FILE-BASED Camera Animation (existing code)
    bool cameraHasAnimation = false;
    if (scene.camera) {
        for (const auto& anim : scene.animationDataList) {
            if (!anim) continue;
            if (anim->positionKeys.count(scene.camera->nodeName) > 0 ||
                anim->rotationKeys.count(scene.camera->nodeName) > 0 ||
                anim->scalingKeys.count(scene.camera->nodeName) > 0) {
                cameraHasAnimation = true;
                break;
            }
        }
    }

    if (scene.camera && cameraHasAnimation && animatedGlobalNodeTransforms.count(scene.camera->nodeName) > 0) {
        // Apply global inverse REMOVED. Static camera works without it.
        // We suspect globalInverse was introducing the "tilt" (sola yat�k).
        // We KEEP the manual UP vector flip because user confirmed it fixed "tepetaklak". (Upside down)

        Matrix4x4 animTransform = animatedGlobalNodeTransforms[scene.camera->nodeName];

        Vec3 pos = animTransform.transform_point(Vec3(0, 0, 0));
        // Blender cameras usually point down -Z.
        Vec3 forward = animTransform.transform_vector(Vec3(0, 0, -1)).normalize();

        // FIX: Force Global Up (0, 1, 0) to prevent unwanted Roll/Tilt (sola yat�k).
        // This mimics "Track To" constraint behavior where the camera stays level.
        // If the camera needs to roll (bank), this line should be reverted to use transformed up.
        // Note on position precision: Small errors (e.g. -0.003 instad of 0) are due to Linear vs Bezier interpolation differences.
        Vec3 up = Vec3(0, 1, 0);

        scene.camera->lookfrom = pos;
        scene.camera->lookat = pos + forward;

        // Handle Gimbal Lock: If looking straight up/down, keep previous up or logic might fail slightly.
        if (abs(Vec3::dot(forward, up)) < 0.99f) {
            scene.camera->vup = up;
        }
        scene.camera->update_camera_vectors();
    }

    // MANUAL KEYFRAME Camera Animation (NEW!)
    if (scene.camera && !cameraHasAnimation && !scene.timeline.tracks.empty()) {
        extern RenderSettings render_settings;
        int current_frame = static_cast<int>(current_time * render_settings.animation_fps);

        auto track_it = scene.timeline.tracks.find(scene.camera->nodeName);
        if (track_it != scene.timeline.tracks.end() && !track_it->second.keyframes.empty()) {
            Keyframe kf = track_it->second.evaluate(current_frame);

            if (kf.has_camera) {
                // Only apply properties that were explicitly keyed
                if (kf.camera.has_position) {
                    scene.camera->lookfrom = kf.camera.position;
                }
                if (kf.camera.has_target) {
                    scene.camera->lookat = kf.camera.target;
                }
                if (kf.camera.has_fov) {
                    scene.camera->vfov = kf.camera.fov;
                }
                if (kf.camera.has_focus) {
                    scene.camera->focus_dist = kf.camera.focus_distance;
                }
                if (kf.camera.has_aperture) {
                    scene.camera->lens_radius = kf.camera.lens_radius;
                }

                // Update camera vectors
                Vec3 forward = (scene.camera->lookat - scene.camera->lookfrom).normalize();
                Vec3 up = Vec3(0, 1, 0);
                if (abs(Vec3::dot(forward, up)) < 0.99f) {
                    scene.camera->vup = up;
                }
                scene.camera->update_camera_vectors();
            }
        }
    }

    // --- MATERIAL KEYFRAME EVALUATION (OPTIMIZED!) ---
    // OPTIMIZATION: Instead of iterating through ALL objects (10M+), iterate only through
    // timeline tracks that have material keyframes. This is O(tracks) instead of O(objects).
    if (!scene.timeline.tracks.empty()) {
        extern RenderSettings render_settings;
        int current_frame = static_cast<int>(current_time * render_settings.animation_fps);

        // Build a cache of node names to material IDs for fast lookup (done once)
        // OPTIMIZATION: This cache should ideally be built once when scene loads,
        // but for now we only process tracks that have keyframes - much faster than 10M objects

        std::vector<uint16_t> keyframedMaterialIds;
        for (auto& [track_name, track] : scene.timeline.tracks) {
            // Skip tracks without material keyframes
            if (track.keyframes.empty()) continue;

            // Evaluate keyframe at current frame
            Keyframe kf = track.evaluate(current_frame);
            if (!kf.has_material) continue;

            // Get material ID from keyframe
            uint16_t mat_id = kf.material.material_id;

            // Fallback: if material_id wasn't stored in the keyframe (INVALID), find it by node name.
            // NOTE: ID 0 is a valid material (first slot) — only INVALID_MATERIAL_ID means "not set".
            if (mat_id == MaterialManager::INVALID_MATERIAL_ID) {
                for (const auto& obj : scene.world.objects) {
                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                    if (tri && tri->getNodeName() == track_name) {
                        mat_id = tri->getMaterialID();
                        break;
                    }
                }
            }

            if (mat_id == MaterialManager::INVALID_MATERIAL_ID) continue;

            // Get material and apply keyframe values
            Material* mat_ptr = MaterialManager::getInstance().getMaterial(mat_id);
            if (mat_ptr && mat_ptr->gpuMaterial) {
                // Apply interpolated material properties to GpuMaterial
                kf.material.applyTo(*mat_ptr->gpuMaterial);

                // Also update CPU-side PrincipledBSDF properties so that
                // capturePBRMaterialSnapshot in updateBackendMaterials reads
                // the keyframed values instead of overriding them with stale CPU data.
                if (auto* pbsdf = dynamic_cast<PrincipledBSDF*>(mat_ptr)) {
                    pbsdf->albedoProperty.color = kf.material.albedo;
                    pbsdf->roughnessProperty.color = Vec3(kf.material.roughness);
                    pbsdf->metallicProperty.intensity = kf.material.metallic;
                    pbsdf->specularProperty.intensity = kf.material.specular;
                    pbsdf->emissionProperty.color = kf.material.emission;
                    pbsdf->ior = kf.material.ior;
                    pbsdf->transmission = kf.material.transmission;
                    pbsdf->opacityProperty.alpha = kf.material.opacity;
                }
                if (std::find(keyframedMaterialIds.begin(), keyframedMaterialIds.end(), mat_id) == keyframedMaterialIds.end()) {
                    keyframedMaterialIds.push_back(mat_id);
                }
            }
        }

        // Push only the animated material slots instead of re-uploading the full
        // material buffer every frame during timeline playback.
        for (uint16_t materialId : keyframedMaterialIds) {
            this->updateBackendMaterial(scene, materialId);
        }

        // --- WORLD KEYFRAME EVALUATION (NEW!) ---
        auto world_track_it = scene.timeline.tracks.find("World");
        if (world_track_it != scene.timeline.tracks.end() && !world_track_it->second.keyframes.empty()) {
            Keyframe kf = world_track_it->second.evaluate(current_frame);
            if (kf.has_world) {
                const WorldKeyframe& wk = kf.world;

                // Apply each property independently based on its flag

                // Background Color
                if (wk.has_background_color) {
                    world.setColor(wk.background_color);
                    scene.background_color = wk.background_color;
                }
                // Background Strength
                if (wk.has_background_strength) {
                    world.setColorIntensity(wk.background_strength);
                }
                // HDRI Rotation
                if (wk.has_hdri_rotation) {
                    world.setHDRIRotation(wk.hdri_rotation);
                }

                // Nishita Sky - Get params once, update as needed, set once at end
                bool need_nishita_update = (wk.has_sun_elevation || wk.has_sun_azimuth ||
                    wk.has_sun_intensity || wk.has_sun_size ||
                    wk.has_atmosphere_intensity ||
                    wk.has_air_density || wk.has_dust_density ||
                    wk.has_ozone_density || wk.has_altitude ||
                    wk.has_mie_anisotropy ||
                    wk.has_cloud_density || wk.has_cloud_coverage ||
                    wk.has_cloud_scale || wk.has_cloud_offset);

                if (need_nishita_update) {
                    NishitaSkyParams np = world.getNishitaParams();

                    // Sun properties
                    if (wk.has_sun_elevation) np.sun_elevation = wk.sun_elevation;
                    if (wk.has_sun_azimuth) np.sun_azimuth = wk.sun_azimuth;
                    if (wk.has_sun_intensity) np.sun_intensity = wk.sun_intensity;
                    if (wk.has_atmosphere_intensity) np.atmosphere_intensity = wk.atmosphere_intensity;
                    if (wk.has_sun_size) np.sun_size = wk.sun_size;

                    // Recalculate sun direction if elevation or azimuth changed
                    if (wk.has_sun_elevation || wk.has_sun_azimuth) {
                        float elRad = np.sun_elevation * 3.14159265f / 180.0f;
                        float azRad = np.sun_azimuth * 3.14159265f / 180.0f;
                        np.sun_direction = make_float3(
                            cosf(elRad) * sinf(azRad),
                            sinf(elRad),
                            cosf(elRad) * cosf(azRad)
                        );
                    }

                    // Atmosphere properties
                    if (wk.has_air_density) np.air_density = wk.air_density;
                    if (wk.has_dust_density) np.dust_density = wk.dust_density;
                    if (wk.has_ozone_density) np.ozone_density = wk.ozone_density;
                    if (wk.has_altitude) np.altitude = wk.altitude;
                    if (wk.has_mie_anisotropy) np.mie_anisotropy = wk.mie_anisotropy;

                    // Cloud properties
                    if (wk.has_cloud_density) np.cloud_density = wk.cloud_density;
                    if (wk.has_cloud_coverage) np.cloud_coverage = wk.cloud_coverage;
                    if (wk.has_cloud_scale) np.cloud_scale = wk.cloud_scale;
                    if (wk.has_cloud_offset) {
                        np.cloud_offset_x = wk.cloud_offset_x;
                        np.cloud_offset_z = wk.cloud_offset_z;
                    }

                    // Enable clouds if coverage > 0
                    if (wk.has_cloud_coverage && np.cloud_coverage > 0.0f) {
                        np.clouds_enabled = 1;
                    }

                    world.setNishitaParams(np);
                }

                if (wk.has_weather_params) {
                    WeatherParams weather = world.getWeatherParams();
                    weather.enabled = wk.weather_enabled;
                    weather.type = wk.weather_type;
                    weather.intensity = wk.weather_intensity;
                    weather.density = wk.weather_density;
                    weather.wind_direction = make_float3(
                        wk.weather_wind_direction.x,
                        wk.weather_wind_direction.y,
                        wk.weather_wind_direction.z);
                    weather.wind_speed = wk.weather_wind_speed;
                    weather.precipitation_scale = wk.weather_precipitation_scale;
                    weather.visibility = wk.weather_visibility;
                    weather.surface_wetness_output = wk.weather_surface_wetness;
                    weather.surface_accumulation_output = wk.weather_surface_accumulation;
                    weather.visual_mode = wk.weather_visual_mode;
                    weather.surface_response_enabled = wk.weather_surface_response_enabled;
                    world.setWeatherParams(weather);
                }
            }
        }
    }

    } // End of Legacy Else

    // --- 4. Ad�m: BVH G�ncelle (only if geometry changed) ---
    // OPTIMIZATION: Skip CPU BVH rebuild for camera-only or material-only animations
    if (geometry_changed) {
        auto embree_ptr = std::dynamic_pointer_cast<EmbreeBVH>(scene.bvh);
        if (embree_ptr) {
            embree_ptr->updateGeometryFromTrianglesFromSource(scene.world.objects);
        }
    }
    
    // ===========================================================================
    // HAIR SYSTEM UPDATE (Rigid & Skeletal Synchronization)
    // ===========================================================================
    // This MUST run after ALL bone/rigid transformations are final.
    if (hairSystem.getTotalStrandCount() > 0) {
        // We pass the final calculated bone matrices.
        // This handles guide skinning and BVH updates.
        hairSystem.updateAllTransforms(scene.world.objects, this->finalBoneMatrices);
        
        if (hairSystem.isBVHDirty()) {
            hairSystem.buildBVH(true);
            uploadHairToGPU();
        }
    }

    return geometry_changed;
}

void Renderer::render_Animation(SDL_Surface* surface, SDL_Window* window, SDL_Texture* raytrace_texture, SDL_Renderer* renderer,
    const int total_samples_per_pixel, const int samples_per_pass, float fps, float duration, int start_frame, int end_frame, SceneData& scene,
    const std::string& output_folder, bool use_denoiser, float denoiser_blend,
    Backend::IBackend* backend, bool use_gpu, UIContext* ui_ctx) {

    render_finished = false;
    rendering_complete = false;
    rendering_in_progress = true;
    rendering_stopped_cpu = false;
    rendering_stopped_gpu = false;

    // [HITTABLEINSTANCE FIX] Force animation_groups rebuild at render start.
    // The new building logic includes HittableInstance source_triangles
    // (which the old Triangle-only logic missed), but the cache is sticky
    // — once built (e.g. during a prior viewport playback), the rebuild
    // branch is skipped on subsequent calls. Without invalidating here,
    // the worker re-uses an older Triangle-only cache and the cube stays
    // frozen at its pre-render pose. Cheap: rebuild costs one O(N) pass.
    animation_groups_dirty = true;
    
    // Reset pause state at start of new animation render
    extern std::atomic<bool> rendering_paused;
    rendering_paused = false;

    // Backend is already synced to this->m_backend in Main.cpp or passed via parameter
    if (!m_backend && backend) m_backend = backend;
      // LOCK VIEWPORT/CAMERA INPUT during animation render
    if (ui_ctx) {
        ui_ctx->render_settings.animation_render_locked = true;
    }

    unsigned int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    auto start_time = std::chrono::steady_clock::now();
    float frame_time = 1.0f / fps;

    extern RenderSettings render_settings;
    extern bool g_hasOptix; // Ensure we can access global flag
    extern bool g_hasVulkan;

    // DISABLE GRID/OVERLAYS FOR ANIMATION RENDER
    bool original_render_mode = render_settings.is_final_render_mode;
    render_settings.is_final_render_mode = true;

    // [SAMPLE COUNT FIX] OptixWrapper::isAccumulationComplete reads the
    // GLOBAL render_settings.max_samples — NOT the per-frame target we
    // pass via setRenderParams (OptixBackend::setRenderParams ignores
    // samplesPerPixel entirely). If the user has the viewport's
    // max_samples set low (e.g. 1 for fast interactive editing), OptiX
    // declares accumulation complete after one sample, the worker exits
    // the inner loop, and what gets denoised+saved is essentially a
    // single noisy sample (denoiser ghosting around moving objects is
    // the giveaway). Override the global for the duration of the
    // animation render and restore it on exit. Vulkan also reads this
    // field for some shader-side limits; aligning it avoids divergent
    // behavior between the two backends.
    const int original_max_samples = render_settings.max_samples;
    render_settings.max_samples = total_samples_per_pixel;

    // Frame range is validated in Main.cpp before calling this function
    // We trust the values passed as parameters
    SCENE_LOG_INFO("render_Animation: Frame range " + std::to_string(start_frame) + " - " + std::to_string(end_frame) + 
                   " (" + std::to_string(end_frame - start_frame + 1) + " frames)");
    SCENE_LOG_INFO("render_Animation: " + std::to_string(total_samples_per_pixel) + " samples per frame, " + 
                   std::to_string(fps) + " FPS, Mode: " + (use_gpu ? (render_settings.use_vulkan ? "Vulkan" : "OptiX") : "CPU"));

    int total_frames = end_frame - start_frame + 1;
    if (total_frames <= 0) {
        SCENE_LOG_ERROR("Invalid frame range! Aborting animation render.");
        render_finished = true;
        rendering_complete = true;
        rendering_in_progress = false;
        render_settings.is_final_render_mode = original_render_mode;
        render_settings.max_samples = original_max_samples;
        return;
    }

    if (!output_folder.empty()) {
        std::filesystem::create_directories(output_folder);
        SCENE_LOG_INFO("Animation frames will be saved to: " + output_folder);
    }

    SCENE_LOG_INFO("Starting animation render: " + std::to_string(total_frames) + " frames (Frame " +
        std::to_string(start_frame) + " to " + std::to_string(end_frame) + ") at " + std::to_string(fps) + " FPS");

    // Sync frame range to UI context for accurate progress display
    if (ui_ctx) {
        ui_ctx->render_settings.animation_start_frame = start_frame;
        ui_ctx->render_settings.animation_end_frame = end_frame;
        ui_ctx->render_settings.animation_total_frames = total_frames;
    }

    // Check if GPU backend is valid (OptiX or Vulkan)
    bool run_gpu = use_gpu && m_backend && (g_hasOptix || g_hasVulkan);
    bool is_vulkan = render_settings.use_vulkan;
    bool is_optix = render_settings.use_optix;

    if (use_gpu && !run_gpu) {
        SCENE_LOG_WARN("GPU backend requested but not available/valid. Falling back to CPU.");
    }

    // Sequence render now uses the active viewport resolution — one resolution
    // setting drives everything (viewport, single-frame final, sequence). The
    // user adjusts resolution via the System panel before starting the render.
    // Keep the legacy final_render_width/height fields synced for project
    // save/load + any other code paths still reading them.
    render_settings.final_render_width = image_width;
    render_settings.final_render_height = image_height;
    int saved_image_width = image_width;
    int saved_image_height = image_height;

    for (int frame = start_frame; frame <= end_frame; ++frame) {

        // Update BOTH global render_settings AND ui_ctx for UI synchronization
        render_settings.animation_current_frame = frame;
        if (ui_ctx) {
            ui_ctx->render_settings.animation_current_frame = frame;
        }

        if (rendering_stopped_cpu || rendering_stopped_gpu) {
            SCENE_LOG_WARN("Animation rendering stopped by user at frame " + std::to_string(frame));
            break;
        }

        // Clear CPU buffers anyway (for safety/consistency)
        std::fill(frame_buffer.begin(), frame_buffer.end(), Vec3(0.0f));
        std::fill(sample_counts.begin(), sample_counts.end(), 0);
        // REMOVED: SDL_FillRect(surface, NULL, 0) to prevent main window blackout

        // Use absolute time based on frame number (not relative to start_frame)
        float current_time = frame * frame_time;

        std::string backend_name = run_gpu ? (is_vulkan ? "Vulkan" : "OptiX") : "CPU";
        SCENE_LOG_INFO("Rendering frame " + std::to_string(frame) + "/" + std::to_string(end_frame) +
            " at time " + std::to_string(current_time) + "s (Mode: " + backend_name + ")");

        // --- SYNC TIMELINE FRAME FOR MATERIAL KEYFRAME EVALUATION ---
        // CRITICAL: updateAnimationState's material keyframe code (lines 797-842) reads
        // render_settings.animation_fps to calculate current_frame. We must sync the 
        // timeline frame counter so material evaluation works correctly!
        extern RenderSettings render_settings;
        render_settings.animation_current_frame = frame;
        render_settings.animation_playback_frame = frame;
        scene.timeline.current_frame = frame;

        // Returns true if geometry changed
        // Disable CPU skinning if running on OptiX to save performance and prevent crashes
        // We only need CPU skinning for CPU rendering or if we need to update CPU BVH
        bool geometry_changed = this->updateAnimationState(scene, current_time, !run_gpu);

        // --- SIMULATION (fluid / gas / particles) PER-FRAME DRIVE ---
        // The sequence-render worker is the SOLE owner of the sim timeline here
        // (the UI driver is gated off by render_owns_sim in SceneUI::draw).
        // render_Animation historically drove only wind / FFT ocean / animator,
        // so fluid splat, foam and the SurfaceSDF surface were ABSENT from
        // sequence output. Deterministically bake the sim to this frame and
        // rebuild the SurfaceSDF volume + particle/foam render instances. The
        // particle/foam bridge self-flags the backend-rebuild globals
        // (g_scene_geometry_generation / g_*_rebuild_pending / g_gpu_refit_pending),
        // which the GPU geometry-sync block below consumes BEFORE traceRays.
        const bool sim_active = scene.anySimulationRuntimeEnabled();
        bool sim_structural_change = false;
        if (sim_active) {
            extern std::atomic<uint64_t> g_scene_geometry_generation;
            const float seq_fps = static_cast<float>(std::max(1, render_settings.animation_fps));
            const uint64_t gen_before = g_scene_geometry_generation.load(std::memory_order_acquire);
            scene.bakeSimulationForRenderFrame(frame, seq_fps, /*enable_rt_geometry*/ true,
                                               /*cache_frames*/ false);
            sim_structural_change =
                g_scene_geometry_generation.load(std::memory_order_acquire) != gen_before;
        }

        // --- WIND ANIMATION ---
        // Apply wind simulation for this frame
        // FIX: Use same pattern as Play Mode (Main.cpp line 691-697) for consistent behavior
        if (run_gpu && m_backend && InstanceManager::getInstance().getGroupCount() > 0) {
            // Calculate wind transforms on CPU (same as Play Mode)
            FoliageWindUpdateStats wind_stats = InstanceManager::getInstance().updateWind(current_time, scene, this->m_backend);

            // Efficiently update instance transforms on GPU (no full rebuild)
            // This is the critical step that was missing - ensures GPU TLAS has updated matrices
            if (wind_stats.any_cpu_update || wind_stats.gpu_deform_applied) {
                m_backend->updateInstanceTransforms(scene.world.objects);
            }

            // Wind updates don't require geometry_changed = true
            // Only instance transforms changed, not vertex data
        }

        // --- VDB VOLUME ANIMATION (FIX) ---
        // Update VDB sequences for current frame (loads new grid from disk if needed)
        scene.updateVDBVolumesFromTimeline(frame);

        // Sync VDBs to GPU if running GPU backend
        // This ensures the new grid data is uploaded to GPU memory
        if (run_gpu && ui_ctx) {
            SceneUI::syncVDBVolumesToGPU(*ui_ctx);
            // Note: syncVDBVolumesToGPU handles geometry flag updates internally if needed
        }

        // --- TERRAIN ANIMATION ---
        // Apply terrain keyframes for this frame (morphing animation)
        for (auto& [track_name, track] : scene.timeline.tracks) {
            // Check if this track has terrain keyframes
            bool has_terrain_kf = false;
            for (auto& kf : track.keyframes) {
                if (kf.has_terrain) {
                    has_terrain_kf = true;
                    break;
                }
            }

            if (has_terrain_kf) {
                // Find terrain by name
                auto& terrains = TerrainManager::getInstance().getTerrains();
                for (auto& terrain : terrains) {
                    if (terrain.name == track_name) {
                        TerrainManager::getInstance().updateFromTrack(&terrain, track, frame);
                        geometry_changed = true;  // Terrain morph = geometry change
                        break;
                    }
                }
            }
        }

        // --- WATER/FFT OCEAN ANIMATION ---
        // Apply water keyframes for this frame (FFT parameter animation)
        for (auto& [track_name, track] : scene.timeline.tracks) {
            // Check if this is a Water track (name starts with "Water_")
            if (track_name.rfind("Water_", 0) != 0) continue;
            
            // Check if this track has water keyframes
            bool has_water_kf = false;
            for (auto& kf : track.keyframes) {
                if (kf.has_water) {
                    has_water_kf = true;
                    break;
                }
            }
            
            if (has_water_kf) {
                // Extract water surface ID from track name (format: "Water_X")
                auto& waters = WaterManager::getInstance().getWaterSurfaces();
                for (auto& water : waters) {
                    std::string expected_name = "Water_" + std::to_string(water.id);
                    if (track_name == expected_name) {
                        WaterManager::getInstance().updateFromTrack(&water, track, frame);
                        if (water.material_id != MaterialManager::INVALID_MATERIAL_ID) {
                            WaterManager::getInstance().syncSurfaceMaterial(&water);
                            this->updateBackendMaterial(scene, water.material_id);
                        }
                        // FFT changes don't need geometry rebuild - they're shader-based
                        // But geometric waves do need BVH update
                        if (water.params.use_geometric_waves && water.animate_mesh) {
                            geometry_changed = true;
                        }
                        break;
                    }
                }
            }
        }

        // --- WATER ANIMATION UPDATE ---
        // Update FFT ocean simulation and geometric wave mesh animation
        // Frame delta: 1.0/fps gives the time for one frame
        float animFps = static_cast<float>(std::max(1, render_settings.animation_fps));
        float frame_delta = 1.0f / animFps;
        float frame_time = static_cast<float>(frame) / animFps;
        WaterUpdateResult waterUpdate = WaterManager::getInstance().update(frame_time);
        if (waterUpdate.mesh_changed) {
            geometry_changed = true;
        }

        // --- RENDER BUFFER SETUP ---
        // Use a dedicated off-screen surface for BOTH CPU and OptiX to prevent
        // access violations when render resolution != window size.
        SDL_Surface* target_surface = surface; // Default fallback
        SDL_Surface* render_surface = nullptr;

        // Sequence render mirrors the active viewport resolution.
        int rw = image_width;
        int rh = image_height;

        // Always create off-screen surface for animation
        render_surface = SDL_CreateRGBSurfaceWithFormat(0, rw, rh, 32, SDL_PIXELFORMAT_RGBA32);
        if (render_surface) {
            target_surface = render_surface;
        }

        if (run_gpu) {
            // --- GPU RENDER PATH (OptiX / Vulkan) ---

            // 1. Update Geometry if needed (skinning + topology changes)
            // Use updateSceneGeometry for both OptiX and Vulkan — it handles
            // bone matrix dispatch (GPU skinning) and TLAS refit/rebuild.
            //
            // [PATH SELECTION] Pick exactly ONE GPU sync path per animation
            // frame based on whether this frame involves skinning. Calling
            // both (as a previous iteration did) does the TLAS rebuild twice
            // — wasted GPU work plus a transient window between the two
            // rebuilds where descriptor sets can briefly reference an
            // about-to-be-replaced AS handle. On Vulkan that occasionally
            // shows up as a silent driver hang mid-render.
            //
            //   • Skinned frame  (finalBoneMatrices non-empty)  →
            //         updateSceneGeometry: dispatches GPU skinning into
            //         BLAS vertex buffers, then refits TLAS. m_vkInstances
            //         transforms stay correct (skin moves vertices, not
            //         instance matrices).
            //   • Rigid keyframe frame  (no bone matrices)     →
            //         updateInstanceTransforms: refreshes m_vkInstances
            //         from scene.world.objects transforms (the
            //         updateSceneGeometry fast-path cannot do this — it
            //         uses the cached values and would refit TLAS with a
            //         frame-0 transform), then refits TLAS once.
            // [FRAME 0 FIX] On the very first iteration of the sequence
            // render we MUST force a full sync regardless of whether
            // updateAnimationState reported a change. The pre-render
            // backend state (uploaded by syncActiveRenderBackendScene or
            // by processAnimations during the launch countdown) may not
            // match what the worker's updateAnimationState computes for
            // start_frame — the worker iterates a different code path
            // (animation_groups, raw Triangles only) than processAnimations
            // (mesh_cache, including HittableInstance wrappers). If
            // geometry_changed is false on the first frame, the unforced
            // path would skip the TLAS refit and the first 1–2 rendered
            // frames render with the stale pre-render transform.
            const bool first_anim_frame = (frame == start_frame);
            const bool needs_geometry_sync = geometry_changed || first_anim_frame;
            if (needs_geometry_sync && m_backend) {
                // HittableInstance wrappers are now synced inside
                // updateAnimationState per-group (see "HITTABLEINSTANCE
                // SYNC" block), so the per-frame scene.world.objects scan
                // that used to live here is no longer needed.
                if (!finalBoneMatrices.empty()) {
                    // Skinned mesh path: GPU skinning + full TLAS refit.
                    m_backend->updateSceneGeometry(scene.world.objects, finalBoneMatrices);
                } else {
                    // [PERF] Pick targeted vs full TLAS sync based on how many
                    // nodes actually moved this frame. This mirrors what
                    // Play-mode's processAnimations does (≤8 changed →
                    // per-node updateObjectTransform; otherwise full scan).
                    // On scatter-dense scenes where only a few cubes / lights
                    // are keyed, the targeted path skips the full
                    // syncInstanceTransforms(force=true) rebuild that scans
                    // every instance + does a dynamic_cast per source — that
                    // scan was the cost gap between Play mode and sequence
                    // render reported earlier.
                    constexpr size_t kTargetedTransformUpdateLimit = 8;
                    const bool use_targeted = !first_anim_frame &&
                        !pending_anim_transform_updates.empty() &&
                        pending_anim_transform_updates.size() <= kTargetedTransformUpdateLimit;

                    if (use_targeted) {
                        for (const auto& [node_name, m] : pending_anim_transform_updates) {
                            m_backend->updateObjectTransform(node_name, m);
                        }
                    } else {
                        m_backend->updateInstanceTransforms(scene.world.objects);
                    }
                }
            }
            if (waterUpdate.material_changed || waterUpdate.mesh_changed) {
                for (auto& water : WaterManager::getInstance().getWaterSurfaces()) {
                    if (water.material_id != MaterialManager::INVALID_MATERIAL_ID) {
                        WaterManager::getInstance().syncSurfaceMaterial(&water);
                        this->updateBackendMaterial(scene, water.material_id);
                    }
                }
            }

            // --- SIMULATION GEOMETRY → BACKEND ---
            // The particle/foam bridge + SurfaceSDF volume were (re)built by the
            // sim drive above. During a sequence render the main loop's rebuild
            // handlers are gated off (skip_backend_for_anim), so the worker must
            // push the sim geometry to the backend itself — and it MUST complete
            // before traceRays: a concurrent / last-minute AS mutation mid-trace
            // is a silent NVIDIA hang (see render_warmup_sequencing).
            if (sim_active && m_backend) {
                extern bool g_optix_rebuild_pending;
                extern bool g_vulkan_rebuild_pending;
                extern bool g_gpu_refit_pending;
                if (sim_structural_change) {
                    // Pool grew / new bridge group: rebuild the TLAS so it
                    // incorporates the InstanceManager particle/foam groups plus
                    // the SurfaceSDF volume. Synchronous — the worker is the sole
                    // backend owner here, so there is no concurrent AS mutation.
                    if (is_vulkan) {
                        // updateGeometry alone is the LIGHT path: it reads the
                        // InstanceManager groups (so it picks up the grown foam /
                        // particle pool) and REUSES cached BLAS via m_meshRegistry,
                        // rebuilding only the TLAS (+ any genuinely new source BLAS,
                        // e.g. the foam icosphere the first time it appears).
                        // We must NOT call rebuildAccelerationStructure() here — it
                        // DESTROYS every BLAS + texture + VDB buffer and clears the
                        // registry, forcing a full scene re-upload. Doing that on
                        // every foam pool-growth frame (foam is volatile) was the
                        // "sık rebuild / render waits rebuild" storm on Vulkan.
                        m_backend->updateGeometry(scene.world.objects);
                        // updateGeometry rebuilt m_orderedVDBInstances → re-sync the
                        // VDB SSBO so volume customIndex matches the new TLAS order.
                        if (ui_ctx) SceneUI::syncVDBVolumesToGPU(*ui_ctx);
                    } else {
                        // OptiX: rebuildBackendGeometry rebuilds the TLAS (incl.
                        // instance groups) + syncs volumetric data + hair, all
                        // synchronously on this thread.
                        this->rebuildBackendGeometry(scene);
                        // Re-apply GPU skinning if this is also a skinned frame —
                        // the full rebuild rebuilt BLAS from base triangles and
                        // dropped the bone deformation updateSceneGeometry applied.
                        if (!finalBoneMatrices.empty()) {
                            m_backend->updateSceneGeometry(scene.world.objects, finalBoneMatrices);
                        }
                    }
                } else if (g_gpu_refit_pending) {
                    // Particle motion only: cheap TLAS refit (no BLAS rebuild).
                    m_backend->updateInstanceTransforms(scene.world.objects);
                }
                // The worker consumed these flags; clear them so the post-render
                // UI loop doesn't re-process a stale pending rebuild.
                g_optix_rebuild_pending = false;
                g_vulkan_rebuild_pending = false;
                g_gpu_refit_pending = false;
            }

            // 2. Set Scene Params
            if (m_backend) {
                m_backend->setTime(static_cast<float>(frame) / animFps, frame_delta);
            }
            this->syncCameraToBackend(*scene.camera);
            if (m_backend) {
                m_backend->setLights(scene.lights);
                WorldData wd = this->world.getGPUData();
                m_backend->setWorldData(&wd);
                // If Vulkan backend, upload AtmosphereLUT host arrays so shaders can sample LUTs
                if (m_backend) {
                    auto* vulkanBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(m_backend);
                    if (vulkanBackend) {
                        auto* al = this->world.getLUT();
                        if (al && al->is_initialized()) vulkanBackend->uploadAtmosphereLUT(al);
                    }
                }
                m_backend->resetAccumulation();

                // [FIX] Set backend target samples to match animation setting.
                // Without this, Vulkan uses whatever m_targetSamples was set before
                // (viewport max_samples or default 1000), causing the render loop to
                // run far too many iterations per frame — appearing locked.
                {
                    Backend::RenderParams rp = {};
                    rp.imageWidth = target_surface->w;
                    rp.imageHeight = target_surface->h;
                    rp.samplesPerPixel = total_samples_per_pixel;
                    rp.minSamples = render_settings.min_samples;
                    rp.maxBounces = std::max(1, render_settings.max_bounces);
                    rp.diffuseBounces = std::clamp(render_settings.diffuse_bounces, 1, rp.maxBounces);
                    rp.transmissionBounces = std::clamp(render_settings.transmission_bounces, 1, rp.maxBounces);
                    // Inherit viewport's adaptive sampling toggle + threshold so a
                    // sequence render can converge early on smooth frames instead
                    // of always burning the full max_samples budget. Disabling
                    // the toggle in the UI still gives fixed-count behavior.
                    rp.useAdaptiveSampling = render_settings.use_adaptive_sampling;
                    rp.adaptiveThreshold = render_settings.variance_threshold;
                    m_backend->setRenderParams(rp);
                }
            }
            // 4. Render Loop (Accumulate Samples until target reached)
            std::vector<uint32_t> temp_framebuffer(target_surface->w * target_surface->h, 0u);

            // Render until max samples reached
            while (m_backend && !m_backend->isAccumulationComplete() && !rendering_stopped_gpu) {
                // PAUSE WAIT - Block here while paused, check stop flag periodically
                while (rendering_paused.load() && !rendering_stopped_gpu.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (rendering_stopped_gpu.load()) break;
                
                // Launch progressive render
                // Pass nullptr for window to disable title bar updates (headless).
                // [THREAD-SAFETY FIX] Pass nullptr for tex too — SDL_Texture is not
                // thread-safe. The worker thread calling SDL_UpdateTexture concurrently
                // with the main thread's SDL_RenderCopy on the same raytrace_texture
                // wedges the main thread inside the SDL2/D3D driver, freezing the UI
                // (frames still save because the worker is alive). Vulkan early-out
                // was relaxed to allow tex==nullptr; pixel data still flows via fb.
                 if (m_backend) {
                     void* framebuffer_ptr = (void*)&temp_framebuffer;
                     m_backend->renderProgressive(target_surface, nullptr, renderer,
                                                 target_surface->w, target_surface->h,
                                                 framebuffer_ptr, nullptr);
                 }
            }

            // Vulkan's rendered viewport path overlaps trace + tonemap/readback with
            // ping-pong frame slots. Sequence rendering owns the backend on this
            // worker thread, so drain the tail once per completed frame before the
            // next frame mutates TLAS/camera/world state. Without this, the viewport
            // can inherit a backlog after the sequence and feel dramatically slower
            // until a Solid<->Rendered mode switch forces a device idle.
            if (is_vulkan && m_backend) {
                m_backend->waitForCompletion();
            }
            
            // IMMEDIATE EXIT CHECK after GPU render loop
            if (rendering_stopped_gpu.load()) {
                SCENE_LOG_WARN("Animation render stopped during GPU frame " + std::to_string(frame));
                if (render_surface) { SDL_FreeSurface(render_surface); }
                break;
            }
        }
        else {
            // --- CPU RENDER PATH ---

            // Reset CPU accumulation for new frame
            resetCPUAccumulation();

            // Render until target samples reached
            // Use direct comparison with total_samples_per_pixel (animation setting)
            // instead of isCPUAccumulationComplete() which uses render_settings.final_render_samples
            while (cpu_accumulated_samples < total_samples_per_pixel && !rendering_stopped_cpu) {
                // PAUSE WAIT - Block here while paused, check stop flag periodically
                while (rendering_paused.load() && !rendering_stopped_cpu.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (rendering_stopped_cpu.load()) break;
                
                // For animation CPU render: pass 'total_samples_per_pixel' as the target
                // This ensures we reach the "final render samples" count, not viewport "max samples"
                render_progressive_pass(target_surface, window, scene, 1, total_samples_per_pixel);
            }
            
            // IMMEDIATE EXIT CHECK after CPU render loop
            if (rendering_stopped_cpu.load()) {
                SCENE_LOG_WARN("Animation render stopped during CPU frame " + std::to_string(frame));
                if (render_surface) { SDL_FreeSurface(render_surface); }
                break;
            }
        }

        // --- COMMON: Denoiser & Save ---

        if (use_denoiser) {
            SCENE_LOG_INFO("Applying denoiser to frame " + std::to_string(frame));
            // Note: For OptiX, launch_random... might have already denoised if internal settings were set,
            // but Renderer::applyOIDNDenoising works on SDL Surface, so it's safe to call again or instead.
            // If OptiX Wrapper already denoised, we might be double denoising?
            // OptixWrapper::launch... doesn't verify denoiser usage inside the loop shown in step 7/38.
            // But Main.cpp calls ray_renderer.applyOIDNDenoising for single frame.
            // So we call it here too.
            // Note: Renderer::applyOIDNDenoising works on SDL Surface
            applyOIDNDenoising(target_surface, 0, true, denoiser_blend);
        }

        // [POST-PROCESS] Apply the same viewport color processing (tonemap,
        // exposure, white balance, saturation, vignette) the user has dialed
        // in to the rendered surface before saving — so the disk output
        // matches what they see in the viewport. Previously sequence frames
        // bypassed this entirely and only got the backend's internal raw
        // tonemap. applyToneMappingToSurface is defined in Main.cpp and
        // forward-declared below; passing the destination surface as both
        // source and destination tells it to read 8-bit pixels in place
        // (the GPU backend already produced display-ready pixels — we just
        // re-grade them).
        if (ui_ctx) {
            extern void applyToneMappingToSurface(SDL_Surface*, SDL_Surface*, ColorProcessor&, Renderer*);
            extern void applyStylizeToSurface(SDL_Surface*, Renderer&, bool);
            applyToneMappingToSurface(target_surface, target_surface, ui_ctx->color_processor, run_gpu ? nullptr : this);
            if (run_gpu && stylizeMode.enabled) {
                // Pull the GPU backend's primary-hit AOVs so the sequence frame gets the
                // full surface-locked stylize (same as the CPU path), not the flat
                // screen-space fallback. Falls back to screen-space if the AOV pull fails.
                // cameraOrigin = lookfrom, matching what syncCameraToBackend pushes as the
                // GPU ray origin (depth is reconstructed from world position).
                // forceRefresh: this fires once per finished frame, and a sequence may keep
                // the camera static while objects move — so always re-pull regardless of hash.
                const bool aov_ready = scene.camera &&
                    fillStylizeAOVFromBackend(m_backend, *scene.camera, /*forceRefresh=*/true);
                applyStylizeToSurface(target_surface, *this, aov_ready);
            }
        }

        if (!output_folder.empty()) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/frame_%04d.png", output_folder.c_str(), frame);
            // Ensure texture is updated for display before saving
           // REMOVED: Unsafe SDL calls from worker thread. Main thread handles display.

           // REMOVED: Unsafe SDL calls from worker thread. Main thread handles display.

            if (SaveSurface(target_surface, filename)) {
                SCENE_LOG_INFO("Frame saved: " + std::string(filename));

                // --- UPDATE ANIMATION PREVIEW ---
                // Access global UI context to update preview buffer
                if (ui_ctx) {
                    std::lock_guard<std::mutex> lock(ui_ctx->animation_preview_mutex);
                    int w = target_surface->w;
                    int h = target_surface->h;
                    size_t pixel_count = w * h;

                    if (ui_ctx->animation_preview_buffer.size() != pixel_count) {
                        ui_ctx->animation_preview_buffer.resize(pixel_count);
                        ui_ctx->animation_preview_width = w;
                        ui_ctx->animation_preview_height = h;
                    }

                    // Copy pixels (ensure thread safety)
                    std::memcpy(ui_ctx->animation_preview_buffer.data(), target_surface->pixels, pixel_count * sizeof(uint32_t));
                    ui_ctx->animation_preview_ready = true;
                }
            }
            else {
                SCENE_LOG_ERROR("Failed to save frame: " + std::string(filename));
            }
        }

        // Cleanup temp surface
        if (render_surface) {
            SDL_FreeSurface(render_surface);
            render_surface = nullptr;
        }
        
        // FINAL STOP CHECK - exit loop immediately if stop was requested during save/denoising
        if (rendering_stopped_cpu.load() || rendering_stopped_gpu.load()) {
            SCENE_LOG_WARN("Animation render stopped after frame " + std::to_string(frame) + " processing");
            break;
        }
    }



    rendering_complete = true;
    // NOTE: rendering_in_progress is set to false AFTER backend state restoration
    // to prevent the main thread from accessing the backend during cleanup.
    render_finished = true;
    
    // RESTORE RENDER MODE
    render_settings.is_final_render_mode = original_render_mode;

    // [SAMPLE COUNT FIX] Restore the global max_samples we overrode at the
    // top so the viewport's interactive accumulation goes back to whatever
    // the user had configured.
    render_settings.max_samples = original_max_samples;

    // [FIX] Restore GPU backend to viewport settings so the viewport can resume
    // normal progressive rendering. Without this, the backend stays at the
    // animation's target sample count with accumulation "complete", causing the
    // viewport to show a stale cached frame indefinitely.
    if (run_gpu && m_backend) {
        if (is_vulkan) {
            m_backend->waitForCompletion();
        }
        Backend::RenderParams rp = {};
        rp.imageWidth = image_width;
        rp.imageHeight = image_height;
        rp.samplesPerPixel = render_settings.max_samples > 0 ? render_settings.max_samples : 100;
        rp.minSamples = render_settings.min_samples;
        rp.maxBounces = std::max(1, render_settings.max_bounces);
        rp.diffuseBounces = std::clamp(render_settings.diffuse_bounces, 1, rp.maxBounces);
        rp.transmissionBounces = std::clamp(render_settings.transmission_bounces, 1, rp.maxBounces);
        rp.useAdaptiveSampling = render_settings.use_adaptive_sampling;
        rp.adaptiveThreshold = render_settings.variance_threshold;
        m_backend->setRenderParams(rp);
        m_backend->resetAccumulation();
    }

    if (rendering_stopped_cpu.load() || rendering_stopped_gpu.load()) {
        extern bool g_optix_rebuild_pending;
        extern bool g_vulkan_rebuild_pending;
        extern bool g_gpu_refit_pending;
        g_optix_rebuild_pending = false;
        g_vulkan_rebuild_pending = false;
        g_gpu_refit_pending = false;
    }

    // Restore viewport resolution for CPU path
    if (!run_gpu && (image_width != saved_image_width || image_height != saved_image_height)) {
        resetResolution(saved_image_width, saved_image_height);
    }

    // UNLOCK viewport/camera input and disable animation mode
    // Set these AFTER all backend cleanup — the main thread checks these flags
    // to decide whether to access the backend.  Setting them early would let
    // the main thread call backend methods while we're still restoring state.
    rendering_in_progress = false;
    if (ui_ctx) {
        ui_ctx->is_animation_mode = false;
        ui_ctx->render_settings.animation_render_locked = false;
    }

    auto end_time = std::chrono::steady_clock::now();

    if (rendering_stopped_cpu || rendering_stopped_gpu) {
        SCENE_LOG_WARN("Animation rendering was stopped by user.");
    }
    else {
        SCENE_LOG_INFO("Animation rendering completed successfully!");
    }
}

void Renderer::rebuildBVH(SceneData& scene, bool use_embree, bool skip_sync) {
    if (!scene.initialized) {
        SCENE_LOG_WARN("Scene not initialized, BVH rebuild skipped.");
        return;
    }

    // Keep CPU geometry in sync with transform handles before building BVH.
    // Open-project deserialization already does this for static meshes, but
    // runtime import/add flows may leave triangles in local space.
    // Caller may pass skip_sync=true when it has just performed this work
    // (e.g. the GPU->CPU switch path) to avoid duplicating millions of
    // dynamic_pointer_cast + matrix-inverse + AABB-rebuild iterations.
    if (!skip_sync) {
        std::for_each(std::execution::par_unseq,
            scene.world.objects.begin(), scene.world.objects.end(),
            [](std::shared_ptr<Hittable>& obj) {
                if (!obj) return;

                if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                    if (tri->getTransformPtr() && !tri->hasAnySkinWeights()) {
                        tri->updateTransformedVertices();
                    }
                    return;
                }

                if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                    if (inst->syncTransformFromSourceTriangles()) {
                        return;
                    }
                    if (inst->source_triangles) {
                        for (auto& srcTri : *inst->source_triangles) {
                            if (srcTri && srcTri->getTransformPtr() && !srcTri->hasAnySkinWeights()) {
                                srcTri->updateTransformedVertices();
                            }
                        }
                    }
                }
            });
    }

    // Create a temporary list of ALL hittable objects for the BVH
    // This includes World Objects (Triangles), VDB Volumes, and Gas Volumes
    std::vector<std::shared_ptr<Hittable>> all_hittables;
    // Reserve estimation: objects + gas + vdb
    all_hittables.reserve(scene.world.objects.size() + scene.gas_volumes.size() + scene.vdb_volumes.size());

    // 1. Add standard objects (Triangles/Meshes)
    for (const auto& obj : scene.world.objects) {
        if (obj && obj->visible) {
            all_hittables.push_back(obj);
        }
    }

    // 2. Add Gas Volumes (they are Hittables)
    for (const auto& gas : scene.gas_volumes) {
        if (gas && gas->visible) all_hittables.push_back(gas);
    }

    // 3. Add VDB Volumes
    for (const auto& vdb : scene.vdb_volumes) {
        if (vdb && vdb->visible) all_hittables.push_back(vdb);
    }

    // Handle empty scene
    if (all_hittables.empty()) {
        scene.bvh = nullptr;  // Clear BVH for empty scene
        SCENE_LOG_INFO("Scene is empty, BVH cleared.");
        return;
    }

    if (use_embree) {
        auto embree_bvh = std::make_shared<EmbreeBVH>();
        embree_bvh->build(all_hittables);
        scene.bvh = embree_bvh;
    }
    else {
        scene.bvh = std::make_shared<ParallelBVHNode>(all_hittables, 0, all_hittables.size(), 0.0, 1.0, 0);
        SCENE_LOG_INFO("[RayTrophi: RT_BVH]  structure built successfully.");
    }

    // IMPORTANT: Always rebuild hair BVH when main BVH is rebuilt
    // This ensures hair-to-mesh and mesh-to-hair shadows are accurate
    if (hairSystem.getTotalStrandCount() > 0) {
        hairSystem.buildBVH(!hideInterpolatedHair);
    }
}

void Renderer::updateBVH(SceneData& scene, bool use_embree) {
    if (scene.world.objects.empty()) {
        scene.bvh = nullptr;
        return;
    }
    rebuildBVH(scene, use_embree);
}

void Renderer::create_scene(SceneData& scene, Backend::IBackend* backend, const std::string& model_path,
    std::function<void(int progress, const std::string& stage)> progress_callback,
    bool append, const std::string& import_prefix) {

    // Helper lambda for progress updates
    auto update_progress = [&](int progress, const std::string& stage) {
        if (progress_callback) {
            progress_callback(progress, stage);
        }
        };

    // Only clear scene if not appending
    if (!append) {
        update_progress(0, "Cleaning previous scene...");
        SCENE_LOG_INFO("========================================");
        SCENE_LOG_INFO("SCENE CLEANUP: Starting comprehensive cleanup...");
        SCENE_LOG_INFO("========================================");

        // ---- 1. Sahne verilerini s�f�rla ----
        scene.world.clear();
        scene.lights.clear();
        scene.animationDataList.clear();
        scene.boneData.clear();

        // ---- 1b. Clear per-model animator caches (prevents bone corruption) ----
        for (auto& ctx : scene.importedModelContexts) {
            if (ctx.animator) {
                ctx.animator->clear();
            }
            if (ctx.graph) {
                ctx.graph.reset();
            }
            ctx.members.clear();
        }
        scene.importedModelContexts.clear();

        // ---- 1c. Clear renderer's cached bone matrices ----
        this->finalBoneMatrices.clear();

        scene.camera = nullptr;
        scene.bvh = nullptr;
        scene.initialized = false;
        SCENE_LOG_INFO("[SCENE CLEANUP] Scene data structures cleaned.");

        update_progress(5, "Clearing materials...");

        // ---- 2. MaterialManager'� temizle ----
        size_t material_count_before = MaterialManager::getInstance().getMaterialCount();
        MaterialManager::getInstance().clear();
        SCENE_LOG_INFO("[MATERIAL CLEANUP] MaterialManager cleared: " + std::to_string(material_count_before) + " materials removed.");

        // ---- 3. CPU Texture Cache'leri temizle ----
        assimpLoader.clearTextureCache();

        // Reset OptiX scene-owned GPU state for the next project load without touching texture
        // ownership. Texture residency itself is cleaned by Texture::cleanup_gpu() above.
        if (g_hasOptix && backend) {
            try {
                Backend::OptixBackend* optixBackend = dynamic_cast<Backend::OptixBackend*>(backend);
                OptixWrapper* optix_gpu = optixBackend ? optixBackend->getOptixWrapper() : nullptr;
                if (optix_gpu) {
                    optix_gpu->clearScene();
                }
            }
            catch (std::exception& e) {
                SCENE_LOG_WARN("[GPU CLEANUP] Exception during OptiX scene reset: " + std::string(e.what()));
            }
        }

        SCENE_LOG_INFO("========================================");
        SCENE_LOG_INFO("SCENE CLEANUP: Completed successfully!");
        SCENE_LOG_INFO("========================================");
    }
    else {
        update_progress(0, "Appending to scene...");
        SCENE_LOG_INFO("Appending model to existing scene (no cleanup)");
    }

    update_progress(10, "Loading model file...");
    SCENE_LOG_INFO("Starting scene creation from: " + model_path);

    std::filesystem::path path(model_path);
    baseDirectory = path.parent_path().string() + "/";
    SCENE_LOG_INFO("Base directory set to: " + baseDirectory);

    // ---- 1. Geometri ve animasyon y�kle ----
    update_progress(15, "Loading geometry & animations...");
    SCENE_LOG_INFO("Loading model geometry and animations...");

    // Create a dedicated loader for this import to keep the aiScene alive
    auto newLoader = std::make_shared<AssimpLoader>();
    auto [loaded_triangles, loaded_animations, loaded_bone_data] = newLoader->loadModelToTriangles(model_path, nullptr, import_prefix);

    // Store the context
    SceneData::ImportedModelContext modelCtx;
    modelCtx.loader = newLoader;
    modelCtx.importName = newLoader->currentImportName;
    modelCtx.animGraphAssetKey = modelCtx.importName;
    modelCtx.hasAnimation = (newLoader->getScene() && newLoader->getScene()->mNumAnimations > 0);
    modelCtx.globalInverseTransform = loaded_bone_data.globalInverseTransform;

    update_progress(40, "Processing triangles...");

    if (loaded_triangles.empty()) {
        SCENE_LOG_ERROR("No triangle data, scene loading failed: " + model_path);
        SCENE_LOG_ERROR("Please provide a valid model file.");
    }

    // --- MERGE ANIMATION DATA & BONES ---
    // Verify bone/animation usage
    bool hasBones = !loaded_bone_data.boneNameToIndex.empty();

    // Calculate Offset for Bone Indices (Append Mode)
    unsigned int boneIndexOffset = 0;
    if (append) {
        boneIndexOffset = static_cast<unsigned int>(scene.boneData.boneNameToIndex.size());
    }
    else {
        // New Scene - already cleared in Step 0, but ensure boneData is fresh
        scene.boneData.boneNameToIndex.clear();
        scene.boneData.boneOffsetMatrices.clear();
        scene.boneData.boneNameToNode.clear();
        scene.boneData.boneParents.clear();
        scene.boneData.boneDefaultTransforms.clear();
        scene.boneData.perModelInverses.clear();
        scene.boneData.weightedBoneNames.clear();
    }

    // 1. Update Triangle Bone Indices with Offset
    if (hasBones && boneIndexOffset > 0) {
        const size_t triCount = loaded_triangles.size();
        auto shiftRange = [&loaded_triangles, boneIndexOffset](size_t s, size_t e) {
            for (size_t i = s; i < e; ++i) {
                auto& tri = loaded_triangles[i];
                if (tri->hasSkinData()) {
                    auto& vertexWeightsList = tri->getVertexBoneWeights();
                    for (auto& vertexWeights : vertexWeightsList) {
                        for (auto& bw : vertexWeights) {
                            bw.first += boneIndexOffset;
                        }
                    }
                }
            }
        };

        constexpr size_t kBoneShiftParallelThreshold = 8192;
        if (triCount < kBoneShiftParallelThreshold) {
            shiftRange(0, triCount);
        } else {
            const unsigned int hw = std::max(2u, std::thread::hardware_concurrency());
            const size_t chunks = std::min<size_t>(hw, (triCount + 8191) / 8192);
            const size_t chunkSize = (triCount + chunks - 1) / chunks;
            std::vector<std::future<void>> futures;
            futures.reserve(chunks);
            for (size_t c = 0; c < chunks; ++c) {
                const size_t s = c * chunkSize;
                const size_t e = std::min(s + chunkSize, triCount);
                if (s >= e) break;
                futures.push_back(std::async(std::launch::async, shiftRange, s, e));
            }
            for (auto& f : futures) f.get();
        }
    }

    // 2. Merge Bone Data
    if (hasBones) {
        for (const auto& [name, id] : loaded_bone_data.boneNameToIndex) {
            scene.boneData.boneNameToIndex[name] = id + boneIndexOffset;
        }
        // Merge Offset Matrices and Node Pointers
        scene.boneData.boneOffsetMatrices.insert(loaded_bone_data.boneOffsetMatrices.begin(), loaded_bone_data.boneOffsetMatrices.end());
        scene.boneData.boneNameToNode.insert(loaded_bone_data.boneNameToNode.begin(), loaded_bone_data.boneNameToNode.end());
        scene.boneData.perModelInverses.insert(loaded_bone_data.perModelInverses.begin(), loaded_bone_data.perModelInverses.end());
        scene.boneData.boneParents.insert(loaded_bone_data.boneParents.begin(), loaded_bone_data.boneParents.end());
        scene.boneData.boneDefaultTransforms.insert(loaded_bone_data.boneDefaultTransforms.begin(), loaded_bone_data.boneDefaultTransforms.end());
        scene.boneData.weightedBoneNames.insert(loaded_bone_data.weightedBoneNames.begin(), loaded_bone_data.weightedBoneNames.end());

        // Only set global inverse if it's the first model or handle separately? 
        // It seems unused for skinning (offset matrix handles it), so valid to leave or overwrite.
        if (!append) scene.boneData.globalInverseTransform = loaded_bone_data.globalInverseTransform;

        // CRITICAL: Rebuild reverse lookup after merge for O(1) index->name queries
        scene.boneData.rebuildReverseLookup();
    }

    // 3. Merge Animations
    if (append) {
        scene.animationDataList.insert(scene.animationDataList.end(), loaded_animations.begin(), loaded_animations.end());
    }
    else {
        scene.animationDataList = loaded_animations;
    }

    SCENE_LOG_INFO("Successfully loaded triangles: " + std::to_string(loaded_triangles.size()));
    SCENE_LOG_INFO("Loaded animations: " + std::to_string(loaded_animations.size()));
    SCENE_LOG_INFO("Total Bones (Merged): " + std::to_string(scene.boneData.boneNameToIndex.size()));

    update_progress(45, "Adding triangles to scene...");
    SCENE_LOG_INFO("Adding triangles to scene world...");

    // Add triangles to scene - animation is handled via TransformHandle and skinning
    // NOTE: AnimatedObject wrappers removed (were unused, wasted memory)
    // Add triangles to scene and model context members
    // Reserve to avoid repeated vector reallocations when importing millions of triangles.
    scene.world.reserve(scene.world.size() + loaded_triangles.size());
    modelCtx.members.reserve(modelCtx.members.size() + loaded_triangles.size());
    for (const auto& tri : loaded_triangles) {
        scene.world.add(tri);
        modelCtx.members.push_back(tri);
    }
    modelCtx.rebuildSkeletonRepresentation(scene.boneData);
    scene.importedModelContexts.push_back(modelCtx);
    SCENE_LOG_INFO("Added " + std::to_string(loaded_triangles.size()) + " triangles to scene member list.");

    // Initialize animation system for the new model
    if (modelCtx.hasAnimation) {
        initializeAnimationSystem(scene);
        // Settle bone matrices for frame 0, applying per-model inverses
        updateAnimationWithGraph(scene, 0.0f, true);
    }

    // ---- 2. Kamera ve ���k verisi ----
    update_progress(55, "Loading camera & lights...");
    SCENE_LOG_INFO("Loading camera and lighting data...");

    // Get new cameras and lights from loaded model using NEW loader
    auto new_lights = newLoader->getLights();
    auto new_cameras = newLoader->getCameras();  // Get ALL cameras

    // Handle cameras: Add all to the list
    if (append) {
        // Append mode: Add new cameras but keep active camera
        for (auto& cam : new_cameras) {
            if (cam) {
                cam->update_camera_vectors();
                scene.cameras.push_back(cam);
                SCENE_LOG_INFO("Append mode: Added camera (total: " + std::to_string(scene.cameras.size()) + ")");
            }
        }
        // If no camera was set before, set the first one as active
        if (!scene.camera && !scene.cameras.empty()) {
            scene.setActiveCamera(0);
        }
    }
    else {
        // New scene: Replace camera list
        scene.cameras.clear();
        for (auto& cam : new_cameras) {
            if (cam) {
                cam->save_initial_state();
                cam->update_camera_vectors();
                scene.cameras.push_back(cam);
            }
        }

        // If no cameras from model, create default
        if (scene.cameras.empty()) {
            auto new_camera = newLoader->getDefaultCamera();
            if (new_camera) {
                new_camera->save_initial_state();
                new_camera->update_camera_vectors();
                scene.cameras.push_back(new_camera);
            }
        }

        // Set first camera as active
        if (!scene.cameras.empty()) {
            scene.setActiveCamera(0);
            SCENE_LOG_INFO("Loaded " + std::to_string(scene.cameras.size()) + " camera(s). Active: Camera #0");
        }
        else {
            SCENE_LOG_WARN("No camera found in model.");
        }
    }

    // Handle lights: In append mode, merge with existing lights
    if (append) {
        // Append new lights to existing ones
        for (auto& light : new_lights) {
            scene.lights.push_back(light);
        }
        SCENE_LOG_INFO("Append mode: Added " + std::to_string(new_lights.size()) + " new lights (total: " + std::to_string(scene.lights.size()) + ")");
    }
    else {
        // Replace lights
        scene.lights = new_lights;
        SCENE_LOG_INFO("Loaded lights: " + std::to_string(scene.lights.size()));
    }

    // CRITICAL: Sync World Sun with first Directional Light from import/new project
    if (!append) { // Only force sync on new scene load, not append
        for (const auto& light : scene.lights) {
            if (light->type() == LightType::Directional) {
                // FORCE Default Sun Intensity to 10.0 (Override file default which is often too low, e.g. 2.4/pi)
                // Also ensures World Sun and Directional Light are coupled at start.
                light->intensity = 10.0f;

                world.setSunDirection(-light->direction);
                world.setSunIntensity(light->intensity);
                SCENE_LOG_INFO("World Sun synced with imported Directional Light (Forced Intensity: 10.0).");
                break; // Only sync to the first one
            }
        }
    }
    // ...
    // Note: OptiX conversion below (in original code) referenced 'assimpLoader' which was the member.
    // We must update that block too, but replacing only up to line 1335 handles the loading/merging logic.
    // The OptiX block is below 1335. I should include it in replacement range or do another replace.
    // The instruction requested updating create_scene. I'll replace the block covering loading to lighting.

    // BUT wait, I need to check if 'assimpLoader.convertTrianglesToOptixData' is called later.
    // Yes, line 1364. That uses member assimpLoader. Since it's a stateless helper (except texture cache maybe?), it MIGHT be okay?
    // BUT convertTrianglesToOptixData uses `MaterialManager` and triangle data. It seems stateless.
    // However, it's safer to use `newLoader`.

    // I will replace up to line 1400.

    // Force initial animation synchronization (poses correctly + updates CPU vertices)
    // CRITICAL: Must happen BEFORE BVH build so BVH sees correctly posed vertices!
    if (!scene.animationDataList.empty() && !scene.boneData.boneNameToIndex.empty()) {
        SCENE_LOG_INFO("[SceneCreation] Forcing initial pose sync for " + std::to_string(scene.boneData.getBoneCount()) + " bones.");
        updateAnimationWithGraph(scene, 0.0f, true); // true = apply_cpu_skinning
    }

    //  Selectable BVH (Embree or in-house BVH)
    update_progress(60, "Building BVH structure...");
    SCENE_LOG_INFO("Building BVH structure...");
    if (use_embree) {
        auto embree_bvh = std::make_shared<EmbreeBVH>();
        embree_bvh->build(scene.world.objects);
        scene.bvh = embree_bvh;
        SCENE_LOG_INFO("[Embree] BVH structure built successfully.");
    }
    else {
        scene.bvh = std::make_shared<ParallelBVHNode>(scene.world.objects, 0, scene.world.size(), 0.0f, 1.0f);
        SCENE_LOG_INFO("[RayTrophi: RT_BVH]  structure built successfully.");
    }

    // ---- 3. GPU setup deferred ----
    // GPU backend build (OptiX GAS / Vulkan AS) is deferred to scene_loading_done in
    // Main.cpp, which calls syncActiveRenderBackendScene() with all dirty flags set.
    // This avoids a double build: create_scene would build from loaded_triangles, then
    // syncActiveRenderBackendScene rebuilds from scene.world.objects immediately after.
    // For the startup path, syncActiveRenderBackendScene(true) is called explicitly via
    // the splash screen prepare block — no action needed here in either case.
    SCENE_LOG_INFO("GPU backend build deferred to post-load sync.");

    // ---- 4. Son bilgiler ----
    update_progress(100, "Complete!");
    SCENE_LOG_INFO("Scene creation completed successfully.");
    SCENE_LOG_INFO("Scene info - Triangles: " + std::to_string(loaded_triangles.size()) +
        ", Lights: " + std::to_string(scene.lights.size()) +
        ", Animations: " + std::to_string(scene.animationDataList.size()));

    scene.initialized = true;
    SCENE_LOG_INFO("Scene initialization flag set to true.");
}


std::uniform_int_distribution<> dis_width(0, image_width - 1);
std::uniform_int_distribution<> dis_height(0, image_height - 1);

namespace {
Vec3 orient_shading_normal(const Vec3& shading_normal, const Vec3& geometric_normal) {
    Vec3 n = shading_normal.normalize();
    Vec3 ng = geometric_normal.normalize();

    // Keep shading normal on the same hemisphere as the face-forwarded geometric normal.
    if (Vec3::dot(n, ng) < 0.0f) {
        n = -n;
    }
    return n;
}
}


void Renderer::apply_normal_map(HitRecord& rec) {
    HitMaterialResolver::resolveMaterialPointers(rec);
    Material* material = rec.materialPtr;
    if (!material) {
        return;
    }

    if (material->has_normal_map()) {
        Vec3 tangent(1.0f, 0.0f, 0.0f);
        Vec3 bitangent(0.0f, 1.0f, 0.0f);
        bool has_uv_tangent = false;

        const Triangle* tri = rec.triangle;
        if (tri) {
            Vec3 v0 = tri->getV0();
            Vec3 v1 = tri->getV1();
            Vec3 v2 = tri->getV2();
            auto [uv0, uv1, uv2] = tri->getUVCoordinates();

            // TBN uses ORIGINAL mesh UVs — do NOT flip V here.
            // Normal maps are authored in original UV space; flipping V
            // would negate the bitangent and invert the Green channel.

            Vec3 edge1 = v1 - v0;
            Vec3 edge2 = v2 - v0;
            Vec2 deltaUV1 = uv1 - uv0;
            Vec2 deltaUV2 = uv2 - uv0;

            float det = deltaUV1.u * deltaUV2.v - deltaUV2.u * deltaUV1.v;
            if (std::abs(det) > 1e-8f) {
                float f = 1.0f / det;
                tangent = (edge1 * deltaUV2.v - edge2 * deltaUV1.v) * f;
                bitangent = (edge2 * deltaUV1.u - edge1 * deltaUV2.u) * f;
                
                tangent = (tangent - rec.normal * Vec3::dot(rec.normal, tangent)).normalize();
                bitangent = Vec3::cross(rec.normal, tangent).normalize();
                
                Vec3 bitangent_test = (edge2 * deltaUV1.u - edge1 * deltaUV2.u) * f;
                if (Vec3::dot(bitangent, bitangent_test) < 0.0f) {
                    bitangent = -bitangent;
                }
                has_uv_tangent = true;
            }
        }

        if (!has_uv_tangent) {
            create_coordinate_system(rec.normal, tangent, bitangent);
        }

        Vec3 normal_from_map = material->get_normal_from_map(rec.u, rec.v);
        normal_from_map = normal_from_map * 2.0 - Vec3(1.0, 1.0, 1.0);

        float normal_strength = material->get_normal_strength();
        normal_from_map.x *= normal_strength;
        normal_from_map.y *= normal_strength;

        Mat3x3 TBN(tangent, bitangent, rec.normal);
        rec.interpolated_normal = orient_shading_normal(TBN * normal_from_map, rec.normal);
    }
    else {
        rec.interpolated_normal = rec.normal;
    }
}

void Renderer::create_coordinate_system(const Vec3& N, Vec3& T, Vec3& B) {
    Vec3 N_norm = N.normalize();

    // E�er normal z eksenine paralelse, �ok k���k d�z y�zeyler i�in �zel durum
    if (N_norm.z < -0.999999f) {
        T = Vec3(0, -1, 0);  // Ters y�nlendirilmi� bir tangent
        B = Vec3(-1, 0, 0);
    }
    else {
        // Normalden tangent ve bitangent hesaplamas�
        float a = 1.0f / (1.0f + N_norm.z);
        float b = -N_norm.x * N_norm.y * a;

        // Daha hassas bir hesaplama, d�z y�zeylerdeki ters d�nme sorunu engellenebilir
        T = Vec3(1.0f - N_norm.x * N_norm.x * a, b, -N_norm.x);
        B = Vec3(b, 1.0f - N_norm.y * N_norm.y * a, -N_norm.y);

        // D�z y�zeylerde y�nleri do�ru tutmak i�in k���k d�zeltme
        if (std::abs(N_norm.z) > 0.9999f) {
            T = Vec3(1.0f, 0.0f, 0.0f);  // x y�n�yle tangent d�zeltmesi
            B = Vec3(0.0f, 1.0f, 0.0f);  // y y�n�yle bitangent d�zeltmesi
        }
    }
}

void Renderer::initialize_halton_cache() {
    halton_cache = std::make_unique<float[]>(MAX_DIMENSIONS * MAX_SAMPLES_HALTON);

    for (int d = 0; d < MAX_DIMENSIONS; ++d) {
        int base = (d == 0) ? 2 : 3;
        for (size_t i = 0; i < MAX_SAMPLES_HALTON; ++i) {
            // Tek boyutlu array'de 2D array gibi indeksleme
            halton_cache[d * MAX_SAMPLES_HALTON + i] = halton(i, base);
        }
    }
}

float Renderer::get_halton_value(size_t index, int dimension) {
    if (dimension < 0 || dimension >= MAX_DIMENSIONS ||
        index >= MAX_SAMPLES_HALTON) {
        return halton(index, dimension == 0 ? 2 : 3);
    }

    return halton_cache[dimension * MAX_SAMPLES_HALTON + index];
}

float Renderer::halton(int index, int base) {
    float r = 0;
    float f = 1;
    int i = index;

    while (i > 0) {
        f = f / base;
        r = r + f * (i % base);
        i = i / base;
    }

    return r;
}

Vec2 Renderer::stratified_halton(int x, int y, int sample_index, int samples_per_pixel) {
    // Daha iyi da��l�m i�in perm�tasyon ekliyoruz
    const uint32_t pixel_hash = (x * 73856093) ^ (y * 19349663); // Basit bir hash fonksiyonu
    const uint32_t sample_hash = sample_index * 83492791;

    // Halton dizisinde farkl� offsetler kullan�yoruz
    const int base_index = (pixel_hash + sample_hash) % MAX_SAMPLES_HALTON;

    // Farkl� asal say� tabanlar� kullanarak daha iyi da��l�m
    const float u = halton_cache[base_index];                     // Taban 2
    const float v = halton_cache[(base_index + MAX_SAMPLES_HALTON / 2) % MAX_SAMPLES_HALTON]; // Taban 3

    // Stratifikasyon eklemek i�in jitter
    const float jitter_u = (rand() / (float)RAND_MAX) * 0.8f / samples_per_pixel;
    const float jitter_v = (rand() / (float)RAND_MAX) * 0.8f / samples_per_pixel;

    return Vec2(
        (x + u + jitter_u) / image_width,
        (y + v + jitter_v) / image_height
    );
}



float Renderer::luminance(const Vec3& color) {
    return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}

// --- Ak�ll� ���k se�imi ---
int Renderer::pick_smart_light(const std::vector<std::shared_ptr<Light>>& lights, const Vec3& hit_position) {
    int light_count = (int)lights.size();
    if (light_count == 0) return -1;

    // --- 1. Directional light varsa %33 ihtimalle se� ---
    for (int i = 0; i < light_count; i++) {
        if (!lights[i]->visible) continue; // Skip invisible lights
        if (lights[i]->type() == LightType::Directional) {
            if (Vec3::random_float() < 0.33) {
                directional_pick_count++;
                return i;
            }
        }
    }

    // --- 2. T�m ���k t�rlerinden a��rl�kl� se�im (GPU ile uyumlu) ---
    std::vector<float> weights(light_count, 0.0f);
    float total_weight = 0.0f;

    for (int i = 0; i < light_count; i++) {
        if (!lights[i]->visible) {
            weights[i] = 0.0f;
            continue; // Skip invisible lights
        }
        Vec3 delta = lights[i]->position - hit_position;
        float distance = std::max(1.0f, delta.length());
        float falloff = 1.0f / (distance * distance);
        float intensity = luminance(lights[i]->color * lights[i]->intensity);

        if (lights[i]->type() == LightType::Point) {
            weights[i] = falloff * intensity;
        }
        else if (lights[i]->type() == LightType::Area) {
            // GPU ile uyumlu: area etkisi
            auto areaLight = std::dynamic_pointer_cast<AreaLight>(lights[i]);
            if (areaLight) {
                float area = areaLight->getWidth() * areaLight->getHeight();
                weights[i] = falloff * intensity * std::min(area, 10.0f);
            }
        }
        else if (lights[i]->type() == LightType::Spot) {
            weights[i] = falloff * intensity * 0.8f;
        }
        else {
            weights[i] = 0.0f;
        }

        total_weight += weights[i];
    }

    // --- E�er a��rl�k yoksa fallback rastgele se�im ---
    if (total_weight < 1e-6f) {
        return std::clamp(int(Vec3::random_float() * light_count), 0, light_count - 1);
    }

    // --- Weighted se�im ---
    float r = Vec3::random_float() * total_weight;
    float accum = 0.0f;
    for (int i = 0; i < light_count; i++) {
        accum += weights[i];
        if (r <= accum) {
            return i;
        }
    }

    // --- G�venlik fallback ---
    return std::clamp(int(Vec3::random_float() * light_count), 0, light_count - 1);
}

Vec3 Renderer::calculate_direct_lighting_single_light(
    const Hittable* bvh,
    const std::shared_ptr<Light>& light,
    const HitRecord& rec,
    const Vec3& normal,
    const Ray& r_in
) {
    Vec3 direct_light(0.0f);

    Vec3 hit_point = rec.point;
    Vec2 uv = Vec2(rec.u, rec.v);

    // Malzeme �zellikleri
    // Malzeme �zellikleri (Blending Support)
    Material* material = rec.materialPtr;
    if (!material) {
        return direct_light;
    }

    Vec3 albedo;
    float metallic;
    float roughness;
    float specularAmount = 0.5f;
    float clearcoat = 0.0f;
    float clearcoatRoughness = 0.03f;

    if (rec.surface_override.valid) {
        albedo = rec.surface_override.albedo;
        metallic = rec.surface_override.metallic;
        roughness = rec.surface_override.roughness;
        clearcoat = rec.surface_override.clearcoat;
        clearcoatRoughness = rec.surface_override.clearcoat_roughness;
    } else {
        albedo = material->getPropertyValue(material->albedoProperty, uv);
        metallic = material->getPropertyValue(material->metallicProperty, uv).z;
        roughness = material->getPropertyValue(material->roughnessProperty, uv).y;
        
        // Try to get clearcoat from material
        if (auto pMat = dynamic_cast<PrincipledBSDF*>(material)) {
            clearcoat = pMat->clearcoat;
            clearcoatRoughness = pMat->clearcoatRoughness;
            specularAmount = pMat->getSpecularValue(uv);
        }
    }
    Vec3 F0 = Vec3::lerp(Vec3(std::clamp(0.08f * specularAmount, 0.0f, 0.08f)), albedo, metallic);

    Vec3 V = -r_in.direction.normalize();
    Vec3 N = normal;

    Vec3 light_sample, to_light, Li;
    float light_distance = 1.0f;
    Vec3 L;

    float pdf_light = 1.0f;
    float pdf_light_select = 1.0f;
    float attenuation = 1.0f;

    // --- Light sampling ---
    if (auto directional = std::dynamic_pointer_cast<DirectionalLight>(light)) {
        L = -directional->random_point();
        light_sample = hit_point + L * 1e8f;
        to_light = L;
        attenuation = 1.0f; // Directional light falloff yok
        light_distance = std::numeric_limits<float>::infinity();
        Li = directional->getIntensity(hit_point, light_sample);
    }
    else if (auto point = std::dynamic_pointer_cast<PointLight>(light)) {
        light_sample = point->random_point();
        to_light = light_sample - hit_point;
        light_distance = to_light.length();
        L = to_light / light_distance;

        // D�ZELTME: PointLight s�n�f� getIntensity i�inde zaten falloff (1/d^2) uyguluyor.
        // Burada tekrar uygularsak ���k �ok zay�fl�yor (1/d^4).
        attenuation = 1.0f;

        // Point Light Specific Boost: Global �arpan kald�r�ld�, sadece Point Light 10 kat g��lendirildi.
        Li = point->getIntensity(hit_point, light_sample) * attenuation;

        float area = 4.0f * M_PI * point->getRadius() * point->getRadius();
        pdf_light = (1.0f / area) * pdf_light_select;
    }
    else if (auto areaLight = std::dynamic_pointer_cast<AreaLight>(light)) {
        // GPU ile uyumlu AreaLight sampling
        light_sample = areaLight->random_point();
        to_light = light_sample - hit_point;
        light_distance = to_light.length();
        L = to_light / light_distance;

        // Light normal (cross of u and v vectors)
        Vec3 light_normal = Vec3::cross(areaLight->getU(), areaLight->getV()).normalize();
        float cos_light = std::fmax(Vec3::dot(-L, light_normal), 0.0f);
        attenuation = cos_light / (light_distance * light_distance);

        Li = areaLight->getIntensity(hit_point, light_sample) * attenuation;

        float area = areaLight->getWidth() * areaLight->getHeight();
        pdf_light = (1.0f / std::fmax(area, 1e-4f)) * pdf_light_select;
    }
    else if (auto spotLight = std::dynamic_pointer_cast<SpotLight>(light)) {
        // GPU ile uyumlu SpotLight sampling
        light_sample = spotLight->position;
        to_light = light_sample - hit_point;
        light_distance = to_light.length();
        L = to_light / light_distance;

        // Spot cone falloff
        float cos_theta = Vec3::dot(-L, spotLight->direction.normalize());
        float angleDeg = spotLight->getAngleDegrees();
        float angleRad = angleDeg * (M_PI / 180.0f);
        float inner_cos = cosf(angleRad * 0.8f);
        float outer_cos = cosf(angleRad);

        float falloff = 0.0f;
        if (cos_theta > inner_cos) falloff = 1.0f;
        else if (cos_theta > outer_cos) {
            float t = (cos_theta - outer_cos) / (inner_cos - outer_cos + 1e-6f);
            falloff = t * t;
        }

        if (falloff < 1e-4f) return direct_light;

        attenuation = falloff / (light_distance * light_distance);
        Li = spotLight->getIntensity(hit_point, light_sample) * attenuation;

        float solid_angle = 2.0f * M_PI * (1.0f - outer_cos);
        pdf_light = (1.0f / std::fmax(solid_angle, 1e-4f)) * pdf_light_select;
    }
    else {
        return direct_light;
    }

    // --- Shadow ---
    // GPU ile uyumlu shadow bias (eski: 0.0001f -> self-shadowing yapabilir)
    // Volumetric Shadow Logic (Transparent Shadows)
    Ray shadow_ray_current(hit_point + N * 0.001f, L);
    float remaining_dist = light_distance;
    Vec3 shadow_transmittance(1.0f); // Changed from float to Vec3 for colored shadows
    int shadow_layers = 0;
    
    // Check hair shadows first (hair casts strong shadows on meshes)
    if (hairSystem.getTotalStrandCount() > 0 && !hairSystem.isBVHDirty()) {
        Hair::HairHitInfo hairShadowHit;
        Vec3 hairShadowOrigin = shadow_ray_current.origin;
        Vec3 hairShadowDir = shadow_ray_current.direction;
        float hairShadowDist = std::min(remaining_dist, 100.0f); // Limit to 100 units
        
        // Trace through multiple hair strands for accumulated shadow
        int hairShadowSamples = 0;
        while (hairShadowSamples < 8 && hairShadowDist > 0.01f) {
            if (hairSystem.intersect(hairShadowOrigin, hairShadowDir, 0.002f, hairShadowDist, hairShadowHit)) {
                // Hair casts stronger shadows - higher opacity per strand
                float hairOpacity = 0.5f + 0.2f * (1.0f - hairShadowHit.v); // 50-70% per strand
                shadow_transmittance = shadow_transmittance * (1.0f - hairOpacity);
                
                // Continue tracing through hair
                hairShadowOrigin = hairShadowHit.position + hairShadowDir * 0.003f;
                hairShadowDist -= (hairShadowHit.t + 0.003f);
                hairShadowSamples++;
                
                if (shadow_transmittance.max_component() < 0.01f) break;
            } else {
                break;
            }
        }
    }


    while (remaining_dist > 0.001f && shadow_layers < 4) {
        HitRecord shadow_rec;
        if (bvh->hit(shadow_ray_current, 0.001f, remaining_dist, shadow_rec)) {
            HitMaterialResolver::resolveMaterialPointers(shadow_rec);

            // Check if blocker is a Volume (VDB or Unified Gas)
            const VDBVolume* vdb = shadow_rec.vdb_volume;
            int live_vol_id = -1;
            std::shared_ptr<VolumeShader> vol_shader = nullptr;
            Matrix4x4 inv_transform = Matrix4x4::identity();
            float den_scale = 1.0f;

            if (vdb) {
                live_vol_id = vdb->getVDBVolumeID();
                vol_shader = vdb->volume_shader;
                inv_transform = vdb->getInverseTransform();
                den_scale = vdb->density_scale;
            }
            else if (shadow_rec.gas_volume && shadow_rec.gas_volume->render_path == GasVolume::VolumeRenderPath::VDBUnified) {
                live_vol_id = shadow_rec.gas_volume->live_vdb_id;
                vol_shader = shadow_rec.gas_volume->getShader();
                Transform* gv_trans = shadow_rec.gas_volume->getTransformPtr();
                if (gv_trans) {
                    Matrix4x4 m = gv_trans->getFinal();
                    Vec3 gsize = shadow_rec.gas_volume->getSettings().grid_size;
                    if (gsize.x > 0 && gsize.y > 0 && gsize.z > 0) {
                        m = m * Matrix4x4::scaling(Vec3(1.0f / gsize.x, 1.0f / gsize.y, 1.0f / gsize.z));
                    }
                    inv_transform = m.inverse();
                }
                den_scale = 1.0f;
            }

            if (live_vol_id >= 0) {
                // Get intersection interval
                float t_enter, t_exit;
                bool hit_box = false;

                if (vdb) {
                    hit_box = vdb->intersectTransformedAABB(shadow_ray_current, 0.001f, remaining_dist, t_enter, t_exit);
                }
                else {
                    // Manual box check for unified gas
                    AABB box; shadow_rec.gas_volume->bounding_box(0, 0, box);
                    hit_box = box.hit_interval(shadow_ray_current, 0.001f, remaining_dist, t_enter, t_exit);
                }

                if (hit_box) {
                    if (t_enter < 0.001f) t_enter = 0.001f;
                    if (t_exit > remaining_dist) t_exit = remaining_dist;

                    float shadow_step = vol_shader ? vol_shader->quality.step_size * 4.0f : 0.4f;
                    if (shadow_step < 0.05f) shadow_step = 0.05f;

                    float density_scale = (vol_shader ? vol_shader->density.multiplier : 1.0f) * den_scale;
                    float shadow_strength = vol_shader ? vol_shader->quality.shadow_strength : 1.0f;

                    float t = t_enter;
                    auto& mgr = VDBVolumeManager::getInstance();
                    t += ((float)rand() / RAND_MAX) * shadow_step;

                    while (t < t_exit) {
                        Vec3 p = shadow_ray_current.at(t);
                        Vec3 local_p = inv_transform.transform_point(p);
                        float density = mgr.sampleDensityCPU(live_vol_id, local_p.x, local_p.y, local_p.z);

                        if (density < 0.01f) density = 0.0f;
                        if (density > 0.0f) {
                            float s_sigma_s = density * density_scale * (vol_shader ? vol_shader->scattering.coefficient : 1.0f);
                            float s_sigma_a = density * (vol_shader ? vol_shader->absorption.coefficient : 0.1f);
                            float sigma_t = (s_sigma_s + s_sigma_a) * shadow_strength;
                            shadow_transmittance = shadow_transmittance * std::exp(-sigma_t * shadow_step);
                        }
                        if (shadow_transmittance.max_component() < 0.01f) break;
                        t += shadow_step;
                    }
                }

                if (shadow_transmittance.max_component() < 0.01f) return direct_light;

                float advance = t_exit + 0.001f;
                shadow_ray_current = Ray(shadow_ray_current.at(advance), L);
                remaining_dist -= advance;
                shadow_layers++;
            }
            else {
                // Check for Generic Mesh Volume Traversal
                bool is_volumetric = false;
                if (shadow_rec.materialPtr && shadow_rec.materialPtr->type() == MaterialType::Volumetric) {
                    is_volumetric = true;
                    auto vol = static_cast<Volumetric*>(shadow_rec.materialPtr);

                    // Find exit point (assume convex/closed mesh for now)
                    Ray exit_ray(shadow_ray_current.at(shadow_rec.t + 0.001f), shadow_ray_current.direction);
                    HitRecord exit_rec;
                    float exit_dist = remaining_dist - shadow_rec.t;

                    // Trace to find the back side of this volume
                    // Note: Ideally we check if we hit the SAME object, but checking material is a good proxy
                    float t_vol_exit = 0.0f;
                    bool found_exit = false;

                    if (bvh->hit(exit_ray, 0.001f, exit_dist, exit_rec)) {
                        HitMaterialResolver::resolveMaterialPointers(exit_rec);
                        // If we hit something, assume it's the exit if it's the same material
                        // Or just treat the segment [enter, next_hit] as the volume
                        t_vol_exit = exit_rec.t;
                        found_exit = true;
                    }
                    else {
                        // Didn't hit anything within light distance -> Volume covers the rest of the path?
                        // Or we exited without hitting a backface (geometry error?). 
                        // For shadows, assume we exit at light distance.
                        t_vol_exit = exit_dist;
                    }

                    // Ray Marching Params
                    float step_size = vol->getStepSize();
                    float density_mult = vol->getDensity();
                    float t = 0.0f; // Relative to exit_ray origin (which is entry + epsilon)

                    // Jitter
                    float jitter = ((float)rand() / RAND_MAX) * step_size;
                    t += jitter;

                    // March
                    while (t < t_vol_exit) {
                        Vec3 p = exit_ray.at(t);
                        float d = vol->calculate_density(p);

                        if (d > 0.001f) {
                            float sigma_t = d * 1.0f; // Assuming extinction = density for shadows
                            shadow_transmittance = shadow_transmittance * std::exp(-sigma_t * step_size);
                        }

                        if (shadow_transmittance.max_component() < 0.01f) break;
                        t += step_size;
                    }

                    // Advance Shadow Ray Logic
                    if (found_exit) {
                        if (exit_rec.materialID == shadow_rec.materialID) {
                            // Hit Volume Backface: Advance PAST it
                            float total_advance = shadow_rec.t + t_vol_exit + 0.001f;
                            shadow_ray_current = Ray(shadow_ray_current.at(total_advance), shadow_ray_current.direction);
                            remaining_dist -= total_advance;
                        }
                        else {
                            // Hit Obstruction (different material): Advance TO it (just before)
                            // We want to hit this obstruction in the next loop iteration.
                            float total_advance = shadow_rec.t + t_vol_exit - 0.001f;
                            if (total_advance < 0.0f) total_advance = 0.0f; // Safety
                            shadow_ray_current = Ray(shadow_ray_current.at(total_advance), shadow_ray_current.direction);
                            remaining_dist -= total_advance;
                        }
                    }
                    else {
                        // Reached Light or end of trace without hitting anything
                        break;
                    }
                }

                if (!is_volumetric) {
                    // Check for Transparent Surface (Glass, Water, Alpha Cutout)
                    bool is_transparent = false;
                    Vec3 transmission_filter(1.0f);

                    if (shadow_rec.materialPtr) {
                        auto pbsdf = dynamic_cast<PrincipledBSDF*>(shadow_rec.materialPtr);
                        if (pbsdf) {
                            Vec2 uv(shadow_rec.u, shadow_rec.v);
                            float tr = pbsdf->getTransmission(uv);
                            float op = pbsdf->get_opacity(uv);

                            if (tr > 0.001f || op < 0.999f) {
                                is_transparent = true;
                                Vec3 base_color = pbsdf->getPropertyValue(pbsdf->albedoProperty, uv);

                                // Simple Transmission approximation for shadows:
                                // Light passes through colored glass
                                // Mix based on opacity and transmission
                                Vec3 tf = (base_color);

                                // 1. Alpha Cutout (Leafs, Fences)
                                if (op < 0.999f) {
                                    // Corrected: Vec3::lerp is static.
                                    // Lerp between 1.0 (air) and MaterialColor (if transmitted) - Simplified approach
                                    transmission_filter = Vec3::lerp(Vec3(1.0f), transmission_filter, op);
                                }

                                // 2. Glass Transmission
                                if (tr > 0.001f) {
                                    // Corrected: No .pow member, do component-wise manually
                                    Vec3 bc = (base_color);
                                    Vec3 bc_pow(std::pow(bc.x, 0.5f), std::pow(bc.y, 0.5f), std::pow(bc.z, 0.5f));
                                    transmission_filter = transmission_filter * bc_pow;
                                }
                            }
                        }
                    }

                    if (is_transparent) {
                        shadow_transmittance *= transmission_filter;
                        if (shadow_transmittance.luminance() < 0.01f) return direct_light; // Absorbed

                        // Advance ray past the transparent surface
                        float advance = shadow_rec.t + 0.001f;
                        shadow_ray_current = Ray(shadow_ray_current.at(advance), L);
                        remaining_dist -= advance;
                        shadow_layers++;
                        continue; // Continue tracing
                    }
                    else {
                        // Opaque blocker found
                        return direct_light;
                    }
                }
            }
        }
        else {
            // No blocker found
            break;
        }
    }

    // Apply calculated transmittance to light intensity
    // shadow_transmittance is now Vec3
    if (shadow_transmittance.max_component() < 0.99f) {
        Li = Li * shadow_transmittance;
    }

    float NdotL = std::fmax(Vec3::dot(N, L), 0.0001f);


    // --- BRDF Hesab� (Specular + Diffuse) ---
    Vec3 H = (L + V).normalize();
    float NdotV = std::fmax(Vec3::dot(N, V), 0.0001f);
    float NdotH = std::fmax(Vec3::dot(N, H), 0.0001f);
    float VdotH = std::fmax(Vec3::dot(V, H), 0.0001f);

    float alpha = max(roughness * roughness, 0.01f);
    PrincipledBSDF psdf;
    // Specular bile�eni
    float D = psdf.DistributionGGX(N, H, roughness);
    float G = psdf.GeometrySmith(N, V, L, roughness);
    Vec3 F = psdf.fresnelSchlickRoughness(VdotH, F0, roughness);

    Vec3 specularBrdf = psdf.evalSpecular(N, V, L, F0, roughness);

    // Diffuse bile�eni - GPU ile uyumlu
    Vec3 F_avg = F0 + (Vec3(1.0f) - F0) / 21.0f;
    // GPU form�l�: k_d = (1 - F_avg) * (1 - metallic)
    Vec3 k_d = (Vec3(1.0f) - F_avg) * (1.0f - metallic);
    Vec3 diffuse = k_d * albedo / M_PI;

    // Toplam BRDF
    Vec3 brdf = diffuse + specularBrdf;

    // Clearcoat Contribution
    if (clearcoat > 0.001f) {
        psdf.clearcoatRoughness = clearcoatRoughness;
        // Check signature: computeClearcoat(V, L, N)
        // V is view vector (-ray.dir), L is light vector, N is normal
        Vec3 cc = psdf.computeClearcoat(V, L, N); 
        brdf = brdf + cc * clearcoat;
    }

    // --- MIS (Multiple Importance Sampling) ---
    // PDF BRDF hesapla
    Vec3 incoming = -L; // Light direction (incoming to surface)
    Vec3 outgoing = V;  // View direction
    float pdf_brdf_val = psdf.pdf(rec, incoming, outgoing);
    float pdf_brdf_val_mis = std::clamp(pdf_brdf_val, 0.01f, 5000.0f);

    float mis_weight = power_heuristic(pdf_light, pdf_brdf_val_mis);

    // I��k katk�s�
    // GPU form�l�: (f * Li * NdotL) * mis_weight
    Vec3 direct = brdf * Li * NdotL * mis_weight;

    return direct;
}


Vec3 Renderer::ray_color(const Ray& r, const Hittable* bvh,
    const std::vector<std::shared_ptr<Light>>& lights,
    const Vec3& background_color, int depth, int sample_index, const SceneData& scene,
    Vec3* primary_albedo, Vec3* primary_normal, bool* primary_hit,
    float* primary_depth, uint32_t* primary_material_id,
    Vec3* primary_world_position) {

    // =========================================================================
    // UNIFIED RAY COLOR - Matches GPU ray_color.cuh exactly
    // =========================================================================

    Vec3f color(0.0f);
    Vec3f throughput(1.0f);
    Vec3f current_medium_absorb(0.0f); // Default: Air (no absorption)
    Ray current_ray = r;
    int transparent_hits = 0;
    float first_hit_t = -1.0f;
    float first_hit_transmission = 0.0f;
    float vol_trans_accum = 1.0f;
    float first_vol_t = -1.0f;
    if (primary_hit) *primary_hit = false;
    if (primary_albedo) *primary_albedo = Vec3(0.0f);
    if (primary_normal) *primary_normal = Vec3(0.0f);
    if (primary_depth) *primary_depth = 0.0f;
    if (primary_material_id) *primary_material_id = 0xFFFFFFFFu;
    if (primary_world_position) *primary_world_position = Vec3(0.0f);

    if (world.data.mode == WORLD_MODE_NISHITA) {
        if (!world.getLUT()) {
            world.initializeLUT();
        } else if (world.needsLUTUpdate()) {
            world.flushLUT();
        }
    }

    int light_count = static_cast<int>(lights.size());

    // Pre-convert lights to unified format for this ray
    // Note: In production, this should be done once per frame, not per ray
    thread_local std::vector<UnifiedLight> unified_lights;
    if (unified_lights.size() != lights.size()) {
        unified_lights.clear();
        unified_lights.reserve(lights.size());
        for (const auto& light : lights) {
            unified_lights.push_back(toUnifiedLight(light));
        }
    }

    int diffuse_bounce_count = 0;
    int transmission_bounce_count = 0;
    const int max_diffuse_bounces = std::clamp(render_settings.diffuse_bounces, 1, std::max(1, render_settings.max_bounces));
    const int max_transmission_bounces = std::clamp(render_settings.transmission_bounces, 1, std::max(1, render_settings.max_bounces));

    for (int bounce = 0; bounce < render_settings.max_bounces; ++bounce) {
        HitRecord rec;
        HitRecord solid_rec;
        bool hit_solid = false;
        bool hit_hair = false;
        Hair::HairHitInfo hairHit;

        bool hit_any = false;
        if (bvh) {
            hit_any = bvh->hit(current_ray, 0.001f, std::numeric_limits<float>::infinity(), rec, false);
            if (hit_any) {
                HitMaterialResolver::resolveSurfaceData(rec);
            }
        }
        
        // Check hair intersection (if hair system has strands)
        if (hairSystem.getTotalStrandCount() > 0) {
            float maxT = hit_any ? rec.t : std::numeric_limits<float>::infinity();
            hit_hair = hairSystem.intersect(
                current_ray.origin, current_ray.direction,
                0.001f, maxT, hairHit
            );

            // Hair hit - process it (simplified condition)
            if (hit_hair) {
                // [MODIFIED] Random variation support using Strand ID
                Hair::HairMaterialParams hairMat = hairHit.material; // Copy material for per-strand mod

                // ===================================================================
                // ROOT UV TEXTURE SAMPLING (Inherit color from scalp mesh)
                // ===================================================================
                if (hairMat.colorMode == Hair::HairMaterialParams::ColorMode::ROOT_UV_MAP) {

                    // --- Custom Independent Texture Support ---
                    // If the user has assigned a specific texture to the hair material itself,
                    // we use that INSTEAD of the mesh texture.

                    bool usedCustomTexture = false;

                    if (hairMat.customAlbedoTexture) {
                        // Sample the custom texture using root UVs
                        // We use the texture's get_color method directly
                        hairMat.color = hairMat.customAlbedoTexture->get_color(hairHit.rootUV.u, hairHit.rootUV.v);
                        hairMat.colorMode = Hair::HairMaterialParams::ColorMode::DIRECT_COLORING;
                        usedCustomTexture = true;
                    }

                    // Apply Roughness Map if exists
                    if (hairMat.customRoughnessTexture) {
                        // Roughness maps are usually grayscale, we take the Red channel or intensity
                        float rMap = hairMat.customRoughnessTexture->get_color(hairHit.rootUV.u, hairHit.rootUV.v).x;
                        hairMat.roughness *= rMap;
                        hairMat.radialRoughness *= rMap;
                    }

                    // Only proceed to Mesh Inheritance if we didn't use a custom albedo
                    if (!usedCustomTexture) {
                        auto& matMgr = MaterialManager::getInstance();
                        const auto& all_mats = matMgr.getAllMaterials();
                        bool textureFound = false;

                        // 1. FAST PATH: Use cached Material ID
                        int matID = hairHit.meshMaterialID;
                        if (matID >= 0 && matID != 0xFFFF && (size_t)matID < all_mats.size()) {
                            const auto& mat = all_mats[matID];
                            if (mat) {
                                if (mat->albedoProperty.texture) {
                                    hairMat.color = mat->albedoProperty.evaluate(hairHit.rootUV);
                                    textureFound = true;
                                }
                                else {
                                    // Material exists but has no texture -> take base color
                                    hairMat.color = mat->albedoProperty.color;
                                    textureFound = true; // technically found, just no texture
                                }
                            }
                        }

                        // 2. SLOW PATH (Fallback): Search geometry by name
                        // This covers legacy grooms or cases where ID sync failed
                        if (!textureFound) {
                            if (const Hair::HairGroom* groom = hairSystem.getGroom(hairHit.groomName)) {
                                std::string scalpName = groom->boundMeshName;
                                for (const auto& h : scene.world.objects) {
                                    if (auto tri = std::dynamic_pointer_cast<Triangle>(h)) {
                                        if (tri->getNodeName() == scalpName) {
                                            int fallbackID = tri->getMaterialID();
                                            if (fallbackID >= 0 && (size_t)fallbackID < all_mats.size()) {
                                                const auto& mat = all_mats[fallbackID];
                                                if (mat && mat->albedoProperty.texture) {
                                                    hairMat.color = mat->albedoProperty.evaluate(hairHit.rootUV);
                                                }
                                                else if (mat) {
                                                    hairMat.color = mat->albedoProperty.color;
                                                }
                                            }
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        // Treat as direct coloring for BSDF evaluation
                        hairMat.colorMode = Hair::HairMaterialParams::ColorMode::DIRECT_COLORING;
                    }
                    // End if (!usedCustomTexture)
                } // End if (hairMat.colorMode == Hair::HairMaterialParams::ColorMode::ROOT_UV_MAP) 

        // Note: Tint is now applied inside HairBSDF::evaluate() as post-process
        // Do NOT modify hairMat.color here to avoid double-tinting

                if (hairMat.randomHue > 0.0f || hairMat.randomValue > 0.0f) {
                    uint32_t id = hairHit.strandID;
                    // Fast integer hash for stable randomness per strand
                    uint32_t h = id * 747796405u + 2891336453u;
                    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
                    float r1 = ((h >> 22u) ^ h) / 4294967296.0f; // [0, 1]

                    uint32_t h2 = id * 123456789u + 987654321u;
                    float r2 = (h2 & 0x00FFFFFF) / 16777216.0f; // [0, 1]

                    if (hairMat.colorMode == Hair::HairMaterialParams::ColorMode::MELANIN) {
                        // For Melanin mode, vary the physical parameters
                        hairMat.melanin = std::clamp(hairMat.melanin + (r1 - 0.5f) * hairMat.randomValue, 0.0f, 1.0f);
                        hairMat.melaninRedness = std::clamp(hairMat.melaninRedness + (r2 - 0.5f) * hairMat.randomHue, 0.0f, 1.0f);
                        
                        // Update the base color for the ambient/fallback term
                        Vec3 sigma = Hair::HairBSDF::melaninToAbsorption(hairMat.melanin, hairMat.melaninRedness);
                        hairMat.color = Vec3(
                            std::exp(-sigma.x * 0.5f),
                            std::exp(-sigma.y * 0.5f),
                            std::exp(-sigma.z * 0.5f)
                        );
                    } else {
                        // Random Brightness (Value) for Direct/Root UV modes
                        if (hairMat.randomValue > 0.0f) {
                            float vScale = 1.0f + (r1 - 0.5f) * hairMat.randomValue * 2.0f;
                            hairMat.color = hairMat.color * vScale;
                        }

                        // Random Hue (Shift) for Direct/Root UV modes
                        if (hairMat.randomHue > 0.0f) {
                            // Rodrigues rotation around Grey Axis (1,1,1)
                            float angle = (r2 - 0.5f) * hairMat.randomHue * 2.0f * 3.14159f;
                            float c = std::cos(angle);
                            float s = std::sin(angle);

                            Vec3 k(0.57735f); // 1/sqrt(3) normalized
                            Vec3& p = hairMat.color;
                            Vec3 crossP = Vec3::cross(k, p);
                            float dotP = Vec3::dot(k, p);

                            hairMat.color = p * c + crossP * s + k * dotP * (1.0f - c);
                        }
                    }

                    // Simple clamp
                    if (hairMat.color.x < 0) hairMat.color.x = 0;
                    if (hairMat.color.y < 0) hairMat.color.y = 0;
                    if (hairMat.color.z < 0) hairMat.color.z = 0;
                }

                Vec3 wo = -current_ray.direction;
                Vec3 baseHairColor = hairMat.color;


                // ===================================================================
                // FULL MARSCHNER HAIR SHADING WITH STRONG SHADOWS
                // ===================================================================

                Vec3 T = hairHit.tangent;
                Vec3 N = hairHit.normal; // Camera-facing normal for shadow offsets

                // Re-verify tangent if needed (Optional, usually hairHit.tangent is stable)
                if (T.length() < 0.1f) T = Vec3(0, 1, 0);

                // Get main light direction (sun or first light)
                Vec3 mainLightDir = Vec3(0.5f, 0.8f, 0.3f).normalize();
                Vec3 mainLightColor = Vec3(1.0f, 0.95f, 0.9f); // Warm sunlight
                float mainLightDist = 1e6f;

                if (!lights.empty()) {
                    auto& firstLight = lights[0];
                    if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(firstLight)) {
                        // DirectionalLight::getDirection() already returns the vector towards the light source (-direction)
                        // This is exactly what we need for L in shading and shadow tracing
                        mainLightDir = dl->getDirection(hairHit.position).normalize();
                        mainLightColor = dl->getIntensity(hairHit.position, Vec3(0));
                        mainLightDist = 1e6f;
                    }
                    else if (auto pl = std::dynamic_pointer_cast<PointLight>(firstLight)) {
                        Vec3 toLight = pl->getPosition() - hairHit.position;
                        mainLightDist = toLight.length();
                        mainLightDir = toLight / (mainLightDist + 1e-6f);
                        mainLightColor = pl->getIntensity(hairHit.position, pl->getPosition());
                    }
                }

                mainLightDir = mainLightDir.normalize();

                // ========================
                // DEEP SHADOW CALCULATION
                // ========================
                float totalShadow = 1.0f;
                {
                    // 1. Mesh Shadow (Solid occlusion)
                    Ray meshShadowRay(hairHit.position + N * 0.0002f, mainLightDir);
                    HitRecord mRec;
                    if (bvh && bvh->hit(meshShadowRay, 0.0002f, mainLightDist, mRec, true)) {
                        totalShadow = 0.0f;
                    }

                    // 2. Hair Transmission Shadow (Light filtering through strands)
                    if (totalShadow > 0.0f && !hairSystem.isBVHDirty()) {
                        Vec3 shadowOrigin = hairHit.position + N * 0.001f; // Larger offset to avoid self-hit
                        Hair::HairHitInfo sHit;
                        int hits = 0;
                        float shadowTraceDist = std::min(mainLightDist, 100.0f);

                        // Trace light through multiple strands for realistic deep shadows
                        int maxHits = 8;
                        while (hits < maxHits && shadowTraceDist > 0.01f) {
                            if (hairSystem.intersect(shadowOrigin, mainLightDir, 0.001f, shadowTraceDist, sHit)) {
                                totalShadow *= 0.4f; // Stronger shadow per strand for visibility
                                shadowOrigin = sHit.position + mainLightDir * 0.002f;
                                shadowTraceDist -= (sHit.t + 0.002f);
                                hits++;
                                if (totalShadow < 0.01f) { totalShadow = 0.0f; break; }
                            }
                            else {
                                break;
                            }
                        }
                    }
                }



                // ========================
                // PURE MARSCHNER SHADING
                // ========================
                // Pass longitudinal (v) and azimuthal (u) correctly
                // Final Color = BSDF * LightColor * Shadow + Ambient
                Vec3 bsdf = Hair::HairBSDF::evaluate(wo, mainLightDir, T, hairMat, hairHit.v, hairHit.u);
                
                // Physically plausible ambient: Small portion of sunlight as 'sky' contribution
                Vec3 hair_color = (bsdf * totalShadow * mainLightColor) + (baseHairColor * mainLightColor * 0.02f);




                // ========================
                // ADDITIONAL LIGHTS
                // ========================
                for (size_t li = 0; li < lights.size(); li++) {
                    const auto& light = lights[li];
                    Vec3 lightDir, lightPos;
                    float lightDist = 0.0f;
                    Vec3 Li(0.0f);

                    if (auto pl = std::dynamic_pointer_cast<PointLight>(light)) {
                        // Random point on the sphere (radius) → soft shadow parity with OptiX/Vulkan
                        lightPos = pl->random_point();
                        Vec3 toLight = lightPos - hairHit.position;
                        lightDist = toLight.length();
                        lightDir = toLight / lightDist;
                        Li = pl->getIntensity(hairHit.position, lightPos);
                    }
                    else if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(light)) {
                        if (li == 0) continue; // Skip main light (already processed)
                        lightDir = -dl->getDirection(hairHit.position);
                        lightDist = 1e6f;
                        Li = dl->getIntensity(hairHit.position, Vec3(0));
                    }
                    else if (auto al = std::dynamic_pointer_cast<AreaLight>(light)) {
                        lightPos = al->random_point();
                        Vec3 toLight = lightPos - hairHit.position;
                        lightDist = toLight.length();
                        lightDir = toLight / lightDist;
                        Li = al->getIntensity(hairHit.position, lightPos);
                    }
                    else if (auto sl = std::dynamic_pointer_cast<SpotLight>(light)) {
                        lightPos = sl->position;
                        Vec3 toLight = lightPos - hairHit.position;
                        lightDist = toLight.length();
                        if (lightDist < 1e-4f) continue;
                        lightDir = toLight / lightDist;
                        // getIntensity already applies cone falloff; returns zero outside the cone
                        Li = sl->getIntensity(hairHit.position, lightPos);
                        if (Li.x <= 0.0f && Li.y <= 0.0f && Li.z <= 0.0f) continue;
                    }
                    else {
                        continue;
                    }

                    if (lightDist < 0.001f) continue;

                    // Shadow check (mesh + hair)
                    Ray shadowRay(hairHit.position + N * 0.003f, lightDir);
                    HitRecord shadowRec;
                    bool inShadow = bvh && bvh->hit(shadowRay, 0.001f, lightDist - 0.01f, shadowRec, true);

                    if (!inShadow) {
                        // Hair self-shadow (only if BVH is ready)
                        float lightShadow = 1.0f;
                        if (!hairSystem.isBVHDirty()) {
                            Hair::HairHitInfo lsh;
                            Vec3 org = shadowRay.origin;
                            float d = std::min(lightDist, 100.0f);
                            int h = 0;
                            // Standard 8-step deep shadow for additional lights
                            while (h < 8 && d > 0.01f && lightShadow > 0.01f) {
                                if (hairSystem.intersect(org, lightDir, 0.002f, d, lsh)) {
                                    lightShadow *= 0.4f;
                                    org = lsh.position + lightDir * 0.003f;
                                    d -= lsh.t + 0.003f;
                                    h++;
                                }
                                else break;
                            }
                        }



                        // Evaluate BSDF for this light (hairHit.v is along, hairHit.u is h)
                        Vec3 lBsdf = Hair::HairBSDF::evaluate(wo, lightDir, T, hairMat, hairHit.v, hairHit.u);

                        // For hair, we don't use standard NdotL with camera-facing N
                        // The BSDF already handles the cylindrical scattering geometry.
                        Vec3 lightContrib = lBsdf * Li * lightShadow;
                        hair_color = hair_color + lightContrib;
                    }
                }

                // Final output
                color += throughput * toVec3f(hair_color);
                return toVec3(color);

            }

        }
        if (!hit_any) {
            // No hit at all? Skip second pass and go to background
        }
        else if (rec.gas_volume || rec.vdb_volume || (rec.materialPtr && rec.materialPtr->type() == MaterialType::Volumetric)) {
            // Hit a volume? 2. Second Pass: Find what's behind/inside (ignore volumes)
            if (bvh) {
                hit_solid = bvh->hit(current_ray, 0.001f, std::numeric_limits<float>::infinity(), solid_rec, true);
                if (hit_solid) {
                    HitMaterialResolver::resolveSurfaceData(solid_rec);
                }
            }
        }
        else {
            // Hit a solid surface directly. Use it for both.
            solid_rec = rec;
            hit_solid = true;
        }

        if (bounce == 0 && hit_solid) {
            // Preserve the primary hit distance for volume-first paths so aerial
            // perspective stays stable while progressive samples accumulate.
            first_hit_t = solid_rec.t;
            if (solid_rec.surface_override.valid) {
                first_hit_transmission = std::clamp(solid_rec.surface_override.transmission, 0.0f, 1.0f);
            }
            else if (auto pbsdf = dynamic_cast<PrincipledBSDF*>(solid_rec.materialPtr)) {
                first_hit_transmission = std::clamp(pbsdf->getTransmission(solid_rec.uv), 0.0f, 1.0f);
            }
        }

        // --- God Rays (Bounce 0 only) ---
        if (bounce == 0 && world.data.mode == WORLD_MODE_NISHITA &&
            world.data.nishita.godrays_enabled && world.data.nishita.godrays_intensity > 0.0f) {
            float hit_dist = hit_any ? rec.t : world.data.nishita.fog_distance * 0.5f;
            Vec3 god_rays = VolumetricRenderer::calculateGodRays(
                scene, world.data, current_ray, hit_dist, bvh, world.getLUT());

            // Check for NaNs/Infs to prevent black screen
            if (std::isfinite(god_rays.x) && std::isfinite(god_rays.y) && std::isfinite(god_rays.z)) {
                color += throughput * toVec3f(god_rays);
            }
        }

        // === VOLUMETRIC ABSORPTION (Beer's Law) ===
        // Apply absorption based on the distance traveled in the current medium (matches GPU)
        if ((rec.gas_volume || rec.vdb_volume || hit_solid) && (current_medium_absorb.x > 0.0f || current_medium_absorb.y > 0.0f || current_medium_absorb.z > 0.0f)) {
            float dist = hit_solid ? solid_rec.t : rec.t;
            Vec3f transmission(
                expf(-current_medium_absorb.x * dist),
                expf(-current_medium_absorb.y * dist),
                expf(-current_medium_absorb.z * dist)
            );
            throughput *= transmission;
        }

        if (!hit_any && !hit_solid) {
            // --- Infinite Grid Logic (Floor Plane Y=0) ---
            Vec3f final_bg_color = render_settings.show_background ?
                toVec3f(world.evaluate(current_ray.direction, current_ray.origin)) :
                Vec3f(0.0f);

            // Only draw grid if looking down (dir.y < 0) AND NOT in final render mode AND grid enabled
            if (render_settings.grid_enabled && !render_settings.is_final_render_mode && current_ray.direction.y < -0.0001f) {
                float t = -current_ray.origin.y / current_ray.direction.y;
                if (t > 0.0f) {
                    Vec3 p = current_ray.origin + current_ray.direction * t;

                    // --- Improved Infinite Grid Shader ---

                    // 1. Distance Fading (Horizon Fog)
                    // Increased fade start significantly to reduce "foggy" look in viewport
                    float fade_start = 100.0f;
                    float fade_end = render_settings.grid_fade_distance;
                    if (fade_end < fade_start) fade_end = fade_start + 100.0f;

                    float dist = t;
                    float alpha_fade = 1.0f - std::clamp((dist - fade_start) / (fade_end - fade_start), 0.0f, 1.0f);

                    if (alpha_fade > 0.0f) {
                        // 2. Grid Structure (Primary & Secondary Lines)
                        float scale_primary = 10.0f;  // Major lines every 10 units
                        float scale_secondary = 1.0f; // Minor lines every 1 unit

                        // Line width scales with distance to reduce aliasing
                        float line_width_base = 0.02f;
                        float line_width = line_width_base * (1.0f + dist * 0.02f);

                        // Modulo coordinates
                        float x_mod_p = abs(fmod(p.x, scale_primary));
                        float z_mod_p = abs(fmod(p.z, scale_primary));
                        float x_mod_s = abs(fmod(p.x, scale_secondary));
                        float z_mod_s = abs(fmod(p.z, scale_secondary));

                        // Check line hints
                        // Handle wrap-around near scale boundary
                        auto is_line = [&](float val, float scale, float width) {
                            return val < width || val >(scale - width);
                            };

                        bool x_line_p = is_line(x_mod_p, scale_primary, line_width);
                        bool z_line_p = is_line(z_mod_p, scale_primary, line_width);
                        bool x_line_s = is_line(x_mod_s, scale_secondary, line_width);
                        bool z_line_s = is_line(z_mod_s, scale_secondary, line_width);

                        // Axis Lines (Thicker)
                        bool x_axis = abs(p.z) < line_width * 2.5f;
                        bool z_axis = abs(p.x) < line_width * 2.5f;

                        // Determine Color & Alpha
                        Vec3f grid_col(0.0f);
                        float grid_alpha = 0.0f;

                        if (x_axis) {
                            grid_col = Vec3f(0.8f, 0.2f, 0.2f); grid_alpha = 0.9f; // Red X
                        }
                        else if (z_axis) {
                            grid_col = Vec3f(0.2f, 0.8f, 0.2f); grid_alpha = 0.9f; // Green Z
                        }
                        else if (x_line_p || z_line_p) {
                            grid_col = Vec3f(0.40f); grid_alpha = 0.5f; // Major Lines (Darker Grey)
                        }
                        else if (x_line_s || z_line_s) {
                            grid_col = Vec3f(0.25f); grid_alpha = 0.2f; // Minor Lines (Subtle)
                        }

                        // Compose
                        if (grid_alpha > 0.0f) {
                            float final_alpha = grid_alpha * alpha_fade;
                            // Alpha blending
                            final_bg_color = final_bg_color * (1.0f - final_alpha) + grid_col * final_alpha;
                        }
                    }
                }
            }

            // --- Background contribution (matching GPU exactly) ---
            float bg_factor = background_factor(bounce);
            Vec3f bg_contribution = final_bg_color * bg_factor;
            color += throughput * bg_contribution;
            break;
        }

        // Limit volumes by solid depth
        bool hit_solid_inside = false;
        float t_solid = std::numeric_limits<float>::infinity();
        if (hit_solid) t_solid = solid_rec.t;

        // ---------------------------------------------------------------------
        // GAS VOLUME RENDERING (Legacy Path)
        // ---------------------------------------------------------------------
        if (rec.gas_volume && rec.gas_volume->render_path == GasVolume::VolumeRenderPath::Legacy) {
            Vec3 min_b, max_b;
            rec.gas_volume->getWorldBounds(min_b, max_b);

            float t_enter = rec.t;
            float t_exit = t_enter;

            float t1 = std::numeric_limits<float>::infinity();
            for (int i = 0; i < 3; ++i) {
                float invD = 1.0f / current_ray.direction[i];
                float t1_slab = (max_b[i] - current_ray.origin[i]) * invD;
                if (invD < 0.0f) {
                    float t0_slab = (min_b[i] - current_ray.origin[i]) * invD;
                    t1_slab = t0_slab;
                }
                t1 = std::min(t1, t1_slab);
            }
            t_exit = t1;

            if (hit_solid && t_solid < t_exit) {
                t_exit = t_solid;
                hit_solid_inside = true;
            }

            float current_transmittance = 1.0f;
            if (t_exit > t_enter) {
                auto shader = rec.gas_volume->getShader();

                // Step size from shader or default (FASTER rendering with larger steps)
                float step_size = shader ? shader->quality.step_size : 0.15f;
                float anisotropy_g = shader ? shader->scattering.anisotropy : 0.0f;
                float anisotropy_back = shader ? shader->scattering.anisotropy_back : -0.3f;
                float lobe_mix = shader ? shader->scattering.lobe_mix : 0.7f;
                float multi_scatter = shader ? shader->scattering.multi_scatter : 0.3f;

                int max_steps = shader ? shader->quality.max_steps : 256;

                float t = t_enter;
                Vec3f accumulated_emission(0.0f);
                t += ((float)rand() / RAND_MAX) * step_size;

                float density_scale = 1.0f;
                float absorption_coeff = 0.1f;
                float blackbody_intensity = 10.0f;
                float temperature_scale_shader = 1.0f;
                bool use_blackbody = false;

                if (shader) {
                    density_scale = shader->density.multiplier;
                    absorption_coeff = shader->absorption.coefficient;
                    blackbody_intensity = shader->emission.blackbody_intensity;
                    temperature_scale_shader = shader->emission.temperature_scale;
                    use_blackbody = (shader->emission.mode == VolumeEmissionMode::Blackbody);
                }

                float ambient_temp = rec.gas_volume->getSettings().ambient_temperature;
                int steps = 0;

                while (t < t_exit && steps < max_steps) {
                    Vec3 pos = current_ray.at(t);
                    float density = rec.gas_volume->sampleDensity(pos);

                    if (density > 0.001f) {
                        float sigma_t = (density * density_scale) * (1.0f + absorption_coeff);
                        float step_trans = exp(-sigma_t * step_size);

                        // Emission from temperature (fire/flame)
                        if (use_blackbody) {
                            float temperature = rec.gas_volume->sampleTemperature(pos);
                            float flame = rec.gas_volume->sampleFlameIntensity(pos);

                            // Use temperature directly in Kelvin for blackbody
                            // Apply temperature_scale as multiplier
                            float temp_k = (temperature - ambient_temp) * temperature_scale_shader;
                            temp_k = std::max(100.0f, std::min(temp_k + 500.0f, 10000.0f)); // +500 baseline for visible color
                            float tk = temp_k / 100.0f;
                            float r, g, b;

                            if (tk <= 66.0f) {
                                r = 255.0f;
                                g = std::max(0.0f, 99.4708f * logf(tk) - 161.12f);
                                b = (tk <= 19.0f) ? 0.0f : std::max(0.0f, 138.52f * logf(tk - 10.0f) - 305.04f);
                            }
                            else {
                                r = 329.7f * powf(tk - 60.0f, -0.133f);
                                g = 288.12f * powf(tk - 60.0f, -0.0755f);
                                b = 255.0f;
                            }

                            Vec3f bb_color(r / 255.0f, g / 255.0f, b / 255.0f);
                            bb_color.x = std::max(0.0f, std::min(bb_color.x, 1.0f));
                            bb_color.y = std::max(0.0f, std::min(bb_color.y, 1.0f));
                            bb_color.z = std::max(0.0f, std::min(bb_color.z, 1.0f));

                            float flame_boost = 1.0f + flame * 5.0f;
                            float brightness = density * density_scale * blackbody_intensity * flame_boost;
                            Vec3f emission = bb_color * brightness;

                            accumulated_emission += emission * (1.0f - step_trans) * current_transmittance;
                        }

                        current_transmittance *= step_trans;
                    }
                    if (current_transmittance < 0.01f) break;
                    t += step_size;
                    steps++;
                }

                // Apply emission to color
                color = color + Vec3f(accumulated_emission.x, accumulated_emission.y, accumulated_emission.z);

                if (bounce == 0) {
                    if (first_vol_t < 0.0f && (1.0f - current_transmittance) > 0.01f) first_vol_t = t_enter;
                    vol_trans_accum *= current_transmittance;
                }

                throughput *= current_transmittance;
                if (current_transmittance < 0.01f) break;
            }

            if (hit_solid_inside) {
                rec = solid_rec;
            }
            else {
                // Move ray to exit point to continue tracing scene
                current_ray = Ray(current_ray.at(t_exit + 0.001f), current_ray.direction);

                // BOUNCE REFUND for Transparent Gas Volumes (Match VDB/GPU)
                if (current_transmittance > 0.01f) {
                    bounce--;
                }

                continue;
            }
        }


        // --- VDB Volume Rendering (High Quality Unified Path) ---
        const VDBVolume* vdb = rec.vdb_volume;
        int live_vol_id = -1;
        std::shared_ptr<VolumeShader> vol_shader = nullptr;
        Matrix4x4 inv_transform = Matrix4x4::identity();
        float den_scale = 1.0f;

        if (vdb) {
            live_vol_id = vdb->getVDBVolumeID();
            vol_shader = vdb->volume_shader;
            inv_transform = vdb->getInverseTransform();
            den_scale = vdb->density_scale;
        }
        else if (rec.gas_volume && rec.gas_volume->render_path == GasVolume::VolumeRenderPath::VDBUnified) {
            live_vol_id = rec.gas_volume->live_vdb_id;
            vol_shader = rec.gas_volume->getShader();
            Transform* gv_trans = rec.gas_volume->getTransformPtr();
            if (gv_trans) {
                Matrix4x4 m = gv_trans->getFinal();
                Vec3 gsize = rec.gas_volume->getSettings().grid_size;
                if (gsize.x > 0 && gsize.y > 0 && gsize.z > 0) {
                    m = m * Matrix4x4::scaling(Vec3(1.0f / gsize.x, 1.0f / gsize.y, 1.0f / gsize.z));
                }
                inv_transform = m.inverse();
            }
            den_scale = 1.0f;
        }
        float actual_step_size = 0.0f;
        const bool procedural_vdb = (vdb && vdb->isProceduralVolume());
        if (live_vol_id >= 0 || procedural_vdb) {
            // Get entry and exit points
            float t_enter, t_exit;
            bool hit_box = false;
           
            if (vdb) {
                hit_box = vdb->intersectTransformedAABB(current_ray, 0.001f, std::numeric_limits<float>::infinity(), t_enter, t_exit);
            }
            else {
                AABB box; rec.gas_volume->bounding_box(0, 0, box);
                hit_box = box.hit_interval(current_ray, 0.001f, std::numeric_limits<float>::infinity(), t_enter, t_exit);
            }

            if (hit_box) {
                if (hit_solid && t_solid < t_exit) {
                    t_exit = t_solid;
                    hit_solid_inside = true;
                }
                if (t_enter < 0.001f) t_enter = 0.001f;

                float step_size = vol_shader ? vol_shader->quality.step_size : 0.1f;
                if (step_size < 0.001f) step_size = 0.001f;

                float density_scale = (vol_shader ? vol_shader->density.multiplier : 1.0f) * den_scale;

                // --- INITIALIZE VOLUME PARAMETERS (Moved up for scoping) ---
                int max_steps = vol_shader ? vol_shader->quality.max_steps : 256;
                Vec3 aabb_size = vdb ? (vdb->getWorldBounds().max - vdb->getWorldBounds().min) : (rec.gas_volume->getSettings().grid_size);
                float volume_size = aabb_size.length();
                actual_step_size = std::max(step_size, volume_size / (float)max_steps);

                auto& mgr = VDBVolumeManager::getInstance();

                // --- VOLUMETRIC RENDERING (CPU) ---
                // Physical Integration using Beer-Lambert Law

                // Access generic Volume Shader properties
                auto shader = vol_shader;

                step_size = shader ? shader->quality.step_size : 0.1f;
                // Avoid infinite loops with bad step size
                if (step_size < 0.001f) step_size = 0.001f;

                density_scale = (shader ? shader->density.multiplier : 1.0f) * den_scale;

                // Shader parameters - Convert to Vec3f for rendering consistency
                Vec3 albedo_raw = shader ? shader->scattering.color : Vec3(1.0f);
                Vec3f volume_albedo = toVec3f(albedo_raw);
                float scattering_intensity = shader ? shader->scattering.coefficient : 1.0f;

                // ===============================================================
                // Absorption parameters (sigma_a)
                // ===============================================================
                Vec3 absorption_color_raw = shader ? shader->absorption.color : Vec3(0.0f);
                Vec3f absorption_color = toVec3f(absorption_color_raw);
                float absorption_coeff = shader ? shader->absorption.coefficient : 0.0f;

                // ===============================================================
                // Emission parameters
                // ===============================================================
                VolumeEmissionMode emission_mode = shader ? shader->emission.mode : VolumeEmissionMode::None;
                Vec3 emission_color_raw = shader ? shader->emission.color : Vec3(1.0f, 0.5f, 0.1f);
                Vec3f emission_color = toVec3f(emission_color_raw);
                float emission_intensity = shader ? shader->emission.intensity : 0.0f;

                // ============================������������������������������������������
                // Density and Remapping Properties
                // ============================������������������������������������������
                float remap_low = shader ? shader->density.remap_low : 0.0f;
                float remap_high = shader ? shader->density.remap_high : 1.0f;
                float remap_range = std::max(1e-5f, remap_high - remap_low);
                float shadow_strength = shader ? shader->quality.shadow_strength : 1.0f;              
                float anisotropy_g = shader ? shader->scattering.anisotropy : 0.0f;
                float anisotropy_back = shader ? shader->scattering.anisotropy_back : -0.3f;
                float lobe_mix = shader ? shader->scattering.lobe_mix : 0.7f;
                float multi_scatter = shader ? shader->scattering.multi_scatter : 0.3f;

                auto blackbody_to_rgb = [](float kelvin) -> Vec3 {
                    kelvin = std::max(1000.0f, std::min(kelvin, 40000.0f));
                    float t = kelvin / 100.0f;
                    float r, g, b;
                    if (t <= 66.0f) {
                        r = 255.0f;
                        g = 99.4708025861f * std::log(std::max(t, 1e-6f)) - 161.1195681661f;
                        if (t <= 19.0f) b = 0.0f;
                        else b = 138.5177312231f * std::log(std::max(t - 10.0f, 1e-6f)) - 305.0447927307f;
                    } else {
                        r = 329.698727446f * std::pow(t - 60.0f, -0.1332047592f);
                        g = 288.1221695283f * std::pow(t - 60.0f, -0.0755148492f);
                        b = 255.0f;
                    }
                    return Vec3(std::clamp(r / 255.0f, 0.0f, 1.0f), 
                                std::clamp(g / 255.0f, 0.0f, 1.0f), 
                                std::clamp(b / 255.0f, 0.0f, 1.0f));
                };

                // Initialize path state
                float current_transparency = 1.0f;
                Vec3f accumulated_vol_color(0.0f);

                // Jitter to reduce banding
                float jitter = ((float)rand() / RAND_MAX) * actual_step_size;
                float t = t_enter + jitter;

                int steps = 0;

                while (t < t_exit && steps < max_steps && current_transparency > 0.01f) {
                    float threshold = ((float)rand() / RAND_MAX) * 0.01f;

                    Vec3 p = current_ray.at(t);
                    Vec3 local_p = inv_transform.transform_point(p);

                    float density = procedural_vdb
                        ? rt_sample_procedural_cloud_cpu(local_p, vdb, world.data.nishita)
                        : mgr.sampleDensityCPU(live_vol_id, local_p.x, local_p.y, local_p.z);

                    float edge_falloff = shader ? shader->density.edge_falloff : 0.0f;
                    if (edge_falloff > 0.0f && density > 0.0f) {
                        Vec3 local_min(0), local_max(1);
                        if (vdb) {
                            local_min = vdb->getLocalBoundsMin();
                            local_max = vdb->getLocalBoundsMax();
                        } else {
                            local_max = rec.gas_volume->getSettings().grid_size;
                        }

                        float dx = std::min(local_p.x - local_min.x, local_max.x - local_p.x);
                        float dy = std::min(local_p.y - local_min.y, local_max.y - local_p.y);
                        float dz = std::min(local_p.z - local_min.z, local_max.z - local_p.z);
                        float edge_dist = std::min({ dx, dy, dz });

                        if (edge_dist < edge_falloff) {
                            float edge_factor = edge_dist / edge_falloff;
                            density *= edge_factor * edge_factor;
                        }
                    }

                    float d_remapped = std::max(0.0f, (density - remap_low) / remap_range);
                    float d = d_remapped * density_scale;
                    float sigma_s = d * scattering_intensity;
                    float cutoff_threshold = shader ? shader->density.cutoff_threshold : 0.04f;
                    if (cutoff_threshold <= 0.0f) cutoff_threshold = 0.04f;
                    float scatter_keep = std::clamp((sigma_s * actual_step_size) / cutoff_threshold, 0.0f, 1.0f);

                    if (((float)rand() / RAND_MAX) <= scatter_keep) {
                        if (bounce == 0 && first_vol_t < 0.0f && d > 0.05f) {
                            first_vol_t = t;
                        }

                        float sigma_a = d * absorption_coeff;
                        float sigma_t = sigma_s + sigma_a;

                        float albedo_avg = volume_albedo.luminance();
                        float T_single = exp(-sigma_t * actual_step_size);
                        float T_multi_p = exp(-sigma_t * actual_step_size * 0.25f);
                        float step_transmittance = T_single * (1.0f - multi_scatter * albedo_avg) + 
                                                   T_multi_p * (multi_scatter * albedo_avg);

                        Vec3f total_radiance(0.0f);

                        // --- LIGHT SAMPLING ---
                        for (const auto& light : lights) {
                            if (!light) continue;

                            Vec3 light_dir;
                            float light_dist;

                            if (light->type() == LightType::Directional) {
                                light_dir = light->getDirection(p);
                                light_dist = 1e9f;
                            } else {
                                Vec3 to_light = light->position - p;
                                light_dist = to_light.length();
                                light_dir = to_light / std::max(light_dist, 0.0001f);
                            }

                            Vec3f light_intensity = toVec3f(light->getIntensity(p, light->position));
                            
                            // --- PARITY: Atmospheric Extinction for Sun (Matches GPU) ---
                            // If this is the main sun light in Nishita mode, apply transmittance
                            if (light->type() == LightType::Directional && world.data.mode == WORLD_MODE_NISHITA && 
                                world.getLUT() && world.getLUT()->is_initialized()) {
                                
                                float Rg = world.data.nishita.planet_radius;
                                if (Rg < 1000.0f) Rg = 6360000.0f;
                                
                                // Coordinates: Planet center is (0, -Rg, 0)
                                Vec3 p_planet = p + Vec3(0, Rg, 0); 
                                float altitude = p_planet.length() - Rg;
                                Vec3 up = p_planet.normalize(); 
                                
                                float cosTheta = Vec3::dot(up, light_dir);
                                float3 t_sun = world.getLUT()->sampleTransmittance(cosTheta, altitude, world.data.nishita.atmosphere_height);
                                light_intensity = light_intensity * toVec3f((t_sun.x, t_sun.y, t_sun.z));
                            }
                            if (light_intensity.luminance() < 1e-5f) continue;

                            Ray shadow_ray_vol(p + light_dir * 0.001f, light_dir);
                            float shadow_transmittance = 1.0f;
                            HitRecord shadow_rec;
                            
                            if (bvh->hit(shadow_ray_vol, 0.001f, light_dist, shadow_rec)) {
                                HitMaterialResolver::resolveMaterialPointers(shadow_rec);
                                if (!shadow_rec.vdb_volume) {
                                    shadow_transmittance = 0.0f;
                                } else {
                                    float density_accum = 0.0f;
                                    float tv_enter, tv_exit;
                                    if (vdb->intersectTransformedAABB(shadow_ray_vol, 0.0f, light_dist, tv_enter, tv_exit)) {
                                        int shadow_steps = shader ? shader->quality.shadow_steps : 8;
                                        float shadow_march_step = volume_size / std::max((float)shadow_steps, 1.0f);
                                        if (shadow_march_step < 0.01f) shadow_march_step = 0.01f;
                                        if (tv_exit > light_dist) tv_exit = light_dist;
                                        float t_shadow = ((float)rand() / RAND_MAX) * shadow_march_step;

                                        while (t_shadow < tv_exit) {
                                            Vec3 slocal_p = inv_transform.transform_point(shadow_ray_vol.at(t_shadow));
                                            float s_density = procedural_vdb
                                                ? rt_sample_procedural_cloud_cpu(slocal_p, vdb, world.data.nishita)
                                                : mgr.sampleDensityCPU(live_vol_id, slocal_p.x, slocal_p.y, slocal_p.z);
                                            float s_rem = std::max(0.0f, (s_density - remap_low) / remap_range);
                                            if (s_rem > 1e-4f) {
                                                density_accum += (s_rem * density_scale * (scattering_intensity + absorption_coeff)) * shadow_march_step;
                                            }
                                            if (density_accum > 10.0f) break; 
                                            t_shadow += shadow_march_step;
                                        }

                                        float beers = exp(-density_accum);
                                        float phys_trans = beers;
                                        
                                        // Match GPU Fix: Only use multi-scatter softening if scattering is actually present
                                        if (scattering_intensity > 1e-6f && multi_scatter > 1e-6f) {
                                            float beers_soft = exp(-density_accum * 0.25f);
                                            phys_trans = beers * (1.0f - multi_scatter * albedo_avg) + beers_soft * (multi_scatter * albedo_avg);
                                        }
                                        shadow_transmittance = 1.0f - shadow_strength * (1.0f - phys_trans);
                                    }
                                }
                            }

                            if (shadow_transmittance > 0.01f) {
                                float cos_theta = Vec3::dot(current_ray.direction.normalize(), light_dir);
                                auto hg = [](float ct, float g) {
                                    float g2 = g * g;
                                    float denom = 1.0f + g2 - 2.0f * g * ct;
                                    return (1.0f - g2) / (4.0f * 3.14159f * std::pow(std::max(denom, 0.0001f), 1.5f));
                                };
                                float phase = hg(cos_theta, anisotropy_g) * lobe_mix + hg(cos_theta, anisotropy_back) * (1.0f - lobe_mix);
                                float powder = 1.0f - std::exp(-d * 2.0f);
                                float forward_bias = 0.5f + 0.5f * std::max(0.0f, cos_theta);
                                phase *= (1.0f + powder * forward_bias * 0.5f);

                                total_radiance += light_intensity * shadow_transmittance * phase;
                            }
                        }

                        // Sky Lighting (directional ambient): blend up with view direction
                        Vec3 sky_dir = (current_ray.direction * 0.45f + Vec3(0, 1, 0) * 0.55f).normalize();
                        total_radiance += toVec3f(world.evaluate(sky_dir, current_ray.origin))
                                       * 0.15f * world.data.nishita.atmosphere_intensity;

                        // --- EMISSION ---
                        Vec3f step_emission(0.0f);
                        if (emission_mode == VolumeEmissionMode::Constant) {
                            step_emission = emission_color * emission_intensity * d;
                        } else if (emission_mode == VolumeEmissionMode::Blackbody || emission_mode == VolumeEmissionMode::ChannelDriven) {
                            float temp_val = mgr.hasTemperatureGrid(live_vol_id) ? mgr.sampleTemperatureCPU(live_vol_id, local_p.x, local_p.y, local_p.z) : density;
                            
                            float kelvin = 0.0f;
                            float t_ramp = 0.0f;
                            float t_scale = shader ? shader->emission.temperature_scale : 1.0f;
                            float t_max = shader ? shader->emission.temperature_max : 1500.0f;

                            if (temp_val > 20.0f) {
                                kelvin = temp_val * t_scale;
                                t_ramp = temp_val / std::max(1.0f, t_max);
                            } else {
                                // Fallback for density-driven or normalized channels (GPU Parity)
                                kelvin = (temp_val * 3000.0f + 1000.0f) * t_scale;
                                // For density-driven fire, we want it to map more aggressively to the hot end
                                t_ramp = std::max(0.0f, std::min(1.0f, temp_val * 2.0f)); 
                            }

                            if (shader && shader->emission.color_ramp.enabled) {
                                step_emission = toVec3f(shader->emission.color_ramp.sample(t_ramp)) * d * shader->emission.blackbody_intensity;
                            } else {
                                step_emission = toVec3f(blackbody_to_rgb(kelvin)) * d * (shader ? shader->emission.blackbody_intensity : 10.0f);
                            }
                        }

                        // --- VOLUMETRIC INTEGRATION: Multi-Scattering Stable (Parity with GPU) ---
                        float sigma_t_safe = std::max(sigma_t, 1e-6f);
                        Vec3f albedo = (volume_albedo);

                        // Multi-scattering energy gain (Simulates diffuse internal bounces)
                        Vec3f ms_boost = Vec3f(1.0f) + albedo * multi_scatter * 2.0f;
                        Vec3f source = (albedo * total_radiance * sigma_s * ms_boost + step_emission);
                        
                        // Stable Analytical Integration over step
                        Vec3f step_color = source * ((1.0f - step_transmittance) );
                        accumulated_vol_color += step_color * current_transparency;

                        current_transparency *= step_transmittance;
                    }

                    if (current_transparency < 0.01f) break;
                    t += actual_step_size;
                    steps++;
                }

                // Apply volumetric result to path tracer state
                color += throughput * accumulated_vol_color;
                if (bounce == 0) {
                    if (first_vol_t < 0.0f && (1.0f - current_transparency) > 0.01f) first_vol_t = t_enter;
                    vol_trans_accum *= current_transparency;
                }
                throughput *= current_transparency;

                if (hit_solid_inside) {
                    // Handoff to solid surface
                    rec = solid_rec;
                    // Fall through to surface shading
                }
                else {
                    // Move ray to exit point to continue tracing background
                    current_ray = Ray(current_ray.at(t_exit + 0.001f), current_ray.direction);

                    // BOUNCE REFUND for Transparent VDBs (Always refund if not fully opaque)
                    if (current_transparency > 0.01f) {
                        bounce--;
                    }

                    if (current_transparency < 0.01f) break;
                    continue; // Continue to next bounce
                }
            }
            else {
                // AABB intersection failed but VDB was hit - move ray past VDB bounding box
                // This prevents falling through to normal material processing
                AABB world_bounds = vdb->getWorldBounds();
                // Find exit point of world AABB and move ray past it
                float t_far = std::numeric_limits<float>::infinity();
                Vec3 inv_dir = Vec3(1.0f / current_ray.direction.x,
                    1.0f / current_ray.direction.y,
                    1.0f / current_ray.direction.z);
                for (int i = 0; i < 3; i++) {
                    float t1 = (world_bounds.min[i] - current_ray.origin[i]) * inv_dir[i];
                    float t2 = (world_bounds.max[i] - current_ray.origin[i]) * inv_dir[i];
                    t_far = std::min(t_far, std::max(t1, t2));
                }
                current_ray = Ray(current_ray.at(t_far + 0.0001f), current_ray.direction);
                continue; // Continue tracing behind VDB
            }
        }

        // --- Mesh Volume Rendering ---
        if (rec.materialPtr && rec.materialPtr->type() == MaterialType::Volumetric) {
            auto vol = static_cast<Volumetric*>(rec.materialPtr);
            float step_size = vol->getStepSize();
            float density_mult = vol->getDensity();

            // Find exit point
            Ray exit_ray(current_ray.at(rec.t + 0.001f), current_ray.direction);
            HitRecord exit_rec;
            float t_vol_enter = rec.t;
            float t_vol_exit = t_vol_enter + 10.0f; // Default if no exit found
            float march_dist = 10.0f;
            bool found_exit = false;

            // Trace to find backface
            if (bvh->hit(exit_ray, 0.001f, std::numeric_limits<float>::infinity(), exit_rec)) {
                HitMaterialResolver::resolveMaterialPointers(exit_rec);
                // Next hit (could be backface or another object)
                march_dist = exit_rec.t;
                t_vol_exit = t_vol_enter + march_dist;
                found_exit = true;
            }
            else {
                march_dist = 10.0f;
            }

            // Global limit from Dual Pass
            if (hit_solid && t_solid < t_vol_exit) {
                float dist_to_solid = t_solid - t_vol_enter;
                if (dist_to_solid < march_dist) {
                    march_dist = dist_to_solid;
                    t_vol_exit = t_solid;
                    hit_solid_inside = true;
                    found_exit = false; // We didn't reach the backface, we hit an obstruction
                }
            }

            float t = 0.0f; // Relative to entry
            float current_transparency = 1.0f;
            Vec3f accumulated_vol_color(0.0f);

            // Jitter
            float jitter = ((float)rand() / RAND_MAX) * step_size;
            t += jitter;

            int steps = 0;
            int max_steps = vol->getMaxSteps();

            Vec3f volume_albedo = toVec3f(vol->getAlbedo());
            float scattering_intensity = vol->getScattering();
            float absorption_coeff = vol->getAbsorption();
            Vec3f emission_color = toVec3f(vol->getEmissionColor());

            while (t < march_dist && steps < max_steps) {
                Vec3 p = exit_ray.at(t);
                float d = vol->calculate_density(p);

                // Edge falloff for procedural noise to avoid hard cuts if needed
                // (Optional: Implement if vol has properties for it)

                if (d > 0.001f) {
                    float sigma_s_scalar = d * density_mult * scattering_intensity;
                    float sigma_a = d * density_mult * absorption_coeff;
                    float sigma_t = sigma_s_scalar + sigma_a;

                    float step_transmittance = exp(-sigma_t * step_size);

                    // In-Scattering
                    Vec3f total_incoming_light(0.0f);

                    if (current_transparency > 0.01f) {
                        for (const auto& light : lights) {
                            if (!light) continue;

                            Vec3 light_dir;
                            float light_dist;

                            if (light->type() == LightType::Directional) {
                                light_dir = light->getDirection(p);
                                light_dist = 1e9f;
                            }
                            else {
                                Vec3 to_light = light->position - p;
                                light_dist = to_light.length();
                                light_dir = to_light / std::max(light_dist, 0.0001f);
                            }

                            Vec3f light_intensity = toVec3f(light->getIntensity(p, light->position));
                            if (light_intensity.luminance() < 1e-5f) continue;

                            // Shadow Ray
                            Ray shadow_ray_vol(p + light_dir * 0.001f, light_dir);
                            float shadow_transmittance = 1.0f;

                            // Shadow Trace
                            // We reuse the logic: Trace, if hit VDB/Vol -> March, else Opaque
                            float dist_trace = light_dist;
                            int shadow_layers = 0;

                            while (dist_trace > 0.001f && shadow_layers < 2) {
                                HitRecord sidx_rec;
                                if (bvh->hit(shadow_ray_vol, 0.001f, dist_trace, sidx_rec)) {
                                    HitMaterialResolver::resolveMaterialPointers(sidx_rec);

                                    bool is_transp_shadow = false;

                                    if (sidx_rec.vdb_volume) {
                                        // Assume transparent by default for safety
                                        is_transp_shadow = true;

                                        const VDBVolume* vdb_s = sidx_rec.vdb_volume;
                                        float t_enter_s, t_exit_s;
                                        if (vdb_s->intersectTransformedAABB(shadow_ray_vol, 0.001f, dist_trace, t_enter_s, t_exit_s)) {
                                            // Quick Shadow March
                                            float s_step = 0.5f;
                                            if (vdb_s->volume_shader) s_step = vdb_s->volume_shader->quality.step_size * 2.0f;
                                            float t_s = t_enter_s + ((float)rand() / RAND_MAX) * s_step;
                                            auto& mgr = VDBVolumeManager::getInstance();
                                            int vid = vdb_s->getVDBVolumeID();
                                            Matrix4x4 inv = vdb_s->getInverseTransform();
                                            float ds = (vdb_s->volume_shader ? vdb_s->volume_shader->density.multiplier : 1.0f) * vdb_s->density_scale;

                                            while (t_s < t_exit_s) {
                                                Vec3 sp = shadow_ray_vol.at(t_s);
                                                Vec3 local_sp = inv.transform_point(sp);
                                                float dens = mgr.sampleDensityCPU(vid, local_sp.x, local_sp.y, local_sp.z);
                                                if (dens > 0.01f) shadow_transmittance *= exp(-dens * ds * s_step);
                                                if (shadow_transmittance < 0.01f) break;
                                                t_s += s_step;
                                            }

                                            // Advance past exit
                                            float adv = t_exit_s + 0.001f;
                                            shadow_ray_vol = Ray(shadow_ray_vol.at(adv), shadow_ray_vol.direction);
                                            dist_trace -= adv;
                                        }
                                        else {
                                            // BVH Hit but Intersect Failed (Edge/Precision)
                                            // Advance past BVH hit to allow continuation
                                            float adv = sidx_rec.t + 0.001f;
                                            shadow_ray_vol = Ray(shadow_ray_vol.at(adv), shadow_ray_vol.direction);
                                            dist_trace -= adv;
                                        }
                                    }
                                    else if (sidx_rec.materialPtr && sidx_rec.materialPtr->type() == MaterialType::Volumetric) {
                                        auto vol_s = static_cast<Volumetric*>(sidx_rec.materialPtr);
                                        is_transp_shadow = true;

                                        // Find Exit Point or Obstruction
                                        Ray s_exit_ray(shadow_ray_vol.at(sidx_rec.t + 0.001f), shadow_ray_vol.direction);
                                        HitRecord s_exit_rec;
                                        float s_march_dist = dist_trace;

                                        if (bvh->hit(s_exit_ray, 0.001f, dist_trace, s_exit_rec)) {
                                            HitMaterialResolver::resolveMaterialPointers(s_exit_rec);
                                            s_march_dist = s_exit_rec.t;
                                        }

                                        // Homogenous Volume Integration (Beer's Law)
                                        float dens = vol_s->getDensity();
                                        if (dens > 0.0f) {
                                            shadow_transmittance *= exp(-dens * s_march_dist);
                                        }

                                        // Advance Ray
                                        float adv = sidx_rec.t + s_march_dist + 0.001f;
                                        shadow_ray_vol = Ray(shadow_ray_vol.at(adv), shadow_ray_vol.direction);
                                        dist_trace -= adv;
                                    }

                                    if (!is_transp_shadow) {
                                        shadow_transmittance = 0.0f; // Opaque
                                    }
                                }
                                else {
                                    break; // No hit
                                }

                                if (shadow_transmittance < 0.01f) break;
                                shadow_layers++;
                            }
                        }

                        current_transparency *= step_transmittance;
                    }

                    if (current_transparency < 0.01f) break;
                    t += actual_step_size;
                    steps++;
                }

                color += throughput * accumulated_vol_color;
                if (bounce == 0) {
                    // For mesh volumes, we use the entry point if we hit density
                    // (Mesh volumes are usually dense throughout, but we could add a check here too)
                    if (first_vol_t < 0.0f && (1.0f - current_transparency) > 0.01f) {
                        first_vol_t = t_vol_enter;
                    }
                    vol_trans_accum *= current_transparency;
                }
                throughput *= current_transparency;

                // ---------------------------------------------------------
                // EXIT LOGIC: Handle Backface vs Obstruction
                // ---------------------------------------------------------

                // Check if we hit the Volume Backface or a different object (Obstruction)
                bool hit_backface = found_exit && (exit_rec.materialID == rec.materialID);

                if (hit_backface || !found_exit) {
                    // CASE A: Standard Volume Exit (Backface) OR Infinite 
                    // We marched through the volume and exited at the other side.
                    // Advance ray PAST the volume backface to continue tracing scene.
                    current_ray = Ray(exit_ray.at(march_dist + 0.001f), current_ray.direction);

                    // BOUNCE REFUND for Transparent Volumes (Match GPU / Gas / VDB)
                    if (current_transparency > 0.01f) {
                        bounce--;
                    }

                    if (current_transparency < 0.01f) break;
                    continue; // Continue loop with new ray

                }
                else {
                    // CASE B: Hit an Obstruction inside/behind Volume (e.g. Wall)
                    // We hit something that is NOT the volume itself.
                    // We must SHADE this object, not skip it.
                    // We have already accumulated volume opacity up to this point.

                    // Update the main HitRecord to the obstruction
                    rec = exit_rec;

                    // Fall through to standard surface shading code below!
                    // (Do NOT continue loop, do NOT update current_ray yet)

                    // Applying volume throughput to the current path
                    // color += ... was already done above.
                    // throughput *= ... was done above.

                    // Proceed to shade 'rec' (The Wall)
                }
            }
        }
        // --- Normal map application ---
        apply_normal_map(rec);

        // --- Ensure correct normal orientation (faceforward) ---
        Vec3f wo = toVec3f(-current_ray.direction.normalize());
        rec.interpolated_normal = orient_shading_normal(rec.interpolated_normal, rec.normal);
        Vec3f N = toVec3f(rec.interpolated_normal);
        Vec3f geom_N = toVec3f(rec.normal);

        Vec3f hit_pos = toVec3f(rec.point);

        // --- Extract material parameters (with texture sampling) ---
        Vec3f albedo(0.8f);
        float roughness = 0.5f;
        float metallic = 0.0f;
        float specular = 0.5f;
        float opacity = 1.0f;
        float transmission = 0.0f;
        float clearcoatValue = 0.0f;
        float clearcoatRoughnessValue = 0.03f;
        float translucentValue = 0.0f;
        float subsurfaceValue = 0.0f;
        Vec3 subsurfaceColorValue(1.0f);
        float iorValue = 1.45f;
        Vec3f emission(0.0f);
        bool is_water = false;

        if (rec.materialPtr) {
            auto pbsdf = dynamic_cast<PrincipledBSDF*>(rec.materialPtr);
            if (pbsdf) {
                Vec2 uv(rec.u, rec.v);

                // Opacity
                opacity = pbsdf->get_opacity(uv);

                // === STOCHASTIC TRANSPARENCY BOUNCE REFUND ===
                // Prevents "Ghost Silhouette" when max bounces is reached on transparent geometry.
                const bool has_opacity_texture = (pbsdf->opacityProperty.texture != nullptr);
                if (!has_opacity_texture && opacity < 0.999f) {
                    if (Vec3::random_float() > opacity) {
                        transparent_hits++;

                        // Pass-through: move ray and refund bounce count if under limit
                        current_ray = Ray(rec.point + current_ray.direction * 0.001f, current_ray.direction);

                        constexpr int kCpuTransparentPassThroughLimit = 10;
                        if (transparent_hits <= kCpuTransparentPassThroughLimit) {
                            bounce--;
                        }
                        continue;
                    }
                }

                // --- Albedo, Roughness, Metallic (Terrain is pre-resolved into custom data) ---
                if (rec.surface_override.valid) {
                    albedo = toVec3f(rec.surface_override.albedo).clamp(0.01f, 1.0f);
                    roughness = std::clamp(rec.surface_override.roughness, 0.01f, 1.0f);
                    metallic = std::clamp(rec.surface_override.metallic, 0.0f, 1.0f);
                    specular = std::clamp(pbsdf->getSpecularValue(uv), 0.0f, 1.0f);
                    transmission = std::clamp(rec.surface_override.transmission, 0.0f, 1.0f);
                    clearcoatValue = rec.surface_override.clearcoat;
                    clearcoatRoughnessValue = rec.surface_override.clearcoat_roughness;
                    translucentValue = rec.surface_override.translucent;
                    subsurfaceValue = rec.surface_override.subsurface;
                    subsurfaceColorValue = rec.surface_override.subsurface_color;
                    iorValue = rec.surface_override.ior;
                } else {
                    // Albedo
                    Vec3 alb = pbsdf->getPropertyValue(pbsdf->albedoProperty, uv);
                    albedo = toVec3f(alb).clamp(0.01f, 1.0f);

                    roughness = pbsdf->getRoughnessValue(uv);
                    metallic = pbsdf->getMetallicValue(uv);
                    specular = pbsdf->getSpecularValue(uv);
                    transmission = pbsdf->getTransmission(uv);
                    clearcoatValue = pbsdf->clearcoat;
                    clearcoatRoughnessValue = pbsdf->clearcoatRoughness;
                    translucentValue = pbsdf->translucent;
                    subsurfaceValue = pbsdf->subsurface;
                    subsurfaceColorValue = pbsdf->subsurfaceColor;
                    iorValue = pbsdf->getIOR();
                }

                // Terrain normal blending still lives here because it needs the
                // final shading basis after hit-resolution but before BRDF usage.
                if (rec.terrain_id != -1) {
                    TerrainObject* terrain = TerrainManager::getInstance().getTerrain(rec.terrain_id);
                    if (terrain && terrain->splatMap && !terrain->layers.empty()) {
                        Vec3 splat_rgb = terrain->splatMap->get_color_bilinear(uv.u, uv.v);
                        float splat_a = terrain->splatMap->get_alpha_bilinear(uv.u, uv.v);
                        float weights[4] = { (float)splat_rgb.x, (float)splat_rgb.y, (float)splat_rgb.z, splat_a };

                        Vec3 blended_n(0.0f);
                        bool has_n = false;

                        for (int i = 0; i < 4 && i < (int)terrain->layers.size(); ++i) {
                            if (weights[i] < 0.001f) continue;
                            auto layer = dynamic_cast<PrincipledBSDF*>(terrain->layers[i].get());
                            if (!layer || !layer->has_normal_map()) continue;

                            float scale = (i < (int)terrain->layer_uv_scales.size()) ? terrain->layer_uv_scales[i] : 1.0f;
                            Vec2 luv(uv.u * scale, uv.v * scale);
                            Vec3 ns = layer->get_normal_from_map(luv.u, luv.v) * 2.0f - Vec3(1.0f);
                            ns.x = -ns.x; // [FIX] Flip X for basis consistency (matching OptiX)
                            ns.y = -ns.y; // [FIX] Flip Y for terrain normal map orientation
                            blended_n += ns * weights[i];
                            has_n = true;
                        }

                        if (has_n) {
                            Vec3 T, B;
                            Renderer::create_coordinate_system(rec.normal, T, B);
                            Mat3x3 TBN(T, B, rec.normal);
                            rec.interpolated_normal = orient_shading_normal(TBN * blended_n.normalize(), rec.normal);
                            N = toVec3f(rec.interpolated_normal);
                        }
                    }
                }

                // === WATER WAVE SHADER (CPU) ===
                // Detect water materials using sheen > 0 (IS_WATER flag)
                is_water = WaterShader::isActiveWater(pbsdf->sheen, transmission);

                if (is_water) {
                    extern RenderSettings render_settings;
                    float fps = static_cast<float>(render_settings.animation_fps > 0 ? render_settings.animation_fps : 24);
                    float water_time = static_cast<float>(render_settings.animation_current_frame) / fps;

                    // Pack parameters (Mirror GPU raygen.cu packing)
                    WaterParamsCPU params;
                    if (rec.materialPtr->gpuMaterial) {
                        auto& g_mat = *rec.materialPtr->gpuMaterial;
                        params.wave_speed = g_mat.anisotropic;
                        params.wave_strength = g_mat.sheen;
                        params.wave_frequency = g_mat.sheen_tint;

                        params.shallow_color = Vec3(g_mat.emission.x, g_mat.emission.y, g_mat.emission.z);
                        params.deep_color = Vec3(g_mat.albedo.x, g_mat.albedo.y, g_mat.albedo.z);
                        params.absorption_color = Vec3(g_mat.subsurface_color.x, g_mat.subsurface_color.y, g_mat.subsurface_color.z);

                        params.depth_max = g_mat.subsurface * 100.0f;
                        params.absorption_density = g_mat.subsurface_scale;
                        params.clarity = std::fmax(0.1f, 1.0f - params.absorption_density);

                        params.foam_level = g_mat.translucent;
                        params.shore_foam_distance = g_mat.subsurface_radius.x;
                        params.shore_foam_intensity = g_mat.clearcoat;

                        params.caustic_intensity_scale = g_mat.clearcoat_roughness;
                        params.caustic_scale = g_mat.subsurface_radius.y;
                        params.caustic_speed = g_mat.subsurface_anisotropy;

                        params.sss_intensity = g_mat.subsurface_radius.z;
                        params.sss_color = params.absorption_color;

                        params.use_fft_ocean = g_mat.fft_height_tex != 0 &&
                                               g_mat.fft_normal_tex != 0 &&
                                               g_mat.fft_ocean_size > 0.001f;
                        params.fft_ocean_size = g_mat.fft_ocean_size;
                        params.fft_choppiness = g_mat.fft_choppiness;

                        params.micro_detail_strength = g_mat.micro_detail_strength;
                        params.micro_detail_scale = g_mat.micro_detail_scale;
                        params.micro_anim_speed = g_mat.micro_anim_speed;
                        params.micro_morph_speed = g_mat.micro_morph_speed;
                        params.foam_noise_scale = g_mat.foam_noise_scale;
                        params.foam_threshold = g_mat.foam_threshold;
                        
                        // Wind animation for micro details
                        params.wind_direction = g_mat.fft_wind_direction;
                        params.wind_speed = g_mat.fft_wind_speed;
                        params.time = water_time;  // Use pre-calculated water time (seconds)
                    } else {
                        // Use PrincipledBSDF fields directly (UI alignment)
                        params.wave_speed = pbsdf->anisotropic;
                        params.wave_strength = pbsdf->sheen;
                        params.wave_frequency = pbsdf->sheen_tint;
                        
                        params.shallow_color = pbsdf->emissionProperty.color;
                        params.deep_color = pbsdf->albedoProperty.color;
                        params.absorption_color = pbsdf->subsurfaceColor;
                        
                        params.depth_max = pbsdf->subsurface * 100.0f;
                        params.absorption_density = pbsdf->subsurfaceScale;
                        params.clarity = std::fmax(0.1f, 1.0f - params.absorption_density);

                        params.foam_level = 0.01f; // High/Starting quality default
                        params.shore_foam_distance = pbsdf->subsurfaceRadius.x;
                        params.shore_foam_intensity = pbsdf->clearcoat;
                        
                        params.caustic_intensity_scale = pbsdf->clearcoatRoughness;
                        params.caustic_scale = pbsdf->subsurfaceRadius.y;
                        params.caustic_speed = pbsdf->subsurfaceAnisotropy;
                        
                        params.sss_intensity = pbsdf->subsurfaceRadius.z;
                        params.sss_color = params.absorption_color;

                        params.use_fft_ocean = false; // FFT always requires GPU
                        params.micro_detail_strength = 0.0f;
                        params.micro_detail_scale = 1.0f;
                        params.micro_anim_speed = 0.1f;
                        params.micro_morph_speed = 1.0f;
                        params.foam_noise_scale = 1.0f;
                        params.foam_threshold = 0.5f;
                        
                        // Wind animation defaults
                        params.wind_direction = 0.0f;
                        params.wind_speed = 10.0f;
                        params.time = water_time;
                    }

                    // Evaluate water through the shared CPU reference contract.
                    Vec3 base_normal(0.0f, 1.0f, 0.0f);
                    WaterShader::SurfaceParams surface_params = toWaterSurfaceParamsCPU(params);
                    surface_params.time = water_time;
                    WaterShader::SurfaceSample wave = evaluateWaterSurfaceCPU(
                        rec.point, base_normal, surface_params
                    );

                    // Apply wave normal
                    rec.interpolated_normal = orient_shading_normal(wave.normal, rec.normal);
                    N = toVec3f(rec.interpolated_normal);

                    // --- Apply Water Appearance (Color Parity) ---
                    // Overwrite base albedo with calculated water color (Shallow/Deep mix)
                    albedo = toVec3f(wave.water_color);

                    // Foam blending
                    float total_foam = wave.foam;
                    if (total_foam > 0.01f) {
                        Vec3f foam_color(0.92f, 0.96f, 1.0f); // Slightly blue-tinted foam
                        albedo = albedo * (1.0f - total_foam) + foam_color * total_foam;
                        roughness = roughness * (1.0f - total_foam) + 0.8f * total_foam; // Foam is rough
                    }
                }

                if (!is_water && rt_weather_active(world.data.weather)) {
                    Vec3 weatherAlbedo = toVec3(albedo);
                    rt_apply_weather_surface(
                        world.data.weather,
                        rec.point,
                        rec.interpolated_normal,
                        rec.normal,
                        weatherAlbedo,
                        roughness,
                        metallic,
                        clearcoatValue,
                        clearcoatRoughnessValue);
                    rec.interpolated_normal = rt_weather_surface_normal(
                        world.data.weather,
                        rec.point,
                        rec.interpolated_normal,
                        rec.normal);
                    N = toVec3f(rec.interpolated_normal);
                    albedo = toVec3f(weatherAlbedo).clamp(0.0f, 1.0f);

                    rec.surface_override.valid = true;
                    rec.surface_override.albedo = weatherAlbedo;
                    rec.surface_override.roughness = roughness;
                    rec.surface_override.metallic = metallic;
                    rec.surface_override.transmission = transmission;
                    rec.surface_override.clearcoat = clearcoatValue;
                    rec.surface_override.clearcoat_roughness = clearcoatRoughnessValue;
                    rec.surface_override.translucent = translucentValue;
                    rec.surface_override.subsurface = subsurfaceValue;
                    rec.surface_override.subsurface_color = subsurfaceColorValue;
                    rec.surface_override.ior = iorValue;
                }

                // NOTE: Emission is now retrieved polymorphically for all materials after scatter
            }
        }

        if (bounce == 0 && primary_hit && !(*primary_hit)) {
            *primary_hit = true;
            if (primary_albedo) *primary_albedo = toVec3(albedo);
            if (primary_normal) *primary_normal = toVec3(N.normalize());
            if (primary_depth) *primary_depth = rec.t;
            if (primary_material_id) *primary_material_id = static_cast<uint32_t>(rec.materialID);
            if (primary_world_position) *primary_world_position = rec.point;
        }

        // --- Add Emission (Volumetric & Surface) ---
        if (rec.materialPtr && rec.materialPtr->type() == MaterialType::Volumetric) {
            auto vol = static_cast<Volumetric*>(rec.materialPtr);

            // Calculate AABB for Object Space Noise
            Vec3 aabb_min(0), aabb_max(0);
            if (rec.triangle) {
                AABB box;
                if (rec.triangle->bounding_box(0, 0, box)) {
                    aabb_min = box.min;
                    aabb_max = box.max;
                    // Add padding to match GPU logic
                    float padding = 0.001f;
                    aabb_min = aabb_min - Vec3(padding);
                    aabb_max = aabb_max + Vec3(padding);
                }
            }

            Vec3 vol_emit = vol->getVolumetricEmission(toVec3(hit_pos), current_ray.direction, aabb_min, aabb_max);
            emission += toVec3f(vol_emit);
        }

        // NOTE: Emission is added to total_contribution AFTER scatter (matching GPU)

        // --- Throughput clamp (matching GPU) ---
        float max_throughput = throughput.max_component();
        if (max_throughput > UnifiedConstants::MAX_CONTRIBUTION) {
            throughput *= (UnifiedConstants::MAX_CONTRIBUTION / max_throughput);
        }

        // --- Russian Roulette (matching GPU exactly) ---
        // GPU: if (bounce > 2) { p = clamp(p, 0.05f, 0.95f); ... }
        if (bounce > UnifiedConstants::RR_START_BOUNCE) {
            float p = russian_roulette_probability(throughput);
            if (Vec3::random_float() > p) {
                break;
            }
            throughput /= p;

            // Post-RR clamp (matching GPU)
            max_throughput = throughput.max_component();
            if (max_throughput > UnifiedConstants::MAX_CONTRIBUTION) {
                throughput *= (UnifiedConstants::MAX_CONTRIBUTION / max_throughput);
            }
        }

        // --- Emissive Contribution (Vulkan parity: computed BEFORE scatter) ---
        // Vulkan closesthit: payload.radiance = emColor * emStrength (before scatter decision)
        // Vulkan raygen: radiance += throughput * payload.radiance (before scatter check)
        // This ensures emissive-only surfaces still contribute even if scatter fails.
        Vec3 emitted = rec.materialPtr ? rec.materialPtr->getEmission(rec.uv, rec.point) : Vec3(0.0f);
        emission = toVec3f(emitted);

        // --- Scatter ray (GPU Parity: Happens before contribution accumulation) ---
        Vec3 attenuation(1.0f);
        Ray scattered;
        bool is_specular = false;
        bool can_scatter = rec.materialPtr && rec.materialPtr->scatter(current_ray, rec, attenuation, scattered, is_specular);

        if (!can_scatter) {
            // Vulkan parity: emissive-only surfaces still emit light.
            // raygen.rgen accumulates payload.radiance THEN checks payload.scattered.
            color += throughput * emission;
            break;
        }

        // Vulkan/OptiX parity: Emission ve direct lighting zaten evaluate_brdf() içeriyor.
        // Scatter attenuation'ı throughput'a SONRA uygulamak gerekir,
        // yoksa BRDF iki kez çarpılır (double-BRDF bug).
        // OptiX: throughput_for_nee = throughput (pre-scatter)
        // Vulkan: payload.radiance att=1.0 ile toplanır, throughput scatter sonrası güncellenir
        Vec3f throughput_for_nee = throughput;

        Vec3f atten_f = toVec3f(attenuation);
        throughput *= atten_f;

        // --- Volumetric Medium Tracking (for Beer's Law) ---
        // Matches GPU logic: Check if we are entering or exiting a transmissive material
        if (is_specular && transmission > 0.01f) {
            Vec3 N = rec.normal;
            Vec3 D = current_ray.direction.normalize();
            float NdotD = Vec3::dot(D, N);
            bool entering = NdotD < 0.0f;

            if (entering) {
                // Entering medium: Set absorption coefficient
                // sigma_a = (1 - color) * density
                auto pbsdf = dynamic_cast<PrincipledBSDF*>(rec.materialPtr);
                if (pbsdf) {
                    Vec3 subColor = pbsdf->subsurfaceColor; 
                    float subScale = pbsdf->subsurfaceScale;
                    
                    Vec3 absorb_base = Vec3(1.0f) - subColor;
                    absorb_base = Vec3(fmaxf(absorb_base.x, 0.0f), fmaxf(absorb_base.y, 0.0f), fmaxf(absorb_base.z, 0.0f));
                    
                    current_medium_absorb = toVec3f(absorb_base * subScale);
                }
            } else {
                // Exiting medium: Reset to air
                current_medium_absorb = Vec3f(0.0f);
            }
        }

        // --- Direct lighting ---
        Vec3f direct_light(0.0f);
        if (!is_specular && light_count > 0 && transmission < 0.99f) {
            // --- Smart Light Selection (Importance Sampling) ---
            int light_index = -1;
            float pdf_select = 1.0f;
            float r = Vec3::random_float();

            // --- SMART LIGHT SELECTION (Same as GPU) ---
            // Use the unified function to pick light based on distance/intensity importance
            // Pass &pdf_select to get the actual probability used for selection
            light_index = pick_smart_light_unified(unified_lights.data(), light_count, toVec3f(rec.point), r, &pdf_select);

            if (light_index >= 0) {
                const UnifiedLight& light = unified_lights[light_index];
                Vec3f wi; float distance; float light_attenuation;
                if (sample_light_direction(light, hit_pos, Vec3::random_float(), Vec3::random_float(), &wi, &distance, &light_attenuation)) {
                    if (dot(N, wi) > 0.001f) {
                        Vec3 shadow_origin = rec.point + rec.interpolated_normal * UnifiedConstants::SHADOW_BIAS;
                        Ray shadow_ray(shadow_origin, toVec3(wi));
                        bool meshOccluded = bvh->occluded(shadow_ray, UnifiedConstants::SHADOW_BIAS, distance);
                        
                        // --- VOLUMETRIC SHADOWING (VDB & Gas) ---
                        float volShadowTransmittance = 1.0f;
                        if (!meshOccluded && (!scene.vdb_volumes.empty() || !scene.gas_volumes.empty())) {
                            for (const auto& v_ptr : scene.vdb_volumes) {
                                if (!v_ptr || !v_ptr->visible) continue;
                                float t0_v, t1_v;
                                if (v_ptr->intersectTransformedAABB(shadow_ray, 0.001f, distance, t0_v, t1_v)) {
                                    // Use a coarser step for shadow marching to preserve performance
                                    float s_step = (v_ptr->volume_shader ? v_ptr->volume_shader->quality.step_size : 0.25f) * 2.5f;
                                    float t_v = t0_v + 0.001f;
                                    auto& mgr = VDBVolumeManager::getInstance();
                                    int vid = v_ptr->getVDBVolumeID();
                                    Matrix4x4 inv = v_ptr->getInverseTransform();
                                    float ds = (v_ptr->volume_shader ? v_ptr->volume_shader->density.multiplier : 1.0f) * v_ptr->density_scale;
                                    float sigma_t = (v_ptr->volume_shader ? (v_ptr->volume_shader->scattering.coefficient + v_ptr->volume_shader->absorption.coefficient) : 1.1f);
                                    
                                    while (t_v < t1_v) {
                                        Vec3 sp = shadow_ray.at(t_v);
                                        Vec3 local_sp = inv.transform_point(sp);
                                        float dens = mgr.sampleDensityCPU(vid, local_sp.x, local_sp.y, local_sp.z);
                                        if (dens > 0.01f) volShadowTransmittance *= expf(-dens * ds * sigma_t * s_step);
                                        if (volShadowTransmittance < 0.01f) {
                                            volShadowTransmittance = 0.0f;
                                            break;
                                        }
                                        t_v += s_step;
                                    }
                                }
                                if (volShadowTransmittance < 0.01f) break;
                            }
                        }

                        // Hair shadow check (hair can cast shadows on meshes)
                        float hairShadowTransmittance = 1.0f;
                        if (!meshOccluded && volShadowTransmittance > 0.01f && hairSystem.getTotalStrandCount() > 0 && !hairSystem.isBVHDirty()) {
                            Hair::HairHitInfo hsh;
                            Vec3 hairShadowOrigin = shadow_origin;
                            Vec3 hairShadowDir = toVec3(wi);
                            float hairShadowDist = std::min(distance, 100.0f);
                            int hHits = 0;
                            while (hHits < 6 && hairShadowDist > 0.01f && hairShadowTransmittance > 0.05f) {
                                if (hairSystem.intersect(hairShadowOrigin, hairShadowDir, 0.002f, hairShadowDist, hsh)) {
                                    hairShadowTransmittance *= 0.4f; // Each strand blocks 60%
                                    hairShadowOrigin = hsh.position + hairShadowDir * 0.003f;
                                    hairShadowDist -= (hsh.t + 0.003f);
                                    hHits++;
                                } else break;
                            }
                        }
                        
                        if (!meshOccluded && hairShadowTransmittance > 0.01f && volShadowTransmittance > 0.01f) {

                            // Light Radiance (Intensity * Color * Attenuation)
                            Vec3f Li = light.color * light.intensity * light_attenuation * hairShadowTransmittance * volShadowTransmittance;

                            // Evaluate BRDF and PDF for MIS
                            Vec3f f = evaluate_brdf_unified(N, wo, wi, albedo, roughness, metallic, specular, transmission, Vec3::random_float());
                            float pdf_brdf = pdf_brdf_unified(N, wo, wi, roughness);

                            // MIS Weight (Currently 1.0 for analytic lights to match GPU, can be enabled for Area lights)
                            float mis_weight = 1.0f;
                            // Vulkan parity: Delta ışıklar (point/directional) için MIS uygulanmaz
                            bool isDelta = (light.type == (int)UnifiedLightType::Point || 
                                          light.type == (int)UnifiedLightType::Directional);
                            if (!isDelta && light.type == (int)UnifiedLightType::Area) {
                                // Vulkan parity: Combined PDF = pdf_geometry * pdf_select
                                // MIS weight ve estimator ikisi de combined PDF kullanmalı
                                float pdf_geo = compute_light_pdf(light, distance, 1.0f);
                                float pdf_combined = pdf_geo * pdf_select;
                                mis_weight = power_heuristic(pdf_combined, std::clamp(pdf_brdf, 0.01f, 5000.0f));
                                // Area light: tam MIS estimator = f * Li * NdotL * w / pdf_combined
                                direct_light = (f * Li * std::max(0.0f, dot(N, wi)) * mis_weight) 
                                             / std::max(pdf_combined, 1e-4f);
                            } else {
                                // Delta ışıklar: mis_weight = 1.0, sadece pdf_select ile böl
                                direct_light = (f * Li * std::max(0.0f, dot(N, wi)) * mis_weight) 
                                             / std::max(pdf_select, 1e-4f);
                            }

                            direct_light = clamp_contribution(direct_light, UnifiedConstants::MAX_CONTRIBUTION);
                        }

                    }
                }
            }
        }
        
        // --- Accumulate contribution (GPU Match: Total = direct + emission) ---
        Vec3f total_contribution = direct_light + emission;

        // Final contribution clamp
        float total_lum = total_contribution.luminance();
        if (total_lum > UnifiedConstants::MAX_CONTRIBUTION * 2.0f) {
            total_contribution *= (UnifiedConstants::MAX_CONTRIBUTION * 2.0f / total_lum);
        }

        // Vulkan/OptiX parity: PRE-scatter throughput kullan.
        // Direct lighting ve emission zaten kendi BRDF değerlendirmesini içerir.
        // Post-scatter throughput kullanmak BRDF'i iki kez uygular (çift-BRDF hatası).
        color += throughput_for_nee * total_contribution;

        if (is_specular && transmission > 0.01f) {
            if (++transmission_bounce_count > max_transmission_bounces) {
                break;
            }
        } else if (!is_specular) {
            if (++diffuse_bounce_count > max_diffuse_bounces) {
                break;
            }
        }

        current_ray = scattered;
    }
    

    // --- Post-Process: Aerial Perspective (Matches GPU) ---
    if (world.data.mode == WORLD_MODE_NISHITA) {
        float ap_dist = first_hit_t;

        // --- WEIGHTED FOG DISTANCE (Ghost Box Protection) ---
        // Match GPU: lerp between background distance and volume distance based on opacity
        if (first_vol_t > 0.0f) {
            float background_t = (first_hit_t > 0.0f) ? first_hit_t : 10000.0f;
            float weight = 1.0f - vol_trans_accum;
            ap_dist = background_t * (1.0f - weight) + first_vol_t * weight;
        }
        else if (ap_dist <= 0.0f) {
            ap_dist = 10000.0f;
        }
        ap_dist *= (1.0f - first_hit_transmission);

        Vec3 final_c = VolumetricRenderer::applyAerialPerspective(scene, world.data, r.origin, r.direction, ap_dist, toVec3(color), world.getLUT());
        color = toVec3f(final_c);
    }

    else if (world.data.nishita.fog_enabled && world.data.nishita.fog_density > 0.0f) {
        // Fallback for simple height fog on other modes
        // ... (Optional: could add simple fog here too if needed, but Nishita Parity is the main goal)
    }

    if (rt_weather_active(world.data.weather)) {
        float weather_dist = first_hit_t > 0.0f ? first_hit_t : 12000.0f;
        if (first_vol_t > 0.0f) weather_dist = std::min(weather_dist, first_vol_t);
        weather_dist *= (1.0f - first_hit_transmission);
        color = toVec3f(rt_apply_weather_atmosphere(
            world.data.weather,
            toVec3(color),
            r.direction,
            weather_dist));
        const float fps = static_cast<float>(render_settings.animation_fps > 0 ? render_settings.animation_fps : 24);
        const float weather_time = render_settings.realtime_weather_preview
            ? static_cast<float>(SDL_GetTicks()) / 1000.0f
            : static_cast<float>(render_settings.animation_current_frame) / fps;
        color = toVec3f(rt_apply_weather_precipitation_overlay(
            world.data.weather,
            toVec3(color),
            r.direction,
            weather_dist,
            weather_time));
    }

    // --- Final NaN/Inf check and clamp (matching GPU) ---
    // GPU: color.x = isfinite(color.x) ? fminf(fmaxf(color.x, 0.0f), 100.0f) : 0.0f;
    if (!color.is_valid()) {
        color = Vec3f(0.0f);
    }
    color = color.clamp(0.0f, 100.0f);

    return toVec3(color);
}

// ============================================================================
// ACCUMULATIVE RENDERING (CPU)
// ============================================================================

uint64_t Renderer::computeCPUCameraHash(const Camera& cam) const {
    // FNV-1a hash of camera parameters
    uint64_t hash = 14695981039346656037ULL;
    auto hashFloat = [&hash](float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        hash ^= bits;
        hash *= 1099511628211ULL;
        };

    hashFloat(cam.lookfrom.x);
    hashFloat(cam.lookfrom.y);
    hashFloat(cam.lookfrom.z);
    hashFloat(cam.lookat.x);
    hashFloat(cam.lookat.y);
    hashFloat(cam.lookat.z);
    hashFloat(cam.vup.x);
    hashFloat(cam.vup.y);
    hashFloat(cam.vup.z);
    hashFloat(cam.vfov);

    return hash;
}

void Renderer::resetCPUAccumulation() {
    cpu_accumulated_samples = 0;
    cpu_accumulation_valid = false;
    cpu_last_camera_hash = 0;  // Reset camera hash for animation frames
    cpu_pixel_list_valid = false;  // Force pixel list rebuild + shuffle on next pass
    invalidateCPUDenoisedBuffer();
    if (!cpu_accumulation_buffer.empty()) {
        std::fill(cpu_accumulation_buffer.begin(), cpu_accumulation_buffer.end(), Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
    }
    if (!cpu_albedo_accumulation_buffer.empty()) {
        std::fill(cpu_albedo_accumulation_buffer.begin(), cpu_albedo_accumulation_buffer.end(), Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
    }
    if (!cpu_normal_accumulation_buffer.empty()) {
        std::fill(cpu_normal_accumulation_buffer.begin(), cpu_normal_accumulation_buffer.end(), Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
    }
    if (!cpu_world_position_accumulation_buffer.empty()) {
        std::fill(cpu_world_position_accumulation_buffer.begin(), cpu_world_position_accumulation_buffer.end(), Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
    }
    if (!cpu_depth_accumulation_buffer.empty()) {
        std::fill(cpu_depth_accumulation_buffer.begin(), cpu_depth_accumulation_buffer.end(), 0.0f);
    }
    if (!cpu_material_id_buffer.empty()) {
        std::fill(cpu_material_id_buffer.begin(), cpu_material_id_buffer.end(), 0xFFFFFFFFu);
    }
    // Reset variance buffer for adaptive sampling
    if (!cpu_variance_buffer.empty()) {
        std::fill(cpu_variance_buffer.begin(), cpu_variance_buffer.end(), 0.0f);
    }
}

// ============================================================================
// Hair System GPU Integration
// ============================================================================



void Renderer::uploadHairToGPU() {
    if (!m_backend) return;

    // ── Raster viewport hair line overlay ────────────────────────────────────
    // Always runs regardless of render backend (Vulkan RT or OptiX).
    // g_viewport_backend is the VulkanViewportBackend used for Solid/Matcap display.
    {
        extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
        auto* vpb = dynamic_cast<Backend::VulkanBackendAdapter*>(g_viewport_backend.get());
        if (vpb) {
            std::vector<float> lineVerts;
            const auto groomNames = hairSystem.getGroomNames();
            for (const auto& gName : groomNames) {
                const Hair::HairGroom* groom = hairSystem.getGroom(gName);
                if (!groom || !groom->isVisible || groom->guides.empty()) continue;
                const Matrix4x4& xf = groom->transform;
                // Mirror the same strand selection logic as the ray-tracing upload
                const auto& viewportStrands = (!groom->interpolated.empty() && !hideInterpolatedHair)
                    ? groom->interpolated : groom->guides;
                for (const auto& strand : viewportStrands) {
                    const size_t n = strand.points.size();
                    if (n < 2) continue;
                    const float invN = 1.0f / float(n - 1);
                    for (size_t i = 0; i + 1 < n; ++i) {
                        Vec3 p0 = xf.transform_point(strand.points[i].position);
                        Vec3 p1 = xf.transform_point(strand.points[i + 1].position);
                        lineVerts.insert(lineVerts.end(),
                            { p0.x, p0.y, p0.z, float(i)     * invN,
                              p1.x, p1.y, p1.z, float(i + 1) * invN });
                    }
                }
            }
            vpb->uploadHairViewportLines(lineVerts, uint32_t(lineVerts.size() / 4));
        }
    }

    // ── Vulkan path ──────────────────────────────────────────────────────────
    auto* vulkanBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(m_backend);
    if (vulkanBackend) {
        const bool noHair = (hairSystem.getTotalStrandCount() == 0);
        auto groomNames = hairSystem.getGroomNames();

        bool hasVisibleHair = false;
        for (const auto& name : groomNames) {
            const Hair::HairGroom* groom = hairSystem.getGroom(name);
            if (!groom || !groom->isVisible) continue;

            const auto& srcStrands = (!groom->interpolated.empty() && !hideInterpolatedHair)
                ? groom->interpolated : groom->guides;
            if (!srcStrands.empty()) {
                hasVisibleHair = true;
                break;
            }
        }

        vulkanBackend->clearHairGeometry(noHair || !hasVisibleHair);
        if (noHair || !hasVisibleHair) {
            SCENE_LOG_INFO("[Hair GPU/Vulkan] Uploaded 0 grooms, 0 strands (combined)");
            vulkanBackend->resetAccumulation();
            return;
        }

        // [FIX] Combine ALL visible grooms into ONE strand list so that only a
        // single AABB BLAS and a single segment SSBO are created.  The old
        // per-groom uploadHairStrands calls overwrote the segment SSBO on every
        // iteration, leaving only the last groom's segments in the buffer while
        // the earlier grooms' BLASes still referenced wrong segment indices.
        // Combining avoids the overwrite and eliminates extra BLAS objects.
        //
        // materialID is set to the groom index so each strand looks up the
        // correct entry in the per-groom hair material buffer.
        std::vector<Backend::HairStrandData> combinedStrands;
        std::vector<Backend::IBackend::HairMaterialData> allMaterials;

        uint32_t groomMaterialIndex = 0;
        for (const auto& name : groomNames) {
            const Hair::HairGroom* groom = hairSystem.getGroom(name);
            if (!groom || !groom->isVisible) continue;

            // Collect strands (prefer interpolated; fall back to guides)
            const auto& srcStrands = (!groom->interpolated.empty() && !hideInterpolatedHair)
                ? groom->interpolated : groom->guides;
            if (srcStrands.empty()) continue;
            // Calculate scale and extract parameters
            const Matrix4x4& transform = groom->transform;
            float sx = Vec3(transform.m[0][0], transform.m[1][0], transform.m[2][0]).length();
            float sy = Vec3(transform.m[0][1], transform.m[1][1], transform.m[2][1]).length();
            float sz = Vec3(transform.m[0][2], transform.m[1][2], transform.m[2][2]).length();
            float avgScale = (sx + sy + sz) / 3.0f;
            bool useBSpline = groom->params.useBSpline;

            for (const auto& s : srcStrands) {
                if (s.points.size() < 4) continue;
                Backend::HairStrandData sd;
                // Override materialID to be the groom's index into allMaterials
                // so the hair intersection shader fetches the correct material.
                sd.materialID = groomMaterialIndex;
                sd.rootUV     = s.rootUV;

                size_t estimatedPoints = s.points.size() + (useBSpline ? 4 : 0);
                sd.points.reserve(estimatedPoints);
                sd.radii.reserve(estimatedPoints);

                auto addPoint = [&](const Hair::HairPoint& hp) {
                    Vec3 p = transform.transform_point(hp.position);
                    float r = std::max((hp.radius > 0.f ? hp.radius : 0.002f) * avgScale, 1e-5f);
                    sd.points.push_back(p);
                    sd.radii.push_back(r);
                };

                if (useBSpline) {
                    addPoint(s.points.front());
                    addPoint(s.points.front());
                }

                for (const auto& hp : s.points) {
                    addPoint(hp);
                }

                if (useBSpline) {
                    addPoint(s.points.back());
                    addPoint(s.points.back());
                }
                
                combinedStrands.push_back(std::move(sd));
            }

            // Collect material for this groom
            const auto& mp = groom->material;
            Backend::IBackend::HairMaterialData mat;
            mat.color           = mp.color;
            mat.melanin         = mp.melanin;
            mat.melaninRedness  = mp.melaninRedness;
            mat.roughness       = mp.roughness;
            mat.radialRoughness = mp.radialRoughness;
            mat.ior             = mp.ior;
            mat.cuticleAngle    = mp.cuticleAngle * 3.14159f / 180.0f;
            mat.colorMode       = (int)mp.colorMode;
            mat.absorption      = mp.absorptionCoefficient;
            mat.specularTint    = mp.specularTint;
            mat.diffuseSoftness = mp.diffuseSoftness;
            mat.tint            = mp.tint;
            mat.tintColor       = mp.tintColor;
            mat.coat            = mp.coat;
            mat.coatTint        = mp.coatTint;
            mat.emission        = mp.emission;
            mat.emissionStrength = mp.emissionStrength;
            mat.enableRootTipGradient = mp.enableRootTipGradient;
            mat.tipColor        = mp.tipColor;
            mat.rootTipBalance  = mp.rootTipBalance;
            mat.randomHue       = mp.randomHue;
            mat.randomValue     = mp.randomValue;

            // 1. Albedo Texture
            if (mp.customAlbedoTexture && mp.customAlbedoTexture->is_loaded()) {
                mat.albedoTexture = reinterpret_cast<int64_t>(mp.customAlbedoTexture.get());
            }
            // 2. Roughness Texture
            if (mp.customRoughnessTexture && mp.customRoughnessTexture->is_loaded()) {
                mat.roughnessTexture = reinterpret_cast<int64_t>(mp.customRoughnessTexture.get());
            }
            // 3. Scalp Mesh Texture (Automatic detection for ROOT_UV_MAP mode)
            if (mp.colorMode == Hair::HairMaterialParams::ColorMode::ROOT_UV_MAP && mat.albedoTexture == -1) {
                auto& matMgr = MaterialManager::getInstance();
                const auto& all_mats = matMgr.getAllMaterials();
                int scalpMatID = groom->params.defaultMaterialID;
                if (scalpMatID >= 0 && (size_t)scalpMatID < all_mats.size()) {
                    const auto& scalpMat = all_mats[scalpMatID];
                    if (scalpMat) {
                        if (scalpMat->albedoProperty.texture && scalpMat->albedoProperty.texture->is_loaded()) {
                            mat.scalpAlbedoTexture = reinterpret_cast<int64_t>(scalpMat->albedoProperty.texture.get());
                        }
                        mat.scalpBaseColor = scalpMat->albedoProperty.color;
                    }
                }
            }

            allMaterials.push_back(mat);
            ++groomMaterialIndex;
        }

        if (!combinedStrands.empty()) {
            // Single upload call → single BLAS, single segment SSBO
            vulkanBackend->uploadHairStrands(combinedStrands, "combined_hair");
        }
        if (!allMaterials.empty()) {
            vulkanBackend->uploadHairMaterials(allMaterials);
        }

        SCENE_LOG_INFO("[Hair GPU/Vulkan] Uploaded " + std::to_string(groomMaterialIndex)
            + " grooms, " + std::to_string(combinedStrands.size()) + " strands (combined)");
        vulkanBackend->resetAccumulation();
        return;
    }

    // ── OptiX / CUDA path ────────────────────────────────────────────────────
    if (!g_hasCUDA) return;

    Backend::OptixBackend* optixBackend = dynamic_cast<Backend::OptixBackend*>(m_backend);
    OptixWrapper* optix_gpu = optixBackend ? optixBackend->getOptixWrapper() : nullptr;

    if (hairSystem.getTotalStrandCount() == 0) {
        if (optix_gpu) optix_gpu->clearHairGeometry();
        return;
    }
    
    // Clear previous hair states in OptiX
    if (optix_gpu) optix_gpu->clearHairGeometry();
    
    auto groomNames = hairSystem.getGroomNames();
    bool first = true;
    
    for (const auto& name : groomNames) {
        std::vector<float> hairVertices4;
        std::vector<unsigned int> hairIndices;
        std::vector<uint32_t> hairStrandIDs;
        std::vector<float> hairTangents3;
        std::vector<float> hairRootUVs2;
        std::vector<float> hairStrandVs;
        size_t vertexCount = 0;
        size_t segmentCount = 0;
        Hair::HairMaterialParams matParams;
        int hairMatID = 0;
        int meshMatID = -1;
        
        // Call with all 13 arguments explicitly
        bool isSpline = hairSystem.getOptiXCurveDataByGroom(
            name,                   // 1
            hairVertices4,          // 2
            hairIndices,            // 3
            hairStrandIDs,          // 4
            hairTangents3,          // 5
            hairRootUVs2,           // 6
            hairStrandVs,           // 7
            vertexCount,            // 8
            segmentCount,           // 9
            matParams,              // 10
            hairMatID,              // 11
            meshMatID,              // 12
            !hideInterpolatedHair   // 13
        );
        
        if (segmentCount > 0) {
            // Convert to float4/float3 arrays
            std::vector<float4> vertices4(vertexCount);
            for (size_t i = 0; i < vertexCount; ++i) {
                vertices4[i] = make_float4(hairVertices4[i * 4 + 0], hairVertices4[i * 4 + 1], hairVertices4[i * 4 + 2], hairVertices4[i * 4 + 3]);
            }
            
            std::vector<float3> tangents3(segmentCount);
            for (size_t i = 0; i < segmentCount; ++i) {
                tangents3[i] = make_float3(hairTangents3[i * 3 + 0], hairTangents3[i * 3 + 1], hairTangents3[i * 3 + 2]);
            }
            
            std::vector<float2> rootUVs2(segmentCount);
            for (size_t i = 0; i < segmentCount; ++i) {
                rootUVs2[i] = make_float2(hairRootUVs2[i * 2 + 0], hairRootUVs2[i * 2 + 1]);
            }
            
            // Generate GPU Hair Material
            GpuHairMaterial hairMat = Hair::HairBSDF::convertToGpu(matParams);

            // Add this groom to OptiX
            if (optix_gpu) optix_gpu->buildHairGeometry(
                vertices4.data(),
                hairIndices.data(),
                hairStrandIDs.data(),
                tangents3.data(),
                rootUVs2.data(),
                hairStrandVs.data(),
                vertexCount,
                segmentCount,
                hairMat,
                name,
                hairMatID,
                meshMatID,
                isSpline,
                false
            );
        }
    }
    
   // SCENE_LOG_INFO("[Hair GPU] Integrated " + std::to_string(groomNames.size()) + " hair grooms into TLAS");
    m_backend->resetAccumulation();
}

void Renderer::updateHairGeometryOnGPU(bool forceRebuild) {
    if (!m_backend || !g_hasCUDA) return;
    
    // Without HairGPUManager, we always perform a full upload if anything changed.
    // In the future, we can implement a CPU-based vertex refit here.
    uploadHairToGPU();
}

void Renderer::setHairMaterial(const Hair::HairMaterialParams& mat) {
    this->hairMaterial = mat;

    if (m_backend) {
        // Collect ALL hair materials from the system to maintain correct indices
        // in the Vulkan SSBO. If we only upload the changed material, it always
        // goes to slot 0, breaking multi-groom setups (e.g. hair and beard).
        std::vector<Backend::IBackend::HairMaterialData> allMaterials;
        auto groomNames = hairSystem.getGroomNames();

        auto convertParams = [&](const Hair::HairMaterialParams& p, const std::string& gName) {
            Backend::IBackend::HairMaterialData h;
            h.color = p.color;
            h.absorption = p.absorptionCoefficient;
            h.melanin = p.melanin;
            h.melaninRedness = p.melaninRedness;
            h.roughness = p.roughness;
            h.radialRoughness = p.radialRoughness;
            h.ior = p.ior;
            h.coat = p.coat;
            h.cuticleAngle = p.cuticleAngle * 3.14159f / 180.0f;
            h.randomHue = p.randomHue;
            h.randomValue = p.randomValue;
            h.colorMode = static_cast<int>(p.colorMode);
            h.tint = p.tint;
            h.tintColor = p.tintColor;
            h.specularTint = p.specularTint;
            h.diffuseSoftness = p.diffuseSoftness;
            h.coatTint = p.coatTint;
            h.emission = p.emission;
            h.emissionStrength = p.emissionStrength;
            h.enableRootTipGradient = p.enableRootTipGradient;
            h.tipColor = p.tipColor;
            h.rootTipBalance = p.rootTipBalance;

            // 1. Albedo Texture
            if (p.customAlbedoTexture && p.customAlbedoTexture->is_loaded()) {
                h.albedoTexture = reinterpret_cast<int64_t>(p.customAlbedoTexture.get());
            }
            // 2. Roughness Texture
            if (p.customRoughnessTexture && p.customRoughnessTexture->is_loaded()) {
                h.roughnessTexture = reinterpret_cast<int64_t>(p.customRoughnessTexture.get());
            }
            // 3. Scalp Mesh Texture (Automatic detection for ROOT_UV_MAP mode)
            if (p.colorMode == Hair::HairMaterialParams::ColorMode::ROOT_UV_MAP && h.albedoTexture == -1) {
                auto* groom = hairSystem.getGroom(gName);
                if (groom) {
                    auto& matMgr = MaterialManager::getInstance();
                    const auto& all_mats = matMgr.getAllMaterials();
                    int scalpMatID = groom->params.defaultMaterialID;
                    if (scalpMatID >= 0 && (size_t)scalpMatID < all_mats.size()) {
                        const auto& scalpMat = all_mats[scalpMatID];
                        if (scalpMat) {
                            if (scalpMat->albedoProperty.texture && scalpMat->albedoProperty.texture->is_loaded()) {
                                h.scalpAlbedoTexture = reinterpret_cast<int64_t>(scalpMat->albedoProperty.texture.get());
                            }
                            h.scalpBaseColor = scalpMat->albedoProperty.color;
                        }
                    }
                }
            }
            return h;
        };

        if (groomNames.empty()) {
            allMaterials.push_back(convertParams(mat, ""));
        } else {
            for (const auto& name : groomNames) {
                const Hair::HairGroom* groom = hairSystem.getGroom(name);
                if (groom) {
                    allMaterials.push_back(convertParams(groom->material, name));
                }
            }
        }

        m_backend->uploadHairMaterials(allMaterials);
        
        // For OptiX, we also need to trigger per-groom updates
        Backend::OptixBackend* optixBackend = dynamic_cast<Backend::OptixBackend*>(m_backend);
        if (optixBackend && optixBackend->getOptixWrapper()) {
            optixBackend->getOptixWrapper()->updateHairMaterialsOnly(hairSystem);
        }
        
        m_backend->resetAccumulation();
    }
    
    resetCPUAccumulation();
}

bool Renderer::isCPUAccumulationComplete() const {
    extern RenderSettings render_settings;
    int target_max_samples = render_settings.max_samples > 0 ? render_settings.max_samples : 100;

    // Check override for final render / animation
    if (render_settings.is_final_render_mode) {
        target_max_samples = render_settings.final_render_samples > 0 ? render_settings.final_render_samples : 128;
    }

    return cpu_accumulated_samples >= target_max_samples;
}

void Renderer::render_progressive_pass(SDL_Surface* surface, SDL_Window* window, SceneData& scene, int samples_this_pass, int override_target_samples) {
    // [SAFETY CHECK] Prevent rendering if stopped or loading a new project
    // This prevents access violations when camera/scene data is being destroyed
    extern std::atomic<bool> rendering_stopped_cpu;
    extern std::atomic<bool> g_scene_loading_in_progress;

    if (rendering_stopped_cpu.load()) {
        // SCENE_LOG_WARN("CPU Render aborted: rendering_stopped_cpu is true");
        return;
    }
    if (g_scene_loading_in_progress.load()) {
        SCENE_LOG_WARN("CPU Render aborted: scene loading in progress");
        return;
    }
    if (!scene.camera) {
        SCENE_LOG_WARN("CPU Render aborted: no camera");
        return;
    }

    extern RenderSettings render_settings;


    // Ensure accumulation buffer is allocated
    const size_t pixel_count = image_width * image_height;
    if (cpu_accumulation_buffer.size() != pixel_count) {
        cpu_accumulation_buffer.resize(pixel_count, Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
    }
    if (cpu_albedo_accumulation_buffer.size() != pixel_count) {
        cpu_albedo_accumulation_buffer.resize(pixel_count, Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
    }
    if (cpu_normal_accumulation_buffer.size() != pixel_count) {
        cpu_normal_accumulation_buffer.resize(pixel_count, Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
    }
    if (cpu_world_position_accumulation_buffer.size() != pixel_count) {
        cpu_world_position_accumulation_buffer.resize(pixel_count, Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
    }
    if (cpu_depth_accumulation_buffer.size() != pixel_count) {
        cpu_depth_accumulation_buffer.resize(pixel_count, 0.0f);
    }
    if (cpu_material_id_buffer.size() != pixel_count) {
        cpu_material_id_buffer.resize(pixel_count, 0xFFFFFFFFu);
    }
    cpu_accumulation_valid = true;

    // Ensure variance buffer is allocated for adaptive sampling
    if (cpu_variance_buffer.size() != pixel_count) {
        cpu_variance_buffer.resize(pixel_count, 0.0f);
    }

    // Camera change detection
    if (scene.camera) {
        uint64_t current_hash = computeCPUCameraHash(*scene.camera);
        bool is_first = (cpu_last_camera_hash == 0);

        if (current_hash != cpu_last_camera_hash) {
            // Camera changed - reset accumulation and variance
            std::fill(cpu_accumulation_buffer.begin(), cpu_accumulation_buffer.end(), Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
            std::fill(cpu_albedo_accumulation_buffer.begin(), cpu_albedo_accumulation_buffer.end(), Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
            std::fill(cpu_normal_accumulation_buffer.begin(), cpu_normal_accumulation_buffer.end(), Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
            std::fill(cpu_world_position_accumulation_buffer.begin(), cpu_world_position_accumulation_buffer.end(), Vec4{ 0.0f, 0.0f, 0.0f, 0.0f });
            std::fill(cpu_depth_accumulation_buffer.begin(), cpu_depth_accumulation_buffer.end(), 0.0f);
            std::fill(cpu_material_id_buffer.begin(), cpu_material_id_buffer.end(), 0xFFFFFFFFu);
            std::fill(cpu_variance_buffer.begin(), cpu_variance_buffer.end(), 0.0f);
            cpu_accumulated_samples = 0;
            cpu_pixel_list_valid = false;  // Force pixel list rebuild + shuffle
            invalidateCPUDenoisedBuffer();

            if (!is_first) {
                // SCENE_LOG_INFO("CPU: Camera changed - resetting accumulation");
            }

            cpu_last_camera_hash = current_hash;
        }
    }

    // Auto-rebuild hair BVH for live updates
    if (hairSystem.isBVHDirty()) {
        hairSystem.buildBVH(!hideInterpolatedHair);
    }

    // Check if already complete
    // Priority: 1. Override (Animation), 2. Final Render Mode (F12), 3. Viewport Settings
    int target_max_samples = 100;
    if (override_target_samples > 0) {
        target_max_samples = override_target_samples;
    }
    else if (render_settings.is_final_render_mode) {
        target_max_samples = render_settings.final_render_samples > 0 ? render_settings.final_render_samples : 128;
    }
    else {
        target_max_samples = render_settings.max_samples > 0 ? render_settings.max_samples : 100;
    }
    if (cpu_accumulated_samples >= target_max_samples) {
        // SCENE_LOG_INFO("CPU Render returned early: cpu_accumulated_samples(" + std::to_string(cpu_accumulated_samples) + ") >= target_max_samples(" + std::to_string(target_max_samples) + ")");
        return; // Already done
    }

    // Multi-threaded rendering with accumulation
    unsigned int num_threads = std::thread::hardware_concurrency();

    // UI/OS Responsiveness Optimization:
    // Leave 2 threads free if we have plenty (e.g., > 4 cores)
    // Leave 1 thread free if we have few (e.g., <= 4 cores)
    if (num_threads > 4) num_threads -= 2;
    else if (num_threads > 1) num_threads -= 1;

    // Safety cap
    if (num_threads > 32) num_threads = 32;

    std::vector<std::thread> threads;

    // ===========================================================================
    // OPTIMIZATION: Cache pixel list - only rebuild + shuffle when necessary
    // This avoids O(n) allocation + O(n) shuffle on EVERY pass
    // ===========================================================================
    if (!cpu_pixel_list_valid || cpu_cached_pixel_list.size() != pixel_count) {
        // Rebuild pixel list (resolution changed or first time)
        cpu_cached_pixel_list.clear();
        cpu_cached_pixel_list.reserve(pixel_count);
        for (int j = 0; j < image_height; ++j) {
            for (int i = 0; i < image_width; ++i) {
                cpu_cached_pixel_list.emplace_back(i, j);
            }
        }

        // Shuffle ONCE for visual distribution (random device seeded)
        std::shuffle(cpu_cached_pixel_list.begin(), cpu_cached_pixel_list.end(),
            std::mt19937(std::random_device{}()));

        cpu_pixel_list_valid = true;
    }

    // Use cached pixel list (no per-pass allocation or shuffle!)
    const auto& pixel_list = cpu_cached_pixel_list;

    int sparse_divisor = 1;
    const bool any_cpu_denoiser_enabled =
        render_settings.use_denoiser ||
        render_settings.timeline_use_denoiser ||
        render_settings.render_use_denoiser;
    if (!any_cpu_denoiser_enabled) {
        // Sparse progressive: First few samples render fewer pixels for faster preview
        // Sample 1: 1/16 pixels, Sample 2: 1/8, Sample 3: 1/4, Sample 5+: all
        if (cpu_accumulated_samples == 0) sparse_divisor = 16;  // First pass: 1/16 pixels
        else if (cpu_accumulated_samples == 1) sparse_divisor = 8;
        else if (cpu_accumulated_samples == 2) sparse_divisor = 4;
        else if (cpu_accumulated_samples == 3) sparse_divisor = 2;
    }

    size_t pixels_to_render = pixel_list.size() / sparse_divisor;
    if (pixels_to_render < 1000) pixels_to_render = pixel_list.size();  // Safety minimum

    std::atomic<int> next_pixel_index{ 0 };
    std::atomic<bool> should_stop{ false };

    // Worker function for progressive accumulation
    auto progressive_worker = [&](int thread_id) {
        std::mt19937 rng(std::random_device{}() + thread_id + cpu_accumulated_samples * 1337);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        while (!should_stop.load(std::memory_order_relaxed)) {
            // Check global stop flag or FORCE STOP from UI
            if (rendering_stopped_cpu.load(std::memory_order_relaxed) || force_stop_rendering.load(std::memory_order_relaxed)) {
                should_stop.store(true, std::memory_order_relaxed);
                break;
            }

            int idx = next_pixel_index.fetch_add(1, std::memory_order_relaxed);
            if (idx == 0) {
                 // Print first pixel trace to confirm thread starts
                 // SCENE_LOG_INFO("Worker thread started processing pixels.");
            }
            if (idx >= static_cast<int>(pixels_to_render)) break;

            int i = pixel_list[idx].first;
            int j = pixel_list[idx].second;
            int pixel_index = j * image_width + i;

            // ===================================================================
            // ADAPTIVE SAMPLING - Early exit for converged pixels
            // Welford running M2 in variance_buffer; convergence by relative standard error
            // rSE = sqrt(variance / n) / mean_lum  with variance = M2 / (n - 1)
            // Requires a warmup so the estimator isn't fooled by 1-2 correlated samples
            // (the single-light NEE failure mode that produced false-converged black dots).
            // ===================================================================
            if (render_settings.use_adaptive_sampling) {
                Vec4& accum_check = cpu_accumulation_buffer[pixel_index];
                float prev_samples_check = accum_check.w;
                float M2_check = cpu_variance_buffer[pixel_index];

                float mean_lum_check = 0.2126f * accum_check.x + 0.7152f * accum_check.y + 0.0722f * accum_check.z;

                constexpr int ADAPTIVE_WARMUP = 4;
                int effective_min = std::max((int)render_settings.min_samples, ADAPTIVE_WARMUP);

                float effective_threshold = render_settings.variance_threshold;
                if (render_settings.use_denoiser) {
                    effective_threshold *= 2.0f;
                }

                float rel_stderr = 1.0f;
                if (prev_samples_check >= 2.0f && M2_check > 0.0f && mean_lum_check > 1e-5f) {
                    float variance = M2_check / (prev_samples_check - 1.0f);
                    rel_stderr = std::sqrt(variance / prev_samples_check) / mean_lum_check;
                }

                if (prev_samples_check >= float(effective_min) &&
                    M2_check > 0.0f &&
                    rel_stderr < effective_threshold) {
                    // Pixel has converged - skip ray tracing but still write existing color to surface
                    // This ensures the display stays updated even when pixels are skipped
                    Vec3 cached_color(accum_check.x, accum_check.y, accum_check.z);

                    // Tone mapping for display (same as below)
                    auto toSRGB_fast = [](float c) {
                        return (c <= 0.0031308f) ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
                        };

                    float exp_factor = scene.camera ? (scene.camera->auto_exposure ?
                        std::pow(2.0f, scene.camera->ev_compensation) : 1.0f) : 1.0f;
                    Vec3 exposed = cached_color * exp_factor;
                    exposed.x = exposed.x / (exposed.x + 1.0f);
                    exposed.y = exposed.y / (exposed.y + 1.0f);
                    exposed.z = exposed.z / (exposed.z + 1.0f);

                    int r = static_cast<int>(255 * std::clamp(toSRGB_fast(std::max(0.0f, exposed.x)), 0.0f, 1.0f));
                    int g = static_cast<int>(255 * std::clamp(toSRGB_fast(std::max(0.0f, exposed.y)), 0.0f, 1.0f));
                    int b = static_cast<int>(255 * std::clamp(toSRGB_fast(std::max(0.0f, exposed.z)), 0.0f, 1.0f));

                    Uint32* pixels_ptr = static_cast<Uint32*>(surface->pixels);
                    int screen_idx = (surface->h - 1 - j) * (surface->pitch / 4) + i;
                    pixels_ptr[screen_idx] = SDL_MapRGB(surface->format, r, g, b);

                    continue;  // Skip to next pixel - FAST PATH!
                }
            }

            Vec3 color_sum(0.0f);
            Vec3 albedo_sum(0.0f);
            Vec3 normal_sum(0.0f);
            Vec3 world_position_sum(0.0f);
            float depth_sum = 0.0f;
            uint32_t first_material_id = 0xFFFFFFFFu;
            int primary_hit_count = 0;
            float batch_lum_sum = 0.0f;
            float batch_lum_sq_sum = 0.0f;

            // Render samples for this pass in 8-wide packets
            for (int s = 0; s < samples_this_pass; ++s) {
                float u = (float(i) + dist(rng)) / (image_width - 1);
                float v = (float(j) + dist(rng)) / (image_height - 1);

                Ray r = scene.camera->get_ray(u, v);

                Vec3 primary_albedo(0.0f);
                Vec3 primary_normal(0.0f);
                Vec3 primary_world_position(0.0f);
                float primary_depth = 0.0f;
                uint32_t primary_material_id = 0xFFFFFFFFu;
                bool primary_hit = false;

                Vec3 sample_color = ray_color(
                    r, scene.bvh.get(), scene.lights, scene.background_color, max_depth, 0, scene,
                    &primary_albedo, &primary_normal, &primary_hit, &primary_depth, &primary_material_id,
                    &primary_world_position);
                color_sum = color_sum + sample_color;

                if (render_settings.use_adaptive_sampling) {
                    float sl = 0.2126f * sample_color.x + 0.7152f * sample_color.y + 0.0722f * sample_color.z;
                    batch_lum_sum += sl;
                    batch_lum_sq_sum += sl * sl;
                }

                if (primary_hit) {
                    albedo_sum = albedo_sum + primary_albedo;
                    normal_sum = normal_sum + primary_normal;
                    world_position_sum = world_position_sum + primary_world_position;
                    depth_sum += primary_depth;
                    if (first_material_id == 0xFFFFFFFFu) {
                        first_material_id = primary_material_id;
                    }
                    primary_hit_count++;
                }
            }

            Vec3 new_color = color_sum / float(samples_this_pass);
            Vec3 new_albedo(0.0f);
            Vec3 new_normal(0.0f);
            Vec3 new_world_position(0.0f);
            float new_depth = 0.0f;
            if (primary_hit_count > 0) {
                new_albedo = albedo_sum / float(primary_hit_count);
                new_normal = (normal_sum / float(primary_hit_count)).normalize();
                new_world_position = world_position_sum / float(primary_hit_count);
                new_depth = depth_sum / float(primary_hit_count);
            }

            // Accumulate with previous samples
            Vec4& accum = cpu_accumulation_buffer[pixel_index];
            float prev_samples = accum.w;
            float prev_mean_lum = 0.2126f * accum.x + 0.7152f * accum.y + 0.0722f * accum.z;
            Vec3 blended_color = new_color;
            float new_total_samples = float(samples_this_pass);

            if (prev_samples > 0.0f) {
                // Progressive blend
                new_total_samples = prev_samples + samples_this_pass;
                Vec3 prev_color(accum.x, accum.y, accum.z);
                blended_color = (prev_color * prev_samples + new_color * samples_this_pass) / new_total_samples;

                accum.x = blended_color.x;
                accum.y = blended_color.y;
                accum.z = blended_color.z;
                accum.w = new_total_samples;
            }
            else {
                // First sample
                accum.x = new_color.x;
                accum.y = new_color.y;
                accum.z = new_color.z;
                accum.w = float(samples_this_pass);
            }

            Vec4& accum_albedo = cpu_albedo_accumulation_buffer[pixel_index];
            Vec4& accum_normal = cpu_normal_accumulation_buffer[pixel_index];
            Vec4& accum_world_position = cpu_world_position_accumulation_buffer[pixel_index];
            float& accum_depth = cpu_depth_accumulation_buffer[pixel_index];
            uint32_t& accum_material_id = cpu_material_id_buffer[pixel_index];
            if (primary_hit_count > 0) {
                if (prev_samples > 0.0f) {
                    Vec3 prev_albedo(accum_albedo.x, accum_albedo.y, accum_albedo.z);
                    Vec3 prev_normal(accum_normal.x, accum_normal.y, accum_normal.z);
                    Vec3 prev_world_position(accum_world_position.x, accum_world_position.y, accum_world_position.z);
                    Vec3 blended_albedo = (prev_albedo * prev_samples + new_albedo * samples_this_pass) / new_total_samples;
                    Vec3 blended_normal = (prev_normal * prev_samples + new_normal * samples_this_pass) / new_total_samples;
                    Vec3 blended_world_position = (prev_world_position * prev_samples + new_world_position * samples_this_pass) / new_total_samples;
                    blended_normal = blended_normal.normalize();
                    accum_depth = (accum_depth * prev_samples + new_depth * samples_this_pass) / new_total_samples;
                    accum_albedo.x = blended_albedo.x;
                    accum_albedo.y = blended_albedo.y;
                    accum_albedo.z = blended_albedo.z;
                    accum_albedo.w = new_total_samples;
                    accum_normal.x = blended_normal.x;
                    accum_normal.y = blended_normal.y;
                    accum_normal.z = blended_normal.z;
                    accum_normal.w = new_total_samples;
                    accum_world_position.x = blended_world_position.x;
                    accum_world_position.y = blended_world_position.y;
                    accum_world_position.z = blended_world_position.z;
                    accum_world_position.w = new_total_samples;
                    if (accum_material_id == 0xFFFFFFFFu) {
                        accum_material_id = first_material_id;
                    }
                } else {
                    accum_albedo.x = new_albedo.x;
                    accum_albedo.y = new_albedo.y;
                    accum_albedo.z = new_albedo.z;
                    accum_albedo.w = float(samples_this_pass);
                    accum_normal.x = new_normal.x;
                    accum_normal.y = new_normal.y;
                    accum_normal.z = new_normal.z;
                    accum_normal.w = float(samples_this_pass);
                    accum_world_position.x = new_world_position.x;
                    accum_world_position.y = new_world_position.y;
                    accum_world_position.z = new_world_position.z;
                    accum_world_position.w = float(samples_this_pass);
                    accum_depth = new_depth;
                    accum_material_id = first_material_id;
                }
            } else if (prev_samples <= 0.0f) {
                accum_albedo = Vec4{ 0.0f, 0.0f, 0.0f, float(samples_this_pass) };
                accum_normal = Vec4{ 0.0f, 0.0f, 0.0f, float(samples_this_pass) };
                accum_world_position = Vec4{ 0.0f, 0.0f, 0.0f, float(samples_this_pass) };
                accum_depth = 0.0f;
                accum_material_id = 0xFFFFFFFFu;
            }

            // ===================================================================
            // ADAPTIVE SAMPLING - Welford M2 update (Chan's parallel merge)
            // variance_buffer stores M2 = sum of squared deviations from running mean.
            // var = M2 / (n - 1), stderr = sqrt(var / n).
            // ===================================================================
            if (render_settings.use_adaptive_sampling) {
                float k = float(samples_this_pass);
                float batch_mean = batch_lum_sum / k;
                float batch_M2 = batch_lum_sq_sum - batch_lum_sum * batch_mean;  // = sum(x^2) - k*mean^2

                float prev_M2 = cpu_variance_buffer[pixel_index];
                float delta = batch_mean - prev_mean_lum;
                float combined_M2 = prev_M2 + batch_M2;
                if (prev_samples > 0.0f) {
                    combined_M2 += delta * delta * (prev_samples * k) / new_total_samples;
                }
                cpu_variance_buffer[pixel_index] = std::clamp(combined_M2, 0.0f, 1.0e8f);
            }

            // Use blended color for display
            new_color = blended_color;

            // Write to surface with tone mapping
            auto toSRGB = [](float c) {
                if (c <= 0.0031308f)
                    return 12.92f * c;
                else
                    return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
                };

            // Calculate Exposure Factor (MUST match GPU path - OptixWrapper for consistency)
            float exposure_factor = 1.0f;
            if (scene.camera) {
                if (scene.camera->auto_exposure) {
                    // Auto Exposure: use EV compensation only
                    exposure_factor = std::pow(2.0f, scene.camera->ev_compensation);
                }
                else if (scene.camera->use_physical_exposure) {
                    // Physical Manual Mode: calculate from ISO/Shutter/F-stop (matches GPU)
                    float iso_mult = (scene.camera->iso_preset_index >= 0 && scene.camera->iso_preset_index < (int)CameraPresets::ISO_PRESET_COUNT) ?
                                     CameraPresets::ISO_PRESETS[scene.camera->iso_preset_index].exposure_multiplier : 1.0f;
                    float shutter_time = (scene.camera->shutter_preset_index >= 0 && scene.camera->shutter_preset_index < (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT) ?
                                         CameraPresets::SHUTTER_SPEED_PRESETS[scene.camera->shutter_preset_index].speed_seconds : 0.004f;

                    // Use F-Stop Number
                    float f_number = 16.0f;
                    if (scene.camera->fstop_preset_index > 0 && scene.camera->fstop_preset_index < (int)CameraPresets::FSTOP_PRESET_COUNT) {
                        f_number = CameraPresets::FSTOP_PRESETS[scene.camera->fstop_preset_index].f_number;
                    }
                    else {
                        // Custom Mode: Estimate f-number from aperture (diameter/radius)
                        if (scene.camera->aperture > 0.001f)
                            f_number = 0.8f / scene.camera->aperture;
                        else
                            f_number = 16.0f;
                    }
                    float aperture_sq = f_number * f_number + 1e-6f;
                    float ev_comp = std::pow(2.0f, scene.camera->ev_compensation);
                    float current_val = (iso_mult * shutter_time) / (aperture_sq + 0.001f);
                    float baseline_val = 0.00003125f; // Sunny 16 baseline
                    exposure_factor = (current_val / baseline_val) * ev_comp;
                }
                else {
                    // Manual Mode (non-physical): use EV compensation only
                    exposure_factor = std::pow(2.0f, scene.camera->ev_compensation);
                }

            }


            // Apply Reinhard Tone Mapping (CPU Parity with GPU)
            // GPU uses: color / (color + 1.0f) in make_color
            // Note: Exposure is applied BEFORE tone mapping in some pipelines, 
            // but GPU make_color applies it AFTER "new_color" is computed.
            // Wait, GPU raygen applies exposure to new_color, THEN calls make_color.
            // So Tone Mapping happens on EXPOSED color.

            Vec3 exposed_color = new_color * exposure_factor;

            // Reinhard Operator
            exposed_color.x = exposed_color.x / (exposed_color.x + 1.0f);
            exposed_color.y = exposed_color.y / (exposed_color.y + 1.0f);
            exposed_color.z = exposed_color.z / (exposed_color.z + 1.0f);

            int r = static_cast<int>(255 * std::clamp(toSRGB(std::max(0.0f, exposed_color.x)), 0.0f, 1.0f));
            int g = static_cast<int>(255 * std::clamp(toSRGB(std::max(0.0f, exposed_color.y)), 0.0f, 1.0f));
            int b = static_cast<int>(255 * std::clamp(toSRGB(std::max(0.0f, exposed_color.z)), 0.0f, 1.0f));

            Uint32* pixels = static_cast<Uint32*>(surface->pixels);
            int screen_index = (surface->h - 1 - j) * (surface->pitch / 4) + i;
            pixels[screen_index] = SDL_MapRGB(surface->format, r, g, b);
        }
        };

    // Launch threads
    auto start_time = std::chrono::high_resolution_clock::now();

    for (unsigned int t = 0; t < num_threads; ++t) {
        threads.emplace_back(progressive_worker, t);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    float pass_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    // Check if stopped
    if (rendering_stopped_cpu.load()) {
        SCENE_LOG_WARN("CPU Render stopped by user");
        return;
    }

    cpu_accumulated_samples += samples_this_pass;
    invalidateCPUDenoisedBuffer();
        // SCENE_LOG_INFO("CPU Render Pass Completed. Total samples: " + std::to_string(cpu_accumulated_samples));

    // Update window title with progress
    if (window) {
        extern std::string active_model_path;
        std::string projectName = active_model_path;
        if (projectName.empty() || projectName == "Untitled") {
            projectName = "Untitled";
        } else {
            // Extract filename from path
            size_t lastSlash = projectName.find_last_of("\\/");
            if (lastSlash != std::string::npos) {
                projectName = projectName.substr(lastSlash + 1);
            }
        }

        float progress = 100.0f * cpu_accumulated_samples / target_max_samples;
        std::string title = "RayTrophi Studio [" + projectName + "] - CPU - Sample " + std::to_string(cpu_accumulated_samples) +
            "/" + std::to_string(target_max_samples) +
            " (" + std::to_string(int(progress)) + "%) - " +
            std::to_string(int(pass_ms)) + "ms/sample";
        SDL_SetWindowTitle(window, title.c_str());
    }
}

// Add this implementation at the end of Renderer.cpp before the closing brace

namespace {
std::vector<std::shared_ptr<Hittable>> filterVisibleGeometryObjects(
    const std::vector<std::shared_ptr<Hittable>>& objects) {
    std::vector<std::shared_ptr<Hittable>> filtered;
    filtered.reserve(objects.size());

    for (const auto& obj : objects) {
        if (!obj || !obj->visible) {
            continue;
        }
        filtered.push_back(obj);
    }

    return filtered;
}
}

// Rebuild OptiX geometry after scene modifications (deletion/addition)
void Renderer::rebuildBackendGeometry(SceneData& scene) {
    // Rebuild geometry TLAS
    rebuildBackendGeometryWithList(filterVisibleGeometryObjects(scene.world.objects));
    // Sync all volumes (VDB/Gas)
    // [VULKAN FIX] For Vulkan, rebuildBackendGeometryWithList() sets g_vulkan_rebuild_pending=true
    // and returns immediately — m_orderedVDBInstances is still stale/empty at this point.
    // Calling syncVolumetricData here would upload an SSBO with incorrect slot ordering or
    // all-inactive entries (if old VDB IDs don't match the freshly loaded scene VDB IDs).
    // The correct sync happens in the pending-block (Main.cpp): after updateGeometry() rebuilds
    // m_orderedVDBInstances, syncVDBVolumesToGPU() + updateBackendMaterials() ensure the SSBO
    // is in the right order. Skip the premature call here for Vulkan.
    if (!render_settings.use_vulkan) {
        WorldData wd = world.getGPUData();
        VolumetricRenderer::syncVolumetricData(scene, m_backend, &wd);
    }
    // Restore hair after rebuild.
    // For Vulkan the actual BLAS/TLAS rebuild is deferred: rebuildBackendGeometryWithList()
    // sets g_vulkan_rebuild_pending and returns immediately — no BLASes exist yet.
    // uploadHairToGPU() would create hair BLASes against the pre-rebuild TLAS and then the
    // pending-block in Main.cpp would immediately destroy them (rebuildAccelerationStructure).
    // The pending-block already calls uploadHairToGPU() after updateGeometry() completes, so
    // skip the call here to avoid the wasted work and the double-clearHairGeometry/waitIdle.
    if (!render_settings.use_vulkan) {
        uploadHairToGPU();
    }
}


void Renderer::rebuildBackendGeometryWithList(const std::vector<std::shared_ptr<Hittable>>& objects) {
    if (!m_backend) {
        SCENE_LOG_WARN("[Backend] Cannot rebuild - no backend pointer");
        return;
    }
    if (render_settings.use_vulkan) {
        extern bool g_vulkan_rebuild_pending;
        g_vulkan_rebuild_pending = true;
        SCENE_LOG_INFO("[Vulkan] Geometry update requested - signaling heavy rebuild flag.");
        return;
    }
    if (!g_hasCUDA) return;
      Backend::OptixBackend* optixBackend = dynamic_cast<Backend::OptixBackend*>(m_backend);
    OptixWrapper* optix_gpu_ptr = optixBackend ? optixBackend->getOptixWrapper() : nullptr;

    // Handle empty list
    size_t hairCount = hairSystem.getTotalStrandCount();
    SCENE_LOG_INFO("[OptiX Rebuild] Start Rebuild. Objects: " + std::to_string(objects.size()) + 
                   ", Hair Strands: " + std::to_string(hairCount));

    if (objects.empty() && hairCount == 0) {
        SCENE_LOG_INFO("[OptiX] Scene empty (No objects, no hair), clearing GPU scene");
        if (optix_gpu_ptr) optix_gpu_ptr->clearScene();
        if (m_backend) m_backend->resetAccumulation();
        return;
    }


    // Global flag to block concurrent updates
    extern std::atomic<bool> g_optix_rebuild_in_progress;
    g_optix_rebuild_in_progress.store(true, std::memory_order_release);

    try {
        // Parallel Extraction of Triangles from Hittable objects
        size_t num_objects = objects.size();
        std::vector<std::shared_ptr<Triangle>> triangles;
        triangles.reserve(num_objects);

        if (num_objects > 1000) {
            unsigned int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 4;

            size_t chunk_size = (num_objects + num_threads - 1) / num_threads;
            std::vector<std::future<std::vector<std::shared_ptr<Triangle>>>> futures;

            for (unsigned int t = 0; t < num_threads; ++t) {
                size_t start = t * chunk_size;
                size_t end = std::min(start + chunk_size, num_objects);
                if (start >= end) continue;

                futures.push_back(std::async(std::launch::async,
                    [&objects, start, end]() {
                        std::vector<std::shared_ptr<Triangle>> local_tris;
                        local_tris.reserve((end - start));
                        for (size_t i = start; i < end; ++i) {
                            if (auto tri = std::dynamic_pointer_cast<Triangle>(objects[i])) {
                                if (tri->visible) {
                                    local_tris.push_back(tri);
                                }
                            }
                        }
                        return local_tris;
                    }
                ));
            }

            for (auto& f : futures) {
                auto part = f.get();
                triangles.insert(triangles.end(), part.begin(), part.end());
            }
        }
        else {
            std::function<void(const std::shared_ptr<Hittable>&)> collect;
            collect = [&](const std::shared_ptr<Hittable>& obj) {
                if (!obj) return;
                if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                    if (tri->visible) triangles.push_back(tri);
                } else if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
                    for (auto& child : list->objects) collect(child);
                } else if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                    // Collect from source if it exists
                    if (inst->source_triangles) {
                        for (auto& tri : *inst->source_triangles) triangles.push_back(tri);
                    }
                } else if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
                    if (bvh->left) collect(bvh->left);
                    if (bvh->right) collect(bvh->right);
                }
            };
            for (const auto& obj : objects) collect(obj);
        }

        OptixGeometryData optix_data;
        if (!triangles.empty()) {
            optix_data = assimpLoader.convertTrianglesToOptixData(triangles);
        }

        // ===================================================================
        // HAIR GEOMETRY DATA (OptiX Curve Primitives)
        // ===================================================================
        // TODO: Fix OptiX 8 compatibility (Error 7001). Disabled for stability.
        // Hair Geometry Generation (OptiX Curves)
        if (hairSystem.getTotalStrandCount() > 0) {
            auto groomNames = hairSystem.getGroomNames();
            for (const auto& groomName : groomNames) {
                std::vector<float> hairVertices4;
                std::vector<unsigned int> hairIndices;
                std::vector<uint32_t> hairStrandIDs;
                std::vector<float> hairTangents3;
                std::vector<float> hairRootUVs2;
                std::vector<float> hairStrandVs;
                size_t vertexCount = 0;
                size_t segmentCount = 0;
                
                Hair::HairMaterialParams matParams;
                int matID = 0;
                int meshMatID = -1;
                
                // Call with all 13 arguments explicitly
                bool isSpline = hairSystem.getOptiXCurveDataByGroom(
                    groomName,          // 1
                    hairVertices4,      // 2
                    hairIndices,        // 3
                    hairStrandIDs,      // 4
                    hairTangents3,      // 5
                    hairRootUVs2,       // 6
                    hairStrandVs,       // 7
                    vertexCount,        // 8
                    segmentCount,       // 9
                    matParams,          // 10
                    matID,              // 11
                    meshMatID,          // 12
                    true                // 13 (includeInterpolated)
                );
                
                if (segmentCount > 0) {
                    CurveGeometry curve_geom;
                    curve_geom.name = groomName; // Use actual groom name
                    curve_geom.vertex_count = vertexCount;
                    curve_geom.segment_count = segmentCount;
                    curve_geom.use_bspline = isSpline; 
                    
                    // Determine material ID from this groom
                    curve_geom.material_id = matID;
                    curve_geom.mesh_material_id = meshMatID;
                    curve_geom.hair_material = Hair::HairBSDF::convertToGpu(matParams); // Fix: assign material properly on load
                    curve_geom.strand_v = hairStrandVs;
                    curve_geom.strand_ids = hairStrandIDs;
                    
                    // Copy Root UVs
                    curve_geom.root_uvs.resize(segmentCount);
                    for (size_t i = 0; i < segmentCount; ++i) {
                        curve_geom.root_uvs[i] = make_float2(hairRootUVs2[i*2], hairRootUVs2[i*2+1]);
                    }

                    // Copy to CurveGeometry vectors
                    curve_geom.vertices.resize(vertexCount);
                    for (size_t i = 0; i < vertexCount; ++i) {
                        curve_geom.vertices[i] = make_float4(hairVertices4[i*4], hairVertices4[i*4+1], hairVertices4[i*4+2], hairVertices4[i*4+3]);
                    }
                    curve_geom.indices = hairIndices;
                    curve_geom.tangents.resize(segmentCount);
                    for (size_t i = 0; i < segmentCount; ++i) {
                        curve_geom.tangents[i] = make_float3(hairTangents3[i*3], hairTangents3[i*3+1], hairTangents3[i*3+2]);
                    }
                    
                    // Copy root UVs
                    curve_geom.root_uvs.resize(segmentCount);
                    for (size_t i = 0; i < segmentCount; ++i) {
                        curve_geom.root_uvs[i] = make_float2(hairRootUVs2[i*2], hairRootUVs2[i*2+1]);
                    }

                    optix_data.curves.push_back(curve_geom);
                   // SCENE_LOG_INFO("[Hair GPU] Uploaded groom '" + groomName + "' (" + std::to_string(segmentCount) + " segments) with MaterialID=" + std::to_string(matID));
                }
            }
        }

        if (optix_gpu_ptr) {
            optix_gpu_ptr->validateMaterialIndices(optix_data);
            optix_gpu_ptr->buildFromDataTLAS(optix_data, objects);
        }
        
        if (m_backend) m_backend->resetAccumulation();


        if (triangles.size() > 1000) {
           // SCENE_LOG_INFO("[OptiX] Geometry rebuilt (Snapshot) - " + std::to_string(triangles.size()) + " triangles");
        }
    }
    catch (std::exception& e) {
        SCENE_LOG_ERROR("[OptiX] Rebuild failed: " + std::string(e.what()));
        optix_gpu_ptr->clearScene();
    }

    g_optix_rebuild_in_progress.store(false, std::memory_order_release);
}


//
// Performance: ~1-5ms vs ~200-500ms for rebuildOptiXGeometry
// ============================================================================
void Renderer::updateBackendMaterials(SceneData& scene) {
    // Only update the active render backend here. The dedicated viewport backend
    // is synced explicitly by Main.cpp when an interactive viewport mode needs
    // Material Preview data. Keeping these paths separate prevents startup and
    // Solid->Rendered transitions from filling both render and raster texture
    // pools with the same high-resolution material set.
    updateBackendMaterials(scene, m_backend);
}

void Renderer::updateBackendMaterials(SceneData& scene, Backend::IBackend* targetBackend) {
    Backend::IBackend* backend = targetBackend ? targetBackend : m_backend;
    if (!backend) return;

    // Prefer OptiX/CUDA path if available
    Backend::OptixBackend* optixBackend = dynamic_cast<Backend::OptixBackend*>(backend);
    OptixWrapper* optix_gpu_ptr = (optixBackend && g_hasCUDA) ? optixBackend->getOptixWrapper() : nullptr;
    const bool canUploadOptixTextures = (optix_gpu_ptr != nullptr) && g_hasOptix;

    if (canUploadOptixTextures) {
        ScopedCudaTextureUpload allowCudaTextureUpload;
        try {
            // Existing OptiX-specific material sync (unchanged)
            auto& mgr = MaterialManager::getInstance();
            const auto& all_materials = mgr.getAllMaterials();
            if (all_materials.empty()) return;

            std::vector<GpuMaterial> gpu_materials;
            std::vector<OptixGeometryData::VolumetricInfo> volumetric_info;
            gpu_materials.reserve(all_materials.size());
            volumetric_info.reserve(all_materials.size());

            for (size_t i = 0; i < all_materials.size(); ++i) {
                const auto& mat = all_materials[i];
                if (!mat) continue;

                GpuMaterial gpu_mat = {};
                OptixGeometryData::VolumetricInfo vol_info = {};

                if (mat->type() == MaterialType::Volumetric) {
                    Volumetric* vol_mat = static_cast<Volumetric*>(mat.get());
                    if (vol_mat) {
                        vol_info.is_volumetric = 1;
                        Vec3 albedo = vol_mat->getAlbedo();
                        Vec3 emission = vol_mat->getEmissionColor();
                        vol_info.density = static_cast<float>(vol_mat->getDensity());
                        vol_info.absorption = static_cast<float>(vol_mat->getAbsorption());
                        vol_info.scattering = static_cast<float>(vol_mat->getScattering());
                        vol_info.albedo = make_float3(albedo.x, albedo.y, albedo.z);
                        vol_info.emission = make_float3(emission.x, emission.y, emission.z);
                        vol_info.g = static_cast<float>(vol_mat->getG());
                        vol_info.step_size = vol_mat->getStepSize();
                        vol_info.max_steps = vol_mat->getMaxSteps();
                        vol_info.noise_scale = vol_mat->getNoiseScale();

                        vol_info.multi_scatter = vol_mat->getMultiScatter();
                        vol_info.g_back = vol_mat->getGBack();
                        vol_info.lobe_mix = vol_mat->getLobeMix();
                        vol_info.light_steps = vol_mat->getLightSteps();
                        vol_info.shadow_strength = vol_mat->getShadowStrength();

                        vol_info.aabb_min = make_float3(0, 0, 0);
                        vol_info.aabb_max = make_float3(1, 1, 1);

                        gpu_mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
                        gpu_mat.roughness = 1.0f;
                        gpu_mat.metallic = 0.0f;
                        gpu_mat.specular = 0.5f;
                        gpu_mat.emission = make_float3(0.0f, 0.0f, 0.0f);
                        gpu_mat.ior = 1.0f;
                        gpu_mat.transmission = 0.0f;
                        gpu_mat.opacity = 1.0f;

                        if (vol_mat->hasVDBVolume()) {
                            int vdb_id = vol_mat->getVDBVolumeID();
                            auto& vdb_mgr = VDBVolumeManager::getInstance();
                            
                            // PROACTIVE CUDA UPLOAD ON BACKEND SWITCH:
                            // If we are in OptiX/CUDA path, and the CUDA grid is not yet uploaded,
                            // but the host grid exists (e.g. loaded under Vulkan mode), trigger CUDA upload.
                            if (!vdb_mgr.getGPUGrid(vdb_id) && vdb_mgr.getHostGrid(vdb_id)) {
                                vdb_mgr.uploadToGPU(vdb_id, true);
                            }
                            
                            void* grid_ptr = vdb_mgr.getGPUGrid(vdb_id);
                            vol_info.nanovdb_grid = grid_ptr;
                            vol_info.has_nanovdb = (grid_ptr != nullptr) ? 1 : 0;
                        }
                    }
                } else {
                    auto getCudaTex = [canUploadOptixTextures](const std::shared_ptr<Texture>& tex) -> cudaTextureObject_t {
                        if (tex && tex->is_loaded()) {
                            if (canUploadOptixTextures && (!tex->is_gpu_uploaded || tex->get_cuda_texture() == 0)) {
                                if (tex->get_cuda_texture() == 0) {
                                    tex->is_gpu_uploaded = false;
                                }
                                tex->upload_to_gpu();
                            }
                            return tex->get_cuda_texture();
                        }
                        return 0;
                    };

                    if (mat->type() == MaterialType::PrincipledBSDF) {
                        PrincipledBSDF* pbsdf = static_cast<PrincipledBSDF*>(mat.get());
                        if (mat->gpuMaterial) gpu_mat = *(mat->gpuMaterial);
                        bool is_water = mat->materialName.find("Water") != std::string::npos || mat->materialName.find("River") != std::string::npos;
                        if (!is_water) {
                            const PBRMaterialSnapshot snapshot = capturePBRMaterialSnapshot(*pbsdf);
                            applyPBRMaterialSnapshotToGpuMaterial(snapshot, gpu_mat);
                        }

                        gpu_mat.albedo_tex      = getCudaTex(pbsdf->albedoProperty.texture);
                        gpu_mat.normal_tex      = getCudaTex(pbsdf->normalProperty.texture);
                        gpu_mat.roughness_tex   = getCudaTex(pbsdf->roughnessProperty.texture);
                        gpu_mat.metallic_tex    = getCudaTex(pbsdf->metallicProperty.texture);
                        gpu_mat.specular_tex    = getCudaTex(pbsdf->specularProperty.texture);
                        gpu_mat.emission_tex    = getCudaTex(pbsdf->emissionProperty.texture);
                        gpu_mat.opacity_tex     = getCudaTex(pbsdf->opacityProperty.texture);
                        gpu_mat.transmission_tex= getCudaTex(pbsdf->transmissionProperty.texture);
                        gpu_mat.height_tex      = getCudaTex(pbsdf->heightProperty.texture);
                    } else if (mat->gpuMaterial) {
                        gpu_mat = *mat->gpuMaterial;
                        vol_info.is_volumetric = 0;
                    } else {
                        gpu_mat.albedo = make_float3(0.8f, 0.8f, 0.8f);
                        gpu_mat.roughness = 0.5f;
                        gpu_mat.metallic = 0.0f;
                        gpu_mat.specular = 0.5f;
                        gpu_mat.emission = make_float3(0.0f, 0.0f, 0.0f);
                        gpu_mat.ior = 1.5f;
                        gpu_mat.transmission = 0.0f;
                        gpu_mat.opacity = 1.0f;
                        vol_info.is_volumetric = 0;
                    }
                }

                gpu_materials.push_back(gpu_mat);
                volumetric_info.push_back(vol_info);
            }

            if (!gpu_materials.empty()) {
                optix_gpu_ptr->updateMaterialBuffer(gpu_materials);
                if (!volumetric_info.empty()) optix_gpu_ptr->updateSBTVolumetricData(volumetric_info);
                optix_gpu_ptr->syncSBTMaterialData(gpu_materials, true);
            }

            optix_gpu_ptr->updateHairMaterialsOnly(hairSystem);
            setHairMaterial(hairMaterial);
            WorldData wd = world.getGPUData();
            VolumetricRenderer::syncVolumetricData(scene, backend, &wd);
            optix_gpu_ptr->resetAccumulation();
        }
        catch (std::exception& e) {
            SCENE_LOG_ERROR("[OptiX] updateOptiXMaterialsOnly failed: " + std::string(e.what()));
        }

        return;
    }

    // Generic backend path (Vulkan, CPU backends etc.)
    try {
        auto& mgr = MaterialManager::getInstance();
        const auto& all_materials = mgr.getAllMaterials();
        if (all_materials.empty()) return;

        // === Build terrain layer data for Vulkan splat-blending ===
        // Map: material_id -> terrain layer buffer index (used to set FLAG_TERRAIN in material flags)
        std::unordered_map<uint16_t, uint32_t> terrainMatToLayerIdx;
        std::vector<Backend::IBackend::TerrainLayerData> terrainLayers;

        const auto& terrainObjects = TerrainManager::getInstance().getTerrains();
        for (const auto& t : terrainObjects) {
            if (t.layers.empty() || !t.splatMap || !t.splatMap->is_loaded()) continue;

            Backend::IBackend::TerrainLayerData ld{};
            uint32_t activeCount = 0;
            for (int k = 0; k < 4 && k < (int)t.layers.size(); ++k) {
                if (t.layers[k]) {
                    uint16_t mid = mgr.getMaterialID(t.layers[k]->materialName);
                    ld.layer_mat_id[k] = mid;
                    ld.layer_uv_scale[k] = (k < (int)t.layer_uv_scales.size()) ? t.layer_uv_scales[k] : 1.0f;
                    ++activeCount;
                } else {
                    ld.layer_mat_id[k] = 0;
                    ld.layer_uv_scale[k] = 1.0f;
                }
            }
            ld.splatMapTexture = reinterpret_cast<int64_t>(t.splatMap.get());
            ld.layer_count = activeCount;

            uint32_t layerIdx = (uint32_t)terrainLayers.size();
            terrainLayers.push_back(ld);
            terrainMatToLayerIdx[t.material_id] = layerIdx;
        }

        std::vector<Backend::IBackend::MaterialData> backendMaterials;
        backendMaterials.reserve(all_materials.size());

        for (size_t i = 0; i < all_materials.size(); ++i) {
            const auto& mat = all_materials[i];
            Backend::IBackend::MaterialData data = {};
            if (!mat) { backendMaterials.push_back(data); continue; }

            data.albedo = mat->albedo;
            data.ior = mat->ior;
            data.opacity = 1.0f;

            if (mat->type() == MaterialType::PrincipledBSDF) {
                PrincipledBSDF* pbsdf = static_cast<PrincipledBSDF*>(mat.get());
                const PBRMaterialSnapshot snapshot = capturePBRMaterialSnapshot(*pbsdf);
                data = makeBackendMaterialDataFromSnapshot(snapshot);

                auto getH = [](const std::shared_ptr<Texture>& tex) -> int64_t { return tex ? reinterpret_cast<int64_t>(tex.get()) : 0; };
                data.albedoTexture = getH(pbsdf->albedoProperty.texture);
                data.normalTexture = getH(pbsdf->normalProperty.texture);
                data.roughnessTexture = getH(pbsdf->roughnessProperty.texture);
                data.metallicTexture = getH(pbsdf->metallicProperty.texture);
                data.specularTexture = getH(pbsdf->specularProperty.texture);
                data.emissionTexture = getH(pbsdf->emissionProperty.texture);
                data.transmissionTexture = getH(pbsdf->transmissionProperty.texture);
                data.opacityTexture = getH(pbsdf->opacityProperty.texture);
                data.heightTexture = getH(pbsdf->heightProperty.texture);
            }

            // Copy water-specific GPU params (live in GpuMaterial, not duplicated in PrincipledBSDF)
            if (mat->gpuMaterial) {
                const auto& gm = *mat->gpuMaterial;
                if ((gm.flags & GPU_MAT_FLAG_WATER) != 0 || gm.sheen > 0.0001f) {
                    WaterShader::SurfaceParams surfaceParams = WaterShader::surfaceParamsFromGpuMaterial(gm);
                    WaterShader::applySurfaceParamsToBackendMaterialData(surfaceParams, data);
                } else {
                    data.micro_detail_strength = gm.micro_detail_strength;
                    data.micro_detail_scale    = gm.micro_detail_scale;
                    data.tile_break_strength   = gm.tile_break_strength;
                }
            }

            if (backend && backend->getInfo().type == Backend::BackendType::VULKAN_RT && mat->gpuMaterial) {
                int64_t fftHeightTexture = 0;
                int64_t fftNormalTexture = 0;
                if (WaterManager::getInstance().syncVulkanFFTTexturesForMaterial(static_cast<uint16_t>(i), backend, fftHeightTexture, fftNormalTexture)) {
                    data.heightTexture = fftHeightTexture;
                    data.normalTexture = fftNormalTexture;
                    data.flags |= Backend::IBackend::MAT_FLAG_WATER_FFT_READY;
                }
            }

            // Mark terrain base materials so the Vulkan shader knows to use splat blending
            auto tit = terrainMatToLayerIdx.find((uint16_t)i);
            if (tit != terrainMatToLayerIdx.end()) {
                data.flags |= Backend::IBackend::MAT_FLAG_TERRAIN;
                data.terrainLayerIdx = tit->second;
            }

            backendMaterials.push_back(data);
        }

        if (backend) {
            backend->uploadMaterials(backendMaterials);
            if (!terrainLayers.empty())
                backend->uploadTerrainLayerMaterials(terrainLayers);
            WorldData wd = world.getGPUData();
            VolumetricRenderer::syncVolumetricData(scene, backend, &wd);
            backend->resetAccumulation();
        }
    }
    catch (std::exception& e) {
        SCENE_LOG_ERROR(std::string("[Renderer] updateBackendMaterials failed: ") + e.what());
    }
}

void Renderer::updateBackendMaterial(SceneData& scene, uint16_t material_id) {
    auto& mgr = MaterialManager::getInstance();
    auto mat = mgr.getMaterial(material_id);

    struct PaintDirtySnapshot {
        Texture* tex;
        bool full;
        int min_x, min_y, max_x, max_y;
    };
    std::vector<PaintDirtySnapshot> dirtySnapshot;

    if (mat && mat->type() == MaterialType::PrincipledBSDF) {
        auto* pbsdf = static_cast<PrincipledBSDF*>(mat);
        auto capture = [&](const std::shared_ptr<Texture>& tex) {
            if (!tex || !tex->vulkan_dirty) return;
            dirtySnapshot.push_back({
                tex.get(),
                tex->vulkan_dirty_full,
                tex->vulkan_dirty_min_x,
                tex->vulkan_dirty_min_y,
                tex->vulkan_dirty_max_x,
                tex->vulkan_dirty_max_y
            });
        };

        capture(pbsdf->albedoProperty.texture);
        capture(pbsdf->normalProperty.texture);
        capture(pbsdf->roughnessProperty.texture);
        capture(pbsdf->metallicProperty.texture);
        capture(pbsdf->emissionProperty.texture);
        capture(pbsdf->opacityProperty.texture);
        capture(pbsdf->transmissionProperty.texture);
        capture(pbsdf->heightProperty.texture);
    }

    updateBackendMaterial(scene, material_id, m_backend);

    extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
    if (g_viewport_backend && g_viewport_backend.get() != m_backend) {
        for (const auto& snapshot : dirtySnapshot) {
            if (!snapshot.tex) continue;
            snapshot.tex->vulkan_dirty = true;
            snapshot.tex->vulkan_dirty_full = snapshot.full;
            snapshot.tex->vulkan_dirty_min_x = snapshot.min_x;
            snapshot.tex->vulkan_dirty_min_y = snapshot.min_y;
            snapshot.tex->vulkan_dirty_max_x = snapshot.max_x;
            snapshot.tex->vulkan_dirty_max_y = snapshot.max_y;
        }
        updateBackendMaterial(scene, material_id, g_viewport_backend.get());
    }
}

void Renderer::updateBackendMaterial(SceneData& scene, uint16_t material_id, Backend::IBackend* targetBackend) {
    Backend::IBackend* backend = targetBackend ? targetBackend : m_backend;
    if (!backend || material_id == MaterialManager::INVALID_MATERIAL_ID) return;

    auto& mgr = MaterialManager::getInstance();
    auto mat = mgr.getMaterial(material_id);
    if (!mat) return;

    Backend::IBackend::MaterialData data{};
    data.albedo = mat->albedo;
    data.ior = mat->ior;
    data.opacity = 1.0f;

    if (mat->type() == MaterialType::PrincipledBSDF) {
        PrincipledBSDF* pbsdf = static_cast<PrincipledBSDF*>(mat);
        const PBRMaterialSnapshot snapshot = capturePBRMaterialSnapshot(*pbsdf);
        data = makeBackendMaterialDataFromSnapshot(snapshot);

        auto getH = [](const std::shared_ptr<Texture>& tex) -> int64_t {
            return tex ? reinterpret_cast<int64_t>(tex.get()) : 0;
        };
        data.albedoTexture = getH(pbsdf->albedoProperty.texture);
        data.normalTexture = getH(pbsdf->normalProperty.texture);
        data.roughnessTexture = getH(pbsdf->roughnessProperty.texture);
        data.metallicTexture = getH(pbsdf->metallicProperty.texture);
        data.specularTexture = getH(pbsdf->specularProperty.texture);
        data.emissionTexture = getH(pbsdf->emissionProperty.texture);
        data.transmissionTexture = getH(pbsdf->transmissionProperty.texture);
        data.opacityTexture = getH(pbsdf->opacityProperty.texture);
        data.heightTexture = getH(pbsdf->heightProperty.texture);
    }

    if (mat->gpuMaterial) {
        const auto& gm = *mat->gpuMaterial;
        if ((gm.flags & GPU_MAT_FLAG_WATER) != 0 || gm.sheen > 0.0001f) {
            WaterShader::SurfaceParams surfaceParams = WaterShader::surfaceParamsFromGpuMaterial(gm);
            WaterShader::applySurfaceParamsToBackendMaterialData(surfaceParams, data);
        } else {
            data.micro_detail_strength = gm.micro_detail_strength;
            data.micro_detail_scale    = gm.micro_detail_scale;
            data.tile_break_strength   = gm.tile_break_strength;
        }
    }

    if (backend && backend->getInfo().type == Backend::BackendType::VULKAN_RT && mat->gpuMaterial) {
        int64_t fftHeightTexture = 0;
        int64_t fftNormalTexture = 0;
        if (WaterManager::getInstance().syncVulkanFFTTexturesForMaterial(material_id, backend, fftHeightTexture, fftNormalTexture)) {
            data.heightTexture = fftHeightTexture;
            data.normalTexture = fftNormalTexture;
            data.flags |= Backend::IBackend::MAT_FLAG_WATER_FFT_READY;
        }
    }

    if (!backend->updateMaterial(material_id, data)) {
        updateBackendMaterials(scene, backend);
    }
}

// ============================================================================
// WIND ANIMATION SYSTEM
// ============================================================================



FoliageWindUpdateStats Renderer::updateWind(SceneData& scene, float time) {
    return FoliageWindSystem::update(scene, time, this->m_backend);
}

namespace {

bool matchesRendererNodeNameForViewportSync(const std::string& instanceNodeName,
                                            const std::string& queryNodeName) {
    if (queryNodeName.empty() || instanceNodeName.empty()) return false;
    if (instanceNodeName == queryNodeName) return true;
    const std::string matPrefix = queryNodeName + "_mat_";
    return instanceNodeName.rfind(matPrefix, 0) == 0;
}

void collectRendererTrianglesForNode(const std::shared_ptr<Hittable>& obj,
                                     const std::string& nodeName,
                                     std::vector<std::shared_ptr<Triangle>>& outTriangles) {
    if (!obj) return;

    if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
        if (matchesRendererNodeNameForViewportSync(inst->node_name, nodeName) &&
            inst->source_triangles) {
            for (const auto& tri : *inst->source_triangles) {
                if (!tri) continue;
                if (matchesRendererNodeNameForViewportSync(tri->getNodeName(), nodeName) ||
                    matchesRendererNodeNameForViewportSync(nodeName, tri->getNodeName()) ||
                    tri->getNodeName().empty()) {
                    outTriangles.push_back(tri);
                }
            }
        }
        return;
    }

    if (auto list = std::dynamic_pointer_cast<HittableList>(obj)) {
        for (const auto& child : list->objects) {
            collectRendererTrianglesForNode(child, nodeName, outTriangles);
        }
        return;
    }

    if (auto bvh = std::dynamic_pointer_cast<ParallelBVHNode>(obj)) {
        collectRendererTrianglesForNode(bvh->left, nodeName, outTriangles);
        collectRendererTrianglesForNode(bvh->right, nodeName, outTriangles);
        return;
    }

    if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
        if (matchesRendererNodeNameForViewportSync(tri->getNodeName(), nodeName)) {
            outTriangles.push_back(tri);
        }
    }
}

void syncViewportMaterialBindingsFromScene(Backend::IBackend* backend,
                                           SceneData& scene,
                                           const std::string& nodeName,
                                           int oldMatID,
                                           int newMatID) {
    auto* viewportBackend = dynamic_cast<Backend::IViewportBackend*>(backend);
    if (!viewportBackend) {
        if (backend) backend->updateInstanceMaterialBinding(nodeName, oldMatID, newMatID);
        return;
    }

    std::vector<std::shared_ptr<Triangle>> triangles;
    for (const auto& obj : scene.world.objects) {
        collectRendererTrianglesForNode(obj, nodeName, triangles);
    }

    if (!triangles.empty() && viewportBackend->updateRasterMeshFromTriangles(nodeName, triangles)) {
        viewportBackend->resetAccumulation();
        return;
    }

    viewportBackend->updateInstanceMaterialBinding(nodeName, oldMatID, newMatID);
}

} // namespace

// ===============================================================================
// GAS VOLUME OPTIX SYNC
// ===============================================================================
// �����������������������������������������������������������������������������
// Volumetric Sync - Unified logic for Gas and VDB
// �����������������������������������������������������������������������������
void Renderer::updateBackendGasVolumes(SceneData& scene) {
    std::unique_ptr<ScopedCudaTextureUpload> allowCudaWorldTextureUpload;
    if (dynamic_cast<Backend::OptixBackend*>(m_backend) != nullptr) {
        allowCudaWorldTextureUpload = std::make_unique<ScopedCudaTextureUpload>();
    }
    WorldData wd = world.getGPUData();
    VolumetricRenderer::syncVolumetricData(scene, m_backend, &wd);
}

void Renderer::updateMeshMaterialBinding(SceneData& scene, const std::string& node_name, int old_mat_id, int new_mat_id) {
    if (m_backend) {
        syncViewportMaterialBindingsFromScene(m_backend, scene, node_name, old_mat_id, new_mat_id);
    }
    extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
    if (g_viewport_backend && g_viewport_backend.get() != m_backend) {
        syncViewportMaterialBindingsFromScene(g_viewport_backend.get(), scene, node_name, old_mat_id, new_mat_id);
    }
    resetCPUAccumulation(); // Ensure CPU path also resets
}

void Renderer::syncCameraToBackend(const Camera& cam) {
    if (!m_backend) return;
    // Publish Stylize state to the global render settings before the backend launch so the
    // OptiX wrapper allocates its AOV buffers (albedo/normal/position) when Stylize is on,
    // independent of the denoiser. Harmless for Vulkan (always allocates its AOV images).
    {
        extern RenderSettings render_settings;
        render_settings.stylize_enabled = stylizeMode.enabled;
    }
    Backend::CameraParams cp;
    cp.origin = cam.lookfrom;
    cp.lookAt = cam.lookat;
    cp.up = cam.vup;
    cp.fov = cam.vfov;
    cp.aperture = cam.aperture;
    cp.focusDistance = cam.focus_dist;
    cp.aspectRatio = cam.aspect_ratio;
    cp.exposureFactor = cam.getPhysicalExposureMultiplier();
    cp.ev_compensation = cam.ev_compensation;
    cp.isoPresetIndex = cam.iso_preset_index;
    cp.shutterPresetIndex = cam.shutter_preset_index;
    cp.fstopPresetIndex = cam.fstop_preset_index;
    cp.autoAE = cam.auto_exposure;
    cp.usePhysicalExposure = cam.use_physical_exposure;
    cp.motionBlurEnabled = cam.enable_motion_blur;
    cp.vignettingEnabled = cam.enable_vignetting;
    cp.chromaticAberrationEnabled = cam.enable_chromatic_aberration;
    
    // Pro Features
    cp.distortion = cam.distortion;
    cp.lens_quality = cam.lens_quality;
    cp.vignetting_amount = cam.vignetting_amount;
    cp.vignetting_falloff = cam.vignetting_falloff;
    cp.chromatic_aberration = cam.chromatic_aberration;
    cp.chromatic_aberration_r = cam.chromatic_aberration_r;
    cp.chromatic_aberration_b = cam.chromatic_aberration_b;
    cp.camera_mode = (int)cam.camera_mode;
    cp.blade_count = cam.blade_count;
    
    // Shake / Handheld
    cp.shake_enabled = cam.enable_camera_shake;
    cp.shake_intensity = cam.shake_intensity;
    cp.shake_frequency = cam.shake_frequency;
    cp.handheld_sway_amplitude = cam.handheld_sway_amplitude;
    cp.handheld_sway_frequency = cam.handheld_sway_frequency;
    cp.breathing_amplitude = cam.breathing_amplitude;
    cp.breathing_frequency = cam.breathing_frequency;
    cp.enable_focus_drift = cam.enable_focus_drift;
    cp.focus_drift_amount = cam.focus_drift_amount;
    cp.operator_skill = (int)cam.operator_skill;
    cp.ibis_enabled = cam.ibis_enabled;
    cp.ibis_effectiveness = cam.ibis_effectiveness;
    cp.rig_mode = (int)cam.rig_mode;

    m_backend->setCamera(cp);
}
