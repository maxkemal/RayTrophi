#pragma once

#include "Backend/IBackend.h"
#include "PrincipledBSDF.h"
#include "WaterShaderCommon.h"
#include "material_gpu.h"
#include <algorithm>
#include <cmath>

namespace WaterShader {

inline float activeWaveStrength(float wave_strength) {
    return std::max(0.001f, wave_strength);
}

inline SurfaceParams surfaceParamsFromGpuMaterial(const GpuMaterial& gpu, float time_seconds = 0.0f) {
    SurfaceParams out;
    out.wave_speed = gpu.anisotropic;
    out.wave_strength = gpu.sheen;
    out.wave_frequency = gpu.sheen_tint;
    out.shallow_color = Vec3(gpu.emission.x, gpu.emission.y, gpu.emission.z);
    out.deep_color = Vec3(gpu.albedo.x, gpu.albedo.y, gpu.albedo.z);
    out.absorption_color = Vec3(gpu.subsurface_color.x, gpu.subsurface_color.y, gpu.subsurface_color.z);
    out.depth_max = gpu.subsurface * 100.0f;
    out.absorption_density = gpu.subsurface_scale;
    out.clarity = std::max(0.1f, 1.0f - out.absorption_density);
    out.ior = gpu.ior;
    out.roughness = gpu.roughness;
    out.foam_level = gpu.translucent;
    out.shore_foam_distance = gpu.subsurface_radius.x;
    out.shore_foam_intensity = gpu.clearcoat;
    out.caustic_intensity = gpu.clearcoat_roughness;
    out.caustic_scale = gpu.subsurface_radius.y;
    out.caustic_speed = gpu.subsurface_anisotropy;
    out.sss_intensity = gpu.subsurface_radius.z;
    out.sss_color = out.absorption_color;
    out.use_fft_ocean = gpu.fft_height_tex != 0 && gpu.fft_normal_tex != 0 && gpu.fft_ocean_size > 0.001f;
    out.fft_ocean_size = resolvedDomain(gpu.fft_ocean_size);
    out.fft_choppiness = gpu.fft_choppiness;
    out.fft_amplitude = gpu.fft_amplitude;
    out.animation_speed = gpu.fft_time_scale;
    out.micro_detail_strength = gpu.micro_detail_strength;
    out.micro_detail_scale = gpu.micro_detail_scale;
    out.micro_anim_speed = gpu.micro_anim_speed;
    out.micro_morph_speed = gpu.micro_morph_speed;
    out.foam_noise_scale = gpu.foam_noise_scale;
    out.foam_threshold = gpu.foam_threshold;
    out.wind_direction = gpu.fft_wind_direction;
    out.wind_speed = gpu.fft_wind_speed;
    out.time = time_seconds;
    return out;
}

inline void applySurfaceParamsToPrincipledBSDF(const SurfaceParams& params, PrincipledBSDF& pbsdf) {
    pbsdf.anisotropic = params.wave_speed;
    pbsdf.sheen = activeWaveStrength(params.wave_strength);
    pbsdf.sheen_tint = params.wave_frequency;
    pbsdf.transmission = params.ior > 1.01f ? 1.0f : 0.0f;
    pbsdf.translucent = params.foam_level;
    pbsdf.clearcoat = params.shore_foam_intensity;
    pbsdf.clearcoatRoughness = params.caustic_intensity;
    pbsdf.subsurface = params.depth_max / 100.0f;
    pbsdf.subsurfaceScale = params.absorption_density;
    pbsdf.subsurfaceColor = params.absorption_color;
    pbsdf.subsurfaceRadius = Vec3(params.shore_foam_distance, params.caustic_scale, params.sss_intensity);
    pbsdf.subsurfaceAnisotropy = params.caustic_speed;
    pbsdf.roughness = params.roughness;
    pbsdf.roughnessProperty.color = Vec3(params.roughness);
    pbsdf.ior = params.ior;
    pbsdf.opacityProperty.alpha = 1.0f;
    pbsdf.albedoProperty.color = params.deep_color;
    pbsdf.emissionProperty.color = params.shallow_color;
    pbsdf.emissionProperty.intensity = 1.0f;
}

inline void applySurfaceParamsToGpuMaterial(
    const SurfaceParams& params,
    GpuMaterial& gpu,
    cudaTextureObject_t fft_height_tex = 0,
    cudaTextureObject_t fft_normal_tex = 0
) {
    gpu.anisotropic = params.wave_speed;
    gpu.sheen = activeWaveStrength(params.wave_strength);
    gpu.sheen_tint = params.wave_frequency;
    gpu.albedo = make_float3(params.deep_color.x, params.deep_color.y, params.deep_color.z);
    gpu.emission = make_float3(params.shallow_color.x, params.shallow_color.y, params.shallow_color.z);
    gpu.subsurface = params.depth_max / 100.0f;
    gpu.subsurface_scale = params.absorption_density;
    gpu.subsurface_color = make_float3(params.absorption_color.x, params.absorption_color.y, params.absorption_color.z);
    gpu.translucent = params.foam_level;
    gpu.clearcoat = params.shore_foam_intensity;
    gpu.clearcoat_roughness = params.caustic_intensity;
    gpu.subsurface_radius = make_float3(params.shore_foam_distance, params.caustic_scale, params.sss_intensity);
    gpu.subsurface_anisotropy = params.caustic_speed;
    gpu.ior = params.ior;
    gpu.roughness = params.roughness;
    gpu.transmission = params.ior > 1.01f ? 1.0f : 0.0f;
    gpu.opacity = 1.0f;
    gpu.metallic = 0.0f;
    gpu.flags |= GPU_MAT_FLAG_WATER;
    gpu.micro_detail_strength = params.micro_detail_strength;
    gpu.micro_detail_scale = params.micro_detail_scale;
    gpu.micro_anim_speed = params.micro_anim_speed;
    gpu.micro_morph_speed = params.micro_morph_speed;
    gpu.foam_noise_scale = params.foam_noise_scale;
    gpu.foam_threshold = params.foam_threshold;
    gpu.fft_height_tex = fft_height_tex;
    gpu.fft_normal_tex = fft_normal_tex;
    gpu.fft_ocean_size = resolvedDomain(params.fft_ocean_size);
    gpu.fft_choppiness = params.fft_choppiness;
    gpu.fft_wind_speed = params.wind_speed;
    gpu.fft_wind_direction = params.wind_direction;
    gpu.fft_amplitude = params.fft_amplitude;
    gpu.fft_time_scale = params.animation_speed;
}

inline void applySurfaceParamsToBackendMaterialData(
    const SurfaceParams& params,
    Backend::IBackend::MaterialData& data
) {
    data.albedo = params.deep_color;
    data.emission = params.shallow_color;
    data.emissionStrength = 1.0f;
    data.roughness = params.roughness;
    data.ior = params.ior;
    data.transmission = params.ior > 1.01f ? 1.0f : 0.0f;
    data.opacity = 1.0f;
    data.metallic = 0.0f;
    data.subsurface = params.depth_max / 100.0f;
    data.subsurfaceColor = params.absorption_color;
    data.subsurfaceRadius = Vec3(params.shore_foam_distance, params.caustic_scale, params.sss_intensity);
    data.subsurfaceScale = params.absorption_density;
    data.subsurfaceAnisotropy = params.caustic_speed;
    data.clearcoat = params.shore_foam_intensity;
    data.clearcoatRoughness = params.caustic_intensity;
    data.translucent = params.foam_level;
    data.anisotropic = params.wave_speed;
    data.sheen = activeWaveStrength(params.wave_strength);
    data.sheenTint = params.wave_frequency;
    data.flags |= Backend::IBackend::MAT_FLAG_WATER;
    data.micro_detail_strength = params.micro_detail_strength;
    data.micro_detail_scale = params.micro_detail_scale;
    data.foam_threshold = params.foam_threshold;
    data.fft_ocean_size = resolvedDomain(params.fft_ocean_size);
    data.fft_choppiness = params.fft_choppiness;
    data.fft_wind_speed = params.wind_speed;
    data.fft_wind_direction = params.wind_direction;
    data.fft_amplitude = params.fft_amplitude;
    data.fft_time_scale = params.animation_speed;
    data.micro_anim_speed = params.micro_anim_speed;
    data.micro_morph_speed = params.micro_morph_speed;
    data.foam_noise_scale = params.foam_noise_scale;
}

} // namespace WaterShader
