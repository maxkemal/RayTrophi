#pragma once

#include "Vec3.h"
#include <algorithm>

namespace WaterShader {

struct SurfaceParams {
    float wave_speed = 1.0f;
    float wave_strength = 0.5f;
    float wave_frequency = 1.0f;

    Vec3 shallow_color = Vec3(2.0f / 255.0f, 3.0f / 255.0f, 3.0f / 255.0f);
    Vec3 deep_color = Vec3(0.01f, 0.02f, 0.05f);
    Vec3 absorption_color = Vec3(0.3f, 0.6f, 0.7f);

    float depth_max = 15.0f;
    float absorption_density = 0.5f;
    float clarity = 0.8f;
    float ior = 1.333f;
    float roughness = 0.02f;

    float foam_level = 0.01f;
    float shore_foam_distance = 1.5f;
    float shore_foam_intensity = 0.6f;

    float caustic_intensity = 0.4f;
    float caustic_scale = 2.0f;
    float caustic_speed = 1.0f;

    float sss_intensity = 0.15f;
    Vec3 sss_color = Vec3(0.1f, 0.4f, 0.5f);

    bool use_fft_ocean = false;
    float fft_ocean_size = 100.0f;
    float fft_choppiness = 1.0f;
    float fft_amplitude = 0.001f;
    float animation_speed = 1.0f;

    float micro_detail_strength = 0.05f;
    float micro_detail_scale = 20.0f;
    float micro_anim_speed = 0.1f;
    float micro_morph_speed = 1.0f;
    float foam_noise_scale = 4.0f;
    float foam_threshold = 0.4f;

    float wind_direction = 0.0f;
    float wind_speed = 10.0f;
    float time = 0.0f;
};

struct SurfaceSample {
    Vec3 normal = Vec3(0.0f, 1.0f, 0.0f);
    float foam = 0.0f;
    float height = 0.0f;
    float depth = 0.0f;
    float shore_factor = 0.0f;
    float caustic_intensity = 0.0f;
    Vec3 water_color = Vec3(0.01f, 0.02f, 0.05f);
    Vec3 absorption = Vec3(1.0f);
};

inline float clamp01(float v) {
    return (std::max)(0.0f, (std::min)(1.0f, v));
}

inline float resolvedDomain(float authored_domain) {
    return (std::max)(authored_domain, 0.001f);
}

inline bool isActiveWater(float water_flag_strength, float transmission) {
    return water_flag_strength > 0.0001f && transmission > 0.1f;
}

} // namespace WaterShader
