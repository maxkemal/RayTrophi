vec3 volumeEmissionAt(VkVolumeInstance vol, vec3 p, float density, float temperature) {
    if (vol.emission_mode <= 0) return vec3(0.0);
    if (vol.emission_mode == 1) {
        return volumeEmissionSource(vol.emission_color, 1.0, density, 1.0,
            vol.emission_intensity, vol._retired_cloud, false);
    }
    if (temperature <= 0.0) temperature = density;
    float lo = vol.source_type == 4 ? 0.0 : max(vol._ext_reserved[6], 0.0);
    float hi = vol.max_temperature > lo + 1.0 ? vol.max_temperature : lo + 1500.0;
    bool kelvinUnits = vol.volume_type == 4 && vol.source_type == 5;
    float kelvin = (kelvinUnits || temperature > 20.0)
        ? clamp(temperature, lo, hi) : mix(lo, hi, clamp(temperature, 0.0, 1.0));
    float u = clamp((kelvin - lo) / max(hi - lo, 1.0), 0.0, 1.0);
    bool ramp = vol.color_ramp_enabled != 0 && vol.ramp_stop_count > 0;
    vec3 color = ramp ? sampleColorRamp(vol, clamp(u * vol.temperature_scale, 0.0, 1.0))
                      : blackbodyToRGB(kelvin * vol.temperature_scale);
    float radiance = ramp ? 1.0 : pow(clamp(kelvin / hi, 0.0, 1.0), 4.0);
    vec3 emission = volumeEmissionSource(color, radiance, density, u,
        vol.blackbody_intensity, vol._retired_cloud, true);
    return clampVolumeRadiance(emission * (1.0 + sampleFlame(vol, p)), 64.0);
}

vec3 volumeEmissionAt(VkVolumeInstance vol, vec3 p, float density) {
    return volumeEmissionAt(vol, p, density, sampleTemperature(vol, p));
}

#define INTERNAL_VOLUME_TYPE VkVolumeInstance
#define INTERNAL_DENSITY(v, p) sampleDensity(v, p)
#define INTERNAL_EMISSION(v, p, d) volumeEmissionAt(v, p, d)
#include "volume_internal_integral.glsl"
#undef INTERNAL_VOLUME_TYPE
#undef INTERNAL_DENSITY
#undef INTERNAL_EMISSION

vec3 volumeInternalLighting(VkVolumeInstance vol, vec3 p, vec3 viewDir) {
    int steps = clamp(vol._ramp_pad[0], 0, 32);
    if (steps == 0 || vol.emission_mode == 0) return vec3(0.0);
    float g = rnd(payload.seed) < vol.scatter_lobe_mix
        ? vol.scatter_anisotropy : vol.scatter_anisotropy_back;
    vec3 direction = sampleHG(viewDir, g, payload.seed);
    float distance = volumeExitDistance(vol, p, direction);
    if (distance <= 1e-4) return vec3(0.0);
    float solidDistance = rwSolidDistance(p, direction, distance);
    if (solidDistance >= 0.0) distance = min(distance, solidDistance);
    vec3 absorption = max(vol.absorption_color, vec3(0.0));
    float peak = max(absorption.r, max(absorption.g, absorption.b));
    vec3 weights = peak > 1e-5 ? absorption / peak : vec3(1.0);
    vec3 extinction = vec3(max(vol.scatter_coefficient, 0.0)) +
                      weights * max(vol.absorption_coefficient, 0.0);
    // Sampled from the dual-HG target phase, so phase/pdf is one.
    return volumeInternalRay(vol, p, direction, distance, steps, extinction);
}

vec3 volumeCachedInternalLighting(VkVolumeInstance vol, vec3 p, vec3 viewDir,
                                  bool supported, int step, inout vec3 cached,
                                  inout bool valid) {
    if (!supported || vol._ramp_pad[0] <= 0 || vol.emission_mode <= 0) return vec3(0.0);
    if (!valid || (step % max(vol._ramp_pad[1], 1)) == 0) {
        cached = volumeInternalLighting(vol, p, viewDir);
        valid = true;
    }
    return cached;
}
