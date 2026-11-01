float volumeEmissionChannelsAcc(VkVolumeInstance vol, vec3 p, pnanovdb_buf_t buf,
    pnanovdb_map_handle_t mapH, inout pnanovdb_readaccessor_t acc,
    out float emissionDensity) {
    vol.density_multiplier = volumeEmissionSamplingMultiplier(
        vol.density_multiplier, vol._retired_cloud[0]);
    return sampleDensityAccChannels(vol, p, buf, mapH, acc, true, emissionDensity);
}

// Gas keeps the ramp's temperature gamma in the slot SDF surfaces use for their
// coordinate space. Zero means an instance that never packed one.
float volumeThermalGamma(VkVolumeInstance vol) {
    return vol.source_type == 4 || vol._accel_reserved[3] <= 0.0 ? 1.0 : vol._accel_reserved[3];
}

vec3 volumeEmissionAt(VkVolumeInstance vol, vec3 p, float density, float temperature) {
    if (vol.emission_mode <= 0) return vec3(0.0);
    float mask = volumeEmissionMaskFactor(density, volumeEmissionSamplingMultiplier(
        vol.density_multiplier, vol._retired_cloud[0]), vol.emission_mask);
    if (mask <= 0.0) {
        return vec3(0.0);
    }
    if (vol.emission_mode == 1) {
        return volumeEmissionSource(vol.emission_color, 1.0, density, 1.0,
            vol.emission_intensity, vol._retired_cloud, false, vol.density_multiplier) * mask;
    }
    float lo = vol.source_type == 4 ? 0.0 : max(vol._ext_reserved[6], 0.0);
    float hi = vol.max_temperature > lo + 1.0 ? vol.max_temperature : lo + 1500.0;
    int units = volumeResolveTemperatureUnits(int(vol.density_curve[10]),
        vol.volume_type == 4 && vol.source_type == 5);
    VolumeTemperatureSample thermal = volumeMapTemperature(temperature,
        volumeEmissionInputDensity(density, vol.density_multiplier, vol._retired_cloud[0]),
        lo, hi, units, vol.vdb_temp_address != uint64_t(0));
    if (!thermal.valid) {
        return vec3(0.0);
    }
    float kelvin = thermal.kelvin;
    float u = pow(thermal.normalized, volumeThermalGamma(vol));
    bool ramp = vol.color_ramp_enabled != 0 && vol.ramp_stop_count > 0;
    vec3 color = ramp ? sampleColorRamp(vol, clamp(u * vol.temperature_scale, 0.0, 1.0))
                      : blackbodyToRGB(kelvin * vol.temperature_scale);
    float radiance = ramp ? 1.0 : pow(clamp(kelvin / hi, 0.0, 1.0), 4.0);
    vec3 emission = volumeEmissionSource(color, radiance, density, u,
        vol.blackbody_intensity, vol._retired_cloud, true, vol.density_multiplier);
    emission *= mask;
    float flameGain = vol.density_curve[11];
    float flame = flameGain > 0.0 ? sampleFlame(vol, p) : 0.0;
    if (flame > 0.0) {
        // A burning cell is at least as hot on the ramp as its flame says.
        float hot = max(clamp(u * vol.temperature_scale, 0.0, 1.0), flame);
        emission += (ramp ? sampleColorRamp(vol, hot) : blackbodyToRGB(hi * hot)) * mask *
            volumeFlameRadiance(flame, flameGain, vol.blackbody_intensity, vol._retired_cloud);
    }
    return emission * volumeEmissionLimitScale(
        dot(emission, vec3(0.2126, 0.7152, 0.0722)), vol._retired_cloud[0]);
}

vec3 volumeEmissionAt(VkVolumeInstance vol, vec3 p, float density) {
    return volumeEmissionAt(vol, p, density, sampleTemperature(vol, p));
}

// Extinction multiplier from soot burnout. Same mapping and coordinate as the
// ramp above. Extinction-only shadow paths read no temperature and stay at 1.
float volumeSootScale(VkVolumeInstance vol, vec3 p, float density, float temperature) {
    if (vol.source_type == 4 || vol.emission_mode < 2 || vol._accel_reserved[0] <= 0.0 ||
        !volumeEmissionIndependent(vol._retired_cloud[0])) {
        return 1.0;
    }
    float lo = max(vol._ext_reserved[6], 0.0);
    float hi = vol.max_temperature > lo + 1.0 ? vol.max_temperature : lo + 1500.0;
    int units = volumeResolveTemperatureUnits(int(vol.density_curve[10]),
        vol.volume_type == 4 && vol.source_type == 5);
    VolumeTemperatureSample thermal = volumeMapTemperature(temperature,
        volumeEmissionInputDensity(density, vol.density_multiplier, vol._retired_cloud[0]),
        lo, hi, units, vol.vdb_temp_address != uint64_t(0));
    if (!thermal.valid) return 1.0;
    float heat = volumeSootExtinctionScale(
        clamp(pow(thermal.normalized, volumeThermalGamma(vol)) * vol.temperature_scale, 0.0, 1.0),
        vol._accel_reserved[0], vol._accel_reserved[1], vol._accel_reserved[2],
        vol._retired_cloud[0]);
    // Smoke clears where it burns, whatever the band says about the temperature.
    return min(heat, 1.0 - vol._accel_reserved[0] * sampleFlame(vol, p));
}

// Burnout thins the gas this light travels through too, so it agrees with the
// camera march. The temperature read exists only on materials that enable it.
float volumeInternalOptical(VkVolumeInstance vol, vec3 p, float density, float emissionDensity) {
    float optical = volumeEmissionOpticalDensity(density, vol.density_multiplier,
        vol._retired_cloud[0]);
    return vol._accel_reserved[0] > 0.0
        ? optical * volumeSootScale(vol, p, emissionDensity, sampleTemperature(vol, p))
        : optical;
}

#define INTERNAL_VOLUME_TYPE VkVolumeInstance
#define INTERNAL_DENSITY(v, p, e) sampleDensityChannels(v, p, e)
#define INTERNAL_OPTICAL(v, p, d, e) volumeInternalOptical(v, p, d, e)
#define INTERNAL_EMISSION(v, p, d) volumeEmissionAt(v, p, d)
#define INTERNAL_FLAGS(v) v._retired_cloud[0]
#define INTERNAL_VOXEL(v) v.voxel_size
#include "volume_internal_integral.glsl"
#undef INTERNAL_VOLUME_TYPE
#undef INTERNAL_DENSITY
#undef INTERNAL_OPTICAL
#undef INTERNAL_EMISSION
#undef INTERNAL_FLAGS
#undef INTERNAL_VOXEL

// Inline, because a nested trace here spills this shader's whole stack: one
// marched voxel still cost +140 ms at frame 69 of the nuke. Solids only: volume
// boxes stay unconfirmed candidates and alpha-tested geometry counts as opaque.
float volumeInternalSolidDistance(vec3 origin, vec3 direction, float tMax) {
    rayQueryEXT query;
    rayQueryInitializeEXT(query, topLevelAS,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0xF5, origin, 1e-4, direction, tMax);
    while (rayQueryProceedEXT(query)) {}
    if (rayQueryGetIntersectionTypeEXT(query, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) return -1.0;
    return rayQueryGetIntersectionTEXT(query, true);
}

vec3 volumeInternalLighting(VkVolumeInstance vol, vec3 p, vec3 viewDir) {
    int steps = clamp(vol._ramp_pad[0], 0, 32);
    if (steps == 0 || vol.emission_mode == 0) {
        return vec3(0.0);
    }
    float g = rnd(payload.seed) < vol.scatter_lobe_mix
        ? vol.scatter_anisotropy : vol.scatter_anisotropy_back;
    vec3 direction = sampleHG(viewDir, g, payload.seed);
    float distance = volumeExitDistance(vol, p, direction);
    if (distance <= 1e-4) return vec3(0.0);
    float solidDistance = volumeInternalSolidDistance(p, direction, distance);
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
