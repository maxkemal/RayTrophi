#ifndef RT_VOLUME_EMISSION_TRANSPORT_H
#define RT_VOLUME_EMISSION_TRANSPORT_H
#include "VolumeEmissionMask.h"
#if defined(__cplusplus)
#include <cmath>
#endif

#if defined(__CUDACC__)
#define RT_EM_FN __host__ __device__ inline
#elif defined(__cplusplus)
#define RT_EM_FN inline
#else
#define RT_EM_FN
#endif

// response[0]: bit 0 = response curve enabled; bit 1 = independent coefficient j.
// Values 0/1 retain the saved-scene source-function convention.
RT_EM_FN bool volumeEmissionIndependent(float flags) {
    return (int(flags) & 2) != 0;
}

RT_EM_FN float volumeEmissionSamplingMultiplier(float multiplier, float flags) {
    return volumeEmissionIndependent(flags) && multiplier <= 0.0f ? 1.0f : multiplier;
}

RT_EM_FN float volumeEmissionOpticalDensity(float density, float multiplier, float flags) {
    return volumeEmissionIndependent(flags) && multiplier <= 0.0f ? 0.0f : density;
}

RT_EM_FN bool volumeEmissionSegmentActive(float sigma, float density, float flags) {
    return sigma > 0.00000001f || (volumeEmissionIndependent(flags) && density > 0.0f);
}

RT_EM_FN float volumeEmissionInputDensity(float density, float multiplier, float flags) {
    if (!volumeEmissionIndependent(flags)) return density;
    float effective = multiplier > 0.0f ? multiplier : 1.0f;
    return density / effective;
}

// Soot burnout: hot gas holds less smoke than cold gas, so a fireball must not
// sit under its own opacity. Multiplies extinction only; u is the same
// temperature coordinate the ramp reads. Independent j is required: in the
// source-function convention emission is sigma*S, so thinning sigma would dim
// the very flame this exists to reveal.
RT_EM_FN float volumeSootExtinctionScale(float u, float strength, float start, float end,
    float flags) {
    if (strength <= 0.0f || !volumeEmissionIndependent(flags)) return 1.0f;
    float span = end - start;
    float x = (u - start) / (span > 0.000001f ? span : 0.000001f);
    x = x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
    return 1.0f - strength * x * x * (3.0f - 2.0f * x);
}

// Thermal radiance limit, as a scale on a luminance. Independent j rolls off
// logarithmically above the knee instead of cutting: a hard ceiling gave every
// hot voxel the same power and left only hue. The slope is continuous at the
// knee, so a material below it is untouched, and order is preserved. The log
// still bounds a tiny hot cell enough to keep it from dominating indirect paths.
RT_EM_FN float volumeEmissionLimitScale(float luminance, float flags) {
    if (!(luminance > 64.0f)) return 1.0f;
    if (!volumeEmissionIndependent(flags)) return 64.0f / luminance;
#if defined(__cplusplus)
    return 64.0f * (1.0f + logf(luminance / 64.0f)) / luminance;
#else
    return 64.0f * (1.0f + log(luminance / 64.0f)) / luminance;
#endif
}

RT_EM_FN float volumeEmissionIntegral(float sigma, float dt, float flags) {
    float tau = sigma * dt;
#if defined(__cplusplus)
    if (!volumeEmissionIndependent(flags)) return 1.0f - expf(-tau);
    float absorbed = -expm1f(-tau);
#else
    float absorbed = 1.0f - exp(-tau);
#endif
    if (!volumeEmissionIndependent(flags)) return absorbed;
    // Stable vacuum limit: j*dt, including exactly zero extinction.
    if (tau < 0.001f) {
        return dt * (1.0f - tau * 0.5f + tau * tau / 6.0f);
    }
    return absorbed / sigma;
}

// Reaction emission, added to the thermal term rather than multiplied into it. A
// burning cell can sit where temperature or density leave the thermal base at
// zero (measured at frame 69 of the nuke: a 40x gain moved mean emission by
// .004%), and a gain over zero stays zero. Carries the same exposure as the
// thermal term, so EV keeps moving the whole fire together.
RT_EM_FN float volumeFlameRadiance(float flame, float gain, float strength,
    const float response[5]) {
    float scale = flame * gain * strength;
    if ((int(response[0]) & 1) == 0) return scale;
#if defined(__cplusplus)
    return scale * exp2f(response[3]);
#else
    return scale * exp2(response[3]);
#endif
}

RT_EM_FN float volumeEmissionChannel(float color, float luminance, float radiance,
    float density, float temperature, float strength, const float response[5],
    bool thermal) {
    if ((int(response[0]) & 1) == 0) {
        return color * radiance * density * strength;
    }
    if (density <= 0.0f) return 0.0f;
    float c = luminance + (color - luminance) * response[4];
    c = c > 0.0f ? c : 0.0f;
    float u = temperature < 0.0f ? 0.0f : (temperature > 1.0f ? 1.0f : temperature);
#if defined(__cplusplus)
    float heat = thermal ? (u > 0.0f ? powf(u, response[1]) : 0.0f) : 1.0f;
    return c * heat * powf(density, response[2]) * strength * exp2f(response[3]);
#else
    float heat = thermal ? (u > 0.0f ? pow(u, response[1]) : 0.0f) : 1.0f;
    return c * heat * pow(density, response[2]) * strength * exp2(response[3]);
#endif
}

#if defined(__cplusplus)
template <typename T>
RT_EM_FN T volumeEmissionFinish(T value, float mask, bool limit) {
    value.x *= mask;
    value.y *= mask;
    value.z *= mask;
    float luminance = value.x * 0.2126f + value.y * 0.7152f + value.z * 0.0722f;
    float scale = limit ? volumeEmissionLimitScale(luminance, 2.0f) : 1.0f;
    value.x *= scale;
    value.y *= scale;
    value.z *= scale;
    return value;
}

template <typename T>
RT_EM_FN T volumeEmissionBuild(T color, float radiance, float density,
    float temperature, float strength, const float response[5], bool thermal) {
    float luminance = color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f;
    T result = color;
    result.x = volumeEmissionChannel(color.x, luminance, radiance, density,
        temperature, strength, response, thermal);
    result.y = volumeEmissionChannel(color.y, luminance, radiance, density,
        temperature, strength, response, thermal);
    result.z = volumeEmissionChannel(color.z, luminance, radiance, density,
        temperature, strength, response, thermal);
    return result;
}

template <typename V, typename T>
RT_EM_FN T volumeEmissionOptInForVolume(const V& volume, T legacy, T color,
    float density, float temperature, float radiance, float strength, bool thermal) {
    if (!volumeEmissionIndependent(volume.emission_response[0])) return legacy;
    T coefficient = volumeEmissionBuild(color, radiance,
        volumeEmissionInputDensity(density, volume.density_multiplier, volume.emission_response[0]),
        temperature, strength, volume.emission_response, thermal);
    float mask = volumeEmissionMaskFactor(density, volumeEmissionSamplingMultiplier(
        volume.density_multiplier, volume.emission_response[0]), volume.emission_mask);
    return volumeEmissionFinish(coefficient, mask, thermal);
}

template <typename T>
RT_EM_FN T volumeEmissionIntegrate(T source, T emission, float sigma, float dt, float flags) {
    // Preserve the legacy operation when off; independent mode replaces only
    // emission's weight, never the scattering/transmittance equation.
    float oldWeight = volumeEmissionIntegral(sigma, dt, 0.0f);
    T result = source;
    if (!volumeEmissionIndependent(flags)) {
        result.x *= oldWeight;
        result.y *= oldWeight;
        result.z *= oldWeight;
        return result;
    }
    float weight = volumeEmissionIntegral(sigma, dt, flags);
    result.x = (source.x - emission.x) * oldWeight + emission.x * weight;
    result.y = (source.y - emission.y) * oldWeight + emission.y * weight;
    result.z = (source.z - emission.z) * oldWeight + emission.z * weight;
    return result;
}
#else
vec3 volumeEmissionIntegrate(vec3 source, vec3 emission, vec3 sigma, float dt, float flags) {
    vec3 oldWeight = vec3(1.0f) - exp(-sigma * dt);
    if (!volumeEmissionIndependent(flags)) return source * oldWeight;
    vec3 weight = vec3(volumeEmissionIntegral(sigma.x, dt, flags),
        volumeEmissionIntegral(sigma.y, dt, flags), volumeEmissionIntegral(sigma.z, dt, flags));
    return (source - emission) * oldWeight + emission * weight;
}

vec3 volumeEmissionCoefficient(vec3 emission, vec3 sigma, float flags) {
    return volumeEmissionIndependent(flags) ? emission : emission * sigma;
}
#endif

#undef RT_EM_FN
#endif
