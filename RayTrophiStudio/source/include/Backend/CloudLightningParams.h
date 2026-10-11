#pragma once

#include "Atmosphere/AtmosphereClouds.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace Backend {

constexpr int kCloudLightningEvents = 4;

struct CloudLightningGPU {
    float start[4];   // scene xyz, pulse radiance
    float end[4];     // scene xyz, bolt radius m
    float color[4];   // linear RGB, deterministic geometry seed
    float stage[4];   // growth 0..1, branching, internal (0 ground/1 cloud), glow radius m
};
static_assert(sizeof(CloudLightningGPU) == 64, "CloudLightningGPU std430 layout");

inline float lightningEffectiveRate(const atmosphere::CloudLightning& lightning) {
    if (lightning.rate_per_minute <= 0.0f) {
        return 0.0f;
    }
    const float period = (std::max)(240.0f / lightning.rate_per_minute,
                                   lightning.duration_s * 1.5f);
    return 240.0f / period;
}

inline uint32_t lightningHash(uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

inline float lightningRandom(uint32_t& seed) {
    seed = lightningHash(seed + 0x9e3779b9u);
    return static_cast<float>(seed & 0x00ffffffu) / 16777216.0f;
}

inline bool lightningSceneHeight(float x, float z, float altitude, float planetRadius,
                                 float altitudeOffset, float& y) {
    const double radius = static_cast<double>(planetRadius) + altitude;
    const double support = radius * radius - static_cast<double>(x) * x -
                           static_cast<double>(z) * z;
    if (radius <= 0.0 || support < 0.0) {
        return false;
    }
    y = static_cast<float>(std::sqrt(support) - planetRadius - altitudeOffset);
    return std::isfinite(y);
}

inline void makeCloudLightningGPU(const atmosphere::CloudState& clouds, float time,
                                 float planetRadius, float altitudeOffset,
                                 CloudLightningGPU (&events)[kCloudLightningEvents]) {
    const auto& lightning = clouds.lightning;
    if (!lightning.enabled || lightning.rate_per_minute <= 0.0f ||
        lightning.intensity <= 0.0f || !std::isfinite(time)) {
        return;
    }
    int source = -1;
    float score = 0.0f;
    for (int i = 0; i < atmosphere::kMaxCloudLayers; ++i) {
        const auto& layer = clouds.layers[i];
        const auto material = atmosphere::layerMaterial(clouds, i);
        const float candidate = layer.coverage * layer.type * layer.thickness_m;
        if (layer.enabled && layer.type >= 0.6f && layer.extinction_per_m > 0.0f &&
            material.density_multiplier > 0.0f &&
            material.scattering_coefficient + material.absorption_coefficient > 0.0f &&
            candidate > score) {
            source = i;
            score = candidate;
        }
    }
    if (source < 0) {
        return;
    }
    const auto& layer = clouds.layers[source];
    // Four staggered lanes preserve long flashes instead of replacing an old
    // event in the middle of its growth. Duration bounds the effective rate.
    const double period = 240.0 / lightningEffectiveRate(lightning);
    for (int i = 0; i < kCloudLightningEvents; ++i) {
        const double offset = period * static_cast<double>(i) / kCloudLightningEvents;
        const double eventSlot = std::floor((static_cast<double>(time) - offset) / period);
        // Hash wraps intentionally without overflowing signed conversion.
        const double wrapped = eventSlot - std::floor(eventSlot / 4294967296.0) * 4294967296.0;
        uint32_t seed = lightningHash(static_cast<uint32_t>(wrapped) ^ clouds.weather.seed ^
                                     lightningHash(static_cast<uint32_t>(i + 1)));
        const double onset = offset + eventSlot * period +
                             period * (0.03 + 0.07 * lightningRandom(seed));
        const float size = 0.65f + lightningRandom(seed) * 0.70f;
        const float duration = lightning.duration_s * size;
        const float age = static_cast<float>(static_cast<double>(time) - onset);
        if (age < 0.0f || age >= duration) {
            continue;
        }
        const float life = age / duration;
        const float envelope = (std::min)(age / 0.012f, 1.0f) *
                               std::exp(-life * 2.8f) * (std::min)((1.0f - life) * 8.0f, 1.0f);
        const float pulses = 0.3f + 0.7f * std::pow(
            0.5f + 0.5f * std::cos(age * 85.0f), 4.0f);
        CloudLightningGPU& event = events[i];
        event.start[0] = lightning.center_m.x + (lightningRandom(seed) - 0.5f) * lightning.extent_m;
        event.start[2] = lightning.center_m.z + (lightningRandom(seed) - 0.5f) * lightning.extent_m;
        const float startAltitude = layer.base_altitude_m + layer.thickness_m *
            (0.35f + lightningRandom(seed) * 0.3f);
        event.start[3] = lightning.intensity * envelope * pulses;
        const bool ground = lightningRandom(seed) < lightning.ground_strike_fraction;
        const float angle = lightningRandom(seed) * 6.28318530718f;
        const float span = ground ? layer.thickness_m * 0.18f :
                                   lightning.horizontal_span_m * size;
        event.end[0] = event.start[0] + std::cos(angle) * span;
        event.end[2] = event.start[2] + std::sin(angle) * span;
        const float endAltitude = ground ? lightning.ground_altitude_m :
            (std::clamp)(startAltitude + (lightningRandom(seed) - 0.5f) *
                         layer.thickness_m * 0.12f, layer.base_altitude_m,
                         layer.base_altitude_m + layer.thickness_m);
        if (!lightningSceneHeight(event.start[0], event.start[2], startAltitude,
                                  planetRadius, altitudeOffset, event.start[1]) ||
            !lightningSceneHeight(event.end[0], event.end[2], endAltitude,
                                  planetRadius, altitudeOffset, event.end[1])) {
            event = {};
            continue;
        }
        event.end[3] = lightning.bolt_radius_m;
        event.color[0] = lightning.color.x;
        event.color[1] = lightning.color.y;
        event.color[2] = lightning.color.z;
        event.color[3] = lightningRandom(seed) * 4096.0f;
        event.stage[0] = (std::min)(life / 0.45f, 1.0f);
        event.stage[1] = lightning.branching;
        event.stage[2] = ground ? 0.0f : 1.0f;
        const float length = std::sqrt(span * span +
            (event.end[1] - event.start[1]) * (event.end[1] - event.start[1]));
        event.stage[3] = (std::clamp)(length * 0.12f, 350.0f, 2500.0f);
    }
}

} // namespace Backend
