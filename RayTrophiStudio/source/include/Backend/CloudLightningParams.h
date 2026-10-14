#pragma once

#include "Atmosphere/AtmosphereClouds.h"
#include "Atmosphere/LightningStrike.h"
#include "Backend/CloudBoltGeometry.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace Backend {

constexpr int kCloudLightningEvents = 4;

struct CloudLightningGPU {
    float start[4];   // scene xyz, pulse radiance
    float end[4];     // scene xyz, bolt radius m
    float color[4];   // linear RGB, channel segment count (CloudBoltSegmentGPU)
    float stage[4];   // growth 0..1.4 (connection at 1), channel brightness, kind (0 ground / 1 cloud horizontal /
                      // 2 upward / 3 cloud vertical), glow radius m
};
static_assert(sizeof(CloudLightningGPU) == 64, "CloudLightningGPU std430 layout");

// The cloud base lit from inside by one event (shaders/cloud_lightning_light.glsl):
// a downward-facing disc that lights the ground, surfaces and the rain
// curtain. Built here once so RT and RayFusion (which has no cloud field
// bound) light the scene from the same emitter.
struct CloudFlashLightGPU {
    float position[4];   // scene xyz on the cloud base, disc radius m (0 = off)
    float radiance[4];   // linear RGB radiance leaving the base, event index
};
static_assert(sizeof(CloudFlashLightGPU) == 32, "CloudFlashLightGPU std430 layout");

// Same formula as cloudFlashProfile() in cloud_lightning_light.glsl.
inline float cloudFlashProfile(float d, float r) {
    const float r2 = r * r;
    const float t = std::clamp((d - 2.5f * r) / (1.5f * r), 0.0f, 1.0f);
    return r2 / (r2 + d * d) * (1.0f - t * t * (3.0f - 2.0f * t));
}

// Diffusion attenuation of the flash through cloud: multiply-scattered light
// falls ~1/(1 + k tau) with k ~ 0.75 (1 - g), g ~ 0.85 for droplets. Shared
// with cloudLightningSource() so the lit base matches the lit interior.
constexpr float kCloudFlashDiffusion = 0.11f;

inline float lightningEffectiveRate(const atmosphere::CloudLightning& lightning) {
    if (lightning.rate_per_minute <= 0.0f) {
        return 0.0f;
    }
    const float period = (std::max)(240.0f / lightning.rate_per_minute,
                                   lightning.duration_s * 1.5f);
    return 240.0f / period;
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

// Inverse of lightningSceneHeight: the altitude of a scene point.
inline float lightningAltitudeOf(float x, float y, float z, float planetRadius,
                                 float altitudeOffset) {
    const double cy = static_cast<double>(y) + planetRadius + altitudeOffset;
    return static_cast<float>(std::sqrt(static_cast<double>(x) * x + cy * cy +
                                        static_cast<double>(z) * z) - planetRadius);
}

// The storm column's span (ATMOSPHERE_CELL_CLOUDS.md §1; CloudParams.h
// packStormColumn): from the lowest enabled layer's base (the LCL) to the
// tropopause. Returns the host layer, or -1 when there is no column.
// The levels themselves are atmosphere::columnLevels (one definition).
inline int stormColumnSpan(const atmosphere::CloudState& c, float& base, float& tropopause) {
    if (!c.storms.enabled || !c.storms.column) return -1;
    const atmosphere::ColumnLevels lv = atmosphere::columnLevels(c);
    if (!lv.valid) return -1;
    base = lv.lcl_m;
    tropopause = lv.tropopause_m;
    return lv.host_layer;
}

// ── Channel network ─────────────────────────────────────────────────────────
// ONE geometry per event, built here and read by every consumer: the drawn
// bolt (cloudApplyLightning), the in-cloud flash source (cloudLightningSource)
// and the lit-base disc below. The shader used to hash its own 24-point bolt
// while the cloud was lit from one straight start->end tube, so the light
// never followed the branches that were drawn.
// Geometry and parent-dependent growth are defined in CloudBoltGeometry.h.

// Stepped leader, then return stroke, then re-strokes; instantaneous value
// at `age`. The leader is faint and grows the channel; the flash peaks when
// it connects (life = leaderFraction).
inline float lightningPulse(float age, float duration, float leaderFraction) {
    if (age < 0.0f || age >= duration) {
        return 0.0f;
    }
    const float life = age / duration;
    if (life < leaderFraction) {
        return 0.08f + 0.14f * (life / leaderFraction);
    }
    const float stroke = (life - leaderFraction) / (1.0f - leaderFraction);
    const float strokeAge = age - leaderFraction * duration;
    const float envelope = std::exp(-stroke * 3.0f) * (std::min)((1.0f - life) * 8.0f, 1.0f);
    return envelope * (0.35f + 0.65f * std::pow(0.5f + 0.5f * std::cos(strokeAge * 85.0f), 4.0f));
}

inline void makeCloudLightningGPU(const atmosphere::CloudState& clouds, float time,
                                 const Vec3& wind_mps, float planetRadius, float altitudeOffset,
                                 CloudLightningGPU (&events)[kCloudLightningEvents],
                                 CloudFlashLightGPU (&flash)[kCloudLightningEvents],
                                 CloudBoltSegmentGPU (&bolt)[kCloudLightningEvents * kCloudBoltSegments]) {
    const auto& lightning = clouds.lightning;
    // Flashes come only from mature storm cells (ATMOSPHERE_STORMS.md §3):
    // no cells, no lightning.
    if (!lightning.enabled || lightning.rate_per_minute <= 0.0f ||
        lightning.intensity <= 0.0f || !std::isfinite(time) || !clouds.storms.enabled) {
        return;
    }
    int source = -1;
    float score = 0.0f;
    for (int i = 0; i < atmosphere::kMaxCloudLayers; ++i) {
        const auto& layer = clouds.layers[i];
        const auto material = atmosphere::layerMaterial(clouds, i);
        // Storm cells raise the genus locally, so any enabled layer can host
        // them; the deepest, most convective one is the flash's layer.
        const float candidate = (std::max)(layer.coverage, 0.05f) *
                                (std::max)(layer.type, 0.1f) * layer.thickness_m;
        if (layer.enabled && layer.extinction_per_m > 0.0f &&
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
    // In a storm column the flashes live in its charge regions, not inside
    // the (often thin) host layer: main negative charge ~30-45 % up the
    // column, upper positive under the anvil, a lower positive pocket near
    // the base.
    float columnBase = 0.0f, columnTop = 0.0f;
    const bool column = stormColumnSpan(clouds, columnBase, columnTop) >= 0;
    const float spanBase = column ? columnBase : layer.base_altitude_m;
    const float spanDepth = column ? columnTop - columnBase : layer.thickness_m;
    const float windLength = std::sqrt(wind_mps.x * wind_mps.x + wind_mps.z * wind_mps.z);
    const auto sourceMaterial = atmosphere::layerMaterial(clouds, source);
    const float layerExtinction = layer.extinction_per_m * sourceMaterial.density_multiplier *
        (sourceMaterial.scattering_coefficient + sourceMaterial.absorption_coefficient);
    // Four staggered lanes preserve long flashes instead of replacing an old
    // event in the middle of its growth. Duration bounds the effective rate.
    const double period = 240.0 / lightningEffectiveRate(lightning);
    for (int i = 0; i < kCloudLightningEvents; ++i) {
        atmosphere::publishLightningStrike(i, {});
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
        const float leaderFraction = lightning.leader_fraction;
        const float age = static_cast<float>(static_cast<double>(time) - onset);
        // Camera exposure: a frame integrates the light over its shutter, so
        // the pulse is averaged over [age - shutter, age]. Instantaneous
        // sampling of the ~74 ms re-strokes at 30-60 fps aliased into random
        // strobing, and a stroke shorter than a frame could vanish.
        const float shutter = lightning.shutter_s;
        if (age < 0.0f || age - shutter >= duration) {
            continue;
        }
        float pulse = 0.0f;
        if (shutter > 0.0f) {
            constexpr int kShutterSamples = 8;
            for (int k = 0; k < kShutterSamples; ++k)
                pulse += lightningPulse(age - shutter * (float(k) + 0.5f) / kShutterSamples,
                                        duration, leaderFraction);
            pulse /= kShutterSamples;
        } else {
            pulse = lightningPulse(age, duration, leaderFraction);
        }
        if (pulse <= 0.0f) {
            continue;
        }
        const float life = (std::min)(age / duration, 1.0f);
        // The cell is chosen from the cells alive at the flash ONSET, so the
        // choice cannot change during the flash; weight = the cell's
        // lightning potential (mature, strong cells flash most).
        atmosphere::StormCellState cells[atmosphere::kMaxStormCells];
        const int cellCount = atmosphere::evaluateStormCells(
            clouds, static_cast<float>(onset), wind_mps, cells);
        float total = 0.0f;
        for (int k = 0; k < cellCount; ++k) total += cells[k].lightning;
        const float pick = lightningRandom(seed) * total;
        if (total <= 0.0f) {
            continue;
        }
        int chosen = 0;
        for (float acc = cells[0].lightning; chosen + 1 < cellCount && pick > acc;
             acc += cells[++chosen].lightning) {}
        const atmosphere::StormCellState& cell = cells[chosen];
        // Inside the core (0.4 r); the flash moves with its cell.
        const float coreAngle = lightningRandom(seed) * 6.28318530718f;
        const float coreRadius = std::sqrt(lightningRandom(seed)) * 0.4f * cell.radius_m;
        CloudLightningGPU& event = events[i];
        event.start[0] = cell.x + std::cos(coreAngle) * coreRadius + wind_mps.x * age;
        event.start[2] = cell.z + std::sin(coreAngle) * coreRadius + wind_mps.z * age;
        float startAltitude = spanBase + spanDepth *
            (column ? 0.3f + lightningRandom(seed) * 0.15f : 0.35f + lightningRandom(seed) * 0.3f);
        event.start[3] = lightning.intensity * pulse;
        const bool ground = lightningRandom(seed) < lightning.ground_strike_fraction;
        // In-cloud kind in a column (own hash, the draw sequence above is
        // unchanged): 1 vertical, negative -> upper positive (the most common
        // flash in a thunderstorm); 2 down to the base; 3 anvil crawler,
        // long and horizontal downwind. 0 = horizontal inside the layer.
        uint32_t kindSeed = lightningHash(seed ^ 0x68e31da4u);
        int icKind = 0;
        if (column && !ground) {
            const float k = lightningRandom(kindSeed);
            icKind = k < 0.5f ? 1 : (k < 0.7f ? 2 : 3);
        }
        if (icKind == 3) startAltitude = spanBase + spanDepth * (0.72f + 0.1f * lightningRandom(kindSeed));
        // Own hash: the existing draw sequence (and so every other flash)
        // stays the same when upward_fraction changes.
        uint32_t upwardSeed = lightningHash(seed ^ 0x2545f491u);
        const bool upward = ground && lightningRandom(upwardSeed) < lightning.upward_fraction;
        const float angle = lightningRandom(seed) * 6.28318530718f;
        float span = ground ? (std::min)(spanDepth * 0.18f, 2500.0f) :
                              lightning.horizontal_span_m * size;
        if (icKind == 1) span = cell.radius_m * (0.15f + 0.35f * lightningRandom(kindSeed));
        if (icKind == 2) span = cell.radius_m * (0.3f + 0.5f * lightningRandom(kindSeed));
        float dirX = std::cos(angle), dirZ = std::sin(angle);
        if (icKind == 3 && windLength > 0.5f) {   // the anvil spreads downwind
            dirX = wind_mps.x / windLength;
            dirZ = wind_mps.z / windLength;
        }
        event.end[0] = event.start[0] + dirX * span;
        event.end[2] = event.start[2] + dirZ * span;
        float endAltitude = ground ? lightning.ground_altitude_m :
            (std::clamp)(startAltitude + (lightningRandom(seed) - 0.5f) *
                         layer.thickness_m * 0.12f, layer.base_altitude_m,
                         layer.base_altitude_m + layer.thickness_m);
        if (icKind == 1) endAltitude = spanBase + spanDepth * (0.72f + 0.15f * lightningRandom(kindSeed));
        if (icKind == 2) endAltitude = spanBase + spanDepth * (0.01f + 0.05f * lightningRandom(kindSeed));
        if (icKind == 3) endAltitude = spanBase + spanDepth * (0.82f + 0.06f * lightningRandom(kindSeed));
        const bool verticalCloud = icKind == 1 || icKind == 2;
        if (!lightningSceneHeight(event.start[0], event.start[2], startAltitude,
                                  planetRadius, altitudeOffset, event.start[1]) ||
            !lightningSceneHeight(event.end[0], event.end[2], endAltitude,
                                  planetRadius, altitudeOffset, event.end[1])) {
            event = {};
            continue;
        }
        const uint32_t geometrySeed = lightningHash(seed ^ 0x5bd1e995u);
        // Attachment (Atmosphere/LightningStrike.h): the ground end is the
        // scene point the leader reaches first, fixed at onset -- it does not
        // drift with the wind like the cloud end. Striking distance from the
        // peak current (Eriksson: 10 I^0.65 m), I 10..60 kA per event: 45-140 m.
        if (ground && lightning.attach_to_scene) {
            const float gx = event.end[0] - wind_mps.x * age;
            const float gz = event.end[2] - wind_mps.z * age;
            float topY = 0.0f;
            if (lightningSceneHeight(gx, gz, spanBase, planetRadius, altitudeOffset, topY)) {
                uint32_t currentSeed = lightningHash(seed ^ 0x1f123bb5u);
                const float peakKiloAmps = 10.0f + 50.0f * lightningRandom(currentSeed);
                const float strikingDistance = 10.0f * std::pow(peakKiloAmps, 0.65f);
                atmosphere::LightningStrikeTarget strike;
                if (atmosphere::resolveLightningStrike(geometrySeed, gx, gz, topY,
                                                       strikingDistance, strike) &&
                    strike.y < topY) {
                    event.end[0] = strike.x;
                    event.end[1] = strike.y;
                    event.end[2] = strike.z;
                    endAltitude = lightningAltitudeOf(strike.x, strike.y, strike.z,
                                                      planetRadius, altitudeOffset);
                    atmosphere::publishLightningStrike(i, strike);
                }
            }
        }
        // Cloud base at the strike, scene y (the planet curves it away from
        // the flat y = base altitude): below it the channel is in clear air.
        float clearAirBelowY = -std::numeric_limits<float>::max();
        if (ground) {
            float baseSceneY = 0.0f;
            if (lightningSceneHeight(event.start[0], event.start[2], spanBase,
                                     planetRadius, altitudeOffset, baseSceneY)) {
                clearAirBelowY = baseSceneY;
            }
        }
        BoltP3 centroid;
        const BoltP3 cloudPoint{event.start[0], event.start[1], event.start[2]};
        const BoltP3 groundPoint{event.end[0], event.end[1], event.end[2]};
        const int segments = buildCloudBolt(
            upward ? groundPoint : cloudPoint, upward ? cloudPoint : groundPoint,
            ground, upward, verticalCloud, lightning.branching,
            lightning.horizontal_span_m * size * 0.5f, clearAirBelowY, geometrySeed,
            &bolt[i * kCloudBoltSegments], centroid);
        event.end[3] = lightning.bolt_radius_m;
        event.color[0] = lightning.color.x;
        event.color[1] = lightning.color.y;
        event.color[2] = lightning.color.z;
        event.color[3] = static_cast<float>(segments);
        event.stage[0] = (std::min)(life / leaderFraction, 1.4f);
        // Channel core radiance relative to the cloud flash.
        event.stage[1] = lightning.channel_brightness;
        event.stage[2] = upward ? 2.0f : (ground ? 0.0f : (verticalCloud ? 3.0f : 1.0f));
        const float length = std::sqrt(span * span +
            (event.end[1] - event.start[1]) * (event.end[1] - event.start[1]));
        // In-cloud glow radius around each lit segment: the flash diffuses
        // over kilometres of cell.
        event.stage[3] = (std::clamp)(length * 0.25f, 800.0f, 4000.0f);

        // Lit base under the network. A ground strike crosses the base: the
        // channel itself is the source there. An in-cloud flash sits above
        // it, centred on its network; its light reaches the base through
        // `depth` metres of cloud.
        float cx, cz, depth;
        if (ground) {
            const float drop = (std::max)(startAltitude - endAltitude, 1.0f);
            const float u = (std::clamp)((startAltitude - spanBase) / drop, 0.0f, 1.0f);
            cx = event.start[0] + (event.end[0] - event.start[0]) * u;
            cz = event.start[2] + (event.end[2] - event.start[2]) * u;
            depth = 0.0f;
        } else {
            cx = centroid.x;
            cz = centroid.z;
            depth = (std::max)(0.5f * (startAltitude + endAltitude) - spanBase, 0.0f);
        }
        float baseY = 0.0f;
        CloudFlashLightGPU& light = flash[i];
        if (lightningSceneHeight(cx, cz, spanBase, planetRadius, altitudeOffset, baseY)) {
            // Mean coverage ~0.5 along the climb to the base.
            const float exitTau = layerExtinction * depth * 0.5f;
            const float exitScale = cloudFlashProfile(depth, event.stage[3]) /
                                    (1.0f + kCloudFlashDiffusion * exitTau);
            // The leader is too faint to light the landscape.
            const float connected = life < leaderFraction ? 0.0f : 1.0f;
            const float L = event.start[3] * exitScale * connected;
            // A spread network lights a wider patch of base.
            float reach = event.stage[3];
            for (int k = 0; k < segments; ++k) {
                const auto& s = bolt[i * kCloudBoltSegments + k];
                reach = (std::max)(reach, std::hypot(s.b[0] - cx, s.b[2] - cz));
            }
            light.position[0] = cx;
            light.position[1] = baseY;
            light.position[2] = cz;
            light.position[3] = L > 0.0f ? (std::min)(reach, 3.0f * event.stage[3]) : 0.0f;
            light.radiance[0] = lightning.color.x * L;
            light.radiance[1] = lightning.color.y * L;
            light.radiance[2] = lightning.color.z * L;
            light.radiance[3] = static_cast<float>(i);
        }
    }
}

} // namespace Backend
