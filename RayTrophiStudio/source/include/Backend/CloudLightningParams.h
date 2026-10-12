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
    float color[4];   // linear RGB, channel segment count (CloudBoltSegmentGPU)
    float stage[4];   // growth 0..1, channel brightness, kind (0 ground / 1 cloud horizontal /
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

// The storm column's span (ATMOSPHERE_CELL_CLOUDS.md §1; CloudParams.h
// packStormColumn): from the lowest enabled layer's base (the LCL) to the
// tropopause. Returns the host layer, or -1 when there is no column.
inline int stormColumnSpan(const atmosphere::CloudState& c, float& base, float& tropopause) {
    if (!c.storms.enabled || !c.storms.column) return -1;
    int host = -1;
    for (int i = 0; i < atmosphere::kMaxCloudLayers; ++i) {
        const atmosphere::CloudLayer& l = c.layers[i];
        if (!l.enabled || l.extinction_per_m <= 0.0f) continue;
        if (host < 0 || l.base_altitude_m < c.layers[host].base_altitude_m) host = i;
    }
    if (host < 0) return -1;
    base = c.layers[host].base_altitude_m;
    tropopause = (std::max)(c.storms.column_top_m, base + 1500.0f);
    return host;
}

// ── Channel network ─────────────────────────────────────────────────────────
// ONE geometry per event, built here and read by every consumer: the drawn
// bolt (cloudApplyLightning), the in-cloud flash source (cloudLightningSource)
// and the lit-base disc below. The shader used to hash its own 24-point bolt
// while the cloud was lit from one straight start->end tube, so the light
// never followed the branches that were drawn.
constexpr int kCloudBoltSegments = 40;   // per event
constexpr int kCloudBoltMainSegments = 16;

struct CloudBoltSegmentGPU {
    float a[4];   // scene xyz, weight (1 = main channel, <1 branch / twig)
    float b[4];   // scene xyz, leader progress (0..1) at which it starts to grow
};
static_assert(sizeof(CloudBoltSegmentGPU) == 32, "CloudBoltSegmentGPU std430 layout");

struct BoltP3 { float x, y, z; };
inline BoltP3 boltAdd(BoltP3 a, BoltP3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline BoltP3 boltSub(BoltP3 a, BoltP3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline BoltP3 boltMul(BoltP3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float boltDot(BoltP3 a, BoltP3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float boltLen(BoltP3 a) { return std::sqrt(boltDot(a, a)); }
inline BoltP3 boltNorm(BoltP3 a) {
    const float l = boltLen(a);
    return l > 1e-6f ? boltMul(a, 1.0f / l) : BoltP3{0.0f, -1.0f, 0.0f};
}

// Midpoint displacement, `levels` octaves (2^levels segments). Each octave
// halves the amplitude: the large zig-zag and the fine crackle of a real
// channel, instead of one jitter scale. `verticalScale` < 1 keeps an in-cloud
// channel inside its layer.
inline void boltDisplacedPath(BoltP3 a, BoltP3 b, int levels, float amplitude,
                              float verticalScale, uint32_t& seed, BoltP3* out) {
    const int n = 1 << levels;
    out[0] = a;
    out[n] = b;
    const BoltP3 dir = boltNorm(boltSub(b, a));
    float amp = amplitude * boltLen(boltSub(b, a));
    for (int step = n; step > 1; step >>= 1) {
        for (int k = 0; k < n; k += step) {
            BoltP3 r{lightningRandom(seed) - 0.5f, (lightningRandom(seed) - 0.5f) * verticalScale,
                     lightningRandom(seed) - 0.5f};
            r = boltSub(r, boltMul(dir, boltDot(r, dir)));
            out[k + step / 2] = boltAdd(boltMul(boltAdd(out[k], out[k + step]), 0.5f),
                                        boltMul(r, 2.0f * amp));
        }
        amp *= 0.5f;
    }
}

// Writes up to kCloudBoltSegments segments; returns the count. The leader
// grows from `start` to `end` and its forks open along that direction: a
// downward strike (cloud -> ground) is a tree hanging from the cloud, an
// `upward` one (ground -> cloud, from a peak or a tower) a tree standing on
// the ground whose crown spreads into the cloud.
// `vertical`: an in-cloud channel between the charge regions of a storm
// column (up to the anvil, or down to the base): it zig-zags in 3D like a
// ground strike, forks along its way and spreads a spider at both ends.
inline int buildCloudBolt(BoltP3 start, BoltP3 end, bool ground, bool upward, bool vertical,
                          float branching,
                          float spiderSpan, uint32_t seed, CloudBoltSegmentGPU* out,
                          BoltP3& centroid) {
    int count = 0;
    float weightSum = 0.0f;
    centroid = {0.0f, 0.0f, 0.0f};
    auto emit = [&](BoltP3 a, BoltP3 b, float weight, float appear) {
        if (count >= kCloudBoltSegments) return;
        CloudBoltSegmentGPU& s = out[count++];
        s.a[0] = a.x; s.a[1] = a.y; s.a[2] = a.z; s.a[3] = weight;
        // The leader progress saturates at 1 when it connects; a late twig
        // must still start before that or it would never grow.
        s.b[0] = b.x; s.b[1] = b.y; s.b[2] = b.z; s.b[3] = (std::min)(appear, 0.94f);
        centroid = boltAdd(centroid, boltMul(boltAdd(a, b), 0.5f * weight));
        weightSum += weight;
    };
    const float verticalScale = (ground || vertical) ? 1.0f : 0.3f;
    BoltP3 main[kCloudBoltMainSegments + 1];
    boltDisplacedPath(start, end, 4, 0.09f, verticalScale, seed, main);
    for (int k = 0; k < kCloudBoltMainSegments; ++k)
        emit(main[k], main[k + 1], 1.0f, float(k) / kCloudBoltMainSegments);

    const BoltP3 mainDir = boltNorm(boltSub(end, start));
    const float mainLen = boltLen(boltSub(end, start));
    BoltP3 side = boltNorm(BoltP3{-mainDir.z, 0.0f, mainDir.x});
    if (boltLen(BoltP3{mainDir.x, 0.0f, mainDir.z}) < 1e-3f) side = {1.0f, 0.0f, 0.0f};

    // Branches: a ground strike forks along its leader's direction and
    // spreads a horizontal "spider" network inside the cloud from its cloud
    // end (that network is what lights the cell in patches); an in-cloud
    // flash is all spider, horizontal and long.
    struct Root { int index; bool spider; };
    Root roots[6];
    int nRoots = 0;
    if (ground) {
        // The cloud end: the first point of a downward leader, the last one
        // of an upward leader (its spider grows once the channel arrives).
        const int top = upward ? kCloudBoltMainSegments : 0;
        const int next = upward ? kCloudBoltMainSegments - 1 : 1;
        roots[nRoots++] = {top, true};
        if (branching > 0.3f) roots[nRoots++] = {next, true};
    } else if (vertical) {
        // Both charge regions spread horizontally.
        roots[nRoots++] = {0, true};
        roots[nRoots++] = {kCloudBoltMainSegments, true};
    }
    const int forks = static_cast<int>(branching * 4.0f + 0.5f);
    for (int f = 0; f < forks && nRoots < 6; ++f)
        roots[nRoots++] = {2 + static_cast<int>(lightningRandom(seed) * 11.0f), !ground && !vertical};
    for (int r = 0; r < nRoots; ++r) {
        const Root root = roots[r];
        const BoltP3 origin = main[root.index];
        const float sign = lightningRandom(seed) < 0.5f ? -1.0f : 1.0f;
        BoltP3 dir;
        float len;
        if (root.spider) {
            const float a = lightningRandom(seed) * 6.28318530718f;
            dir = boltNorm(BoltP3{std::cos(a), (lightningRandom(seed) - 0.5f) * 0.1f, std::sin(a)});
            len = spiderSpan * (0.25f + 0.25f * lightningRandom(seed));
        } else {
            dir = boltNorm(boltAdd(boltMul(mainDir, 0.65f),
                                   boltMul(side, sign * (0.45f + 0.4f * lightningRandom(seed)))));
            len = mainLen * (0.12f + 0.16f * lightningRandom(seed));
        }
        BoltP3 path[5];
        boltDisplacedPath(origin, boltAdd(origin, boltMul(dir, len)), 2, 0.14f,
                          root.spider ? 0.3f : 1.0f, seed, path);
        const float rootAppear = float(root.index) / kCloudBoltMainSegments;
        const float weight = root.spider ? 0.6f : 0.5f;
        for (int k = 0; k < 4; ++k)
            emit(path[k], path[k + 1], weight, rootAppear + 0.06f * float(k + 1));
        // A twig off the first two forks when branching is high.
        if (r < 2 && branching > 0.5f) {
            const BoltP3 tdir = boltNorm(boltAdd(dir, boltMul(side, -sign * 0.8f)));
            BoltP3 twig[3];
            boltDisplacedPath(path[2], boltAdd(path[2], boltMul(tdir, len * 0.4f)), 1, 0.15f,
                              root.spider ? 0.3f : 1.0f, seed, twig);
            for (int k = 0; k < 2; ++k)
                emit(twig[k], twig[k + 1], 0.3f, rootAppear + 0.18f + 0.06f * float(k + 1));
        }
    }
    if (weightSum > 0.0f) centroid = boltMul(centroid, 1.0f / weightSum);
    return count;
}

// Stepped leader, then return stroke, then re-strokes; instantaneous value
// at `age`. The leader is faint and grows the channel; the flash peaks when
// it connects (life = kCloudLeaderLife).
constexpr float kCloudLeaderLife = 0.12f;
inline float lightningPulse(float age, float duration) {
    if (age < 0.0f || age >= duration) return 0.0f;
    const float life = age / duration;
    if (life < kCloudLeaderLife) return 0.06f + 0.10f * (life / kCloudLeaderLife);
    const float stroke = (life - kCloudLeaderLife) / (1.0f - kCloudLeaderLife);
    const float strokeAge = age - kCloudLeaderLife * duration;
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
                pulse += lightningPulse(age - shutter * (float(k) + 0.5f) / kShutterSamples, duration);
            pulse /= kShutterSamples;
        } else {
            pulse = lightningPulse(age, duration);
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
        BoltP3 centroid;
        const BoltP3 cloudPoint{event.start[0], event.start[1], event.start[2]};
        const BoltP3 groundPoint{event.end[0], event.end[1], event.end[2]};
        const int segments = buildCloudBolt(
            upward ? groundPoint : cloudPoint, upward ? cloudPoint : groundPoint,
            ground, upward, verticalCloud, lightning.branching,
            lightning.horizontal_span_m * size * 0.5f, geometrySeed,
            &bolt[i * kCloudBoltSegments], centroid);
        event.end[3] = lightning.bolt_radius_m;
        event.color[0] = lightning.color.x;
        event.color[1] = lightning.color.y;
        event.color[2] = lightning.color.z;
        event.color[3] = static_cast<float>(segments);
        event.stage[0] = (std::min)(life / kCloudLeaderLife, 1.0f);
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
            const float connected = life < kCloudLeaderLife ? 0.0f : 1.0f;
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
