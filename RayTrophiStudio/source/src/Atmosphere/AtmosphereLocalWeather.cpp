// Local weather: suitability, recent rain and local fog
// (docs/dev/ATMOSPHERE_LOCAL_WINDS_FOG.md §2-4). See the header for the
// contract. The pure functions come first; syncLocalWeather() at the end is
// the main-thread glue that copies the terrain and publishes the snapshots.
#include "Atmosphere/AtmosphereLocalWeather.h"

#include "Atmosphere/AtmosphereClimate.h"
#include "Atmosphere/AtmosphereClouds.h"
#include "TerrainManager.h"
#include "World.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

namespace atmosphere {

namespace {

float smooth01(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

std::mutex g_snapshotMutex;
std::shared_ptr<const LocalWeatherEnvironment> g_environment;
std::shared_ptr<const LocalFogMap> g_fogMap;

void publishEnvironment(std::shared_ptr<const LocalWeatherEnvironment> env) {
    std::lock_guard<std::mutex> lock(g_snapshotMutex);
    g_environment = std::move(env);
}

void publishFogMap(std::shared_ptr<const LocalFogMap> map) {
    std::lock_guard<std::mutex> lock(g_snapshotMutex);
    g_fogMap = std::move(map);
}

uint64_t fnv(uint64_t h, const void* data, size_t size) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < size; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

template <typename T>
uint64_t fnvValue(uint64_t h, const T& v) { return fnv(h, &v, sizeof(T)); }

constexpr uint64_t kFnvBasis = 1469598103934665603ull;

// Cover envelope of a cell (cloud_cells.comp cellCover).
float cellCoverEnvelope(float d) { return 1.0f - smooth01(0.55f, 1.15f, d); }

} // namespace

std::shared_ptr<const LocalWeatherEnvironment> localWeatherEnvironment() {
    std::lock_guard<std::mutex> lock(g_snapshotMutex);
    return g_environment;
}

std::shared_ptr<const LocalFogMap> localFogMapSnapshot() {
    std::lock_guard<std::mutex> lock(g_snapshotMutex);
    return g_fogMap;
}

// ── Terrain field ──────────────────────────────────────────────────────────

bool TerrainField::sample(float x, float z, float& height_out, float& mean_out) const {
    if (!valid || n <= 1 || size_m <= 0.0f) return false;
    const float fx = (x - origin_x) / size_m * static_cast<float>(n) - 0.5f;
    const float fz = (z - origin_z) / size_m * static_cast<float>(n) - 0.5f;
    if (fx < -0.5f || fz < -0.5f || fx > static_cast<float>(n) - 0.5f ||
        fz > static_cast<float>(n) - 0.5f) return false;
    const int nx = std::clamp(static_cast<int>(std::lround(fx)), 0, n - 1);
    const int nz = std::clamp(static_cast<int>(std::lround(fz)), 0, n - 1);
    if (!covered[static_cast<size_t>(nz) * n + nx]) return false;
    const float cx = std::clamp(fx, 0.0f, static_cast<float>(n - 1));
    const float cz = std::clamp(fz, 0.0f, static_cast<float>(n - 1));
    const int x0 = std::min(static_cast<int>(cx), n - 2);
    const int z0 = std::min(static_cast<int>(cz), n - 2);
    const float tx = cx - static_cast<float>(x0);
    const float tz = cz - static_cast<float>(z0);
    auto lerp2 = [&](const std::vector<float>& v) {
        const size_t i00 = static_cast<size_t>(z0) * n + x0;
        const float a = v[i00] * (1.0f - tx) + v[i00 + 1] * tx;
        const float b = v[i00 + n] * (1.0f - tx) + v[i00 + n + 1] * tx;
        return a * (1.0f - tz) + b * tz;
    };
    height_out = lerp2(height);
    mean_out = lerp2(local_mean);
    return true;
}

bool TerrainField::gradient(float x, float z, float& gx, float& gz) const {
    const float e = texel_m();
    float h0, h1, m;
    if (!sample(x + e, z, h1, m) || !sample(x - e, z, h0, m)) return false;
    gx = (h1 - h0) / (2.0f * e);
    if (!sample(x, z + e, h1, m) || !sample(x, z - e, h0, m)) return false;
    gz = (h1 - h0) / (2.0f * e);
    return true;
}

// ── §2 Suitability ─────────────────────────────────────────────────────────

SuitabilityTerms birthSuitability(const LocalWeatherEnvironment* env, float x, float z,
                                  const Vec3& wind_mps) {
    SuitabilityTerms s;
    if (!env || !env->terrain || !env->terrain->valid) return s;
    const TerrainField& f = *env->terrain;
    float h, mean, gx, gz;
    if (!f.sample(x, z, h, mean) || !f.gradient(x, z, gx, gz)) return s;
    s.terrain = true;
    const float speed = std::sqrt(wind_mps.x * wind_mps.x + wind_mps.z * wind_mps.z);
    // Upslope component of the wind: the slope the air is pushed up, weighted
    // by how hard it is pushed (8 m/s = full, saturating at 1.5x).
    float lift = 0.0f;
    if (speed > 0.1f)
        lift = (gx * wind_mps.x + gz * wind_mps.z) / speed * std::min(speed / 8.0f, 1.5f);
    s.orographic = smooth01(0.0f, 0.12f, lift);
    s.lee = smooth01(0.0f, 0.12f, -lift);
    s.ridge = smooth01(0.0f, 200.0f, h - mean);
    // Differential heating: a slope facing the sun gets more than flat
    // ground under the same sun (daytime only).
    const Vec3 sun = env->sun_direction;
    const float nLen = std::sqrt(gx * gx + 1.0f + gz * gz);
    const float facing = (-gx * sun.x + sun.y - gz * sun.z) / nLen;
    const float day = smooth01(0.05f, 0.4f, sun.y);
    s.heating = std::clamp((facing - sun.y) * 5.0f, 0.0f, 1.0f) * day;
    s.value = std::clamp(0.3f + 0.7f * s.orographic + 0.25f * s.ridge + 0.45f * s.heating -
                         0.25f * s.lee, 0.05f, 1.0f);
    return s;
}

// ── §3 Rain ────────────────────────────────────────────────────────────────

float cellRainAt(const StormCellState* cells, int count, float x, float z, const Vec3& wind_mps) {
    const float speed = std::sqrt(wind_mps.x * wind_mps.x + wind_mps.z * wind_mps.z);
    const float wx = speed > 1e-3f ? wind_mps.x / speed : 0.0f;
    const float wz = speed > 1e-3f ? wind_mps.z / speed : 0.0f;
    float rain = 0.0f;
    for (int i = 0; i < count; ++i) {
        const StormCellState& c = cells[i];
        const float r = std::max(c.radius_m, 1.0f);
        const float d = std::hypot(x - c.x, z - c.z) / r;
        // Heaviest rain on the forward flank: the core sits 0.3 r downwind.
        const float df = std::hypot(x - (c.x + wx * 0.3f * r), z - (c.z + wz * 0.3f * r)) / r;
        rain = std::max(rain, std::max(c.convective_rain * (1.0f - smooth01(0.0f, 0.55f, df)),
                                       c.stratiform_rain * (1.0f - smooth01(0.3f, 1.2f, d))));
    }
    return rain;
}

float recentRainAt(const RecentRainCells& past, float x, float z, const Vec3& wind_mps) {
    float sum = 0.0f, weight = 0.0f;
    for (int k = 0; k < kRecentRainSamples; ++k) {
        const float w = std::exp(-static_cast<float>(k) * kRecentRainStepS / kRecentRainDecayS);
        if (past.lists[k]) sum += w * cellRainAt(past.lists[k], past.counts[k], x, z, wind_mps);
        weight += w;
    }
    return weight > 0.0f ? sum / weight : 0.0f;
}

// ── §4 Fog ─────────────────────────────────────────────────────────────────

float localFogExtinctionAt(const LocalFogTexel& t, float y) {
    if (y < t.ground - 5.0f) return 0.0f;
    return t.fog * std::clamp((t.fog_top - y) / kLocalFogTopSoftM, 0.0f, 1.0f) +
           t.mist * std::exp(-std::max(y - t.ground, 0.0f) / kLocalMistScaleHeightM);
}

bool LocalFogMap::sample(float x, float z, LocalFogTexel& out) const {
    out = {};
    // Not gated on `active`: an inactive map holds zeros, and "inside the map,
    // no fog" is a different answer from "outside the map".
    if (n <= 1 || size_m <= 0.0f || texels.size() != static_cast<size_t>(n) * n) return false;
    // Texel centres, clamped: the linear filter of cloud_common.glsl cloudFogMapAt.
    const float u = (x - origin_x) / size_m;
    const float v = (z - origin_z) / size_m;
    if (u < 0.0f || v < 0.0f || u > 1.0f || v > 1.0f) return false;
    const float fx = std::clamp(u * static_cast<float>(n) - 0.5f, 0.0f, static_cast<float>(n - 1));
    const float fz = std::clamp(v * static_cast<float>(n) - 0.5f, 0.0f, static_cast<float>(n - 1));
    const int x0 = std::min(static_cast<int>(fx), n - 2);
    const int z0 = std::min(static_cast<int>(fz), n - 2);
    const float tx = fx - static_cast<float>(x0);
    const float tz = fz - static_cast<float>(z0);
    const LocalFogTexel& a = texels[static_cast<size_t>(z0) * n + x0];
    const LocalFogTexel& b = texels[static_cast<size_t>(z0) * n + x0 + 1];
    const LocalFogTexel& c = texels[static_cast<size_t>(z0 + 1) * n + x0];
    const LocalFogTexel& d = texels[static_cast<size_t>(z0 + 1) * n + x0 + 1];
    auto bl = [&](float LocalFogTexel::*f) {
        const float top = a.*f * (1.0f - tx) + b.*f * tx;
        const float bot = c.*f * (1.0f - tx) + d.*f * tx;
        return top * (1.0f - tz) + bot * tz;
    };
    out.fog = bl(&LocalFogTexel::fog);
    out.fog_top = bl(&LocalFogTexel::fog_top);
    out.mist = bl(&LocalFogTexel::mist);
    out.ground = bl(&LocalFogTexel::ground);
    return true;
}

namespace {

// Peak envelope extinctions at full potential (1/m); the march's shape noise
// is <= 1, so these are also the majorant. Radiation fog 0.012 = visibility
// ~330 m (3.9 / sigma) in a valley; mist 0.0015 at the ground after rain
// (~2.6 km), thinning upward over 250 m.
// ★ No rain-shaft mist: the precipitation curtain (cloud_rt.glsl) already
//   is the medium inside the shaft. A froxel-lit copy (0.0035 1/m up to the
//   cloud base) washed a raining valley in sunlit haze (2026-10-13).
constexpr float kRadiationFogPeak = 0.012f;
constexpr float kPostRainMistPeak = 0.0015f;
constexpr float kFogTimeQuantumS = 10.0f;

LocalFogDrivers fogDrivers(const World& world, const CloudState& clouds, const Vec3& sun) {
    LocalFogDrivers d;
    const ClimateState& climate = world.getClimate();
    // Radiative cooling persists into the early morning; by ~17 deg of sun
    // the fog has burnt off.
    d.burn_off = smooth01(-0.02f, 0.30f, sun.y);
    d.night = 1.0f - d.burn_off;
    d.calm = 1.0f - smooth01(1.0f, 4.0f, climate.wind_speed_mps);
    d.humidity = climate.surface_relative_humidity;
    const float altitude = world.getNishitaParams().altitude;
    float lowestBase = 1e9f;
    for (int i = 0; i < kMaxCloudLayers; ++i) {
        const CloudLayer& l = clouds.layers[i];
        if (!l.enabled || l.extinction_per_m <= 0.0f) continue;
        lowestBase = std::min(lowestBase, l.base_altitude_m);
        // Low and mid layers keep the ground from cooling; cirrus does not.
        if (l.base_altitude_m < 6000.0f) d.layer_cover = std::max(d.layer_cover, l.coverage);
    }
    d.cloud_base_y = (lowestBase < 1e8f ? lowestBase : 2000.0f) - altitude;
    return d;
}

struct FogPlacement {
    float origin_x = 0.0f, origin_z = 0.0f, size_m = 0.0f;
    bool terrain = false;
};

FogPlacement placeFogMap(const TerrainField* field, const CloudState& clouds, const Vec3& camera) {
    FogPlacement p;
    if (field && field->valid) {
        p.origin_x = field->origin_x;
        p.origin_z = field->origin_z;
        p.size_m = field->size_m;
        p.terrain = true;
    } else if (clouds.storms.enabled) {
        p.size_m = clouds.storms.extent_m + 4.0f * clouds.storms.cell_radius_m;
        p.origin_x = clouds.storms.center_m.x - 0.5f * p.size_m;
        p.origin_z = clouds.storms.center_m.z - 0.5f * p.size_m;
    } else {
        // Flat world, no storms: 40 km around the camera, re-centred in 4 km
        // steps so a moving camera does not rebuild every frame.
        p.size_m = 40000.0f;
        p.origin_x = std::floor(camera.x / 4000.0f) * 4000.0f - 0.5f * p.size_m;
        p.origin_z = std::floor(camera.z / 4000.0f) * 4000.0f - 0.5f * p.size_m;
    }
    return p;
}

// The 8 past cell lists recentRainAt reads, at the quantized time.
struct PastCells {
    StormCellState cells[kRecentRainSamples][kMaxStormCells];
    RecentRainCells view;
    int now = 0;
    void evaluate(const CloudState& clouds, float time_s, const Vec3& wind) {
        for (int k = 0; k < kRecentRainSamples; ++k) {
            view.counts[k] = evaluateStormCells(clouds, time_s - static_cast<float>(k) * kRecentRainStepS,
                                                wind, cells[k]);
            view.lists[k] = cells[k];
        }
        now = view.counts[0];
    }
};

LocalFogTexel fogTexel(const LocalFogDrivers& d, float strength, float ground, float mean,
                       bool on_terrain, float cover, float recent, float now) {
    LocalFogTexel t;
    t.ground = ground;
    // Hollows (below the ~1.5 km local mean) collect the cold air; on flat
    // ground without a terrain every place is average.
    const float hollow = on_terrain ? smooth01(15.0f, -60.0f, ground - mean) : 0.5f;
    const float fill = on_terrain ? std::clamp(mean - ground, 0.0f, 250.0f) : 0.0f;
    const float humid = smooth01(0.80f, 0.97f, d.humidity + 0.12f * recent);
    const float clear = 1.0f - smooth01(0.3f, 0.85f, cover);
    const float potential = d.night * d.calm * humid * clear;
    t.fog = strength * kRadiationFogPeak * potential * (0.25f + 0.75f * hollow);
    // The fog fills the hollow; in the morning its top lifts as it thins.
    t.fog_top = ground + (20.0f + 0.6f * fill) * (0.6f + 0.4f * hollow) + 60.0f * d.burn_off;
    // Mist: after the rain has stopped -- wet ground evaporating into humid
    // air. Where it still rains, the curtain is the medium.
    const float post = std::clamp(recent - now, 0.0f, 1.0f) *
                       smooth01(0.55f, 0.85f, d.humidity + 0.25f * recent);
    t.mist = strength * kPostRainMistPeak * post;
    return t;
}

std::shared_ptr<LocalFogMap> buildFogMap(const World& world, const LocalWeatherEnvironment& env,
                                         const FogPlacement& place, float time_q,
                                         const LocalFogDrivers& drivers) {
    auto map = std::make_shared<LocalFogMap>();
    const CloudState& clouds = world.getClouds();
    map->origin_x = place.origin_x;
    map->origin_z = place.origin_z;
    map->size_m = place.size_m;
    map->terrain = place.terrain;
    map->time_s = time_q;
    map->drivers = drivers;
    const int n = map->n;
    map->texels.assign(static_cast<size_t>(n) * n, LocalFogTexel{});
    if (!clouds.local_weather.fog_enabled || clouds.local_weather.fog_strength <= 0.0f) return map;

    const Vec3 wind = world.cloudWindVelocity();
    auto past = std::make_unique<PastCells>();
    const bool storms = clouds.storms.enabled;
    if (storms) past->evaluate(clouds, time_q, wind);
    const TerrainField* field = env.terrain.get();
    const float texel = place.size_m / static_cast<float>(n);

    // Cell cover, recent rain and rain now on the coarse grid (texel centres).
    const int g = kLocalRainGridSize;
    const float gTexel = place.size_m / static_cast<float>(g);
    std::vector<float> gCover(static_cast<size_t>(g) * g, 0.0f);
    std::vector<float> gRecent(gCover.size(), 0.0f), gNow(gCover.size(), 0.0f);
    if (storms) {
        for (int iz = 0; iz < g; ++iz) {
            for (int ix = 0; ix < g; ++ix) {
                const float x = place.origin_x + (static_cast<float>(ix) + 0.5f) * gTexel;
                const float z = place.origin_z + (static_cast<float>(iz) + 0.5f) * gTexel;
                float cover = 0.0f;
                for (int i = 0; i < past->now; ++i) {
                    const StormCellState& c = past->cells[0][i];
                    const float d = std::hypot(x - c.x, z - c.z) / std::max(c.radius_m, 1.0f);
                    cover = std::max(cover, c.coverage * cellCoverEnvelope(d));
                }
                const size_t k = static_cast<size_t>(iz) * g + ix;
                gCover[k] = cover;
                gRecent[k] = recentRainAt(past->view, x, z, wind);
                gNow[k] = cellRainAt(past->cells[0], past->now, x, z, wind);
            }
        }
    }
    auto coarse = [&](const std::vector<float>& v, float x, float z) {
        const float fx = std::clamp((x - place.origin_x) / gTexel - 0.5f, 0.0f, static_cast<float>(g - 1));
        const float fz = std::clamp((z - place.origin_z) / gTexel - 0.5f, 0.0f, static_cast<float>(g - 1));
        const int x0 = std::min(static_cast<int>(fx), g - 2), z0 = std::min(static_cast<int>(fz), g - 2);
        const float tx = fx - static_cast<float>(x0), tz = fz - static_cast<float>(z0);
        const size_t i = static_cast<size_t>(z0) * g + x0;
        return (v[i] * (1.0f - tx) + v[i + 1] * tx) * (1.0f - tz) +
               (v[i + g] * (1.0f - tx) + v[i + g + 1] * tx) * tz;
    };

    const float fade = 0.06f * static_cast<float>(n);
    int fogged = 0;
    float bandLo = 1e30f, bandHi = -1e30f;
    for (int iz = 0; iz < n; ++iz) {
        for (int ix = 0; ix < n; ++ix) {
            const float x = place.origin_x + (static_cast<float>(ix) + 0.5f) * texel;
            const float z = place.origin_z + (static_cast<float>(iz) + 0.5f) * texel;
            float ground = 0.0f, mean = 0.0f;
            const bool onTerrain = field && field->sample(x, z, ground, mean);
            float cover = drivers.layer_cover, recent = 0.0f, now = 0.0f;
            if (storms) {
                cover = std::max(cover, coarse(gCover, x, z));
                recent = coarse(gRecent, x, z);
                now = coarse(gNow, x, z);
            }
            LocalFogTexel t = fogTexel(drivers, clouds.local_weather.fog_strength, ground, mean,
                                       onTerrain, cover, recent, now);
            // Fade over the outer 6% of the map: no wall at its edge.
            const float edge = static_cast<float>(std::min({ix, iz, n - 1 - ix, n - 1 - iz}));
            const float e = std::clamp(edge / fade, 0.0f, 1.0f);
            t.fog *= e;
            t.mist *= e;
            map->texels[static_cast<size_t>(iz) * n + ix] = t;
            const float ext = t.fog + t.mist;
            map->max_extinction = std::max(map->max_extinction, ext);
            if (ext > 1e-6f) {
                bandLo = std::min(bandLo, ground - 5.0f);
                // The march raises the lid up to 35% of the fog depth.
                if (t.fog > 1e-6f)
                    bandHi = std::max(bandHi, t.fog_top + 0.35f * std::max(t.fog_top - ground, 10.0f) +
                                              kLocalFogTopSoftM);
                if (t.mist > 1e-6f) bandHi = std::max(bandHi, ground + kLocalMistReachM);
            }
            if (ext > 1e-4f) ++fogged;
        }
    }
    map->fogged_fraction = static_cast<float>(fogged) / static_cast<float>(n * n);
    map->active = map->max_extinction > 1e-6f && bandHi > bandLo;
    if (map->active) {
        map->band_bottom_y = bandLo;
        map->band_top_y = bandHi;
    }
    return map;
}

// ── Terrain copy (main thread: TerrainManager is not thread-safe) ──────────

uint64_t terrainFingerprint(std::vector<TerrainObject>& terrains) {
    uint64_t h = fnvValue(kFnvBasis, terrains.size());
    for (const TerrainObject& t : terrains) {
        const Heightmap& hm = t.heightmap;
        h = fnvValue(h, hm.width);
        h = fnvValue(h, hm.height);
        h = fnvValue(h, hm.scale_xz);
        h = fnvValue(h, hm.scale_y);
        h = fnvValue(h, hm.data.size());
        if (t.transform) h = fnv(h, &t.transform->getFinal().m[0][0], sizeof(float) * 16);
        // Strided samples: an edit anywhere moves at least some of them
        // (a sculpt stroke spans many texels).
        const size_t count = hm.data.size();
        if (count > 0) {
            const size_t stride = std::max<size_t>(count / 257, 1);
            for (size_t i = 0; i < count; i += stride) h = fnvValue(h, hm.data[i]);
        }
    }
    return h;
}

std::shared_ptr<TerrainField> buildTerrainField(std::vector<TerrainObject>& terrains, uint64_t fingerprint) {
    auto field = std::make_shared<TerrainField>();
    field->fingerprint = fingerprint;
    struct Source {
        const Heightmap* hm;
        Matrix4x4 toWorld;
        Matrix4x4 toLocal;
        bool transformed;
    };
    std::vector<Source> sources;
    float minX = 1e30f, minZ = 1e30f, maxX = -1e30f, maxZ = -1e30f;
    for (const TerrainObject& t : terrains) {
        const Heightmap& hm = t.heightmap;
        if (hm.data.empty() || hm.width <= 1 || hm.height <= 1 || hm.scale_xz <= 0.0f) continue;
        Source s{&hm, Matrix4x4(), Matrix4x4(), t.transform != nullptr};
        if (s.transformed) {
            s.toWorld = t.transform->getFinal();
            s.toLocal = s.toWorld.inverse();
        }
        for (int c = 0; c < 4; ++c) {
            const Vec3 local((c & 1) ? hm.scale_xz : 0.0f, 0.0f, (c & 2) ? hm.scale_xz : 0.0f);
            const Vec3 w = s.transformed ? s.toWorld.multiplyVector(Vec4(local, 1.0f)).xyz() : local;
            minX = std::min(minX, w.x); maxX = std::max(maxX, w.x);
            minZ = std::min(minZ, w.z); maxZ = std::max(maxZ, w.z);
        }
        sources.push_back(s);
    }
    if (sources.empty()) return field;
    const int n = field->n;
    field->size_m = std::max(maxX - minX, maxZ - minZ) * 1.02f;
    field->origin_x = 0.5f * (minX + maxX) - 0.5f * field->size_m;
    field->origin_z = 0.5f * (minZ + maxZ) - 0.5f * field->size_m;
    field->height.assign(static_cast<size_t>(n) * n, 0.0f);
    field->local_mean.assign(static_cast<size_t>(n) * n, 0.0f);
    field->covered.assign(static_cast<size_t>(n) * n, 0);
    const float texel = field->texel_m();
    for (int iz = 0; iz < n; ++iz) {
        for (int ix = 0; ix < n; ++ix) {
            const float x = field->origin_x + (static_cast<float>(ix) + 0.5f) * texel;
            const float z = field->origin_z + (static_cast<float>(iz) + 0.5f) * texel;
            for (const Source& s : sources) {
                const Heightmap& hm = *s.hm;
                const Vec3 lp = s.transformed ? s.toLocal.multiplyVector(Vec4(x, 0.0f, z, 1.0f)).xyz()
                                              : Vec3(x, 0.0f, z);
                if (lp.x < 0.0f || lp.z < 0.0f || lp.x > hm.scale_xz || lp.z > hm.scale_xz) continue;
                // Same bilinear as TerrainManager::sampleHeight.
                const float gx = lp.x / hm.scale_xz * static_cast<float>(hm.width - 1);
                const float gz = lp.z / hm.scale_xz * static_cast<float>(hm.height - 1);
                const int x0 = std::clamp(static_cast<int>(gx), 0, hm.width - 1);
                const int z0 = std::clamp(static_cast<int>(gz), 0, hm.height - 1);
                const int x1 = std::min(x0 + 1, hm.width - 1);
                const int z1 = std::min(z0 + 1, hm.height - 1);
                const float fx = gx - static_cast<float>(x0);
                const float fz = gz - static_cast<float>(z0);
                const float h0 = hm.data[z0 * hm.width + x0] * (1.0f - fx) + hm.data[z0 * hm.width + x1] * fx;
                const float h1 = hm.data[z1 * hm.width + x0] * (1.0f - fx) + hm.data[z1 * hm.width + x1] * fx;
                const float local = (h0 * (1.0f - fz) + h1 * fz) * hm.scale_y;
                const float y = s.transformed ? s.toWorld.multiplyVector(Vec4(lp.x, local, lp.z, 1.0f)).y
                                              : local;
                const size_t i = static_cast<size_t>(iz) * n + ix;
                field->height[i] = y;
                field->covered[i] = 1;
                break;
            }
        }
    }
    // Local mean over ~1.5 km: separable box over covered texels only.
    const int radius = std::clamp(static_cast<int>(std::lround(1500.0f / texel)), 2, 16);
    std::vector<float> sum(static_cast<size_t>(n) * n, 0.0f), cnt(static_cast<size_t>(n) * n, 0.0f);
    for (int iz = 0; iz < n; ++iz) {
        for (int ix = 0; ix < n; ++ix) {
            float s = 0.0f, c = 0.0f;
            for (int k = std::max(ix - radius, 0); k <= std::min(ix + radius, n - 1); ++k) {
                const size_t i = static_cast<size_t>(iz) * n + k;
                if (field->covered[i]) { s += field->height[i]; c += 1.0f; }
            }
            sum[static_cast<size_t>(iz) * n + ix] = s;
            cnt[static_cast<size_t>(iz) * n + ix] = c;
        }
    }
    for (int iz = 0; iz < n; ++iz) {
        for (int ix = 0; ix < n; ++ix) {
            float s = 0.0f, c = 0.0f;
            for (int k = std::max(iz - radius, 0); k <= std::min(iz + radius, n - 1); ++k) {
                s += sum[static_cast<size_t>(k) * n + ix];
                c += cnt[static_cast<size_t>(k) * n + ix];
            }
            const size_t i = static_cast<size_t>(iz) * n + ix;
            field->local_mean[i] = c > 0.0f ? s / c : field->height[i];
        }
    }
    field->valid = true;
    return field;
}

uint64_t g_terrainFingerprint = 0;
std::shared_ptr<const TerrainField> g_terrainField;
uint64_t g_environmentKey = 0;
uint64_t g_environmentRevision = 0;
uint64_t g_fogKey = 0;
uint64_t g_fogRevision = 0;

Vec3 worldSunDirection(const World& world) {
    const NishitaSkyParams n = world.getNishitaParams();
    Vec3 s(n.sun_direction.x, n.sun_direction.y, n.sun_direction.z);
    const float len = s.length();
    return len > 1e-6f ? s / len : Vec3(0.0f, 1.0f, 0.0f);
}

float quantizeTime(float t) { return std::floor(t / kFogTimeQuantumS) * kFogTimeQuantumS; }

} // namespace

void syncLocalWeather(const World& world, float cloud_time_s, const Vec3& camera) {
    // 1. Terrain copy, only when the terrain changed.
    // A terrain graph computing its heights on a worker (defer_mesh_updates)
    // may be writing the heightmap right now: keep the last field until it
    // finalizes on the main thread.
    std::vector<TerrainObject>& terrains = TerrainManager::getInstance().getTerrains();
    bool terrainBusy = false;
    for (const TerrainObject& t : terrains) terrainBusy = terrainBusy || t.defer_mesh_updates;
    uint64_t fingerprint = g_terrainFingerprint;
    if (!terrainBusy) {
        fingerprint = terrainFingerprint(terrains);
        if (fingerprint != g_terrainFingerprint) {
            g_terrainFingerprint = fingerprint;
            g_terrainField = buildTerrainField(terrains, fingerprint);
        }
    }

    // 2. Environment (terrain + sun): what the storm births read.
    const Vec3 sun = worldSunDirection(world);
    uint64_t envKey = fnvValue(kFnvBasis, fingerprint);
    envKey = fnvValue(envKey, sun.x);
    envKey = fnvValue(envKey, sun.y);
    envKey = fnvValue(envKey, sun.z);
    if (envKey != g_environmentKey || !localWeatherEnvironment()) {
        g_environmentKey = envKey;
        auto env = std::make_shared<LocalWeatherEnvironment>();
        env->terrain = g_terrainField && g_terrainField->valid ? g_terrainField : nullptr;
        env->sun_direction = sun;
        env->revision = ++g_environmentRevision;
        publishEnvironment(env);
    }
    const std::shared_ptr<const LocalWeatherEnvironment> env = localWeatherEnvironment();

    // 3. Fog map, when any input changed.
    const CloudState& clouds = world.getClouds();
    const FogPlacement place = placeFogMap(env->terrain.get(), clouds, camera);
    const float timeQ = clouds.storms.enabled ? quantizeTime(cloud_time_s) : 0.0f;
    const LocalFogDrivers drivers = fogDrivers(world, clouds, sun);
    uint64_t key = fnvValue(kFnvBasis, env->revision);
    key = fnvValue(key, world.cloudRevision());
    key = fnvValue(key, timeQ);
    key = fnvValue(key, place.origin_x);
    key = fnvValue(key, place.origin_z);
    key = fnvValue(key, place.size_m);
    key = fnv(key, &drivers, sizeof(drivers));
    const Vec3 wind = world.cloudWindVelocity();
    key = fnvValue(key, wind.x);
    key = fnvValue(key, wind.z);
    if (key == g_fogKey && localFogMapSnapshot()) return;
    g_fogKey = key;
    std::shared_ptr<LocalFogMap> map = buildFogMap(world, *env, place, timeQ, drivers);
    map->revision = ++g_fogRevision;
    publishFogMap(map);
}

LocalWeatherPoint sampleLocalWeather(const World& world, float cloud_time_s, const Vec3& p) {
    LocalWeatherPoint out;
    const std::shared_ptr<const LocalWeatherEnvironment> env = localWeatherEnvironment();
    const CloudState& clouds = world.getClouds();
    const Vec3 wind = world.cloudWindVelocity();
    out.suitability = birthSuitability(env.get(), p.x, p.z, wind);
    if (env && env->terrain) {
        out.on_terrain = env->terrain->sample(p.x, p.z, out.ground_y, out.local_mean_y);
    }
    if (clouds.storms.enabled) {
        // The fog map's time quantum: the answer matches the picture.
        auto past = std::make_unique<PastCells>();
        past->evaluate(clouds, quantizeTime(cloud_time_s), wind);
        out.recent_rain = recentRainAt(past->view, p.x, p.z, wind);
        out.rain_now = cellRainAt(past->cells[0], past->now, p.x, p.z, wind);
    }
    if (const std::shared_ptr<const LocalFogMap> map = localFogMapSnapshot()) {
        out.in_fog_map = map->sample(p.x, p.z, out.fog);
        out.extinction_at_y = out.in_fog_map ? localFogExtinctionAt(out.fog, p.y) : 0.0f;
    }
    return out;
}

} // namespace atmosphere
