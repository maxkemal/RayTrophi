#pragma once
/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Atmosphere/AtmosphereLocalWeather.h
* =========================================================================
*
* Local weather (docs/dev/ATMOSPHERE_LOCAL_WINDS_FOG.md §2-4): where storm
* cells are born (suitability), where it has rained (recent rain) and where
* fog lies (local fog).
*
* ★ Everything here is a PURE function of the storm cells (themselves a pure
*   function of seed and time), the climate, the sun and the terrain height.
*   No step-by-step integration: scrubbing the timeline gives the same fog.
*   Ground wetness is never READ from the physics: "it rained here" is the
*   cells' own past, re-evaluated (climate -> physics stays one way).
*
* ★ The terrain is a boundary condition, not simulation state: a 128^2 copy
*   of its height (TerrainField), published by the main thread whenever the
*   terrain changes. Without terrain the field is invalid and every answer
*   falls back to flat ground at scene y = 0 (suitability 1 = old behaviour).
*
* ★ The fog is a VOLUME of the cloud field, not a screen-space veil: the map
*   (envelope) is sheet 3 of the cloud cell map and cloud_common.glsl's
*   cloudFogDensity() shapes it with the cloud noise; the cloud march lights
*   it (sun march, cloud shadow, multiple scattering) on both Vulkan
*   devices. The first version lived in the aerial froxel: 32x32 screen
*   cells, no shape, no shadow -- it read as paint on the ground.
*   world.sample_weather_local reports the envelope; world.sample_clouds
*   measures the shaped density the renderers draw.
*/
#include "Vec3.h"
#include <cstdint>
#include <memory>
#include <vector>

class World;

namespace atmosphere {

struct ClimateState;
struct CloudState;
struct StormCellState;

constexpr int kTerrainFieldSize = 128;
// = Backend::kCloudCellMapSize: the map is uploaded as one sheet of it.
constexpr int kLocalFogMapSize = 256;
// Recent rain and cell cover vary over kilometres: evaluated on a coarse grid
// and interpolated (a full-resolution cell loop x 8 past instants was the
// build cost).
constexpr int kLocalRainGridSize = 64;

// Low-resolution copy of the scene terrain (union of all terrains, square).
struct TerrainField {
    bool valid = false;
    float origin_x = 0.0f, origin_z = 0.0f;   // world x/z of texel (0,0)'s corner
    float size_m = 0.0f;                       // square side
    int n = kTerrainFieldSize;
    std::vector<float> height;       // scene y (m); ground of the nearest terrain where uncovered
    std::vector<float> local_mean;   // height box-averaged over ~1.5 km (hollows lie below it)
    std::vector<uint8_t> covered;    // 1 = a terrain lies under this texel
    uint64_t fingerprint = 0;

    float texel_m() const { return size_m / static_cast<float>(n); }
    // Bilinear; false outside the field or over an uncovered texel.
    bool sample(float x, float z, float& height_out, float& mean_out) const;
    // Height gradient (m per m) by central differences on the field.
    bool gradient(float x, float z, float& gx, float& gz) const;
};

// What the storm births and the fog read besides the cloud state. Published
// by syncLocalWeather() every frame it changes.
struct LocalWeatherEnvironment {
    std::shared_ptr<const TerrainField> terrain;   // may be null
    Vec3 sun_direction = Vec3(0.0f, 1.0f, 0.0f);    // towards the sun, normalized
    uint64_t revision = 0;
};
std::shared_ptr<const LocalWeatherEnvironment> localWeatherEnvironment();

// §2: birth suitability 0..1 at world x/z (1 = no preference). Windward
// slopes (orographic lift), ridges and sun-facing slopes raise it; the lee
// lowers it. `wind_mps` = the climate wind the cells drift with.
struct SuitabilityTerms {
    float value = 1.0f;
    float orographic = 0.0f;   // upslope component of the wind, 0..1
    float lee = 0.0f;          // downslope component, 0..1
    float ridge = 0.0f;        // above the local mean, 0..1
    float heating = 0.0f;      // sun-facing slope beyond flat ground, 0..1
    bool terrain = false;
};
SuitabilityTerms birthSuitability(const LocalWeatherEnvironment* env, float x, float z,
                                  const Vec3& wind_mps);

// Rain potential 0..1 of the live cells at x/z: the CPU mirror of the
// cell map's B channel (cloud_cells.comp). Cells only: the stratiform rain
// of the GPU weather map is not known on the CPU.
float cellRainAt(const StormCellState* cells, int count, float x, float z, const Vec3& wind_mps);

// §3: exponentially weighted rain of the last hour, 0..1 (1 = it has rained
// at full potential all along). Evaluates the cells at 8 past instants.
constexpr int kRecentRainSamples = 8;
constexpr float kRecentRainStepS = 450.0f;    // 8 x 450 s = 60 min
constexpr float kRecentRainDecayS = 1800.0f;
struct RecentRainCells {
    StormCellState* lists[kRecentRainSamples] = {};
    int counts[kRecentRainSamples] = {};
};
float recentRainAt(const RecentRainCells& past, float x, float z, const Vec3& wind_mps);

// §4: fog map texel = the ENVELOPE the cloud march shapes (RGBA16F sheet 3
// of the cloud cell map, cloud_common.glsl cloudFogMapAt).
//   fog      radiation fog extinction (1/m), filling hollows up to fog_top
//   fog_top  radiation fog top, scene y (m)
//   mist     mist extinction at the ground (1/m) over recently rained-on land
//   ground   ground, scene y (m): the mist thins upward from it
// Light needs no channel: the march shadows the fog through the clouds.
struct LocalFogTexel {
    float fog = 0.0f, fog_top = 0.0f, mist = 0.0f, ground = 0.0f;
};

// Vertical profile, shared with cloud_common.glsl cloudFogEnvelope.
constexpr float kLocalFogTopSoftM = 25.0f;        // radiation fog: soft lid
constexpr float kLocalMistScaleHeightM = 250.0f;  // mist: exp(-height above ground / H)
constexpr float kLocalMistReachM = 1000.0f;       // 4 H: where the band stops
float localFogExtinctionAt(const LocalFogTexel& t, float y);

// The scene-wide inputs of the fog, kept for the IPC answer.
struct LocalFogDrivers {
    float night = 0.0f;          // 1 = radiative cooling (sun low / below the horizon)
    float burn_off = 0.0f;       // 0..1 morning lift and thinning
    float calm = 0.0f;           // 1 = wind under ~1 m/s, 0 above 4 m/s
    float humidity = 0.0f;       // surface RH
    float layer_cover = 0.0f;    // low/mid layer cover (clear sky lets the ground cool; shades the sun)
    float cloud_base_y = 0.0f;   // scene y of the lowest enabled layer base
};

struct LocalFogMap {
    bool active = false;          // false: no fog anywhere (or disabled)
    float origin_x = 0.0f, origin_z = 0.0f, size_m = 0.0f;
    int n = kLocalFogMapSize;
    std::vector<LocalFogTexel> texels;
    float max_extinction = 0.0f;   // 1/m, fog + mist at the ground (= the march's majorant)
    float fogged_fraction = 0.0f;  // share of texels with ground extinction > 1e-4
    float band_bottom_y = 0.0f;    // scene y range holding any fog: the march
    float band_top_y = 0.0f;       // interval of the fog layer (GPU slot 4)
    bool terrain = false;          // placed over the terrain field
    float time_s = 0.0f;           // cloud time quantum it was built for
    LocalFogDrivers drivers;
    uint64_t revision = 0;

    // Bilinear over texels, the SAME addressing as the shader. False
    // outside the map.
    bool sample(float x, float z, LocalFogTexel& out) const;
};
std::shared_ptr<const LocalFogMap> localFogMapSnapshot();

// Main thread, every frame: refreshes the terrain field when the terrain
// changed (cheap fingerprint), publishes the environment and rebuilds the fog
// map when one of its inputs changed (cloud time is quantized to 10 s: fog
// evolves over minutes). `camera` places the map when there is neither
// terrain nor a storm domain.
void syncLocalWeather(const World& world, float cloud_time_s, const Vec3& camera);

// Recent rain at a point, evaluated now (IPC): the same function the map uses.
struct LocalWeatherPoint {
    SuitabilityTerms suitability;
    float recent_rain = 0.0f;
    float rain_now = 0.0f;
    float ground_y = 0.0f;
    float local_mean_y = 0.0f;
    bool on_terrain = false;
    bool in_fog_map = false;
    LocalFogTexel fog;            // envelope from the published map (what the march shapes)
    float extinction_at_y = 0.0f; // envelope 1/m at the queried height (noise not applied)
};
LocalWeatherPoint sampleLocalWeather(const World& world, float cloud_time_s, const Vec3& p);

} // namespace atmosphere
