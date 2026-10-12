#pragma once
/*
* =========================================================================
* Project:       RayTrophi Studio
* File:          Backend/CloudParams.h
* =========================================================================
*
* GPU packet for the cloud field (docs/dev/ATMOSPHERE_CLOUDS.md). ONE
* definition, mirrored field for field by shaders/cloud_common.glsl
* (CloudParams block, std430). Both Vulkan devices get the same packet, so
* the same density at the same point is a property of this struct, not of
* two hand-kept copies.
*
* Everything is in metres / 1/m / seconds. The shader never sees the
* CloudState struct itself.
*/

#include "Atmosphere/AtmosphereClouds.h"
#include "Backend/CloudLightningParams.h"
#include "Backend/WeatherRenderParams.h"
#include "Vec3.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace Backend {

// Resolutions of the generated textures. Fixed: the density function's
// frequency constants assume them.
constexpr uint32_t kCloudBaseNoiseSize   = 192;   // 3D, RGBA8 (28 MB; 128 blurred the 16-cell octave)
constexpr uint32_t kCloudDetailNoiseSize = 64;    // 3D, RGBA8 (32 = 4 texels per finest cell: soft blobs)
constexpr uint32_t kCloudCurlNoiseSize   = 128;   // 2D, RGBA8
constexpr uint32_t kCloudWeatherMapSize  = 2048;  // 2D, RGBA8 (decision 3)
// RT delta-tracking bound: max coverage per tile over its 3x3 neighbourhood.
// Tile = extent / 256 (~590 m at the 150 km default); cloud_rt.glsl steps
// at most one tile per majorant fetch.
constexpr uint32_t kCloudMajorantSize    = 256;   // 2D, RGBA8 (R used)
// RayFusion cloud shadow map: 512^2 over 32 km (62.5 m texels) around the
// camera; cloud_shadow.comp and material_preview_frag.frag hold the extent.
constexpr uint32_t kCloudShadowMapSize   = 512;
constexpr float    kCloudShadowExtentM   = 32000.0f;

struct CloudLayerGPU {
    float a[4];   // base_altitude_m, thickness_m, coverage, type
    float b[4];   // extinction_per_m, droplet_diameter_um, erosion, enabled (0/1)
    float c[4];   // 1/cell_size_m, 1/detail_size_m, weather_offset_u, weather_offset_v
    float d[4];   // inverse yaw cos, sin; formation role, group
    float e[4];   // effective layer RGB albedo, phase override (2 = droplet fit)
    float f[4];   // effective layer emission RGB, reserved
    float g[4];   // group base, height, inverse footprint period, base raggedness
};
static_assert(sizeof(CloudLayerGPU) == 112, "CloudLayerGPU std430 layout");

// GPU layer slots: the authored layers, then the storm column shell
// (ATMOSPHERE_CELL_CLOUDS.md §1, cloud_common.glsl cloudShellDensity).
constexpr int kCloudGpuLayers = atmosphere::kMaxCloudLayers + 1;
constexpr int kCloudStormShellLayer = atmosphere::kMaxCloudLayers;

struct CloudParamsGPU {
    CloudLayerGPU layers[kCloudGpuLayers];
    float weather[4]; // 1/extent_m, type_variation, evolution (s * speed), rt_steps
    float wind[4];    // drift_x_m, drift_z_m, planet_radius_m, altitude_offset_m
    float cirrus[4];  // enabled, altitude_m, coverage, optical depth
    float misc[4];    // 1/cirrus_scale_m, weather_seed, rt_max_bounces, rt_flags
                      // rt_flags: +1 render (set by the adapter when the
                      // textures are ready for clouds or fast weather),
                      // +2 secondary rays march fully (quality),
                      // +4 reference path tracer instead of the marcher
    float precip[4];  // rate_mm_h, extinction_per_m at that rate, trail_x, trail_z
                      // (trail = upwind lag behind the drifting cloud, metres
                      //  per metre of fall, at the ground; see makeCloudParamsGPU)
    float precip2[4]; // reserved, ground_fraction, shear_x, shear_z
                      // (shear = horizontal metres per metre of height, Faz 3e)
    float scatterAlbedo[4]; // linear RGB albedo, HG g override (2 = droplet fit)
    float emissionSource[4]; // linear RGB source radiance, reserved
    float cirrusDirection[4]; // authored wind x,z; camera weather time, reserved
    float storms[4];   // cell map centre x, z, 1/size (m), enabled (0/1)
    float storms2[4];  // stratiform rain share, tropopause height fraction in the
                       // storm column, wall-cloud allowance m, mammatus 0..1
    float precip3[4];  // sea-level air temperature C, lapse K/m, forced form (0 auto),
                       // reserved (ATMOSPHERE_PRECIP_TYPES.md)
    CloudLightningGPU lightning[kCloudLightningEvents];
    CloudFlashLightGPU flash[kCloudLightningEvents];   // lit cloud base per event
    // Channel network per event (kCloudBoltSegments each; count in color[3]).
    CloudBoltSegmentGPU bolt[kCloudLightningEvents * kCloudBoltSegments];
};
static_assert(sizeof(CloudParamsGPU) == 6144, "CloudParamsGPU std430 layout (cloud_common.glsl)");

// ── Storm cells (docs/dev/ATMOSPHERE_STORMS.md) ─────────────────────────────
// Input of cloud_cells.comp, which paints the cell map every update: three
// 256^2 fields stacked in one 256 x 768 image (cells, cell structure, storm
// column).
constexpr uint32_t kCloudCellMapSize = 256;
constexpr uint32_t kCloudCellMapRows = 3 * kCloudCellMapSize;

struct CloudCellGPU {
    float a[4];   // world x, z, radius m, cover floor
    float b[4];   // tower, convective rain, stratiform rain, lightning weight
    float c[4];   // id (0..65535), age s, rotation (signed), turret detail
    float d[4];   // life 0..1, peak, anvil spread, wall cloud
};
static_assert(sizeof(CloudCellGPU) == 64, "CloudCellGPU std430 layout");

struct CloudCellsGPU {
    float header[4];   // count, map centre x, z, map size m
    float wind[4];     // wind direction x, z (forward flank), majorant dilation m, reserved
    CloudCellGPU cells[atmosphere::kMaxStormCells];
};
static_assert(sizeof(CloudCellsGPU) == 32 + 64 * atmosphere::kMaxStormCells,
              "CloudCellsGPU std430 layout (cloud_cells.comp)");

// Square the cell map covers: the domain plus how far a cell drifts in
// half a life (cells are born upwind by that much) and two radii, capped so
// a texel stays well under a cell radius.
inline float cloudCellMapSize(const atmosphere::CloudStorms& s, const Vec3& wind_mps) {
    const float wind = std::sqrt(wind_mps.x * wind_mps.x + wind_mps.z * wind_mps.z);
    const float reach = wind * s.lifetime_s * 0.5f * 1.3f + 2.0f * s.cell_radius_m * 1.3f;
    return (std::min)(s.extent_m + 2.0f * reach, 400000.0f);
}

inline CloudCellsGPU makeCloudCellsGPU(const atmosphere::CloudState& c, float time_seconds,
                                       const Vec3& wind_mps) {
    CloudCellsGPU g{};
    if (!c.storms.enabled) return g;
    atmosphere::StormCellState cells[atmosphere::kMaxStormCells];
    const int n = atmosphere::evaluateStormCells(c, time_seconds, wind_mps, cells);
    const float size = cloudCellMapSize(c.storms, wind_mps);
    g.header[0] = static_cast<float>(n);
    g.header[1] = c.storms.center_m.x;
    g.header[2] = c.storms.center_m.z;
    g.header[3] = size;
    const float wind = std::sqrt(wind_mps.x * wind_mps.x + wind_mps.z * wind_mps.z);
    g.wind[0] = wind > 1e-3f ? wind_mps.x / wind : 0.0f;
    g.wind[1] = wind > 1e-3f ? wind_mps.z / wind : 0.0f;
    // A majorant fetch must bound a whole segment (<= one weather-majorant
    // tile, cloudSegmentLen) plus the linear filter of two cell texels.
    g.wind[2] = c.weather.extent_m / static_cast<float>(kCloudMajorantSize) +
                2.0f * size / static_cast<float>(kCloudCellMapSize);
    for (int i = 0; i < n; ++i) {
        const auto& s = cells[i];
        CloudCellGPU& o = g.cells[i];
        o.a[0] = s.x; o.a[1] = s.z; o.a[2] = s.radius_m; o.a[3] = s.coverage;
        o.b[0] = s.tower; o.b[1] = s.convective_rain; o.b[2] = s.stratiform_rain; o.b[3] = s.lightning;
        o.c[0] = static_cast<float>(s.id & 0xffffu);
        o.c[1] = s.age_s;
        o.c[2] = c.storms.rotation;
        o.c[3] = c.storms.turrets;
        o.d[0] = s.life;
        o.d[1] = s.peak;
        o.d[2] = c.storms.column ? c.storms.anvil : 0.0f;
        o.d[3] = c.storms.column ? c.storms.wall_cloud : 0.0f;
    }
    return g;
}

inline void clearTransientCloudKey(CloudParamsGPU& key) {
    for (auto& event : key.lightning) {
        event = {};
    }
    for (auto& light : key.flash) {
        light = {};
    }
    for (auto& segment : key.bolt) {
        segment = {};
    }
}

// Storm column shell (ATMOSPHERE_CELL_CLOUDS.md §1): layer slot 3, from the
// lowest cloud base (minus the wall-cloud allowance) to the tropopause plus
// the overshooting tops. Its look (noise scales, erosion, material, phase)
// is the host layer's -- the lowest enabled one, whose base is the LCL. The
// layers must already be packed. Off (enabled 0) without a host or with
// storms.column off; the cells then paint into the layers.
constexpr float kStormOvershootM = 800.0f;
constexpr float kStormWallCloudM = 300.0f;
inline void packStormColumn(const atmosphere::CloudState& c, CloudParamsGPU& p) {
    CloudLayerGPU& s = p.layers[kCloudStormShellLayer];
    s = CloudLayerGPU{};
    float hostBase = 0.0f, tropo = 0.0f;
    const int host = stormColumnSpan(c, hostBase, tropo);
    if (host < 0 || p.layers[host].b[0] <= 0.0f) return;
    const atmosphere::CloudLayer& hl = c.layers[host];
    s = p.layers[host];
    const float wall = kStormWallCloudM * c.storms.wall_cloud;
    const float base = hl.base_altitude_m - wall;
    const float top = tropo + kStormOvershootM;
    s.a[0] = base;
    s.a[1] = top - base;
    s.a[2] = 1.0f;   // coverage is the cells'; > 0 keeps the shell in the interval tests
    s.a[3] = 1.0f;
    s.b[3] = 1.0f;
    s.d[2] = 0.0f;
    s.d[3] = 0.0f;
    s.f[3] = 1.0f;   // the column flag (cloudShellActive)
    s.g[0] = base;
    s.g[1] = top - base;
    p.storms2[1] = (tropo - base) / (top - base);
    p.storms2[2] = wall;
    p.storms2[3] = c.storms.mammatus;
}

inline CloudParamsGPU makeCloudParamsGPU(const atmosphere::CloudState& c, float time_seconds,
                                         const Vec3& drift_m, const Vec3& wind_mps,
                                         float planet_radius_m,
                                         float altitude_offset_m) {
    CloudParamsGPU p{};
    const atmosphere::CloudMaterial& m = c.material;
    const float total = m.scattering_coefficient + m.absorption_coefficient;
    const float extinctionScale = m.density_multiplier * total;
    const float albedo = total > 0.0f ? m.scattering_coefficient / total : 0.0f;
    p.scatterAlbedo[0] = m.scattering_color.x * albedo;
    p.scatterAlbedo[1] = m.scattering_color.y * albedo;
    p.scatterAlbedo[2] = m.scattering_color.z * albedo;
    p.scatterAlbedo[3] = m.override_anisotropy ? m.anisotropy : 2.0f;
    p.emissionSource[0] = m.emission_color.x * m.emission_strength;
    p.emissionSource[1] = m.emission_color.y * m.emission_strength;
    p.emissionSource[2] = m.emission_color.z * m.emission_strength;
    for (int i = 0; i < atmosphere::kMaxCloudLayers; ++i) {
        const atmosphere::CloudLayer& l = c.layers[i];
        CloudLayerGPU& g = p.layers[i];
        const atmosphere::CloudMaterial lm = atmosphere::layerMaterial(c, i);
        const float lt = lm.scattering_coefficient + lm.absorption_coefficient;
        const float la = lt > 0.0f ? lm.scattering_coefficient / lt : 0.0f;
        g.a[0] = l.base_altitude_m;
        g.a[1] = l.thickness_m;
        g.a[2] = l.coverage;
        g.a[3] = l.type;
        g.b[0] = l.extinction_per_m * lm.density_multiplier * lt;
        // Diameter, not g: the full phase (HG + Draine blend) needs all four
        // fit parameters, and they all derive from the diameter.
        g.b[1] = l.droplet_diameter_um;
        g.b[2] = l.erosion;
        g.b[3] = l.enabled ? 1.0f : 0.0f;
        g.c[0] = 1.0f / (l.cell_size_m > 1.0f ? l.cell_size_m : 1.0f);
        g.c[1] = 1.0f / (l.detail_size_m > 1.0f ? l.detail_size_m : 1.0f);
        // Each layer reads the shared weather map at its own offset, so two
        // layers are not the same pattern stacked (golden-ratio spread).
        const int fieldIndex = l.formation_group > 0 ? l.formation_group - 1 : i;
        g.c[2] = 0.6180339f * static_cast<float>(fieldIndex);
        g.c[3] = 0.3819660f * static_cast<float>(fieldIndex);
        const float yaw = -l.rotation_degrees * 0.017453292519943295f;
        g.d[0] = std::cos(yaw);
        g.d[1] = std::sin(yaw);
        g.d[2] = static_cast<float>(l.formation_role);
        g.d[3] = static_cast<float>(l.formation_group);
        g.e[0] = lm.scattering_color.x * la;
        g.e[1] = lm.scattering_color.y * la;
        g.e[2] = lm.scattering_color.z * la;
        g.e[3] = lm.override_anisotropy ? lm.anisotropy : 2.0f;
        g.f[0] = lm.emission_color.x * lm.emission_strength;
        g.f[1] = lm.emission_color.y * lm.emission_strength;
        g.f[2] = lm.emission_color.z * lm.emission_strength;
        float groupBase = l.base_altitude_m;
        float groupTop = l.base_altitude_m + l.thickness_m;
        float groupPeriod = l.cell_size_m;
        int firstMember = i;
        if (l.formation_group > 0) {
            for (int j = 0; j < atmosphere::kMaxCloudLayers; ++j) {
                const auto& member = c.layers[j];
                if (!member.enabled || member.formation_group != l.formation_group) {
                    continue;
                }
                groupBase = (std::min)(groupBase, member.base_altitude_m);
                groupTop = (std::max)(groupTop, member.base_altitude_m + member.thickness_m);
                if (j < firstMember) {
                    firstMember = j;
                    groupPeriod = member.cell_size_m;
                }
            }
        }
        g.g[0] = groupBase;
        g.g[1] = (std::max)(groupTop - groupBase, 1.0f);
        g.g[2] = 1.0f / (std::max)(groupPeriod, 1.0f);
        g.g[3] = l.base_raggedness;
    }
    p.weather[0] = 1.0f / (c.weather.extent_m > 1.0f ? c.weather.extent_m : 1.0f);
    p.weather[1] = c.weather.type_variation;
    p.weather[2] = time_seconds * c.weather.evolution_speed;
    p.weather[3] = static_cast<float>(c.quality.rt_steps);
    p.wind[0] = drift_m.x;
    p.wind[1] = drift_m.z;
    p.wind[2] = planet_radius_m;
    p.wind[3] = altitude_offset_m;
    p.cirrus[0] = c.cirrus.enabled ? 1.0f : 0.0f;
    p.cirrus[1] = c.cirrus.altitude_m;
    p.cirrus[2] = c.cirrus.coverage;
    p.cirrus[3] = c.cirrus.opacity * extinctionScale;
    const float windLength = std::sqrt(wind_mps.x * wind_mps.x + wind_mps.z * wind_mps.z);
    const float wx = windLength > 1e-5f ? wind_mps.x / windLength : 1.0f;
    const float wz = windLength > 1e-5f ? wind_mps.z / windLength : 0.0f;
    const float cirrusYaw = c.cirrus.rotation_degrees * 0.017453292519943295f;
    p.cirrusDirection[0] = wx * std::cos(cirrusYaw) - wz * std::sin(cirrusYaw);
    p.cirrusDirection[1] = wx * std::sin(cirrusYaw) + wz * std::cos(cirrusYaw);
    p.misc[0] = 1.0f / (c.cirrus.scale_m > 1.0f ? c.cirrus.scale_m : 1.0f);
    p.misc[1] = static_cast<float>(c.weather.seed % 65536u);
    p.misc[2] = static_cast<float>(c.quality.rt_max_bounces);
    p.misc[3] = (c.quality.rt_secondary_full_march ? 2.0f : 0.0f) +
                (c.quality.rt_reference_path_trace ? 4.0f : 0.0f);
    // Precipitation (ATMOSPHERE_WEATHER.md §3.1, ATMOSPHERE_PRECIP_TYPES.md).
    // The form -- and with it the optics (Marshall-Palmer rain sigma ~ R^0.63,
    // snow far more per mm of water, hail far less) and the fall speed -- is
    // decided per point in the shader from the climate's temperature
    // profile. precip[1] is only an upper bound of the extinction over every
    // form (snow), for the cheap segment reject.
    const float rate = c.layers[0].enabled ? c.precipitation.rate_mm_h : 0.0f;
    p.precip[0] = rate;
    p.precip[1] = rate > 0.0f ? 2.6e-3f * (std::max)(std::pow(rate, 0.6f), std::pow(rate, 0.7f)) : 0.0f;
    p.precip3[0] = c.precipitation.air_temperature_c;
    p.precip3[1] = c.precipitation.lapse_rate_k_per_m;
    p.precip3[2] = static_cast<float>(c.precipitation.form);
    // precip[2..3] = how far the drops TRAIL their cloud per metre of fall,
    // at the ground, for a 1 m/s faller (cloudPrecipSample divides by the
    // local form's fall speed and caps it at ~56 degrees: snow in a gale
    // still reads as a fall, hail falls nearly straight). The weather map
    // drifts with the wind and the drops leave the base at that speed, so
    // uniform wind gives a vertical shaft in the cloud frame; only the slower
    // near-surface air (~60% of the cloud-level wind, linear deficit) drags
    // the foot back.
    const float kSurfaceWindFraction = 0.6f;
    p.precip[2] = -wind_mps.x * (1.0f - kSurfaceWindFraction);
    p.precip[3] = -wind_mps.z * (1.0f - kSurfaceWindFraction);
    p.precip2[0] = 0.0f;   // reserved
    p.precip2[1] = c.precipitation.ground_fraction;
    // Wind shear: crowns lean downwind ~0.015 m/m per m/s at the top (quadratic
    // in height and position-modulated in cloudShear),
    // capped at 0.5 so a gale does not lay the clouds flat.
    const float shear = 0.015f;
    float sx = wind_mps.x * shear, sz = wind_mps.z * shear;
    const float sl = std::sqrt(sx * sx + sz * sz);
    if (sl > 0.5f) { sx *= 0.5f / sl; sz *= 0.5f / sl; }
    p.precip2[2] = sx;
    p.precip2[3] = sz;
    makeCloudLightningGPU(c, time_seconds, wind_mps, planet_radius_m, altitude_offset_m,
                          p.lightning, p.flash, p.bolt);
    if (c.storms.enabled) {
        const float size = cloudCellMapSize(c.storms, wind_mps);
        p.storms[0] = c.storms.center_m.x;
        p.storms[1] = c.storms.center_m.z;
        p.storms[2] = 1.0f / size;
        p.storms[3] = 1.0f;
        p.storms2[0] = c.storms.stratiform_rain;
        packStormColumn(c, p);
    }
    return p;
}

inline bool sharedWeatherCloudRendering(const WorldData& world,
                                       const atmosphere::CloudState& clouds) {
    return (world.mode == WORLD_MODE_NISHITA && clouds.anyEnabled()) ||
           weatherNeedsRendering(world.weather);
}

inline CloudParamsGPU makeSharedCloudParamsGPU(const atmosphere::CloudState& clouds,
                                              float time, const Vec3& drift,
                                              const Vec3& wind, const WorldData& world,
                                              float weatherTime) {
    CloudParamsGPU p = makeCloudParamsGPU(clouds, time, drift, wind,
                                        world.nishita.planet_radius, world.nishita.altitude);
    p.cirrusDirection[2] = weatherTime; // reserved slot, shared camera-weather clock
    if (world.mode != WORLD_MODE_NISHITA || !clouds.anyEnabled()) {
        for (auto& layer : p.layers) {
            layer.b[3] = 0.0f;
        }
        p.cirrus[0] = 0.0f;
        p.precip[0] = p.precip[1] = 0.0f;
        for (auto& event : p.lightning) {
            event.start[3] = 0.0f;
        }
        for (auto& light : p.flash) {
            light = {};
        }
    }
    return p;
}

// Inputs of the WEATHER MAP only (its regeneration key). Wind drift and
// evolution move the lookup, never the map, so they are not in here.
struct CloudWeatherGenParams {
    uint32_t seed = 1;
    uint32_t feature_cells = 12;   // clusters across one map tile
    uint32_t size = kCloudWeatherMapSize;
    uint32_t pad = 0;
};
static_assert(sizeof(CloudWeatherGenParams) == 16, "CloudWeatherGenParams push layout");

inline CloudWeatherGenParams makeCloudWeatherGenParams(const atmosphere::CloudState& c) {
    CloudWeatherGenParams w;
    w.seed = c.weather.seed;
    const float cells = c.weather.extent_m / (c.weather.feature_size_m > 1.0f ? c.weather.feature_size_m : 1.0f);
    // Integer: the map must tile seamlessly, and a tileable noise needs a
    // whole number of cells per period.
    w.feature_cells = static_cast<uint32_t>(cells < 1.0f ? 1.0f : (cells > 256.0f ? 256.0f : cells + 0.5f));
    return w;
}

inline uint64_t hashCloudWeatherGen(const CloudWeatherGenParams& w) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&w);
    for (size_t i = 0; i < sizeof(w); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

// Per-frame block of cloud_raster.comp (RayFusion clouds, Faz 3c). The first
// 32 bytes carry the names cloud_rt.glsl reads from the RT world struct.
struct CloudRasterFrameGPU {
    float sunDir[3];
    float sunIntensity;
    float atmosphereHeight;
    int32_t lutReady;
    float shadowCenter[2];   // texel-snapped camera x, z (cloud_shadow.comp)
    float camPos[4];      // w = tan(vfov/2)
    float camRight[4];    // w = aspect
    float camUp[4];       // w = wind drift since last frame, x (m)
    float camFwd[4];      // w = history valid
    float prevPos[4];
    float prevRight[4];
    float prevUp[4];      // w = wind drift since last frame, z (m)
    float prevFwd[4];
    uint32_t info[4];     // width, height, frame index, steps
    WeatherRenderGPU weatherEffects;
    // Copy of CloudParamsGPU::flash: the RayFusion surface pass has no cloud
    // field bound but must be lit by the same flash (material_preview_frag).
    CloudFlashLightGPU flash[kCloudLightningEvents];
};
static_assert(sizeof(CloudRasterFrameGPU) == 384, "CloudRasterFrameGPU std430 layout (cloud_raster.comp)");

// One query for world.sample_clouds. mode 0: density at `a`; mode 1:
// transmittance along a -> b.
struct CloudQueryGPU {
    float a[4];   // xyz, unused
    float b[4];   // xyz, unused
};
static_assert(sizeof(CloudQueryGPU) == 32, "CloudQueryGPU std430 layout");

} // namespace Backend
