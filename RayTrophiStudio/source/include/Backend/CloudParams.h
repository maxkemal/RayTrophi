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
    float g[4];   // group base, height, inverse footprint period, reserved
};
static_assert(sizeof(CloudLayerGPU) == 112, "CloudLayerGPU std430 layout");

struct CloudParamsGPU {
    CloudLayerGPU layers[atmosphere::kMaxCloudLayers];
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
    float precip2[4]; // snow (0/1), ground_fraction, shear_x, shear_z
                      // (shear = horizontal metres per metre of height, Faz 3e)
    float scatterAlbedo[4]; // linear RGB albedo, HG g override (2 = droplet fit)
    float emissionSource[4]; // linear RGB source radiance, reserved
    float cirrusDirection[4]; // authored wind x,z; camera weather time, reserved
    CloudLightningGPU lightning[kCloudLightningEvents];
};
static_assert(sizeof(CloudParamsGPU) == 736, "CloudParamsGPU std430 layout (cloud_common.glsl)");

inline void clearTransientCloudKey(CloudParamsGPU& key) {
    for (auto& event : key.lightning) {
        event = {};
    }
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
    // Precipitation (ATMOSPHERE_WEATHER.md §3.1). Optical extinction of the
    // drop population: Marshall-Palmer rain sigma ~ R^0.63 (2.5e-4 /m at
    // 1 mm/h -> ~4 km visibility at 10 mm/h); snow scatters far more per mm of
    // water (sigma ~ R^0.7, ~1.5 km at 1 mm/h). Terminal fall speed: rain
    // ~6.5 m/s, snow ~1 m/s; the wind carries the drops while they fall.
    const float rate = c.layers[0].enabled ? c.precipitation.rate_mm_h : 0.0f;
    const bool snow = c.precipitation.snow;
    p.precip[0] = rate;
    p.precip[1] = rate > 0.0f ? (snow ? 2.6e-3f * std::pow(rate, 0.7f) : 2.5e-4f * std::pow(rate, 0.63f)) : 0.0f;
    // precip[2..3] = how far the drops TRAIL their cloud per metre of fall,
    // at the ground (cloudPrecipSample). The weather map drifts with the
    // wind and the drops leave the base at that speed, so uniform wind gives
    // a vertical shaft in the cloud frame; only the slower near-surface air
    // (~60% of the cloud-level wind, linear deficit) drags the foot back.
    // Capped at ~56 degrees so snow in a gale still reads as a fall.
    const float fall = snow ? 1.0f : 6.5f;
    const float kSurfaceWindFraction = 0.6f;
    float tx = -wind_mps.x * (1.0f - kSurfaceWindFraction) / fall;
    float tz = -wind_mps.z * (1.0f - kSurfaceWindFraction) / fall;
    const float tl = std::sqrt(tx * tx + tz * tz);
    if (tl > 1.5f) { tx *= 1.5f / tl; tz *= 1.5f / tl; }
    p.precip[2] = tx;
    p.precip[3] = tz;
    p.precip2[0] = snow ? 1.0f : 0.0f;
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
    makeCloudLightningGPU(c, time_seconds, planet_radius_m, altitude_offset_m, p.lightning);
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
};
static_assert(sizeof(CloudRasterFrameGPU) == 256, "CloudRasterFrameGPU std430 layout (cloud_raster.comp)");

// One query for world.sample_clouds. mode 0: density at `a`; mode 1:
// transmittance along a -> b.
struct CloudQueryGPU {
    float a[4];   // xyz, unused
    float b[4];   // xyz, unused
};
static_assert(sizeof(CloudQueryGPU) == 32, "CloudQueryGPU std430 layout");

} // namespace Backend
