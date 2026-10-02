#pragma once
/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Atmosphere/AtmosphereClouds.h
* =========================================================================
*
* The cloud half of the atmosphere authority (docs/dev/ATMOSPHERE_CLOUDS.md).
*
* ★★★ CloudState is the ONLY owner of cloud parameters. The old
*   NishitaSkyParams::cloud_* / cloud2_* fields are a PACKET written by
*   World::syncCloudPacket() from this struct (cloudsToLegacyPacket) and read
*   by the paths that have not moved yet: the in-TLAS Vulkan cloud volume
*   (until Faz 3b), OptiX (frozen, decision (a)), Stylize and the CPU path.
*   Nothing writes those fields any more.
*
* Physical units live in the field names. There are no "silver intensity"
* or "shadow strength" dials: those looks come from the light transport
* (Faz 3b/3c), and a dial for them would fight it.
*
* Wind is NOT here: it is the climate's (Faz 2 rule -- the atmosphere
* produces shared data, consumers read it).
*/

#include "Vec3.h"
#include "json.hpp"
#include <cstdint>
#include <string>
#include <vector>

struct NishitaSkyParams;

namespace atmosphere {

constexpr int kMaxCloudLayers = 3;

// One volumetric layer: a spherical shell between base and base+thickness.
struct CloudLayer {
    bool  enabled = false;
    float base_altitude_m = 1500.0f;   // above sea level
    float thickness_m = 1500.0f;
    // Fraction of the sky this layer covers, 0..1. Applied to the weather
    // map's coverage channel, so the pattern comes from the map and the
    // amount from here.
    float coverage = 0.35f;
    // 0 = stratus (flat sheet), 0.5 = cumulus (heaped), 1 = cumulonimbus
    // (towering). Selects the vertical density profile; the weather map
    // varies it locally by CloudWeather::type_variation.
    float type = 0.5f;
    // Extinction at full density, 1/m. Cumulus ~0.05: a 1 km column has an
    // optical depth of ~50 and is effectively opaque (acceptance test).
    float extinction_per_m = 0.05f;
    // Droplet diameter in micrometres, 5..50. Drives the phase function
    // (Jendersie & d'Eon 2023): larger drops -> sharper forward peak.
    float droplet_diameter_um = 10.0f;
    // How much the detail noise eats the edges, 0..1.
    float erosion = 0.5f;
    // Characteristic cloud size (base-shape noise period) and detail size, m.
    float cell_size_m = 1800.0f;
    float detail_size_m = 180.0f;
};

// High, thin ice layer (cirrus / cirrostratus). Rendered as a thin shell
// (Faz 3c); the data exists from 3a so presets and keys are complete.
struct CirrusLayer {
    bool  enabled = false;
    float altitude_m = 9000.0f;
    float coverage = 0.3f;
    float opacity = 0.3f;          // optical depth of the sheet at full coverage
    float scale_m = 8000.0f;       // streak length
};

// World-anchored weather map (procedural source in 3a).
struct CloudWeather {
    uint32_t seed = 1;
    float extent_m = 150000.0f;     // one tile of the map; it repeats beyond
    float feature_size_m = 12000.0f;// size of cloud "fields" (clusters)
    float type_variation = 0.25f;   // local +- around each layer's type
    // How fast the cloud SHAPES evolve, independent of wind drift (noise
    // coordinates per second along the evolution axis).
    float evolution_speed = 0.002f;
};

struct CloudQuality {
    // RT (Faz 3b): production marcher -- steps along the view ray, jittered
    // per sample so accumulation converges. rt_reference_path_trace switches
    // to the unbounded volumetric path tracer (slow; calibration only), where
    // rt_max_bounces applies.
    int rt_steps = 128;
    bool rt_reference_path_trace = false;
    int rt_max_bounces = 64;
    // RayFusion (Faz 3c).
    int realtime_steps = 128;
    int realtime_resolution_divisor = 1;   // 2 shook under wind drift in play, 4 blurred detail (by eye)
    // RT secondary rays: false = cloud-aware sky panorama (decision 2),
    // true = full march (slow, unbiased).
    bool rt_secondary_full_march = false;
};

// Rain / snow falling from layer 0 (docs/dev/ATMOSPHERE_WEATHER.md §3.1).
// Derived from the climate while derive_from_climate is on; set directly
// (script / panel) otherwise. Rendered as shafts under the base, where the
// weather map's B channel (cluster cores) is.
struct CloudPrecipitation {
    float rate_mm_h = 0.0f;        // peak rate under a core, water equivalent
    bool  snow = false;            // frozen at the ground (slower fall, denser veil)
    float ground_fraction = 1.0f;  // share of the base->ground column a shaft
                                   // survives before evaporating (< 1: virga)
};

struct CloudState {
    // Layer 0 follows the climate (docs/dev/ATMOSPHERE_WEATHER.md §2): base =
    // LCL, genus / coverage / depth from instability + humidity. World
    // re-derives it on every climate or altitude change; the panel shows those
    // four fields read-only. A preset turns it off (an explicit look).
    bool         derive_from_climate = true;
    CloudLayer   layers[kMaxCloudLayers];
    CirrusLayer  cirrus;
    CloudPrecipitation precipitation;
    CloudWeather weather;
    CloudQuality quality;

    bool anyEnabled() const;
};

// Rejects (does not clamp) out-of-range input.
bool validateClouds(const CloudState& c, std::string* error);

void cloudsToJson(const CloudState& c, nlohmann::json& j);
// Missing keys keep the defaults; an out-of-range present value fails.
bool cloudsFromJson(const nlohmann::json& j, CloudState& out, std::string* error);

// Keyframe blend. Floats lerp; bools and the seed come from `a` below t=1.
CloudState lerpClouds(const CloudState& a, const CloudState& b, float t);

// Writes the legacy packet (NishitaSkyParams::cloud_*) from the authority.
// wind_offset_m is the accumulated weather drift (x,z).
void cloudsToLegacyPacket(const CloudState& c, float wind_offset_x_m, float wind_offset_z_m,
                          NishitaSkyParams& out);

// Henyey-Greenstein g of the forward lobe for a droplet diameter
// (Jendersie & d'Eon 2023 fit, valid 5..50 um). Shared with the shader.
float cloudPhaseForwardG(float droplet_diameter_um);

// Presets. Names are stable IPC values.
std::vector<std::string> cloudPresetNames();
bool cloudPreset(const std::string& name, CloudState& out);

} // namespace atmosphere
