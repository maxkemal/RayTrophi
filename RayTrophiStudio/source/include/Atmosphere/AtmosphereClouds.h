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

// Coefficients scale nominal layer extinction; colours are linear RGB.
struct CloudMaterial {
    float density_multiplier = 1.0f;
    float scattering_coefficient = 1.0f;
    Vec3 scattering_color = Vec3(1.0f);
    float absorption_coefficient = 0.0f;
    bool override_anisotropy = false;
    float anisotropy = 0.75f;
    Vec3 emission_color = Vec3(1.0f);
    float emission_strength = 0.0f;
};

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
    // Ragged base, 0..1: 0 = flat condensation base; higher = the base breaks
    // into large hanging lobes with small detached fractus pieces between
    // them (cloud_common.glsl). Carved by the detail pass only, so it never
    // adds density: the authored base altitude is the lowest lobe.
    float base_raggedness = 0.6f;
    // Characteristic cloud size (base-shape noise period) and detail size, m.
    float cell_size_m = 1800.0f;
    float detail_size_m = 180.0f;
    float rotation_degrees = 0.0f;  // world Y rotation of the layer field
    int formation_role = 0;  // 0 generic, 1 storm base, 2 tower, 3 anvil
    int formation_group = 0; // 0 independent; 1..3 share a storm footprint/span
    bool use_custom_material = false;
    CloudMaterial material;
};

// High, thin ice layer (cirrus / cirrostratus). Rendered as a thin shell
// (Faz 3c); the data exists from 3a so presets and keys are complete.
struct CirrusLayer {
    bool  enabled = false;
    float altitude_m = 9000.0f;
    float coverage = 0.3f;
    float opacity = 0.3f;          // optical depth of the sheet at full coverage
    float scale_m = 8000.0f;       // streak length
    float rotation_degrees = 0.0f;  // relative to the climate wind direction
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
    // Marched pixels per frame = 1 / (N x N) of the cloud target; the rest
    // are carried by a full-resolution history (cloud_resolve.comp). 2 costs
    // what divisor 2 cost, at full resolution. 1 = every pixel every frame.
    int realtime_interleave = 2;
    // RT secondary rays: false = cloud-aware sky panorama (decision 2),
    // true = full march (slow, unbiased).
    bool rt_secondary_full_march = false;
};

// Rain / snow falling from layer 0 (docs/dev/ATMOSPHERE_WEATHER.md §3.1).
// Derived from the climate while derive_from_climate is on; set directly
// (script / panel) otherwise. Rendered as shafts under the base, where the
// weather map's B channel (cluster cores) is.
// Precipitation forms (docs/dev/ATMOSPHERE_PRECIP_TYPES.md). Ids shared with
// cloud_common.glsl (PRECIP_*) and IPC world.sample_clouds precipitation_form.
enum PrecipForm : int {
    kPrecipAuto = 0, kPrecipRain = 1, kPrecipDrizzle = 2, kPrecipSnow = 3, kPrecipSleet = 4,
    kPrecipGraupel = 5, kPrecipHail = 6, kPrecipFreezingRain = 7, kPrecipIcePellets = 8,
};
const char* precipFormName(int form);
int precipFormFromName(const std::string& name);   // -1 unknown

struct CloudPrecipitation {
    float rate_mm_h = 0.0f;        // peak rate under a core, water equivalent
    // kPrecipAuto: each point's form follows its column through the climate's
    // temperature profile (snow line, melting band, freezing rain under an
    // inversion, hail / graupel from updrafts, drizzle from shallow
    // stratiform cloud). Otherwise forced everywhere.
    int   form = kPrecipAuto;
    float ground_fraction = 1.0f;  // share of the base->ground column a shaft
                                   // survives before evaporating (< 1: virga)
    // DERIVED, read-only (World::rederiveClouds, every climate sync): the
    // profile the forms follow, and whether it reaches the scene ground
    // frozen (audio, panel).
    float air_temperature_c = 15.0f;     // at sea level
    float lapse_rate_k_per_m = 0.0065f;
    bool  snow = false;
};

// Deterministic timeline events, rendered through the shared Vulkan cloud path.
struct CloudLightning {
    bool enabled = false;
    bool derive_rate_from_climate = false;
    float rate_per_minute = 6.0f;
    float intensity = 30.0f;
    Vec3 color = Vec3(0.65f, 0.78f, 1.0f);
    // Where: only from mature storm cells (CloudStorms). The old authored
    // centre/extent square moved there as the storm domain.
    float ground_altitude_m = 0.0f;
    float ground_strike_fraction = 0.15f;
    // Share of the ground strikes that are UPWARD lightning: the leader
    // starts on the ground (a peak, a tower) and branches up into the cloud.
    float upward_fraction = 0.1f;
    float bolt_radius_m = 2.0f;
    float duration_s = 0.8f;
    float horizontal_span_m = 12000.0f;
    float branching = 0.75f;
    // Camera exposure the flash is averaged over, s (0 = instantaneous).
    // ~1/30: what a video frame integrates; removes re-stroke strobing.
    float shutter_s = 1.0f / 30.0f;
    // Radiance of the channel core relative to the lit cloud (`intensity`).
    // A real channel is far brighter than the cloud it lights.
    float channel_brightness = 8.0f;
};

// Storm systems (docs/dev/ATMOSPHERE_STORMS.md). Convective cells that are a
// pure function of (seed, time): born in the storm domain, carried by the
// wind like the weather map under them, growing -> mature -> dissipating.
// They raise the local cloud cover and genus, place the convective rain and
// are the only source of lightning.
constexpr int kMaxAutoStormCells = 16;
constexpr int kMaxManualStormCells = 8;
constexpr int kMaxStormCells = kMaxAutoStormCells + kMaxManualStormCells;

// An authored cell. lifetime_s <= 0: permanently mature (art direction).
struct StormCell {
    Vec3 position_m = Vec3(0.0f);   // world x/z at birth; y unused
    float radius_m = 3500.0f;
    float intensity = 1.0f;         // peak 0..1
    float birth_s = 0.0f;           // cloud time
    float lifetime_s = 0.0f;
};

struct CloudStorms {
    bool enabled = false;
    // max_cells, intensity, stratiform_rain and enabled follow the climate
    // (instability, humidity, genus); the panel shows them read-only.
    bool derive_from_climate = true;
    Vec3 center_m = Vec3(0.0f);     // storm domain, world x/z; y unused
    float extent_m = 40000.0f;
    int max_cells = 6;              // concurrent automatic cells, 0..16
    float cell_radius_m = 3500.0f;  // mean mature radius
    float lifetime_s = 2400.0f;     // mean cell life
    float intensity = 0.8f;         // mean peak strength 0..1
    // Broad light rain under dense cover OUTSIDE the cells, as a share of
    // the peak rate (stratiform / frontal rain). 0 = only cells rain.
    float stratiform_rain = 0.1f;
    // Cell structure (ATMOSPHERE_STORMS.md §3.1). rotation: mesocyclone
    // strength, signed (+ = counter-clockwise seen from above). turrets: how
    // much the cell is a few rising updraft towers (1) instead of one uniform
    // column (0).
    float rotation = 0.4f;
    float turrets = 0.8f;
    // Storm column (ATMOSPHERE_CELL_CLOUDS.md §1): each cell is drawn as its
    // own column from the cloud base to the tropopause (towers, overshooting
    // tops, anvil, wall cloud, mammatus) instead of raising the cover of the
    // layers it sits in. Off = the cells paint into the layers (old look).
    bool column = true;
    float column_top_m = 11000.0f;  // tropopause, above sea level: the anvil level
    float anvil = 0.7f;             // anvil spread downwind, 0..1
    float wall_cloud = 0.5f;        // lowered base under the main updraft, 0..1
    float mammatus = 0.5f;          // pouches under the anvil, 0..1
    std::vector<StormCell> cells;   // authored, <= kMaxManualStormCells
};

// One cell evaluated at a time: what the GPU cell map, the lightning and
// world.cloud_stats all read (one definition).
struct StormCellState {
    float x = 0.0f, z = 0.0f;       // world, current (drifted)
    float radius_m = 0.0f;
    float life = 0.0f;              // age / lifetime, 0..1 (0.5 for permanent)
    float coverage = 0.0f;          // cover floor at the core, 0..1
    float tower = 0.0f;             // genus floor (towering), 0..1
    float convective_rain = 0.0f;   // core rain potential, 0..1
    float stratiform_rain = 0.0f;   // wide light rain potential, 0..1
    float lightning = 0.0f;         // flash weight, 0..1
    float age_s = 0.0f;             // since birth (turret cycles)
    float peak = 0.0f;              // peak strength the phase curves scale
    uint32_t id = 0;                // stable for the cell's life (turret seeds)
    bool manual = false;
    int manual_index = -1;          // index in CloudStorms::cells, or -1
};

// Live cells at `time_s` (cloud time) moving with `wind_mps`. Writes at most
// kMaxStormCells, returns the count. Pure: same inputs, same cells.
struct CloudState;
int evaluateStormCells(const CloudState& clouds, float time_s, const Vec3& wind_mps,
                       StormCellState* out);

// Turns the automatic cells alive at `time_s` into authored cells (same
// birth, position, radius, peak and life: the sky does not change), up to the
// authored limit, and stops the automatic lanes (max_cells 0, storms no
// longer derived). `first`: index in evaluateStormCells' list to keep first
// (-1 none). Returns how many were added; they are appended to storms.cells.
int freezeStormCells(CloudState& clouds, float time_s, const Vec3& wind_mps, int first = -1);

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
    CloudMaterial material;

    CloudLightning lightning;
    CloudStorms storms;

    bool anyEnabled() const;
};

// Rejects (does not clamp) out-of-range input.
bool validateClouds(const CloudState& c, std::string* error);
bool validateCloudMaterial(const CloudMaterial& material, std::string* error);
void cloudMaterialToJson(const CloudMaterial& material, nlohmann::json& out);
bool cloudMaterialFromJson(const nlohmann::json& json, CloudMaterial& out,
                          std::string* error);
CloudMaterial lerpCloudMaterial(const CloudMaterial& a, const CloudMaterial& b, float t);

inline CloudMaterial layerMaterial(const CloudState& state, int index) {
    CloudMaterial m = state.material;
    const CloudLayer& layer = state.layers[index];
    if (!layer.use_custom_material) {
        return m;
    }
    const CloudMaterial& local = layer.material;
    m.density_multiplier *= local.density_multiplier;
    m.scattering_coefficient *= local.scattering_coefficient;
    m.scattering_color = Vec3(m.scattering_color.x * local.scattering_color.x,
                             m.scattering_color.y * local.scattering_color.y,
                             m.scattering_color.z * local.scattering_color.z);
    m.absorption_coefficient += local.absorption_coefficient;
    if (local.override_anisotropy) {
        m.override_anisotropy = true;
        m.anisotropy = local.anisotropy;
    }
    const Vec3 emission = m.emission_color * m.emission_strength +
                          local.emission_color * local.emission_strength;
    m.emission_color = emission;
    m.emission_strength = 1.0f;
    return m;
}

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
