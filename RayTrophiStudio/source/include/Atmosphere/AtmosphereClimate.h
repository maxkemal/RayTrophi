#pragma once
/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          Atmosphere/AtmosphereClimate.h
* =========================================================================
*
* The climate half of the atmosphere authority (docs/dev/ATMOSPHERE_SYSTEM.md).
*
* ★★★ This struct is the ONLY owner of ambient temperature, humidity, pressure
*   and wind for the world. The render packets (NishitaSkyParams::derived_*,
*   WeatherParams::derived_*) are written from it by World::syncClimatePacket()
*   and nothing else. Before this existed the same quantities had 3-6 unaware
*   owners (sky 15 C vs gas 293 K, CloudManager's hard-coded 20 m/s wind) and
*   the sky's humidity slider was read by NO renderer at all.
*
* Units live in the field names on purpose: a consumer that hands a Kelvin
* value to a normalized-temperature solver will not crash, it will just look
* "too strong" -- the unit must be visible at every call site.
*
* Altitude convention: metres above sea level = scene Y (metres) +
* NishitaSkyParams::altitude (the scene's elevation offset the sky already uses).
*/

#include "Vec3.h"
#include "json.hpp"
#include <cstdint>
#include <string>

namespace atmosphere {

// International Standard Atmosphere constants.
constexpr float kGravity                 = 9.80665f;   // m/s^2
constexpr float kMolarMassAir            = 0.0289644f; // kg/mol
constexpr float kUniversalGasConstant    = 8.3144598f; // J/(mol K)
constexpr float kSpecificGasConstantAir  = 287.05f;    // J/(kg K)
constexpr float kTropopauseAltitudeM     = 11000.0f;   // lapse rate stops here
constexpr float kKelvinOffset            = 273.15f;

struct ClimateState {
    float surface_temperature_k     = 288.15f;   // ISA sea level (15 C)
    float lapse_rate_k_per_m        = 0.0065f;   // ISA troposphere; 0 = isothermal
    float surface_relative_humidity = 0.1f;      // 0..1
    float surface_pressure_pa       = 101325.0f; // ISA sea level
    Vec3  wind_direction            = Vec3(1.0f, 0.0f, 0.0f); // horizontal, normalized on set
    float wind_speed_mps            = 0.0f;      // at the surface; no shear yet (Faz 2)
    // Convective potential, dimensionless 0..1 (a CAPE proxy; docs/dev/
    // ATMOSPHERE_WEATHER.md decision 1). 0 = stable (stratiform sky),
    // 1 = violently unstable (towering cumulonimbus).
    float instability               = 0.3f;
};

// Weather DERIVED from the climate (docs/dev/ATMOSPHERE_WEATHER.md §2).
// Never stored: a pure function of ClimateState + the scene altitude.
struct DerivedWeather {
    float dew_point_k          = 273.15f;
    float cloud_base_m         = 1000.0f;  // above sea level (LCL)
    float cloud_thickness_m    = 1000.0f;
    float cloud_type           = 0.5f;     // 0 stratus .. 1 cumulonimbus (cloud_common.glsl)
    float cloud_coverage       = 0.0f;
    float freezing_level_m     = 3000.0f;  // above sea level
    float precipitation_mm_h   = 0.0f;     // peak rate under a cluster core
    bool  snow                 = false;    // precipitation reaches the ground frozen
    float lightning_per_minute = 0.0f;
};

DerivedWeather deriveWeather(const ClimateState& c, float surface_altitude_m);
void derivedWeatherToJson(const DerivedWeather& w, nlohmann::json& j);

// Everything a consumer may ask the climate for at one point. Derived, never stored.
struct ClimateSample {
    float altitude_m          = 0.0f;
    float temperature_k       = 288.15f;
    float relative_humidity   = 0.1f;    // constant with altitude until Faz 2
    float pressure_pa         = 101325.0f;
    float air_density_kg_m3   = 1.225f;
    Vec3  wind_mps            = Vec3(0.0f, 0.0f, 0.0f);
};

// Rejects (does not clamp) out-of-range input. Silent clamping is how a panel
// ends up showing a value the renderer never used.
bool validateClimate(const ClimateState& c, std::string* error);

// Normalizes the wind direction onto the horizontal plane. Returns false (and
// leaves `c` untouched) when the direction has no horizontal component.
bool normalizeWindDirection(ClimateState& c);

ClimateSample sampleClimate(const ClimateState& c, float altitude_m);

// Builds the climate a keyframe asks for. Starts from `current` so a keyed
// wind direction with no horizontal component (two opposite keys blend to
// zero halfway) keeps the current direction instead of failing the block.
ClimateState climateFromKeyed(const ClimateState& current,
                              float temperature_k, float lapse_rate_k_per_m,
                              float relative_humidity, float pressure_pa,
                              const Vec3& wind_direction, float wind_speed_mps);

// Hygroscopic growth of aerosol extinction (Hanel 1976): wet particles swell
// and scatter more, f(RH) = (1 - RH)^-gamma. Reference is DRY aerosol (RH=0),
// so the scale is 1 at RH=0 and ~1.05 at the old 0.1 default. RH is capped
// at 0.95: above that the medium is fog/cloud, not haze, and the formula
// diverges.
float hygroscopicMieScale(float relative_humidity);

// ── Ambient snapshot: the physics side's read door (Faz 2) ────────────────
//
// Simulations (MSF, fluids, particles, foliage, ocean) live in modules that
// never see the renderer's World, and some of them step on worker threads.
// They read the climate through this snapshot instead. World publishes it from
// syncClimatePacket(), which every climate/altitude write already goes
// through, so the snapshot can never lag a set_climate or a keyframe.
//
// ★ ONE-WAY (ATMOSPHERE_SYSTEM.md §3.1): consumers read, nothing here writes
//   back. A local override lives on the consumer (inherit_atmosphere = false).
// ★ Before the first publish the snapshot is the ClimateState defaults
//   (288.15 K, calm) -- a defined value, never garbage.
struct AmbientSnapshot {
    ClimateState climate;
    float    altitude_offset_m = 0.0f;  // NishitaSkyParams::altitude
    uint64_t revision = 0;              // bumps on every publish
};

void publishAmbient(const ClimateState& c, float altitude_offset_m);
AmbientSnapshot ambientSnapshot();
// sampleClimate at a scene position (scene Y in metres + altitude offset).
ClimateSample sampleAmbient(const Vec3& scene_pos);

// Scene-origin temperature, for consumers whose ambient is ONE number (the MSF
// world ambient, fluid emission). The lapse rate over a scene is sub-kelvin
// (0.0065 K/m x 100 m = 0.65 K), below what those consumers resolve.
float ambientSurfaceKelvin();

// Wind velocity (m/s, world space) at the scene origin. No shear yet: the
// climate has one surface wind, so position would not change the answer.
Vec3 ambientWindMps();

void climateToJson(const ClimateState& c, nlohmann::json& j);
// Missing keys keep the defaults. Returns false (and leaves `out` at defaults)
// when a present value is out of range, so a corrupt file cannot inject an
// invalid climate past validateClimate().
bool climateFromJson(const nlohmann::json& j, ClimateState& out, std::string* error);

} // namespace atmosphere
