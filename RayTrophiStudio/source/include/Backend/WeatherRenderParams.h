#pragma once

#include "World.h"
#include "Atmosphere/AtmosphereClimate.h"
#include <cstring>

namespace Backend {

// std430 mirror of WeatherRenderData in weather_render_data.glsl.
struct WeatherRenderGPU {
    float state[4];     // enabled, type, intensity, density
    float wind[4];      // xyz direction, speed m/s
    float visual[4];    // pattern scale, sea-level C, lapse K/m, scene altitude offset
    float surface[4];   // wetness, accumulation, settling, height
    float flags[4];     // surface response, reserved
};
static_assert(sizeof(WeatherRenderGPU) == 80, "WeatherRenderGPU std430 layout");

// Manual SURFACE weather only (falling precipitation is the cloud packet's,
// ATMOSPHERE_NEAR_PRECIP.md). True keeps the RayFusion frame block that
// carries this payload to the surface shader alive without clouds.
inline bool weatherNeedsRendering(const WeatherParams& w) {
    return w.enabled != 0 && w.type != WEATHER_NONE &&
        ((w.intensity > 0.0f && w.density > 0.0f) ||
         (w.surface_response_enabled != 0 &&
          (w.surface_wetness_output > 0.0f || w.surface_accumulation_output > 0.0f)));
}

inline WeatherRenderGPU makeWeatherRenderGPU(const WeatherParams& w) {
    WeatherRenderGPU p{};
    p.state[0] = static_cast<float>(w.enabled);
    p.state[1] = static_cast<float>(w.type);
    p.state[2] = w.intensity;
    p.state[3] = w.density;
    p.wind[0] = w.derived_wind_direction.x;
    p.wind[1] = w.derived_wind_direction.y;
    p.wind[2] = w.derived_wind_direction.z;
    p.wind[3] = w.derived_wind_speed_mps;
    p.visual[0] = w.precipitation_scale;
    const auto ambient = atmosphere::ambientSnapshot();
    p.visual[1] = ambient.climate.surface_temperature_k - 273.15f;
    p.visual[2] = ambient.climate.lapse_rate_k_per_m;
    p.visual[3] = ambient.altitude_offset_m;
    p.surface[0] = w.surface_wetness_output;
    p.surface[1] = w.surface_accumulation_output;
    p.surface[2] = w.surface_settling_output;
    p.surface[3] = w.surface_height_output;
    p.flags[0] = static_cast<float>(w.surface_response_enabled);
    return p;
}

// Reuse retired weather padding without changing any GPUWorldData offsets.
// _weatherPad1 is an int in the ABI; transport the float lapse rate bitwise.
template <typename WorldGPU>
inline void fillWeatherClimateGPU(WorldGPU& world) {
    const auto ambient = atmosphere::ambientSnapshot();
    world._weatherPad0 = ambient.climate.surface_temperature_k - 273.15f;
    const float lapse = ambient.climate.lapse_rate_k_per_m;
    static_assert(sizeof(world._weatherPad1) == sizeof(lapse), "Weather lapse ABI");
    std::memcpy(&world._weatherPad1, &lapse, sizeof(lapse));
}

} // namespace Backend
