#pragma once

#include "World.h"

namespace Backend {

// std430 mirror of WeatherRenderData in weather_render_data.glsl.
struct WeatherRenderGPU {
    float state[4];     // enabled, type, intensity, density
    float wind[4];      // xyz direction, speed m/s
    float visual[4];    // scale, visibility, mode, time seconds
    float surface[4];   // wetness, accumulation, settling, height
    float flags[4];     // surface response, reserved
};
static_assert(sizeof(WeatherRenderGPU) == 80, "WeatherRenderGPU std430 layout");

inline bool weatherNeedsRendering(const WeatherParams& w) {
    return w.enabled != 0 && w.type != WEATHER_NONE &&
        ((w.intensity > 0.0f && w.density > 0.0f) ||
         (w.surface_response_enabled != 0 &&
          (w.surface_wetness_output > 0.0f || w.surface_accumulation_output > 0.0f)));
}

inline WeatherRenderGPU makeWeatherRenderGPU(const WeatherParams& w, float time) {
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
    p.visual[1] = w.visibility;
    p.visual[2] = static_cast<float>(w.visual_mode);
    p.visual[3] = time;
    p.surface[0] = w.surface_wetness_output;
    p.surface[1] = w.surface_accumulation_output;
    p.surface[2] = w.surface_settling_output;
    p.surface[3] = w.surface_height_output;
    p.flags[0] = static_cast<float>(w.surface_response_enabled);
    return p;
}

} // namespace Backend
