#ifndef WEATHER_FROM_WORLD_GLSL
#define WEATHER_FROM_WORLD_GLSL
// Manual surface amount + canonical climate profile: visual.yzw = sea-level
// temperature C, lapse K/m and scene altitude offset. ABI padding is reused.
#define WEATHER_DATA WeatherRenderData( \
    vec4(worldData.w.weatherEnabled, worldData.w.weatherType, \
         worldData.w.weatherIntensity, worldData.w.weatherDensity), \
    vec4(worldData.w.weatherWindDirection, worldData.w.weatherWindSpeed), \
    vec4(worldData.w.weatherPrecipitationScale, worldData.w._weatherPad0, \
         intBitsToFloat(worldData.w._weatherPad1), worldData.w.altitude), \
    vec4(worldData.w.weatherSurfaceWetness, worldData.w.weatherSurfaceAccumulation, \
         worldData.w.weatherSurfaceSettling, worldData.w.weatherSurfaceHeight), \
    vec4(worldData.w.weatherSurfaceResponseEnabled, 0.0, 0.0, 0.0))
#include "weather_render_data.glsl"
#endif
