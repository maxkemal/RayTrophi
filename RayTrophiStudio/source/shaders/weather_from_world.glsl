#ifndef WEATHER_FROM_WORLD_GLSL
#define WEATHER_FROM_WORLD_GLSL
#ifndef WEATHER_TIME
#define WEATHER_TIME 0.0
#endif
#define WEATHER_DATA WeatherRenderData( \
    vec4(worldData.w.weatherEnabled, worldData.w.weatherType, \
         worldData.w.weatherIntensity, worldData.w.weatherDensity), \
    vec4(worldData.w.weatherWindDirection, worldData.w.weatherWindSpeed), \
    vec4(worldData.w.weatherPrecipitationScale, worldData.w.weatherVisibility, \
         worldData.w.weatherVisualMode, WEATHER_TIME), \
    vec4(worldData.w.weatherSurfaceWetness, worldData.w.weatherSurfaceAccumulation, \
         worldData.w.weatherSurfaceSettling, worldData.w.weatherSurfaceHeight), \
    vec4(worldData.w.weatherSurfaceResponseEnabled, 0.0, 0.0, 0.0))
#include "weather_render_data.glsl"
#endif
