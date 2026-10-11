#ifndef WEATHER_RENDER_TYPES_GLSL
#define WEATHER_RENDER_TYPES_GLSL
struct WeatherRenderData {
    vec4 state;
    vec4 wind;
    vec4 visual;
    vec4 surface;
    vec4 flags;
};
#endif
