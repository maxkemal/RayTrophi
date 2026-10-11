#ifndef WEATHER_RENDER_DATA_GLSL
#define WEATHER_RENDER_DATA_GLSL

#include "weather_render_types.glsl"

// WEATHER_DATA is supplied by the caller from the shared GPU packet.
bool weatherActive() {
    return WEATHER_DATA.state.x > 0.5 && WEATHER_DATA.state.y > 0.5 &&
           WEATHER_DATA.state.z > 0.0 && WEATHER_DATA.state.w > 0.0;
}

bool weatherVisualActive() {
    return weatherActive() && WEATHER_DATA.visual.z < 0.5;
}

vec3 weatherTintColor(int type) {
    if (type == 1) return vec3(0.50, 0.56, 0.62);
    if (type == 2) return vec3(0.86, 0.90, 0.96);
    if (type == 3) return vec3(0.74, 0.58, 0.38);
    return vec3(0.70, 0.76, 0.82);
}

#endif
