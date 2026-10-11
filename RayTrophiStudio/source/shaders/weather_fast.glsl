// Bounded depth layers, shared by RT camera rays and RayFusion scene post.
// A visual approximation, not simulated particles or a second volume march.
#ifndef WEATHER_FAST_GLSL
#define WEATHER_FAST_GLSL
#include "weather_render_data.glsl"

float weatherFastHash(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

vec3 applyWeatherAtmosphere(vec3 color, vec3 rayDir, float distance) {
    if (!weatherVisualActive()) return color;
    float sigma = WEATHER_DATA.state.z * WEATHER_DATA.state.w *
        (0.00016 + (1.0 - clamp(WEATHER_DATA.visual.y, 0.0, 1.0)) * 0.00034);
    float amount = min(0.78, 1.0 - exp(-max(distance, 0.0) * sigma));
    int type = int(WEATHER_DATA.state.y + 0.5);
    vec3 tint = weatherTintColor(type);
    if (type == 1) tint *= 0.72;
    return mix(color, tint, amount);
}

vec3 applyWeatherPrecipitation(vec3 color, vec3 origin, vec3 direction, float distance) {
    if (!weatherVisualActive() || distance <= 0.5) return color;
    int type = int(WEATHER_DATA.state.y + 0.5);
    float density = clamp(WEATHER_DATA.state.z * WEATHER_DATA.state.w, 0.0, 1.0);
    vec3 wind = WEATHER_DATA.wind.xyz * WEATHER_DATA.wind.w;
    float time = WEATHER_DATA.visual.w;
    float scale = max(WEATHER_DATA.visual.x, 0.1);
    float fallSpeed = type == 1 ? 6.5 : (type == 2 ? 0.9 : 0.0);
    vec3 velocity = wind + vec3(0.0, -fallSpeed, 0.0);
    vec3 tint = weatherTintColor(type);
    // World coordinates move with wind/fall; each depth has its own seed and
    // size. Foreground surfaces reject every layer behind their hit distance.
    for (int layer = 2; layer >= 0; --layer) {
        float depth = layer == 0 ? 2.5 : (layer == 1 ? 8.0 : 22.0);
        if (depth >= distance) continue;
        vec3 p = (origin + direction * depth - velocity * time) / scale;
        vec2 q = vec2(p.x + p.z * 0.713, p.y);
        q *= type == 1 ? vec2(2.5, 0.65) : vec2(1.6, 1.6);
        vec2 cell = floor(q);
        vec2 f = fract(q);
        float rnd = weatherFastHash(vec3(cell, float(layer) + 17.0));
        float spawn = smoothstep(1.0 - density * 0.6, 1.0, rnd);
        float amount;
        if (type == 1) {
            float slant = clamp((wind.x + wind.z * 0.713) / 6.5, -2.0, 2.0);
            float x = abs(f.x - 0.5 - (f.y - 0.5) * slant * 0.12);
            float width = mix(0.025, 0.06, rnd);
            amount = spawn * (1.0 - smoothstep(width, width * 2.0, x)) *
                     (1.0 - smoothstep(0.08, 0.5, abs(f.y - 0.5)));
        } else if (type == 2) {
            vec2 center = vec2(0.35 + rnd * 0.3, 0.5);
            center.x += sin(time * 1.3 + rnd * 23.0) * 0.12;
            float radius = mix(0.04, 0.12, rnd);
            amount = spawn * (1.0 - smoothstep(radius * 0.4, radius, length(f - center)));
        } else {
            // Soft dust/mist patches avoid a uniform full-screen colour wash.
            vec2 s = f * f * (3.0 - 2.0 * f);
            float a = weatherFastHash(vec3(cell, float(layer)));
            float b = weatherFastHash(vec3(cell + vec2(1.0, 0.0), float(layer)));
            float c = weatherFastHash(vec3(cell + vec2(0.0, 1.0), float(layer)));
            float e = weatherFastHash(vec3(cell + 1.0, float(layer)));
            amount = smoothstep(0.3, 0.9, mix(mix(a, b, s.x), mix(c, e, s.x), s.y)) *
                     density * (type == 3 ? 0.12 : 0.07);
        }
        amount *= smoothstep(depth, depth + 1.0, distance);
        if (type == 1) {
            color = color * (1.0 - amount * 0.12) + tint * amount * 0.30;
        } else {
            color = mix(color, tint, min(amount * (type == 2 ? 0.65 : 1.0), 0.8));
        }
    }
    return color;
}

#endif
