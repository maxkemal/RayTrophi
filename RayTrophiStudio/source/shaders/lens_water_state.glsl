#ifndef LENS_WATER_STATE_GLSL
#define LENS_WATER_STATE_GLSL
const uint LENS_CAPACITY = 256u;
const uint LENS_HISTORY = 601u;
struct LensDrop {
    vec4 a; // uv, radius cubed (m^3), age seconds
    vec4 b; // velocity uv/s, snow water-equivalent radius cubed, ice equivalent
};
struct LensSnapshot {
    uvec4 meta;
    vec4 stats;
    vec4 previous;
    LensDrop drops[256];
};
layout(std430, set = 0, binding = 0) buffer LensState {
    vec4 nearState[8];
    uvec4 meta;
    vec4 stats;
    vec4 previous;
    uvec4 history;
    LensDrop drops[256];
    LensSnapshot snapshots[601];
} lens;
layout(set = 0, binding = 1, rgba16f) uniform image2D lensField;
layout(set = 0, binding = 2, rgba16f) uniform image2D lensSolidField;
layout(push_constant) uniform LensPush {
    vec4 forward;
    vec4 up;
    vec4 originTime;
    vec4 windGlass; // wind x,z, diameter m, slip threshold m
    vec4 optics;    // blur uv, output aspect ratio, local atmospheric temperature C, reserved
    vec4 right;
    uvec4 control;  // enabled, reset key, reserved, reserved
} lp;
uint lensHash(uint v) {
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    return v ^ (v >> 16u);
}
float lensRandom(uint v) {
    return float(lensHash(v) >> 8u) / 16777216.0;
}
#endif
