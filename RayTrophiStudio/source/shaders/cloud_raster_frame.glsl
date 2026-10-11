// Shared std430 frame ABI: Backend::CloudRasterFrameGPU (176 bytes).
#ifndef CLOUD_RASTER_FRAME_GLSL
#define CLOUD_RASTER_FRAME_GLSL
struct CloudWorldLite {
    vec3  sunDir;
    float sunIntensity;
    float atmosphereHeight;
    int   _pad5;          // 1 = atmosphere LUTs bound (same meaning as RT)
    float shadowCenterX;  // cloud_shadow.comp
    float shadowCenterZ;
};

layout(set = 0, binding = 7, std430) readonly buffer CloudRasterFrame {
    CloudWorldLite w;
    vec4  camPos;         // xyz, w = tan(vertical fov / 2)
    vec4  camRight;       // xyz, w = aspect
    vec4  camUp;
    vec4  camFwd;         // xyz, w = 1 when the history is valid
    vec4  prevPos;        // previous frame, same layout
    vec4  prevRight;
    vec4  prevUp;
    vec4  prevFwd;
    uvec4 info;           // width, height, frame index, march steps
} worldData;

layout(set = 0, binding = 6) uniform sampler2D atmosphereLUTs[2];   // 0 transmittance, 1 sky-view

#endif
