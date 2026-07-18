#version 460
#extension GL_EXT_ray_tracing : require

#extension GL_EXT_nonuniform_qualifier : require

// Push Constants (match other shaders)
layout(push_constant) uniform CameraPC {
    vec4 origin;
    vec4 lowerLeft;
    vec4 horizontal;
    vec4 vertical;
    uint frameCount;
    uint minSamples;
    uint lightCount;
    float varianceThreshold;
    uint maxSamples;
    float exposure_factor;
} cam;

// Payload structure — shared ABI, single source of truth (declared for layout
// completeness; this shader only writes the location-1 shadow payload).
#include "rt_payload.glsl"

layout(location = 0) rayPayloadInEXT RayPayload payload;
// Shadow payload: rgb = transmissive tint accumulated by any-hits, w = reached-light flag.
layout(location = 1) rayPayloadInEXT vec4 shadowPayload;

// Descriptor bindings (minimal, mirror main shader)
layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 6) uniform sampler2D materialTextures[];
// World data (minimal) — must match miss.rmiss / C++ VkWorldDataSimple
struct WorldDataSimple {
    vec3 sunDir; float pad0;
    vec3 sunColor; float sunIntensity;
    float sunSize; float airDensity; float dustDensity; float mieAnisotropy;
    int  mode;
    int  envTexSlot;
    float envIntensity;
    float pad1;
};
layout(set = 0, binding = 7) readonly buffer WorldBuffer { WorldDataSimple w[]; } world;

void main() {
    // Ray escaped to the light: keep the accumulated rgb tint, flag as visible.
    shadowPayload.w = 1.0;
}
