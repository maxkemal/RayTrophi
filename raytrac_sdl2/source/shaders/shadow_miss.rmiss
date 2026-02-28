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

// Payload structure (match closesthit)
struct RayPayload {
    vec3     radiance;
    vec3     attenuation;
    vec3     scatterOrigin;
    vec3     scatterDir;
    uint     seed;
    bool     scattered;
    bool     hitEmissive;
    uint     occluded;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;
layout(location = 1) rayPayloadInEXT bool shadowOccluded;

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
    // No hit -> not occluded
    shadowOccluded = false;
}
