#version 450
#extension GL_GOOGLE_include_directive : require

// Camera-facing particle billboards built on the CPU (particle roadmap
// Phase 1.5). The CPU sends particle CENTRES; this stage expands the quad and
// reads the particle's look from the appearance LUT at (lut_row, normalized
// age). Used for systems that are not device resident (CPU policy, host
// consumers, a simulation on another VkDevice) and for fluid-domain particles.
// Device-resident systems are drawn by particle_viewport_pull.vert instead.

layout(location = 0) in vec3 inCenter;   // world-space particle centre
layout(location = 1) in vec2 inCorner;   // [-1,1] quad corner (also the UV)
layout(location = 2) in vec3 inLook;     // age, lut_row, size_scale

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec4 vColor;
layout(location = 2) flat out vec4 vShade;   // roughness, lit
layout(location = 3) flat out vec3 vL1;      // strongest light, billboard frame
layout(location = 4) flat out vec3 vC1;      // its relative radiance
layout(location = 5) flat out vec3 vL2;      // second light
layout(location = 6) flat out vec3 vC2;
layout(location = 7) flat out vec3 vAmb;     // the other lights + bounce
layout(location = 8) flat out vec3 vUp;      // world up, billboard frame

layout(set = 0, binding = 0, std430) readonly buffer AppearanceLut {
    vec4 lut[];
};

#include "include/particle_appearance_lut.glsl"

void main() {
    float halfSize;
    int row = int(inLook.y + 0.5);
    vColor = particleLook(row, inLook.x, inLook.z, halfSize);
    particleLighting(row, inCenter, vShade, vL1, vC1, vL2, vC2, vAmb, vUp);
    vUV = particleEffectCorner(inCorner, vColor.a, inLook.x, inLook.z, vShade);
    gl_Position = particleClip(inCenter, vUV, halfSize);
}
