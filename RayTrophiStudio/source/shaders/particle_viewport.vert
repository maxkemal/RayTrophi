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

layout(set = 0, binding = 0, std430) readonly buffer AppearanceLut {
    vec4 lut[];
};

#include "include/particle_appearance_lut.glsl"

void main() {
    float halfSize;
    vColor = particleLook(int(inLook.y + 0.5), inLook.x, inLook.z, halfSize);
    gl_Position = particleClip(inCenter, inCorner, halfSize);
    vUV = inCorner;
}
