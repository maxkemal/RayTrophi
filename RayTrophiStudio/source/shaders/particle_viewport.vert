#version 450

// Camera-facing particle billboards. Quad corners are expanded on the CPU into
// world-space positions, so the vertex stage only needs the view-projection.
layout(location = 0) in vec3 inPosition;  // world-space corner
layout(location = 1) in vec2 inUV;        // [-1,1] across the sprite
layout(location = 2) in vec4 inColor;     // rgb + opacity

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec4 vColor;

layout(push_constant) uniform PC {
    mat4 viewProj;
    mat4 view;
    int useMatcap;
    float overrideR, overrideG, overrideB;
} pc;

void main() {
    gl_Position = pc.viewProj * vec4(inPosition, 1.0);
    vUV = inUV;
    vColor = inColor;
}
