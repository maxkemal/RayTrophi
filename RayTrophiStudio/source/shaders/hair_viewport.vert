#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in float inVCoord;

layout(location = 0) out float vVCoord;

layout(push_constant) uniform PC {
    mat4 viewProj;
    mat4 view;
    int useMatcap;
    float overrideR, overrideG, overrideB;
} pc;

void main() {
    gl_Position = pc.viewProj * vec4(inPosition, 1.0);
    vVCoord = inVCoord;
}
