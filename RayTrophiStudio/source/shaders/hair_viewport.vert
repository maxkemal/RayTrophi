#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in float inVCoord;
layout(location = 2) in float inThickness;
layout(location = 3) in vec3 inColor;

layout(location = 0) out float vVCoord;
layout(location = 1) out float vThickness;
layout(location = 2) out vec3 vColor;

layout(push_constant) uniform PC {
    mat4 viewProj;
    mat4 view;
    int useMatcap;
    float overrideR, overrideG, overrideB;
} pc;

void main() {
    gl_Position = vec4(inPosition, 1.0); // Output world space for geom shader
    vVCoord = inVCoord;
    vThickness = inThickness;
    vColor = inColor;
}
