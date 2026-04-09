#version 450

layout(location = 0) in float vVCoord;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    mat4 viewProj;
    mat4 view;
    int useMatcap;
    float overrideR, overrideG, overrideB;
} pc;

void main() {
    // Flat color override (used for selection highlight etc.)
    if (pc.useMatcap == -1) {
        outColor = vec4(pc.overrideR, pc.overrideG, pc.overrideB, 1.0);
        return;
    }
    // Root: dark warm brown -> Tip: light golden
    vec3 rootColor = vec3(0.22, 0.15, 0.10);
    vec3 tipColor  = vec3(0.80, 0.68, 0.48);
    outColor = vec4(mix(rootColor, tipColor, vVCoord), 1.0);
}
