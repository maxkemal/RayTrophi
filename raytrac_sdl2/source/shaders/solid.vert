#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inModelCol0;
layout(location = 3) in vec4 inModelCol1;
layout(location = 4) in vec4 inModelCol2;
layout(location = 5) in vec4 inModelCol3;

layout(location = 0) out vec3 vNormal;

layout(push_constant) uniform SolidPushConstants {
    mat4 viewProj;
    mat4 view;
    int useMatcap;         // -1 = flat color, 0 = solid, 1 = matcap texture, 2..9 = procedural
    float overrideR, overrideG, overrideB; // flat color (useMatcap == -1)
} pc;

void main() {
    mat4 model = mat4(inModelCol0, inModelCol1, inModelCol2, inModelCol3);
    gl_Position = pc.viewProj * model * vec4(inPosition, 1.0);
    mat4 modelOrModelView = (pc.useMatcap != 0) ? (pc.view * model) : model;
    mat3 normalMatrix = mat3(modelOrModelView);
    vNormal = normalize(normalMatrix * inNormal);
}
