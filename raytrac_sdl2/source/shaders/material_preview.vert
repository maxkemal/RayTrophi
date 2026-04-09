#version 450

// Per-vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in uint inMaterialID;

// Per-instance attributes (model matrix as 4 columns)
layout(location = 4) in vec4 inModelCol0;
layout(location = 5) in vec4 inModelCol1;
layout(location = 6) in vec4 inModelCol2;
layout(location = 7) in vec4 inModelCol3;

layout(location = 0) out vec3 vWorldPos;
layout(location = 1) out vec3 vWorldNormal;
layout(location = 2) out vec2 vUV;
layout(location = 3) flat out uint vMaterialID;

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;
    mat4 view;
    vec4 cameraPos;    // xyz = world position, w = unused
    vec4 lightDir0;    // xyz = direction, w = intensity
    vec4 lightDir1;    // xyz = direction (fill), w = intensity
    vec4 lightDir2;    // xyz = direction (rim), w = intensity
} pc;

void main() {
    mat4 model = mat4(inModelCol0, inModelCol1, inModelCol2, inModelCol3);
    vec4 worldPos = model * vec4(inPosition, 1.0);
    gl_Position = pc.viewProj * worldPos;
    vWorldPos = worldPos.xyz;

    mat3 normalMatrix = transpose(inverse(mat3(model)));
    vWorldNormal = normalize(normalMatrix * inNormal);

    vUV = inUV;
    vMaterialID = inMaterialID;
}
