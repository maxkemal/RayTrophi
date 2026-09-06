#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 2) in uint inMaterialID;
layout(location = 3) in vec4 inModelCol0;
layout(location = 4) in vec4 inModelCol1;
layout(location = 5) in vec4 inModelCol2;
layout(location = 6) in vec4 inModelCol3;
layout(location = 7) in vec2 inTexCoord;

layout(location = 0) flat out uint vMaterialID;
layout(location = 1) out vec2 vTexCoord;

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;
    mat4 view;
    vec4 cameraPos;
    vec4 lightDir0;
    vec4 lightDir1;
    vec4 lightDir2;
    uvec4 materialMeta;
} pc;

void main() {
    mat4 model = mat4(inModelCol0, inModelCol1, inModelCol2, inModelCol3);
    gl_Position = pc.viewProj * model * vec4(inPosition, 1.0);
    vMaterialID = inMaterialID;
    vTexCoord = vec2(inTexCoord.x, 1.0 - inTexCoord.y);
}
