#version 450

// Per-vertex attributes
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in uint inMaterialID;

// Per-instance attributes (model matrix as 4 columns)
layout(location = 3) in vec4 inModelCol0;
layout(location = 4) in vec4 inModelCol1;
layout(location = 5) in vec4 inModelCol2;
layout(location = 6) in vec4 inModelCol3;

// UV coordinates (binding 4)
layout(location = 7) in vec2 inTexCoord;

layout(location = 0) out vec3 vWorldNormal;
layout(location = 1) flat out uint vMaterialID;
layout(location = 2) out vec2 vTexCoord;
layout(location = 3) out vec3 vWorldPos;

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;
    mat4 view;
    vec4 cameraPos;    // xyz = world position, w = unused
    vec4 lightDir0;    // xyz = direction, w = intensity
    vec4 lightDir1;    // xyz = direction (fill), w = intensity
    vec4 lightDir2;    // xyz = direction (rim), w = intensity
    uvec4 materialMeta; // x = material count, y = quality, z = lighting preset
} pc;

void main() {
    mat4 model = mat4(inModelCol0, inModelCol1, inModelCol2, inModelCol3);
    vec4 worldPos = model * vec4(inPosition, 1.0);
    gl_Position = pc.viewProj * worldPos;

    mat3 normalMatrix = transpose(inverse(mat3(model)));
    vec3 worldNormal = normalMatrix * inNormal;
    float nlen = length(worldNormal);
    if (nlen < 1e-5) {
        worldNormal = vec3(0.0, 1.0, 0.0);
    } else {
        worldNormal /= nlen;
    }
    vWorldNormal = worldNormal;
    vMaterialID  = inMaterialID;
    // Flip V: Blender/OpenGL UV origin is bottom-left, Vulkan sampler origin is top-left
    vTexCoord    = vec2(inTexCoord.x, 1.0 - inTexCoord.y);
    vWorldPos    = worldPos.xyz;
}
