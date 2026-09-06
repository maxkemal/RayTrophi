#version 450
#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) flat in uint vMaterialID;
layout(location = 1) in vec2 vTexCoord;

#define GpuMaterial Material
#include "material_struct.glsl"

layout(set = 0, binding = 0, std430) readonly buffer MaterialBuffer {
    GpuMaterial materials[];
};
layout(set = 0, binding = 1) uniform sampler2D textures[];

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;
    mat4 view;
    vec4 cameraPos;
    vec4 lightDir0;
    vec4 lightDir1;
    vec4 lightDir2;
    uvec4 materialMeta;
} pc;

bool validTexture(uint id) {
    return id > 0u && id < max(pc.materialMeta.w, 1u);
}

vec2 applyUVTransform(vec2 originalUV, const GpuMaterial mat) {
    vec2 uv = originalUV - vec2(0.5);
    uv *= vec2(mat.uv_scale_x != 0.0 ? mat.uv_scale_x : 1.0,
               mat.uv_scale_y != 0.0 ? mat.uv_scale_y : 1.0);
    float a = radians(mat.uv_rotation_degrees);
    float c = cos(a), s = sin(a);
    uv = vec2(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
    return uv + vec2(0.5 + mat.uv_offset_x, 0.5 + mat.uv_offset_y);
}

void main() {
    bool impostor = (vMaterialID & 0x80000000u) != 0u;
    uint count = max(pc.materialMeta.x, 1u);
    uint index = min(vMaterialID & 0x7fffffffu, count - 1u);
    GpuMaterial mat = materials[index];
    if (impostor) return;

    float opacity = clamp(mat.opacity, 0.0, 1.0);
    if (validTexture(mat.opacity_tex)) {
        vec4 texel = texture(textures[nonuniformEXT(mat.opacity_tex)],
                             applyUVTransform(vTexCoord, mat));
        bool alpha = ((mat.flags & 256u) != 0u) || (mat.opacity_tex == mat.albedo_tex);
        opacity *= alpha ? texel.a : texel.r;
        if (opacity < 0.1) discard;
    }
    if (opacity == 0.0) discard;
}
