#version 450

layout(location = 0) flat in uint vMaterialID;
layout(location = 1) in vec2 vTexCoord;

#define GpuMaterial Material
#include "material_struct.glsl"

layout(set = 0, binding = 0, std430) readonly buffer MaterialBuffer {
    GpuMaterial materials[];
};
#define MATERIAL_TEXTURE_FRAGMENT
#define MATERIAL_TEXTURE_BINDING 1
#define MATERIAL_TEXTURE_REF_BINDING 29
#include "material_texture_arrays.glsl"
#undef MATERIAL_TEXTURE_BINDING
#undef MATERIAL_TEXTURE_REF_BINDING

layout(push_constant) uniform MaterialPreviewPushConstants {
    mat4 viewProj;

    vec4 cameraPos;
    vec4 lightDir0;
    vec4 lightDir1;

    uvec4 materialMeta;
} pc;

bool validTexture(uint id) {
    return materialTextureValid(id);
}

#include "material_preview_uv.glsl"
#include "material_preview_opacity.glsl"

void main() {
    bool impostor = (vMaterialID & 0x80000000u) != 0u;
    uint count = max(pc.materialMeta.x, 1u);
    uint index = min(vMaterialID & 0x7fffffffu, count - 1u);
    GpuMaterial mat = materials[index];
    vec2 uv = applyUVTransform(vTexCoord, mat);
    vec2 uvDx = dFdx(uv);
    vec2 uvDy = dFdy(uv);
    if (impostor) return;
    // Only the camera prepass excludes transmission. Atlas behavior is retained.
    if ((pc.materialMeta.y & (1u << 30u)) != 0u &&
        (((mat.flags & MATERIAL_FLAGS_PREVIEW_CUTOUT) == 0u && mat.opacity < 0.999) || mat.transmission > 0.001 ||
         materialTextureValid(uint(mat.transmission_tex)) || (mat.flags & ((1u << 17u) | (1u << 19u) | (1u << 24u))) != 0u)) discard;

    float opacity = previewSurfaceOpacity(mat, uv, uvDx, uvDy, false);
    if (opacity == 0.0) discard;
    // Partial-alpha receivers must not hide the opaque surface behind them.
    if ((pc.materialMeta.y & (1u << 30u)) != 0u && opacity < 0.999) discard;
}
