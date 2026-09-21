#ifndef MATERIAL_PREVIEW_OPACITY_GLSL
#define MATERIAL_PREVIEW_OPACITY_GLSL

float previewSurfaceOpacity(GpuMaterial mat, vec2 uv, bool impostor) {
    // Proxies carry a single representative UV, not an alpha silhouette.
    if (impostor) return 1.0;
    float opacity = clamp(mat.opacity, 0.0, 1.0);
    if (validTexture(mat.opacity_tex)) {
        vec4 texel = texture(textures[nonuniformEXT(mat.opacity_tex)], uv);
        bool useAlpha = (mat.flags & 256u) != 0u || mat.opacity_tex == mat.albedo_tex;
        opacity *= useAlpha ? texel.a : texel.r;
        if (opacity < 0.1) opacity = 0.0;
    }
    return materialCoverageOpacity(opacity, (mat.flags & MATERIAL_FLAGS_PREVIEW_CUTOUT) != 0u);
}
#endif
