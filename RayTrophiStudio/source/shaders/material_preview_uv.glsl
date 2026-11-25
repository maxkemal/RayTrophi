#ifndef RT_PREVIEW_UV
#define RT_PREVIEW_UV
vec2 applyUVTransform(vec2 originalUV, const GpuMaterial mat) {
    vec2 uv = originalUV - vec2(0.5);

    // Scale
    float sx = (mat.uv_scale_x != 0.0) ? mat.uv_scale_x : 1.0;
    float sy = (mat.uv_scale_y != 0.0) ? mat.uv_scale_y : 1.0;
    uv *= vec2(sx, sy);

    // Rotation
    if (mat.uv_rotation_degrees != 0.0) {
        float angle = mat.uv_rotation_degrees * (3.14159265359 / 180.0);
        float c = cos(angle), s = sin(angle);
        uv = vec2(c * uv.x - s * uv.y, s * uv.x + c * uv.y);
    }

    // Offset and Pivot
    uv += vec2(0.5);
    uv += vec2(mat.uv_offset_x, mat.uv_offset_y);

    // Tiling
    float tx = (mat.uv_tiling_x != 0.0) ? mat.uv_tiling_x : 1.0;
    float ty = (mat.uv_tiling_y != 0.0) ? mat.uv_tiling_y : 1.0;
    uv *= vec2(tx, ty);

    return uv;
}

// UV of the material's texture SLOTS. A wet paint canvas material
// (paint_base_tex != 0, docs/dev/WET_PAINT_CANVAS_UV.md) keeps its slots at the
// raw mesh UV: they are paint canvas pages laid out on the mesh's own unwrap,
// and the UV transform belongs to the tiled base under them.
vec2 materialSlotUV(vec2 originalUV, const GpuMaterial mat) {
    return mat.paint_base_tex != 0u ? originalUV : applyUVTransform(originalUV, mat);
}

#endif
