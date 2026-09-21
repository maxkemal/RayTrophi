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

#endif
