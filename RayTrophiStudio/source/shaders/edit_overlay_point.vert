#version 450

// Edit-mesh overlay: vertex markers as instanced screen-space billboards.
// One instance per editable vertex (position + flags are per-instance
// attributes); gl_VertexIndex 0..5 expands a two-triangle quad with a
// constant pixel radius. The fragment shader carves an AA disc out of it.

layout(location = 0) in vec3 inPosition; // per-instance
layout(location = 1) in uint inFlags;    // per-instance: bit0 = selected, bits 8..15 = soft weight

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vUV; // -1..1 across the quad

layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 baseColor;
    vec4 selectColor;
    vec4 params;   // x = pointRadiusPx, y = viewportW, z = viewportH, w = mode
    vec4 params2;  // x = relative depth pull, y = soft-heat alpha base (0 = off), z = x-ray (>0.5), w reserved
} pc;

const vec2 kCorners[6] = vec2[](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0)
);

void main() {
    vec4 clip = pc.mvp * vec4(inPosition, 1.0);
    if (pc.params2.z > 0.5) {
        clip.z = 0.0; // x-ray: marker always wins the depth test
    } else {
        clip.z -= pc.params2.x * abs(clip.z); // distance-proportional pull (see edit_overlay.vert)
    }

    bool selected = (inFlags & 1u) != 0u;
    float radiusPx = pc.params.x * (selected ? 1.35 : 1.0);

    vec2 corner = kCorners[gl_VertexIndex];
    // Offset in clip space scaled by w so the marker stays a constant pixel size.
    clip.xy += corner * radiusPx * 2.0 * clip.w / vec2(pc.params.y, pc.params.z);
    gl_Position = clip;
    vUV = corner;

    vec4 col = selected ? pc.selectColor : pc.baseColor;
    if (!selected && pc.params2.y > 0.0) {
        float soft = float((inFlags >> 8) & 255u) / 255.0;
        if (soft > 0.003) {
            // Same ramp as the ImGui weightToColor fallback (blue -> orange).
            vec3 heat = vec3(0.274 + 0.726 * soft,
                             0.863 - 0.176 * soft,
                             1.0   - 0.745 * soft);
            col = vec4(heat, max(0.08, pc.params2.y * soft));
        }
    }
    // Sculpt protection mask (bits 16..23): tint frozen vertex markers cool grey.
    float mask = float((inFlags >> 16) & 255u) / 255.0;
    if (mask > 0.01) {
        col.rgb = mix(col.rgb, vec3(0.20, 0.40, 0.62), mask * 0.85);
        col.a = max(col.a, mask * 0.9);
    }
    vColor = col;
}
