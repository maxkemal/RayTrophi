#version 450

// Edit-mesh overlay: shared vertex shader for the edge (LINE_LIST) and
// face-fill (TRIANGLE_LIST) pipelines. Positions are object-local; the
// model transform is pre-multiplied into pc.mvp on the CPU so object
// moves never force a vertex re-upload.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in uint inFlags; // bit0 = selected, bits 8..15 = soft-selection weight

layout(location = 0) out vec4 vColor;

layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 baseColor;
    vec4 selectColor;
    vec4 params;   // x = pointRadiusPx, y = viewportW, z = viewportH, w = mode (0 = per-vertex flags, 1 = force selectColor)
    vec4 params2;  // x = relative depth pull, y = soft-heat alpha base (0 = off), z = x-ray (>0.5), w reserved
} pc;

void main() {
    vec4 clip = pc.mvp * vec4(inPosition, 1.0);
    if (pc.params2.z > 0.5) {
        clip.z = 0.0; // x-ray: overlay always wins the depth test
    } else {
        // Distance-proportional pull toward the camera. A constant NDC offset
        // dwarfs the mesh's own depth range at typical view distances and
        // turns the overlay into accidental x-ray; scaling by |z| keeps the
        // pull just above z-fighting at every distance.
        clip.z -= pc.params2.x * abs(clip.z);
    }
    gl_Position = clip;

    vec4 col = pc.baseColor;
    if (pc.params.w > 0.5) {
        col = pc.selectColor;
    } else if ((inFlags & 1u) != 0u) {
        col = pc.selectColor;
    } else if (pc.params2.y > 0.0) {
        float soft = float((inFlags >> 8) & 255u) / 255.0;
        if (soft > 0.003) {
            // Same ramp as the ImGui weightToColor fallback (blue -> orange).
            vec3 heat = vec3(0.274 + 0.726 * soft,
                             0.863 - 0.176 * soft,
                             1.0   - 0.745 * soft);
            col = vec4(heat, max(0.08, pc.params2.y * soft));
        }
    }
    // Sculpt protection mask (bits 16..23): tint frozen regions cool grey so
    // the user can see what the brush will skip. Suppressed in force-select mode.
    if (pc.params.w <= 0.5) {
        float mask = float((inFlags >> 16) & 255u) / 255.0;
        if (mask > 0.01) {
            col.rgb = mix(col.rgb, vec3(0.20, 0.40, 0.62), mask * 0.8);
            col.a = max(col.a, mask * 0.85);
        }
    }
    vColor = col;
}
