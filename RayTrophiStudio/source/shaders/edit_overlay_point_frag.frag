#version 450

// Edit-mesh overlay: anti-aliased disc with a dark rim, carved from the
// billboard quad. Replaces the per-vertex double AddCircleFilled the ImGui
// overlay used to draw.

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vUV;

layout(location = 0) out vec4 outColor;

void main() {
    float d = length(vUV); // 0 at centre, 1 at quad edge
    float aa = fwidth(d);
    float alpha = 1.0 - smoothstep(1.0 - aa, 1.0, d);
    if (alpha <= 0.0) {
        discard;
    }
    // Dark rim keeps markers readable on bright surfaces.
    float rim = smoothstep(0.58, 0.92, d);
    vec3 rgb = mix(vColor.rgb, vec3(0.06, 0.08, 0.10), rim * 0.85);
    outColor = vec4(rgb, vColor.a * alpha);
}
