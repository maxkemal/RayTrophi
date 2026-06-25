#version 450

// Selection-outline composite: screen-space edge detect on the R8G8 mask,
// blended over the finished viewport color image.
//   mask.g = full silhouette (no depth test)  — 1.0 primary, 0.5 secondary
//   mask.r = depth-tested visible coverage
// A pixel is on the outline when the silhouette boundary crosses its
// neighborhood (some samples inside, some outside). Where the local visible
// coverage is zero the object is behind something there, so the outline
// switches to the desaturated "occluded" style — the per-pixel equivalent of
// the old CPU BVH occlusion probes.

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D maskTex;

layout(push_constant) uniform SelectionOutlinePushConstants {
    vec4 primaryColor;
    vec4 secondaryColor;
    vec4 occludedColor;
    vec4 params; // x = outline thickness (px), y/z = texel size, w unused
} pc;

void main() {
    const vec2 texel = pc.params.yz;
    // Kernel radius in pixels: half the requested thickness (outline grows
    // symmetrically outward+inward from the silhouette boundary).
    const float radius = max(1.0, pc.params.x * 0.5 + 0.5);

    float gMax = 0.0;
    float gMin = 1.0;
    float rMax = 0.0;

    // 5x5 neighborhood scaled to the requested thickness. The mask sampler is
    // bilinear, so fractional offsets soften the boundary into cheap AA.
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            vec2 uv = vUV + vec2(float(dx), float(dy)) * (radius * 0.5) * texel;
            vec2 m = texture(maskTex, uv).rg;
            gMax = max(gMax, m.g);
            gMin = min(gMin, m.g);
            rMax = max(rMax, m.r);
        }
    }

    // Boundary straddle test; bilinear filtering makes gMax/gMin partially
    // fractional near the edge, which feeds directly into coverage alpha.
    float edge = clamp((gMax - gMin) * 2.0, 0.0, 1.0) * step(0.004, gMax) * step(gMin, 0.996);
    if (edge <= 0.0) {
        discard;
    }

    vec4 tier = (gMax >= 0.75) ? pc.primaryColor : pc.secondaryColor;
    // Occluded portion: silhouette present but no visible (depth-tested)
    // coverage anywhere in the neighborhood.
    vec4 col = (rMax > 0.004) ? tier : pc.occludedColor;
    outColor = vec4(col.rgb, col.a * edge);
}
