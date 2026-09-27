#version 450
#extension GL_GOOGLE_include_directive : require

// Vertex pulling for device-resident particle systems (particle roadmap
// Phase 1.5 Batch B). No vertex buffer: the draw is 6 x capacity vertices and
// each vertex reads its particle straight from the simulation's storage
// buffers (set 1), which live on this same VkDevice. Nothing is read back to
// the host to draw a resident system.
//
// A draw covers ONE blend pass: particles whose profile has the other blend,
// and dead slots, collapse to a clipped point.

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec4 vColor;

layout(set = 0, binding = 0, std430) readonly buffer AppearanceLut {
    vec4 lut[];
};
// Profile id -> LUT row for this system (at pc.draw.x, pc.draw.y entries).
// Bit 31 set = alpha blend. Unknown ids read row 0 (fallback, additive).
layout(set = 0, binding = 1, std430) readonly buffer RowLookup {
    uint rowLookup[];
};

layout(set = 1, binding = 0) readonly buffer PX { float px[]; };
layout(set = 1, binding = 1) readonly buffer PY { float py[]; };
layout(set = 1, binding = 2) readonly buffer PZ { float pz[]; };
layout(set = 1, binding = 3) readonly buffer Age { float age[]; };
layout(set = 1, binding = 4) readonly buffer Lifetime { float lifetime[]; };
layout(set = 1, binding = 5) readonly buffer Alive { uint alive[]; };
layout(set = 1, binding = 6) readonly buffer Profile { uint profile[]; };
layout(set = 1, binding = 7) readonly buffer SizeScale { float sizeScale[]; };

#include "include/particle_appearance_lut.glsl"

const vec2 kCorners[6] = vec2[](
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));

void collapse() {
    // Outside the clip volume on every vertex: the triangle is discarded.
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    vUV = vec2(0.0);
    vColor = vec4(0.0);
}

void main() {
    uint p = uint(gl_VertexIndex) / 6u;
    vec2 corner = kCorners[uint(gl_VertexIndex) % 6u];
    if (p >= pc.draw.z || alive[p] == 0u) {
        collapse();
        return;
    }
    uint id = profile[p];
    uint entry = id < pc.draw.y ? rowLookup[pc.draw.x + id] : 0u;
    bool alphaBlend = (entry & 0x80000000u) != 0u;
    if (alphaBlend != (pc.draw.w == 1u)) {
        collapse();
        return;
    }
    vec3 center = vec3(px[p], py[p], pz[p]);
    if (any(isnan(center)) || any(isinf(center))) {
        collapse();
        return;
    }
    float life = lifetime[p];
    float normalizedAge = life > 1e-6 ? age[p] / life : 0.0;
    float halfSize;
    vColor = particleLook(int(entry & 0x7fffffffu), normalizedAge, sizeScale[p], halfSize);
    gl_Position = particleClip(center, corner, halfSize);
    vUV = corner;
}
