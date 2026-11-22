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
layout(location = 2) flat out vec4 vShade;   // roughness, lit
layout(location = 3) flat out vec3 vL1;      // strongest light, billboard frame
layout(location = 4) flat out vec3 vC1;      // its relative radiance
layout(location = 5) flat out vec3 vL2;      // second light
layout(location = 6) flat out vec3 vC2;
layout(location = 7) flat out vec3 vAmb;     // the other lights + bounce
layout(location = 8) flat out vec3 vUp;      // world up, billboard frame

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
    vShade = vec4(0.0);
    vL1 = vec3(0.0, 0.0, 1.0); vL2 = vL1;
    vC1 = vec3(0.0); vC2 = vec3(0.0); vAmb = vec3(0.0);
    vUp = vec3(0.0, 1.0, 0.0);
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
    int row = int(entry & 0x7fffffffu);
    vColor = particleLook(row, normalizedAge, sizeScale[p], halfSize);
    particleLighting(row, center, vShade, vL1, vC1, vL2, vC2, vAmb, vUp);
    vUV = particleEffectCorner(corner, vColor.a, normalizedAge, sizeScale[p], vShade);
    gl_Position = particleClip(center, vUV, halfSize);
}
