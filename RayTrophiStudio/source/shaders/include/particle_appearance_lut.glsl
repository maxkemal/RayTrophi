// Particle appearance LUT lookup shared by the raster billboard vertex shaders
// (particle_viewport.vert: CPU-built quads, particle_viewport_pull.vert: reads
// the simulation buffers). One copy of the interpolation, and it must match
// sampleParticleAppearanceLut on the CPU.
//
// LUT layout (ParticleAppearanceProfile.h, keep in sync): a row is 64 samples,
// a sample is two vec4: [0] = (r, g, b, opacity), [1] = (size, emission, 0, 0).
// Sample s sits at age s / 63; neighbours are interpolated linearly.
//
// The including shader declares `vec4 lut[]` (set 0, binding 0) and the push
// constant block below.

layout(push_constant) uniform ParticlePC {
    mat4 viewProj;
    vec4 cameraRight;   // xyz: first row of the view rotation
    vec4 cameraUp;      // xyz: second row of the view rotation
    // x: row-lookup offset, y: row-lookup count, z: particle count (pull only),
    // w: blend pass drawn by this call (0 additive, 1 alpha; pull only)
    uvec4 draw;
} pc;

const int kParticleLutSamples = 64;

// Returns colour * emission in rgb, opacity in a; writes the billboard half
// size (metres) to halfSize.
vec4 particleLook(int row, float normalizedAge, float sizeScale, out float halfSize) {
    float age = clamp(normalizedAge, 0.0, 1.0);
    float x = age * float(kParticleLutSamples - 1);
    int i0 = min(int(x), kParticleLutSamples - 1);
    int i1 = min(i0 + 1, kParticleLutSamples - 1);
    float w = x - float(i0);
    int base = row * kParticleLutSamples * 2;
    vec4 a0 = lut[base + i0 * 2];
    vec4 a1 = lut[base + i0 * 2 + 1];
    vec4 b0 = lut[base + i1 * 2];
    vec4 b1 = lut[base + i1 * 2 + 1];
    vec4 colorOpacity = mix(a0, b0, w);
    vec2 sizeEmission = mix(a1.xy, b1.xy, w);
    halfSize = 0.5 * sizeEmission.x * max(sizeScale, 0.0);
    return vec4(colorOpacity.rgb * sizeEmission.y, colorOpacity.a);
}

// Lit-drop lighting (ParticleBillboardBuilder.cpp markLitRow). The builder
// writes into the unused [1].zw slots of a row's samples:
//   sample 0: (roughness, lit)   sample 1: (light count, reference E)
//   light i (< 8): 2+3i: (pos.x, pos.y)  3+3i: (pos.z, R)  4+3i: (G, B)
// RGB = colour * intensity, the RT point falloff / d^2 (a directional light is
// a point 1e4 m away). Weighed at the particle centre (a drop is small): the
// two strongest lights come out as direction + colour, the rest tint the
// ambient. Brightness is RELATIVE (the overlay is drawn after the tonemap):
// sqrt(E / reference E), clamped. Directions in the billboard frame (x camera
// right, y camera up, z toward the camera).
// ★ OFF (2026-11-18, user): the lit bead read as distinct spheres and broke the
// look of flowing paint; the flat soft sprite is back. The lighting below and
// the builder's row data stay for a later, stronger version (velocity-aligned
// stretch needs the velocity stream in the pulled draw - a full rebuild).
// 1 = lit drops for non-emissive sphere systems, 0 = every billboard flat.
#define PARTICLE_LIT_DROPS 0

vec2 lutZW(int base, int sample_index) { return lut[base + sample_index * 2 + 1].zw; }

float lum3(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void particleLighting(int row, vec3 center, out vec4 shade, out vec3 l1, out vec3 c1,
                      out vec3 l2, out vec3 c2, out vec3 amb, out vec3 upB) {
    int base = row * kParticleLutSamples * 2;
    vec2 s0 = lutZW(base, 0);
    shade = vec4(s0.x, s0.y, 0.0, 0.0);
    l1 = vec3(0.0, 0.0, 1.0); l2 = l1;
    c1 = vec3(0.0); c2 = vec3(0.0); amb = vec3(0.0);
    vec3 r = pc.cameraRight.xyz;
    vec3 u = pc.cameraUp.xyz;
    vec3 f = cross(r, u);   // view +z: from the scene toward the camera
    upB = vec3(r.y, u.y, f.y);
    if (s0.y < 0.5) return;
#if !PARTICLE_LIT_DROPS
    shade.y = 0.0;
    return;
#endif
    vec2 s1 = lutZW(base, 1);
    int count = clamp(int(s1.x + 0.5), 0, 8);
    float e_ref = max(s1.y, 1e-12);
    vec3 total = vec3(0.0);
    float e1 = -1.0, e2 = -1.0;
    vec3 d1 = vec3(0.0, 1.0, 0.0), d2 = d1, r1 = vec3(0.0), r2 = vec3(0.0);
    for (int i = 0; i < count; ++i) {
        vec2 a = lutZW(base, 2 + 3 * i);
        vec2 b = lutZW(base, 3 + 3 * i);
        vec2 c = lutZW(base, 4 + 3 * i);
        vec3 d = vec3(a.x, a.y, b.x) - center;
        float dd = max(dot(d, d), 0.04);
        vec3 rad = vec3(b.y, c.x, c.y) / dd;
        float e = lum3(rad);
        vec3 dir = d * inversesqrt(dd);
        total += rad;
        if (e > e1) { e2 = e1; d2 = d1; r2 = r1; e1 = e; d1 = dir; r1 = rad; }
        else if (e > e2) { e2 = e; d2 = dir; r2 = rad; }
    }
    float e_total = lum3(total);
    if (!(e_total > 0.0)) {
        // No usable light (none in the scene, or an older builder's layout):
        // a neutral key from above-front, never a black drop.
        vec3 key = normalize(vec3(0.45, 0.8, 0.35));
        l1 = vec3(dot(key, r), dot(key, u), dot(key, f));
        c1 = vec3(0.8);
        amb = vec3(0.4);
        return;
    }
    float bright = clamp(sqrt(e_total / e_ref), 0.55, 1.6);
    float k = bright / e_total;
    c1 = r1 * k;
    c2 = max(e2, 0.0) > 0.0 ? r2 * k : vec3(0.0);
    // Lights beyond the two strongest fill the ambient in their own colour; a
    // little of the whole mix fills it too (room bounce).
    amb = max(total - r1 - max(r2, vec3(0.0)), vec3(0.0)) * k * 0.6 + total * k * 0.22;
    l1 = vec3(dot(d1, r), dot(d1, u), dot(d1, f));
    l2 = vec3(dot(d2, r), dot(d2, u), dot(d2, f));
}

vec2 particleEffectCorner(vec2 corner, float opacity, float age, float scale,
                          inout vec4 shade) {
    if (shade.y < 0.5 && shade.x >= 3.0) {
        // Confetti uses a stable silhouette variation from its per-particle size.
        shade.x = 3.0 + floor(3.0 * fract(scale * 17.413));
        float first = 1.0 - smoothstep(0.01, 0.082, abs(age - 0.20));
        float second = 1.0 - smoothstep(0.01, 0.075, abs(age - 0.56));
        shade.z = max(first, second * 0.68);
        shade.w = 1.0 + 0.62 * shade.z;
        corner.x *= shade.w;
    } else if (shade.y < 0.5 && shade.x > 1.5) {
        shade.z = smoothstep(0.43, 0.72, opacity);
        shade.w = 1.0 + 1.8 * shade.z;
        corner.x *= shade.w;
    }
    return corner;
}

vec4 particleClip(vec3 center, vec2 corner, float halfSize) {
    vec3 world = center + (pc.cameraRight.xyz * corner.x + pc.cameraUp.xyz * corner.y) * halfSize;
    return pc.viewProj * vec4(world, 1.0);
}
