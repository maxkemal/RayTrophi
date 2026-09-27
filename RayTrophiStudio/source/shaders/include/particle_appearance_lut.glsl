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

vec4 particleClip(vec3 center, vec2 corner, float halfSize) {
    vec3 world = center + (pc.cameraRight.xyz * corner.x + pc.cameraUp.xyz * corner.y) * halfSize;
    return pc.viewProj * vec4(world, 1.0);
}
