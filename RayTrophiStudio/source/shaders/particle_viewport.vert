#version 450

// Camera-facing particle billboards (particle roadmap Phase 1.5 Batch A).
// The CPU sends particle CENTRES; this stage expands the quad and reads the
// particle's look from the appearance LUT at (lut_row, normalized age).
//
// LUT layout (ParticleAppearanceProfile.h, keep in sync): a row is 64 samples,
// a sample is two vec4: [0] = (r, g, b, opacity), [1] = (size, emission, 0, 0).
// Sample s sits at age s / 63; neighbours are interpolated linearly, exactly
// like sampleParticleAppearanceLut on the CPU.

layout(location = 0) in vec3 inCenter;   // world-space particle centre
layout(location = 1) in vec2 inCorner;   // [-1,1] quad corner (also the UV)
layout(location = 2) in vec3 inLook;     // age, lut_row, size_scale

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec4 vColor;

layout(set = 0, binding = 0, std430) readonly buffer AppearanceLut {
    vec4 lut[];
};

layout(push_constant) uniform PC {
    mat4 viewProj;
    mat4 view;
} pc;

const int kSamples = 64;

void main() {
    float age = clamp(inLook.x, 0.0, 1.0);
    int row = int(inLook.y + 0.5);
    float sizeScale = max(inLook.z, 0.0);

    float x = age * float(kSamples - 1);
    int i0 = min(int(x), kSamples - 1);
    int i1 = min(i0 + 1, kSamples - 1);
    float w = x - float(i0);
    int base = row * kSamples * 2;
    vec4 a0 = lut[base + i0 * 2];
    vec4 a1 = lut[base + i0 * 2 + 1];
    vec4 b0 = lut[base + i1 * 2];
    vec4 b1 = lut[base + i1 * 2 + 1];
    vec4 colorOpacity = mix(a0, b0, w);
    vec2 sizeEmission = mix(a1.xy, b1.xy, w);

    // Camera right / up are the first two rows of the view rotation.
    vec3 right = vec3(pc.view[0][0], pc.view[1][0], pc.view[2][0]);
    vec3 up = vec3(pc.view[0][1], pc.view[1][1], pc.view[2][1]);
    float halfSize = 0.5 * sizeEmission.x * sizeScale;
    vec3 world = inCenter + (right * inCorner.x + up * inCorner.y) * halfSize;

    gl_Position = pc.viewProj * vec4(world, 1.0);
    vUV = inCorner;
    vColor = vec4(colorOpacity.rgb * sizeEmission.y, colorOpacity.a);
}
