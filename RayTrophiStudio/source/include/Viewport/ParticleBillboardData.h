#pragma once

// Raster particle billboard hand-off (UI -> Vulkan viewport backend).
//
// The CPU sends particle CENTRES plus the inputs of the appearance lookup; the
// vertex shader (shaders/particle_viewport.vert) expands the camera-facing quad
// and reads colour / opacity / size / emission from the LUT at (lut_row, age).
// Before Phase 1.5 the CPU lerped colour/size/opacity every step and expanded
// finished world-space corners; nothing on the GPU knew what a profile was.

#include <cstdint>
#include <vector>

struct ParticleBillboardVertex {
    float center[3];   // world-space particle centre (same for all 6 vertices)
    float corner[2];   // [-1, 1] quad corner; also the sprite UV
    float age;         // normalized age, 0 = birth, 1 = death
    float lut_row;     // row in ParticleBillboardUpload::lut (integer valued)
    float size_scale;  // multiplies the LUT size (per-particle jitter)
};
static_assert(sizeof(ParticleBillboardVertex) == 8 * sizeof(float),
              "ParticleBillboardVertex must match the pipeline's vertex input layout");

struct ParticleBillboardUpload {
    std::vector<ParticleBillboardVertex> additive;
    std::vector<ParticleBillboardVertex> alpha;
    // Concatenated LUT rows (kParticleAppearanceLutFloatsPerRow floats each).
    // Row 0 is always the fallback appearance.
    std::vector<float> lut;
};
