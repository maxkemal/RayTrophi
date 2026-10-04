#pragma once

// Raster particle billboard hand-off (UI -> Vulkan viewport backend).
//
// Two ways a particle reaches the raster viewport, one appearance contract:
// - CPU quads (`additive` / `alpha`): the CPU sends particle CENTRES plus the
//   inputs of the appearance lookup; particle_viewport.vert expands the
//   camera-facing quad and reads colour / opacity / size / emission from the
//   LUT at (lut_row, age). For systems whose state is on the host.
// - Vertex pulling (`pulled`): a device-resident system is drawn straight from
//   the simulation's own storage buffers by particle_viewport_pull.vert. Only
//   possible when the simulation runs on the viewport's VkDevice; the builder
//   decides, so the backend never binds a foreign device's buffer.

#include <cstdint>
#include <vector>

struct SphereImpostorInstance {
    float centerRadius[4];
};
static_assert(sizeof(SphereImpostorInstance) == 16, "Sphere instance layout");

struct ParticleBillboardVertex {
    float center[3];   // world-space particle centre (same for all 6 vertices)
    float corner[2];   // [-1, 1] quad corner; also the sprite UV
    float age;         // normalized age, 0 = birth, 1 = death
    float lut_row;     // row in ParticleBillboardUpload::lut (integer valued)
    float size_scale;  // multiplies the LUT size (per-particle jitter)
};
static_assert(sizeof(ParticleBillboardVertex) == 8 * sizeof(float),
              "ParticleBillboardVertex must match the pipeline's vertex input layout");

// Binding order of particle_viewport_pull.vert set 1.
enum ParticlePulledStream : uint32_t {
    kPulledPositionX = 0,
    kPulledPositionY,
    kPulledPositionZ,
    kPulledAge,
    kPulledLifetime,
    kPulledAlive,
    kPulledProfile,
    kPulledSizeScale,
    kPulledStreamCount
};

struct ParticlePulledDraw {
    uint64_t buffers[kPulledStreamCount] = {};  // VkBuffer handles, simulation-owned
    uint32_t particle_count = 0;                // device slots (dead ones collapse)
    uint32_t lookup_offset = 0;                 // into ParticleBillboardUpload::row_lookup
    uint32_t lookup_count = 0;                  // entries = profile ids covered
    bool has_alpha = false;                     // any profile with alpha blend
    // Changes whenever the device state changed (a resident step ran), so the
    // backend re-renders only then instead of every frame.
    uint64_t state_version = 0;
};

// A render-only refinement of APIC carrier particles. RayFusion expands each
// parent deterministically in the vertex shader; no child list is stored and
// no child feeds back into the solver.
struct FluidSphereProxyDraw {
    uint64_t position_buffer = 0;  // simulation-owned VkBuffer of packed Vec3
    uint32_t parent_count = 0;
    uint32_t children_per_parent = 1;
    float child_radius = 0.0f;
    float spread_radius = 0.0f;
    float size_variation = 0.0f;
    uint64_t state_version = 0;
};

struct ParticleBillboardUpload {
    std::vector<SphereImpostorInstance> spheres;
    std::vector<ParticleBillboardVertex> additive;
    std::vector<ParticleBillboardVertex> alpha;
    // Concatenated LUT rows (kParticleAppearanceLutFloatsPerRow floats each).
    // Row 0 is always the fallback appearance.
    std::vector<float> lut;
    // Profile id -> LUT row, per pulled system (bit 31 = alpha blend).
    // Never empty: entry 0 keeps the binding valid.
    std::vector<uint32_t> row_lookup;
    std::vector<ParticlePulledDraw> pulled;
    std::vector<FluidSphereProxyDraw> fluid_sphere_proxies;
};
