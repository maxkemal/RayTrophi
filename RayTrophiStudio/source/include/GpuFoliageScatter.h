#pragma once

#include <cstdint>
#include <cstring>

struct InstanceGroup;
struct TerrainObject;

namespace FoliageGPU {

struct ParityStats {
    bool gpuAvailable = false;
    bool completed = false;
    uint32_t candidateCount = 0;
    uint32_t rngMismatches = 0;
    uint32_t acceptanceMismatches = 0;
    uint32_t rejectionMaskMismatches = 0;
    double cpuMs = 0.0;
    double gpuDispatchReadbackMs = 0.0;
};

ParityStats runParityTest(uint32_t candidateCount = 1000000u);
const ParityStats& getLastParityStats();

struct TerrainScatterStats {
    bool gpuAvailable = false;
    bool gpuPathUsed = false;
    uint32_t candidateCount = 0;
    uint32_t acceptedBeforeSpacing = 0;
    uint32_t spawned = 0;
    uint64_t uploadBytes = 0;
    double gpuMs = 0.0;
    double cpuCompactMs = 0.0;
};

int scatterFillTerrainGPU(::InstanceGroup& group, ::TerrainObject* terrain, bool& attempted);
const TerrainScatterStats& getLastTerrainScatterStats();

enum class ScatterMode : uint32_t {
    TerrainFill = 0,
    TerrainBrushAdd = 1,
    TerrainBrushErase = 2,
    MeshFill = 3,
    MeshBrushAdd = 4,
    MeshBrushErase = 5
};

enum ScatterFlags : uint32_t {
    HasDensityMask       = 1u << 0,
    HasExclusionMask     = 1u << 1,
    HasScaleMask         = 1u << 2,
    HasSplatInclude      = 1u << 3,
    HasSplatExclusion    = 1u << 4,
    CheckMinimumDistance = 1u << 5,
    AlignToNormal        = 1u << 6,
    AllowRidges          = 1u << 7,
    AllowFlats           = 1u << 8,
    AllowGullies         = 1u << 9
};

// Shared CPU/GLSL ABI. Every member is a vec4/uvec4-sized lane so std430 has
// no compiler-dependent padding. Matrix arrays are column-neutral raw values;
// the shader helpers explicitly apply the same row-major convention as Matrix4x4.
struct alignas(16) ScatterSettingsGPU {
    uint32_t meta[4]{};          // mode, targetCount, seed, candidateCount
    uint32_t terrain[4]{};       // width, height, sourceCount, ScatterFlags
    float worldToLocal[16]{};
    float localToWorld[16]{};
    float heightSlopeEdge[4]{};  // heightMin, heightMax, slopeMaxDeg, edgeMarginUV
    float curvature[4]{};        // min, max, step, reserved
    float direction[4]{};        // angleDeg, influence, reserved, reserved
    float scale[4]{};            // min, max, normalInfluence, density
    float rotation[4]{};         // randomY, randomXZ, reserved, reserved
    float offsetMask[4]{};       // yOffsetMin, yOffsetMax, exclusionThreshold, scaleMaskInfluence
    float brush[4]{};            // centerX, centerZ, radius, falloff
    int32_t maskSlots[4]{-1,-1,-1,-1}; // density, exclusion, scale, splat include/exclude packed separately
};
static_assert(sizeof(ScatterSettingsGPU) == 288, "ScatterSettingsGPU std430 ABI changed");

struct alignas(16) ScatterInstanceGPU {
    float position[4]{};         // xyz, alive (0/1)
    float rotation[4]{};         // xyz degrees, candidate id bits in w
    float scaleSource[4]{};      // xyz, source index bits in w
};
static_assert(sizeof(ScatterInstanceGPU) == 48, "ScatterInstanceGPU std430 ABI changed");

struct alignas(16) ScatterDecisionGPU {
    uint32_t accepted = 0;
    uint32_t rejectionMask = 0;
    uint32_t candidateId = 0;
    uint32_t reserved = 0;
};
static_assert(sizeof(ScatterDecisionGPU) == 16, "ScatterDecisionGPU std430 ABI changed");

// Integer-only candidate RNG shared verbatim with GLSL. It is independent of
// execution order, so parallel dispatch and prefix compaction remain reproducible.
inline uint32_t hash32(uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

inline uint32_t candidateRandomBits(uint32_t seed, uint32_t candidateId, uint32_t stream) {
    return hash32(seed ^ hash32(candidateId + 0x9e3779b9u * (stream + 1u)));
}

inline float candidateRandom01(uint32_t seed, uint32_t candidateId, uint32_t stream) {
    const uint32_t mantissa = candidateRandomBits(seed, candidateId, stream) >> 8u;
    return static_cast<float>(mantissa) * (1.0f / 16777216.0f);
}

inline float uintBitsToFloat(uint32_t value) {
    float result = 0.0f;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

} // namespace FoliageGPU
