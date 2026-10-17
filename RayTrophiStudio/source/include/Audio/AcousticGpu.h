#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace audio {
inline constexpr std::size_t kAcousticGpuMaxInstances = 4096;
inline constexpr std::size_t kAcousticGpuMaxSources = 32;
inline constexpr std::uint32_t kAcousticIgnoredVisual = 2;
inline constexpr std::size_t kAcousticDiffuseBins = 8;
inline constexpr std::size_t kAcousticPlaybackTaps = 6 + kAcousticDiffuseBins;
// Derived GPU views, never owners of scene geometry or acceleration structures.
struct alignas(16) AcousticGpuInstance {
    std::uint64_t positions = 0;
    std::uint64_t indices = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t triangleCount = 0;
    std::uint32_t object = 0;
    std::uint32_t defined = 0; // 0 unknown, 1 boundary, 2 ignored simulation visual
    float energy[8]{};
    float reflection[8]{};
    float diffuse[8]{};
};
static_assert(sizeof(AcousticGpuInstance) == 128, "Acoustic instance shader ABI");
static_assert(offsetof(AcousticGpuInstance, diffuse) == 96);

struct AcousticGpuGeometry {
    std::vector<AcousticGpuInstance> instances;
    std::vector<std::string> names;
    std::string revision;
    std::string boundaryRevision;
    std::string error;
    std::uint32_t ignoredSimulationInstances = 0;
    std::uint32_t ignoredSimulationObjects = 0;
};

struct alignas(16) AcousticGpuRay {
    float source[4]{}; // xyz, meters per scene unit
    float listener[4]{}; // xyz, object-barrier mode (otherwise thin surfaces)
    float options[4]{}; // specular enabled, max excess seconds, diffuse enabled, reserved
};
static_assert(sizeof(AcousticGpuRay) == 48, "Acoustic ray shader ABI");

struct alignas(16) AcousticGpuReflection {
    float energy[8]{};
    float path[4]{}; // distance metres, excess delay seconds, valid, kind (0 specular, 1 diffuse)
};
static_assert(sizeof(AcousticGpuReflection) == 48);

// Six specular candidates and eight diffuse delay bins per source.
struct alignas(16) AcousticGpuSummary {
    float energy[8]{};
    float path[4]{}; // distance metres, delay seconds, reserved
    std::uint32_t state[4]{}; // error, retained crossings, raw candidates, reserved
    AcousticGpuReflection reflections[kAcousticPlaybackTaps]{};
    std::uint32_t diffuseState[4]{}; // samples, contributions, rejected queries, near receiver
};
static_assert(sizeof(AcousticGpuSummary) == 752, "Acoustic result shader ABI");
static_assert(offsetof(AcousticGpuSummary, reflections) == 64);
static_assert(offsetof(AcousticGpuSummary, diffuseState) == 736);

struct AcousticGpuBatch {
    std::vector<AcousticGpuInstance> instances;
    std::vector<AcousticGpuRay> rays;
    std::vector<std::string> ids;
    std::string stamp;
    std::function<bool()> stillCurrent;
};

struct AcousticGpuCompletion {
    std::vector<AcousticGpuSummary> summaries;
    std::vector<std::string> ids;
    std::string stamp;
};
} // namespace audio
