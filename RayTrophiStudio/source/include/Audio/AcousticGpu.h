#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace audio {
inline constexpr std::size_t kAcousticGpuMaxInstances = 4096;
// Derived GPU views, never owners of scene geometry or acceleration structures.
struct alignas(16) AcousticGpuInstance {
    std::uint64_t positions = 0;
    std::uint64_t indices = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t triangleCount = 0;
    std::uint32_t object = 0;
    std::uint32_t defined = 0;
    float energy[8]{};
};
static_assert(sizeof(AcousticGpuInstance) == 64, "Acoustic instance shader ABI");

struct AcousticGpuGeometry {
    std::vector<AcousticGpuInstance> instances;
    std::vector<std::string> names;
    std::string revision;
    std::string error;
};

struct alignas(16) AcousticGpuRay {
    float source[4]{}; // xyz, meters per scene unit
    float listener[4]{}; // xyz, object-barrier mode (otherwise thin surfaces)
};
static_assert(sizeof(AcousticGpuRay) == 32, "Acoustic ray shader ABI");

// Exactly 64 bytes per source, at most four sources per submitted batch.
struct alignas(16) AcousticGpuSummary {
    float energy[8]{};
    float path[4]{}; // distance metres, delay seconds, reserved
    std::uint32_t state[4]{}; // error, retained crossings, raw candidates, reserved
};
static_assert(sizeof(AcousticGpuSummary) == 64, "Acoustic result shader ABI");

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
