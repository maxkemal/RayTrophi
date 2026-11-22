#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace RayTrophiSim {

// Capabilities precede every other instruction in a SPIR-V module. These
// wire values keep the check independent of optional SPIR-V SDK headers.
constexpr const char* simulationSpirvRejection(const std::vector<uint32_t>& words,
                                               bool atomicFloatAdd, bool shaderFloat64) {
    constexpr uint32_t magic = 0x07230203;
    constexpr uint32_t opCapability = 17;
    constexpr uint32_t float64 = 10;
    constexpr uint32_t atomicFloat32AddEXT = 6033;
    if (words.size() < 5 || words[0] != magic) {
        return "module.invalid_spirv";
    }

    for (std::size_t index = 5; index < words.size();) {
        const uint32_t instruction = words[index];
        const uint32_t wordCount = instruction >> 16;
        if (wordCount == 0 || wordCount > words.size() - index) {
            return "module.invalid_spirv";
        }
        if ((instruction & 0xffff) != opCapability) {
            break;
        }
        if (wordCount != 2) {
            return "module.invalid_spirv";
        }
        const uint32_t capability = words[index + 1];
        if (capability == atomicFloat32AddEXT && !atomicFloatAdd) {
            return "module.AtomicFloat32AddEXT_not_supported_on_this_GPU";
        }
        if (capability == float64 && !shaderFloat64) {
            return "module.Float64_not_supported_on_this_GPU";
        }
        index += wordCount;
    }
    return nullptr;
}

} // namespace RayTrophiSim
