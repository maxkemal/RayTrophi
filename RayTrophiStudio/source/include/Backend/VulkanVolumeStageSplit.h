#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace VulkanRT {

// Process-start developer opt-in until split-stage driver/image acceptance.
bool configureSplitVolumeShaders(const std::string& shaderDirectory,
    const std::vector<std::uint32_t>& raygen,
    std::vector<std::uint32_t>& fog, std::vector<std::uint32_t>& surface);

constexpr std::uint32_t routedVolumeSbtOffset(bool split, bool volume,
    bool hair, bool sphere, std::uint32_t mask, std::uint32_t original) {
    if (volume && (mask == 0x02u || mask == 0x08u)) {
        if (split && mask == 0x08u) {
            return 2u + (hair ? 1u : 0u) + (sphere ? 1u : 0u);
        }
        return 1u;
    }
    return original;
}

} // namespace VulkanRT
