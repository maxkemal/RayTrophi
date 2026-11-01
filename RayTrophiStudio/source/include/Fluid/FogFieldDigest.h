#pragma once

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace RayTrophiSim::Fluid {

// On-demand fingerprint of every voxel, including source-empty temperature.
// Hash float bits in defined byte order; sums/counts cannot identify a field.
inline std::string fogFieldDigest(const std::vector<float>& values) {
    uint64_t hash = 14695981039346656037ull;
    for (float value : values) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= (bits >> (byte * 8)) & 0xffu;
            hash *= 1099511628211ull;
        }
    }
    std::ostringstream result;
    result << std::hex << std::setfill('0') << std::setw(16) << hash;
    return result.str();
}

} // namespace RayTrophiSim::Fluid
