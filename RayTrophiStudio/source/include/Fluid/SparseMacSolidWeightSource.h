#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace RayTrophiSim::Fluid {
inline constexpr uint32_t kSolidWeightWordsPerAxis = 144;
inline constexpr uint32_t kSolidWeightWordsPerTile = 432;

// Physical-tile lookup followed by byte-packed XYZ ownership pages. Lookup 0
// means fully open (255). Nonzero lookup values are absolute uint-word offsets.
// This immutable collider metadata is not a dense floating-point MAC field.
bool packSparseMacSolidWeightSource(const std::array<int, 3>& cells,
    const std::array<const std::vector<uint8_t>*, 3>& weights,
    std::vector<uint32_t>& output, std::string& error, uint64_t max_bytes = 0);
} // namespace RayTrophiSim::Fluid
