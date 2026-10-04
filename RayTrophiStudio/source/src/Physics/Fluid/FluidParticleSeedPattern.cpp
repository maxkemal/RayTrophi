#include "Fluid/FluidParticleSeedPattern.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace RayTrophiSim::Fluid {
namespace {

std::uint32_t mixBits(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

std::uint32_t cellHash(int x, int y, int z, std::uint32_t seed) {
    std::uint32_t value = seed ^ 0x9e3779b9u;
    value = mixBits(value ^ (static_cast<std::uint32_t>(x) * 0x85ebca6bu));
    value = mixBits(value ^ (static_cast<std::uint32_t>(y) * 0xc2b2ae35u));
    return mixBits(value ^ (static_cast<std::uint32_t>(z) * 0x27d4eb2fu));
}

float slotDistanceSquared(const FluidSeedSlot& a, const FluidSeedSlot& b) {
    const float dx = static_cast<float>(a.x - b.x);
    const float dy = static_cast<float>(a.y - b.y);
    const float dz = static_cast<float>(a.z - b.z);
    return dx * dx + dy * dy + dz * dz;
}

} // namespace

FluidSeedPattern buildFluidSeedPattern(int particles_per_cell) {
    FluidSeedPattern pattern;
    const int count = std::max(particles_per_cell, 1);
    pattern.subdivisions = std::max(
        1,
        static_cast<int>(std::ceil(std::cbrt(static_cast<double>(count)))));

    std::vector<FluidSeedSlot> candidates;
    candidates.reserve(static_cast<std::size_t>(pattern.subdivisions) *
                       static_cast<std::size_t>(pattern.subdivisions) *
                       static_cast<std::size_t>(pattern.subdivisions));
    for (int z = 0; z < pattern.subdivisions; ++z) {
        for (int y = 0; y < pattern.subdivisions; ++y) {
            for (int x = 0; x < pattern.subdivisions; ++x) {
                candidates.push_back({x, y, z});
            }
        }
    }

    pattern.slots.reserve(static_cast<std::size_t>(count));
    while (static_cast<int>(pattern.slots.size()) < count && !candidates.empty()) {
        std::size_t best_index = 0;
        float best_distance = -1.0f;
        float best_total_distance = -1.0f;
        for (std::size_t candidate_index = 0;
             candidate_index < candidates.size();
             ++candidate_index) {
            float nearest = std::numeric_limits<float>::max();
            float total_distance = 0.0f;
            if (pattern.slots.empty()) {
                nearest = 0.0f;
            } else {
                for (const FluidSeedSlot& selected : pattern.slots) {
                    const float distance =
                        slotDistanceSquared(candidates[candidate_index], selected);
                    nearest = std::min(nearest, distance);
                    total_distance += distance;
                }
            }
            if (nearest > best_distance ||
                (nearest == best_distance && total_distance > best_total_distance)) {
                best_distance = nearest;
                best_total_distance = total_distance;
                best_index = candidate_index;
            }
        }
        pattern.slots.push_back(candidates[best_index]);
        candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(best_index));
    }
    return pattern;
}

FluidSeedSlot orientFluidSeedSlot(const FluidSeedSlot& slot,
                                  int subdivisions,
                                  int cell_x,
                                  int cell_y,
                                  int cell_z,
                                  std::uint32_t seed) {
    const std::uint32_t hash = cellHash(cell_x, cell_y, cell_z, seed);
    const int source[3] = {slot.x, slot.y, slot.z};
    static constexpr int permutations[6][3] = {
        {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
        {1, 2, 0}, {2, 0, 1}, {2, 1, 0}
    };
    const int* permutation = permutations[hash % 6u];
    FluidSeedSlot result{
        source[permutation[0]],
        source[permutation[1]],
        source[permutation[2]]
    };
    const int high = std::max(subdivisions - 1, 0);
    if ((hash & (1u << 3u)) != 0u) result.x = high - result.x;
    if ((hash & (1u << 4u)) != 0u) result.y = high - result.y;
    if ((hash & (1u << 5u)) != 0u) result.z = high - result.z;
    return result;
}

} // namespace RayTrophiSim::Fluid
