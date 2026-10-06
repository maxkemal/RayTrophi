#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace RayTrophiSim::Fluid {

struct MatterEmissionRequest {
    std::size_t particles = 0;
    double weight = 1.0;
};

const char* validateMatterPoolWeight(float weight);

// Weighted max-min sharing of free capacity. Unused shares return to the pool.
// Rotation resolves integer ties without always favouring the first source.
std::vector<std::size_t> allocateMatterEmissionBudget(
    const std::vector<MatterEmissionRequest>& requests, std::size_t capacity,
    uint64_t rotation);

} // namespace RayTrophiSim::Fluid
