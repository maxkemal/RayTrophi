#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace RayTrophiSim::Fluid {

// Domain-local, monotonically allocated. The allocator is part of the particle
// snapshot, so restoring a cache frame restores both identities and allocation.
inline uint64_t allocateMatterParticleId(uint64_t& next_id) {
    if (next_id == 0 || next_id == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error("matter particle identity exhausted");
    }
    return next_id++;
}

inline void extendMatterParticleIds(std::vector<uint64_t>& ids,
                                   uint64_t& next_id, std::size_t count) {
    if (ids.size() > count) {
        throw std::logic_error("matter identity sidecar exceeds particle count");
    }
    while (ids.size() < count) {
        ids.push_back(allocateMatterParticleId(next_id));
    }
}

} // namespace RayTrophiSim::Fluid
