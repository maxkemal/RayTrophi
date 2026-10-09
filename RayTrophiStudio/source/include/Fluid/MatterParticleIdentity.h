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

// Identity -> index over one snapshot of ids, open addressing in two flat
// arrays. A std::unordered_map of 1.3M grains allocated a node per id and was
// a large share of each frame's host time. Identity 0 is never allocated
// (allocateMatterParticleId), so it marks an empty slot.
class MatterParticleIdIndex {
public:
    static constexpr uint32_t kMissing = 0xffffffffu;

    // False when an id repeats (or is 0): the snapshot is not an identity set.
    bool build(const std::vector<uint64_t>& ids) {
        if (ids.size() >= kMissing) {
            throw std::length_error("matter identity index exceeds 32-bit indices");
        }
        std::size_t capacity = 16;
        while (capacity < 2 * ids.size()) {
            capacity *= 2;
        }
        keys_.assign(capacity, 0);
        values_.assign(capacity, kMissing);
        mask_ = capacity - 1;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == 0) {
                return false;
            }
            std::size_t slot = home(ids[i]);
            while (keys_[slot] != 0) {
                if (keys_[slot] == ids[i]) {
                    return false;
                }
                slot = (slot + 1) & mask_;
            }
            keys_[slot] = ids[i];
            values_[slot] = static_cast<uint32_t>(i);
        }
        return true;
    }

    uint32_t find(uint64_t id) const {
        if (keys_.empty() || id == 0) {
            return kMissing;
        }
        for (std::size_t slot = home(id); keys_[slot] != 0; slot = (slot + 1) & mask_) {
            if (keys_[slot] == id) {
                return values_[slot];
            }
        }
        return kMissing;
    }

private:
    std::size_t home(uint64_t id) const {
        return static_cast<std::size_t>((id * 0x9E3779B97F4A7C15ull) >> 32) & mask_;
    }

    std::vector<uint64_t> keys_;
    std::vector<uint32_t> values_;
    std::size_t mask_ = 0;
};

} // namespace RayTrophiSim::Fluid
