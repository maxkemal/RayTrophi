#pragma once

#include "ParticleSimulation.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace RayTrophiSim {

// Short-lived cursor for consecutive spawns in one simulation system. A spawn
// fills the lowest dead slot, so all slots before it are known to be live.
// Any intervening system update invalidates that guarantee.
class ParticleSpawnSequence {
public:
    std::size_t spawn(ParticleSimulationSystem& system, const ParticleSpawnDesc& desc) {
        if (system_ != &system || version_ != system.data_version_) {
            system_ = &system;
            next_slot_ = 0;
        }

        const std::size_t slot = system.spawnFrom(desc, next_slot_);
        if (slot != std::numeric_limits<std::size_t>::max()) {
            next_slot_ = slot + 1;
        }
        version_ = system.data_version_;
        return slot;
    }

private:
    const ParticleSimulationSystem* system_ = nullptr;
    std::uint64_t version_ = 0;
    std::size_t next_slot_ = 0;
};

} // namespace RayTrophiSim
