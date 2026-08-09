#pragma once

#include "Vec3.h"

#include <cstddef>
#include <cstdint>

namespace RayTrophiSim {
class ParticleSimulationSystem;

struct AshDebrisSettings {
    bool enabled = true;
    std::size_t max_particles = 4096;
    float particles_per_kg = 120.0f;
    float near_distance = 12.0f;
    float far_lod_scale = 0.25f;
    float lifetime_seconds = 5.0f;
};

struct AshDebrisStats {
    uint64_t events = 0;
    uint64_t requested_particles = 0;
    uint64_t spawned_particles = 0;
    uint64_t lod_reduced_particles = 0;
    uint64_t budget_rejected_particles = 0;
    float accepted_mass_kg = 0.0f;
};

class AshDebrisSystem {
public:
    AshDebrisSettings& settings() { return settings_; }
    const AshDebrisSettings& settings() const { return settings_; }
    const AshDebrisStats& stats() const { return stats_; }
    std::size_t emit(ParticleSimulationSystem& particles, const Vec3& center,
                     const Vec3& velocity, float mass_kg,
                     float camera_distance, uint32_t seed);
    void resetStats() { stats_ = {}; }

private:
    AshDebrisSettings settings_;
    AshDebrisStats stats_;
};
} // namespace RayTrophiSim
