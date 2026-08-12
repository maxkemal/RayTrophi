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
    // Mass held back because the particle budget was full, waiting for a later
    // event to carry it. See AshDebrisSystem::reservoir_mass_kg_.
    float reservoir_mass_kg = 0.0f;
};

class AshDebrisSystem {
public:
    AshDebrisSettings& settings() { return settings_; }
    const AshDebrisSettings& settings() const { return settings_; }
    const AshDebrisStats& stats() const { return stats_; }
    // Returns the number of particles actually spawned. Mass that could not be
    // represented is NOT lost — it accumulates in the reservoir and rides along
    // with the next event that has budget.
    std::size_t emit(ParticleSimulationSystem& particles, const Vec3& center,
                     const Vec3& velocity, float mass_kg,
                     float camera_distance, uint32_t seed);
    void resetStats() { stats_ = {}; }
    // Timeline reset: the reservoir is simulation state, not settings.
    void resetReservoir() { reservoir_mass_kg_ = 0.0f; }
    float reservoirMassKg() const { return reservoir_mass_kg_; }

private:
    AshDebrisSettings settings_;
    AshDebrisStats stats_;
    // ★ THE AshReservoir THE ROADMAP ASKS FOR, and the fix for a real leak.
    //
    // When the particle budget was full this system simply returned 0 and the
    // debris mass evaporated — no particle, no reservoir, no counter. That
    // silently violates "mass is spent exactly once": the MSF side had already
    // debited it. Budget is a BUDGET FOR VISUAL DETAIL; it must never be a mass
    // sink. Unrepresented mass waits here and is folded into the next emit.
    float reservoir_mass_kg_ = 0.0f;
};
} // namespace RayTrophiSim
