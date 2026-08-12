#include "AshDebrisSystem.h"
#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim {
namespace {
float unitHash(uint32_t value) {
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return static_cast<float>(value & 0x00ffffffu) / 16777215.0f;
}
}

std::size_t AshDebrisSystem::emit(
    ParticleSimulationSystem& particles, const Vec3& center,
    const Vec3& velocity, float mass_kg, float camera_distance, uint32_t seed) {
    ++stats_.events;
    if (!settings_.enabled || mass_kg <= 0.0f) return 0;

    // Mass held back by an earlier full-budget event rides along with this one.
    // Detail count still comes from THIS event's mass, so a backlog does not
    // suddenly inflate the particle count; it makes each particle heavier.
    const float carried = reservoir_mass_kg_;
    const float total_mass = mass_kg + carried;

    const std::size_t requested = static_cast<std::size_t>(std::ceil(
        mass_kg * std::max(settings_.particles_per_kg, 0.0f)));
    stats_.requested_particles += requested;
    const float lod = camera_distance > settings_.near_distance
        ? std::clamp(settings_.far_lod_scale, 0.0f, 1.0f) : 1.0f;
    const std::size_t after_lod = static_cast<std::size_t>(std::ceil(requested * lod));
    stats_.lod_reduced_particles += requested - after_lod;
    const std::size_t alive = particles.aliveCount();
    const std::size_t available = alive < settings_.max_particles
        ? settings_.max_particles - alive : 0;
    const std::size_t count = std::min(after_lod, available);
    stats_.budget_rejected_particles += after_lod - count;
    if (count == 0) {
        // No detail slots at all: keep the mass, do not invent particles and do
        // not drop it. accepted_mass_kg deliberately does NOT move — nothing was
        // handed to the particle system this call.
        reservoir_mass_kg_ = total_mass;
        stats_.reservoir_mass_kg = reservoir_mass_kg_;
        return 0;
    }
    reservoir_mass_kg_ = 0.0f;
    stats_.reservoir_mass_kg = 0.0f;

    const float mass_each = total_mass / static_cast<float>(count);
    // ★ An aggregate particle carries the mass of the many it stands in for, so
    // drawing it at the size of a single grain reads as a speck that falls like a
    // stone. Radius tracks the cube root of the mass ratio (a grain is a volume),
    // capped so one huge aggregate cannot fill the screen.
    const float nominal_mass = std::max(
        mass_kg / std::max<float>(1.0f, static_cast<float>(after_lod)), 1e-9f);
    const float size_scale = std::clamp(
        std::cbrt(mass_each / nominal_mass), 1.0f, 4.0f);
    for (std::size_t i = 0; i < count; ++i) {
        const uint32_t key = seed + static_cast<uint32_t>(i * 3u);
        const Vec3 jitter(unitHash(key) - 0.5f, unitHash(key + 1u),
                          unitHash(key + 2u) - 0.5f);
        ParticleSpawnDesc desc;
        desc.position = center + jitter * 0.18f;
        desc.velocity = velocity + jitter * 0.8f + Vec3(0.0f, 0.35f, 0.0f);
        desc.lifetime_seconds = std::max(settings_.lifetime_seconds, 0.05f);
        desc.mass = std::max(mass_each, 1e-6f);
        desc.start_size = 0.025f * size_scale;
        desc.end_size = 0.008f * size_scale;
        desc.start_opacity = 0.75f;
        desc.end_opacity = 0.0f;
        desc.start_color = Vec3(0.12f, 0.10f, 0.08f);
        desc.end_color = Vec3(0.28f, 0.27f, 0.25f);
        particles.spawn(desc);
    }
    stats_.spawned_particles += count;
    stats_.accepted_mass_kg += total_mass;
    return count;
}
} // namespace RayTrophiSim
