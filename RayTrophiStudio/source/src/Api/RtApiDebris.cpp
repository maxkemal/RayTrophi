#include "RtApiInternal.h"

#include <algorithm>

namespace rtapi {

Result configureAshDebris(bool enabled, uint64_t max_particles,
                          float particles_per_kg, float near_distance,
                          float far_lod_scale, float lifetime_seconds) {
    if (!g_ctx) return notBound();
    auto& settings = g_ctx->scene.ashDebrisSystem().settings();
    settings.enabled = enabled;
    settings.max_particles = static_cast<std::size_t>(max_particles);
    settings.particles_per_kg = std::max(particles_per_kg, 0.0f);
    settings.near_distance = std::max(near_distance, 0.0f);
    settings.far_lod_scale = std::clamp(far_lod_scale, 0.0f, 1.0f);
    settings.lifetime_seconds = std::max(lifetime_seconds, 0.05f);
    return Result::success();
}

Result emitAshDebris(Vec3 center, Vec3 velocity, float mass_kg,
                     float camera_distance, uint32_t seed,
                     uint64_t& out_spawned) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (mass_kg <= 0.0f) return Result::fail("ash mass_kg must be positive");
    auto& particles = g_ctx->scene.ensureParticleSimulationSystem();
    out_spawned = static_cast<uint64_t>(g_ctx->scene.ashDebrisSystem().emit(
        particles, center, velocity, mass_kg,
        std::max(camera_distance, 0.0f), seed));
    return Result::success();
}

Result getAshDebrisInfo(AshDebrisInfo& out) {
    if (!g_ctx) return notBound();
    const auto& system = g_ctx->scene.ashDebrisSystem();
    const auto& settings = system.settings();
    const auto& stats = system.stats();
    out.enabled = settings.enabled;
    out.max_particles = settings.max_particles;
    out.particles_per_kg = settings.particles_per_kg;
    out.near_distance = settings.near_distance;
    out.far_lod_scale = settings.far_lod_scale;
    out.lifetime_seconds = settings.lifetime_seconds;
    const auto runtime = g_ctx->scene.getParticleSimulationSystem();
    out.alive_particles = runtime ? runtime->aliveCount() : 0;
    out.events = stats.events;
    out.requested_particles = stats.requested_particles;
    out.spawned_particles = stats.spawned_particles;
    out.lod_reduced_particles = stats.lod_reduced_particles;
    out.budget_rejected_particles = stats.budget_rejected_particles;
    out.accepted_mass_kg = stats.accepted_mass_kg;
    out.reservoir_mass_kg = system.reservoirMassKg();
    return Result::success();
}
} // namespace rtapi
