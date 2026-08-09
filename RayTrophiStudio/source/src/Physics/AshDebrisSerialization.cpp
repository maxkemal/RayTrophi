#include "AshDebrisSerialization.h"
#include "AshDebrisSystem.h"

#include <algorithm>

namespace RayTrophiSim {
nlohmann::json serializeAshDebris(const AshDebrisSystem& system) {
    const auto& s = system.settings();
    return {{"enabled", s.enabled}, {"max_particles", s.max_particles},
            {"particles_per_kg", s.particles_per_kg},
            {"near_distance", s.near_distance},
            {"far_lod_scale", s.far_lod_scale},
            {"lifetime_seconds", s.lifetime_seconds}};
}

void deserializeAshDebris(const nlohmann::json& root, AshDebrisSystem& system) {
    if (!root.contains("ash_debris") || !root["ash_debris"].is_object()) return;
    const auto& j = root["ash_debris"];
    auto& s = system.settings();
    s.enabled = j.value("enabled", s.enabled);
    s.max_particles = j.value("max_particles", s.max_particles);
    s.particles_per_kg = std::max(j.value("particles_per_kg", s.particles_per_kg), 0.0f);
    s.near_distance = std::max(j.value("near_distance", s.near_distance), 0.0f);
    s.far_lod_scale = std::clamp(j.value("far_lod_scale", s.far_lod_scale), 0.0f, 1.0f);
    s.lifetime_seconds = std::max(j.value("lifetime_seconds", s.lifetime_seconds), 0.05f);
}
} // namespace RayTrophiSim
