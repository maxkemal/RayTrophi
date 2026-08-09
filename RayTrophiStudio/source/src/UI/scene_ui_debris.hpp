#pragma once

#include <algorithm>

namespace DebrisUI {
inline void drawSceneControls(SceneData& scene) {
    if (!ImGui::CollapsingHeader("Ash & Small Debris (Scene)")) return;
    auto& system = scene.ashDebrisSystem();
    auto& settings = system.settings();
    const auto& stats = system.stats();
    ImGui::Checkbox("Enable Ash/Debris##AshScene", &settings.enabled);
    int budget = static_cast<int>(std::min<std::size_t>(
        settings.max_particles, 1000000u));
    if (ImGui::DragInt("Particle Budget##AshScene", &budget, 32.0f, 0, 1000000))
        settings.max_particles = static_cast<std::size_t>(std::max(budget, 0));
    ImGui::DragFloat("Particles / kg##AshScene", &settings.particles_per_kg,
                     1.0f, 0.0f, 10000.0f, "%.0f");
    ImGui::DragFloat("Near Distance##AshScene", &settings.near_distance,
                     0.25f, 0.0f, 10000.0f, "%.1f m");
    ImGui::DragFloat("Far LOD Scale##AshScene", &settings.far_lod_scale,
                     0.01f, 0.0f, 1.0f, "%.2f");
    ImGui::DragFloat("Lifetime##AshScene", &settings.lifetime_seconds,
                     0.1f, 0.05f, 120.0f, "%.1f s");
    ImGui::TextDisabled("events %llu | spawned %llu / requested %llu",
                        static_cast<unsigned long long>(stats.events),
                        static_cast<unsigned long long>(stats.spawned_particles),
                        static_cast<unsigned long long>(stats.requested_particles));
    ImGui::TextDisabled("LOD reduced %llu | budget rejected %llu | mass %.3f kg",
                        static_cast<unsigned long long>(stats.lod_reduced_particles),
                        static_cast<unsigned long long>(stats.budget_rejected_particles),
                        stats.accepted_mass_kg);
    if (ImGui::Button("Reset Debris Telemetry##AshScene")) system.resetStats();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Clears counters only; live particles and authored settings remain.");
}
} // namespace DebrisUI
