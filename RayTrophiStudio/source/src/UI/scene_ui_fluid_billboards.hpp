#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

namespace FluidBillboardUI {
inline void pushVertex(std::vector<float>& out, const Vec3& p, float u, float v,
                       const Vec3& color, float alpha) {
    out.push_back(p.x); out.push_back(p.y); out.push_back(p.z);
    out.push_back(u); out.push_back(v);
    out.push_back(color.x); out.push_back(color.y); out.push_back(color.z);
    out.push_back(alpha);
}

inline void appendGridDomainParticles(const SceneData& scene,
                                      const Vec3& right, const Vec3& up,
                                      std::vector<float>& alpha_data,
                                      std::size_t& drawn,
                                      std::size_t max_billboards) {
    for (const auto& system : scene.particle_systems) {
        if (!system.visible || !system.enabled || !system.runtime) continue;
        const auto& domains = system.runtime->gridDomains();
        const auto& states = system.runtime->gridDomainStates();
        for (std::size_t d = 0; d < domains.size() && d < states.size(); ++d) {
            const auto& desc = domains[d];
            const auto& state = states[d];
            if (!desc.enabled || desc.type != RayTrophiSim::SimulationDomainType::Fluid ||
                desc.fluid_render_mode != RayTrophiSim::Fluid::FluidRenderMode::Particles ||
                !state.valid) continue;
            const float radius = std::max(1.0e-4f,
                std::max(state.voxel_size, 1.0e-4f) *
                desc.fluid_particle_radius_factor *
                desc.fluid_particle_size_multiplier);
            const Vec3 rh = right * radius;
            const Vec3 uh = up * radius;
            for (const Vec3& center : state.particles.position) {
                if (drawn >= max_billboards) return;
                if (!std::isfinite(center.x) || !std::isfinite(center.y) ||
                    !std::isfinite(center.z)) continue;
                const Vec3 c00 = center - rh - uh;
                const Vec3 c10 = center + rh - uh;
                const Vec3 c11 = center + rh + uh;
                const Vec3 c01 = center - rh + uh;
                constexpr float alpha = 0.92f;
                pushVertex(alpha_data, c00, -1.f, -1.f, desc.fluid_particle_color, alpha);
                pushVertex(alpha_data, c10,  1.f, -1.f, desc.fluid_particle_color, alpha);
                pushVertex(alpha_data, c11,  1.f,  1.f, desc.fluid_particle_color, alpha);
                pushVertex(alpha_data, c00, -1.f, -1.f, desc.fluid_particle_color, alpha);
                pushVertex(alpha_data, c11,  1.f,  1.f, desc.fluid_particle_color, alpha);
                pushVertex(alpha_data, c01, -1.f,  1.f, desc.fluid_particle_color, alpha);
                ++drawn;
            }
        }
    }
}
} // namespace FluidBillboardUI
