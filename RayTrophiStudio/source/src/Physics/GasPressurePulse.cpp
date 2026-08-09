#include "ParticleSimulation.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim {

bool ParticleSimulationSystem::injectGasPressurePulse(
    const std::string& domain, const Vec3& center,
    float radius, float peak_pressure_kpa) {
    if (radius <= 0.0f || peak_pressure_kpa <= 0.0f) return false;
    for (std::size_t i = 0; i < grid_domains_.size() &&
                            i < grid_domain_states_.size(); ++i) {
        if (grid_domains_[i].name != domain ||
            grid_domains_[i].type != SimulationDomainType::Gas) continue;
        auto& state = grid_domain_states_[i];
        if (!state.valid) return false;
        auto& grid = state.grid;
        const float h = std::max(state.voxel_size, 1e-4f);
        const float velocity_peak = std::min(peak_pressure_kpa * 0.02f, 80.0f);
        bool touched = false;
        for (int z = 0; z < grid.nz; ++z) {
            for (int y = 0; y < grid.ny; ++y) {
                for (int x = 0; x < grid.nx; ++x) {
                    const Vec3 p = state.bounds_min +
                        Vec3((x + 0.5f) * h, (y + 0.5f) * h, (z + 0.5f) * h);
                    Vec3 radial = p - center;
                    const float distance = radial.length();
                    if (distance >= radius || distance < 1e-5f) continue;
                    radial = radial * (1.0f / distance);
                    const float falloff = 1.0f - distance / radius;
                    const Vec3 dv = radial * (velocity_peak * falloff * 0.5f);
                    grid.velXAt(x, y, z) += dv.x;
                    grid.velXAt(x + 1, y, z) += dv.x;
                    grid.velYAt(x, y, z) += dv.y;
                    grid.velYAt(x, y + 1, z) += dv.y;
                    grid.velZAt(x, y, z) += dv.z;
                    grid.velZAt(x, y, z + 1) += dv.z;
                    if (grid.pressure.size() ==
                        static_cast<std::size_t>(grid.getCellCount())) {
                        grid.pressure[grid.cellIndex(x, y, z)] +=
                            peak_pressure_kpa * falloff;
                    }
                    touched = true;
                }
            }
        }
        if (touched) ++state.version;
        return touched;
    }
    return false;
}

} // namespace RayTrophiSim
