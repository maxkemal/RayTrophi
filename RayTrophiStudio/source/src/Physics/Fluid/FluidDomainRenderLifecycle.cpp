#include "scene_data.h"

#include "globals.h"

#include <type_traits>

bool SceneData::hasAuthoritativeGridFluidDomain(const std::string& name) const {
    for (const auto& system : particle_systems) {
        if (!system.runtime) continue;
        for (const auto& domain : system.runtime->gridDomains()) {
            if (domain.type == RayTrophiSim::SimulationDomainType::Fluid &&
                domain.name == name) {
                return true;
            }
        }
    }
    return false;
}

void SceneData::retireDomainSurfaceRepresentation(
    SceneData::ParticleSystemObject& system,
    std::size_t domain_index) {
    const bool had_live_id =
        domain_index < system.domain_vdb_ids.size() &&
        system.domain_vdb_ids[domain_index] >= 0;
    const bool had_volume =
        domain_index < system.domain_volumes.size() &&
        static_cast<bool>(system.domain_volumes[domain_index]);

    // A representation switch is not an empty/cache-miss frame. Remove the
    // SurfaceSDF hittable and its live NanoVDB registration so a later Vulkan
    // TLAS/SSBO rebuild cannot resurrect stale dielectric geometry.
    removeDomainVolume(system, domain_index);

    auto release = [domain_index](auto& buffers) {
        if (domain_index >= buffers.size()) return;
        using Buffer = typename std::decay_t<decltype(buffers)>::value_type;
        Buffer{}.swap(buffers[domain_index]);
    };
    release(system.domain_sdf_buffers);
    release(system.domain_uvw_buffers);
    release(system.domain_composition_buffers);
    release(system.domain_foam_density);

    if (domain_index < system.domain_sdf_stats.size()) {
        system.domain_sdf_stats[domain_index] =
            RayTrophiSim::Fluid::LevelSetStats{};
    }
    if (domain_index < system.domain_last_fluid_render_mode.size()) {
        system.domain_last_fluid_render_mode[domain_index] = static_cast<int>(
            RayTrophiSim::Fluid::FluidRenderMode::Particles);
    }

    if (had_live_id || had_volume) {
        g_gas_volumes_dirty = true;
    }
}
