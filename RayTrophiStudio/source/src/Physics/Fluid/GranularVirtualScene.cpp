#include "scene_data.h"

#include "Fluid/GranularVirtualSurface.h"
#include "TriangleMesh.h"
#include "globals.h"

#include <algorithm>

namespace {

void markGranularTopologyChanged() {
    g_geometry_dirty = true;
    g_vulkan_rebuild_pending = true;
    g_optix_rebuild_pending = true;
    g_viewport_raster_rebuild_pending = true;
    g_bvh_rebuild_pending = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
}

void removeGranularMesh(
    SceneData& scene,
    std::shared_ptr<TriangleMesh>& mesh) {
    if (!mesh) {
        return;
    }
    const auto hittable = std::static_pointer_cast<Hittable>(mesh);
    const auto found = std::find(
        scene.world.objects.begin(),
        scene.world.objects.end(),
        hittable);
    if (found != scene.world.objects.end()) {
        scene.world.objects.erase(found);
    }
    mesh.reset();
    markGranularTopologyChanged();
}

} // namespace

void SceneData::syncGranularVirtualRepresentations() {
    for (auto& system : particle_systems) {
        if (!system.runtime) {
            continue;
        }
        const auto& states = system.runtime->gridDomainStates();
        const auto& domains = system.runtime->gridDomains();
        const std::size_t count = std::max(states.size(), domains.size());
        system.domain_granular_virtual.resize(count);
        system.domain_granular_meshes.resize(count);
        system.domain_granular_versions.resize(count, 0);
        system.domain_granular_particle_counts.resize(count, 0);
        system.domain_granular_material_ids.resize(count, -1);

        for (std::size_t domain_index = 0; domain_index < count; ++domain_index) {
            const bool available =
                domain_index < states.size() &&
                domain_index < domains.size();
            const bool virtual_granular = available &&
                domains[domain_index].type ==
                    RayTrophiSim::SimulationDomainType::Fluid &&
                domains[domain_index].fluid_render_mode ==
                    RayTrophiSim::Fluid::FluidRenderMode::VirtualParticles &&
                domains[domain_index].fluid_params.granular_enabled;
            if (!virtual_granular) {
                removeGranularMesh(
                    *this,
                    system.domain_granular_meshes[domain_index]);
                system.domain_granular_virtual[domain_index].clear();
                system.domain_granular_versions[domain_index] = 0;
                system.domain_granular_particle_counts[domain_index] = 0;
                system.domain_granular_material_ids[domain_index] = -1;
                continue;
            }

            const auto& state = states[domain_index];
            const auto& domain = domains[domain_index];
            const std::size_t particle_count = state.particles.size();
            const int authored_material = domain.fluid_surface_material_id >= 0
                ? domain.fluid_surface_material_id
                : domain.fluid_particle_material_id;
            if (system.domain_granular_versions[domain_index] == state.version &&
                system.domain_granular_particle_counts[domain_index] ==
                    particle_count &&
                system.domain_granular_material_ids[domain_index] ==
                    authored_material &&
                system.domain_granular_virtual[domain_index].stats.measured) {
                continue;
            }

            std::string error;
            RayTrophiSim::Fluid::GranularVirtualSettings settings;
            if (!RayTrophiSim::Fluid::buildGranularVirtualRepresentation(
                    state.particles,
                    state.grid,
                    settings,
                    system.domain_granular_virtual[domain_index],
                    error)) {
                removeGranularMesh(
                    *this,
                    system.domain_granular_meshes[domain_index]);
                system.domain_granular_versions[domain_index] = state.version;
                system.domain_granular_particle_counts[domain_index] = particle_count;
                continue;
            }

            auto& representation =
                system.domain_granular_virtual[domain_index];
            if (representation.stats.surface_columns == 0) {
                removeGranularMesh(
                    *this,
                    system.domain_granular_meshes[domain_index]);
            } else {
                const std::uint16_t material_id = static_cast<std::uint16_t>(
                    std::clamp(authored_material, 0, 65535));
                const std::string node_name =
                    "[GranularVirtual] " + system.name + " D" +
                    std::to_string(domain_index);
                const auto previous_mesh =
                    system.domain_granular_meshes[domain_index];
                bool topology_changed = false;
                if (RayTrophiSim::Fluid::updateGranularVirtualSurface(
                        representation,
                        material_id,
                        node_name,
                        system.domain_granular_meshes[domain_index],
                        topology_changed,
                        error)) {
                    auto& mesh = system.domain_granular_meshes[domain_index];
                    if (!previous_mesh || previous_mesh != mesh) {
                        if (previous_mesh) {
                            const auto old_hittable =
                                std::static_pointer_cast<Hittable>(previous_mesh);
                            const auto found = std::find(
                                world.objects.begin(),
                                world.objects.end(),
                                old_hittable);
                            if (found != world.objects.end()) {
                                world.objects.erase(found);
                            }
                        }
                        world.objects.push_back(
                            std::static_pointer_cast<Hittable>(mesh));
                        markGranularTopologyChanged();
                    } else if (topology_changed) {
                        markGranularTopologyChanged();
                    } else {
                        markBodyGeometryDirty(node_name);
                        g_viewport_raster_rebuild_pending = true;
                    }
                } else {
                    removeGranularMesh(
                        *this,
                        system.domain_granular_meshes[domain_index]);
                    representation.clear();
                }
            }
            system.domain_granular_versions[domain_index] = state.version;
            system.domain_granular_particle_counts[domain_index] = particle_count;
            system.domain_granular_material_ids[domain_index] = authored_material;
        }
    }
}

void SceneData::releaseGranularVirtualRepresentations() {
    for (auto& system : particle_systems) {
        for (auto& mesh : system.domain_granular_meshes) {
            removeGranularMesh(*this, mesh);
        }
        system.domain_granular_meshes.clear();
        system.domain_granular_virtual.clear();
        system.domain_granular_versions.clear();
        system.domain_granular_particle_counts.clear();
        system.domain_granular_material_ids.clear();
    }
}

void SceneData::removeGranularVirtualRepresentation(
    ParticleSystemObject& system,
    std::size_t domain_index) {
    if (domain_index < system.domain_granular_meshes.size()) {
        removeGranularMesh(*this, system.domain_granular_meshes[domain_index]);
    }
    if (domain_index < system.domain_granular_virtual.size()) {
        system.domain_granular_virtual[domain_index].clear();
    }
    if (domain_index < system.domain_granular_versions.size()) {
        system.domain_granular_versions[domain_index] = 0;
    }
    if (domain_index < system.domain_granular_particle_counts.size()) {
        system.domain_granular_particle_counts[domain_index] = 0;
    }
    if (domain_index < system.domain_granular_material_ids.size()) {
        system.domain_granular_material_ids[domain_index] = -1;
    }
}
