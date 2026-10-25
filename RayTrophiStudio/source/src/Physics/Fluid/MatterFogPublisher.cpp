#include "scene_data.h"
#include "Fluid/MatterFogSnapshot.h"
#include "Fluid/MatterFogStreams.h"
#include "globals.h"
#include <algorithm>
#include <exception>

namespace {
void dirtyFogGeometry() {
    g_geometry_dirty = true;
    g_vulkan_rebuild_pending = true;
    g_optix_rebuild_pending = true;
    g_bvh_rebuild_pending = true;
    g_gas_volumes_dirty = true;
}
}

RayTrophiSim::SimulationGridDomainDesc* SceneData::matterFogDomainOwningVolume(
    const VDBVolume* volume, bool* fog_view) {
    if (!volume) {
        return nullptr;
    }
    for (auto& system : particle_systems) {
        if (!system.runtime) {
            continue;
        }
        auto& domains = system.runtime->gridDomains();
        for (std::size_t d = 0; d < system.matter_fog_streams.size() &&
                d < domains.size(); ++d) {
            for (const auto& [key, stream] : system.matter_fog_streams[d]) {
                if (stream.volume.get() == volume) {
                    if (fog_view) {
                        *fog_view = true;
                    }
                    return &domains[d];
                }
            }
        }
    }
    return nullptr;
}

void SceneData::removeMatterFogStreams(ParticleSystemObject& system, std::size_t d) {
    if (d >= system.matter_fog_streams.size()) {
        return;
    }
    auto& manager = VDBVolumeManager::getInstance();
    for (auto& [key, stream] : system.matter_fog_streams[d]) {
        if (stream.vdb_id >= 0) {
            manager.unloadVDB(stream.vdb_id);
        }
        if (stream.volume) {
            removeVDBVolume(stream.volume);
            world.objects.erase(std::remove(world.objects.begin(), world.objects.end(),
                std::static_pointer_cast<Hittable>(stream.volume)), world.objects.end());
            dirtyFogGeometry();
        }
    }
    system.matter_fog_streams[d].clear();
}

void SceneData::invalidateMatterFogStreams(ParticleSystemObject& system) {
    auto& manager = VDBVolumeManager::getInstance();
    for (auto& streams : system.matter_fog_streams) {
        for (auto& [key, stream] : streams) {
            if (stream.vdb_id >= 0) {
                manager.unloadVDB(stream.vdb_id);
            }
            stream.vdb_id = -1;
            if (stream.volume) {
                stream.volume->setVDBVolumeID(-1);
                stream.volume->awaiting_live_rebind = true;
            }
        }
    }
    g_gas_volumes_dirty = true;
}

void SceneData::syncMatterFogStreams(ParticleSystemObject& system, std::size_t d,
    const RayTrophiSim::SimulationGridDomainState& state,
    RayTrophiSim::SimulationGridDomainDesc& desc, VDBVolumeManager& manager,
    bool render_enabled, bool force_sync) {
    using namespace RayTrophiSim::Fluid;
    if (system.matter_fog_streams.size() <= d) {
        system.matter_fog_streams.resize(d + 1);
    }
    auto& streams = system.matter_fog_streams[d];
    auto setVisible = [](MatterFogLiveStream& stream, bool visible) {
        if (stream.volume && stream.volume->visible != visible) {
            stream.volume->visible = visible;
            dirtyFogGeometry();
        }
    };
    if (!render_enabled || !system.visible || !state.valid) {
        for (auto& [key, stream] : streams) {
            setVisible(stream, false);
        }
        return;
    }
    const bool current = !force_sync && !streams.empty() &&
        std::all_of(streams.begin(), streams.end(), [&](const auto& entry) {
            const auto& stream = entry.second;
            return stream.state_version == state.version &&
                (!stream.populated || (stream.vdb_id >= 0 && stream.volume));
        });
    if (current) {
        for (auto& [key, stream] : streams) {
            auto& appearance = ensureFogAppearance(desc.matter_fog_streams,
                key, desc.fluid_fog_shader);
            if (stream.volume) {
                stream.volume->setShader(appearance.shader);
            }
            setVisible(stream, stream.populated && appearance.visible);
        }
        return;
    }
    MatterFogFields fields;
    std::string snapshot;
    std::string error;
    try {
        if (!prepareMatterFogSnapshot(*system.runtime, d, simulation_world.compute(),
                fields, snapshot, error)) {
            SCENE_LOG_WARN("Matter Fog publication refused: " + error);
            return;
        }
    } catch (const std::exception& exception) {
        SCENE_LOG_WARN(std::string("Matter Fog snapshot failed: ") + exception.what());
        return;
    }
    std::size_t new_streams = 0;
    std::size_t new_appearances = 0;
    for (const auto& field : fields.fields) {
        if (streams.find(field.key) == streams.end()) {
            ++new_streams;
        }
        if (desc.matter_fog_streams.find(field.key) == desc.matter_fog_streams.end()) {
            ++new_appearances;
        }
    }
    if (desc.matter_fog_streams.size() + new_appearances > 32) {
        SCENE_LOG_WARN("Matter Fog persistent appearance capacity exceeded");
        return;
    }
    // The existing RT candidate scan has sixteen slots. Refuse overflow before
    // registering anything rather than silently publishing an omitted medium.
    if (vdb_volumes.size() + gas_volumes.size() + new_streams > 16) {
        SCENE_LOG_WARN("Matter Fog publication exceeds the sixteen-volume render capacity");
        return;
    }
    for (auto& [key, stream] : streams) {
        stream.populated = false;
    }
    for (const auto& field : fields.fields) {
        auto& appearance = ensureFogAppearance(desc.matter_fog_streams,
            field.key, desc.fluid_fog_shader);
        auto& stream = streams[field.key];
        const bool populated = std::any_of(field.density.begin(), field.density.end(),
            [](float density) { return density > 0.0f; });
        stream.populated = populated;
        if (!populated) {
            stream.state_version = state.version;
            setVisible(stream, false);
            continue;
        }
        const auto& grid = field.grid;
        const std::string name = desc.name + " / " + fogFieldName(field.key);
        const int id = manager.registerOrUpdateLiveVolume(stream.vdb_id, name,
            grid.nx, grid.ny, grid.nz, grid.voxel, field.density.data(),
            field.temperature_kelvin.empty() ? nullptr : field.temperature_kelvin.data(),
            nullptr);
        if (id < 0) {
            continue;
        }
        stream.vdb_id = id;
        stream.state_version = state.version;
        manager.clearLiveDenseGpuFields(id);
        if (!stream.volume) {
            stream.volume = std::make_shared<VDBVolume>();
            stream.volume->transient = true;
            stream.volume->name = name;
            addVDBVolume(stream.volume);
            world.objects.push_back(stream.volume);
            dirtyFogGeometry();
        }
        auto& volume = stream.volume;
        volume->name = name;
        volume->render_as_isosurface = false;
        volume->render_joint_fog = true;
        volume->cpu_render_skip = false;
        volume->setShader(appearance.shader);
        // Before bindLiveVolume, which resets the box to the full grid.
        const AABB boxBefore = volume->getWorldBounds();
        volume->bindLiveVolume(id, grid.voxel, grid.origin, grid.origin + Vec3(
            static_cast<float>(grid.nx), static_cast<float>(grid.ny),
            static_cast<float>(grid.nz)) * grid.voxel);
        // The full grid box made every ray in the domain pay for a stream that
        // may occupy a few hundred cells (VDBVolume::shrinkLiveBoundsToOccupied).
        volume->shrinkLiveBoundsToOccupied(field.density, grid.nx, grid.ny, grid.nz, grid.voxel);
        // Vulkan refreshes its TLAS box from getWorldBounds on every volume
        // publish and OptiX reads the box per ray, but the CPU Embree BVH bakes
        // it at build time: a GROWING box would clip the plume there. A
        // shrinking one only leaves a looser box, which is harmless.
        const AABB boxAfter = volume->getWorldBounds();
        if (boxAfter.min.x < boxBefore.min.x || boxAfter.min.y < boxBefore.min.y ||
            boxAfter.min.z < boxBefore.min.z || boxAfter.max.x > boxBefore.max.x ||
            boxAfter.max.y > boxBefore.max.y || boxAfter.max.z > boxBefore.max.z) {
            g_bvh_rebuild_pending = true;
        }
        setVisible(stream, appearance.visible);
        simulation_render_updated = true;
        g_gas_volumes_dirty = true;
    }
    for (auto& [key, stream] : streams) {
        if (!stream.populated) {
            stream.state_version = state.version;
            setVisible(stream, false);
        }
    }
}
