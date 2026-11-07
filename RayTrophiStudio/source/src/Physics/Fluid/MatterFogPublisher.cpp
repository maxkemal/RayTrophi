#include "scene_data.h"
#include "Fluid/MatterFogSnapshot.h"
#include "Fluid/MatterFogStreams.h"
#include "Fluid/FogOpticalBounds.h"
#include "VolumeShader.h"
#include "globals.h"
#include "imgui.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <iterator>
#include <vector>

namespace {
void dirtyFogGeometry() {
    g_geometry_dirty = true;
    g_vulkan_rebuild_pending = true;
    g_optix_rebuild_pending = true;
    g_bvh_rebuild_pending = true;
    g_gas_volumes_dirty = true;
}

// Gas split into per-source streams renders through the mixture path at about
// twice the cost of the same gas as one volume (RT_GAS_STREAM_MERGE_20261030.md).
// Streams that differ only in density response can be one volume: the response
// is applied here, per stream, and the mapped densities add. Raw densities must
// not be summed, a saturating curve gives f(a) + f(b) != f(a + b).
using RayTrophiSim::Fluid::FogFieldKind;
using RayTrophiSim::Fluid::FogFieldKey;
using RayTrophiSim::Fluid::MatterFogAppearance;
using RayTrophiSim::Fluid::MatterFogAppearances;
using RayTrophiSim::Fluid::MatterFogFields;

nlohmann::json fogMergeIdentity(const VolumeShader& shader) {
    nlohmann::json identity = shader.toJson();
    identity.erase("name");
    identity.erase("density");
    identity["material_graph"] = shader.material_graph;
    identity["density_channel"] = shader.density.channel;
    identity["edge_falloff"] = shader.density.edge_falloff;
    if (!shader.emission_response.mask_enabled ||
        shader.emission_response.mask_strength <= 0.0f ||
        shader.emission.mode == VolumeEmissionMode::None) {
        auto& response = identity["emission_response"];
        response.erase("mask_enabled");
        response.erase("mask_center");
        response.erase("mask_width");
        response.erase("mask_strength");
    }
    return identity;
}

// Empty when the gas streams cannot merge. Otherwise it changes with every input
// of the merged density, so an edit republishes.
std::string gasMergeKey(const MatterFogAppearances& appearances,
    std::string* reason = nullptr) {
    const auto refuse = [&](const char* why) {
        if (reason) {
            *reason = why;
        }
        return std::string{};
    };
    const VolumeShader* first = nullptr;
    nlohmann::json firstIdentity;
    std::string key;
    std::size_t count = 0;
    for (const auto& [field, appearance] : appearances) {
        if (field.kind == FogFieldKind::Carrier) {
            continue;
        }
        const auto* shader = appearance.shader.get();
        if (!shader) {
            return refuse("a source has no material");
        }
        if (!appearance.visible) {
            return refuse("a source is hidden");
        }
        if (!shader->material_graph.empty()) {
            return refuse("a source uses a material graph");
        }
        if (shader->emission.mode != VolumeEmissionMode::None &&
            shader->emission_response.independent_emission) {
            return refuse("independent emission coefficients must retain each source density");
        }
        if (shader->emission.mode != VolumeEmissionMode::None &&
            shader->emission_response.mask_enabled &&
            shader->emission_response.mask_strength > 0.0f) {
            return refuse("density emission masks must be evaluated per source");
        }
        if (!first) {
            first = shader;
            firstIdentity = fogMergeIdentity(*shader);
            key += firstIdentity.dump();
        } else if (fogMergeIdentity(*shader) != firstIdentity) {
            return refuse("their color, scattering, absorption or emission differ");
        }
        key += shader->density.toJson().dump();
        ++count;
    }
    return count >= 2 ? key : refuse("only one gas source");
}

// volumeNanoFinishDensity, on the host. Keep the two in the same order.
float mappedFogDensity(const VolumeShader& shader, float raw) {
    if (!std::isfinite(raw)) {
        return 0.0f;
    }
    const auto& density = shader.density;
    const float range = std::max(density.remap_high - density.remap_low, 1e-6f);
    const float x = std::max((raw - density.remap_low) / range, 0.0f);
    float mapped = x;
    if (density.curve.enabled && x > 0.0f) {
        if (x >= 1.0f) {
            mapped = x * density.curve.values[7];
        } else {
            const float t = x * 7.0f;
            const int i = std::min(static_cast<int>(t), 6);
            const float f = t - static_cast<float>(i);
            mapped = density.curve.values[i] +
                (density.curve.values[i + 1] - density.curve.values[i]) * f;
        }
    }
    const float cutoff = std::max(density.cutoff_threshold, 0.0f);
    if (mapped <= cutoff) {
        return 0.0f;
    }
    float fade = 1.0f;
    if (cutoff > 0.0f) {
        const float u = std::clamp((mapped - cutoff) / cutoff, 0.0f, 1.0f);
        fade = u * u * (3.0f - 2.0f * u);
    }
    return mapped * density.multiplier * fade;
}

// One thin wisp above the cutoff keeps its whole leaf from being skipped and can
// stretch the volume box across the domain: measured 210 -> 80 ms with no visible
// change. A block is dropped only when no axis column through it reaches this
// optical depth, which bounds what it can do to a ray. Emissive media are kept:
// their light does not scale with extinction.
constexpr float kInvisibleBlockOpticalDepth = 0.01f;
constexpr int kFogCullBlock = 8;

void cullInvisibleFogBlocks(std::vector<float>& density,
    const RayTrophiSim::Fluid::FogFieldGrid& grid, const VolumeShader& shader) {
    if (shader.emission.mode != VolumeEmissionMode::None) {
        return;
    }
    const float sigma = std::max(shader.scattering.coefficient, 0.0f) +
        std::max(shader.absorption.coefficient, 0.0f);
    const std::size_t cells = static_cast<std::size_t>(grid.nx) * grid.ny * grid.nz;
    if (sigma <= 0.0f || density.size() != cells) {
        return;
    }
    const int b = kFogCullBlock;
    const float perCell = sigma * grid.voxel;
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;
    const std::size_t stepY = static_cast<std::size_t>(nx);
    const std::size_t stepZ = stepY * ny;
    const int blocksX = (nx + b - 1) / b;
    const int blocksY = (ny + b - 1) / b;
    const int blocksZ = (nz + b - 1) / b;
    float* values = density.data();
    #pragma omp parallel for collapse(3) schedule(static)
    for (int bz = 0; bz < blocksZ; ++bz) {
        for (int by = 0; by < blocksY; ++by) {
            for (int bx = 0; bx < blocksX; ++bx) {
                float columnX[kFogCullBlock * kFogCullBlock] = {};
                float columnY[kFogCullBlock * kFogCullBlock] = {};
                float columnZ[kFogCullBlock * kFogCullBlock] = {};
                float deepest = 0.0f;
                const int endX = std::min((bx + 1) * b, nx);
                const int endY = std::min((by + 1) * b, ny);
                const int endZ = std::min((bz + 1) * b, nz);
                for (int z = bz * b; z < endZ; ++z) {
                    for (int y = by * b; y < endY; ++y) {
                        for (int x = bx * b; x < endX; ++x) {
                            const std::size_t cell = x + y * stepY + z * stepZ;
                            const float tau = mappedFogDensity(shader, values[cell]) * perCell;
                            float& alongX = columnX[(y % b) * b + (z % b)];
                            float& alongY = columnY[(x % b) * b + (z % b)];
                            float& alongZ = columnZ[(x % b) * b + (y % b)];
                            alongX += tau;
                            alongY += tau;
                            alongZ += tau;
                            deepest = std::max({deepest, alongX, alongY, alongZ});
                        }
                    }
                }
                if (deepest >= kInvisibleBlockOpticalDepth) {
                    continue;
                }
                for (int z = bz * b; z < endZ; ++z) {
                    for (int y = by * b; y < endY; ++y) {
                        for (int x = bx * b; x < endX; ++x) {
                            values[x + y * stepY + z * stepZ] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

// Folds every gas field into the first. Returns the neutral shader of the merged
// stream, or null when the fields stay separate.
std::shared_ptr<VolumeShader> mergeGasFields(MatterFogFields& fields,
    const MatterFogAppearances& appearances, FogFieldKey& mergedKey) {
    std::vector<std::size_t> gas;
    for (std::size_t i = 0; i < fields.fields.size(); ++i) {
        if (fields.fields[i].key.kind != FogFieldKind::Carrier) {
            gas.push_back(i);
        }
    }
    if (gas.size() < 2) {
        return nullptr;
    }
    const auto& first = fields.fields[gas.front()];
    std::vector<float> merged(first.density.size(), 0.0f);
    for (std::size_t index : gas) {
        const auto& field = fields.fields[index];
        const auto appearance = appearances.find(field.key);
        if (appearance == appearances.end() || !appearance->second.shader ||
            field.density.size() != merged.size()) {
            return nullptr;
        }
        const auto& shader = *appearance->second.shader;
        for (std::size_t cell = 0; cell < merged.size(); ++cell) {
            merged[cell] += mappedFogDensity(shader, field.density[cell]);
        }
    }
    auto neutral = std::make_shared<VolumeShader>(*appearances.at(first.key).shader);
    neutral->density.curve.enabled = false;
    neutral->density.multiplier = 1.0f;
    neutral->density.remap_low = 0.0f;
    neutral->density.remap_high = 1.0f;
    neutral->density.cutoff_threshold = 0.0f;
    mergedKey = first.key;
    fields.fields[gas.front()].density = std::move(merged);
    for (auto index = gas.rbegin(); index != std::prev(gas.rend()); ++index) {
        fields.fields.erase(fields.fields.begin() + static_cast<std::ptrdiff_t>(*index));
    }
    return neutral;
}
}

namespace RayTrophiSim::Fluid {
std::string matterGasMergeStatus(const MatterFogAppearances& appearances) {
    std::string reason;
    if (!gasMergeKey(appearances, &reason).empty()) {
        return "Merged: sources share one volume, materials differ only in density.";
    }
    if (reason == "only one gas source") {
        return "One gas source.";
    }
    if (reason == "density emission masks must be evaluated per source") {
        return "Separate volumes: density emission masks need each source's own density. "
            "Disabling masks allows compatible sources to merge again.";
    }
    return "Separate volumes, " + reason +
        ". Overlapping sources cost about twice as much. To merge, make every setting "
        "except density equal on all sources.";
}
}

// Requests once per key: a refused publication must not retry every frame.
bool SceneData::matterFogMergeChanged() {
    // A publish resets the render and would cut a slider drag or curve edit short.
    // The key stays different, so the request lands on the frame after release.
    if (ImGui::GetCurrentContext() &&
        (ImGui::IsAnyItemActive() || ImGui::IsMouseDown(ImGuiMouseButton_Left))) {
        return false;
    }
    bool changed = false;
    for (auto& system : particle_systems) {
        if (!system.runtime) {
            continue;
        }
        auto& domains = system.runtime->gridDomains();
        for (std::size_t d = 0; d < system.matter_fog_streams.size() &&
                d < domains.size(); ++d) {
            auto& streams = system.matter_fog_streams[d];
            if (streams.empty()) {
                continue;
            }
            const std::string key = gasMergeKey(domains[d].matter_fog_streams);
            for (auto& [field, stream] : streams) {
                const auto appearance = domains[d].matter_fog_streams.find(field);
                if (appearance == domains[d].matter_fog_streams.end()) {
                    continue;
                }
                // The volume box is cut from the density at publish time, so a
                // cutoff edit needs a publish as much as a merge change does.
                const auto& shader = stream.merged_shader
                    ? stream.merged_shader : appearance->second.shader;
                const std::string bounds = RayTrophiSim::Fluid::fogOpticalBoundsKey(shader.get(),
                    stream.volume ? stream.volume->density_scale : 1.0f);
                const std::string wanted = key + "|" + bounds;
                if (stream.gas_merge_key + "|" + stream.optical_bounds_key != wanted &&
                    stream.gas_merge_requested != wanted) {
                    stream.gas_merge_requested = wanted;
                    changed = true;
                }
            }
        }
    }
    return changed;
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
    const auto streamShader = [](const MatterFogLiveStream& stream,
        const MatterFogAppearance& appearance) {
        return stream.merged_shader ? stream.merged_shader : appearance.shader;
    };
    const std::string gasMerge = gasMergeKey(desc.matter_fog_streams);
    const bool current = !force_sync && !streams.empty() &&
        std::all_of(streams.begin(), streams.end(), [&](const auto& entry) {
            const auto& stream = entry.second;
            const auto appearance = desc.matter_fog_streams.find(entry.first);
            if (appearance == desc.matter_fog_streams.end()) {
                return false;
            }
            const float scale = stream.volume ? stream.volume->density_scale : 1.0f;
            return stream.state_version == state.version &&
                stream.gas_merge_key == gasMerge &&
                stream.optical_bounds_key == fogOpticalBoundsKey(
                    streamShader(stream, appearance->second).get(), scale) &&
                (!stream.populated || (stream.vdb_id >= 0 && stream.volume));
        });
    if (current) {
        for (auto& [key, stream] : streams) {
            auto& appearance = ensureFogAppearance(desc.matter_fog_streams,
                key, desc.fluid_fog_shader);
            if (stream.volume) {
                stream.volume->setPublishedShader(streamShader(stream, appearance),
                    appearance.shader);
            }
            setVisible(stream,
                stream.populated && stream.optically_populated && appearance.visible);
        }
        return;
    }
    MatterFogFields fields;
    std::string snapshot;
    std::string error;
    try {
        if (!prepareMatterFogSnapshot(*system.runtime, d, simulation_world.compute(),
                fields, snapshot, error,
                RayTrophiSim::Fluid::MatterFogSnapshotPurpose::Publication)) {
            SCENE_LOG_WARN("Matter Fog publication refused: " + error);
            return;
        }
    } catch (const std::exception& exception) {
        SCENE_LOG_WARN(std::string("Matter Fog snapshot failed: ") + exception.what());
        return;
    }
    // After the merge, so the folded streams retire below like any other
    // representation change. Their appearances stay authored in the domain.
    for (const auto& field : fields.fields) {
        if (field.key.kind != FogFieldKind::Carrier) {
            ensureFogAppearance(desc.matter_fog_streams, field.key, desc.fluid_fog_shader);
        }
    }
    for (auto& field : fields.fields) {
        const auto appearance = desc.matter_fog_streams.find(field.key);
        if (field.key.kind != FogFieldKind::Carrier &&
            appearance != desc.matter_fog_streams.end() && appearance->second.shader) {
            cullInvisibleFogBlocks(field.density, field.grid, *appearance->second.shader);
        }
    }
    const std::string gasMergeNow = gasMergeKey(desc.matter_fog_streams);
    FogFieldKey mergedKey;
    const auto mergedShader = gasMergeNow.empty() ? nullptr
        : mergeGasFields(fields, desc.matter_fog_streams, mergedKey);
    // A representation change retires its live volume, rather than keeping a
    // hidden VDB panel entry forever. Preserve authored appearances separately
    // so switching back to Fog restores the user's material.
    for (auto it = streams.begin(); it != streams.end();) {
        const bool retained = std::any_of(fields.fields.begin(), fields.fields.end(),
            [&](const auto& field) { return field.key == it->first; });
        if (retained) {
            ++it;
            continue;
        }
        auto& stream = it->second;
        if (stream.vdb_id >= 0) {
            manager.unloadVDB(stream.vdb_id);
        }
        if (stream.volume) {
            removeVDBVolume(stream.volume);
            world.objects.erase(std::remove(world.objects.begin(), world.objects.end(),
                std::static_pointer_cast<Hittable>(stream.volume)), world.objects.end());
            dirtyFogGeometry();
            simulation_render_updated = true;
        }
        it = streams.erase(it);
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
        appearance.producer_name = field.producer_name;
        auto& stream = streams[field.key];
        stream.merged_shader = mergedShader && field.key == mergedKey ? mergedShader : nullptr;
        stream.gas_merge_key = gasMergeNow;
        const auto shader = streamShader(stream, appearance);
        const bool populated = std::any_of(field.density.begin(), field.density.end(),
            [](float density) { return density > 0.0f; });
        stream.populated = populated;
        const float densityScale = stream.volume ? stream.volume->density_scale : 1.0f;
        const auto boundsKey = fogOpticalBoundsKey(shader.get(), densityScale);
        if (!populated) {
            stream.optical_bounds_key = boundsKey;
            stream.state_version = state.version;
            setVisible(stream, false);
            continue;
        }
        const auto& grid = field.grid;
        const std::string name = desc.name + " / " +
            (field.producer_name.empty() ? fogFieldName(field.key) : field.producer_name);
        const int id = manager.registerOrUpdateLiveVolume(stream.vdb_id, name,
            grid.nx, grid.ny, grid.nz, grid.voxel, field.density.data(),
            field.temperature_kelvin.empty() ? nullptr : field.temperature_kelvin.data(),
            nullptr);
        if (id < 0) {
            continue;
        }
        stream.vdb_id = id;
        stream.optical_bounds_key = boundsKey;
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
        volume->setPublishedShader(shader, appearance.shader);
        // Before bindLiveVolume, which resets the box to the full grid.
        const AABB boxBefore = volume->getWorldBounds();
        volume->bindLiveVolume(id, grid.voxel, grid.origin, grid.origin + Vec3(
            static_cast<float>(grid.nx), static_cast<float>(grid.ny),
            static_cast<float>(grid.nz)) * grid.voxel);
        const auto bounds = fogOpticalBounds(field.density, grid.nx, grid.ny, grid.nz,
            shader.get(), volume->density_scale);
        stream.optically_populated = !bounds.empty;
        if (!bounds.empty) {
            volume->setLocalBounds(Vec3(static_cast<float>(bounds.first[0]),
                static_cast<float>(bounds.first[1]), static_cast<float>(bounds.first[2])) *
                    grid.voxel,
                Vec3(static_cast<float>(bounds.end[0]), static_cast<float>(bounds.end[1]),
                    static_cast<float>(bounds.end[2])) * grid.voxel);
        }
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
        setVisible(stream, stream.optically_populated && appearance.visible);
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
