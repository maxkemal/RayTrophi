#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "scene_data.h"

namespace RayTrophiSim {

// Deduplicates identical ObjectMeshSDF cooks across multiple particle systems
// during project load. In scenes like pastel-paint-workshop, 7 particle systems
// share the same 3 toy ObjectMeshSDF colliders (21 cooks -> 3 cooks, ~7x faster).
class ParticleSdfLoadDeduplicator {
public:
    struct Ref {
        std::shared_ptr<ParticleSimulationSystem> runtime;
        std::size_t collider_index = 0;
    };

    struct Group {
        Ref primary;
        std::vector<Ref> duplicates;
    };

    void enqueue(SceneData& scene,
                 ParticleColliderDesc& collider,
                 const std::shared_ptr<ParticleSimulationSystem>& runtime,
                 std::size_t collider_index,
                 bool system_has_domains) {
        if (collider.source_mode != ParticleColliderSourceMode::ObjectMeshSDF ||
            collider.source_name.empty() || !runtime) {
            return;
        }
        // Auto resolution (mode 4) depends on the system's finest domain voxel.
        // Only deduplicate mode 4 when the system has no grid domains (voxel = 0).
        const bool can_dedup = (collider.sdf_resolution_mode != 4) || !system_has_domains;
        if (!can_dedup) {
            scene.rebuildSDFColliderAsync(collider, runtime);
            return;
        }
        const std::string key =
            collider.source_name + "#" + std::to_string(collider.sdf_resolution_mode);
        auto it = groups_.find(key);
        if (it == groups_.end()) {
            scene.rebuildSDFColliderAsync(collider, runtime);
            Group group;
            group.primary = Ref{runtime, collider_index};
            groups_.emplace(key, std::move(group));
        } else {
            it->second.duplicates.push_back(Ref{runtime, collider_index});
        }
    }

    void resolveDuplicates(SceneData& scene) {
        bool any_duplicates = false;
        for (const auto& kv : groups_) {
            if (!kv.second.duplicates.empty()) {
                any_duplicates = true;
                break;
            }
        }
        if (!any_duplicates) {
            return;
        }

        // Wait for the primary cooks and publish them before sharing.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (true) {
            scene.publishCompletedSdfBakes();
            bool still_busy = false;
            for (const auto& kv : groups_) {
                if (kv.second.duplicates.empty()) {
                    continue;
                }
                const Ref& primary = kv.second.primary;
                if (!primary.runtime || primary.collider_index >= primary.runtime->colliders().size()) {
                    continue;
                }
                const auto& coll = primary.runtime->colliders()[primary.collider_index];
                if (coll.sdf_bake_busy && coll.sdf_bake_busy->load(std::memory_order_acquire)) {
                    still_busy = true;
                    break;
                }
            }
            if (!still_busy || std::chrono::steady_clock::now() >= deadline) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        // Share the immutable cooked grid buffer (shared_ptr refcount bump, zero copy)
        // and metadata with every duplicate collider.
        for (const auto& kv : groups_) {
            const Group& group = kv.second;
            if (group.duplicates.empty() || !group.primary.runtime ||
                group.primary.collider_index >= group.primary.runtime->colliders().size()) {
                continue;
            }
            const auto& src = group.primary.runtime->colliders()[group.primary.collider_index];
            if (!src.sdf_grid_data || src.sdf_grid_data->empty()) {
                for (const Ref& dup : group.duplicates) {
                    if (dup.runtime && dup.collider_index < dup.runtime->colliders().size()) {
                        scene.rebuildSDFColliderAsync(
                            dup.runtime->colliders()[dup.collider_index], dup.runtime);
                    }
                }
                continue;
            }
            for (const Ref& dup : group.duplicates) {
                if (!dup.runtime || dup.collider_index >= dup.runtime->colliders().size()) {
                    continue;
                }
                auto& dst = dup.runtime->colliders()[dup.collider_index];
                dst.sdf_grid_data = src.sdf_grid_data;
                dst.sdf_origin = src.sdf_origin;
                dst.sdf_extents = src.sdf_extents;
                dst.sdf_mesh_size = src.sdf_mesh_size;
                dst.sdf_nx = src.sdf_nx;
                dst.sdf_ny = src.sdf_ny;
                dst.sdf_nz = src.sdf_nz;
                dst.sdf_cooked_on_gpu = src.sdf_cooked_on_gpu;
                dst.sdf_cook_ms = 0.0f;
                if (dst.sdf_bake_busy) {
                    dst.sdf_bake_busy->store(false, std::memory_order_release);
                }
            }
        }
    }

private:
    std::unordered_map<std::string, Group> groups_;
};

} // namespace RayTrophiSim
