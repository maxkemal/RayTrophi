/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_data.h
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#pragma once
#include <HittableList.h>
#include <AssimpLoader.h>
#include "AnimationController.h"
#include "KeyframeSystem.h"
#include "VDBVolume.h"
#include "GasVolume.h"
#include "ForceField.h"
#include "ParticleSimulation.h"
#include "SimCache.h"
#include "SimulationSystems.h"
#include "SimulationWorld.h"
#include "SimulationComputeVulkanContext.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>   // std::memcpy — bit-exact float hashing in the sim signatures

// Global atomics to track and cancel active background SDF bakes during scene destruction/clearance.
inline std::atomic<bool> g_cancel_sdf_bakes{false};
inline std::atomic<int> g_active_sdf_bakes{0};
#include "Fluid/FluidObject.h"
#include "Fluid/FluidSimulationSystem.h"
#include "Fluid/SubstanceTag.h"
#include "RigidBodySystem.h"
#include "StructuralImpulse.h"
#include "AshDebrisSystem.h"
#include "MeltSurfaceFlow.h"
#include "Core/RenderStateManager.h"
#include "globals.h"
#include "SurfaceMeshCache.h"
#include "HittableInstance.h"
#include "ColliderMeshBVH.h"
#include "MeshModifiers.h"
#include "GeometryNodesV2.h"
#include "MaterialNodesV2.h"
#include "NodeSystem/SimulationNodes.h"
#include "Paint/PaintTextureSet.h"
#include "Paint/PaintLayerStack.h"

#include <functional>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iterator>
#include <utility>
#include <set>
#include <unordered_map>
#include <map>
#include <array>
#include <unordered_set>
#include <filesystem>

namespace AnimationGraph {
    class AnimationNodeGraph;
}

namespace OzzRuntime {
    struct AnimationSet;
}

/**
 * @brief Central container for all scene data.
 * 
 * Contains:
 * - world: All renderable objects (triangles)
 * - bvh: Acceleration structure for ray tracing
 * - animationDataList: File-based animation data (from FBX/GLTF)
 * - boneData: Skeletal animation bone hierarchy
 * - timeline: Manual keyframe animation data
 * - cameras/lights: Scene lighting and viewpoints
 * - importedModelContexts: Keeps AssimpLoaders alive for animation
 */
struct SceneData {
    SceneData() {
        syncSimulationWorld();
    }

    // UI Settings Serialization Helper
    std::string ui_settings_json_str;  // JSON string storing UI settings
    int load_counter = 0;              // Incremented when a project is loaded
    // =========================================================================
    // Core Geometry
    // =========================================================================
    HittableList world;                                    // All renderable objects
    std::shared_ptr<Hittable> bvh;                         // Acceleration structure
    // Separate, synchronously-rebuilt BVH holding ONLY discrete sim particles
    // (InstanceManager transient groups) for the CPU reference renderer. Kept out
    // of the async `bvh` because particles move every frame: the async build's
    // topology-generation guard discards every in-flight result during playback,
    // so particles would freeze. This small particle-only structure rebuilds in a
    // few ms on the main thread each frame they move; the CPU integrator queries it
    // alongside `bvh`. Null when there are no live particles.
    std::shared_ptr<Hittable> particle_bvh;
    
    // Non-destructive Modeling Cache
    std::unordered_map<std::string, std::vector<std::shared_ptr<Triangle>>> base_mesh_cache; // nodeName -> list of Triangles
    mutable std::unordered_map<std::string, RayTrophiSim::SurfaceMeshCache> surface_mesh_cache; // shared wet/particle/collider surface cache
    mutable uint64_t surface_mesh_cache_version = 1;
    // Per-epoch rebuild memo for getSurfaceMeshCacheForObject(refresh=true): the
    // collider OBB/bounds resolvers re-derive an object's world-space surface
    // (a full O(scene objects) rescan + rebuild) several times per sim step. Within
    // one "geometry epoch" (g_scene_geometry_generation unchanged AND no keyframe
    // re-pose) the world triangles can't change, so an object is rebuilt at most
    // ONCE per epoch and reused. A static high-poly ground/beach collider then
    // resolves once for the whole bake instead of N times per frame.
    mutable std::unordered_set<std::string> surface_cache_epoch_done_;
    mutable uint64_t surface_cache_epoch_gen_ = ~0ull;
    struct SimulationLocalBounds {
        Vec3 min = Vec3(0.0f);
        Vec3 max = Vec3(0.0f);
        uint64_t geometry_generation = 0;
        bool valid = false;
    };
    // Lightweight transform-only collider path. Local bounds are rebuilt only
    // when topology changes; animation then costs one matrix fetch + 8 points.
    mutable std::unordered_map<std::string, SimulationLocalBounds> simulation_local_bounds_;
    // node name -> the scene object that owns it, for
    // resolveObjectTransformForSimulation. Handle only, never a matrix: parented
    // emitters must see live motion. weak_ptr so a deleted object simply misses.
    mutable std::unordered_map<std::string, std::weak_ptr<Hittable>>
        simulation_transform_lookup_;
    // Last sim-source pose matrix actually pushed onto each object by
    // applySimSourceObjectPosesForFrame. Lets that pass be a cheap no-op when the
    // evaluated pose is unchanged (so it can be called every idle UI frame to keep
    // gizmos live), and only erase the surface-cache memo / re-push when the pose
    // truly changed — either the playhead moved OR a keyframe at the current frame
    // was added/edited (which doesn't change the frame number).
    mutable std::unordered_map<std::string, Matrix4x4> last_sim_pose_applied_;
    std::unordered_map<std::string, MeshModifiers::ModifierStack> mesh_modifiers;          // nodeName -> Modifier Stack
    // Faz 8a Geo-DAG: parallel, optional per-object node graph (Base Mesh -> Subdivide (CC) -> ...).
    // Fully additive alongside mesh_modifiers above — the linear ModifierStack panel is untouched
    // and keeps working; this is a separate, opt-in way to build the same kind of geometry chain.
    std::unordered_map<std::string, std::shared_ptr<GeometryNodesV2::GeometryNodeGraphV2>> geometry_node_graphs; // nodeName -> Geo-DAG graph
    // Material node graphs (Faz 1): per-MATERIAL (not per-object) graph that folds
    // into the existing PrincipledBSDF on Apply — see MaterialNodesV2.h header.
    std::unordered_map<std::string, std::shared_ptr<MaterialNodesV2::MaterialNodeGraphV2>> material_node_graphs; // materialName -> graph
    // ── Simulation node graphs, SCOPED ───────────────────────────────────────
    // Decision record: docs/dev/SIMULATION_NODE_OBJECT_MODEL.md, section 8 step 1.
    //
    // ★★★ There is no "the" simulation graph and NO "active domain" default.
    // A graph belongs to a named scene entity, so every caller says which one it
    // means. An implicit default here would be the exact silent-assumption
    // pattern this repository keeps paying for: the call succeeds, edits the
    // wrong domain, and nothing reports it.
    //
    // ★★ Keyed by NAME, like material_node_graphs above — not by pointer or id.
    // A name survives a solver rebuild and means the same thing over IPC.
    //
    // ★★★ A graph whose owner is gone is the "fracture UI state survived the
    // scene change" shape: it keeps drawing, keeps accepting edits, and drives
    // nothing. Domain removal drops its graph at that one call site
    // (rtapi::removeFluidDomain). Object deletion and renaming reach the scene
    // through several paths, so rather than claim they are all hooked, the
    // condition is MEASURED and reported as `owner_missing` on
    // rtapi::simGraphList(). Making it visible is what keeps it from being the
    // silent kind of stale.
    std::unordered_map<std::string, std::shared_ptr<NodeSystem::Sim::SimulationNodeGraph>> simulation_domain_graphs; // domainName -> graph
    std::unordered_map<std::string, std::shared_ptr<NodeSystem::Sim::SimulationNodeGraph>> simulation_object_graphs; // objectName -> graph
    // One world, so one graph and no key. Null until something asks for it.
    std::shared_ptr<NodeSystem::Sim::SimulationNodeGraph> simulation_world_graph;

    std::unordered_map<std::string, Paint::PaintTextureSet> mesh_paint_texture_sets;       // nodeName#materialID -> texture set
    std::unordered_map<std::string, Paint::PaintLayerStack> mesh_paint_layer_stacks;      // nodeName#materialID -> layer stack

    // =========================================================================
    // Animation Data
    // =========================================================================
    std::vector<std::shared_ptr<AnimationData>> animationDataList; // File-based animations
    BoneData boneData;                                     // Bone hierarchy and matrices
    
    // Multi-camera support
    std::vector<std::shared_ptr<Camera>> cameras;  // All cameras in scene
    size_t active_camera_index = 0;                 // Index of currently active camera
    
    // Convenience accessor for active camera (for backward compatibility)
    std::shared_ptr<Camera> camera;  // Points to active camera
    
    std::vector<std::shared_ptr<Light>> lights;
    Vec3 background_color = Vec3(0.2f, 0.2f, 0.2f);
    bool initialized = false;
    ColorProcessor color_processor;
    std::unordered_set<std::string> editor_pending_delete_object_names;
    
    // =========================================================================
    // Object Grouping System
    // =========================================================================
    struct SceneGroup {
        std::string name;
        std::vector<std::string> member_names;  // nodeName list of grouped objects
        bool expanded = true;                    // UI expand state
        
        bool contains(const std::string& obj_name) const {
            return std::find(member_names.begin(), member_names.end(), obj_name) 
                   != member_names.end();
        }
    };
    std::vector<SceneGroup> object_groups;
    
    // Keyframe animation system
    TimelineManager timeline;

    bool isEditorPendingDeleteObjectName(const std::string& nodeName) const {
        return !nodeName.empty() &&
               editor_pending_delete_object_names.find(nodeName) != editor_pending_delete_object_names.end();
    }

    void markObjectPendingDelete(const std::string& nodeName) {
        if (!nodeName.empty()) {
            editor_pending_delete_object_names.insert(nodeName);
            removeParticleBindingsForObjectName(nodeName);
            invalidateSurfaceMeshCache(nodeName);
        }
    }

    void restoreObjectPendingDelete(const std::string& nodeName) {
        if (!nodeName.empty()) {
            editor_pending_delete_object_names.erase(nodeName);
        }
    }

    size_t compactPendingDeletedObjects() {
        if (editor_pending_delete_object_names.empty()) {
            return 0;
        }

        auto matchesPendingDelete = [&](const std::shared_ptr<Hittable>& obj) -> bool {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                return isEditorPendingDeleteObjectName(tri->getNodeName());
            }
            // Flat (direct SoA) node: no per-face facade, so without this branch a deleted flat
            // object never matched here and stayed physically in world.objects forever (even
            // across saves) — only hidden via visible=false, still counted by anything that
            // scans world.objects without checking visibility (e.g. the HUD triangle stats).
            if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                return isEditorPendingDeleteObjectName(tm->nodeName);
            }
            return false;
        };

        const size_t beforeCount = world.objects.size();
        world.objects.erase(
            std::remove_if(world.objects.begin(), world.objects.end(), matchesPendingDelete),
            world.objects.end());

        for (auto& group : object_groups) {
            group.member_names.erase(
                std::remove_if(group.member_names.begin(), group.member_names.end(),
                    [&](const std::string& name) { return isEditorPendingDeleteObjectName(name); }),
                group.member_names.end());
        }

        for (const auto& nodeName : editor_pending_delete_object_names) {
            base_mesh_cache.erase(nodeName);
            invalidateSurfaceMeshCache(nodeName);
            removeParticleBindingsForObjectName(nodeName);
        }

        for (auto& [nodeName, stack] : mesh_modifiers) {
            (void)stack;
            if (isEditorPendingDeleteObjectName(nodeName)) {
                base_mesh_cache.erase(nodeName);
                invalidateSurfaceMeshCache(nodeName);
            }
        }
        for (auto it = mesh_modifiers.begin(); it != mesh_modifiers.end();) {
            if (isEditorPendingDeleteObjectName(it->first)) {
                it = mesh_modifiers.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = mesh_paint_texture_sets.begin(); it != mesh_paint_texture_sets.end();) {
            const std::string& key = it->first;
            const size_t sep = key.find('#');
            const std::string nodeName = (sep == std::string::npos) ? key : key.substr(0, sep);
            if (isEditorPendingDeleteObjectName(nodeName)) {
                it = mesh_paint_texture_sets.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = mesh_paint_layer_stacks.begin(); it != mesh_paint_layer_stacks.end();) {
            const std::string& key = it->first;
            const size_t sep = key.find('#');
            const std::string nodeName = (sep == std::string::npos) ? key : key.substr(0, sep);
            if (isEditorPendingDeleteObjectName(nodeName)) {
                it = mesh_paint_layer_stacks.erase(it);
            } else {
                ++it;
            }
        }

        for (auto& model : importedModelContexts) {
            model.members.erase(
                std::remove_if(model.members.begin(), model.members.end(), matchesPendingDelete),
                model.members.end());
        }

        for (const auto& nodeName : editor_pending_delete_object_names) {
            timeline.tracks.erase(nodeName);
        }

        const size_t removedCount = beforeCount - world.objects.size();
        editor_pending_delete_object_names.clear();
        return removedCount;
    }
    
    // Get active camera (safely)
    std::shared_ptr<Camera> getActiveCamera() const {
        if (cameras.empty()) return camera;  // Fallback to legacy pointer
        if (active_camera_index >= cameras.size()) return cameras[0];
        return cameras[active_camera_index];
    }
    
    // Set active camera by index
    void setActiveCamera(size_t index) {
        if (index < cameras.size()) {
            active_camera_index = index;
            camera = cameras[index];
        }
    }
    
    // Add a camera to the scene
    void addCamera(std::shared_ptr<Camera> cam) {
        cameras.push_back(cam);
        if (cameras.size() == 1) {
            active_camera_index = 0;
            camera = cam;
        }
    }
    
    // Remove a camera from the scene (returns true if successful)
    // SAFETY: Cannot delete the active camera or the last remaining camera
    bool removeCamera(std::shared_ptr<Camera> cam) {
        if (!cam) return false;
        if (cameras.size() <= 1) return false;  // Cannot delete last camera
        
        // Find camera index
        auto it = std::find(cameras.begin(), cameras.end(), cam);
        if (it == cameras.end()) return false;  // Camera not found
        
        size_t index = std::distance(cameras.begin(), it);
        
        // Cannot delete active camera
        if (index == active_camera_index) return false;
        
        // Remove camera
        cameras.erase(it);
        
        // Adjust active_camera_index if needed
        if (active_camera_index > index) {
            active_camera_index--;
        }
        
        // Update camera pointer
        if (!cameras.empty()) {
            camera = cameras[active_camera_index];
        }
        
        return true;
    }
    
    // Imported Model Contexts for Multi-Model Animation
    struct ImportedModelContext {
        struct SkeletonNode {
            std::string name;
            std::string parentName;
            Matrix4x4 localBindTransform = Matrix4x4::identity();
            Matrix4x4 globalBindTransform = Matrix4x4::identity();
            int boneIndex = -1;
            bool weightedBone = false;
            std::vector<int> children;
        };

        std::shared_ptr<class AssimpLoader> loader; // Keep loader alive (owns aiScene)
        std::string importName;
        bool hasAnimation = false;                  // True if this model has animation data
        Matrix4x4 globalInverseTransform;           // Matrix to correct FBX axis/scale (from Root node)
        bool animationOnlyImport = false;          // True when the import has animation/skeleton but no mesh members
        bool hasSkeletonRepresentation = false;    // True when a runtime/editor skeleton view was built
        size_t weightedBoneCount = 0;
        
        // --- Multi-Animator Logic ---
        std::shared_ptr<class AnimationController> animator;  // Per-model animator state
        std::shared_ptr<OzzRuntime::AnimationSet> ozzAnimationSet; // Optional future runtime bridge
        std::string animGraphAssetKey;                 // Editor asset key for this character
        std::shared_ptr<AnimationGraph::AnimationNodeGraph> runtimeGraph; // Per-character runtime graph instance
        std::shared_ptr<AnimationGraph::AnimationNodeGraph> graph; // Legacy alias, keep in sync with runtimeGraph
        bool useAnimGraph = false;                  // Toggle between Controller and Node Graph
        bool preferOzzRuntime = true;              // Future opt-in path for Ozz sampling
        bool loggedOzzRuntimeUsage = false;        // Avoid per-frame runtime path logs
        bool restPoseApplied = false;              // True after rest pose written once; prevents per-frame reset when no clip is active
        bool animGraphFollowTimeline = false;       // Timeline-driven when true, autonomous when false
        bool useRootMotion = false;                 // Move object transform with character
        std::string rootMotionBone;                 // Optional override. Empty = auto detect.
        bool visible = true;                        // Visibility toggle for the whole model
        
        // Link to scene world objects (Triangles/Meshes) belonging to this model
        // This allows applying root motion to the correct TransformHandle
        std::vector<std::shared_ptr<class Hittable>> members; 
        std::vector<SkeletonNode> skeletonNodes;
        std::vector<int> skeletonRootNodes;

        void rebuildSkeletonRepresentation(const BoneData& allBoneData) {
            skeletonNodes.clear();
            skeletonRootNodes.clear();
            weightedBoneCount = 0;
            hasSkeletonRepresentation = false;

            if (importName.empty()) {
                animationOnlyImport = members.empty() && hasAnimation;
                return;
            }

            const std::string prefix = importName + "_";
            std::unordered_map<std::string, int> nodeLookup;

            auto ensureNode = [&](const std::string& fullName) -> int {
                if (fullName.find(prefix) != 0) {
                    return -1;
                }

                auto existing = nodeLookup.find(fullName);
                if (existing != nodeLookup.end()) {
                    return existing->second;
                }

                SkeletonNode node;
                node.name = fullName;

                auto localIt = allBoneData.boneDefaultTransforms.find(fullName);
                if (localIt != allBoneData.boneDefaultTransforms.end()) {
                    node.localBindTransform = localIt->second;
                }

                auto boneIt = allBoneData.boneNameToIndex.find(fullName);
                if (boneIt != allBoneData.boneNameToIndex.end()) {
                    node.boneIndex = static_cast<int>(boneIt->second);
                    node.weightedBone = allBoneData.weightedBoneNames.find(fullName) != allBoneData.weightedBoneNames.end();
                    if (node.weightedBone) {
                        ++weightedBoneCount;
                    }
                }

                int index = static_cast<int>(skeletonNodes.size());
                skeletonNodes.push_back(node);
                nodeLookup[fullName] = index;
                return index;
            };

            for (const auto& [name, local] : allBoneData.boneDefaultTransforms) {
                (void)local;
                ensureNode(name);
            }

            for (const auto& [name, boneIndex] : allBoneData.boneNameToIndex) {
                (void)boneIndex;
                ensureNode(name);
            }

            for (auto& node : skeletonNodes) {
                auto parentIt = allBoneData.boneParents.find(node.name);
                if (parentIt != allBoneData.boneParents.end() && parentIt->second.find(prefix) == 0) {
                    node.parentName = parentIt->second;
                }
            }

            for (int i = 0; i < static_cast<int>(skeletonNodes.size()); ++i) {
                auto& node = skeletonNodes[i];
                if (node.parentName.empty()) {
                    skeletonRootNodes.push_back(i);
                    continue;
                }

                auto parentIt = nodeLookup.find(node.parentName);
                if (parentIt == nodeLookup.end()) {
                    skeletonRootNodes.push_back(i);
                    continue;
                }

                skeletonNodes[parentIt->second].children.push_back(i);
            }

            std::function<void(int, const Matrix4x4&)> computeGlobal = [&](int nodeIndex, const Matrix4x4& parentGlobal) {
                auto& node = skeletonNodes[nodeIndex];
                node.globalBindTransform = parentGlobal * node.localBindTransform;
                for (int childIndex : node.children) {
                    computeGlobal(childIndex, node.globalBindTransform);
                }
            };

            for (int rootIndex : skeletonRootNodes) {
                computeGlobal(rootIndex, Matrix4x4::identity());
            }

            hasSkeletonRepresentation = !skeletonNodes.empty();
            animationOnlyImport = members.empty() && hasAnimation;
        }
    };
    std::vector<ImportedModelContext> importedModelContexts;

    // =========================================================================
    // VDB Volume Objects (Industry-Standard Volumetrics)
    // =========================================================================
    std::vector<std::shared_ptr<VDBVolume>> vdb_volumes;
    
    // Add a VDB volume to the scene
    void addVDBVolume(std::shared_ptr<VDBVolume> vol) {
        if (vol) {
            vdb_volumes.push_back(vol);
        }
    }
    
    // Remove a VDB volume from the scene
    bool removeVDBVolume(std::shared_ptr<VDBVolume> vol) {
        auto it = std::find(vdb_volumes.begin(), vdb_volumes.end(), vol);
        if (it != vdb_volumes.end()) {
            vdb_volumes.erase(it);
            return true;
        }
        return false;
    }
    
    // Find VDB volume by name
    std::shared_ptr<VDBVolume> findVDBVolumeByName(const std::string& name) const {
        for (const auto& vol : vdb_volumes) {
            if (vol && vol->name == name) {
                return vol;
            }
        }
        return nullptr;
    }
    
    // Update VDB volumes from timeline (for animation)
    void updateVDBVolumesFromTimeline(int frame) {
        for (auto& vol : vdb_volumes) {
            if (vol && vol->isLinkedToTimeline()) {
                vol->updateFromTimeline(frame);
            }
        }
    }

    // =========================================================================
    // Gas Simulation Volumes (Real-time/Baked Gas/Smoke)
    // =========================================================================
    std::vector<std::shared_ptr<GasVolume>> gas_volumes;
    std::shared_ptr<RayTrophiSim::GasVolumeSimulationSystem> gas_simulation_system;

    void ensureGasSimulationSystem() {
        syncSimulationWorld();
        if (!gas_simulation_system) {
            gas_simulation_system = std::make_shared<RayTrophiSim::GasVolumeSimulationSystem>();
            simulation_world.addSystem(gas_simulation_system);
        }
        gas_simulation_system->setVolumes(&gas_volumes);
    }
    
    // Add a gas volume to the scene
    void addGasVolume(std::shared_ptr<GasVolume> gas) {
        if (gas) {
            static int gas_id_counter = 0;
            gas->id = gas_id_counter++;
            
            // LINK TO FORCE FIELDS: Critical for simulation to respond to fields
            gas->getSimulator().setExternalForceFieldManager(&this->force_field_manager);
            
            gas_volumes.push_back(gas);
            ensureGasSimulationSystem();

            // Keep gas volumes in the shared hittable list so backend geometry/TLAS rebuilds
            // and viewport picking see the same object set regardless of creation path.
            auto it = std::find(world.objects.begin(), world.objects.end(), gas);
            if (it == world.objects.end()) {
                world.objects.push_back(gas);
            }
        }
    }
    
    // Remove a gas volume from the scene
    bool removeGasVolume(std::shared_ptr<GasVolume> gas) {
        auto it = std::find(gas_volumes.begin(), gas_volumes.end(), gas);
        if (it != gas_volumes.end()) {
            gas_volumes.erase(it);
            if (gas_simulation_system) {
                gas_simulation_system->setVolumes(&gas_volumes);
            }
            world.objects.erase(std::remove(world.objects.begin(), world.objects.end(), gas), world.objects.end());
            return true;
        }
        return false;
    }
    
    // Find gas volume by name
    std::shared_ptr<GasVolume> findGasVolumeByName(const std::string& name) const {
        for (const auto& gas : gas_volumes) {
            if (gas && gas->name == name) {
                return gas;
            }
        }
        return nullptr;
    }
    
    // Update all gas volumes (call from main loop)
    void updateGasVolumes(float dt) {
        ensureGasSimulationSystem();
        simulation_world.stepOnce(dt);
    }
    
    // Update all gas volumes from timeline (for animation sync)
    void updateGasVolumesFromTimeline(int frame) {
        for (auto& gas : gas_volumes) {
            if (gas && gas->isLinkedToTimeline()) {
                gas->updateFromTimeline(frame);
            }
        }
    }

    // =========================================================================
    // Fluid (APIC Liquid) Objects
    // -------------------------------------------------------------------------
    // Lives parallel to gas_volumes / particle_systems. Each FluidObject owns
    // its own particle set + MAC grid; FluidSimulationSystem ticks them all
    // through the shared SimulationWorld on every stepOnce(), so as soon as
    // the system is registered the existing main-loop tick drives it.
    // Render bridges (viewport overlay, NanoVDB SDF, RT shaders) are wired in
    // later phases — at this stage simulation runs but is not visible.
    // =========================================================================
    std::vector<RayTrophiSim::Fluid::FluidObject> fluid_objects;
    std::shared_ptr<RayTrophiSim::FluidSimulationSystem> fluid_simulation_system;
    uint32_t next_fluid_object_id = 1;
    int active_fluid_object_index = -1;

    void ensureFluidSimulationSystem() {
        syncSimulationWorld();
        if (!fluid_simulation_system) {
            fluid_simulation_system = std::make_shared<RayTrophiSim::FluidSimulationSystem>();
            simulation_world.addSystem(fluid_simulation_system);
        }
        fluid_simulation_system->setObjects(&fluid_objects);
    }

    RayTrophiSim::Fluid::FluidObject* addFluidObject(const std::string& name = "Fluid") {
        fluid_objects.emplace_back();
        auto& obj = fluid_objects.back();
        obj.id = next_fluid_object_id++;
        obj.name = name;
        active_fluid_object_index = static_cast<int>(fluid_objects.size()) - 1;
        ensureFluidSimulationSystem();
        return &obj;
    }

    bool removeFluidObject(uint32_t id) {
        auto it = std::find_if(fluid_objects.begin(), fluid_objects.end(),
                               [id](const RayTrophiSim::Fluid::FluidObject& o) { return o.id == id; });
        if (it == fluid_objects.end()) return false;
        destroyFluidRenderVolume(id);
        destroyFluidParticleRenderGroup(*it);
        const int removed_index = static_cast<int>(std::distance(fluid_objects.begin(), it));
        fluid_objects.erase(it);
        if (fluid_simulation_system) fluid_simulation_system->setObjects(&fluid_objects);
        if (fluid_objects.empty()) {
            active_fluid_object_index = -1;
        } else if (active_fluid_object_index >= static_cast<int>(fluid_objects.size())) {
            active_fluid_object_index = static_cast<int>(fluid_objects.size()) - 1;
        } else if (removed_index < active_fluid_object_index) {
            --active_fluid_object_index;
        }
        return true;
    }

    RayTrophiSim::Fluid::FluidObject* activeFluidObject() {
        if (active_fluid_object_index < 0 ||
            active_fluid_object_index >= static_cast<int>(fluid_objects.size())) {
            return nullptr;
        }
        return &fluid_objects[static_cast<std::size_t>(active_fluid_object_index)];
    }

    RayTrophiSim::Fluid::FluidObject* findFluidObjectByName(const std::string& name) {
        for (auto& obj : fluid_objects) {
            if (obj.name == name) return &obj;
        }
        return nullptr;
    }

    // =========================================================================
    // Rigid Bodies (Jolt Physics)
    // -------------------------------------------------------------------------
    // Each RigidBodyObject drives a scene mesh (by nodeName). The RigidBodySystem
    // sizes/poses a Jolt body from the object's oriented bounds, steps it through
    // the shared SimulationWorld, and writes the rigid motion back onto the
    // object's transform. Same lifecycle as gas/fluid systems.
    // =========================================================================
    std::vector<RayTrophiSim::RigidBodyObject> rigid_bodies;
    std::shared_ptr<RayTrophiSim::RigidBodySystem> rigid_body_system;

    // Per-step rigid-fluid coupling level sets (sim resolution), rebuilt by the
    // RigidBodySystem's prepare() callback and read by its sampler() within the
    // SAME step. Covers both FluidObjects and grid-domain fluids. The borrowed
    // grid pointer is only valid during that step (the fluid systems step after
    // the rigid system, so the grids are stable while these are in use).
    struct FluidCouplingField {
        const FluidSim::FluidGrid* grid = nullptr;  // borrowed (within-step only)
        std::vector<float> sdf;                       // sim-res level set (phi<0 inside)
        // Per-XZ-column free-surface height (world Y), size nx*nz, indexed
        // i + k*nx. -1e30 marks a dry column. Lets the rigid sampler measure
        // submersion against where the surface IS rather than against the
        // particle field a sunk body has displaced out of its own cells.
        std::vector<float> surface_h;
    };
    std::vector<FluidCouplingField> fluid_coupling_fields_;

    // Soft-body weld topology cache (node -> mesh mapping). Built by the soft mesh
    // resolver so the writer can scatter the solver's UNIQUE deformed vertices back
    // to every (shared) triangle corner. Keyed by source nodeName.
    struct SoftWeldCache {
        std::vector<std::shared_ptr<Triangle>> tris;  // triangles for this node, in order
        std::vector<uint32_t> corner_unique;          // 3 per triangle: unique vertex idx per corner
        std::vector<Vec3> rest_world_unique;          // rest WORLD pos per unique vertex (Jolt seed + reference)
        std::size_t unique_count = 0;                 // welded vertex count (== solver vertex count)
        // True bind-pose LOCAL positions/normals captured at resolve time (the writer
        // overwrites `original` with the deformed-local geometry the BLAS reads, so we
        // keep the rest here to restore on reset). Parallel to `tris`, 3 per triangle.
        std::vector<std::array<Vec3, 3>> rest_local_pos;
        std::vector<std::array<Vec3, 3>> rest_local_nrm;
        uint64_t geometry_generation = 0;             // g_scene_geometry_generation when captured
        // Flat (direct SoA) rigid body: a flat TriangleMesh-as-Hittable has no per-face facades, so
        // `tris` is empty and the bake writes straight into the mesh's GeometryDetail. flat_mesh is
        // the non-owning mesh; flat_rest_pos/nrm are the per-SoA-vertex rest LOCAL pos/normal.
        class TriangleMesh* flat_mesh = nullptr;
        std::vector<Vec3> flat_rest_pos;
        std::vector<Vec3> flat_rest_nrm;
        // Flat SOFT/cloth body: a flat mesh straight from facadesToFlatMesh is an UNWELDED triangle
        // soup (indices[v]=v, vc = 3*tris). The cloth solver needs a CONNECTED mesh (shared verts)
        // or every triangle is 3 free particles with no constraints. So the flat soft path welds by
        // rest position into `rest_world_unique` (+ remaps corner_unique to those unique ids) and
        // keeps this per-SoA-vertex -> unique map to scatter the welded solver result back onto every
        // duplicate SoA vertex. Empty for the rigid flat bake (which moves the whole mesh, no weld).
        std::vector<uint32_t> flat_soa_to_unique;
    };
    std::unordered_map<std::string, SoftWeldCache> soft_weld_cache_;
    // Rest-pose cache for RIGID bodies that render via vertex baking (see
    // applyRigidBakedTransform). Kept SEPARATE from soft_weld_cache_ so the soft
    // kind==Rigid guards and the soft frame/disk caches never see rigid nodes. Only
    // tris + rest_local_pos/nrm are populated (rigid needs no welding/Jolt seed).
    std::unordered_map<std::string, SoftWeldCache> rigid_bake_cache_;
    // Phase 6c: weld topology for MELTING objects. Separate map for the same
    // reason rigid_bake_cache_ is separate — a melting object is not a soft body
    // and must never be seen as one by the soft freeze/reset/frame-cache paths.
    std::unordered_map<std::string, SoftWeldCache> melt_weld_cache_;
    // Nodes currently displaced by melt. Used only to decide whether a field that
    // has cooled to melt == 0 still needs one final write to put it back at rest.
    std::unordered_map<std::string, uint8_t> melt_displaced_;
    struct MeltSdfRefreshStamp {
        uint64_t revision = 0;
        float mean_melt = 0.0f;
        bool displaced = false;
    };
    std::unordered_map<std::string, MeltSdfRefreshStamp> melt_sdf_refresh_stamp_;
    struct PendingSdfBake {
        std::string node_name;
        std::shared_ptr<RayTrophiSim::ParticleSimulationSystem> runtime;
        std::shared_ptr<std::vector<float>> grid;
        Vec3 origin;
        Vec3 extents;
        int nx = 0, ny = 0, nz = 0;
        std::shared_ptr<std::atomic<uint64_t>> serial;
        std::shared_ptr<std::atomic<bool>> busy;
        uint64_t request_serial = 0;
    };
    std::mutex pending_sdf_bakes_mutex_;
    std::vector<PendingSdfBake> pending_sdf_bakes_;
    struct MeltAppliedStamp {
        uint64_t topology = 0;
        uint64_t content = 0;
    };
    // The render loop calls applyMeltDisplacement every frame so timeline
    // restores are visible while paused. Do not rewrite identical vertices:
    // that would dirty/refit BLAS every frame and permanently reset both Vulkan
    // and OptiX accumulation even though the simulation state is stationary.
    std::unordered_map<std::string, MeltAppliedStamp> melt_applied_stamp_;
    // Rewind restores fluid/render resources in a burst. Hold melt mesh
    // write-back for one render frame so a heavy source BLAS refit does not land
    // in the same Vulkan submission window as fluid surface destruction/rebuild.
    bool defer_melt_displacement_once_ = false;
    uint32_t fracture_summary_tick_ = 0;
    uint64_t structural_impulse_sequence_ = 0;
    std::vector<RayTrophiSim::StructuralImpulseEvent> structural_impulse_events_;
    // fracture group -> the source object it was cut from. MSF fields are keyed
    // by object, fracture groups by cluster, and the two stopped being the same
    // string the moment clustering arrived.
    std::map<std::string, std::string> fracture_group_source_;
    // Per gas domain: seconds since it last emitted a blast event. Keyed by name
    // rather than index because domains are added and removed underneath us.
    std::map<std::string, float> combustion_event_clock_;
    RayTrophiSim::StructuralImpulseStats structural_impulse_stats_;
    RayTrophiSim::AshDebrisSystem ash_debris_system_;

    // ── Per-mesh deform refit (all backends) ─────────────────────────────────
    // A simulated body (rigid / soft / fracture shard) bakes new verts into its
    // source mesh EVERY step. Routing that through markDirty(Geometry) destroyed +
    // rebuilt EVERY BLAS / raster mesh / CPU BVH in the scene per frame — so a few
    // moving bodies (e.g. a shattered wall) froze playback in Solid AND render modes.
    // Instead we record the changed node here and refit ONLY that node in place on
    // whichever path is active (raster Solid: updateRasterMeshFromTriangles, OptiX /
    // Vulkan RT: updateInteractiveMesh) + a cheap CPU Embree refit — promoting to a
    // full rebuild only if a refit fails (topology change). Set per-frame by Main via
    // setDeformRefitActive() (true whenever the active path supports a per-mesh refit).
    std::unordered_set<std::string> pending_deform_nodes_;
    bool deform_refit_active_ = false;
    // Bumped whenever ANY body's mesh verts change (sim write-back / reset). The
    // selection gizmo/outline memoizes a body's world-AABB against this so a STATIC
    // (stopped) body costs O(1) per frame instead of re-walking its triangles every
    // frame — that per-frame walk was why a selected body pinned the idle UI at ~6%
    // CPU while a plain (bbox-cached) object stayed at ~0%.
    uint64_t body_geom_version_ = 1;
    bool ui_mesh_cache_rebuild_request_ = false;  // see requestUiMeshCacheRebuild()

    // Free-surface height (world Y) the fluid reaches AROUND a world point — the
    // MAX column surface over a small XZ neighbourhood, NOT the local column top.
    // A floating/sunk body displaces fluid out of its own column, so that
    // column's top reads at the body's underside (or is dry); the body's true
    // submersion reference is the SURROUNDING water level, which the neighbourhood
    // max recovers. Robust for both floaters (surrounding waterline) and sunk
    // bodies (the tank surface above them). Returns -1e30 if no fluid is near.
    float sampleFluidColumnSurface(const FluidCouplingField& f, const Vec3& wp) const {
        if (!f.grid) return -1.0e30f;
        const auto& g = *f.grid;
        if (f.surface_h.size() != static_cast<size_t>(g.nx) * g.nz) return -1.0e30f;
        const int ic = static_cast<int>(std::floor((wp.x - g.origin.x) / g.voxel_size));
        const int kc = static_cast<int>(std::floor((wp.z - g.origin.z) / g.voxel_size));
        constexpr int R = 3;  // ~few cells reaches open water beside a small body
        float best = -1.0e30f;
        for (int dk = -R; dk <= R; ++dk) {
            const int k = kc + dk;
            if (k < 0 || k >= g.nz) continue;
            for (int di = -R; di <= R; ++di) {
                const int i = ic + di;
                if (i < 0 || i >= g.nx) continue;
                const float h = f.surface_h[static_cast<size_t>(i) + static_cast<size_t>(k) * g.nx];
                if (h > best) best = h;
            }
        }
        return best;
    }

    // AMBIENT (wave) fluid velocity around a world point. A floating/submerged body
    // displaces fluid out of its own cells AND the solver stamps the body's own
    // velocity (solid_vel) into them, so sampling the grid velocity AT the body
    // reads ~the body's own motion → zero relative velocity → drag can only damp
    // the body, never let WAVES drag it. Instead we average the grid velocity over
    // genuine-fluid samples in a small neighbourhood (the surrounding water column
    // + sides + below), which carries the wave flow. Returns false if no fluid is
    // near (point not really in/under water). voxel-scaled offsets, cheap (8 taps).
    bool sampleFluidAmbientVelocity(const FluidCouplingField& f, const Vec3& wp, Vec3& out_vel) const {
        if (!f.grid) return false;
        const auto& g = *f.grid;
        const float h = g.voxel_size;
        const Vec3 taps[8] = {
            Vec3(0.0f, 0.0f, 0.0f),
            Vec3( 2.0f * h, 0.0f, 0.0f), Vec3(-2.0f * h, 0.0f, 0.0f),
            Vec3(0.0f, 0.0f,  2.0f * h), Vec3(0.0f, 0.0f, -2.0f * h),
            Vec3(0.0f, -1.5f * h, 0.0f), Vec3(0.0f, -3.0f * h, 0.0f),
            Vec3(0.0f,  1.0f * h, 0.0f)
        };
        Vec3 acc(0.0f, 0.0f, 0.0f);
        int n = 0;
        for (const Vec3& o : taps) {
            const Vec3 p = wp + o;
            if (g.sampleCellCentered(f.sdf, p) < 0.0f) {  // genuine fluid (not cavity/air)
                const Vec3 v = g.sampleVelocity(p);
                if (std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z)) {
                    acc = acc + v;
                    ++n;
                }
            }
        }
        if (n == 0) return false;
        out_vel = acc * (1.0f / static_cast<float>(n));
        return true;
    }

    std::string rigidBodyProxyColliderName(const std::string& node_name) const {
        return node_name.empty() ? "Rigid Body Proxy Collider" : node_name + " Rigid Body Proxy Collider";
    }

    void upsertRigidBodyProxyCollider(RayTrophiSim::ParticleSimulationSystem& runtime,
                                      const RayTrophiSim::RigidBodyObject& rb) {
        if (rb.source_name.empty()) return;

        RayTrophiSim::ParticleColliderDesc desc;
        desc.name = rigidBodyProxyColliderName(rb.source_name);
        desc.source_mode = RayTrophiSim::ParticleColliderSourceMode::ObjectOBB;
        desc.source_name = rb.source_name;
        desc.enabled = rb.enabled;
        desc.restitution = rb.restitution;
        desc.friction = rb.friction;
        desc.thickness = 0.02f;
        fitParticleColliderToObjectBounds(desc, rb.source_name, true);
        desc.name = rigidBodyProxyColliderName(rb.source_name);

        auto& colliders = runtime.colliders();
        for (auto& collider : colliders) {
            if (collider.name == desc.name && collider.source_name == rb.source_name) {
                collider = desc;
                return;
            }
        }
        runtime.addCollider(desc);
    }

    void removeRigidBodyProxyColliders(const std::string& node_name) {
        if (node_name.empty()) return;
        const std::string proxy_name = rigidBodyProxyColliderName(node_name);
        for (auto& system : particle_systems) {
            if (!system.runtime) continue;
            auto& colliders = system.runtime->colliders();
            colliders.erase(
                std::remove_if(colliders.begin(), colliders.end(),
                    [&](const RayTrophiSim::ParticleColliderDesc& collider) {
                        return collider.name == proxy_name && collider.source_name == node_name;
                    }),
                colliders.end());
        }
    }

    bool isRigidBodyProxyCollider(const RayTrophiSim::ParticleColliderDesc& collider,
                                  const std::string& node_name) const {
        return !node_name.empty() &&
               collider.name == rigidBodyProxyColliderName(node_name) &&
               collider.source_name == node_name;
    }

    RayTrophiSim::ParticleColliderDesc* findAuthoredColliderForRigidBody(RayTrophiSim::RigidBodyObject& rb) {
        for (auto& system : particle_systems) {
            if (!system.runtime) continue;
            for (auto& collider : system.runtime->colliders()) {
                if (!collider.enabled) continue;
                if (!rb.collider_name.empty() && collider.name == rb.collider_name) return &collider;
            }
        }
        for (auto& system : particle_systems) {
            if (!system.runtime) continue;
            for (auto& collider : system.runtime->colliders()) {
                if (!collider.enabled || isRigidBodyProxyCollider(collider, rb.source_name)) continue;
                if (!rb.source_name.empty() && collider.source_name == rb.source_name) return &collider;
            }
        }
        return nullptr;
    }

    const RayTrophiSim::ParticleColliderDesc* findAuthoredColliderForRigidBody(const RayTrophiSim::RigidBodyObject& rb) const {
        for (const auto& system : particle_systems) {
            if (!system.runtime) continue;
            for (const auto& collider : system.runtime->colliders()) {
                if (!collider.enabled) continue;
                if (!rb.collider_name.empty() && collider.name == rb.collider_name) return &collider;
            }
        }
        for (const auto& system : particle_systems) {
            if (!system.runtime) continue;
            for (const auto& collider : system.runtime->colliders()) {
                if (!collider.enabled || isRigidBodyProxyCollider(collider, rb.source_name)) continue;
                if (!rb.source_name.empty() && collider.source_name == rb.source_name) return &collider;
            }
        }
        return nullptr;
    }

    Matrix4x4 rigidPoseFromCenter(const Vec3& center) const {
        Matrix4x4 pose = Matrix4x4::identity();
        pose.m[0][3] = center.x;
        pose.m[1][3] = center.y;
        pose.m[2][3] = center.z;
        return pose;
    }

    Matrix4x4 rigidPoseFromCapsuleSegment(const Vec3& start, const Vec3& end) const {
        const Vec3 center = (start + end) * 0.5f;
        const Vec3 segment = end - start;
        const float len = segment.length();
        const Vec3 axis_y = len > 1e-6f ? segment * (1.0f / len) : Vec3(0.0f, 1.0f, 0.0f);
        const Vec3 helper = std::fabs(axis_y.y) < 0.95f ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
        Vec3 axis_x = Vec3::cross(helper, axis_y);
        const float x_len = axis_x.length();
        axis_x = x_len > 1e-6f ? axis_x * (1.0f / x_len) : Vec3(1.0f, 0.0f, 0.0f);
        Vec3 axis_z = Vec3::cross(axis_y, axis_x);
        const float z_len = axis_z.length();
        axis_z = z_len > 1e-6f ? axis_z * (1.0f / z_len) : Vec3(0.0f, 0.0f, 1.0f);

        Matrix4x4 pose = Matrix4x4::identity();
        pose.m[0][0] = axis_x.x; pose.m[1][0] = axis_x.y; pose.m[2][0] = axis_x.z;
        pose.m[0][1] = axis_y.x; pose.m[1][1] = axis_y.y; pose.m[2][1] = axis_y.z;
        pose.m[0][2] = axis_z.x; pose.m[1][2] = axis_z.y; pose.m[2][2] = axis_z.z;
        pose.m[0][3] = center.x; pose.m[1][3] = center.y; pose.m[2][3] = center.z;
        return pose;
    }

    bool resolveRigidBodyColliderShape(const RayTrophiSim::RigidBodyObject& rb,
                                       Matrix4x4& out_box_pose,
                                       Vec3& out_half,
                                       RayTrophiSim::RigidBodyShape& out_shape) const {
        const auto* collider = findAuthoredColliderForRigidBody(rb);
        if (!collider) return false;

        const float kMinHalf = 0.025f;
        switch (collider->source_mode) {
            case RayTrophiSim::ParticleColliderSourceMode::Sphere:
                out_shape = RayTrophiSim::RigidBodyShape::Sphere;
                out_half = Vec3(std::max(collider->sphere_radius, kMinHalf),
                                std::max(collider->sphere_radius, kMinHalf),
                                std::max(collider->sphere_radius, kMinHalf));
                out_box_pose = rigidPoseFromCenter(collider->sphere_center);
                return true;
            case RayTrophiSim::ParticleColliderSourceMode::Capsule: {
                const float len = (collider->capsule_end - collider->capsule_start).length();
                const float radius = std::max(collider->capsule_radius, kMinHalf);
                out_shape = RayTrophiSim::RigidBodyShape::Capsule;
                out_half = Vec3(radius, std::max(kMinHalf, len * 0.5f + radius), radius);
                out_box_pose = rigidPoseFromCapsuleSegment(collider->capsule_start, collider->capsule_end);
                return true;
            }
            case RayTrophiSim::ParticleColliderSourceMode::PlaneY:
                out_shape = RayTrophiSim::RigidBodyShape::Box;
                out_half = Vec3(500.0f, kMinHalf, 500.0f);
                out_box_pose = rigidPoseFromCenter(Vec3(0.0f, collider->plane_y - kMinHalf, 0.0f));
                return true;
            case RayTrophiSim::ParticleColliderSourceMode::ObjectAABB: {
                const Vec3 mn = Vec3::min(collider->bounds_min, collider->bounds_max);
                const Vec3 mx = Vec3::max(collider->bounds_min, collider->bounds_max);
                out_shape = RayTrophiSim::RigidBodyShape::Box;
                out_half = (mx - mn) * 0.5f;
                out_half.x = std::max(out_half.x, kMinHalf);
                out_half.y = std::max(out_half.y, kMinHalf);
                out_half.z = std::max(out_half.z, kMinHalf);
                out_box_pose = rigidPoseFromCenter((mn + mx) * 0.5f);
                return true;
            }
            case RayTrophiSim::ParticleColliderSourceMode::ObjectOBB:
            case RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF:
            case RayTrophiSim::ParticleColliderSourceMode::ObjectConvexDecomp:
            case RayTrophiSim::ParticleColliderSourceMode::ObjectMeshBVH: {
                RayTrophiSim::ParticleColliderOBB obb;
                if (!resolveObjectOBBForSimulation(collider->source_name, obb)) return false;
                const Vec3 mn = obb.local_bounds_min;
                const Vec3 mx = obb.local_bounds_max;
                // ObjectOBB stays an oriented box. The mesh-derived modes (SDF /
                // convex-decomp / mesh-BVH) want the ACTUAL mesh boundary, so route
                // them to the Mesh shape (exact triangle mesh when static, convex
                // hull when dynamic). The OBB-derived half-extents/pose below are
                // still emitted: they are the fluid-coupling volume fallback and the
                // shape used if the source triangles can't be resolved this tick.
                out_shape = (collider->source_mode ==
                             RayTrophiSim::ParticleColliderSourceMode::ObjectOBB)
                                ? RayTrophiSim::RigidBodyShape::Box
                                : RayTrophiSim::RigidBodyShape::Mesh;
                out_half = (mx - mn) * 0.5f;
                out_half.x = std::max(out_half.x, kMinHalf);
                out_half.y = std::max(out_half.y, kMinHalf);
                out_half.z = std::max(out_half.z, kMinHalf);
                // POINT transform (must include the centroid translation). Using
                // operator* (vector transform, drops translation) placed the box
                // centre at ~world origin instead of the object's centre; the
                // re-pose then swung the body around the world origin the instant
                // it rotated (rotated source object "jumped to -Y").
                const Vec3 center_world = obb.local_to_world.transform_point((mn + mx) * 0.5f);
                out_box_pose = obb.local_to_world;
                out_box_pose.m[0][3] = center_world.x;
                out_box_pose.m[1][3] = center_world.y;
                out_box_pose.m[2][3] = center_world.z;
                return true;
            }
        }
        return false;
    }

    void syncRigidBodyProxyColliders() {
        for (auto& rb : rigid_bodies) {
            if (!rb.enabled) {
                removeRigidBodyProxyColliders(rb.source_name);
                continue;
            }
            // Only Rigid bodies expose a proxy collider so the fluid/particle
            // solver can see them. Soft / Cloth bodies are deformable (and not
            // simulated yet) — a rigid box proxy would misrepresent them.
            if (rb.kind != RayTrophiSim::BodyKind::Rigid) {
                removeRigidBodyProxyColliders(rb.source_name);
                continue;
            }

            RayTrophiSim::ParticleColliderDesc* authored = nullptr;
            for (auto& system : particle_systems) {
                if (!system.runtime) continue;
                for (auto& collider : system.runtime->colliders()) {
                    if (!collider.enabled || isRigidBodyProxyCollider(collider, rb.source_name)) continue;
                    if (!rb.collider_name.empty() && collider.name == rb.collider_name) {
                        authored = &collider;
                        break;
                    }
                }
                if (authored) break;
            }
            if (!authored) {
                for (auto& system : particle_systems) {
                    if (!system.runtime) continue;
                    for (auto& collider : system.runtime->colliders()) {
                        if (!collider.enabled || isRigidBodyProxyCollider(collider, rb.source_name)) continue;
                        if (!rb.source_name.empty() && collider.source_name == rb.source_name) {
                            authored = &collider;
                            break;
                        }
                    }
                    if (authored) break;
                }
            }

            if (authored) {
                const std::string authored_name = authored->name;
                const float authored_friction = authored->friction;
                const float authored_restitution = authored->restitution;
                removeRigidBodyProxyColliders(rb.source_name);
                rb.collider_name = authored_name;
                rb.friction = authored_friction;
                rb.restitution = authored_restitution;
                continue;
            }

            const std::string proxy_name = rigidBodyProxyColliderName(rb.source_name);
            for (auto& system : particle_systems) {
                if (!system.runtime) continue;
                for (auto& collider : system.runtime->colliders()) {
                    if (collider.name == proxy_name && collider.source_name == rb.source_name) {
                        rb.friction = collider.friction;
                        rb.restitution = collider.restitution;
                        rb.collider_name = proxy_name;
                    }
                }
            }
            rb.collider_name = proxy_name;
            for (auto& system : particle_systems) {
                if (system.runtime) upsertRigidBodyProxyCollider(*system.runtime, rb);
            }
        }
    }

    bool captureRigidBodyRestPose(RayTrophiSim::RigidBodyObject& rb) {
        if (rb.source_name.empty()) return false;

        Matrix4x4 pivot = Matrix4x4::identity();
        bool have_pivot = false;
        for (auto& obj : world.objects) {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                tri && tri->getNodeName() == rb.source_name) {
                if (Transform* th = tri->getTransformPtr()) {
                    pivot = th->getPivotMatrix();
                    have_pivot = true;
                }
                break;
            } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj);
                       tm && tm->nodeName == rb.source_name) {
                if (Transform* th = tm->transform.get()) {
                    pivot = th->getPivotMatrix();
                    have_pivot = true;
                }
                break;
            }
        }
        if (!have_pivot) return false;

        Matrix4x4 body_pose = Matrix4x4::identity();
        Vec3 half;
        RayTrophiSim::RigidBodyShape resolved_shape = rb.shape;
        if (!resolveRigidBodyColliderShape(rb, body_pose, half, resolved_shape)) {
            RayTrophiSim::ParticleColliderOBB obb;
            if (!resolveObjectOBBForSimulation(rb.source_name, obb)) return false;
            half = (obb.local_bounds_max - obb.local_bounds_min) * 0.5f;
        }
        const float kMinHalf = 0.025f;
        half.x = std::max(half.x, kMinHalf);
        half.y = std::max(half.y, kMinHalf);
        half.z = std::max(half.z, kMinHalf);

        rb.initial_pivot = pivot;
        rb.rest_half_extents = half;
        rb.shape = resolved_shape;
        rb.rest_captured = true;
        rb.created = false;
        rb.handle = 0xffffffffu;
        rb.has_written = false;
        return true;
    }

    void invalidateRigidBodySimulationCache() {
        clearSimFrameCache();
        sim_timeline_frame_ = -1;
        rigid_timeline_frame_ = -1;
        sim_cache_valid_ = false;
        sim_cache_dir_.clear();
        sim_cache_valid_system_ids_.clear();
        last_sim_config_sig_ = 0;
        last_fluid_coupling_sig_ = 0;
        // Back to "nothing baked yet": the next tick adopts whatever frame rate
        // it is handed instead of reporting a change against a dead bake.
        last_sim_bake_fps_ = 0.0f;
    }

    void ensureRigidBodySystem() {
        syncSimulationWorld();
        if (!rigid_body_system) {
            rigid_body_system = std::make_shared<RayTrophiSim::RigidBodySystem>();

            // Shape + initial pose: derive an oriented box from the object's live
            // world verts (same OBB the particle colliders use), then move the pose
            // to the box CENTRE and report half-extents.
            rigid_body_system->setShapeResolver(
                [this](const RayTrophiSim::RigidBodyObject& rb,
                       Matrix4x4& out_box_pose,
                       Vec3& out_half,
                       RayTrophiSim::RigidBodyShape& out_shape) -> bool {
                    if (resolveRigidBodyColliderShape(rb, out_box_pose, out_half, out_shape)) {
                        return true;
                    }

                    RayTrophiSim::ParticleColliderOBB obb;
                    const std::string& node = rb.source_name;
                    if (!resolveObjectOBBForSimulation(node, obb)) return false;
                    out_shape = rb.shape;
                    const Vec3 mn = obb.local_bounds_min;
                    const Vec3 mx = obb.local_bounds_max;
                    out_half = (mx - mn) * 0.5f;
                    // Clamp thin axes: a flat ground plane has ~0 thickness, which
                    // would make a degenerate 2D box the rigid body tunnels through.
                    // A solid slab (min 2.5cm half-thickness) blocks reliably.
                    const float kMinHalf = 0.025f;
                    out_half.x = std::max(out_half.x, kMinHalf);
                    out_half.y = std::max(out_half.y, kMinHalf);
                    out_half.z = std::max(out_half.z, kMinHalf);
                    const Vec3 c_local = (mn + mx) * 0.5f;
                    // POINT transform (include centroid translation). operator* is a
                    // vector transform that DROPS translation, which put the box
                    // centre at ~world origin → the body swung around the origin as
                    // soon as it rotated (rotated source "jumped to -Y" then rose).
                    const Vec3 center_world = obb.local_to_world.transform_point(c_local);
                    out_box_pose = obb.local_to_world;  // keep orthonormal rotation columns
                    out_box_pose.m[0][3] = center_world.x;
                    out_box_pose.m[1][3] = center_world.y;
                    out_box_pose.m[2][3] = center_world.z;
                    return true;
                });

            rigid_body_system->setPivotGetter(
                [this](const std::string& node, Matrix4x4& out_pivot) -> bool {
                    for (auto& obj : world.objects) {
                        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                            if (tri->getNodeName() == node) {
                                if (Transform* th = tri->getTransformPtr()) {
                                    out_pivot = th->getPivotMatrix();
                                    return true;
                                }
                            }
                        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                            // Flat (direct SoA) mesh: pivot lives on its own Transform handle.
                            if (tm->nodeName == node && tm->transform) {
                                out_pivot = tm->transform->getPivotMatrix();
                                return true;
                            }
                        }
                    }
                    return false;
                });

            rigid_body_system->setPivotSetter(
                [this](const std::string& node, const Matrix4x4& pivot) {
                    auto matrixEqual = [](const Matrix4x4& a, const Matrix4x4& b) {
                        for (int r = 0; r < 4; ++r) {
                            for (int c = 0; c < 4; ++c) {
                                if (a.m[r][c] != b.m[r][c]) return false;
                            }
                        }
                        return true;
                    };
                    bool changed = false;
                    for (auto& obj : world.objects) {
                        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                            if (tri->getNodeName() == node) {
                                if (Transform* th = tri->getTransformPtr()) {
                                    if (!matrixEqual(th->getPivotMatrix(), pivot)) {
                                        th->setPivotMatrix(pivot);
                                        changed = true;
                                    }
                                }
                            }
                        } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                            // Flat (direct SoA) mesh: pivot lives on its own Transform handle.
                            if (tm->nodeName == node && tm->transform) {
                                if (!matrixEqual(tm->transform->getPivotMatrix(), pivot)) {
                                    tm->transform->setPivotMatrix(pivot);
                                    changed = true;
                                }
                            }
                        }
                    }
                    if (!changed) return;
                    // Object moved without bumping geometry generation: drop the
                    // surface-cache epoch memo so any future resolve rebuilds.
                    surface_cache_epoch_done_.erase(node);
                    // The mesh moved, but topology did not. Request transform/refit
                    // only; a full geometry rebuild every Jolt step is far too heavy.
                    Core::RenderStateManager::instance().markDirty(Core::DirtyScope::Transforms);
                });

            // Rigid render write-back: bake the body's world-space rigid delta into
            // the source mesh verts (NOT the transform handle — that corrupted
            // imported/non-TRS meshes from frame 0). Mirrors the soft render path but
            // preserves authored per-corner normals. See applyRigidBakedTransform.
            rigid_body_system->setRigidMeshBaker(
                [this](const std::string& node, const Matrix4x4& world_delta) {
                    applyRigidBakedTransform(node, world_delta);
                });

            // ── Soft-body geometry I/O ────────────────────────────────────────
            // resolver(): (re)build the weld cache from the rest mesh and hand Jolt
            // the unique rest world vertices + face indices. The weld key is the REST
            // world position (transform * original), stable across the sim (we never
            // touch a soft source's transform/original), so shared corners collapse to
            // one particle and the mesh stays connected. See rebuildSoftWeldCache.
            rigid_body_system->setSoftMeshResolver(
                [this](const RayTrophiSim::RigidBodyObject& rb,
                       std::vector<Vec3>& out_vertices,
                       std::vector<uint32_t>& out_indices) -> bool {
                    if (!rebuildSoftWeldCache(rb.source_name)) return false;
                    const SoftWeldCache& cache = soft_weld_cache_[rb.source_name];
                    out_vertices = cache.rest_world_unique;
                    out_indices = cache.corner_unique;
                    return out_vertices.size() >= 3;
                });

            // writer(): scatter the solver's unique deformed vertices back onto every
            // triangle corner. The GPU BLAS is built from LOCAL vertices
            // (getOriginalVertexPosition) + the object's instance transform, so the
            // deformation must go into `original` (= inverse(transform) * world); we
            // also set `position` (world) so the CPU/world paths agree. Then request a
            // geometry rebuild. A hand-rolled flat normal per triangle (Vec3::normalize
            // zeroes tiny vectors, which would blank thin triangles).
            rigid_body_system->setSoftMeshWriter(
                [this](const std::string& node, const std::vector<Vec3>& world_verts) {
                    applySoftDeformedVerts(node, world_verts);
                });

            // resetToRest(): the writer overwrote each triangle's `original` (local)
            // with the deformed geometry, so restore the cached bind-pose local first,
            // then recompute world positions; finally drop the cache so the next create
            // re-resolves a clean rest.
            rigid_body_system->setSoftMeshResetToRest(
                [this](const std::string& node) { restoreSoftRestMesh(node); });

            // resumeState(): when playback runs PAST the RAM cache, the soft body was
            // left uncreated during the cached replay and is rebuilt from REST. Hand
            // back the cached deformed verts of the frame we're resuming FROM
            // (soft_resume_frame_) plus a finite-difference velocity, so the body
            // continues from there instead of re-animating from rest (the reported
            // "sim recomputes from the start past frame N" bug). Returns false on a
            // first bake / when that frame isn't cached.
            rigid_body_system->setSoftResumeProvider(
                [this](const std::string& node,
                       std::vector<Vec3>& out_positions,
                       std::vector<Vec3>& out_velocities) -> bool {
                    out_positions.clear();
                    out_velocities.clear();
                    if (soft_resume_frame_ < 1) return false;
                    auto it = soft_frame_cache_.find(soft_resume_frame_);
                    if (it == soft_frame_cache_.end()) return false;
                    auto nit = it->second.find(node);
                    if (nit == it->second.end() || nit->second.empty()) return false;
                    out_positions = nit->second;
                    // Velocity from the previous cached frame (so the resume keeps the
                    // body's momentum). Skip if absent or topology changed.
                    auto pit = soft_frame_cache_.find(soft_resume_frame_ - 1);
                    if (pit != soft_frame_cache_.end()) {
                        auto pnit = pit->second.find(node);
                        if (pnit != pit->second.end() &&
                            pnit->second.size() == out_positions.size()) {
                            const float inv_dt =
                                (soft_resume_dt_ > 1.0e-6f) ? (1.0f / soft_resume_dt_) : 0.0f;
                            out_velocities.resize(out_positions.size());
                            for (std::size_t i = 0; i < out_positions.size(); ++i)
                                out_velocities[i] = (out_positions[i] - pnit->second[i]) * inv_dt;
                        }
                    }
                    return true;
                });

            // ── Rigid-fluid coupling (buoyancy + drag) ────────────────────────
            // prepare(): once per step (only when a coupled body exists), rebuild
            // a sim-resolution coupling level set for EVERY fluid source — both
            // standalone FluidObjects AND grid-domain fluids (particle systems).
            // Built into SceneData-side scratch (fluid_coupling_fields_) because
            // grid-domain states are exposed const; borrowed grid pointers stay
            // valid for the duration of this step (the owning systems step later,
            // at order >= 100). Separate from the render SDF (refined + render-
            // mode gated), so it can be sampled with grid.sampleCellCentered().
            rigid_body_system->setFluidCouplingPrepare([this]() {
                fluid_coupling_fields_.clear();
                auto build_for = [this](const RayTrophiSim::Fluid::FluidParticles& parts,
                                        const FluidSim::FluidGrid& grid,
                                        const RayTrophiSim::Fluid::LevelSetParams& base) {
                    if (parts.size() == 0 || grid.nx <= 0) return;
                    RayTrophiSim::Fluid::LevelSetParams lp = base;
                    lp.surface_resolution_multiplier = 1;  // sim grid (sample-able)
                    lp.anisotropy_enabled = false;         // cheap + robust, not pretty
                    // CRITICAL for coupling: empty cells far above the water must
                    // read as clearly OUTSIDE, not the small +narrow_band sentinel
                    // (buildLevelSet clamps phi to ±narrow_band and fills empty
                    // cells with far_value=narrow_band). With the default 3 voxels
                    // a body reads partial submersion at ANY height and floats
                    // mid-domain. Push the sentinel far out and disable smoothing
                    // so the far-cell value can't smear back down toward the
                    // surface. Real (in-water) cells are bounded by the kernel
                    // radius, so widening the clamp never affects them.
                    lp.narrow_band_voxels = 64.0f;
                    lp.smoothing_iterations = 0;
                    FluidCouplingField field;
                    field.grid = &grid;
                    if (RayTrophiSim::Fluid::buildLevelSet(parts, grid, lp, field.sdf, nullptr)) {
                        // Per-column free-surface height: scan each XZ column from
                        // the top down for the highest fluid cell (phi<0) and
                        // refine to the zero-crossing into the cell above. This is
                        // the height fluid reaches in that column independent of
                        // any cavity a sunk body carved, so submersion stays
                        // correct for fully submerged bodies.
                        const int gnx = grid.nx, gny = grid.ny, gnz = grid.nz;
                        const float vs = grid.voxel_size;
                        const float oy = grid.origin.y;
                        field.surface_h.assign(static_cast<size_t>(gnx) * gnz, -1.0e30f);
                        for (int k = 0; k < gnz; ++k) {
                            for (int i = 0; i < gnx; ++i) {
                                int j_top = -1;
                                for (int j = gny - 1; j >= 0; --j) {
                                    if (field.sdf[static_cast<size_t>(i) +
                                                  static_cast<size_t>(j) * gnx +
                                                  static_cast<size_t>(k) * gnx * gny] < 0.0f) {
                                        j_top = j;
                                        break;
                                    }
                                }
                                if (j_top < 0) continue;  // dry column
                                float surf = oy + (j_top + 0.5f) * vs;  // top fluid cell centre
                                if (j_top + 1 < gny) {
                                    const float phi0 = field.sdf[static_cast<size_t>(i) +
                                        static_cast<size_t>(j_top) * gnx +
                                        static_cast<size_t>(k) * gnx * gny];
                                    const float phi1 = field.sdf[static_cast<size_t>(i) +
                                        static_cast<size_t>(j_top + 1) * gnx +
                                        static_cast<size_t>(k) * gnx * gny];
                                    if (phi1 > phi0) {
                                        float frac = -phi0 / (phi1 - phi0);
                                        frac = std::min(1.0f, std::max(0.0f, frac));
                                        surf += frac * vs;
                                    }
                                } else {
                                    surf = oy + gny * vs;  // fluid reaches domain top
                                }
                                field.surface_h[static_cast<size_t>(i) +
                                                static_cast<size_t>(k) * gnx] = surf;
                            }
                        }
                        fluid_coupling_fields_.push_back(std::move(field));
                    }
                };
                // Standalone APIC FluidObjects.
                for (auto& obj : fluid_objects) {
                    if (!obj.enabled) continue;
                    obj.ensureGrid();
                    build_for(obj.particles, obj.grid, obj.level_set_params);
                }
                // Grid-domain fluids living inside particle systems.
                const RayTrophiSim::Fluid::LevelSetParams kDomainLevelSet{};
                for (auto& sys : particle_systems) {
                    if (!sys.runtime) continue;
                    for (const auto& gd : sys.runtime->gridDomainStates()) {
                        if (gd.type != RayTrophiSim::SimulationDomainType::Fluid || !gd.valid) continue;
                        build_for(gd.particles, gd.grid, kDomainLevelSet);
                    }
                }
            });

            // sampler(): query the fluid at a world point. signed_distance is the
            // point's height relative to the free SURFACE in its column (<0 below
            // it); velocity drives drag. Reads the fields prepare() built.
            rigid_body_system->setFluidSampler(
                [this](const Vec3& wp,
                       RayTrophiSim::RigidBodySystem::FluidSample& out) -> bool {
                    for (const auto& f : fluid_coupling_fields_) {
                        if (!f.grid) continue;
                        const auto& g = *f.grid;
                        const size_t cells = static_cast<size_t>(g.nx) * g.ny * g.nz;
                        if (f.sdf.size() != cells) continue;  // stale size guard
                        Vec3 lo, hi;
                        g.getWorldBounds(lo, hi);
                        // Only the XZ footprint must be in-domain — a body falling
                        // from above the domain top is "not submerged" (handled by
                        // the surface height), not "outside the fluid".
                        if (wp.x < lo.x || wp.z < lo.z || wp.x > hi.x || wp.z > hi.z) continue;

                        // Submersion vs the free surface, NOT vs the particle field
                        // at this interior point (a sunk body displaces particles
                        // out of its own cells, so that test gave zero buoyancy).
                        const float surf = sampleFluidColumnSurface(f, wp);
                        if (surf <= -1.0e30f) continue;  // no fluid in this column area
                        out.signed_distance = wp.y - surf;  // <0 => below surface

                        // Velocity for drag = the AMBIENT (wave) flow in the water
                        // AROUND the body, not the grid velocity AT the sample point.
                        // The body's own cells are solid (cavity) and carry the
                        // body's stamped velocity, so sampling there gives ~zero
                        // relative velocity and waves can't drag it. The neighbourhood
                        // average picks up the surrounding wave flow; if no fluid is
                        // near, fall back to still water (0 → drag just damps).
                        Vec3 amb;
                        out.velocity = sampleFluidAmbientVelocity(f, wp, amb)
                                           ? amb : Vec3(0.0f, 0.0f, 0.0f);
                        out.valid = true;
                        return true;
                    }
                    return false;
                });

            simulation_world.addSystem(rigid_body_system);
        }
        rigid_body_system->setBodies(&rigid_bodies);
    }

    // Mark a scene object as a rigid body (dynamic) or static collider. Returns a
    // pointer to the descriptor (existing one updated if the object already has it).
    RayTrophiSim::RigidBodyObject* addRigidBodyForObject(const std::string& node_name, bool dynamic = true) {
        if (node_name.empty()) return nullptr;
        ensureRigidBodySystem();
        for (auto& rb : rigid_bodies) {
            if (rb.source_name == node_name) {
                rb.dynamic = dynamic;
                rb.motion_type = dynamic ? RayTrophiSim::RigidBodyMotionType::Dynamic
                                         : RayTrophiSim::RigidBodyMotionType::Static;
                rb.enabled = true;
                syncRigidBodyProxyColliders();
                captureRigidBodyRestPose(rb);
                if (rigid_body_system) {
                    rigid_body_system->resetRuntime(true);
                    rigid_body_system->setBodies(&rigid_bodies);
                }
                invalidateRigidBodySimulationCache();
                return &rb;
            }
        }
        RayTrophiSim::RigidBodyObject rb;
        rb.source_name = node_name;
        rb.name = node_name + (dynamic ? " (Rigid)" : " (Static)");
        rb.dynamic = dynamic;
        rb.motion_type = dynamic ? RayTrophiSim::RigidBodyMotionType::Dynamic
                                 : RayTrophiSim::RigidBodyMotionType::Static;
        rigid_bodies.push_back(rb);
        rigid_body_system->setBodies(&rigid_bodies);  // vector may have reallocated
        syncRigidBodyProxyColliders();
        captureRigidBodyRestPose(rigid_bodies.back());
        rigid_body_system->resetRuntime(true);
        rigid_body_system->setBodies(&rigid_bodies);
        invalidateRigidBodySimulationCache();
        return &rigid_bodies.back();
    }

    // ── Destruction: fracture-group break (Faz 2) ───────────────────────────
    // True when any breakable body exists. Breakable scenes bypass the rigid
    // frame cache (deterministic re-sim) so the shatter replays correctly on a
    // loop / rewind instead of fighting cached pre-break (static) poses.
    bool hasBreakableBodies() const {
        for (const auto& rb : rigid_bodies) if (rb.getBreakable()) return true;
        return false;
    }

    // World AABBs for a whole set of nodes at once, keyed by the group each node
    // maps to. ONE pass over world.objects regardless of how many groups are
    // asked for, because the caller that needs every group is the blast path,
    // which fires on the busiest frame in the scene.
    void accumulateFractureGroupBounds(
        const std::unordered_map<std::string, std::string>& node_to_group,
        std::unordered_map<std::string, RayTrophiSim::FractureGroupBounds>& out) const;

    // World AABB of a fracture group's shards, from shard GEOMETRY (never from
    // shard centres — see FractureGroupBounds). False when the group has no
    // surviving geometry.
    bool fractureGroupBounds(const std::string& group,
                             Vec3& out_min, Vec3& out_max) const;

    // World-space centre of a scene node's geometry (mesh AABB centre).
    Vec3 nodeWorldCenter(const std::string& node) const {
        Vec3 mn(1e30f, 1e30f, 1e30f), mx(-1e30f, -1e30f, -1e30f);
        bool any = false;
        for (const auto& o : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(o);
            if (!tri || tri->getNodeName() != node) continue;
            for (int i = 0; i < 3; ++i) {
                const Vec3 p = tri->getVertexPosition(i);
                mn = Vec3(std::min(mn.x, p.x), std::min(mn.y, p.y), std::min(mn.z, p.z));
                mx = Vec3(std::max(mx.x, p.x), std::max(mx.y, p.y), std::max(mx.z, p.z));
                any = true;
            }
        }
        return any ? (mn + mx) * 0.5f : Vec3(0.0f, 0.0f, 0.0f);
    }

    // Register each shard node of a fractured mesh as a STATIC, breakable rigid
    // body (intact until an impact exceeds `threshold`). Shards share `group` so a
    // hit on any one shatters them all. Shape = Mesh → ConvexHull once dynamic.
    // `source_object` is the node the shards were cut FROM. It is what carries
    // the MSF field, so without it a cluster whose group name differs from the
    // object name can find no damage and never weakens. Empty = the group name
    // is the object name (the single-group case).
    void makeFractureGroupBreakable(const std::string& group,
                                    const std::vector<std::string>& shard_nodes,
                                    float break_velocity,
                                    bool integrity_weakening = true,
                                    float integrity_exponent = 1.5f,
                                    float minimum_threshold_scale = 0.15f,
                                    const std::string& source_object = "") {
        if (shard_nodes.empty()) return;
        ensureRigidBodySystem();
        fracture_group_source_[group] =
            source_object.empty() ? group : source_object;
        for (const auto& node : shard_nodes) {
            RayTrophiSim::RigidBodyObject* rb = addRigidBodyForObject(node, /*dynamic=*/false);
            if (!rb) continue;
            rb->setBreakable(true);
            rb->broken = false;
            rb->setFractureGroup(group);
            rb->setBreakVelocity(break_velocity);
            rb->setIntegrityWeakening(integrity_weakening);
            rb->setIntegrityExponent(std::max(integrity_exponent, 0.01f));
            rb->setMinimumThresholdScale(
                std::clamp(minimum_threshold_scale, 0.0f, 1.0f));
            rb->shape = RayTrophiSim::RigidBodyShape::Mesh;   // convex hull when broken
            rb->motion_type = RayTrophiSim::RigidBodyMotionType::Static;
            rb->dynamic = false;
            // ★ MASS FROM VOLUME, or every shard weighs the same 1 kg.
            //
            // `mass` defaults to 1.0 and `auto_mass_from_density` to false, so a
            // 5 cm chip and a 2 m slab came out identical. Nothing looks heavier
            // than anything else, big pieces drift like polystyrene and small
            // ones plough through them: the "weightless" reading is exactly this
            // default, not a solver problem. Shards are the one case where the
            // authored default cannot be right — their sizes span two orders of
            // magnitude BY CONSTRUCTION, and nobody is going to type a mass per
            // shard for forty of them.
            rb->auto_mass_from_density = true;
            // ★ And WEIGH IT NOW, rather than leaving it to body creation.
            //
            // The volume-derived mass is computed in ensureBodyCreated, which
            // only runs once the simulation starts. Until then rb.mass is the
            // authored 1.0 — so before anyone presses play, the panel, the
            // scripting API and the break threshold all see a 40-shard object
            // weighing exactly 40 kg. That reads identically to the "every shard
            // weighs one kilogram" bug, and a threshold derived from it is wrong
            // by whatever the real mass turns out to be.
            const float volume = nodeMeshVolume(node);
            if (volume > 0.0f) rb->mass = std::max(0.001f, rb->density * volume);
        }
        if (rigid_body_system) {
            rigid_body_system->setContactEventsEnabled(true);  // impacts drive the break
            rigid_body_system->resetRuntime(true);
            rigid_body_system->setBodies(&rigid_bodies);
        }
        invalidateRigidBodySimulationCache();
    }

    // Shatter a fracture group NOW: flip every still-intact shard to Dynamic and
    // give it a one-shot blast velocity (radial from `impact_point`, blended with
    // `impact_dir`). Re-creation happens on the next step (created=false).
    void breakFractureGroup(const std::string& group, const Vec3& impact_point,
                            const Vec3& impact_dir, float strength) {
        bool any = false;
        for (auto& rb : rigid_bodies) {
            if (!rb.getBreakable() || rb.broken || rb.getFractureGroup() != group) continue;
            rb.broken = true;
            rb.motion_type = RayTrophiSim::RigidBodyMotionType::Dynamic;
            rb.dynamic = true;
            rb.created = false;  // recreate as a dynamic convex hull next step
            const Vec3 c = nodeWorldCenter(rb.source_name);
            Vec3 radial = c - impact_point;
            const float r = radial.length();
            const Vec3 dir = (r > 1e-4f) ? radial * (1.0f / r) : Vec3(0.0f, 1.0f, 0.0f);
            const float falloff = 1.0f / (1.0f + r);  // nearer shards fly faster
            rb.pending_launch_velocity =
                dir * (strength * falloff) + impact_dir * (strength * 0.3f);
            rb.has_pending_launch = true;
            any = true;
        }
        if (any && rigid_body_system) rigid_body_system->setBodies(&rigid_bodies);
    }

    // Manual shatter (UI "Break Now"): explode the group radially from its own
    // centre. Takes effect on the next sim step (recreates shards dynamic).
    void breakFractureGroupNow(const std::string& group, float strength) {
        Vec3 sum(0.0f, 0.0f, 0.0f);
        int n = 0;
        for (auto& rb : rigid_bodies)
            if (rb.getBreakable() && !rb.broken && rb.getFractureGroup() == group) {
                sum += nodeWorldCenter(rb.source_name);
                ++n;
            }
        if (n == 0) return;
        breakFractureGroup(group, sum * (1.0f / static_cast<float>(n)),
                           Vec3(0.0f, 0.0f, 0.0f), strength);
    }

    // ── Fracture bookkeeping the PROJECT has to know about ───────────────────
    // group -> source object it was cut from. Without this on reload every
    // clustered group looks up its MSF field under its own name, finds nothing,
    // and reports pristine integrity forever — the exact silent failure the map
    // was introduced to kill, resurrected by a save/load round trip.
    const std::map<std::string, std::string>& fractureGroupSources() const {
        return fracture_group_source_;
    }
    void setFractureGroupSource(const std::string& group, const std::string& source) {
        if (!group.empty()) fracture_group_source_[group] = source;
    }

    // Nodes whose geometry is PARKED by an active fracture: alive, but pulled
    // out of world.objects so the shards render in their place.
    //
    // ★ The project saver walks each imported model's nodes and records any it
    // cannot find in world.objects as DELETED. A parked original is exactly that
    // — absent but not gone — so fracturing and saving marked the source mesh
    // deleted, and reopening dropped it for good. The shards do not survive
    // either (they belong to no model and no procedural record), so the object
    // simply vanished from the reopened project.
    std::set<std::string> fracture_parked_nodes;

    // ── How a saved fracture comes back ──────────────────────────────────────
    //
    // ★★ MEASURED, not assumed: with the default save settings
    // (`save_geometry = true`) a project writes a binary geometry sidecar
    // (`.rtp.bin`) and the shard meshes come back from it whole. So the primary
    // job of this record is NOT to re-cut — it is to restore the BOOKKEEPING,
    // which no amount of geometry brings back: which shards belong to which
    // object, and which structural cluster each one is in. Without that the
    // scene looks perfectly correct and yet the object does not know it is
    // fractured, so Break Now, un-fracture and every cluster query fail.
    //
    // ★ Re-cutting is the FALLBACK, for projects saved with geometry off (where
    // meshes are re-imported or procedurally regenerated instead). Never do both:
    // cutting an object whose shards were already restored produces a second set
    // of shards with the SAME names.
    //
    // ★★ The sites are what makes the fallback honest. Seed + pattern + count do
    // not determine them: the scatter rejects candidates outside the hull and
    // tops up from a second pass, and the thermal pattern draws from the MSF
    // damage field as it stood when the artist pressed the button — a state that
    // no longer exists after a reload. A re-derived cut would look plausible and
    // be different, and the rigid bodies saved alongside are bound to shard
    // NAMES, so a different cut silently rebinds them to other pieces.
    struct FractureRecipe {
        int      site_count = 15;
        uint32_t seed = 1337u;
        int      pattern = 0;        // 0 uniform, 1 impact-clustered, 2 thermal
        int      cluster_count = 4;
        bool     exact_surface = true;
        float    preview_gap = 0.02f;
        std::vector<Vec3> sites;     // world space, verbatim (re-cut fallback)
        // The shards this cut produced and their cluster index, parallel arrays.
        // This is what adoption rebuilds the UI bookkeeping from.
        std::vector<std::string> shard_nodes;
        std::vector<int>         shard_clusters;
    };
    std::map<std::string, FractureRecipe> fracture_recipes;

    RayTrophiSim::MaterialIntegritySummary fractureIntegritySummary(
        const std::string& group) const;
    // Summed authored mass [kg] of a fracture group's shards. The break
    // threshold is derived from it, so a heavy group resists a blast that
    // scatters a light one without either being tuned separately.
    float fractureGroupMass(const std::string& group) const;
    // Closed-mesh volume [m^3] of a node's geometry. Handles flat SoA and the
    // legacy facade form; returns 0 for a node with no geometry.
    float nodeMeshVolume(const std::string& node) const;
    bool applyFractureImpulse(const std::string& group,
                              const Vec3& point,
                              const Vec3& direction,
                              float impulse);

    // After a sim step: read the contact events and shatter any breakable group
    // whose shard took an impact above its threshold. No-op without breakables.
    void processFractureImpacts();
    // Turn this step's combustion into structural blast events. The producer the
    // chain never had: burning weakened the material and dropped its threshold,
    // but nothing outside a script ever delivered the load.
    void emitCombustionStructuralImpulses(float dt);
    void queueStructuralImpulse(RayTrophiSim::StructuralImpulseEvent event);
    // ★ MUST be pumped by every loop that steps the sim, next to
    // processFractureImpacts().
    //
    // It used to be called from ONE place: rt.physics.step. So a blast queued
    // inside the app was never consumed — it sat in the queue while the timeline
    // played, and the queue grew. Contact could break a group, overpressure
    // could not, and the scripted gate passed while the same scene did nothing
    // interactively. Two consumers for two kinds of load is one consumer too
    // few: they have to be pumped together or "it works in the test" means
    // nothing about the app.
    void processStructuralImpulseEvents();
    const RayTrophiSim::StructuralImpulseStats& structuralImpulseStats() const {
        return structural_impulse_stats_;
    }
    RayTrophiSim::AshDebrisSystem& ashDebrisSystem() { return ash_debris_system_; }
    const RayTrophiSim::AshDebrisSystem& ashDebrisSystem() const {
        return ash_debris_system_;
    }

    // Reset every breakable group back to intact (Static, unbroken) — called on a
    // rewind to frame 0 so a replay re-derives the shatter deterministically.
    void resetFractureToIntact() {
        fracture_summary_tick_ = 0;
        structural_impulse_events_.clear();
        // The ash reservoir is simulation state: mass waiting for a particle
        // slot. Replaying from frame 0 must not inherit debris the rewound run
        // produced, or the first budgeted event would spawn it a second time.
        ash_debris_system_.resetReservoir();
        bool any = false;
        for (auto& rb : rigid_bodies) {
            if (!rb.getBreakable()) continue;
            rb.broken = false;
            rb.motion_type = RayTrophiSim::RigidBodyMotionType::Static;
            rb.dynamic = false;
            rb.created = false;
            rb.has_pending_launch = false;
            rb.pending_launch_velocity = Vec3(0.0f, 0.0f, 0.0f);
            any = true;
        }
        if (any && rigid_body_system) rigid_body_system->setBodies(&rigid_bodies);
    }

    // Mark a scene object as a deformable body (soft body or cloth). Mirrors
    // addRigidBodyForObject but sets `kind`; foundation only — the soft solver is
    // not wired yet, so the body is authored/serialized but inert. Returns the
    // descriptor (existing one converted in place if the object already has one).
    RayTrophiSim::RigidBodyObject* addSoftBodyForObject(const std::string& node_name,
                                                        RayTrophiSim::BodyKind kind = RayTrophiSim::BodyKind::SoftBody) {
        if (node_name.empty()) return nullptr;
        ensureRigidBodySystem();
        const char* suffix = (kind == RayTrophiSim::BodyKind::Cloth) ? " (Cloth)" : " (Soft)";
        for (auto& rb : rigid_bodies) {
            if (rb.source_name == node_name) {
                rb.kind = kind;
                rb.dynamic = true;
                rb.motion_type = RayTrophiSim::RigidBodyMotionType::Dynamic;
                rb.enabled = true;
                syncRigidBodyProxyColliders();   // drops any stale rigid proxy
                captureRigidBodyRestPose(rb);
                if (rigid_body_system) {
                    rigid_body_system->resetRuntime(true);
                    rigid_body_system->setBodies(&rigid_bodies);
                }
                invalidateRigidBodySimulationCache();
                return &rb;
            }
        }
        RayTrophiSim::RigidBodyObject rb;
        rb.source_name = node_name;
        rb.name = node_name + suffix;
        rb.kind = kind;
        rb.dynamic = true;
        rb.motion_type = RayTrophiSim::RigidBodyMotionType::Dynamic;
        rigid_bodies.push_back(rb);
        rigid_body_system->setBodies(&rigid_bodies);  // vector may have reallocated
        syncRigidBodyProxyColliders();
        captureRigidBodyRestPose(rigid_bodies.back());
        rigid_body_system->resetRuntime(true);
        rigid_body_system->setBodies(&rigid_bodies);
        invalidateRigidBodySimulationCache();
        return &rigid_bodies.back();
    }

    bool removeRigidBodyForObject(const std::string& node_name) {
        // Determine the body's kind BEFORE resetRuntime so we can restore the
        // mesh through the correct cache (rigid vs soft/cloth). resetRuntime
        // routes by the body's current `kind`; if we let it run blindly it
        // would use the right path, but we also need to clean up our own caches
        // afterwards, and the body is about to be erased, so do it explicitly.
        RayTrophiSim::BodyKind removed_kind = RayTrophiSim::BodyKind::Rigid;
        for (const auto& rb : rigid_bodies) {
            if (rb.source_name == node_name) { removed_kind = rb.kind; break; }
        }
        if (rigid_body_system) {
            rigid_body_system->resetRuntime(true);
        }
        // Explicitly restore this node's mesh to rest using the correct cache
        // for its kind, then drop both caches so no stale deformation data
        // can leak back (e.g. if the same node is later re-added as a body).
        restoreBodyMeshToRest(node_name, removed_kind);
        removeRigidBodyProxyColliders(node_name);
        const size_t before = rigid_bodies.size();
        rigid_bodies.erase(
            std::remove_if(rigid_bodies.begin(), rigid_bodies.end(),
                [&](const RayTrophiSim::RigidBodyObject& rb) { return rb.source_name == node_name; }),
            rigid_bodies.end());
        if (rigid_body_system) {
            rigid_body_system->setBodies(&rigid_bodies);
        }
        if (rigid_bodies.size() != before) {
            syncRigidBodyProxyColliders();
            invalidateRigidBodySimulationCache();
        }
        return rigid_bodies.size() != before;
    }

    // Freeze a physics body at its CURRENT frame: commit the deformed/posed mesh as
    // the object's permanent geometry and remove the body operator. The sim already
    // baked the current frame into the source mesh's `original` verts, so "apply" is:
    // destroy just this Jolt body (others keep simulating), drop the descriptor + its
    // rest/topology caches, and DON'T restore the rest pose — the frozen shape stays
    // and, since save-rest-restore iterates rigid_bodies, the now-removed body's mesh
    // serializes as-is. Returns false if no body drives `node`.
    bool applyBodyAtCurrentFrame(std::string node) {  // BY VALUE: the caller passes a
        // rigid_bodies element's source_name; erasing it below would dangle a reference.
        auto it = std::find_if(rigid_bodies.begin(), rigid_bodies.end(),
            [&](const RayTrophiSim::RigidBodyObject& rb) { return rb.source_name == node; });
        if (it == rigid_bodies.end()) return false;

        // Remove only THIS body from the Jolt world (leaves others mid-sim intact).
        if (rigid_body_system) rigid_body_system->destroyBodyForNode(node);

        removeRigidBodyProxyColliders(node);
        rigid_bodies.erase(it);

        // Drop the body's rest/topology caches so nothing can later restore the old
        // rest or replay a stale deformed frame onto the now-frozen mesh.
        rigid_bake_cache_.erase(node);
        soft_weld_cache_.erase(node);
        melt_weld_cache_.erase(node);
        melt_displaced_.erase(node);
        melt_sdf_refresh_stamp_.erase(node);
        melt_applied_stamp_.erase(node);
        for (auto& kv : soft_frame_cache_) kv.second.erase(node);

        if (rigid_body_system) rigid_body_system->setBodies(&rigid_bodies);
        syncRigidBodyProxyColliders();
        // Cached sim frames reference the old body set; drop them so a later play
        // re-sims the remaining bodies cleanly (and never re-deforms the frozen one).
        invalidateRigidBodySimulationCache();

        // The frozen verts are already in the mesh; flag a one-shot geometry refresh
        // (backend BLAS) + a SceneUI mesh/bbox cache rebuild (the object's bounds
        // changed shape but its triangle count didn't, so SceneUI won't auto-rebuild).
        ++body_geom_version_;
        requestUiMeshCacheRebuild();
        Core::RenderStateManager::instance().markDirty(Core::DirtyScope::Geometry);
        return true;
    }

    struct FluidRenderBinding {
        int vdb_id = -1;
        std::shared_ptr<VDBVolume> volume;
        std::vector<float> density;
        // Render mode the bound shader preset matches; -1 = uninitialised.
        // Lets syncFluidRenderVolumes re-tune the preset (smoke vs water look)
        // when the user toggles the render mode without tearing the volume
        // down (volume + surfaceSDF both ride this binding).
        int last_render_mode = -1;
    };
    std::unordered_map<uint32_t, FluidRenderBinding> fluid_render_bindings;

    // =========================================================================
    // Particle Simulation (shared simulation world testbed)
    // =========================================================================
    // Viewport render blend look for a particle system's billboards.
    enum class ParticleBlendMode { Additive = 0, Alpha = 1 };

    // What geometry each alive particle is instanced as in the real RT render
    // paths. Built-in primitives are cheap, view-independent meshes generated
    // once; SceneMeshes picks from a weighted list of scene nodes (explosion
    // debris, scattered chunks). All routes flow through a transient
    // InstanceGroup that every RT backend already consumes.
    enum class ParticleRenderShape {
        Sphere = 0,   // emissive/diffuse round droplet — default, view-independent
        Cube = 1,
        Tetra = 2,
        Quad = 3,     // flat card (camera-facing handled later; world-aligned for now)
        SceneMeshes = 4  // weighted list of scene meshes (debris)
    };

    // One weighted entry of the SceneMeshes render source list.
    struct ParticleRenderMeshSource {
        std::string node_name;   // scene node to instance
        float weight = 1.0f;     // selection probability weight
    };

    // Per-system configuration for how particles appear in the real render.
    struct ParticleRenderSettings {
        bool render_in_raytrace = true;       // bridge into the RT instance channel
        ParticleRenderShape shape = ParticleRenderShape::Sphere;
        float size_multiplier = 1.0f;         // scales SoA per-particle size
        int sphere_subdivisions = 1;          // icosphere refinement for Sphere shape
        // Built-in primitive material look. Per-particle color variety comes from
        // sampling the base_color -> color_end gradient into `color_buckets`
        // materials; each particle picks one by a stable hash (spark look, no
        // shader change). Set color_end == base_color for a uniform color.
        bool emissive = true;                 // sparks glow; granular -> false
        // When true, the bucket gradient endpoints are pulled from the first
        // emitter's start_color/end_color so the appearance panel is the single
        // source of truth (Solid billboards + RT instances stay in sync without
        // the user having to edit two color pairs). Toggle off to author RT-only
        // colors that diverge from the billboard fade.
        bool inherit_color_from_emitter = true;
        Vec3 base_color = Vec3(1.0f, 0.6f, 0.2f);   // gradient start (orange)
        Vec3 color_end = Vec3(1.0f, 0.25f, 0.08f);  // gradient end (deep red)
        int color_buckets = 8;                // distinct colors sampled along the gradient
        // Over-life color: each particle's bucket follows its AGE (start->end as it
        // ages, like the Solid billboards) + emissive dims out. This needs the
        // material to change per frame, which the cheap TLAS refit can't do, so it
        // forces a full rebuild each motion frame — opt-in (heavier). Off = stable
        // per-particle color variety (cheap refit). Ignored for SceneMeshes.
        bool over_life_color = false;
        float emission_strength = 6.0f;       // used when emissive
        float roughness = 0.6f;               // used when not emissive
        // SceneMeshes source list (weighted). Only used when shape == SceneMeshes.
        std::vector<ParticleRenderMeshSource> mesh_sources;
    };

    struct ParticleSystemObject {
        uint32_t id = 0;
        std::string name = "Particle System";
        bool visible = true;
        bool enabled = true;
        ParticleBlendMode blend_mode = ParticleBlendMode::Additive;
        // How the particles are drawn in the real RT render paths (OptiX +
        // Vulkan). Serialized; the live instance group it drives is not.
        ParticleRenderSettings render;
        // Transient InstanceManager group id mirroring this system's alive
        // particles as instances (render bridge; -1 = none). Not serialized.
        int render_instance_group_id = -1;
        // Each system owns its own runtime solver. All systems are registered
        // with the SimulationWorld and simulate concurrently; the runtime is the
        // single source of truth for gravity/drag/collision/emitters/colliders.
        std::shared_ptr<RayTrophiSim::ParticleSimulationSystem> runtime;
        // Live VDB volume id per grid domain (render bridge; -1 = none). Parallel
        // to runtime->gridDomainStates(). Not serialized.
        std::vector<int> domain_vdb_ids;
        // Transient VDBVolume hittable per grid domain, bound to the live id so
        // the existing VDB render path (TLAS + volume pass) draws the gas.
        std::vector<std::shared_ptr<VDBVolume>> domain_volumes;
        // Whitewater Volume render mode: a SECOND live VDB per domain — foam
        // splatted to a white scattering density. Separate id/volume/buffer so
        // it composites independently of the liquid volume/surface.
        std::vector<int> domain_foam_vdb_ids;
        std::vector<std::shared_ptr<VDBVolume>> domain_foam_volumes;
        std::vector<std::vector<float>> domain_foam_density;
        // Transient InstanceManager group id per grid domain for the Particles
        // render mode (only used when the domain is type=Fluid AND
        // fluid_render_mode == Particles; -1 otherwise). Parallel-indexed to
        // gridDomainStates(); see SceneData::syncDomainFluidParticleInstances.
        std::vector<int>    domain_particle_render_group_ids;
        // Peak-seen alive particle count per domain. Pool only grows so cheap
        // TLAS refit stays valid across reseed-driven shrinks (matches the
        // ParticleSystemObject contract documented in ParticleRenderBridge.cpp).
        std::vector<size_t> domain_particle_pool_capacities;
        // Whitewater (foam/spray/bubble) render instances — same pooling
        // contract, a SEPARATE InstanceGroup with a white scattering material.
        // Independent of fluid_render_mode (foam shows over any liquid render).
        std::vector<int>    domain_foam_render_group_ids;
        std::vector<size_t> domain_foam_pool_capacities;
        // Per-domain narrow-band SDF buffer + stats, populated by the
        // SurfaceSDF render route. Transient; not serialized.
        std::vector<std::vector<float>>        domain_sdf_buffers;
        std::vector<RayTrophiSim::Fluid::LevelSetStats> domain_sdf_stats;
        // Per-domain material-coordinate (UVW) field, gathered alongside the
        // SDF. Interleaved xyz triples at SIM-grid resolution — NOT the refined
        // surface resolution, see buildMaterialCoordinateGrid for why.
        //
        // These are staging buffers only: the contents are SWAPPED into the
        // render volume that consumes them, so what stays here between rebuilds
        // is the previous frame's storage being recycled, not a live copy.
        // Nothing may read these expecting current data.
        std::vector<std::vector<float>>        domain_uvw_buffers;
        // Composition staging, same swap-on-rebuild contract as above.
        std::vector<std::vector<float>>        domain_composition_buffers;
        // SurfaceSDF rebuild gates. The first signature tracks particle +
        // surfacing params, so buildLevelSet is skipped when the generated SDF
        // would be identical. The second tracks the already-converted density
        // proxy upload, so NanoVDB conversion/upload is not repeated just
        // because syncSimulationRenderVolumes was called again.
        std::vector<uint64_t> domain_sdf_signatures;
        std::vector<uint64_t> domain_vdb_upload_signatures;
        // Last mode the bridge re-tuned the per-domain shader for. -1 = uninit;
        // any change vs current desc.fluid_render_mode triggers a one-shot
        // preset re-apply so the bridge never stomps on the user's live shader
        // edits across frames (the previous code re-applied every frame, so
        // UI slider edits were instantly overwritten).
        std::vector<int>    domain_last_fluid_render_mode;
        // Which shader instance the bridge last applied a render-mode preset to.
        // The mode alone is not enough: a domain whose shader is (re)created while
        // the mode is UNCHANGED — a fresh smoke preset from the lazy-init below, or a
        // project reload — would never be tuned, because the mode-change gate does not
        // fire. Identity, not equality: compared as a raw pointer, never dereferenced.
        std::vector<const void*> domain_last_tuned_shader;
    };

    std::vector<ParticleSystemObject> particle_systems;
    int active_particle_system_index = -1;  // UI selection focus only; does NOT gate simulation
    uint32_t next_particle_system_id = 1;

    // ── Grid-domain render bridge ────────────────────────────────────────────
    // Each grid domain with content is mirrored as a transient (never serialized)
    // VDBVolume hittable in vdb_volumes + world.objects, bound to a live NanoVDB
    // volume rebuilt each step from the runtime's FluidGrid density. This reuses
    // the existing VDB render path (TLAS instance + volume pass) on every backend
    // with no backend edits. The sim layer stays render-agnostic; this bridge
    // lives in the scene layer.
    uint64_t sim_render_frame_counter = 0;
    bool simulation_render_updated = false;  // a live volume's content changed this step
    bool force_simulation_render_sync_ = false;
    bool preserve_script_simulation_preview_ = false;
    // Authoring gate for Flow Source keyframes. While enabled, timeline scrub
    // moves the playhead without restoring/resimulating expensive gas state;
    // the UI keeps staged slider values until the user commits a diamond key.
    bool simulation_key_authoring_mode_ = false;

    // ── Timeline bake / scrub cache (memory) ─────────────────────────────────
    // sim_timeline_frame_ < 0 means "free-run" (interactive realtime preview, the
    // default). Playing the timeline switches to a deterministic bake from frame
    // 0: each frame's grid state is cached; scrubbing restores from the cache (or
    // resimulates the gap). "Reset Simulation" returns to free-run.
    std::map<int, std::vector<std::vector<RayTrophiSim::SimulationGridDomainState>>> sim_frame_cache_;
    // Rigid bodies are frame-cached in LOCKSTEP with sim_frame_cache_ (captured in
    // captureSimFrame, replayed in restoreRigidFrame). This is what keeps the rigid
    // motion identical on replay: the bake is the only pass where rigid (order 50)
    // and fluid (order 100) step coupled together, so we record the rigid result
    // then play it back verbatim instead of re-simulating it against a frozen
    // fluid frame (which diverges from the cached fluid). Cleared with the fluid
    // cache in clearSimFrameCache().
    std::map<int, std::vector<RayTrophiSim::RigidBodyFrameState>> rigid_frame_cache_;
    // Material State Field (burn/heat surface damage) is frame-cached in the same
    // lockstep. It is per-OBJECT runtime state, not per-domain: Phase 4 settled
    // that an object outside every domain is still simulated, so MSF cannot be
    // folded into SimulationGridDomainState and rides alongside it instead.
    // Outer vector is per particle system, matching sim_frame_cache_.
    // Cleared with the fluid cache in clearSimFrameCache().
    std::map<int, std::vector<std::vector<RayTrophiSim::MaterialStateFieldSnapshot>>>
        msf_frame_cache_;
    // Soft/cloth bodies are frame-cached alongside the fluid+rigid caches: the
    // deformed UNIQUE world vertices per soft node per frame (captured post-step in
    // captureSimFrame, scattered back to the mesh on replay in restoreSimFrame). The
    // deformation lives in the mesh, not in a pose, so it must be recorded per frame
    // or a cached-frame replay would freeze the cloth. Cleared with the fluid cache.
    std::map<int, std::map<std::string, std::vector<Vec3>>> soft_frame_cache_;

    // When playback steps PAST the cached range, soft bodies are rebuilt from rest
    // (see RigidBodySystem::restoreFrameState). These tell the soft-resume provider
    // which cached frame to teleport a freshly created soft body onto so it CONTINUES
    // instead of restarting from rest. soft_resume_frame_ < 1 disables the resume
    // (first bake / nothing to resume). Set just before a forward resim, reset after.
    int   soft_resume_frame_ = -1;
    float soft_resume_dt_    = 1.0f / 24.0f;
    // Discrete particles are frame-cached alongside the fluid+rigid+soft caches:
    // the full per-system SoA + alive count, captured post-step in captureSimFrame
    // and restored in restoreSimFrame. WHY: sim_frame_cache_ holds only grid-domain
    // states, so a cached-frame replay (loop-back / scrub within the baked range)
    // restored the grid but left the discrete particle SoA empty (clear()ed on the
    // rewind) — frames up to the previously-played head showed NO particles until
    // the sim re-simulated PAST the cache (the reported "empty until played frame"
    // bug). Runtime emitter/flow counters are part of the same snapshot: without
    // them, crossing from cached playback into the first uncached frame re-fired
    // one-shot bursts as though the simulation had returned to frame zero.
    struct ParticleFrameSnapshot {
        RayTrophiSim::ParticleSoABuffers buffers;
        std::size_t alive_count = 0;
        RayTrophiSim::ParticleSimulationRuntimeState runtime;
    };
    std::map<int, std::vector<ParticleFrameSnapshot>> particle_frame_cache_;
    int sim_timeline_frame_ = -1;
    int rigid_timeline_frame_ = -1;
    static constexpr int kMaxCachedSimFrames = 600;
    // Config signature for automatic memory-cache invalidation: when the sim
    // SETUP changes (add/remove of any sim element, rigid-body param edits, …)
    // the bake cache is dropped automatically instead of relying on manual reset.
    // Live sim state (per-step positions) is deliberately excluded so the
    // signature is stable while a sim is running.
    uint64_t last_sim_config_sig_ = 0;
    // Sub-signature of only the fluid-bake inputs (grid/emitter/collider config +
    // fluid-coupled rigid bodies). When the global signature changes but THIS one
    // doesn't, the change was a non-coupling rigid edit/move: the cheap rigid
    // re-sim runs but the expensive fluid cache is preserved.
    uint64_t last_fluid_coupling_sig_ = 0;
    // Frame rate the current bake was produced with. Every system steps once per
    // timeline frame at fixed_dt = 1/fps, so this IS a physics input — but it
    // arrives as an argument rather than as scene state, so it cannot live in
    // computeSimConfigSignature(). 0 = nothing baked yet, so the first tick
    // never counts as a change.
    float last_sim_bake_fps_ = 0.0f;
    // Last g_scene_geometry_generation value consumed by refreshRigidRestPosesOnUserEdit;
    // lets the idle user-edit detector skip work when no geometry edit happened.
    uint64_t last_user_edit_gen_ = 0;

    // ── On-disk bake cache (render-only point cache; see SimCache.h) ──────────
    // When a project is loaded with a valid <project>.simcache/ folder, the
    // baked sim is streamed from disk instead of re-simulated: restoreSimFrame
    // falls back to SimCache::readSystemFrame when the in-RAM cache misses. Set
    // by the loader (setSimDiskCache) after validating per-system config hashes.
    std::string sim_cache_dir_;
    bool        sim_cache_valid_ = false;
    std::unordered_set<uint32_t> sim_cache_valid_system_ids_;
    int         sim_cache_start_frame_ = 0;
    int         sim_cache_end_frame_ = 0;
    // Set when a fluid-affecting edit rewinds the sim to frame 0; the UI consumes
    // it (consumeSimRewindRequest) to move the timeline playhead back to start.
    bool        sim_rewind_request_ = false;
    // Short settle window after playback Start/End edits. Range metadata must
    // never invalidate simulation state; the extra ticks cover ImGui commit and
    // timeline-to-flow-key synchronization occurring on adjacent UI frames.
    uint8_t     timeline_range_edit_grace_ = 0;

    // ── Cooperative (frame-driven) disk bake state machine ───────────────────
    // A disk bake re-simulates the whole timeline range; one blocking loop would
    // freeze the UI for the entire bake. Instead it runs as a state machine
    // advanced a few frames per UI tick (tickSimulationDiskBake, time-budgeted)
    // so the progress bar + Cancel stay live and the app never freezes for long.
    // Everything runs on the main thread → no GPU/Vulkan/CUDA cross-thread hazard.
    bool        sim_bake_active_ = false;
    bool        sim_bake_cancel_ = false;
    bool        sim_bake_ok_ = true;
    std::string sim_bake_dir_;
    int         sim_bake_start_ = 0;
    int         sim_bake_end_ = 0;
    int         sim_bake_cur_ = 0;          // last frame stepped/written
    float       sim_bake_fps_ = 24.0f;
    float       sim_bake_dt_ = 1.0f / 24.0f;
    std::vector<std::pair<uint32_t, uint64_t>> sim_bake_hashes_;

    void syncSimulationRenderVolumes() {
        // The bridge does CUDA work (registerOrUpdateLiveVolume -> uploadToGPU) and
        // mutates world.objects. Doing either while a backend is tearing down /
        // rebuilding GPU state poisons the CUDA context (hangs / error 700) — the
        // same hazard the viewport denoiser guards against. Skip; resume next frame.
        const bool sync_blocked =
            g_optix_rebuild_in_progress.load() || g_viewport_rebuild_in_progress.load();
        // ★ A silent early-out is indistinguishable from "the loop ran and found
        // nothing", and every gate below lives INSIDE the loop — so a skip here
        // reads in the capture as though the producer agreed there was nothing to
        // do. Say it out loud; this is the switch-into-Vulkan-RT window.
        SCENE_LOG_ON_CHANGE("simrendersync.blocked", sync_blocked ? 1 : 0,
            std::string("[VolumeGate -1] syncSimulationRenderVolumes ") +
            (sync_blocked
                ? std::string("SKIPPED (optix_rebuild=") +
                  (g_optix_rebuild_in_progress.load() ? "1" : "0") +
                  " viewport_rebuild=" +
                  (g_viewport_rebuild_in_progress.load() ? "1" : "0") + ")"
                : std::string("running again")));
        if (sync_blocked) {
            return;
        }

        auto& mgr = VDBVolumeManager::getInstance();
        const int frame = static_cast<int>(sim_render_frame_counter++);
        const bool force_sync = force_simulation_render_sync_;
        simulation_render_updated = false;


        for (auto& system : particle_systems) {
            if (!system.runtime) {
                destroyDomainVolumes(system);
                continue;
            }

            const auto& states = system.runtime->gridDomainStates();
            auto& domains = system.runtime->gridDomains();  // per-domain shader lives here
            // Drop volumes for domains that no longer exist.
            //
            // ★★★ ASK THE DESCRIPTORS, NOT THE RUNTIME STATE.
            //
            // `states` is transient: resetGridDomainStates() clears and re-resizes
            // it, and synchronizeGridDomains() rebuilds it, so between a sim reset
            // and the next sync it can be shorter than — or empty relative to —
            // the domain DESCRIPTORS while every one of those domains still
            // exists. Treating it as the "does this domain exist" authority
            // unloaded every live volume for that one frame. The next frame
            // re-registered them, and registerOrUpdateLiveVolume then MISSES on
            // findVolumeIndex and hands out a BRAND NEW id
            // (VDBVolumeManager.cpp:488-497).
            //
            // Measured consequence: volume slot keys marching 42 → 43 → 44 → 46 →
            // 48 across consecutive publishes, a continuous
            // "[VolumeSSBO] volume slot ORDER changed" storm, and an identity
            // cache that can never hit — while the TLAS customIndex stayed baked
            // against the old order. That is volumes reading each other's slots,
            // i.e. the black band at the domain edge and its cost blow-up.
            //
            // A genuine deletion does NOT rely on this loop: removeSimulationGridDomain
            // erases the descriptor, the state and the volume together at its own
            // call site. So requiring BOTH lists to agree the domain is gone costs
            // nothing and removes the transient window entirely.
            const std::size_t live_domain_count =
                (std::max)(states.size(), domains.size());
            for (std::size_t d = live_domain_count; d < system.domain_vdb_ids.size(); ++d) {
                removeDomainVolume(system, d);
            }
            system.domain_vdb_ids.resize(live_domain_count, -1);
            system.domain_volumes.resize(live_domain_count);
            system.domain_sdf_buffers.resize(live_domain_count);
            system.domain_uvw_buffers.resize(live_domain_count);
            system.domain_composition_buffers.resize(live_domain_count);
            system.domain_sdf_stats.resize(live_domain_count);
            system.domain_sdf_signatures.resize(live_domain_count, 0);
            // ★ Same length as the arrays above. These are all per-domain
            // parallel arrays and removeSimulationGridDomain erases them at one
            // shared index, so letting some track the transient state count and
            // others the descriptor count reintroduces exactly the skew this
            // block just removed.
            system.domain_vdb_upload_signatures.resize(live_domain_count, 0);
            system.domain_last_fluid_render_mode.resize(live_domain_count, -1);
            system.domain_last_tuned_shader.resize(live_domain_count, nullptr);
            // Reused below to carry foam into the fluid-surface volume's
            // temperature channel (single-volume whitewater compositing).
            system.domain_foam_density.resize(live_domain_count);

            for (std::size_t d = 0; d < states.size(); ++d) {
                const auto& state = states[d];
                const bool domain_render_enabled =
                    d >= domains.size() || domains[d].render_to_nanovdb;
                const bool has_density =
                    (state.channels & static_cast<uint32_t>(RayTrophiSim::SimulationGridDomainChannelFlags::Density)) != 0u;
                // Fluid render mode gates the NanoVDB route. Particles mode is
                // handled entirely by ParticleRenderBridge — the volume route
                // must tear its contribution down or the two paths fight.
                const bool is_fluid_domain =
                    state.type == RayTrophiSim::SimulationDomainType::Fluid;
                RayTrophiSim::Fluid::FluidRenderMode fluid_mode =
                    (is_fluid_domain && d < domains.size())
                        ? domains[d].fluid_render_mode
                        : RayTrophiSim::Fluid::FluidRenderMode::Volume;
                // ★★ `Volume` is a DEAD configuration for a liquid domain, and it
                // is the struct default (ParticleSimulation.h), so any domain
                // created without an explicit render mode — a scripted one, for
                // instance — lands in it.
                //
                // In that state fluid_surface_route is false, so no SDF is ever
                // built, and `renderable` falls through to active_density_cells>0
                // which a fluid domain never splats. The volume object is then
                // never created, no gate downstream ever sees it, and the liquid
                // is simply absent with nothing reporting a failure.
                //
                // The UI already treats Volume as invalid (its combo offers only
                // Particles / SurfaceSDF) and used to REPAIR it by writing
                // SurfaceSDF while drawing the panel — which is why the surface
                // appeared only after opening the fluid panel or round-tripping
                // the render mode. Panel visibility must not be what fixes scene
                // data: normalise it here, where the mode is consumed.
                if (is_fluid_domain && d < domains.size() &&
                    fluid_mode == RayTrophiSim::Fluid::FluidRenderMode::Volume) {
                    fluid_mode = RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
                    domains[d].fluid_render_mode = fluid_mode;
                    SCENE_LOG_ON_CHANGE(
                        "domainmode." + system.name + ".D" + std::to_string(d), 1,
                        std::string("[VolumeGate 0a] domain '") + system.name + " D" +
                        std::to_string(d) + "' had the invalid liquid render mode "
                        "'Volume'; normalised to SurfaceSDF.");
                }
                bool has_sdf_override = false;
                if (is_fluid_domain && d < domains.size()) {
                    for (const auto& b : domains[d].fluid_substance_materials)
                        has_sdf_override |= b.representation ==
                            RayTrophiSim::Fluid::SubstanceRepresentation::SurfaceSDF;
                }
                const bool fluid_surface_route = is_fluid_domain &&
                    (fluid_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF || has_sdf_override);
                const bool fluid_skip_volume = is_fluid_domain && !fluid_surface_route;
                // Volume whitewater: foam rides THIS surface volume's temperature
                // channel (single volume → no coincident-volume drop). Only on the
                // surface route (that is the volume it rides) and only when the foam
                // panel's render mode is Volume.
                const bool volume_foam_active =
                    fluid_surface_route && d < domains.size() &&
                    domains[d].fluid_foam_params.enabled &&
                    domains[d].fluid_foam_params.render_mode ==
                        RayTrophiSim::Fluid::FoamRenderMode::Volume;

                const bool renderable =
                    domain_render_enabled && system.visible && state.valid &&
                    state.grid.nx > 0 && !fluid_skip_volume &&
                    // Volume mode needs density splatted; SurfaceSDF rebuilds
                    // its own density-proxy from particles, so it only needs
                    // particles to be present.
                    (fluid_surface_route
                        ? !state.particles.empty()
                        : (has_density && state.active_density_cells > 0));

                // ★ GATE 0 — the domain producer. This is the decision that
                // determines whether the domain volume exists at all, upstream
                // of every other gate; if it says no, gates 3/4 never even see
                // the volume and stay silent, which reads as "nothing happened".
                SCENE_LOG_ON_CHANGE(
                    "domainvol." + system.name + ".D" + std::to_string(d),
                    (renderable ? 1 : 0) + (fluid_surface_route ? 2 : 0) +
                        (fluid_skip_volume ? 4 : 0) + (is_fluid_domain ? 8 : 0),
                    std::string("[VolumeGate 0] domain '") + system.name + " D" +
                    std::to_string(d) + "' " + (renderable ? "RENDERABLE" : "NOT renderable") +
                    " | fluid=" + (is_fluid_domain ? "1" : "0") +
                    " route=" + (fluid_skip_volume ? "Particles"
                                 : (fluid_surface_route ? "SurfaceSDF" : "Volume")) +
                    " render_enabled=" + (domain_render_enabled ? "1" : "0") +
                    " sys_visible=" + (system.visible ? "1" : "0") +
                    " valid=" + (state.valid ? "1" : "0") +
                    " has_density=" + (has_density ? "1" : "0") +
                    " res=" + std::to_string(state.grid.nx) + "x" +
                        std::to_string(state.grid.ny) + "x" + std::to_string(state.grid.nz) +
                    // ★ Geometry of the domain itself. The black-band repro is
                    // decided at domain CREATION (a solid inside it at that
                    // moment triggers it; deleting the solid afterwards does
                    // not cure it), so the first thing to compare between a
                    // triggering and a clean run is whether the domain box came
                    // out the same size and in the same place at all.
                    " voxel=" + std::to_string(state.grid.voxel_size) +
                    " origin=(" + std::to_string(state.grid.origin.x) + "," +
                        std::to_string(state.grid.origin.y) + "," +
                        std::to_string(state.grid.origin.z) + ")" +
                    " particles=" + std::to_string(state.particles.size()) +
                    " active_cells=" + std::to_string(state.active_density_cells));

                if (!renderable) {
                    // A real representation switch is different from a
                    // temporarily empty SurfaceSDF frame. The latter keeps its
                    // stable slot; Particles mode must retire the SDF resource
                    // before the splat bridge publishes its geometry.
                    if (fluid_skip_volume) {
                        retireDomainSurfaceRepresentation(system, d);
                        continue;
                    }
                    // Keep the VDB registered and visible so the TLAS customIndex→SSBO
                    // slot mapping stays valid. Setting visible=false here causes a
                    // became_visible rebuild on the very next renderable frame, and
                    // unloading the VDB causes the same full-rebuild cycle every time
                    // the fluid transitions between empty/non-empty (e.g. on timeline
                    // loop or when density briefly hits 0 at simulation start).
                    // Keep the last-uploaded buffers registered, but mark their
                    // content inactive for this empty/cache-miss frame. The slot
                    // reactivates naturally when a solved frame is published.
                    // ★A frame that is NOT published must invalidate the upload gate.
                    // The SurfaceSDF upload signature is derived from the particle
                    // POSITIONS, so a fluid that is parked (paused, or sitting on a
                    // restored cache frame) produces the identical signature forever.
                    // If a single frame skips publication — and the surface route
                    // uniquely can, because `renderable` demands !particles.empty()
                    // where the gas route only needs active_density_cells — then on
                    // the next frame upload_changed is FALSE, registerOrUpdateLiveVolume
                    // is never called again, the volume stays out of the volume packet
                    // and its SSBO slot stays is_active=0: a transparent surface, with
                    // the nested gas domain still rendering perfectly beside it.
                    // Recovery required a render-mode toggle (which clears these
                    // signatures via removeDomainVolume) — exactly the workaround users
                    // found. Clearing the gate here makes the next good frame republish.
                    if (d < system.domain_vdb_upload_signatures.size()) {
                        system.domain_vdb_upload_signatures[d] = 0;
                    }
                    if (system.domain_vdb_ids[d] >= 0) {
                        // This early-out happens before the normal dense-field
                        // publication below. Clear the occupied-content gate or a
                        // rewind to empty frame 0 retains `true` from the last
                        // non-empty frame and Vulkan marches the stale domain AABB
                        // as a black box. Buffer addresses and slot identity stay
                        // intact; only this empty frame is omitted from the packet.
                        mgr.setLiveDenseContentActive(system.domain_vdb_ids[d], false);
                        g_gas_volumes_dirty = true;
                    }
                    continue;
                }

                // Each domain owns its volume shader (created lazily, editable in
                // the domain panel, serialized with the domain).
                bool shader_created_for_missing_domain = false;
                if (d < domains.size() && !domains[d].shader) {
                    // ★ Follow the domain's INTENT. The smoke preset has no
                    // blackbody emission, so a fire domain that fell back to it
                    // rendered its temperature field as flat grey smoke — the
                    // fire was simulating correctly and simply could not be
                    // seen. Presets always set a shader explicitly; this path is
                    // what a UI- or script-created domain lands on.
                    domains[d].shader = domains[d].fire_enabled
                        ? VolumeShader::createFirePreset()
                        : VolumeShader::createSmokePreset();
                    shader_created_for_missing_domain = true;
                }
                std::shared_ptr<VolumeShader> domain_shader =
                    (d < domains.size()) ? domains[d].shader : nullptr;

                // Re-tune the per-domain shader ONLY when the fluid render mode
                // crosses a boundary (or on first sight). Otherwise the user's
                // live UI edits would be stomped by the preset every frame.
                if (is_fluid_domain && domain_shader && d < system.domain_last_fluid_render_mode.size()) {
                    const int cur_mode = static_cast<int>(fluid_mode);
                    const bool had_previous_render_mode =
                        system.domain_last_fluid_render_mode[d] != -1;
                    const bool mode_changed =
                        had_previous_render_mode &&
                        cur_mode != system.domain_last_fluid_render_mode[d];
                    // ★A shader we have never tuned must be tuned even when the mode did
                    // NOT change. The lazy init a few lines above hands a fresh SMOKE
                    // preset to any domain whose shader is missing, and a smoke preset
                    // carries quality.max_steps ~16 — far too few for an iso walk that has
                    // to cross the whole domain. The walk then dies inside the medium and
                    // the liquid renders as a solid BLACK body with a correctly-placed
                    // surface: exactly the "black band where the fluid is" report.
                    // The mode gate alone missed this because the mode was SurfaceSDF the
                    // whole time; only the shader object was new. Toggling the render mode
                    // by hand fixed it because that forced this same preset to re-apply —
                    // which is why the workaround worked and pointed here.
                    // Identity check, so this still fires at most once per shader instance
                    // and never stomps the user's live slider edits on later frames.
                    const bool shader_untuned =
                        d >= system.domain_last_tuned_shader.size() ||
                        system.domain_last_tuned_shader[d] !=
                            static_cast<const void*>(domain_shader.get());
                    // A shader restored from a project is already authored;
                    // its density/scattering values must survive the first
                    // post-load sync. Only auto-created shaders receive the
                    // mode defaults on first sight.
                    if (mode_changed ||
                        (shader_untuned && shader_created_for_missing_domain)) {
                        // Mid-flight mode change: the SurfaceSDF density-proxy
                        // layout is dramatically different (very thin sharp
                        // band, 0..1) from the splatted-particle density layout
                        // (broad soft 0..N), and the backends cache SBT/BLAS
                        // entries off the existing volume binding. Reusing the
                        // same vdb_id + scene volume across the crossover let
                        // the Vulkan/OptiX driver see a half-updated state
                        // (new density layout, old descriptor binding) and
                        // crashed without an exception. Tear the binding down
                        // here and let the next sync iteration build a fresh
                        // upload + new scene volume from scratch. -1 sentinel
                        // means "first sight", no prior binding to clear.
                        // Only a genuine mid-flight MODE change invalidates the device
                        // binding; re-tuning a newly created shader must not tear the
                        // volume down (that would restart the upload every reload).
                        if (mode_changed && system.domain_last_fluid_render_mode[d] != -1) {
                            removeDomainVolume(system, d);
                            if (d < system.domain_sdf_buffers.size()) {
                                system.domain_sdf_buffers[d].clear();
                                system.domain_sdf_buffers[d].shrink_to_fit();
                            }
                            if (d < system.domain_uvw_buffers.size()) {
                                system.domain_uvw_buffers[d].clear();
                                system.domain_uvw_buffers[d].shrink_to_fit();
                            }
                            if (d < system.domain_sdf_signatures.size()) {
                                system.domain_sdf_signatures[d] = 0;
                            }
                            if (d < system.domain_vdb_upload_signatures.size()) {
                                system.domain_vdb_upload_signatures[d] = 0;
                            }
                        }
                        switch (fluid_mode) {
                            case RayTrophiSim::Fluid::FluidRenderMode::Volume:
                                // Fluid splat density is in [0,1] (per-particle
                                // = 1/8 with default ppc=8, trilinear-spread
                                // across 8 cells = 0.125 max per particle).
                                // Gas presets assume density 1..10, so the
                                // fluid Volume mode needs a much higher
                                // multiplier to read at all — ~50 makes a
                                // packed cell fully opaque, partial cells
                                // tint as fog. Absorption is pumped + tinted
                                // so accumulated water reads blue.
                                domain_shader->name = "Liquid NanoVDB Preview";
                                domain_shader->density.multiplier = 50.0f;
                                domain_shader->density.cutoff_threshold = 0.01f;
                                domain_shader->scattering.color = Vec3(0.55f, 0.74f, 0.92f);
                                domain_shader->scattering.coefficient = 1.0f;
                                domain_shader->scattering.anisotropy = 0.0f;
                                domain_shader->absorption.color = Vec3(0.15f, 0.42f, 0.78f);
                                domain_shader->absorption.coefficient = 2.0f;
                                domain_shader->emission.mode = VolumeEmissionMode::None;
                                break;
                            case RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF:
                                // Refractive water surface. In isosurface mode
                                // the shader interprets these fields as:
                                //   scattering.color = mild surface cast (near
                                //     white so thin sheets stay clear),
                                //   absorption.color = per-channel ABSORPTION
                                //     (high red, low blue -> blue transmitted
                                //     with depth, the real reason water is blue),
                                //   absorption.coefficient = depth tint strength.
                                domain_shader->name = "Liquid Surface (SDF)";
                                domain_shader->density.multiplier = 60.0f;
                                domain_shader->density.cutoff_threshold = 0.05f;
                                domain_shader->scattering.color = Vec3(0.92f, 0.96f, 1.0f);
                                domain_shader->scattering.coefficient = 0.4f;
                                domain_shader->scattering.anisotropy = 0.0f;
                                domain_shader->absorption.color = Vec3(0.85f, 0.40f, 0.12f);
                                domain_shader->absorption.coefficient = 2.5f;
                                domain_shader->emission.mode = VolumeEmissionMode::None;
                                // Gas presets default max_steps ~16 — far too
                                // few for an iso walk that must cross the whole
                                // domain to reach the back surface. ~256 covers
                                // a 64-128 voxel domain at ~voxel*0.5 fineness.
                                domain_shader->quality.max_steps = 256;
                                domain_shader->quality.step_size = 0.05f;
                                break;
                            case RayTrophiSim::Fluid::FluidRenderMode::Particles:
                                // Volume is torn down anyway; leave shader as-is.
                                break;
                        }
                        system.domain_last_fluid_render_mode[d] = cur_mode;
                        if (d < system.domain_last_tuned_shader.size()) {
                            system.domain_last_tuned_shader[d] =
                                static_cast<const void*>(domain_shader.get());
                        }
                        g_gas_volumes_dirty = true;
                    }
                }

                // SurfaceSDF route rebuilds the density-proxy band each step
                // from a Zhu-Bridson level set. The proxy buffer lives alongside
                // the system's transient render state so the upload call below
                // can swap density_ptr without mutating the const sim state's
                // own density (which the splat pass owns).
                const float* density_ptr_override = nullptr;
                bool surface_sdf_changed = false;
                // Set only on frames the material-coordinate field was actually
                // regathered, so the handoff to the render volume below is a
                // swap on rebuild frames and nothing at all on the others.
                bool uvw_rebuilt = false;
                bool composition_rebuilt = false;
                auto hash_combine_local = [](uint64_t h, uint64_t v) {
                    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
                    return h;
                };
                auto quantize_local = [](float v) {
                    return static_cast<uint64_t>(static_cast<int64_t>(std::lround(v * 1000.0f)));
                };
                if (fluid_surface_route && d < system.domain_sdf_buffers.size()) {
                    auto& sdf_buf = system.domain_sdf_buffers[d];
                    auto& sdf_stats = system.domain_sdf_stats[d];
                    const auto& lsp = domains[d].fluid_level_set_params;
                    uint64_t sdf_sig = 1469598103934665603ull;
                    sdf_sig = hash_combine_local(sdf_sig, static_cast<uint64_t>(state.particles.size()));
                    sdf_sig = hash_combine_local(sdf_sig, static_cast<uint64_t>(state.grid.nx));
                    sdf_sig = hash_combine_local(sdf_sig, static_cast<uint64_t>(state.grid.ny));
                    sdf_sig = hash_combine_local(sdf_sig, static_cast<uint64_t>(state.grid.nz));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(state.grid.voxel_size));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(state.grid.origin.x));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(state.grid.origin.y));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(state.grid.origin.z));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(domains[d].fluid_surface_band_voxels));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(lsp.kernel_radius_voxels));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(lsp.particle_radius_voxels));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(lsp.surface_offset_voxels));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(lsp.narrow_band_voxels));
                    sdf_sig = hash_combine_local(sdf_sig, static_cast<uint64_t>(lsp.smoothing_iterations));
                    sdf_sig = hash_combine_local(sdf_sig, static_cast<uint64_t>(lsp.surface_resolution_multiplier));
                    sdf_sig = hash_combine_local(sdf_sig, lsp.anisotropy_enabled ? 1ull : 0ull);
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(lsp.anisotropy_radius_voxels));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(lsp.anisotropy_max_stretch));
                    sdf_sig = hash_combine_local(sdf_sig, static_cast<uint64_t>(lsp.anisotropy_neighbor_min));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(lsp.position_smoothing));
                    // ★★★ THE DOMAIN MATERIAL AND EVERY SUBSTANCE MATERIAL ID
                    // BELONG HERE, and their absence was a reported bug: the
                    // composition field carries a per-cell material INDEX, so
                    // assigning a material from the list changes what this
                    // buffer must contain — while the tags, positions and
                    // representations it was keyed on are all unchanged. On a
                    // PAUSED timeline nothing else moves either, so the
                    // signature matched, the rebuild was skipped, and the pick
                    // did not reach the picture until the user rewound and
                    // re-simulated. The domain-level material updates live (it
                    // is pushed straight onto the volume), so the same panel
                    // answered two ways depending on which row you clicked.
                    sdf_sig = hash_combine_local(sdf_sig,
                        static_cast<uint64_t>(
                            static_cast<int64_t>(domains[d].fluid_surface_material_id)));
                    for (const auto& b : domains[d].fluid_substance_materials) {
                        sdf_sig = hash_combine_local(sdf_sig,
                            RayTrophiSim::Fluid::substanceTag(b.substance));
                        sdf_sig = hash_combine_local(sdf_sig,
                            static_cast<uint64_t>(b.representation));
                        sdf_sig = hash_combine_local(sdf_sig,
                            static_cast<uint64_t>(static_cast<int64_t>(b.material_id)));
                        // Miscibility narrows the composition ramp in the
                        // producer, so it must invalidate the cached fields
                        // exactly like the material binding does.
                        sdf_sig = hash_combine_local(sdf_sig,
                            quantize_local(b.miscibility));
                        // ★ Substance VISCOSITY and PHASE are deliberately
                        // absent: they feed the solver, not the surface. Their
                        // effect reaches these fields only by moving particles,
                        // and the positions are already hashed below — adding
                        // them would rebuild the surface on parameters that
                        // cannot change it. A solid chunk is still drawn by
                        // whatever `representation` says, which IS hashed.
                    }
                    for (std::size_t particle_index = 0;
                         particle_index < state.particles.position.size(); ++particle_index) {
                        const auto& p = state.particles.position[particle_index];
                        sdf_sig = hash_combine_local(sdf_sig, quantize_local(p.x));
                        sdf_sig = hash_combine_local(sdf_sig, quantize_local(p.y));
                        sdf_sig = hash_combine_local(sdf_sig, quantize_local(p.z));
                        // Mass fraction changes the rendered level-set radius and
                        // density mask, so it must invalidate the cached SDF too.
                        if (particle_index < state.particles.mass_fraction.size()) {
                            sdf_sig = hash_combine_local(
                                sdf_sig,
                                quantize_local(state.particles.mass_fraction[particle_index]));
                        }
                    }

                    const bool needs_sdf_rebuild =
                        force_sync ||
                        d >= system.domain_sdf_signatures.size() ||
                        system.domain_sdf_signatures[d] != sdf_sig ||
                        sdf_buf.empty();
                    if (needs_sdf_rebuild) {
                        std::vector<uint32_t> excluded_splat_tags;
                        if (fluid_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles) {
                            for (uint32_t tag : state.particles.substance_tag) {
                                bool explicit_sdf = false;
                                for (const auto& b : domains[d].fluid_substance_materials) {
                                    if (RayTrophiSim::Fluid::substanceTag(b.substance) == tag &&
                                        b.representation == RayTrophiSim::Fluid::SubstanceRepresentation::SurfaceSDF) {
                                        explicit_sdf = true; break;
                                    }
                                }
                                if (!explicit_sdf && std::find(excluded_splat_tags.begin(),
                                        excluded_splat_tags.end(), tag) == excluded_splat_tags.end())
                                    excluded_splat_tags.push_back(tag);
                            }
                        }
                        for (const auto& b : domains[d].fluid_substance_materials) {
                            bool exclude = b.representation ==
                                RayTrophiSim::Fluid::SubstanceRepresentation::Splat;
                            if (fluid_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles)
                                exclude = false; // handled from live tags above
                            if (exclude) excluded_splat_tags.push_back(
                                RayTrophiSim::Fluid::substanceTag(b.substance));
                        }
                        RayTrophiSim::Fluid::buildLevelSet(
                            state.particles, state.grid,
                            lsp, sdf_buf, &sdf_stats, &excluded_splat_tags);
                        if (d < system.domain_sdf_signatures.size()) {
                            system.domain_sdf_signatures[d] = sdf_sig;
                        }
                        surface_sdf_changed = true;
                        // Material coordinates ride the SAME rebuild gate as the
                        // surface, deliberately. They are gathered from the same
                        // particles with the same kernel, so a coordinate field
                        // one rebuild older than the surface it anchors would
                        // slide the texture by exactly one step of motion —
                        // small, plausible-looking, and untraceable.
                        if (d < system.domain_uvw_buffers.size()) {
                            uvw_rebuilt =
                                RayTrophiSim::Fluid::buildMaterialCoordinateGrid(
                                    state.particles, state.grid, lsp,
                                    system.domain_uvw_buffers[d],
                                    &excluded_splat_tags);
                        }
                        // Composition rides the SAME rebuild gate, for the same
                        // reason the coordinate does: it is gathered from the
                        // same particles with the same kernel, and a mixture one
                        // rebuild older than the surface it describes would put
                        // yesterday's boundary on today's shape.
                        if (d < system.domain_composition_buffers.size()) {
                            RayTrophiSim::Fluid::SubstanceMaterialEntry entries[
                                RayTrophiSim::Fluid::kMaxFluidSubstanceMaterials];
                            std::size_t entry_count = 0;
                            for (const auto& b : domains[d].fluid_substance_materials) {
                                if (entry_count >= RayTrophiSim::Fluid::kMaxFluidSubstanceMaterials)
                                    break;
                                if (b.substance.empty()) continue;
                                entries[entry_count].tag =
                                    RayTrophiSim::Fluid::substanceTag(b.substance);
                                entries[entry_count].material_id = b.material_id;
                                entries[entry_count].miscibility = b.miscibility;
                                ++entry_count;
                            }
                            composition_rebuilt =
                                RayTrophiSim::Fluid::buildCompositionGrid(
                                    state.particles, state.grid, lsp,
                                    entries, entry_count,
                                    domains[d].fluid_surface_material_id,
                                    system.domain_composition_buffers[d],
                                    // ★ THE SAME list the level set was given.
                                    // All three fields describe ONE surface, so
                                    // they must agree about which particles it
                                    // is made of; a splat substance voting here
                                    // tints a surface it has no part in.
                                    &excluded_splat_tags);
                            // ★★★ The post-pass that used to live here — collapse
                            // every cell's pair to its dominant slot when the
                            // domain flag `fluid_blend_substance_materials` was
                            // off — IS GONE, and the flag with it. It set the
                            // weight to exactly 0 or 1 in every cell, which left
                            // the shader's trilinear filter nothing to filter:
                            // the boundary landed on cell FACES and rendered as
                            // axis-aligned cubes of colour. A control labelled
                            // "Dominant Cell Material" that delivers "voxelised
                            // material" is worse than no control, because the
                            // artefact reads as a mixing bug rather than as the
                            // setting doing what it says.
                            // Sharpness is now per-substance MISCIBILITY, applied
                            // inside buildCompositionGrid as a gain on the ramp:
                            // an immiscible front is sharp AND sub-cell smooth.
                        }
                    }
                    // SDF may be refined above the sim grid (surface_resolution_
                    // multiplier), so size the proxy loop from the buffer itself,
                    // not the sim grid cell count.
                    const std::size_t cells = sdf_buf.size();
                    if (cells > 0 && needs_sdf_rebuild) {
                        // Reuse the buffer in-place as a density proxy centred
                        // on the surface:
                        //   density = clamp(0.5 - 0.5 * phi / grad_width, 0, 1)
                        // phi=0 (surface) -> 0.5 (the shader iso threshold),
                        // phi=-grad_width (interior) -> 1.0, +grad_width (air)
                        // -> 0.0. A SYMMETRIC ramp over a few voxels is the key
                        // fix for "matte pooled water": the old `1 - phi/band`
                        // saturated the whole interior to 1.0, so the gradient
                        // (hence the surface normal) was zero everywhere except
                        // a razor-thin shell — deep/settled fluid lost its
                        // normal and shaded flat. The symmetric ramp keeps a
                        // smooth gradient across the full band so the
                        // finite-difference normal is valid for both thin
                        // flowing sheets AND thick accumulated pools.
                        const float grad_width =
                            std::max(1.0f, domains[d].fluid_surface_band_voxels)
                            * state.grid.voxel_size;
                        const float inv_w = 0.5f / grad_width;
                        for (std::size_t ci = 0; ci < cells; ++ci) {
                            float dval = 0.5f - sdf_buf[ci] * inv_w;
                            if (dval < 0.0f) dval = 0.0f;
                            if (dval > 1.0f) dval = 1.0f;
                            sdf_buf[ci] = dval;
                        }
                    }
                    if (cells > 0) {
                        density_ptr_override = sdf_buf.data();
                    }
                    // Shader tuning happens above on mode-change only — don't
                    // re-apply per frame, that would stomp UI edits.
                } else if (d < system.domain_sdf_buffers.size() &&
                           !system.domain_sdf_buffers[d].empty()) {
                    // Mode left SurfaceSDF: free the proxy buffer to keep the
                    // memory footprint honest with the current route.
                    system.domain_sdf_buffers[d].clear();
                    system.domain_sdf_buffers[d].shrink_to_fit();
                    if (d < system.domain_uvw_buffers.size()) {
                        system.domain_uvw_buffers[d].clear();
                        system.domain_uvw_buffers[d].shrink_to_fit();
                    }
                    if (d < system.domain_sdf_signatures.size()) {
                        system.domain_sdf_signatures[d] = 0;
                    }
                    if (d < system.domain_vdb_upload_signatures.size()) {
                        system.domain_vdb_upload_signatures[d] = 0;
                    }
                }
                const std::string volume_name =
                    system.name + " Domain " + std::to_string(d) +
                    (state.type == RayTrophiSim::SimulationDomainType::Fluid
                         ? " [Fluid NanoVDB]"
                         : " [Gas NanoVDB]");

                // Throttle the expensive (OpenVDB + NanoVDB + upload) rebuild.
                const long long cells =
                    static_cast<long long>(state.grid.nx) * state.grid.ny * state.grid.nz;
                int stride = 1;
                if (cells >= 160LL * 160 * 160) stride = 3;
                else if (cells >= 104LL * 104 * 104) stride = 2;

                const int prev_id = system.domain_vdb_ids[d];
                const auto dense_gpu_view =
                    system.runtime->gasGpuFieldView(d, simulation_world.compute());
                const bool use_live_dense_gpu =
                    !render_settings.use_optix &&
                    state.type == RayTrophiSim::SimulationDomainType::Gas &&
                    dense_gpu_view.valid();
                // Live Render Update OFF freezes the volume so the path tracer can
                // converge instead of resetting forever. Always do the first upload
                // (prev_id < 0) so a frozen domain still shows a static frame.
                // The driver only calls the bridge when the grid actually changed
                // (bake/scrub/free-run step), so upload on first sight + stride.
                uint64_t upload_sig = 0;
                if (density_ptr_override) {
                    const auto& up_stats = system.domain_sdf_stats[d];
                    upload_sig = 1469598103934665603ull;
                    upload_sig = hash_combine_local(upload_sig, system.domain_sdf_signatures[d]);
                    upload_sig = hash_combine_local(upload_sig, static_cast<uint64_t>(up_stats.eff_nx));
                    upload_sig = hash_combine_local(upload_sig, static_cast<uint64_t>(up_stats.eff_ny));
                    upload_sig = hash_combine_local(upload_sig, static_cast<uint64_t>(up_stats.eff_nz));
                    upload_sig = hash_combine_local(upload_sig, quantize_local(up_stats.eff_voxel));
                    // Volume foam rides the temperature channel and moves every frame,
                    // so make any foam-bearing frame re-upload — fold the live foam
                    // count + the frame so the temp grid tracks the whitewater motion.
                    if (volume_foam_active) {
                        upload_sig = hash_combine_local(upload_sig, static_cast<uint64_t>(state.foam.size()));
                        upload_sig = hash_combine_local(upload_sig, static_cast<uint64_t>(frame));
                        // Fold the deposit-per-particle + per-class weights so a Foam
                        // Density / Bubble Froth / Spray slider edit re-uploads the temp
                        // grid even on a non-forced (stride) frame.
                        upload_sig = hash_combine_local(upload_sig,
                            quantize_local(domains[d].fluid_foam_params.volume_density));
                        upload_sig = hash_combine_local(upload_sig,
                            quantize_local(domains[d].fluid_foam_params.volume_bubble_strength));
                        upload_sig = hash_combine_local(upload_sig,
                            quantize_local(domains[d].fluid_foam_params.volume_spray_strength));
                    }
                }
                const bool upload_changed =
                    density_ptr_override &&
                    (d >= system.domain_vdb_upload_signatures.size() ||
                     system.domain_vdb_upload_signatures[d] != upload_sig ||
                     surface_sdf_changed);
                const bool do_update =
                    (!use_live_dense_gpu || prev_id < 0) &&
                    (force_sync || (prev_id < 0) ||
                    (density_ptr_override
                        ? (upload_changed && ((frame % stride) == 0))
                        : ((frame % stride) == 0)));
                if (do_update) {
                    // Upload temperature too when the shader maps it to emission
                    // (blackbody / channel-driven fire). registerOrUpdateLiveVolume
                    // keeps temperature voxels only above ~300 (Kelvin gas
                    // heuristic), so scale our 0-based heat into a Kelvin-ish range.
                    const float* temp_ptr = nullptr;
                    std::vector<float> scaled_temp;
                    const bool wants_temp =
                        domain_shader &&
                        (domain_shader->emission.mode == VolumeEmissionMode::Blackbody ||
                         domain_shader->emission.mode == VolumeEmissionMode::ChannelDriven) &&
                        !state.grid.temperature.empty();
                    if (wants_temp) {
                        constexpr float kHeatToKelvin = 3000.0f;
                        scaled_temp.resize(state.grid.temperature.size());
                        for (std::size_t ci = 0; ci < scaled_temp.size(); ++ci) {
                            scaled_temp[ci] = state.grid.temperature[ci] * kHeatToKelvin;
                        }
                        temp_ptr = scaled_temp.data();
                    }

                    const float* density_ptr = density_ptr_override
                        ? density_ptr_override
                        : state.grid.density.data();
                    // The SurfaceSDF proxy may be on a refined grid; upload at its
                    // effective resolution. Same origin/extent, finer voxels. The
                    // sim-sized temperature array can't ride a refined upload, so
                    // drop it on the surface route (water surfaces don't emit).
                    const auto& up_stats = system.domain_sdf_stats[d];
                    const bool refined_upload =
                        density_ptr_override && up_stats.eff_nx > 0 &&
                        static_cast<std::size_t>(up_stats.eff_nx) *
                        static_cast<std::size_t>(up_stats.eff_ny) *
                        static_cast<std::size_t>(up_stats.eff_nz) ==
                            system.domain_sdf_buffers[d].size();
                    const int   up_nx    = refined_upload ? up_stats.eff_nx : state.grid.nx;
                    const int   up_ny    = refined_upload ? up_stats.eff_ny : state.grid.ny;
                    const int   up_nz    = refined_upload ? up_stats.eff_nz : state.grid.nz;
                    const float up_voxel = refined_upload ? up_stats.eff_voxel : state.grid.voxel_size;
                    const float* up_temp = density_ptr_override ? nullptr : temp_ptr;

                    // ── Volume whitewater: foam → THIS volume's temperature channel ──
                    // Trilinear-splat the foam particles into a density field at the
                    // SAME resolution/extent as the SDF density grid, pre-scaled by
                    // FOAM_TEMP_SCALE. volume_closesthit.rchit (source_type==4) marches
                    // it as a bright white single-scatter medium (÷FOAM_TEMP_SCALE).
                    // One volume carries BOTH water (density/SDF) and foam (temp), so
                    // there is no coincident-volume drop on any backend. copyFromDense
                    // culls temp voxels < 300 → a ~0.03 foam-density floor (faint
                    // foam dropped — acceptable).
                    auto& foam_density = system.domain_foam_density[d];
                    if (volume_foam_active && !state.foam.empty()) {
                        constexpr float kFoamTempScale = 10000.0f; // MUST match FOAM_TEMP_SCALE in volume_closesthit.rchit
                        const float dpp = std::max(0.01f,
                            domains[d].fluid_foam_params.volume_density);
                        RayTrophiSim::Fluid::splatFoamDensity(
                            state.foam, up_nx, up_ny, up_nz, up_voxel, state.grid.origin,
                            foam_density, dpp,
                            domains[d].fluid_foam_params.volume_bubble_strength,
                            domains[d].fluid_foam_params.volume_spray_strength);
                        for (float& v : foam_density) v *= kFoamTempScale;
                        up_temp = foam_density.data();
                    } else if (!foam_density.empty()) {
                        foam_density.clear();
                        foam_density.shrink_to_fit();
                    }
                    const int new_id = mgr.registerOrUpdateLiveVolume(
                        prev_id,
                        volume_name,
                        up_nx, up_ny, up_nz,
                        up_voxel,
                        density_ptr,
                        up_temp,
                        nullptr);
                    // ★registerOrUpdateLiveVolume returns -1 on ANY exception inside the
                    // OpenVDB→NanoVDB conversion. Writing that -1 back would ORPHAN the
                    // still-valid previous binding: the next call gets prev_id=-1 and
                    // allocates a brand new volume entry, leaking the old one, while this
                    // frame renders nothing. Keep the last good id and let the retry below
                    // update it in place.
                    if (new_id >= 0) {
                        system.domain_vdb_ids[d] = new_id;
                        if (density_ptr_override &&
                            d < system.domain_vdb_upload_signatures.size()) {
                            system.domain_vdb_upload_signatures[d] = upload_sig;
                        }
                    } else if (d < system.domain_vdb_upload_signatures.size()) {
                        // Failed publication — clear the gate so the next frame retries
                        // instead of believing this signature is already on the device.
                        system.domain_vdb_upload_signatures[d] = 0;
                    }
                    simulation_render_updated = true;
                    // The host/GPU NanoVDB grid changed: force the backend volume
                    // table re-sync. OptiX stores device pointers in that table,
                    // so a live grid reallocation must refresh it before launch.
                    g_gas_volumes_dirty = true;
                }
                const int id = system.domain_vdb_ids[d];
                if (id < 0) {
                    // No device binding yet: same rule as the !renderable early-out —
                    // never leave the upload gate claiming this frame was published.
                    if (d < system.domain_vdb_upload_signatures.size()) {
                        system.domain_vdb_upload_signatures[d] = 0;
                    }
                    continue;
                }
                if (use_live_dense_gpu) {
                    mgr.setLiveDenseGpuFields(
                        id,
                        dense_gpu_view.density_address,
                        dense_gpu_view.temperature_address,
                        dense_gpu_view.fuel_address,
                        dense_gpu_view.flame_address,
                        dense_gpu_view.resolution_x,
                        dense_gpu_view.resolution_y,
                        dense_gpu_view.resolution_z,
                        dense_gpu_view.origin.x,
                        dense_gpu_view.origin.y,
                        dense_gpu_view.origin.z,
                        dense_gpu_view.voxel_size,
                        dense_gpu_view.version,
                        state.active_density_cells > 0,
                        state.active_density_min,
                        state.active_density_max);
                    mgr.setLiveDenseMajorant(
                        id,
                        dense_gpu_view.majorant_address,
                        dense_gpu_view.majorant_dim_x,
                        dense_gpu_view.majorant_dim_y,
                        dense_gpu_view.majorant_dim_z,
                        dense_gpu_view.majorant_block,
                        dense_gpu_view.emissive_list_address,
                        dense_gpu_view.emissive_capacity);
                    simulation_render_updated = true;
                    g_gas_volumes_dirty = true;
                } else {
                    mgr.clearLiveDenseGpuFields(id);
                }

                const Vec3 world_min = state.grid.origin;
                const Vec3 world_max = state.grid.origin +
                    Vec3(static_cast<float>(state.grid.nx) * state.grid.voxel_size,
                         static_cast<float>(state.grid.ny) * state.grid.voxel_size,
                         static_cast<float>(state.grid.nz) * state.grid.voxel_size);

                bool created = false;
                bool became_visible = false;
                if (!system.domain_volumes[d]) {
                    auto vol = std::make_shared<VDBVolume>();
                    vol->transient = true;
                    vol->name = volume_name;
                    system.domain_volumes[d] = vol;
                    addVDBVolume(vol);
                    world.objects.push_back(vol);
                    created = true;
                } else if (!system.domain_volumes[d]->visible) {
                    became_visible = true;
                }

                auto& vol = system.domain_volumes[d];
                vol->name = volume_name;
                vol->visible = true;
                // Renderable on CPU again (Volume/SurfaceSDF route): clear any skip
                // left over from a previous Particles-mode frame so the CPU BVH
                // treats it as a real volume occluder once more.
                vol->cpu_render_skip = false;
                // Mark the volume's render route every frame — the SDF proxy
                // density is the same NanoVDB channel either way; the shader
                // picks "fog raymarch" vs "isosurface walk + refraction" based
                // on render_as_isosurface (mapped to source_type=4 downstream).
                // ★The Vulkan TLAS bakes this flag into the INSTANCE MASK at build
                // time (VulkanBackend.cpp: mask = render_as_isosurface ? 0x08 : 0x02),
                // while source_type is refreshed in the SSBO every frame. If the flag
                // flips without a TLAS rebuild the two disagree, and the gas closest-hit's
                // skipGasVolumes handoff — which masks out 0x02 — then masks out the
                // LIQUID as well: the gas hands the ray off and nothing is behind it, so
                // the overlap region renders BLACK.
                // This is why it depended on where the scene was authored: in Solid the
                // RT rebuild is deferred (`!interactive_viewport_active`), so the flag
                // could change with the mask left behind; authoring in Rendered rebuilt
                // in the same breath and looked fine.
                // A flag whose value lives in the acceleration structure must force the
                // rebuild itself rather than trusting whoever set it to remember.
                const bool iso_was = vol->render_as_isosurface;
                vol->render_as_isosurface = fluid_surface_route;
                if (iso_was != fluid_surface_route && !created) {
                    g_geometry_dirty = true;
                    g_vulkan_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    g_gas_volumes_dirty = true;
                }
                if (fluid_surface_route && d < domains.size()) {
                    vol->render_isosurface_ior = domains[d].fluid_surface_ior;
                    vol->render_isosurface_roughness = domains[d].fluid_surface_roughness;
                    vol->render_isosurface_foam = domains[d].fluid_surface_foam;
                    vol->render_isosurface_material_id =
                        domains[d].fluid_surface_material_id;
                    vol->render_isosurface_pore_amount = domains[d].fluid_surface_pore_amount;
                    vol->render_isosurface_pore_scale  = domains[d].fluid_surface_pore_scale;
                    vol->render_isosurface_pore_detail = domains[d].fluid_surface_pore_detail;
                    vol->render_isosurface_coord_space = domains[d].fluid_surface_coord_space;
                    // ── Material coordinates: hand the field over ─────────────
                    // SWAP, not copy: the staging buffer's whole purpose is to
                    // become the volume's, and the volume's previous storage
                    // going back the other way is capacity the next rebuild
                    // reuses. Only on frames it was regathered — otherwise the
                    // volume keeps the field it already has, which is still the
                    // one matching the surface currently uploaded.
                    if (uvw_rebuilt && d < system.domain_uvw_buffers.size()) {
                        vol->render_isosurface_uvw_residual.swap(system.domain_uvw_buffers[d]);
                        vol->render_isosurface_uvw_dim[0] = state.grid.nx;
                        vol->render_isosurface_uvw_dim[1] = state.grid.ny;
                        vol->render_isosurface_uvw_dim[2] = state.grid.nz;
                        // ★ Placement travels WITH the buffer, from the same
                        // grid, in the same statement. The consumer indexes in
                        // world space through these; taking them from anywhere
                        // else (the volume's render bounds, a cached copy) is
                        // how the field ends up stretched relative to the
                        // surface it describes.
                        vol->render_isosurface_uvw_origin[0] = state.grid.origin.x;
                        vol->render_isosurface_uvw_origin[1] = state.grid.origin.y;
                        vol->render_isosurface_uvw_origin[2] = state.grid.origin.z;
                        vol->render_isosurface_uvw_voxel = state.grid.voxel_size;
                        ++vol->render_isosurface_uvw_version;
                    } else if (!uvw_rebuilt && surface_sdf_changed) {
                        // The surface rebuilt but the coordinate gather refused
                        // (no uvw sidecar, or every particle off-grid). Drop the
                        // field rather than leaving the previous one attached to
                        // a surface it no longer describes — a STALE coordinate
                        // is worse than none, because none falls back to world
                        // anchoring and looks merely old-fashioned, while stale
                        // pins the texture to where the liquid used to be.
                        vol->render_isosurface_uvw_residual.clear();
                        vol->render_isosurface_uvw_dim[0] = 0;
                        vol->render_isosurface_uvw_dim[1] = 0;
                        vol->render_isosurface_uvw_dim[2] = 0;
                        vol->render_isosurface_uvw_voxel = 0.0f;
                        ++vol->render_isosurface_uvw_version;
                    }

                    // ── Composition: its OWN gate, deliberately ───────────────
                    // ★ Not folded into the coordinate's if/else above. The
                    // composition gather legitimately refuses on a frame the
                    // coordinate succeeded — a single-material domain has no
                    // mixture to describe — so sharing a branch would either
                    // publish nothing as if it were data, or drop a perfectly
                    // good coordinate because there happened to be no mixture.
                    if (d < system.domain_composition_buffers.size()) {
                        if (composition_rebuilt) {
                            vol->render_isosurface_composition.swap(
                                system.domain_composition_buffers[d]);
                            ++vol->render_isosurface_uvw_version;
                        } else if (surface_sdf_changed &&
                                   !vol->render_isosurface_composition.empty()) {
                            // Same stale-is-worse-than-absent rule as the
                            // coordinate: a mixture pinned to where the liquid
                            // used to be reads as a shading bug, while none at
                            // all falls back to the domain material.
                            vol->render_isosurface_composition.clear();
                            ++vol->render_isosurface_uvw_version;
                        }
                    }
                    // Volume whitewater look (temperature-channel particle foam):
                    // tint + extinction drive the white single-scatter medium the
                    // shader marches (_ext_reserved[3..6]). Only meaningful when the
                    // foam render mode is Volume.
                    if (volume_foam_active) {
                        const auto& fp = domains[d].fluid_foam_params;
                        vol->render_isosurface_foam_color   = fp.volume_color;
                        vol->render_isosurface_foam_opacity = fp.volume_opacity;
                    }
                }
                if (domain_shader) {
                    vol->setShader(domain_shader);  // pick up live shader edits
                }
                vol->bindLiveVolume(id, state.grid.voxel_size, world_min, world_max);

                if (created) {
                    // New hittable added to world.objects: rebuild GPU TLAS so it
                    // gets primary-ray hits and re-sync volume buffers.
                    // CPU BVH: the live volume IS CPU-sampleable — VDBVolume::hit +
                    // the ray_color VDB ray-march read it through sampleDensityCPU on
                    // the manager's host NanoVDB handle (exactly like a disk VDB, which
                    // is also dual-listed in world.objects + vdb_volumes). So flag the
                    // CPU BVH for a rebuild too; only on CREATE (a structural change),
                    // not on per-step density updates — those are picked up at shade
                    // time and need no BVH work as long as the domain bounds are fixed.
                    // The async builder (Main.cpp) snapshots world.objects only, so the
                    // volume is added exactly once. This is what lets gas/fluid Volume +
                    // SurfaceSDF render in the CPU reference backend (offline).
                    // became_visible is NOT treated as a structural change — the TLAS
                    // instance was already registered on 'created', so showing it again
                    // only requires an SSBO update (g_gas_volumes_dirty) to re-activate
                    // the slot. Triggering a full rebuild on became_visible caused a
                    // rebuild every time density went 0→non-zero (e.g. on timeline loop).
                    g_geometry_dirty = true;
                    g_vulkan_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    g_gas_volumes_dirty = true;
                    g_bvh_rebuild_pending = true;
                } else if (became_visible) {
                    g_gas_volumes_dirty = true;
                }


            }
        }

        syncFluidRenderVolumes(mgr, frame, force_sync);
        syncFluidFoamVolumes(mgr, frame, force_sync);

        force_simulation_render_sync_ = false;
    }

    // The old whitewater "Volume" mode used a SEPARATE foam NanoVDB volume that sat
    // coincident with the fluid surface volume and got dropped by the Vulkan
    // integrator (black cube). The Volume mode is back, but it now rides the fluid
    // SURFACE volume's TEMPERATURE channel instead (single volume → no coincidence;
    // see volume_foam_active in syncSimulationRenderVolumes). This routine therefore
    // only TEARS DOWN any leftover SEPARATE foam volumes (domain_foam_volumes) from
    // an older session/save; it does NOT touch the surface-volume temp-channel foam.
    // (domain_foam_density is a transient splat scratch refilled per frame, safe to
    // clear here.)
    void syncFluidFoamVolumes(VDBVolumeManager& mgr, int frame, bool force_sync) {
        (void)frame; (void)force_sync;
        for (auto& system : particle_systems) {
            bool removed_any = false;
            for (std::size_t d = 0; d < system.domain_foam_volumes.size(); ++d) {
                if (system.domain_foam_volumes[d]) {
                    auto vol = system.domain_foam_volumes[d];
                    removeVDBVolume(vol);
                    auto it = std::find(world.objects.begin(), world.objects.end(),
                                        std::static_pointer_cast<Hittable>(vol));
                    if (it != world.objects.end()) world.objects.erase(it);
                    system.domain_foam_volumes[d].reset();
                    removed_any = true;
                }
                if (d < system.domain_foam_vdb_ids.size() && system.domain_foam_vdb_ids[d] >= 0) {
                    mgr.unloadVDB(system.domain_foam_vdb_ids[d]);
                    system.domain_foam_vdb_ids[d] = -1;
                }
            }
            if (!system.domain_foam_density.empty()) {
                system.domain_foam_density.clear();
                system.domain_foam_density.shrink_to_fit();
            }
            if (removed_any) {
                g_geometry_dirty = true;
                g_vulkan_rebuild_pending = true;
                g_optix_rebuild_pending = true;
                g_gas_volumes_dirty = true;
            }
        }
    }

    // ── Discrete-particle render bridge (defined in ParticleRenderBridge.cpp) ──
    // Mirrors each visible system's alive SoA into a transient InstanceManager
    // group (one instance per particle), consumed by every RT backend. Driven
    // INDEPENDENTLY from the per-frame render loop (next to the billboard upload),
    // NOT from the timeline sim driver — so it just reflects the live SoA and does
    // not inherit the timeline bake/scrub/cache gating or touch its global state.
    // enable_rt_geometry=false suppresses the instanced geometry (e.g. Debug
    // display mode shows the overlay instead) without destroying the groups.
    void syncParticleRenderInstances(bool enable_rt_geometry = true);
    // Drop every transient particle render group (reload / clear).
    void releaseParticleRenderInstances();
    // Drop a single system's render group (per-system removal).
    void destroyParticleRenderGroup(ParticleSystemObject& system);

    // Mirror every FluidObject in Particles render mode as instanced spheres
    // through the same ParticleRenderBridge group/source mechanism. Called from
    // inside syncParticleRenderInstances so a single UI tick covers both kinds.
    void syncFluidParticleRenderInstances(bool enable_rt_geometry);
    void destroyFluidParticleRenderGroup(RayTrophiSim::Fluid::FluidObject& obj);
    void releaseFluidParticleRenderInstances();

    // Mirror SimulationGridDomain particles (type=Fluid, render_mode=Particles)
    // as instanced spheres. Lives next to the FluidObject loop so both kinds
    // of fluid particle systems flow through the same GPU pipeline.
    void syncDomainFluidParticleInstances(bool enable_rt_geometry);
    void releaseDomainFluidParticleInstances();

    // Mirror SimulationGridDomain whitewater (foam/spray/bubble) particles as
    // instanced spheres with a SEPARATE white scattering material. Independent
    // of the liquid render mode — runs whenever fluid_foam_params.enabled.
    void syncFluidFoamRenderInstances(bool enable_rt_geometry);
    void releaseDomainFluidFoamInstances();

    // CPU reference render bridge for discrete particles. The GPU backends iterate
    // InstanceManager transient groups directly, but the CPU BVH is built only from
    // world.objects (rebuildSceneObjects skips transient groups). This expands every
    // live transient particle/fluid/foam instance into a HittableInstance (one shared
    // child EmbreeBVH per primitive source, cached on the source) and appends them to
    // `out`. Callers append into a snapshot used solely for the CPU BVH build, so the
    // particles never reach world.objects (no double-render on GPU / no selection-list
    // churn). Builds the child BVH lazily and reuses it until the bridge clears sources.
    void appendParticleCPUHittables(std::vector<std::shared_ptr<Hittable>>& out);

    // (Re)build the particle-only `particle_bvh` synchronously from the current live
    // particle set. Cheap (particles only, no static geometry). Sets particle_bvh to
    // null when no particles are live. Called per-frame on particle motion (CPU path)
    // and after any full scene-BVH rebuild so the composite stays in sync.
    void rebuildParticleBVH(bool use_embree);

    // Free all live grid-domain volumes and their scene objects (reload / clear).
    void releaseSimulationRenderVolumes() {
        for (auto& system : particle_systems) {
            destroyDomainVolumes(system);
        }
        destroyAllFluidRenderVolumes();
        releaseParticleRenderInstances();
        releaseFluidParticleRenderInstances();
        releaseDomainFluidParticleInstances();
        releaseDomainFluidFoamInstances();
    }

    // ── Timeline simulation driver ───────────────────────────────────────────
    void clearSimFrameCache() {
        sim_frame_cache_.clear();
        rigid_frame_cache_.clear();  // rigid is cached in lockstep; never outlive the fluid cache
        soft_frame_cache_.clear();   // soft deformation cache, same lockstep
        particle_frame_cache_.clear(); // discrete particle SoA, same lockstep
        msf_frame_cache_.clear();    // burn/heat surface damage, same lockstep
    }

    // Remove damage history without throwing away costly gas/fluid/rigid frame
    // caches. An empty object key means every MSF snapshot. Disk snapshots cannot
    // be edited safely in place, so detach the bake binding; otherwise a later RAM
    // cache miss would stream the cleared damage back from disk.
    void clearMaterialDamageHistory(const std::string& object_key = std::string()) {
        if (object_key.empty()) {
            msf_frame_cache_.clear();
            melt_applied_stamp_.clear();
        } else {
            for (auto& frame : msf_frame_cache_) {
                for (auto& system_snapshots : frame.second) {
                    system_snapshots.erase(std::remove_if(
                        system_snapshots.begin(), system_snapshots.end(),
                        [&](const auto& snapshot) {
                            return snapshot.object_key == object_key;
                        }), system_snapshots.end());
                }
            }
            melt_applied_stamp_.erase(object_key);
        }
        sim_cache_valid_ = false;
        sim_cache_dir_.clear();
        sim_cache_valid_system_ids_.clear();
    }

    // Live source-object pivot (frame-0 spawn pose) for a rigid body. Returns
    // false if the bound scene object can't be resolved yet. Used by the config
    // signatures to detect REPOSITIONING (moving the object without editing any
    // rigid param) and by the fluid-coupling overlap test.
    bool getRigidBodySourcePivot(const RayTrophiSim::RigidBodyObject& rb, Matrix4x4& out) const {
        if (rb.source_name.empty()) return false;
        for (const auto& obj : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->getNodeName() == rb.source_name) {
                if (Transform* th = tri->getTransformPtr()) { out = th->getPivotMatrix(); return true; }
                return false;
            }
        }
        return false;
    }

    // Refresh cached rigid rest poses (rb.initial_pivot) after a USER transform
    // edit — e.g. dragging a rigid's source object with the gizmo. The geometry
    // generation counter bumps on any world-vertex edit, but the simulation bumps
    // it too while stepping, so a bump is only trusted as a user edit by the IDLE
    // gate at the call site. Cheap: returns immediately when the generation hasn't
    // changed; on a real change it does an O(1) pivot read per rigid and only the
    // (heavier) full rest-pose recapture for bodies whose source actually moved.
    // The recapture updates initial_pivot so the config signatures pick up the
    // move on this same tick (no per-tick world scan in the hot path).
    void refreshRigidRestPosesOnUserEdit() {
        const uint64_t gen = g_scene_geometry_generation.load(std::memory_order_acquire);
        if (gen == last_user_edit_gen_) return;
        last_user_edit_gen_ = gen;
        auto q = [](float f) { return static_cast<int64_t>(f * 1000.0f); };
        auto poseDiffers = [&](const Matrix4x4& a, const Matrix4x4& b) {
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 4; ++c)
                    if (q(a.m[r][c]) != q(b.m[r][c])) return true;
            return false;
        };
        for (auto& rb : rigid_bodies) {
            Matrix4x4 live;
            if (!getRigidBodySourcePivot(rb, live)) continue;
            // CRITICAL: the sim writes its simulated pose back onto the source
            // object every step (last_written_pivot). That is NOT a user edit — if
            // we recaptured it as the spawn pose, initial_pivot would drift to the
            // body's current position each frame, the signature would change every
            // tick, and the body would be reset+re-simulated endlessly (it appears
            // to vibrate / be in two places at once). So skip when the live pose
            // still matches the last sim write, and skip when it already matches
            // the cached spawn pose. Only a pose differing from BOTH is a genuine
            // user reposition that should redefine the spawn point.
            if (rb.has_written && !poseDiffers(live, rb.last_written_pivot)) continue;
            if (rb.rest_captured && !poseDiffers(live, rb.initial_pivot)) continue;
            captureRigidBodyRestPose(rb);
        }
    }

    // Does editing/moving this rigid body change any FLUID bake? A Static body
    // can never move, so it only couples when its rest sphere overlaps a Fluid
    // grid-domain AABB — a far static prop's edits leave the (expensive) fluid
    // cache intact. Dynamic/Kinematic bodies may fall or animate into the tank
    // later, so they are treated as coupled whenever ANY fluid domain exists.
    // Conservative (returns true) when the rest pose can't be resolved.
    bool rigidCouplesToFluid(const RayTrophiSim::RigidBodyObject& rb) const {
        const bool can_move = (rb.motion_type != RayTrophiSim::RigidBodyMotionType::Static);
        Vec3 center;
        float radius_sq = 0.0f;
        bool have_sphere = false;
        if (!can_move && rb.rest_captured) {
            // Cached rest pose — no live world scan (refreshed on user edit).
            center = rb.initial_pivot.getTranslation();
            const Vec3 hh = rb.rest_half_extents;
            radius_sq = hh.x * hh.x + hh.y * hh.y + hh.z * hh.z;
            have_sphere = true;
        }
        for (const auto& s : particle_systems) {
            if (!s.runtime) continue;
            for (const auto& d : s.runtime->gridDomains()) {
                if (d.type != RayTrophiSim::SimulationDomainType::Fluid) continue;
                if (can_move) return true;        // dynamic/kinematic + any fluid => coupled
                if (!have_sphere) return true;    // unresolved static pose => conservative
                // Closest point on the domain AABB to the rest-sphere centre.
                const float cx = std::max(d.bounds_min.x, std::min(center.x, d.bounds_max.x));
                const float cy = std::max(d.bounds_min.y, std::min(center.y, d.bounds_max.y));
                const float cz = std::max(d.bounds_min.z, std::min(center.z, d.bounds_max.z));
                const float dx = center.x - cx, dy = center.y - cy, dz = center.z - cz;
                if (dx * dx + dy * dy + dz * dz <= radius_sq) return true;
            }
        }
        return false;
    }

    // Signature of ONLY the inputs a fluid bake actually depends on: the grid /
    // emitter / collider / flow-source config, the gas/fluid/force-field element
    // counts, and the rigid bodies that couple to fluid (their collider geometry,
    // dynamics, and live spawn pose). A non-coupling rigid (e.g. a far static
    // prop) is deliberately absent here, so editing or moving it leaves this
    // signature — and therefore the fluid cache — unchanged.
    // -- Fluid domain solver config -> cache signature ----------------------
    // Everything the APIC solve READS off a domain descriptor, folded into both
    // simulation signatures so editing it drops the stale bake instead of
    // replaying it.
    //
    // ★★★ THE BUG THIS CLOSES: the two signatures hashed emitters, colliders,
    // flow sources, force fields and rigid bodies -- and NOTHING off the grid
    // domain itself except its thermal override. So viscosity, gravity, FLIP
    // blend, boundary mode, seed density, per-substance physics... every one of
    // them was edited, ignored, and the previously baked frames replayed
    // unchanged. The control appeared dead until the user pressed Reset by
    // hand, which is exactly the workflow the auto-invalidation was written to
    // make unnecessary.
    //
    // ★★★ LOOK IS NOT PHYSICS, and that split is the whole design here:
    //   physics (this function) -> invalidate the bake, rewind, re-simulate
    //   look (material id, representation, porosity, coord space, IOR, ...)
    //        -> refreshFluidSurfaceMaterial + a render resync, NO re-sim
    // Hashing a look value here would throw away an expensive bake every time
    // somebody clicked a material, and the user would learn to fear the panel.
    // Leaving a physics value OUT -- the state this replaces -- silently shows
    // the wrong simulation, which is worse because nothing announces it.
    //
    // ★★ BOUNDS ARE DELIBERATELY ABSENT. A domain bound to an object tracks it
    // every frame, so hashing bounds_min/max would re-key the cache on every
    // frame of a MOVING domain and the bake could never complete. Domain motion
    // has its own path (domain_motion_delta). Resolution and voxel size are
    // authored and safe: a translation does not change them.
    //
    // ★ Bit-exact, not quantized. The rest of these signatures quantize to
    // 1/1000 to keep live poses from jittering the hash -- but a kinematic
    // viscosity of 1e-6 (water) quantizes to ZERO, and so does 9e-4. Half the
    // physical range of the very control the user reported as dead would have
    // stayed dead. These are AUTHORED values, never written per step, so there
    // is no jitter to filter and the exact bits are the honest key.
    static uint64_t hashFluidDomainSolverConfig(uint64_t h,
                                                const RayTrophiSim::SimulationGridDomainDesc& d) {
        auto mix = [](uint64_t hv, uint64_t v) {
            hv ^= v + 0x9e3779b97f4a7c15ull + (hv << 6) + (hv >> 2);
            return hv;
        };
        auto fb = [](float f) {
            uint32_t bits = 0;
            std::memcpy(&bits, &f, sizeof(bits));
            return static_cast<uint64_t>(bits);
        };
        if (d.type != RayTrophiSim::SimulationDomainType::Fluid) return h;
        h = mix(h, d.enabled ? 1ull : 0ull);
        h = mix(h, static_cast<uint64_t>(d.backend));
        h = mix(h, static_cast<uint64_t>(d.boundary_mode));

        // ★★★ voxel_size / resolution_* / padding ARE NOT HASHED, and leaving
        // them out is not an oversight — the runtime WRITES THEM BACK every
        // sync, recomputed from the domain extent
        // (ParticleSimulation.cpp: domain.voxel_size = max(extent_i / res_i)).
        // For a domain bound to a moving object the extent drifts, so their bits
        // change every frame and this signature would re-key the cache forever:
        // the bake would rewind to frame 0 on every tick and never complete.
        // That failure is far worse than the one it would fix, and grid-shape
        // edits already have their own path (the seed-settled auto-reseed in the
        // domain panel, which reseeds and snaps the playhead to start).
        //
        // Same reason fluid_params.particles_per_cell is absent while
        // fluid_seed_particles_per_cell below IS hashed: the solver overwrites
        // the former at seed time with the budget-clamped value. Hash what the
        // USER authored, never what the runtime derived from it.
        const auto& fp = d.fluid_params;
        h = mix(h, fb(fp.gravity.x)); h = mix(h, fb(fp.gravity.y)); h = mix(h, fb(fp.gravity.z));
        h = mix(h, fb(fp.cfl));
        h = mix(h, static_cast<uint64_t>(fp.max_substeps));
        h = mix(h, static_cast<uint64_t>(fp.pressure_iterations));
        h = mix(h, fb(fp.pressure_relative_residual));
        h = mix(h, fp.pressure_multigrid_preconditioner ? 1ull : 0ull);
        h = mix(h, fb(fp.apic_blend));
        h = mix(h, fb(fp.flip_blend));
        h = mix(h, fb(fp.max_velocity));
        h = mix(h, fb(fp.velocity_damping));
        h = mix(h, fb(fp.wall_damping));
        h = mix(h, fb(fp.density_correction));
        h = mix(h, fb(fp.internal_friction));
        h = mix(h, fb(fp.air_drag));
        h = mix(h, fp.reseed_enabled ? 1ull : 0ull);
        h = mix(h, static_cast<uint64_t>(fp.reseed_target_per_cell));
        h = mix(h, static_cast<uint64_t>(fp.reseed_min_per_cell));
        h = mix(h, static_cast<uint64_t>(fp.reseed_max_per_cell));
        h = mix(h, static_cast<uint64_t>(fp.max_particles));
        h = mix(h, static_cast<uint64_t>(fp.uvw_refresh_period));
        h = mix(h, fb(fp.domain_motion_coupling));
        h = mix(h, fb(fp.kinematic_viscosity));
        h = mix(h, static_cast<uint64_t>(fp.viscosity_sweeps));
        h = mix(h, fb(fp.viscosity_wall_slip));
        h = mix(h, fb(fp.affine_damping));
        h = mix(h, fb(fp.max_affine));
        h = mix(h, static_cast<uint64_t>(fp.boundary));
        h = mix(h, static_cast<uint64_t>(fp.chemistry_preset));
        h = mix(h, fp.free_surface ? 1ull : 0ull);
        h = mix(h, fp.variational_solids ? 1ull : 0ull);
        h = mix(h, fp.ghost_fluid_surface ? 1ull : 0ull);

        // ★★★ GRANULAR MATERIAL. Every one of these changes the constitutive
        // law, so a cache baked at one friction angle must never replay as
        // another. Their absence here is why editing Young modulus or cohesion
        // left the old bake on screen and forced a manual Reset + Seed: the
        // signature never moved, so the rewind never fired and the panel looked
        // like it did nothing.
        //
        // ★ These are AUTHORED values, safe to hash. sanitizeGranularMaterial
        // clamps them at edit time (deterministically, from the authored value),
        // and unlike voxel_size or particles_per_cell above, the runtime never
        // writes them back per frame -- ParticleSimulation takes a COPY of
        // fluid_params before the substep loop rescales its damping fields, so
        // that rewrite cannot reach the domain and re-key the cache every tick.
        h = mix(h, fp.granular_enabled ? 1ull : 0ull);
        h = mix(h, fb(fp.granular_friction_angle_degrees));
        h = mix(h, fb(fp.granular_cohesion));
        h = mix(h, fb(fp.granular_dilatancy_degrees));
        h = mix(h, fb(fp.granular_young_modulus));
        h = mix(h, fb(fp.granular_poisson_ratio));
        h = mix(h, fb(fp.granular_tensile_cutoff));
        h = mix(h, fb(fp.granular_hardening));
        h = mix(h, fb(fp.granular_fracture_strain));
        h = mix(h, fb(fp.granular_damage_rate));
        h = mix(h, fb(fp.granular_healing_rate));
        h = mix(h, fp.granular_rebonding ? 1ull : 0ull);
        // The substep ceiling is not cosmetic: it decides how much of the
        // authored stiffness the solver can actually deliver (measured -- at
        // h = 0.05 m and 24 fps, E = 2e5 Pa needs 27 substeps, so a ceiling of
        // 16 quietly ran the material at ~72 kPa). Two bakes at different
        // ceilings are two different simulations.
        h = mix(h, static_cast<uint64_t>(fp.granular_max_solver_substeps));
        // Softening changes the material's whole history, so a bake made with
        // it off must not replay as one made with it on. These are AUTHORED;
        // the per-particle multiplier they drive lives on the particles and is
        // deliberately never written back here (see APICSolverParams).
        h = mix(h, fb(fp.granular_softening_temperature));
        h = mix(h, fb(fp.granular_softening_range));
        h = mix(h, fb(fp.granular_residual_strength));
        h = mix(h, fb(fp.granular_tack_peak));
        h = mix(h, fb(fp.granular_thermal_conductivity));

        // Authored initial state. A seed box or fill level that does not re-key
        // the cache means the tank the user just resized replays at its old
        // level -- the setting looks like it did nothing.
        h = mix(h, static_cast<uint64_t>(d.fluid_seed_mode));
        // ★★★ THE SEED AABB IS NOT HASHED, and this one was learned the
        // expensive way: fluid_seed_min/max FOLLOW THE DOMAIN. When the domain
        // is translated the sync glues the seed box to it
        // (ParticleSimulation.cpp: "Fluid seed AABB follows the domain's
        // translation"), so hashing it re-keys the cache on every frame that
        // the domain moves. Symptom, reported from a real session: dragging the
        // domain DURING PLAYBACK snapped the playhead to frame 0 every tick and
        // the simulation could never advance.
        //
        // ★★ Same family as the bounds and voxel_size exclusions above, and the
        // rule they all serve is one line: HASH WHAT THE USER AUTHORED, NEVER
        // WHAT THE RUNTIME DERIVED FROM IT. A field that the sim writes back is
        // a live pose wearing a config field's name, and every one of them
        // turns this signature into a cache shredder.
        //
        // Editing the seed box is not lost as a result: it has its own path,
        // the panel's seed-settled auto-reseed, which reseeds and snaps the
        // playhead to start once the drag is released.
        h = mix(h, static_cast<uint64_t>(d.fluid_seed_particles_per_cell));
        h = mix(h, static_cast<uint64_t>(d.fluid_max_particles));
        h = mix(h, d.fluid_reseed_on_reset ? 1ull : 0ull);
        h = mix(h, fb(d.fluid_fill_level));
        h = mix(h, fb(d.fluid_fill_wall_margin));

        // Liquid -> gas combustion coupling: all authored, all change the solve.
        h = mix(h, d.fluid_flammable ? 1ull : 0ull);
        h = mix(h, d.fluid_extinguishing ? 1ull : 0ull);
        h = mix(h, d.fluid_auto_ignite ? 1ull : 0ull);
        h = mix(h, fb(d.fluid_ignition_temperature));
        h = mix(h, fb(d.fluid_evaporation_rate));
        h = mix(h, fb(d.fluid_surface_fuel_capacity));
        h = mix(h, fb(d.fluid_combustion_heat_release));
        h = mix(h, fb(d.fluid_combustion_smoke_yield));
        h = mix(h, fb(d.fluid_surface_cooling));
        h = mix(h, fb(d.fluid_cooling_power));
        h = mix(h, fb(d.fluid_oxygen_dilution));

        // Solid-phase coupling: both are physics, both change the solve.
        h = mix(h, d.fluid_solid_phase_enabled ? 1ull : 0ull);
        h = mix(h, fb(d.fluid_solid_phase_fill));

        // -- Per-substance PHYSICS only -------------------------------------
        // ★★★ material_id and representation are NOT here on purpose. They are
        // look: assigning a material must repaint the current frame, never
        // discard a bake. They reach the picture through the composition field
        // instead -- see the surface rebuild signature, which DOES hash them.
        for (const auto& b : d.fluid_substance_materials) {
            h = mix(h, static_cast<uint64_t>(
                RayTrophiSim::Fluid::substanceTag(b.substance)));
            h = mix(h, fb(b.kinematic_viscosity));
            h = mix(h, fb(b.miscibility));
            h = mix(h, static_cast<uint64_t>(b.phase));
        }
        return h;
    }

    uint64_t computeFluidCouplingSignature() const {
        auto mix = [](uint64_t h, uint64_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            return h;
        };
        auto qf = [](float f) { return static_cast<uint64_t>(static_cast<int64_t>(f * 1000.0f)); };
        uint64_t h = 1469598103934665603ull;
        h = mix(h, particle_systems.size());
        for (const auto& s : particle_systems) {
            if (!s.runtime) { h = mix(h, 0); continue; }
            h = mix(h, s.runtime->gridDomains().size());
            h = mix(h, s.runtime->emitters().size());
            // Emitter config (rate/velocity/spread/lifetime/shape/etc.) must
            // invalidate the bake — editing it otherwise replays the stale RAM
            // cache. Skip the LIVE `accumulator` (the sim writes it every step, so
            // hashing it would reset the cache every frame — the same live-pose
            // thrash trap the force-field/rigid hashes avoid). source_name/point/
            // direction are authored config (the resolver reads them, never writes).
            for (const auto& em : s.runtime->emitters()) {
                h = mix(h, em.enabled ? 1ull : 0ull);
                h = mix(h, static_cast<uint64_t>(em.source_mode));
                h = mix(h, static_cast<uint64_t>(em.spawn_mode));
                for (char ch : em.source_name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                h = mix(h, qf(em.point.x)); h = mix(h, qf(em.point.y)); h = mix(h, qf(em.point.z));
                h = mix(h, qf(em.local_offset.x)); h = mix(h, qf(em.local_offset.y)); h = mix(h, qf(em.local_offset.z));
                h = mix(h, qf(em.direction.x)); h = mix(h, qf(em.direction.y)); h = mix(h, qf(em.direction.z));
                h = mix(h, qf(em.surface_offset));
                h = mix(h, qf(em.rate_per_second));
                h = mix(h, static_cast<uint64_t>(em.burst_count));
                h = mix(h, qf(em.speed));
                h = mix(h, qf(em.spread));
                h = mix(h, qf(em.lifetime_seconds));
                h = mix(h, qf(em.mass));
                h = mix(h, qf(em.angular_velocity));
                h = mix(h, qf(em.angular_jitter));
                h = mix(h, static_cast<uint64_t>(em.seed));
            }
            h = mix(h, s.runtime->colliders().size());
            for (const auto& c : s.runtime->colliders()) {
                for (char ch : c.name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                for (char ch : c.source_name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                h = mix(h, static_cast<uint64_t>(c.source_mode));
                h = mix(h, c.enabled ? 1ull : 0ull);
                h = mix(h, c.fluid_collision_enabled ? 1ull : 0ull);
                h = mix(h, qf(c.plane_y));
                h = mix(h, qf(c.sphere_center.x)); h = mix(h, qf(c.sphere_center.y)); h = mix(h, qf(c.sphere_center.z));
                h = mix(h, qf(c.sphere_radius));
                h = mix(h, qf(c.capsule_start.x)); h = mix(h, qf(c.capsule_start.y)); h = mix(h, qf(c.capsule_start.z));
                h = mix(h, qf(c.capsule_end.x)); h = mix(h, qf(c.capsule_end.y)); h = mix(h, qf(c.capsule_end.z));
                h = mix(h, qf(c.capsule_radius));
                h = mix(h, qf(c.bounds_min.x)); h = mix(h, qf(c.bounds_min.y)); h = mix(h, qf(c.bounds_min.z));
                h = mix(h, qf(c.bounds_max.x)); h = mix(h, qf(c.bounds_max.y)); h = mix(h, qf(c.bounds_max.z));
                h = mix(h, qf(c.friction));
                h = mix(h, qf(c.restitution));
                h = mix(h, qf(c.thickness));
                h = mix(h, c.gas_interaction_enabled ? 1ull : 0ull);
                h = mix(h, qf(c.gas_density_rate));
                h = mix(h, qf(c.gas_temperature_rate));
                h = mix(h, qf(c.gas_fuel_rate));
                h = mix(h, qf(c.gas_flame_rate));
                h = mix(h, qf(c.gas_surface_band_voxels));
                // Substance selection changes the derived ignition/fuel/burn
                // values, so a cache baked as "Wood" must not replay as "Iron".
                for (char ch : c.msf_substance) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                h = mix(h, c.msf_override.override_ignition ? 1ull : 0ull);
                h = mix(h, qf(c.msf_override.ignition_kelvin));
                h = mix(h, qf(c.msf_override.burn_rate_scale));
                h = mix(h, qf(c.msf_override.fuel_capacity_scale));
                h = mix(h, static_cast<uint64_t>(c.msf_mask_resolution));
                h = mix(h, c.msf_auto_transfer ? 1ull : 0ull);
                for (char ch : c.msf_transfer_domain) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                h = mix(h, qf(c.msf_transfer_rate_kg_s));
                h = mix(h, qf(c.msf_transfer_min_mass_kg));
                h = mix(h, qf(c.msf_transfer_particles_per_kg));
                h = mix(h, c.msf_transfer_max_batch_particles);
                h = mix(h, c.msf_melt_flow_enabled ? 1ull : 0ull);
                h = mix(h, qf(c.msf_melt_height_loss));
                h = mix(h, qf(c.msf_melt_spread));
                h = mix(h, c.msf_melt_sdf_refresh ? 1ull : 0ull);
                h = mix(h, c.msf_melt_sdf_revision_interval);
                h = mix(h, qf(c.msf_melt_sdf_change_threshold));
                h = mix(h, qf(c.msf_transfer_velocity.x));
                h = mix(h, qf(c.msf_transfer_velocity.y));
                h = mix(h, qf(c.msf_transfer_velocity.z));
                h = mix(h, c.msf_generate_char_mask ? 1ull : 0ull);
            }
            // World thermal boundary conditions (Phase 4). Ambient, the Kelvin
            // calibration, convection and oxygen all change how much an object
            // heats and how fast it burns, so a cache baked in a cold draughty
            // room must not replay as one baked in a sealed hot one.
            {
                const auto& wt = s.runtime->worldThermal();
                h = mix(h, qf(wt.ambient_kelvin));
                h = mix(h, qf(wt.kelvin_per_unit));
                h = mix(h, qf(wt.convection_coefficient));
                h = mix(h, qf(wt.oxygen_availability));
            }
            // Per-domain ambient/oxygen override, for the same reason.
            for (const auto& d : s.runtime->gridDomains()) {
                h = mix(h, d.thermal_override_enabled ? 1ull : 0ull);
                if (d.thermal_override_enabled) {
                    h = mix(h, qf(d.thermal_ambient_kelvin));
                    h = mix(h, qf(d.thermal_oxygen));
                }
                h = hashFluidDomainSolverConfig(h, d);
            }
            h = mix(h, s.runtime->flowSources().size());
            // Flow sources are domain-owned emitters. Hash their authored
            // routing, fields and animation keys, otherwise moving/keying one
            // while keeping the same source count can replay a stale gas/fluid
            // cache (especially visible with multiple domains/systems).
            // Runtime counters/accumulators are deliberately excluded.
            for (const auto& f : s.runtime->flowSources()) {
                h = mix(h, f.enabled ? 1ull : 0ull);
                h = mix(h, static_cast<uint64_t>(f.source_mode));
                h = mix(h, static_cast<uint64_t>(static_cast<int64_t>(f.domain_index)));
                for (char ch : f.source_name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                h = mix(h, qf(f.position.x)); h = mix(h, qf(f.position.y)); h = mix(h, qf(f.position.z));
                h = mix(h, qf(f.velocity.x)); h = mix(h, qf(f.velocity.y)); h = mix(h, qf(f.velocity.z));
                h = mix(h, qf(f.radius));
                h = mix(h, qf(f.density));
                h = mix(h, qf(f.temperature));
                h = mix(h, qf(f.fuel));
                h = mix(h, qf(f.falloff));
                h = mix(h, qf(f.velocity_coupling));
                h = mix(h, qf(f.fluid_particles_per_second));
                h = mix(h, qf(f.fluid_velocity_spread));
                h = mix(h, f.fluid_emit_along_normal ? 1ull : 0ull);
                // ★ The substance changes what the liquid IS, so a cached
                // sequence baked under the old name no longer describes it.
                h = mix(h, static_cast<uint64_t>(
                    RayTrophiSim::Fluid::substanceTag(f.fluid_substance)));
                h = mix(h, f.use_time_limit ? 1ull : 0ull);
                h = mix(h, qf(f.start_time)); h = mix(h, qf(f.end_time));
                h = mix(h, f.use_particle_limit ? 1ull : 0ull);
                h = mix(h, static_cast<uint64_t>(f.max_emitted_particles));
                h = mix(h, f.keyframes.size());
                for (const auto& [frame, k] : f.keyframes) {
                    h = mix(h, static_cast<uint64_t>(static_cast<int64_t>(frame)));
                    h = mix(h, k.has_enabled ? 1ull : 0ull);
                    h = mix(h, k.has_position ? 1ull : 0ull);
                    h = mix(h, k.has_velocity ? 1ull : 0ull);
                    h = mix(h, k.has_radius ? 1ull : 0ull);
                    h = mix(h, k.has_density ? 1ull : 0ull);
                    h = mix(h, k.has_temperature ? 1ull : 0ull);
                    h = mix(h, k.has_fuel ? 1ull : 0ull);
                    h = mix(h, k.has_falloff ? 1ull : 0ull);
                    h = mix(h, k.has_velocity_coupling ? 1ull : 0ull);
                    h = mix(h, k.enabled ? 1ull : 0ull);
                    h = mix(h, qf(k.position.x)); h = mix(h, qf(k.position.y)); h = mix(h, qf(k.position.z));
                    h = mix(h, qf(k.velocity.x)); h = mix(h, qf(k.velocity.y)); h = mix(h, qf(k.velocity.z));
                    h = mix(h, qf(k.radius)); h = mix(h, qf(k.density));
                    h = mix(h, qf(k.temperature)); h = mix(h, qf(k.fuel));
                    h = mix(h, qf(k.falloff)); h = mix(h, qf(k.velocity_coupling));
                }
            }
        }
        h = mix(h, gas_volumes.size());
        h = mix(h, fluid_objects.size());
        // Force fields drive the fluid too, so editing a field that affects fluid
        // (strength / direction / position / wind-coupling knobs / noise) must
        // re-bake the FLUID — not just bump the count. Without this the fluid
        // cache replayed stale after any wind tweak (only the rigid/soft caches
        // dropped). Gated on affects_fluid: a field that doesn't touch fluid never
        // invalidates the (expensive) fluid bake.
        h = mix(h, force_field_manager.force_fields.size());
        for (const auto& ff : force_field_manager.force_fields) {
            // ★ A Thermal field never "affects fluid" — it exerts no force at all
            // (its affect mask is zeroed in SimulationForceFieldSnapshot). It DOES
            // drive surface heating and therefore pyrolysis, so it has to be
            // hashed on its own terms or moving a burner would replay a stale
            // burn. This is exactly the trap the affects_fluid gate was written
            // to avoid in the other direction.
            if (ff && ff->type == Physics::ForceFieldType::Thermal) {
                h = mix(h, ff->enabled ? 1ull : 0ull);
                h = mix(h, static_cast<uint64_t>(ff->shape));
                h = mix(h, static_cast<uint64_t>(ff->falloff_type));
                h = mix(h, qf(ff->thermal_delta_kelvin));
                h = mix(h, qf(ff->position.x)); h = mix(h, qf(ff->position.y)); h = mix(h, qf(ff->position.z));
                h = mix(h, qf(ff->falloff_radius)); h = mix(h, qf(ff->inner_radius));
                h = mix(h, qf(ff->start_frame)); h = mix(h, qf(ff->end_frame));
                continue;
            }
            if (!ff || !ff->affects_fluid) { h = mix(h, 0); continue; }
            h = mix(h, ff->enabled ? 1ull : 0ull);
            h = mix(h, static_cast<uint64_t>(ff->type));
            h = mix(h, static_cast<uint64_t>(ff->shape));
            h = mix(h, static_cast<uint64_t>(ff->falloff_type));
            h = mix(h, qf(ff->strength));
            h = mix(h, qf(ff->position.x)); h = mix(h, qf(ff->position.y)); h = mix(h, qf(ff->position.z));
            h = mix(h, qf(ff->rotation.x)); h = mix(h, qf(ff->rotation.y)); h = mix(h, qf(ff->rotation.z));
            h = mix(h, qf(ff->scale.x)); h = mix(h, qf(ff->scale.y)); h = mix(h, qf(ff->scale.z));
            h = mix(h, qf(ff->direction.x)); h = mix(h, qf(ff->direction.y)); h = mix(h, qf(ff->direction.z));
            h = mix(h, qf(ff->falloff_radius)); h = mix(h, qf(ff->inner_radius));
            h = mix(h, qf(ff->axis.x)); h = mix(h, qf(ff->axis.y)); h = mix(h, qf(ff->axis.z));
            h = mix(h, qf(ff->inward_force)); h = mix(h, qf(ff->upward_force));
            h = mix(h, qf(ff->linear_drag)); h = mix(h, qf(ff->quadratic_drag));
            h = mix(h, ff->use_noise ? 1ull : 0ull);
            h = mix(h, qf(ff->noise.frequency)); h = mix(h, qf(ff->noise.amplitude));
            h = mix(h, qf(ff->noise.speed)); h = mix(h, static_cast<uint64_t>(ff->noise.octaves));
            // Wind→fluid surface-drag knobs.
            h = mix(h, ff->fluid_surface_drag ? 1ull : 0ull);
            h = mix(h, qf(ff->fluid_drag_coupling));
            h = mix(h, qf(ff->fluid_surface_depth));
            h = mix(h, qf(ff->fluid_curl_detail));
        }
        for (const auto& rb : rigid_bodies) {
            if (!rigidCouplesToFluid(rb)) continue;
            for (char c : rb.source_name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(c)));
            for (char c : rb.collider_name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(c)));
            // Breakable bodies flip motion_type/dynamic when they shatter at runtime;
            // hash the authored-static intent so a break doesn't invalidate the fluid
            // coupling cache (and re-bake/reset). See computeSimConfigSignature.
            if (rb.getBreakable()) {
                h = mix(h, 0xB4EAC0DEull);
            } else {
                h = mix(h, static_cast<uint64_t>(rb.motion_type));
                h = mix(h, rb.dynamic ? 1ull : 0ull);
            }
            h = mix(h, static_cast<uint64_t>(rb.shape));
            h = mix(h, rb.enabled ? 1ull : 0ull);
            h = mix(h, qf(rb.mass));
            h = mix(h, rb.auto_mass_from_density ? 1ull : 0ull);
            h = mix(h, qf(rb.density));
            h = mix(h, qf(rb.linear_damping));
            h = mix(h, qf(rb.angular_damping));
            h = mix(h, qf(rb.gravity_scale));
            h = mix(h, qf(rb.friction));
            h = mix(h, qf(rb.restitution));
            h = mix(h, qf(rb.initial_linear_velocity.x));
            h = mix(h, qf(rb.initial_linear_velocity.y));
            h = mix(h, qf(rb.initial_linear_velocity.z));
            h = mix(h, qf(rb.initial_angular_velocity.x));
            h = mix(h, qf(rb.initial_angular_velocity.y));
            h = mix(h, qf(rb.initial_angular_velocity.z));
            h = mix(h, rb.lock_translation_x ? 1ull : 0ull);
            h = mix(h, rb.lock_translation_y ? 1ull : 0ull);
            h = mix(h, rb.lock_translation_z ? 1ull : 0ull);
            h = mix(h, rb.lock_rotation_x ? 1ull : 0ull);
            h = mix(h, rb.lock_rotation_y ? 1ull : 0ull);
            h = mix(h, rb.lock_rotation_z ? 1ull : 0ull);
            h = mix(h, rb.fluid_coupling_enabled ? 1ull : 0ull);
            h = mix(h, qf(rb.getBuoyancyScale()));
            h = mix(h, qf(rb.getFluidDensity()));
            h = mix(h, qf(rb.getFluidDrag()));
            h = mix(h, qf(rb.getFluidQuadraticDrag()));
            h = mix(h, qf(rb.getFluidAngularDrag()));
            // Cached spawn pose (see computeSimConfigSignature) — O(1), no scan.
            {
                const Vec3 t = rb.initial_pivot.getTranslation();
                h = mix(h, qf(t.x)); h = mix(h, qf(t.y)); h = mix(h, qf(t.z));
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 4; ++c) h = mix(h, qf(rb.initial_pivot.m[r][c]));
            }
        }
        return h;
    }

    // Cheap content hash of the simulation SETUP (not its live state). Changes
    // when sim elements are added/removed or a rigid body's params are edited, so
    // updateSimulationTimeline can auto-drop a stale bake cache. Excludes anything
    // that mutates per step (particle counts, positions) so it stays stable while
    // the sim runs. NOTE: deep per-domain particle/gas/fluid param edits are not
    // all hashed yet — full content signature is part of the Faz 5 cache hardening.
    uint64_t computeSimConfigSignature() const {
        auto mix = [](uint64_t h, uint64_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
            return h;
        };
        auto qf = [](float f) { return static_cast<uint64_t>(static_cast<int64_t>(f * 1000.0f)); };
        uint64_t h = 1469598103934665603ull;
        h = mix(h, particle_systems.size());
        for (const auto& s : particle_systems) {
            if (!s.runtime) { h = mix(h, 0); continue; }
            h = mix(h, s.runtime->gridDomains().size());
            // ★★★ The DISCRETISATION the simulation lives in. Only the domain
            // COUNT used to be hashed here, so changing a domain's voxel size or
            // bounds left the signature identical: a bake survived a grid
            // resolution change and kept serving frames solved on a different
            // grid, while still reporting itself valid. Measured 2026-08-17 with
            // the N6 cache node — that is stale physics reaching a render with
            // nothing to report it.
            //
            // ★★ AUTHORED fields only. Nothing listed here mutates per step;
            // hashing a derived or live field would drop the cache every frame
            // (the same thrash trap the emitter hash below avoids by skipping
            // `accumulator`).
            for (const auto& gd : s.runtime->gridDomains()) {
                for (char ch : gd.name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                h = mix(h, static_cast<uint64_t>(gd.type));
                h = mix(h, static_cast<uint64_t>(gd.backend));
                h = mix(h, static_cast<uint64_t>(gd.boundary_mode));
                h = mix(h, static_cast<uint64_t>(gd.source_mode));
                h = mix(h, gd.enabled ? 1ull : 0ull);
                h = mix(h, qf(gd.bounds_min.x)); h = mix(h, qf(gd.bounds_min.y)); h = mix(h, qf(gd.bounds_min.z));
                h = mix(h, qf(gd.bounds_max.x)); h = mix(h, qf(gd.bounds_max.y)); h = mix(h, qf(gd.bounds_max.z));
                h = mix(h, static_cast<uint64_t>(gd.resolution_x));
                h = mix(h, static_cast<uint64_t>(gd.resolution_y));
                h = mix(h, static_cast<uint64_t>(gd.resolution_z));
                h = mix(h, qf(gd.voxel_size));
            }
            h = mix(h, s.runtime->emitters().size());
            // Emitter config (rate/velocity/spread/lifetime/shape/etc.) must
            // invalidate the bake — editing it otherwise replays the stale RAM
            // cache. Skip the LIVE `accumulator` (the sim writes it every step, so
            // hashing it would reset the cache every frame — the same live-pose
            // thrash trap the force-field/rigid hashes avoid). source_name/point/
            // direction are authored config (the resolver reads them, never writes).
            for (const auto& em : s.runtime->emitters()) {
                h = mix(h, em.enabled ? 1ull : 0ull);
                h = mix(h, static_cast<uint64_t>(em.source_mode));
                h = mix(h, static_cast<uint64_t>(em.spawn_mode));
                for (char ch : em.source_name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                h = mix(h, qf(em.point.x)); h = mix(h, qf(em.point.y)); h = mix(h, qf(em.point.z));
                h = mix(h, qf(em.local_offset.x)); h = mix(h, qf(em.local_offset.y)); h = mix(h, qf(em.local_offset.z));
                h = mix(h, qf(em.direction.x)); h = mix(h, qf(em.direction.y)); h = mix(h, qf(em.direction.z));
                h = mix(h, qf(em.surface_offset));
                h = mix(h, qf(em.rate_per_second));
                h = mix(h, static_cast<uint64_t>(em.burst_count));
                h = mix(h, qf(em.speed));
                h = mix(h, qf(em.spread));
                h = mix(h, qf(em.lifetime_seconds));
                h = mix(h, qf(em.mass));
                h = mix(h, qf(em.angular_velocity));
                h = mix(h, qf(em.angular_jitter));
                h = mix(h, static_cast<uint64_t>(em.seed));
            }
            h = mix(h, s.runtime->colliders().size());
            for (const auto& c : s.runtime->colliders()) {
                for (char ch : c.name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                for (char ch : c.source_name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                h = mix(h, static_cast<uint64_t>(c.source_mode));
                h = mix(h, c.enabled ? 1ull : 0ull);
                h = mix(h, c.fluid_collision_enabled ? 1ull : 0ull);
                h = mix(h, qf(c.plane_y));
                h = mix(h, qf(c.sphere_center.x)); h = mix(h, qf(c.sphere_center.y)); h = mix(h, qf(c.sphere_center.z));
                h = mix(h, qf(c.sphere_radius));
                h = mix(h, qf(c.capsule_start.x)); h = mix(h, qf(c.capsule_start.y)); h = mix(h, qf(c.capsule_start.z));
                h = mix(h, qf(c.capsule_end.x)); h = mix(h, qf(c.capsule_end.y)); h = mix(h, qf(c.capsule_end.z));
                h = mix(h, qf(c.capsule_radius));
                h = mix(h, qf(c.bounds_min.x)); h = mix(h, qf(c.bounds_min.y)); h = mix(h, qf(c.bounds_min.z));
                h = mix(h, qf(c.bounds_max.x)); h = mix(h, qf(c.bounds_max.y)); h = mix(h, qf(c.bounds_max.z));
                h = mix(h, qf(c.friction));
                h = mix(h, qf(c.restitution));
                h = mix(h, qf(c.thickness));
                h = mix(h, c.gas_interaction_enabled ? 1ull : 0ull);
                h = mix(h, qf(c.gas_density_rate));
                h = mix(h, qf(c.gas_temperature_rate));
                h = mix(h, qf(c.gas_fuel_rate));
                h = mix(h, qf(c.gas_flame_rate));
                h = mix(h, qf(c.gas_surface_band_voxels));
                // Substance selection changes the derived ignition/fuel/burn
                // values, so a cache baked as "Wood" must not replay as "Iron".
                for (char ch : c.msf_substance) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                h = mix(h, c.msf_override.override_ignition ? 1ull : 0ull);
                h = mix(h, qf(c.msf_override.ignition_kelvin));
                h = mix(h, qf(c.msf_override.burn_rate_scale));
                h = mix(h, qf(c.msf_override.fuel_capacity_scale));
                h = mix(h, static_cast<uint64_t>(c.msf_mask_resolution));
                h = mix(h, c.msf_auto_transfer ? 1ull : 0ull);
                for (char ch : c.msf_transfer_domain) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                h = mix(h, qf(c.msf_transfer_rate_kg_s));
                h = mix(h, qf(c.msf_transfer_min_mass_kg));
                h = mix(h, qf(c.msf_transfer_particles_per_kg));
                h = mix(h, c.msf_transfer_max_batch_particles);
                h = mix(h, c.msf_melt_flow_enabled ? 1ull : 0ull);
                h = mix(h, qf(c.msf_melt_height_loss));
                h = mix(h, qf(c.msf_melt_spread));
                h = mix(h, c.msf_melt_sdf_refresh ? 1ull : 0ull);
                h = mix(h, c.msf_melt_sdf_revision_interval);
                h = mix(h, qf(c.msf_melt_sdf_change_threshold));
                h = mix(h, qf(c.msf_transfer_velocity.x));
                h = mix(h, qf(c.msf_transfer_velocity.y));
                h = mix(h, qf(c.msf_transfer_velocity.z));
                h = mix(h, c.msf_generate_char_mask ? 1ull : 0ull);
            }
            // World thermal boundary conditions (Phase 4). Ambient, the Kelvin
            // calibration, convection and oxygen all change how much an object
            // heats and how fast it burns, so a cache baked in a cold draughty
            // room must not replay as one baked in a sealed hot one.
            {
                const auto& wt = s.runtime->worldThermal();
                h = mix(h, qf(wt.ambient_kelvin));
                h = mix(h, qf(wt.kelvin_per_unit));
                h = mix(h, qf(wt.convection_coefficient));
                h = mix(h, qf(wt.oxygen_availability));
            }
            // Per-domain ambient/oxygen override, for the same reason.
            for (const auto& d : s.runtime->gridDomains()) {
                h = mix(h, d.thermal_override_enabled ? 1ull : 0ull);
                if (d.thermal_override_enabled) {
                    h = mix(h, qf(d.thermal_ambient_kelvin));
                    h = mix(h, qf(d.thermal_oxygen));
                }
                h = hashFluidDomainSolverConfig(h, d);
            }
            h = mix(h, s.runtime->flowSources().size());
            // Keep the general simulation-cache signature sensitive to the
            // same authored flow-source state as the fluid coupling signature.
            // Per-step accumulator/emitted counters must not participate.
            for (const auto& f : s.runtime->flowSources()) {
                h = mix(h, f.enabled ? 1ull : 0ull);
                h = mix(h, static_cast<uint64_t>(f.source_mode));
                h = mix(h, static_cast<uint64_t>(static_cast<int64_t>(f.domain_index)));
                for (char ch : f.source_name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(ch)));
                h = mix(h, qf(f.position.x)); h = mix(h, qf(f.position.y)); h = mix(h, qf(f.position.z));
                h = mix(h, qf(f.velocity.x)); h = mix(h, qf(f.velocity.y)); h = mix(h, qf(f.velocity.z));
                h = mix(h, qf(f.radius));
                h = mix(h, qf(f.density));
                h = mix(h, qf(f.temperature));
                h = mix(h, qf(f.fuel));
                h = mix(h, qf(f.falloff));
                h = mix(h, qf(f.velocity_coupling));
                h = mix(h, qf(f.fluid_particles_per_second));
                h = mix(h, qf(f.fluid_velocity_spread));
                h = mix(h, f.fluid_emit_along_normal ? 1ull : 0ull);
                h = mix(h, static_cast<uint64_t>(
                    RayTrophiSim::Fluid::substanceTag(f.fluid_substance)));
                h = mix(h, f.use_time_limit ? 1ull : 0ull);
                h = mix(h, qf(f.start_time)); h = mix(h, qf(f.end_time));
                h = mix(h, f.use_particle_limit ? 1ull : 0ull);
                h = mix(h, static_cast<uint64_t>(f.max_emitted_particles));
                h = mix(h, f.keyframes.size());
                for (const auto& [frame, k] : f.keyframes) {
                    h = mix(h, static_cast<uint64_t>(static_cast<int64_t>(frame)));
                    h = mix(h, k.has_enabled ? 1ull : 0ull);
                    h = mix(h, k.has_position ? 1ull : 0ull);
                    h = mix(h, k.has_velocity ? 1ull : 0ull);
                    h = mix(h, k.has_radius ? 1ull : 0ull);
                    h = mix(h, k.has_density ? 1ull : 0ull);
                    h = mix(h, k.has_temperature ? 1ull : 0ull);
                    h = mix(h, k.has_fuel ? 1ull : 0ull);
                    h = mix(h, k.has_falloff ? 1ull : 0ull);
                    h = mix(h, k.has_velocity_coupling ? 1ull : 0ull);
                    h = mix(h, k.enabled ? 1ull : 0ull);
                    h = mix(h, qf(k.position.x)); h = mix(h, qf(k.position.y)); h = mix(h, qf(k.position.z));
                    h = mix(h, qf(k.velocity.x)); h = mix(h, qf(k.velocity.y)); h = mix(h, qf(k.velocity.z));
                    h = mix(h, qf(k.radius)); h = mix(h, qf(k.density));
                    h = mix(h, qf(k.temperature)); h = mix(h, qf(k.fuel));
                    h = mix(h, qf(k.falloff)); h = mix(h, qf(k.velocity_coupling));
                }
            }
        }
        h = mix(h, gas_volumes.size());
        h = mix(h, fluid_objects.size());
        // Force fields now drive rigid + soft/cloth bodies too, so editing a field
        // (strength / position / direction / masks) must invalidate the body bake.
        // Cheap: a handful of fields folded into the hash that already runs each
        // frame — no extra structure or pass. Force fields are not keyframed, so
        // hashing their LIVE pose only changes on a real user edit (no playback
        // thrash). Catches both panel edits and viewport gizmo drags.
        h = mix(h, force_field_manager.force_fields.size());
        for (const auto& ff : force_field_manager.force_fields) {
            if (!ff) { h = mix(h, 0); continue; }
            h = mix(h, ff->enabled ? 1ull : 0ull);
            h = mix(h, static_cast<uint64_t>(ff->type));
            h = mix(h, static_cast<uint64_t>(ff->shape));
            h = mix(h, static_cast<uint64_t>(ff->falloff_type));
            h = mix(h, qf(ff->strength));
            h = mix(h, qf(ff->position.x)); h = mix(h, qf(ff->position.y)); h = mix(h, qf(ff->position.z));
            h = mix(h, qf(ff->rotation.x)); h = mix(h, qf(ff->rotation.y)); h = mix(h, qf(ff->rotation.z));
            h = mix(h, qf(ff->scale.x)); h = mix(h, qf(ff->scale.y)); h = mix(h, qf(ff->scale.z));
            h = mix(h, qf(ff->direction.x)); h = mix(h, qf(ff->direction.y)); h = mix(h, qf(ff->direction.z));
            h = mix(h, qf(ff->falloff_radius)); h = mix(h, qf(ff->inner_radius));
            h = mix(h, qf(ff->axis.x)); h = mix(h, qf(ff->axis.y)); h = mix(h, qf(ff->axis.z));
            h = mix(h, qf(ff->inward_force)); h = mix(h, qf(ff->upward_force));
            h = mix(h, qf(ff->linear_drag)); h = mix(h, qf(ff->quadratic_drag));
            h = mix(h, ff->use_noise ? 1ull : 0ull);
            h = mix(h, qf(ff->noise.frequency)); h = mix(h, qf(ff->noise.amplitude));
            h = mix(h, qf(ff->noise.speed)); h = mix(h, static_cast<uint64_t>(ff->noise.octaves));
            // Wind→fluid surface-drag knobs (editing them must invalidate the bake).
            h = mix(h, ff->fluid_surface_drag ? 1ull : 0ull);
            h = mix(h, qf(ff->fluid_drag_coupling));
            h = mix(h, qf(ff->fluid_surface_depth));
            h = mix(h, qf(ff->fluid_curl_detail));
            h = mix(h, ff->affects_rigidbody ? 1ull : 0ull);
            h = mix(h, ff->affects_cloth ? 1ull : 0ull);
            h = mix(h, ff->affects_fluid ? 1ull : 0ull);
            h = mix(h, ff->affects_gas ? 1ull : 0ull);
            h = mix(h, ff->affects_particles ? 1ull : 0ull);
        }
        h = mix(h, rigid_bodies.size());
        for (const auto& rb : rigid_bodies) {
            for (char c : rb.source_name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(c)));
            for (char c : rb.collider_name) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(c)));
            if (rb.getBreakable()) {
                h = mix(h, 0xB4EAC0DEull);  // stable "breakable, authored static" marker
                h = mix(h, qf(rb.getBreakVelocity()));
                h = mix(h, rb.getIntegrityWeakening() ? 1ull : 0ull);
                h = mix(h, qf(rb.getIntegrityExponent()));
                h = mix(h, qf(rb.getMinimumThresholdScale()));
                for (char c : rb.getFractureGroup()) h = mix(h, static_cast<uint64_t>(static_cast<unsigned char>(c)));
            } else {
                h = mix(h, static_cast<uint64_t>(rb.motion_type));
                h = mix(h, rb.dynamic ? 1ull : 0ull);
            }
            h = mix(h, static_cast<uint64_t>(rb.shape));
            h = mix(h, qf(rb.mass));
            h = mix(h, rb.auto_mass_from_density ? 1ull : 0ull);
            h = mix(h, qf(rb.density));
            h = mix(h, qf(rb.linear_damping));
            h = mix(h, qf(rb.angular_damping));
            h = mix(h, qf(rb.gravity_scale));
            h = mix(h, qf(rb.friction));
            h = mix(h, qf(rb.restitution));
            h = mix(h, qf(rb.initial_linear_velocity.x));
            h = mix(h, qf(rb.initial_linear_velocity.y));
            h = mix(h, qf(rb.initial_linear_velocity.z));
            h = mix(h, qf(rb.initial_angular_velocity.x));
            h = mix(h, qf(rb.initial_angular_velocity.y));
            h = mix(h, qf(rb.initial_angular_velocity.z));
            h = mix(h, rb.sleep_enabled ? 1ull : 0ull);
            h = mix(h, rb.lock_translation_x ? 1ull : 0ull);
            h = mix(h, rb.lock_translation_y ? 1ull : 0ull);
            h = mix(h, rb.lock_translation_z ? 1ull : 0ull);
            h = mix(h, rb.lock_rotation_x ? 1ull : 0ull);
            h = mix(h, rb.lock_rotation_y ? 1ull : 0ull);
            h = mix(h, rb.lock_rotation_z ? 1ull : 0ull);
            h = mix(h, rb.fluid_coupling_enabled ? 1ull : 0ull);
            h = mix(h, qf(rb.getBuoyancyScale()));
            h = mix(h, qf(rb.getFluidDensity()));
            h = mix(h, qf(rb.getFluidDrag()));
            h = mix(h, qf(rb.getFluidQuadraticDrag()));
            h = mix(h, qf(rb.getFluidAngularDrag()));
            h = mix(h, rb.enabled ? 1ull : 0ull);
            // Force-field coupling knobs (drive every body kind).
            h = mix(h, rb.force_field_enabled ? 1ull : 0ull);
            h = mix(h, qf(rb.force_field_scale));
            // Cloth/soft pins: editing/adding/removing a pin must rebuild the body.
            h = mix(h, rb.getSoftPins().size());
            for (const auto& pin : rb.getSoftPins()) {
                h = mix(h, pin.enabled ? 1ull : 0ull);
                h = mix(h, qf(pin.radius));
                h = mix(h, qf(pin.center.x)); h = mix(h, qf(pin.center.y)); h = mix(h, qf(pin.center.z));
            }
            // Body kind + soft params: changing Rigid<->Soft/Cloth or editing any
            // soft authoring value must invalidate the bake (the deformation cache
            // is keyed off these). Cheap to fold in here.
            h = mix(h, static_cast<uint64_t>(rb.kind));
            if (rb.kind != RayTrophiSim::BodyKind::Rigid) {
                h = mix(h, qf(rb.getSoftStiffness()));
                h = mix(h, qf(rb.getSoftCompliance()));
                h = mix(h, qf(rb.getSoftPressure()));
                h = mix(h, qf(rb.getSoftDamping()));
                h = mix(h, qf(rb.getSoftVertexRadius()));
                h = mix(h, static_cast<uint64_t>(rb.getSoftIterations()));
                h = mix(h, qf(rb.getSoftFriction()));
                h = mix(h, qf(rb.getSoftRestitution()));
                h = mix(h, qf(rb.getSoftGravityFactor()));
                h = mix(h, qf(rb.getSoftMass()));
                h = mix(h, rb.getSoftTwoSided() ? 1ull : 0ull);
            }
            // CACHED spawn pose (rb.initial_pivot) so MOVING a rigid changes the
            // signature without a per-tick world scan — O(1) here. The cache is
            // refreshed only on a real user transform edit by
            // refreshRigidRestPosesOnUserEdit() (called from updateSimulationTimeline).
            {
                const Vec3 t = rb.initial_pivot.getTranslation();
                h = mix(h, qf(t.x)); h = mix(h, qf(t.y)); h = mix(h, qf(t.z));
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 4; ++c) h = mix(h, qf(rb.initial_pivot.m[r][c]));
            }
        }
        return h;
    }

    // Apply the keyframed transform of every object bound as a simulation source
    // (collider / emitter / grid domain / flow source) for an ARBITRARY timeline
    // frame, so the sim sees its animated pose at that exact frame.
    //
    // Why this is needed: the per-tick UI driver (TimelineWidget) and the
    // sequence-render worker (updateAnimationState) only ever apply the SINGLE
    // currently-displayed frame's pose. A sim bake, however, advances many
    // sub-steps per applied pose — a scrub catch-up (capped resim loop), a fresh
    // 0..N bake, or the sequence-render first frame (0..start_frame). Without
    // re-posing the source objects per sub-step, every step of that bake sees the
    // collider/emitter frozen at one pose, so a keyframed collider only interacts
    // with the fluid at that single position (the reported bug).
    //
    // Mirrors TimelineWidget's transform-apply: setPivotMatrix on the shared
    // transform handle. The SurfaceMeshCache the voxelizer reads computes world
    // verts as getTransformMatrix()*original, so updating the handle is enough for
    // moved geometry to reach the solid mask — no CPU vertex bake required.
    // True when a node is the source object of an ENABLED, DYNAMIC rigid body —
    // i.e. the rigid sim writes its pose every step and owns its transform.
    // Such nodes must not be re-posed by keyframe / serialize-cached tracks.
    bool isSimOwnedRigidSource(const std::string& name) const {
        if (name.empty()) return false;
        for (const auto& rb : rigid_bodies)
            if (rb.enabled && rb.dynamic && rb.source_name == name) return true;
        return false;
    }

    void applySimSourceObjectPosesForFrame(int frame) {
        if (timeline.tracks.empty() || particle_systems.empty()) return;

        // Unique node names referenced by any sim source across all systems.
        std::vector<std::string> source_names;
        auto addName = [&](const std::string& n) {
            if (n.empty()) return;
            if (std::find(source_names.begin(), source_names.end(), n) == source_names.end())
                source_names.push_back(n);
        };
        for (auto& system : particle_systems) {
            if (!system.runtime) continue;
            for (const auto& c : system.runtime->colliders())  addName(c.source_name);
            for (const auto& e : system.runtime->emitters())   addName(e.source_name);
            for (const auto& d : system.runtime->gridDomains()) addName(d.source_name);
            for (const auto& f : system.runtime->flowSources()) addName(f.source_name);
        }
        if (source_names.empty()) return;

        // Evaluate each source's transform track ONCE for this frame; drop names
        // with no transform track so the world.objects pass below stays cheap.
        // Also drop names whose evaluated pose is identical to what we last pushed
        // (same playhead AND no keyframe edit) — those need no re-push and, crucially,
        // no surface-cache memo erase, so this can run every idle frame for free.
        std::vector<std::string> posed_names;
        std::vector<Matrix4x4> posed_mats;
        posed_names.reserve(source_names.size());
        posed_mats.reserve(source_names.size());
        auto matrixEqual = [](const Matrix4x4& a, const Matrix4x4& b) {
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    if (a.m[r][c] != b.m[r][c]) return false;
            return true;
        };
        for (const auto& name : source_names) {
            // A DYNAMIC rigid body owns its source object's transform — the rigid
            // sim writes the simulated pose every step. If we ALSO pushed a keyframe
            // (or a serialize-cached frame-0 pose) onto it here, the two drivers
            // fight and the object flickers between the authored and the simulated
            // pose ("two places at once" vibration). Kinematic/static bodies are
            // keyframe-driven and must still be posed, so gate on rb.dynamic only.
            if (isSimOwnedRigidSource(name)) continue;
            auto track_it = timeline.tracks.find(name);
            if (track_it == timeline.tracks.end() || track_it->second.keyframes.empty()) continue;
            Keyframe kf = track_it->second.evaluate(frame);
            if (!kf.has_transform) continue;
            Matrix4x4 mat = Matrix4x4::fromTRS(kf.transform.position,
                                               kf.transform.rotation,
                                               kf.transform.scale);
            auto prev = last_sim_pose_applied_.find(name);
            if (prev != last_sim_pose_applied_.end() && matrixEqual(prev->second, mat)) {
                continue;  // pose unchanged since last push — nothing to do
            }
            last_sim_pose_applied_[name] = mat;
            posed_names.push_back(name);
            posed_mats.push_back(mat);
        }
        if (posed_names.empty()) return;

        // Single pass over world.objects; transform handles are shared per mesh,
        // but set on every matching triangle to stay correct if they aren't.
        //
        // The evaluated keyframe matrix is an ABSOLUTE object pose. Transform's
        // final contract is current * base; leaving a previous animation/current
        // matrix alive and then writing the same absolute pose into base applies
        // translation twice when play starts. Besides shifting analytical
        // colliders by +delta, this made SDF diverge in the opposite direction
        // because its sampling path uses inverse(final). Reset current before
        // installing the absolute base/pivot pose.
        for (auto& obj : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri) {
                const std::string& nn = tri->getNodeName();
                if (nn.empty()) continue;
                for (std::size_t i = 0; i < posed_names.size(); ++i) {
                    if (posed_names[i] != nn) continue;
                    if (Transform* th = tri->getTransformPtr()) {
                        th->setCurrent(Matrix4x4::identity());
                        th->setPivotMatrix(posed_mats[i]);
                    }
                    break;
                }
                continue;
            }
            auto flat_mesh = std::dynamic_pointer_cast<TriangleMesh>(obj);
            if (flat_mesh && flat_mesh->transform) {
                const std::string& nn = flat_mesh->nodeName;
                for (std::size_t i = 0; i < posed_names.size(); ++i) {
                    if (posed_names[i] != nn) continue;
                    flat_mesh->transform->setCurrent(Matrix4x4::identity());
                    flat_mesh->transform->setPivotMatrix(posed_mats[i]);
                    break;
                }
            }
        }
        // These objects just moved without bumping g_scene_geometry_generation, so
        // drop them from the surface-cache epoch memo — their next resolve must
        // rebuild from the new world verts. Static (un-posed) objects keep their
        // memo and stay cheap.
        for (const auto& name : posed_names) surface_cache_epoch_done_.erase(name);

        // Cache reset/restore can reach this outside TimelineWidget's normal
        // backend transform-update path. Keep the rendered object and the
        // collider resolver on the same newly-applied CPU pose.
        g_gpu_refit_pending = true;
        g_bvh_rebuild_pending = true;
    }

    bool hasSimFrame(int frame) const {
        return sim_frame_cache_.find(frame) != sim_frame_cache_.end();
    }

    // ── Bake / cache state, read-only (node layer Faz N6) ────────────────────
    // ★ A script driving the simulation has to be able to tell three states
    // apart that all look like "nothing is cached": nothing baked yet, a bake in
    // progress, and a bake INVALIDATED because the authored config changed. The
    // last one is the interesting one and it is invisible without the signature.
    bool simCacheValid() const { return sim_cache_valid_; }
    const std::string& simCacheDir() const { return sim_cache_dir_; }
    bool simBakeActive() const { return sim_bake_active_; }
    uint64_t simConfigSignature() const { return last_sim_config_sig_; }
    std::size_t simFrameCacheCount() const { return sim_frame_cache_.size(); }
    bool simFrameCacheRange(int& out_first, int& out_last) const {
        if (sim_frame_cache_.empty()) return false;
        out_first = sim_frame_cache_.begin()->first;
        out_last = sim_frame_cache_.rbegin()->first;
        return true;
    }

    int nearestCachedSimFrameAtOrBelow(int frame) const {
        int best = -1;
        for (const auto& kv : sim_frame_cache_) {
            if (kv.first <= frame && kv.first > best) best = kv.first;
        }
        return best;
    }

    // Rough RAM footprint of the in-memory sim frame caches, for the UI "cache is
    // getting big — bake to disk" nudge. Covers the per-frame body (soft/cloth verts)
    // + particle (SoA columns) + rigid (poses) snapshots — the ones that balloon with
    // crowded/long scenes. Fluid/gas GRID states (sim_frame_cache_) are NOT included:
    // their per-cell size isn't cheaply known here, so this under-reports pure-fluid
    // scenes (a disk bake is recommended there regardless).
    std::size_t estimateSimCacheBytes() const {
        std::size_t bytes = 0;
        for (const auto& f : soft_frame_cache_)
            for (const auto& n : f.second)
                bytes += n.second.size() * sizeof(Vec3);
        for (const auto& f : particle_frame_cache_)
            for (const auto& snap : f.second)
                bytes += snap.buffers.position_x.size() * 80;  // ~20 float columns + flags
        for (const auto& f : rigid_frame_cache_)
            bytes += f.second.size() * sizeof(RayTrophiSim::RigidBodyFrameState);
        // MSF is six float channels per surface ELEMENT, and elements are the
        // texels of the char mask — a single 128x128 mask is ~6k elements, i.e.
        // ~150 KB per object per frame. That balloons faster than anything else
        // here, so it belongs in the "bake to disk" nudge rather than hiding.
        for (const auto& f : msf_frame_cache_)
            for (const auto& per_system : f.second)
                for (const auto& snap : per_system)
                    bytes += static_cast<std::size_t>(snap.element_count) * 6u * sizeof(float);
        return bytes;
    }
    int cachedSimFrameCount() const { return static_cast<int>(sim_frame_cache_.size()); }
    // The UI uses this only to decide whether a paused timeline scrub is about
    // to replace GPU-visible simulation buffers. It is not a second playhead.
    int simulationTimelineFrame() const { return sim_timeline_frame_; }

    // Scatter the solver's UNIQUE deformed world vertices back onto every triangle
    // corner of a soft body's mesh. The GPU BLAS reads LOCAL vertices
    // (getOriginalVertexPosition) + the instance transform, so the deformation goes
    // into `original` (= inverse(transform) * world); `position` (world) is set too
    // for the CPU/world-space paths. Flat per-triangle normals in each space.
    // (Re)build the weld topology cache for a soft body's source mesh: gather its
    // triangles, weld corners by REST world position (~0.1 mm), and record the
    // corner->unique map, the unique rest world positions (Jolt seed), and the bind-
    // pose local pos/normals (for reset). Used by the resolver AND by disk replay
    // (where the body was never live-created, so no cache exists yet). Returns false
    // if the mesh isn't available or is degenerate.
    bool rebuildSoftWeldCache(const std::string& node) {
        return rebuildWeldCache(node, soft_weld_cache_);
    }

    // ★ The cache map is a PARAMETER so Phase 6c can weld a melting object without
    // filing it under soft_weld_cache_. A non-soft node living in that map would
    // be seen by the soft body's freeze/reset/frame-cache paths as a soft body —
    // the same reason rigid_bake_cache_ was kept separate. Same welder, separate
    // ledger.
    bool rebuildWeldCache(const std::string& node,
                          std::unordered_map<std::string, SoftWeldCache>& cache_map) {
        if (node.empty()) return false;

        extern std::atomic<uint64_t> g_scene_geometry_generation;
        const uint64_t current_gen = g_scene_geometry_generation.load(std::memory_order_acquire);

        // Flat (direct SoA) node: no per-face facades, so the facade weld below would find no
        // triangles and bail (soft/cloth body never created, no pins). We must still WELD by rest
        // position: a flat mesh from facadesToFlatMesh is an UNWELDED soup (indices[v]=v, vc=3*tris),
        // and handing that to the cloth solver makes every triangle 3 free particles (no shared edge
        // = no constraint = unpinned verts free-fall, no collision — the reported bug). The welded
        // unique set drives the solver; flat_soa_to_unique scatters its result back to every duplicate
        // SoA vertex. flat_rest_pos/nrm keep the per-SoA-vertex authored rest for the reset path.
        if (TriangleMesh* fm = getFlatNodeMesh(node)) {
            if (!fm->geometry) return false;
            DNA::GeometryDetail* g = fm->geometry.get();
            const size_t vc = g->get_vertex_count();
            const auto& idx = g->indices;
            if (vc == 0 || idx.size() < 3) return false;

            const Matrix4x4 xf = fm->transform ? fm->transform->getFinal() : Matrix4x4::identity();
            const Vec3* Po = g->get_attribute_data<Vec3>("P_orig");
            if (!Po) return false;

            auto it = cache_map.find(node);
            const bool have_rest = (it != cache_map.end() &&
                                    it->second.flat_mesh == fm &&
                                    it->second.flat_rest_pos.size() == vc &&
                                    it->second.flat_soa_to_unique.size() == vc &&
                                    !it->second.rest_world_unique.empty());
            if (have_rest) {
                SoftWeldCache& cache = it->second;
                if (cache.geometry_generation == current_gen) return true; // unchanged — reuse
                // Topology unchanged, generation bumped (almost always a sim write-back): keep the
                // AUTHORED rest local + weld, only refresh the rest WORLD seed from the current
                // transform — re-deriving rest from the now-deformed live SoA would freeze the
                // deformed frame in as the new rest (the soft-edit-at-frame-N corruption).
                cache.geometry_generation = current_gen;
                for (size_t v = 0; v < vc; ++v) {
                    const uint32_t u = cache.flat_soa_to_unique[v];
                    if (u < cache.rest_world_unique.size())
                        cache.rest_world_unique[u] = xf.transform_point(cache.flat_rest_pos[v]);
                }
                return cache.unique_count >= 3;
            }

            const Vec3* No = g->get_attribute_data<Vec3>("N_orig");
            SoftWeldCache cache;
            cache.geometry_generation = current_gen;
            cache.flat_mesh = fm;
            cache.flat_rest_pos.assign(Po, Po + vc);
            if (No) cache.flat_rest_nrm.assign(No, No + vc);

            // Weld every SoA corner to a shared unique vertex by quantized rest WORLD position.
            std::map<std::array<int64_t, 3>, uint32_t> weld;
            const double kQuant = 10000.0;  // ~0.1 mm tolerance (matches the facade weld)
            cache.flat_soa_to_unique.assign(vc, 0);
            std::vector<uint8_t> seen(vc, 0);
            cache.corner_unique.reserve(idx.size());
            for (std::size_t k = 0; k < idx.size(); ++k) {
                const uint32_t soa_vid = idx[k];
                if (soa_vid >= vc) { cache.corner_unique.push_back(0); continue; }
                uint32_t u;
                if (seen[soa_vid]) {
                    u = cache.flat_soa_to_unique[soa_vid];
                } else {
                    const Vec3 rest = xf.transform_point(Po[soa_vid]);
                    const std::array<int64_t, 3> key{
                        (int64_t)std::llround((double)rest.x * kQuant),
                        (int64_t)std::llround((double)rest.y * kQuant),
                        (int64_t)std::llround((double)rest.z * kQuant)};
                    auto wit = weld.find(key);
                    if (wit == weld.end()) {
                        u = (uint32_t)cache.rest_world_unique.size();
                        cache.rest_world_unique.push_back(rest);
                        weld.emplace(key, u);
                    } else {
                        u = wit->second;
                    }
                    cache.flat_soa_to_unique[soa_vid] = u;
                    seen[soa_vid] = 1;
                }
                cache.corner_unique.push_back(u);
            }
            cache.unique_count = cache.rest_world_unique.size();
            const bool ok = cache.unique_count >= 3;
            cache_map[node] = std::move(cache);
            return ok;
        }

        std::size_t current_tri_count = 0;
        for (const auto& obj : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->getNodeName() == node) current_tri_count++;
        }

        auto it = cache_map.find(node);
        const bool have_rest = (it != cache_map.end() &&
                                !it->second.rest_local_pos.empty() &&
                                it->second.tris.size() == current_tri_count);
        if (have_rest && it->second.geometry_generation == current_gen) {
            return true; // Nothing changed since capture — reuse as-is.
        }
        if (have_rest) {
            // Topology is UNCHANGED but the geometry generation moved. That bump is
            // almost always a SIM deformation write-back (or a reset) — and the live
            // `original` verts now hold the CURRENT DEFORMED shape. Re-deriving the rest
            // from them would freeze that deformed frame in as the new "rest", so editing
            // a body param / adding a force at frame N made frame N the baseline and
            // frame 0 stopped returning to the original (the reported soft/cloth bug;
            // rigid was immune because rigid_bake_cache_ is captured once and never
            // re-derived). Keep the AUTHORED rest_local + topology; only refresh the rest
            // WORLD seed (Jolt) from rest_local * the CURRENT transform so moving the
            // object before play still relocates the soft body. (A genuine REST-mesh edit
            // changes the triangle COUNT or drops the cache; pure vertex edits of a soft
            // rest aren't picked up here — acceptable vs. the deformation-corruption bug.)
            SoftWeldCache& cache = it->second;
            cache.geometry_generation = current_gen;
            cache.rest_world_unique.assign(cache.unique_count, Vec3(0.0f, 0.0f, 0.0f));
            std::size_t corner = 0;
            for (std::size_t t = 0; t < cache.tris.size(); ++t) {
                const Matrix4x4 xf = cache.tris[t] ? cache.tris[t]->getTransformMatrix()
                                                   : Matrix4x4::identity();
                const bool has_lp = (t < cache.rest_local_pos.size());
                for (int i = 0; i < 3; ++i, ++corner) {
                    if (!has_lp || corner >= cache.corner_unique.size()) continue;
                    const uint32_t u = cache.corner_unique[corner];
                    if (u < cache.unique_count)
                        cache.rest_world_unique[u] = xf.transform_point(cache.rest_local_pos[t][i]);
                }
            }
            return cache.unique_count >= 3;
        }

        SoftWeldCache cache;
        cache.geometry_generation = current_gen;
        for (auto& obj : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->getNodeName() == node) cache.tris.push_back(tri);
        }
        if (cache.tris.empty()) return false;

        std::map<std::array<int64_t, 3>, uint32_t> weld;  // quantized rest pos -> idx
        const double kQuant = 10000.0;  // ~0.1 mm weld tolerance
        cache.corner_unique.reserve(cache.tris.size() * 3);
        cache.rest_local_pos.reserve(cache.tris.size());
        cache.rest_local_nrm.reserve(cache.tris.size());
        for (auto& tri : cache.tris) {
            const Matrix4x4 xf = tri->getTransformMatrix();
            std::array<Vec3, 3> lp, ln;
            for (int i = 0; i < 3; ++i) {
                lp[i] = tri->getOriginalVertexPosition(i);
                ln[i] = tri->getOriginalVertexNormal(i);
                const Vec3 rest = xf.transform_point(tri->getOriginalVertexPosition(i));
                const std::array<int64_t, 3> key{
                    (int64_t)std::llround((double)rest.x * kQuant),
                    (int64_t)std::llround((double)rest.y * kQuant),
                    (int64_t)std::llround((double)rest.z * kQuant)};
                uint32_t idx;
                auto it = weld.find(key);
                if (it == weld.end()) {
                    idx = (uint32_t)cache.rest_world_unique.size();
                    cache.rest_world_unique.push_back(rest);
                    weld.emplace(key, idx);
                } else {
                    idx = it->second;
                }
                cache.corner_unique.push_back(idx);
            }
            cache.rest_local_pos.push_back(lp);
            cache.rest_local_nrm.push_back(ln);
        }
        cache.unique_count = cache.rest_world_unique.size();
        const bool ok = cache.unique_count >= 3;
        cache_map[node] = std::move(cache);
        return ok;
    }

    // Snapshot every soft body's deformed UNIQUE world vertices (read from the
    // meshes the writer just updated). Shared by the in-memory capture and disk bake.
    void snapshotSoftBodies(std::map<std::string, std::vector<Vec3>>& out) const {
        out.clear();
        for (const auto& kv : soft_weld_cache_) {
            const SoftWeldCache& cache = kv.second;
            if (cache.unique_count == 0) continue;
            // Flat (direct SoA) soft body: the writer set P (world) per SoA vertex; gather the WELDED
            // unique world verts back via flat_soa_to_unique (duplicates share one unique slot).
            if (cache.flat_mesh && cache.flat_mesh->geometry) {
                const Vec3* P = cache.flat_mesh->geometry->get_attribute_data<Vec3>("P");
                const size_t vc = cache.flat_mesh->geometry->get_vertex_count();
                if (P && cache.flat_soa_to_unique.size() == vc) {
                    std::vector<Vec3> uniq(cache.unique_count, Vec3(0.0f, 0.0f, 0.0f));
                    for (size_t v = 0; v < vc; ++v) {
                        const uint32_t u = cache.flat_soa_to_unique[v];
                        if (u < uniq.size()) uniq[u] = P[v];
                    }
                    out[kv.first] = std::move(uniq);
                }
                continue;
            }
            std::vector<Vec3> uniq(cache.unique_count, Vec3(0.0f, 0.0f, 0.0f));
            std::size_t corner = 0;
            for (const auto& tri : cache.tris) {
                if (!tri) { corner += 3; continue; }
                for (int i = 0; i < 3; ++i, ++corner) {
                    const uint32_t u = cache.corner_unique[corner];
                    if (u < uniq.size()) uniq[u] = tri->getVertexPosition(i);  // world (writer set this)
                }
            }
            out[kv.first] = std::move(uniq);
        }
    }

    void applySoftDeformedVerts(const std::string& node, const std::vector<Vec3>& world_verts) {
        auto it = soft_weld_cache_.find(node);
        if (it == soft_weld_cache_.end()) return;
        applyDeformedVertsToCache(it->second, node, world_verts);
    }

    // Write welded world vertices back onto a mesh: positions (world + local),
    // area-weighted SMOOTH normals accumulated on the welded topology, and the
    // geometry-dirty flag. Split out of applySoftDeformedVerts so Phase 6c melt
    // displacement reuses the exact same writer instead of copying it — two
    // writers would be two places for the local/world or normal handling to drift.
    void applyDeformedVertsToCache(SoftWeldCache& cache,
                                   const std::string& node,
                                   const std::vector<Vec3>& world_verts) {
        if (world_verts.size() != cache.unique_count) return;  // stale topology

        // Flat (direct SoA) soft body: the solver deforms the WELDED unique verts; scatter each one
        // back onto every duplicate SoA vertex via flat_soa_to_unique. Write P (world) + P_orig
        // (local), with area-weighted smooth normals accumulated on the WELDED topology
        // (corner_unique) so shared verts shade smooth. No facades, so we never touch cache.tris.
        if (cache.flat_mesh && cache.flat_mesh->geometry) {
            DNA::GeometryDetail* g = cache.flat_mesh->geometry.get();
            const size_t vc = g->get_vertex_count();
            if (cache.flat_soa_to_unique.size() != vc) return;
            Vec3* Po = g->get_attribute_data_mut<Vec3>("P_orig");
            Vec3* P  = g->get_attribute_data_mut<Vec3>("P");
            Vec3* No = g->get_attribute_data_mut<Vec3>("N_orig");
            Vec3* N  = g->get_attribute_data_mut<Vec3>("N");
            const Matrix4x4 xf = cache.flat_mesh->transform ? cache.flat_mesh->transform->getFinal()
                                                            : Matrix4x4::identity();
            const Matrix4x4 inv_xf = xf.inverse();

            // Per-unique local positions + area-weighted smooth normals (world + local).
            const size_t uc = cache.unique_count;
            std::vector<Vec3> uniq_local(uc);
            for (size_t u = 0; u < uc; ++u) uniq_local[u] = inv_xf.transform_point(world_verts[u]);
            std::vector<Vec3> nw(uc, Vec3(0.0f, 0.0f, 0.0f)), nl(uc, Vec3(0.0f, 0.0f, 0.0f));
            auto cross = [](const Vec3& a, const Vec3& b) {
                return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
            };
            const auto& cu = cache.corner_unique;
            for (size_t t = 0; t + 2 < cu.size(); t += 3) {
                const uint32_t a = cu[t], b = cu[t + 1], c = cu[t + 2];
                if (a >= uc || b >= uc || c >= uc) continue;
                const Vec3 fnw = cross(world_verts[b] - world_verts[a], world_verts[c] - world_verts[a]);
                const Vec3 fnl = cross(uniq_local[b] - uniq_local[a], uniq_local[c] - uniq_local[a]);
                nw[a] += fnw; nw[b] += fnw; nw[c] += fnw;
                nl[a] += fnl; nl[b] += fnl; nl[c] += fnl;
            }
            auto norm = [](const Vec3& v, const Vec3& fb) {
                const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
                return (len > 1e-12f) ? Vec3(v.x / len, v.y / len, v.z / len) : fb;
            };
            for (size_t u = 0; u < uc; ++u) {
                nw[u] = norm(nw[u], Vec3(0.0f, 1.0f, 0.0f));
                nl[u] = norm(nl[u], Vec3(0.0f, 1.0f, 0.0f));
            }

            // Scatter the welded result onto every SoA vertex.
            for (size_t v = 0; v < vc; ++v) {
                const uint32_t u = cache.flat_soa_to_unique[v];
                if (u >= uc) continue;
                if (P)  P[v]  = world_verts[u];
                if (Po) Po[v] = uniq_local[u];
                if (N)  N[v]  = nw[u];
                if (No) No[v] = nl[u];
            }
            markBodyGeometryDirty(node);
            return;
        }

        const Transform* last_xf = nullptr;
        bool inv_valid = false;
        Matrix4x4 inv_xf = Matrix4x4::identity();

        // Pass 1: write positions (world `position` + local `original` the BLAS reads)
        // and accumulate AREA-WEIGHTED face normals per shared vertex so the surface
        // shades SMOOTH (welded corners share a normal) instead of faceted flat.
        std::vector<Vec3> nw_acc(cache.unique_count, Vec3(0.0f, 0.0f, 0.0f));
        std::vector<Vec3> nl_acc(cache.unique_count, Vec3(0.0f, 0.0f, 0.0f));
        std::size_t corner = 0;
        for (auto& tri : cache.tris) {
            if (!tri) { corner += 3; continue; }
            const Transform* xfp = tri->getTransformPtr();
            if (!inv_valid || xfp != last_xf) {
                inv_xf = tri->getTransformMatrix().inverse();
                last_xf = xfp;
                inv_valid = true;
            }
            uint32_t u[3];
            Vec3 wp[3], lp[3];
            for (int i = 0; i < 3; ++i, ++corner) {
                u[i] = cache.corner_unique[corner];
                wp[i] = (u[i] < world_verts.size()) ? world_verts[u[i]] : tri->getVertexPosition(i);
                lp[i] = inv_xf.transform_point(wp[i]);
                tri->setVertexPosition(i, wp[i]);
                tri->setOriginalVertexPosition(i, lp[i]);
            }
            // Unnormalized cross == 2*area*unit_normal => area weighting for free.
            auto cross = [](const Vec3& a, const Vec3& b) {
                return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
            };
            const Vec3 fnw = cross(wp[1] - wp[0], wp[2] - wp[0]);
            const Vec3 fnl = cross(lp[1] - lp[0], lp[2] - lp[0]);
            for (int i = 0; i < 3; ++i) {
                if (u[i] < cache.unique_count) { nw_acc[u[i]] += fnw; nl_acc[u[i]] += fnl; }
            }
        }
        auto norm = [](const Vec3& v, const Vec3& fallback) {
            const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
            return (len > 1e-12f) ? Vec3(v.x / len, v.y / len, v.z / len) : fallback;
        };
        for (std::size_t u = 0; u < cache.unique_count; ++u) {
            nw_acc[u] = norm(nw_acc[u], Vec3(0.0f, 1.0f, 0.0f));
            nl_acc[u] = norm(nl_acc[u], Vec3(0.0f, 1.0f, 0.0f));
        }

        // Pass 2: assign the smooth (shared) normal to each corner.
        corner = 0;
        for (auto& tri : cache.tris) {
            if (!tri) { corner += 3; continue; }
            for (int i = 0; i < 3; ++i, ++corner) {
                const uint32_t uu = cache.corner_unique[corner];
                if (uu < cache.unique_count) {
                    tri->setVertexNormal(i, nw_acc[uu]);
                    tri->setOriginalVertexNormal(i, nl_acc[uu]);
                }
            }
        }
        markBodyGeometryDirty(node);
    }

    // ── Phase 6c: melt geometry (slump) ──────────────────────────────────────
    //
    // ★ The displacement is DERIVED from `melt`, never accumulated. That single
    // choice pays for itself three times over:
    //   - `melt` is monotonic (Phase 6b only ever raises it), so a derived
    //     displacement is monotonic too — melting still reads as irreversible.
    //   - `melt` is already in the Phase 4b frame cache, so scrubbing the timeline
    //     replays the geometry for free. No new cache, no new serialization.
    //   - Clear Damage sets melt to 0 and the mesh returns to rest by itself.
    // Accumulating into the live vertices would have needed all three built by
    // hand, and would have compounded every frame the sim ran at a different dt.
    //
    // ★ SCOPE, stated plainly: this is quasi-static molten-shell FLOW. The liquid
    // share of each vertex's remaining material is transported downhill through
    // triangle connectivity, conserving volume exactly, and the surface sits at
    // whatever thickness the material under it adds up to. Because that thickness
    // is driven by the MSF mass fractions, the geometric volume the mesh loses
    // EQUALS the mass that actually left it — one number, not two tunings that
    // drift apart. (It was previously a slump approximation whose volume had to be
    // reconciled with APIC's by hand, and never quite was.)
    //
    // Still out of scope, and belonging to the APIC bridge: topology loss,
    // detached droplets, and pooling against anything other than the object's own
    // rest floor.
    void applyMeltDisplacement() {
        if (defer_melt_displacement_once_) {
            defer_melt_displacement_once_ = false;
            return;
        }
        for (auto& system : particle_systems) {
            if (!system.runtime || !system.runtime->hasMaterialStateFields()) continue;
            for (const auto& entry : system.runtime->materialStateFields()) {
                applyMeltDisplacementToNode(entry.first, entry.second, system.runtime);
            }
        }
    }

    void applyMeltDisplacementToNode(const std::string& node,
                                     const RayTrophiSim::MaterialStateField& field,
                                     const std::shared_ptr<RayTrophiSim::ParticleSimulationSystem>& runtime) {
        if (!runtime) return;
        if (node.empty()) return;
        // ★ Two writers to one mesh would fight every frame, each overwriting the
        // other's result. A soft body already owns its vertices; melt defers.
        if (soft_weld_cache_.count(node) != 0) return;

        const auto stamp_it = melt_applied_stamp_.find(node);
        if (stamp_it != melt_applied_stamp_.end() &&
            stamp_it->second.topology == field.topology_generation &&
            stamp_it->second.content == field.mask_revision) {
            return;
        }

        const bool was_displaced = melt_displaced_.count(node) != 0;
        float peak_melt = 0.0f;
        for (float m : field.melt_texel) peak_melt = std::max(peak_melt, m);
        // Nothing melted and nothing to put back: the common case, and it must
        // cost no more than this scan.
        if (peak_melt <= 0.0f && !was_displaced) {
            melt_applied_stamp_[node] = { field.topology_generation, field.mask_revision };
            return;
        }
        // No UV layout: Phase 6a can answer no query for this object, so there is
        // nothing to displace FROM. Leaving the mesh alone is the honest outcome.
        if (field.mask_resolution <= 0) return;

        if (!rebuildWeldCache(node, melt_weld_cache_)) return;
        auto cit = melt_weld_cache_.find(node);
        if (cit == melt_weld_cache_.end()) return;
        SoftWeldCache& cache = cit->second;
        const std::size_t uc = cache.unique_count;
        if (uc < 3 || cache.rest_world_unique.size() != uc) return;

        // ── Per-object opt-out, evaluated BEFORE any work ─────────────────────
        // ★ This gate used to sit at the bottom, after the whole UV sample pass,
        // and it was a bare `return`. Two consequences, both wrong:
        //   - the sampling cost was paid every revision even with the feature off;
        //   - turning the switch off on an object that was ALREADY displaced left
        //     the melted shape frozen on screen forever. melt_displaced_ still
        //     claimed it was deformed, the stamp was never advanced, and nothing
        //     downstream ever put the rest pose back.
        // Disabling a visual option must undo its geometry, exactly as Clear
        // Damage does — the MSF chemistry underneath is untouched either way,
        // which is the whole point of the "performance options do not silently
        // destroy physical state" rule.
        RayTrophiSim::ParticleColliderDesc* authoring = nullptr;
        for (auto& collider : runtime->colliders()) {
            if (collider.source_name == node) { authoring = &collider; break; }
        }
        if (authoring && !authoring->msf_melt_flow_enabled) {
            if (was_displaced) {
                applyDeformedVertsToCache(cache, node, cache.rest_world_unique);
                melt_displaced_.erase(node);
            }
            melt_applied_stamp_[node] = { field.topology_generation,
                                          field.mask_revision };
            return;
        }

        // A high-poly source otherwise rewrites all positions/normals and refits
        // its BLAS on every MSF mask revision while the APIC surface BLAS changes
        // too. Keep chemistry/fluid at full rate, but budget visual mesh refits.
        // Rewinds (revision decreases) bypass the cadence and restore exactly.
        const std::size_t triangle_count = cache.corner_unique.size() / 3u;
        const uint64_t refit_interval = std::clamp<uint64_t>(
            1u + triangle_count / 50000u, 1u, 8u);
        if (stamp_it != melt_applied_stamp_.end() &&
            stamp_it->second.topology == field.topology_generation &&
            field.mask_revision > stamp_it->second.content &&
            field.mask_revision - stamp_it->second.content < refit_interval) {
            return;
        }

        // ── Melt per UNIQUE vertex ────────────────────────────────────────────
        // ★★ THE trap of this phase: a UV seam is exactly where one spatial vertex
        // carries two different UVs. Sampling per SoA vertex and displacing each
        // one by its own answer tears the mesh open along every seam. So sample
        // per SoA vertex (each has its own UV, which is the point) and REDUCE to
        // the unique vertex with max — the same rule the texel scatter uses at a
        // seam, so the two agree.
        std::vector<float> melt_unique(uc, 0.0f);
        std::vector<float> local_mass_unique(uc, 1.0f);
        // ★ Which vertices the field could actually answer for. Without this the
        // solver cannot tell "melted none" from "no reading", and a vertex whose
        // UV lands between islands stands up as a spike while its neighbours sink.
        std::vector<uint8_t> sampled_unique(uc, 0u);
        bool sampled_any = false;

        if (cache.flat_mesh && cache.flat_mesh->geometry &&
            cache.flat_soa_to_unique.size() == cache.flat_mesh->geometry->get_vertex_count()) {
            const DNA::GeometryDetail* g = cache.flat_mesh->geometry.get();
            const Vec2* uv = g->get_uvs();
            if (!uv) return;  // welded, but unwrapped: same "cannot displace" case
            const std::size_t vc = g->get_vertex_count();
            for (std::size_t v = 0; v < vc; ++v) {
                const uint32_t u = cache.flat_soa_to_unique[v];
                if (u >= uc) continue;
                float m = 0.0f;
                if (!RayTrophiSim::MaterialStateFieldSystem::sampleMeltAtUV(
                        field, uv[v].x, uv[v].y, m)) continue;
                melt_unique[u] = std::max(melt_unique[u], m);
                float local_mass = 1.0f;
                if (RayTrophiSim::MaterialStateFieldSystem::sampleLocalMassAtUV(
                        field, uv[v].x, uv[v].y, local_mass))
                    local_mass_unique[u] = std::min(local_mass_unique[u], local_mass);
                sampled_unique[u] = 1u;
                sampled_any = true;
            }
        } else {
            std::size_t corner = 0;
            for (const auto& tri : cache.tris) {
                if (!tri) { corner += 3; continue; }
                const auto uvs = tri->getUVCoordinates();
                const Vec2 c_uv[3] = { std::get<0>(uvs), std::get<1>(uvs), std::get<2>(uvs) };
                for (int i = 0; i < 3; ++i, ++corner) {
                    if (corner >= cache.corner_unique.size()) break;
                    const uint32_t u = cache.corner_unique[corner];
                    if (u >= uc) continue;
                    float m = 0.0f;
                    if (!RayTrophiSim::MaterialStateFieldSystem::sampleMeltAtUV(
                            field, c_uv[i].x, c_uv[i].y, m)) continue;
                    melt_unique[u] = std::max(melt_unique[u], m);
                    float local_mass = 1.0f;
                    if (RayTrophiSim::MaterialStateFieldSystem::sampleLocalMassAtUV(
                            field, c_uv[i].x, c_uv[i].y, local_mass))
                        local_mass_unique[u] = std::min(local_mass_unique[u], local_mass);
                    sampled_unique[u] = 1u;
                    sampled_any = true;
                }
            }
        }
        if (!sampled_any && !was_displaced) return;

        // ── Slump ─────────────────────────────────────────────────────────────
        // Sag scales with the object's own size so the same substance behaves the
        // same on a coffee cup and on a cathedral, and with (1 - melt_viscosity)
        // so the substance library's authored viscosity finally does something.
        // No new UI parameter, and therefore no script/IPC parity surface.
        const RayTrophiSim::SubstanceProfile& prof =
            RayTrophiSim::findSubstance(field.substance_name);
        std::vector<Vec3> displaced;
        RayTrophiSim::MeltSurfaceFlowSettings flow_settings;
        flow_settings.viscosity = prof.melt_viscosity;
        if (authoring) {
            flow_settings.maximum_height_loss = authoring->msf_melt_height_loss;
            flow_settings.maximum_lateral_gain = authoring->msf_melt_spread;
        }
        if (!RayTrophiSim::solveMeltSurfaceFlow(cache.rest_world_unique,
                cache.corner_unique, melt_unique, local_mass_unique,
                sampled_unique, flow_settings, displaced)) return;
        applyDeformedVertsToCache(cache, node, displaced);
        if (peak_melt > 0.0f) melt_displaced_[node] = 1u;
        else melt_displaced_.erase(node);
        melt_applied_stamp_[node] = { field.topology_generation, field.mask_revision };

        // Disabled: repeated melt-driven ObjectMeshSDF recooks raced Vulkan RT
        // geometry consumption during rewind. The explicit Force Rebuild SDF
        // action is now the only authoritative refresh path.
        // The render mesh above is live geometry, while an ObjectMeshSDF is a
        // cooked snapshot. Refresh only after meaningful thermal change and a
        // revision interval; otherwise a melting object would start an expensive
        // 3D cook every readback/frame. Reset-to-rest is always allowed through.
        if (false && authoring && authoring->source_mode ==
                RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF &&
            authoring->msf_melt_sdf_refresh) {
            float mean_melt = 0.0f;
            for (float m : melt_unique) mean_melt += std::clamp(m, 0.0f, 1.0f);
            mean_melt /= std::max<std::size_t>(1u, melt_unique.size());
            auto& refresh = melt_sdf_refresh_stamp_[node];
            const bool now_displaced = peak_melt > 0.0f;
            const uint64_t interval = std::max<uint32_t>(1u,
                authoring->msf_melt_sdf_revision_interval);
            const bool interval_due = field.mask_revision >= refresh.revision + interval;
            const bool shape_due = std::abs(mean_melt - refresh.mean_melt) >=
                std::max(0.001f, authoring->msf_melt_sdf_change_threshold);
            const bool restoring = refresh.displaced && !now_displaced;
            if ((restoring || (interval_due && shape_due))) {
                invalidateSurfaceMeshCache(node);
                if (rebuildSDFColliderAsync(*authoring, runtime, true)) {
                    refresh = { field.mask_revision, mean_melt, now_displaced };
                }
            }
        }
    }

    // Cache the rest-pose LOCAL verts/normals of a RIGID body's source mesh (called
    // lazily on the first bake, while the mesh is still at rest). Mirrors the
    // rest_local capture in rebuildSoftWeldCache but skips welding (a rigid mesh is
    // moved as a whole, so corners need no merging and normals must stay per-corner).
    bool rebuildRigidBakeCache(const std::string& node) {
        if (node.empty()) return false;
        SoftWeldCache cache;
        for (auto& obj : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->getNodeName() == node) cache.tris.push_back(tri);
        }
        if (cache.tris.empty()) {
            // Flat (direct SoA) mesh: no per-face facades. Capture the SoA rest local pos/normal so
            // applyRigidBakedTransform can bake the body delta straight into the GeometryDetail.
            for (auto& obj : world.objects) {
                auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj);
                if (!tm || tm->nodeName != node || !tm->geometry) continue;
                DNA::GeometryDetail* g = tm->geometry.get();
                const size_t vc = g->get_vertex_count();
                const Vec3* Po = g->get_attribute_data<Vec3>("P_orig");
                const Vec3* No = g->get_attribute_data<Vec3>("N_orig");
                if (!Po || vc == 0) return false;
                cache.flat_mesh = tm.get();
                cache.flat_rest_pos.assign(Po, Po + vc);
                if (No) cache.flat_rest_nrm.assign(No, No + vc);
                break;
            }
            if (!cache.flat_mesh) return false;
            rigid_bake_cache_[node] = std::move(cache);
            return true;
        }
        cache.rest_local_pos.reserve(cache.tris.size());
        cache.rest_local_nrm.reserve(cache.tris.size());
        for (auto& tri : cache.tris) {
            std::array<Vec3, 3> lp, ln;
            for (int i = 0; i < 3; ++i) {
                lp[i] = tri->getOriginalVertexPosition(i);
                ln[i] = tri->getOriginalVertexNormal(i);
            }
            cache.rest_local_pos.push_back(lp);
            cache.rest_local_nrm.push_back(ln);
        }
        rigid_bake_cache_[node] = std::move(cache);
        return true;
    }

    // Render write-back for a RIGID body: apply the body's world-space rigid delta
    // D = B(t)*inv(B0) to the source mesh by baking transformed vertices into BOTH
    // `original` (LOCAL — what the GPU BLAS reads) and `position` (WORLD — CPU path),
    // leaving the object's TRANSFORM HANDLE untouched. This is the soft-body render
    // path adapted for rigid: it renders imported/non-TRS meshes correctly in every
    // backend (moving the transform corrupted them from frame 0), while PRESERVING
    // the mesh's authored per-corner normals (no welding/smoothing — a flat cube
    // stays flat). D == identity restores the rest pose.
    void applyRigidBakedTransform(const std::string& node, const Matrix4x4& D) {
        auto it = rigid_bake_cache_.find(node);
        if (it == rigid_bake_cache_.end()) {
            if (!rebuildRigidBakeCache(node)) return;   // captured at rest (first call)
            it = rigid_bake_cache_.find(node);
        }
        SoftWeldCache& cache = it->second;

        // Flat (direct SoA) rigid bake: no facades, so apply the body's world delta D straight to
        // the mesh's GeometryDetail. Convert D to a LOCAL transform (Mlocal = inv(Th)*D*Th, Th = the
        // untouched spawn world matrix) and push every SoA vertex's rest local pos/normal through it,
        // writing P_orig/N_orig (authoritative local) + the world-baked P/N mirrors — the same SoA
        // write the flat sculpt path uses. Mirrors the facade math below, vertex-indexed instead of
        // per-corner. The transform handle stays untouched (matches the facade rigid path).
        if (cache.flat_mesh && cache.flat_mesh->geometry) {
            DNA::GeometryDetail* g = cache.flat_mesh->geometry.get();
            const size_t vc = g->get_vertex_count();
            Vec3* Po = g->get_attribute_data_mut<Vec3>("P_orig");
            Vec3* P  = g->get_attribute_data_mut<Vec3>("P");
            Vec3* No = g->get_attribute_data_mut<Vec3>("N_orig");
            Vec3* N  = g->get_attribute_data_mut<Vec3>("N");
            const Matrix4x4 ThF = cache.flat_mesh->transform ? cache.flat_mesh->transform->getFinal()
                                                             : Matrix4x4::identity();
            const Matrix4x4 NTF = cache.flat_mesh->transform ? cache.flat_mesh->transform->getNormalTransform()
                                                             : Matrix4x4::identity();
            const Matrix4x4 Mlocal = ThF.inverse() * D * ThF;       // rest LOCAL -> deformed LOCAL
            const Matrix4x4 Mlocal_n = Mlocal.inverse().transpose();
            auto unit = [](const Vec3& v) {
                const float l = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
                return (l > 1e-12f) ? Vec3(v.x / l, v.y / l, v.z / l) : Vec3(0.0f, 1.0f, 0.0f);
            };
            const size_t n = std::min(vc, cache.flat_rest_pos.size());
            for (size_t v = 0; v < n; ++v) {
                const Vec3 lp = Mlocal.transform_point(cache.flat_rest_pos[v]);
                if (Po) Po[v] = lp;
                if (P)  P[v]  = ThF.transform_point(lp);
                if (v < cache.flat_rest_nrm.size()) {
                    const Vec3 ln = unit(Mlocal_n.transform_vector(cache.flat_rest_nrm[v]));
                    if (No) No[v] = ln;
                    if (N)  N[v]  = unit(NTF.transform_vector(ln));
                }
            }
            markBodyGeometryDirty(node);
            return;
        }

        const Transform* last_xf = nullptr;
        Matrix4x4 Th = Matrix4x4::identity();
        Matrix4x4 Mlocal = Matrix4x4::identity();      // rest LOCAL pos -> deformed LOCAL pos
        Matrix4x4 Mlocal_n = Matrix4x4::identity();    // LOCAL normal transform
        Matrix4x4 NT = Matrix4x4::identity();          // LOCAL normal -> WORLD normal
        bool have = false;

        for (std::size_t t = 0; t < cache.tris.size(); ++t) {
            auto& tri = cache.tris[t];
            if (!tri) continue;
            const Transform* xfp = tri->getTransformPtr();
            if (!have || xfp != last_xf) {
                Th = tri->getTransformMatrix();                 // unchanged spawn world matrix
                // new_local = inv(Th) * D * Th * rest_local  (apply D in world, back to local)
                Mlocal = Th.inverse() * D * Th;
                Mlocal_n = Mlocal.inverse().transpose();
                NT = xfp ? xfp->getNormalTransform() : Matrix4x4::identity();
                last_xf = xfp;
                have = true;
            }
            const std::array<Vec3, 3>& rlp = cache.rest_local_pos[t];
            const std::array<Vec3, 3>& rln = cache.rest_local_nrm[t];
            auto unit = [](const Vec3& v) {
                const float l = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
                return (l > 1e-12f) ? Vec3(v.x / l, v.y / l, v.z / l) : Vec3(0.0f, 1.0f, 0.0f);
            };
            for (int i = 0; i < 3; ++i) {
                const Vec3 lp = Mlocal.transform_point(rlp[i]);
                const Vec3 ln = unit(Mlocal_n.transform_vector(rln[i]));
                const Vec3 wp = Th.transform_point(lp);
                const Vec3 wn = unit(NT.transform_vector(ln));
                tri->setOriginalVertexPosition(i, lp);
                tri->setVertexPosition(i, wp);
                tri->setOriginalVertexNormal(i, ln);
                tri->setVertexNormal(i, wn);
            }
        }
        markBodyGeometryDirty(node);
    }

    // Restore a soft/cloth body's source mesh to its undeformed rest pose. The
    // writer overwrote each triangle's LOCAL `original` with the deformed geometry,
    // so we restore the cached bind-pose local first, then recompute world. Shared
    // by the soft reset-to-rest callback AND the save-time rest restore.
    void restoreSoftRestMesh(const std::string& node) {
        ++body_geom_version_;  // verts change → invalidate the gizmo's memoized AABB
        auto it = soft_weld_cache_.find(node);
        if (it != soft_weld_cache_.end()) {
            SoftWeldCache& cache = it->second;
            // Flat (direct SoA) soft body: restore the SoA from the captured rest local pos/normal
            // (the writer overwrote P_orig/N_orig with the deformed shape), then re-bake world P/N.
            if (cache.flat_mesh && cache.flat_mesh->geometry) {
                DNA::GeometryDetail* g = cache.flat_mesh->geometry.get();
                const size_t vc = std::min(g->get_vertex_count(), cache.flat_rest_pos.size());
                Vec3* Po = g->get_attribute_data_mut<Vec3>("P_orig");
                Vec3* P  = g->get_attribute_data_mut<Vec3>("P");
                Vec3* No = g->get_attribute_data_mut<Vec3>("N_orig");
                Vec3* N  = g->get_attribute_data_mut<Vec3>("N");
                const Matrix4x4 xf = cache.flat_mesh->transform ? cache.flat_mesh->transform->getFinal()
                                                                : Matrix4x4::identity();
                const Matrix4x4 NT = cache.flat_mesh->transform ? cache.flat_mesh->transform->getNormalTransform()
                                                                : Matrix4x4::identity();
                auto unit = [](const Vec3& v) {
                    const float l = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
                    return (l > 1e-12f) ? Vec3(v.x / l, v.y / l, v.z / l) : Vec3(0.0f, 1.0f, 0.0f);
                };
                for (size_t v = 0; v < vc; ++v) {
                    if (Po) Po[v] = cache.flat_rest_pos[v];
                    if (P)  P[v]  = xf.transform_point(cache.flat_rest_pos[v]);
                    if (v < cache.flat_rest_nrm.size()) {
                        if (No) No[v] = cache.flat_rest_nrm[v];
                        if (N)  N[v]  = unit(NT.transform_vector(cache.flat_rest_nrm[v]));
                    }
                }
                markBodyGeometryDirty(node);
                return;
            }
            for (std::size_t t = 0; t < cache.tris.size(); ++t) {
                auto& tri = cache.tris[t];
                if (!tri) continue;
                if (t < cache.rest_local_pos.size()) {
                    for (int i = 0; i < 3; ++i) {
                        tri->setOriginalVertexPosition(i, cache.rest_local_pos[t][i]);
                        tri->setOriginalVertexNormal(i, cache.rest_local_nrm[t][i]);
                    }
                }
                tri->updateTransformedVertices();  // position = transform * restored original
            }
            // Keep the weld topology: a cached-frame replay still needs it to
            // scatter, and the resolver overwrites it on the next create.
            Core::RenderStateManager::instance().markDirty(Core::DirtyScope::Geometry);
            return;
        }
        // No cache (never simulated): just recompute from current original.
        bool any = false;
        for (auto& obj : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->getNodeName() == node) {
                tri->updateTransformedVertices();
                any = true;
            }
        }
        if (any) Core::RenderStateManager::instance().markDirty(Core::DirtyScope::Geometry);
    }

    // Restore a body's source mesh to its rest pose using the cache appropriate
    // for its CURRENT kind — MUST be called BEFORE kind is changed so the right
    // restore path is taken. Both rigid (rigid_bake_cache_) and soft/cloth
    // (soft_weld_cache_) caches are dropped afterwards so the new kind (or the
    // removal path) starts with a clean mesh and rebuilds from rest geometry.
    void restoreBodyMeshToRest(const std::string& node, RayTrophiSim::BodyKind current_kind) {
        if (node.empty()) return;
        if (current_kind == RayTrophiSim::BodyKind::Rigid) {
            // Rigid body: identity delta restores the rest mesh via the bake cache.
            applyRigidBakedTransform(node, Matrix4x4::identity());
        } else {
            // Soft / Cloth: restore from the weld cache's saved rest_local_pos.
            restoreSoftRestMesh(node);
        }
        // Drop both caches so the new kind (or a fresh add) rebuilds from the
        // now-clean rest mesh, not from stale deformed geometry.
        soft_weld_cache_.erase(node);
        rigid_bake_cache_.erase(node);
        melt_weld_cache_.erase(node);
        melt_displaced_.erase(node);
        melt_sdf_refresh_stamp_.erase(node);
        melt_applied_stamp_.erase(node);
    }

    // Route a body's per-frame geometry change to the cheapest correct refresh.
    // Vulkan RT active → record the node for an in-place per-mesh BLAS refit
    // (consumed in the render loop), avoiding the full-scene BLAS teardown that
    // markDirty(Geometry) forces every frame. Any other backend (OptiX / CPU) →
    // the proven full-rebuild path. See pending_deform_nodes_.
    void markBodyGeometryDirty(const std::string& node) {
        ++body_geom_version_;  // invalidate the gizmo's memoized world-AABB
        if (deform_refit_active_ && !node.empty()) {
            // Cheap per-node refit on the active path (consumed in Main's render loop)
            // + a true CPU Embree refit for the picking / CPU-render BVH. Avoids the
            // full-scene teardown markDirty(Geometry) triggers every frame.
            pending_deform_nodes_.insert(node);
            g_geometry_deform_pending = true;
            g_cpu_bvh_refit_pending = true;
        } else {
            Core::RenderStateManager::instance().markDirty(Core::DirtyScope::Geometry);
        }
    }

    // Monotonic counter; the selection gizmo memoizes a body's world-AABB against it.
    uint64_t bodyGeomVersion() const { return body_geom_version_; }

    // One-shot request for SceneUI to rebuild its mesh/bbox caches. Set by data-side
    // ops (e.g. applyBodyAtCurrentFrame) that change an object's geometry WITHOUT
    // changing the object count — the free-function panels can't reach SceneUI's
    // caches directly, and SceneUI only auto-rebuilds on a membership-count change.
    void requestUiMeshCacheRebuild() { ui_mesh_cache_rebuild_request_ = true; }
    bool consumeUiMeshCacheRebuild() {
        const bool v = ui_mesh_cache_rebuild_request_;
        ui_mesh_cache_rebuild_request_ = false;
        return v;
    }

    // Set each frame by the render loop: true only when the active RENDER backend
    // is Vulkan RT and a per-mesh BLAS refit is valid for this frame.
    void setDeformRefitActive(bool v) { deform_refit_active_ = v; }
    void deferMeltDisplacementOnce() { defer_melt_displacement_once_ = true; }
    bool hasPendingDeformNodes() const { return !pending_deform_nodes_.empty(); }
    void clearPendingDeformNodes() { pending_deform_nodes_.clear(); }
    std::vector<std::string> takePendingDeformNodes() {
        std::vector<std::string> out(pending_deform_nodes_.begin(), pending_deform_nodes_.end());
        pending_deform_nodes_.clear();
        return out;
    }
    // Triangles of a body's source mesh, in the same order the BLAS was built from
    // (world.objects order — what the body caches also capture), for the per-mesh
    // refit path. Empty when the node has no mesh yet.
    std::vector<std::shared_ptr<Triangle>> collectNodeTriangles(const std::string& node) {
        auto rit = rigid_bake_cache_.find(node);
        if (rit != rigid_bake_cache_.end() && !rit->second.tris.empty()) return rit->second.tris;
        auto sit = soft_weld_cache_.find(node);
        if (sit != soft_weld_cache_.end() && !sit->second.tris.empty()) return sit->second.tris;
        std::vector<std::shared_ptr<Triangle>> tris;
        for (auto& obj : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->getNodeName() == node) tris.push_back(tri);
        }
        return tris;
    }

    // The flat (direct SoA) TriangleMesh-as-Hittable for a node, or null when the node is facade-
    // backed / absent. Lets the per-mesh deform refit route a flat mesh (collectNodeTriangles is
    // empty for it) straight to a cheap SoA refit instead of a full per-frame rebuild.
    TriangleMesh* getFlatNodeMesh(const std::string& node) {
        for (auto& obj : world.objects) {
            auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj);
            if (tm && tm->nodeName == node) return tm.get();
        }
        return nullptr;
    }

    // ── Save-time rest restore ───────────────────────────────────────────────
    // The sim bakes its deformed/posed result straight into the source meshes'
    // LOCAL `original` verts (applySoftDeformedVerts / applyRigidBakedTransform),
    // and the project serializer writes those verts verbatim (writeGeometryBinary
    // dumps getOriginalVertexPosition/Normal). Saving mid-sim — or after pausing on
    // a non-rest frame — therefore persisted the FINAL sim pose into the file; on
    // reload the body was stuck in it (the load-time resetRuntime then cached the
    // deformed mesh as the new "rest", so even removing the body restored to the
    // corrupted pose). Before geometry is written we restore every body to its rest
    // mesh; reapplyBodyRestSnapshot() puts the live deformation back afterwards so
    // the on-screen simulation is undisturbed by the save.
    struct BodyRestSnapshot {
        std::shared_ptr<Triangle> tri;
        std::array<Vec3, 3> orig_pos;
        std::array<Vec3, 3> orig_nrm;
    };

    std::vector<BodyRestSnapshot> snapshotAndRestoreBodiesToRest() {
        std::vector<BodyRestSnapshot> snaps;
        if (rigid_bodies.empty()) return snaps;
        for (auto& rb : rigid_bodies) {
            const std::string& node = rb.source_name;
            if (node.empty()) continue;
            bool any = false;
            for (auto& obj : world.objects) {
                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                if (!tri || tri->getNodeName() != node) continue;
                BodyRestSnapshot s;
                s.tri = tri;
                for (int i = 0; i < 3; ++i) {
                    s.orig_pos[i] = tri->getOriginalVertexPosition(i);
                    s.orig_nrm[i] = tri->getOriginalVertexNormal(i);
                }
                snaps.push_back(std::move(s));
                any = true;
            }
            if (!any) continue;
            if (rb.kind == RayTrophiSim::BodyKind::Rigid)
                applyRigidBakedTransform(node, Matrix4x4::identity());
            else
                restoreSoftRestMesh(node);
        }
        return snaps;
    }

    void reapplyBodyRestSnapshot(const std::vector<BodyRestSnapshot>& snaps) {
        if (snaps.empty()) return;
        for (const auto& s : snaps) {
            if (!s.tri) continue;
            for (int i = 0; i < 3; ++i) {
                s.tri->setOriginalVertexPosition(i, s.orig_pos[i]);
                s.tri->setOriginalVertexNormal(i, s.orig_nrm[i]);
            }
            s.tri->updateTransformedVertices();
        }
        ++body_geom_version_;  // verts changed → invalidate the gizmo's memoized AABB
        Core::RenderStateManager::instance().markDirty(Core::DirtyScope::Geometry);
    }

    // Snapshot the deformed UNIQUE world vertices of every soft body for `frame`
    // (read from the meshes the writer just updated). No-op without soft bodies.
    void captureSoftFrame(int frame) {
        if (soft_weld_cache_.empty()) return;
        snapshotSoftBodies(soft_frame_cache_[frame]);
    }

    // Replay the cached soft deformation for `frame` back onto the meshes. Returns
    // false when the frame isn't cached.
    bool restoreSoftFrame(int frame) {
        auto it = soft_frame_cache_.find(frame);
        if (it == soft_frame_cache_.end()) return false;
        for (auto& kv : it->second) applySoftDeformedVerts(kv.first, kv.second);
        return true;
    }

    void captureSimFrame(int frame) {
        if (static_cast<int>(sim_frame_cache_.size()) >= kMaxCachedSimFrames &&
            sim_frame_cache_.find(frame) == sim_frame_cache_.end()) {
            return; // cache cap reached; keep what we have
        }
        auto& entry = sim_frame_cache_[frame];
        entry.clear();
        entry.reserve(particle_systems.size());
        for (auto& system : particle_systems) {
            if (system.runtime) {
                entry.push_back(system.runtime->captureGridDomainStatesForCache(
                    simulation_world.compute()));
            }
            else entry.emplace_back();
        }
        // Capture the discrete particle SoA in the SAME pass so a cached-frame
        // replay restores the actual particles (grid states alone left them empty).
        auto& psnap = particle_frame_cache_[frame];
        psnap.clear();
        psnap.reserve(particle_systems.size());
        for (auto& system : particle_systems) {
            if (system.runtime) {
                ParticleFrameSnapshot snapshot;
                snapshot.buffers = system.runtime->buffers();
                snapshot.alive_count = system.runtime->aliveCount();
                snapshot.runtime = system.runtime->captureRuntimeState();
                psnap.emplace_back(std::move(snapshot));
            } else {
                psnap.emplace_back();
            }
        }
        // Burn/heat surface damage, same pass. Without this a scrub or loop-back
        // silently un-burned every object: MSF is permanent by design and there
        // is no other path that reproduces it, so an uncached frame replayed as
        // pristine geometry with a fully-simulated fire around it.
        auto& msnap = msf_frame_cache_[frame];
        msnap.clear();
        msnap.reserve(particle_systems.size());
        for (auto& system : particle_systems) {
            if (system.runtime) {
                msnap.push_back(system.runtime->captureMaterialStateFieldsForCache(
                    simulation_world.compute()));
            } else {
                msnap.emplace_back();
            }
        }
        // Capture the rigid bodies in the SAME pass so replay restores them in
        // lockstep with this fluid frame (see rigid_frame_cache_).
        captureRigidFrame(frame);
        // Soft/cloth deformation is mesh-resident, so record it per frame too.
        captureSoftFrame(frame);
    }

    // Snapshot the dynamic rigid bodies for `frame`, keyed alongside the fluid
    // cache. No-op when there are no rigid bodies.
    void captureRigidFrame(int frame) {
        if (!rigid_body_system || rigid_bodies.empty()) return;
        rigid_body_system->captureFrameState(rigid_frame_cache_[frame]);
    }

    // Replay the rigid bodies for `frame` from the cache (pose + velocities), so
    // their motion matches the cached fluid exactly instead of being re-simulated
    // against a frozen fluid frame. Returns false when the frame isn't cached.
    bool restoreRigidFrame(int frame) {
        if (!rigid_body_system) return false;
        auto it = rigid_frame_cache_.find(frame);
        if (it != rigid_frame_cache_.end()) {
            if (!rigid_body_system->restoreFrameState(it->second)) return false;
            rigid_timeline_frame_ = frame;
            return true;
        }
        // RAM cache misses on every reopened project and after any cache reset,
        // and the fallback below this is a capped catch-up re-sim that neither
        // matches the baked run nor arrives on the frame being displayed. The
        // on-disk bake has the answer — use it before resorting to re-simulating.
        if (sim_cache_valid_ && !sim_cache_dir_.empty() &&
            frame >= sim_cache_start_frame_ && frame <= sim_cache_end_frame_) {
            std::vector<RayTrophiSim::RigidBodyFrameState> disk;
            if (RayTrophiSim::SimCache::readRigidFrame(sim_cache_dir_, frame, disk) &&
                rigid_body_system->restoreFrameState(disk)) {
                rigid_timeline_frame_ = frame;
                return true;
            }
        }
        return false;
    }

    // Bring the rigid timeline to `frame`: replay it from the cache when present
    // (the deterministic, fluid-matching path), otherwise fall back to the cheap
    // re-sim (used by the disk-cache path and any uncached gap).
    void syncRigidToFrame(int frame, float fixed_dt, int max_steps) {
        if (frame < 0) frame = 0;
        if (rigid_timeline_frame_ == frame) return;
        // Breakable scenes bypass the cache and re-sim: the Static->Dynamic shatter
        // transition isn't captured by the (dynamic-only) rigid frame cache, so a
        // cached replay would freeze shards pre-break. Deterministic re-sim re-breaks
        // at the same frame instead. Rigid is cheap; the fluid/grid cache is untouched.
        if (!hasBreakableBodies() && restoreRigidFrame(frame)) return;
        advanceRigidTimelineToFrame(frame, fixed_dt, max_steps);
    }

    bool restoreSimFrame(int frame, float fixed_dt = 1.0f / 24.0f) {
        // Frame 0 is reconstructed from the rest pose rather than from a cache
        // entry: it is the one frame that is exactly reproducible, and resetting
        // here is what makes a loop-back put a fallen body back at the top.
        // (Rigid frames ARE cached now — RAM in lockstep with the fluid cache,
        // and on disk beside it; see restoreRigidFrame.)
        if (frame <= 0 && rigid_body_system) { rigid_body_system->resetRuntime(); resetFractureToIntact(); }
        // Soft/cloth deformation is mesh-resident and cached per frame; replay it so
        // a cached-frame scrub/loop shows the cloth's shape instead of a frozen mesh.
        restoreSoftFrame(frame);
        auto it = sim_frame_cache_.find(frame);
        if (it != sim_frame_cache_.end() && it->second.size() == particle_systems.size()) {
            // Restore the discrete particle SoA from the lockstep cache (if present
            // for this frame) so a cached-frame replay shows the actual particles
            // instead of an empty SoA.
            auto pit = particle_frame_cache_.find(frame);
            const bool have_particles =
                (pit != particle_frame_cache_.end() && pit->second.size() == particle_systems.size());
            auto mit = msf_frame_cache_.find(frame);
            const bool have_msf =
                (mit != msf_frame_cache_.end() && mit->second.size() == particle_systems.size());
            for (std::size_t i = 0; i < particle_systems.size(); ++i) {
                if (particle_systems[i].runtime) {
                    particle_systems[i].runtime->setGridDomainStates(it->second[i]);
                    if (have_particles) {
                        particle_systems[i].runtime->restoreSoA(
                            pit->second[i].buffers, pit->second[i].alive_count);
                        particle_systems[i].runtime->restoreRuntimeState(pit->second[i].runtime);
                    }
                    if (have_msf) {
                        particle_systems[i].runtime->restoreMaterialStateFields(
                            mit->second[i], simulation_world.compute());
                    }
                    invalidateSimulationRenderBindings(particle_systems[i]);
                }
            }
            simulation_world.resetTime(static_cast<float>(frame) * fixed_dt, frame);
            // A cached grid/particle snapshot does not contain the authored
            // transform of object-bound colliders/emitters. Pose those sources
            // to the same frame before exposing the restored state; otherwise
            // the collider gizmo/voxel mask starts at the previous live pose and
            // only converges toward the object over subsequent simulation steps.
            applySimSourceObjectPosesForFrame(frame);
            return true;
        }
        // Disk fallback: stream the frame from the on-disk bake cache (render-only).
        if (restoreSimFrameFromDisk(frame, fixed_dt)) {
            return true;
        }
        return false;
    }

    // Read every system's domain states for `frame` from the on-disk bake cache
    // and install them. Returns false (silently) when no valid disk cache is
    // bound, the frame is out of the baked range, or any system file is missing/
    // corrupt — callers then fall back to resimulation as before.
    bool restoreSimFrameFromDisk(int frame, float fixed_dt = 1.0f / 24.0f) {
        if (!sim_cache_valid_ || sim_cache_dir_.empty()) return false;
        if (frame < sim_cache_start_frame_ || frame > sim_cache_end_frame_) return false;

        for (std::size_t i = 0; i < particle_systems.size(); ++i) {
            if (!particle_systems[i].runtime) continue;
            if (sim_cache_valid_system_ids_.count(particle_systems[i].id) > 0) {
                std::vector<RayTrophiSim::SimulationGridDomainState> loaded;
                std::vector<RayTrophiSim::MaterialStateFieldSnapshot> loaded_msf;
                if (RayTrophiSim::SimCache::readSystemFrame(
                        sim_cache_dir_, particle_systems[i].id, frame, loaded, loaded_msf)) {
                    particle_systems[i].runtime->setGridDomainStates(loaded);
                    // Burn/heat damage rides alongside the grid states; without
                    // this a disk-replayed frame shows pristine geometry inside a
                    // fully simulated fire.
                    particle_systems[i].runtime->restoreMaterialStateFields(
                        loaded_msf, simulation_world.compute());
                    invalidateSimulationRenderBindings(particle_systems[i]);
                } else {
                    particle_systems[i].runtime->resetGridDomainStates();
                    particle_systems[i].runtime->clear();
                    invalidateSimulationRenderBindings(particle_systems[i]);
                }
            } else {
                particle_systems[i].runtime->resetGridDomainStates();
                particle_systems[i].runtime->clear();
                invalidateSimulationRenderBindings(particle_systems[i]);
            }
        }
        // Soft bodies: replay the baked deformation. On a freshly reopened project
        // the body was never live-created, so build the weld topology on demand.
        std::vector<RayTrophiSim::SimCache::SoftBodyFrame> soft;
        if (RayTrophiSim::SimCache::readSoftFrame(sim_cache_dir_, frame, soft)) {
            for (auto& b : soft) {
                if (soft_weld_cache_.find(b.name) == soft_weld_cache_.end())
                    rebuildSoftWeldCache(b.name);
                applySoftDeformedVerts(b.name, b.vertices);
            }
        }
        // Rigid bodies come from the SAME frame on disk, before the source-object
        // poses are applied — a body restored here is the authority for its own
        // object's pose, and anything parented to it (flow sources, emitters)
        // resolves against this frame rather than against wherever a capped
        // catch-up re-sim happened to leave the body.
        std::vector<RayTrophiSim::RigidBodyFrameState> rigid;
        if (rigid_body_system &&
            RayTrophiSim::SimCache::readRigidFrame(sim_cache_dir_, frame, rigid) &&
            rigid_body_system->restoreFrameState(rigid)) {
            rigid_timeline_frame_ = frame;
        }
        simulation_world.resetTime(static_cast<float>(frame) * fixed_dt, frame);
        // Disk cache stores simulation fields, not source-object transforms.
        // Keep collider/emitter/domain bindings on the exact restored frame.
        applySimSourceObjectPosesForFrame(frame);
        return true;
    }

    void stepRigidBodiesOnly(float fixed_dt, int frame) {
        if (!rigid_body_system || rigid_bodies.empty()) return;
        if (fixed_dt <= 0.0f) fixed_dt = 1.0f / 24.0f;

        RayTrophiSim::SimulationContext ctx = simulation_world.makeContext(fixed_dt, 0, 1);
        ctx.dt = fixed_dt;
        ctx.fixed_dt = fixed_dt;
        ctx.time_seconds = static_cast<float>(frame) * fixed_dt;
        ctx.frame = frame;
        ctx.timeline_frame = frame;
        ctx.substep_index = 0;
        ctx.substep_count = 1;

        rigid_body_system->prepare(ctx);
        rigid_body_system->step(ctx);
        rigid_body_system->finalize(ctx);
    }

    bool advanceRigidTimelineToFrame(int target_frame, float fixed_dt, int max_steps) {
        if (target_frame < 0) target_frame = 0;
        if (!rigid_body_system || rigid_bodies.empty()) {
            rigid_timeline_frame_ = target_frame;
            return true;
        }
        if (rigid_timeline_frame_ < 0 || target_frame < rigid_timeline_frame_) {
            rigid_body_system->resetRuntime();
            resetFractureToIntact();  // rewind un-shatters; re-sim re-breaks deterministically
            rigid_timeline_frame_ = 0;
        }
        int steps = 0;
        while (rigid_timeline_frame_ < target_frame && steps < max_steps) {
            const int next_frame = rigid_timeline_frame_ + 1;
            applySimSourceObjectPosesForFrame(next_frame);
            syncSimulationWorld();
            stepRigidBodiesOnly(fixed_dt, next_frame);
            processFractureImpacts();  // shatter on contact above threshold
            emitCombustionStructuralImpulses(fixed_dt);  // fire -> blast
            processStructuralImpulseEvents();  // ...and by blast overpressure
            rigid_timeline_frame_ = next_frame;
            ++steps;
        }
        return rigid_timeline_frame_ == target_frame;
    }

    void resetSimulationToStart(bool clear_cache = true, bool capture_frame = true) {
        if (clear_cache) {
            clearSimFrameCache();
        }
        for (auto& system : particle_systems) {
            if (system.runtime) {
                invalidateSimulationRenderBindings(system);
                system.runtime->resetGridDomainStates();
                system.runtime->clear();  // particles back to empty for a deterministic bake
                // Re-arm authored initial fluid seeds and synchronize the
                // domain states NOW, so frame 0 already carries the full tank +
                // correct grid metadata — exactly what the disk bake does
                // (bakeSimulationToDisk). Without this the interactive play path
                // started frame 0 EMPTY (clear() wiped the seeded particles and the
                // seed wasn't re-applied until the first step), so a fill domain
                // only refilled a couple of frames in and its SurfaceSDF volume
                // wasn't built/raymarched until ~frame 2 (the reported bug).
                for (auto& dom : system.runtime->gridDomains()) {
                    if (dom.type == RayTrophiSim::SimulationDomainType::Fluid &&
                        (dom.fluid_seed_mode == RayTrophiSim::FluidSeedMode::FillLevel ||
                         dom.fluid_reseed_on_reset)) {
                        dom.fluid_pending_seed = true;
                    }
                }
                system.runtime->synchronizeGridDomainsNow();
            }
        }
        for (auto& obj : fluid_objects) {
            obj.resetState();
            if (obj.pending_seed) {
                obj.ensureGrid();
                RayTrophiSim::Fluid::seedBox(obj.particles, obj.grid, obj.seed_min, obj.seed_max, obj.seed_particles_per_cell);
            }
        }
        // Rigid bodies respawn at their source objects' poses on the next step.
        if (rigid_body_system) rigid_body_system->resetRuntime();
        resetFractureToIntact();  // un-shatter breakable groups back to intact
        rigid_timeline_frame_ = 0;
        simulation_world.resetTime(0.0f, 0);
        applySimSourceObjectPosesForFrame(0);
        if (capture_frame) {
            captureSimFrame(0);
        }
    }

    // Return to interactive free-run preview (default mode).
    void resetSimulation() {
        resetSimulationToStart();
        sim_timeline_frame_ = -1;
        rigid_timeline_frame_ = -1;
        syncSimulationRenderVolumes();
    }

    void requestSimulationTimelineRenderResync() {
        force_simulation_render_sync_ = true;
    }

    // A script can deliberately author a setup and then advance its live runtime
    // with rt.fluid.step(). The first UI tick must publish that resulting state,
    // not interpret the preceding authoring edits as a request to restore frame
    // zero and erase it. Consumed once by updateSimulationTimeline().
    void preserveScriptSimulationPreview() {
        preserve_script_simulation_preview_ = true;
    }

    void setSimulationKeyAuthoringMode(bool enabled) {
        if (simulation_key_authoring_mode_ == enabled) return;
        simulation_key_authoring_mode_ = enabled;
        if (!enabled) force_simulation_render_sync_ = true;
    }

    bool simulationKeyAuthoringMode() const {
        return simulation_key_authoring_mode_;
    }

    // Start/End are playback-range metadata, not simulation inputs. Editing
    // them can end ImGui's generic `ui_editing` settle gate and expose a benign
    // runtime signature drift (for example an asynchronously published collider
    // or melt-owned geometry) as if the user changed fluid authoring. Rebase the
    // signatures without dropping caches or restoring frame zero.
    void preserveSimulationForTimelineRangeEdit() {
        last_sim_config_sig_ = computeSimConfigSignature();
        last_fluid_coupling_sig_ = computeFluidCouplingSignature();
        timeline_range_edit_grace_ = 3;
        sim_rewind_request_ = false;
    }

    // A fluid-affecting setup edit rewinds the sim to frame 0 (see
    // updateSimulationTimeline) instead of auto-resimming up to a high parked
    // frame. The UI layer consumes this to move the timeline playhead to start.
    bool consumeSimRewindRequest() {
        const bool r = sim_rewind_request_;
        sim_rewind_request_ = false;
        return r;
    }

    // Push appearance-only SurfaceSDF params (IOR + VolumeShader edits) to the
    // already-bound live volumes and flag a volume-SSBO re-upload — WITHOUT
    // rebuilding the level set. The VolumeShader is bound by shared_ptr so its
    // edits propagate for free; only the IOR needs copying onto the volume.
    // Use this for IOR / colour / density / absorption slider edits so the
    // viewport updates live and cheaply. Geometry-affecting params (kernel /
    // particle / narrow / surface band) must use
    // requestSimulationTimelineRenderResync() instead — they rebuild the SDF.
    void refreshFluidSurfaceMaterial() {
        for (auto& system : particle_systems) {
            if (!system.runtime) continue;
            auto& domains = system.runtime->gridDomains();
            const std::size_t n = std::min(domains.size(), system.domain_volumes.size());
            for (std::size_t d = 0; d < n; ++d) {
                if (!system.domain_volumes[d]) continue;
                if (domains[d].type != RayTrophiSim::SimulationDomainType::Fluid) continue;
                if (domains[d].fluid_render_mode !=
                    RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) continue;
                system.domain_volumes[d]->render_isosurface_ior = domains[d].fluid_surface_ior;
                system.domain_volumes[d]->render_isosurface_roughness = domains[d].fluid_surface_roughness;
                system.domain_volumes[d]->render_isosurface_foam = domains[d].fluid_surface_foam;
                // Live like the IOR: picking a material must repaint without a
                // re-sim, otherwise the control reads as broken.
                system.domain_volumes[d]->render_isosurface_material_id =
                    domains[d].fluid_surface_material_id;
                // Porosity is pure shader state too — a crumb-size slider must
                // repaint the current frame without a re-sim.
                system.domain_volumes[d]->render_isosurface_pore_amount =
                    domains[d].fluid_surface_pore_amount;
                system.domain_volumes[d]->render_isosurface_pore_scale =
                    domains[d].fluid_surface_pore_scale;
                system.domain_volumes[d]->render_isosurface_pore_detail =
                    domains[d].fluid_surface_pore_detail;
                // Coordinate space is pure shader state as well — the level set
                // is untouched, only the coordinate the patterns are addressed
                // in changes. It MUST be pushed here and not only in the sync
                // loop: the sync loop runs on simulation frames, so without this
                // the combo would appear dead on a paused timeline, which is the
                // state an artist is in while choosing a look.
                system.domain_volumes[d]->render_isosurface_coord_space =
                    domains[d].fluid_surface_coord_space;
                // Volume whitewater look (tint + extinction) is pure shader state —
                // push it live like the IOR so a Foam Color / Foam Opacity slider
                // updates the current frame without a re-splat (those ride
                // g_gas_volumes_dirty into the volume table). Foam Density is NOT
                // here: it changes the deposited temp grid and needs a re-upload
                // (the UI routes it through requestSimulationTimelineRenderResync).
                if (domains[d].fluid_foam_params.enabled &&
                    domains[d].fluid_foam_params.render_mode ==
                        RayTrophiSim::Fluid::FoamRenderMode::Volume) {
                    system.domain_volumes[d]->render_isosurface_foam_color =
                        domains[d].fluid_foam_params.volume_color;
                    system.domain_volumes[d]->render_isosurface_foam_opacity =
                        domains[d].fluid_foam_params.volume_opacity;
                }
            }
        }
        for (auto& obj : fluid_objects) {
            auto it = fluid_render_bindings.find(obj.id);
            if (it == fluid_render_bindings.end() || !it->second.volume) continue;
            if (obj.render_mode != RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF) continue;
            if (obj.shader) it->second.volume->setShader(obj.shader);
            it->second.volume->render_isosurface_ior = obj.surface_ior;
            it->second.volume->render_isosurface_roughness = obj.surface_roughness;
            it->second.volume->render_isosurface_foam = obj.surface_foam;
        }
        g_gas_volumes_dirty = true;
    }

    // Per-tick simulation driver.
    //   live_mode == true  : continuous free-run interactive preview (heavier).
    //   live_mode == false : Timeline (default) — play bakes into the cache, scrub
    //                        restores/resimulates, and a stopped timeline stays
    //                        frozen (no stepping, no render churn → cheap/idle).
    void syncFlowSourceKeysFromTimeline(bool preserve_cache = false) {
        bool keys_changed = false;
        auto keyEqual = [](const auto& a, const auto& b) {
            return a.has_enabled == b.has_enabled &&
                   a.has_position == b.has_position &&
                   a.has_velocity == b.has_velocity &&
                   a.has_radius == b.has_radius &&
                   a.has_density == b.has_density &&
                   a.has_temperature == b.has_temperature &&
                   a.has_fuel == b.has_fuel &&
                   a.has_falloff == b.has_falloff &&
                   a.has_velocity_coupling == b.has_velocity_coupling &&
                   a.enabled == b.enabled &&
                   a.position.x == b.position.x &&
                   a.position.y == b.position.y &&
                   a.position.z == b.position.z &&
                   a.velocity.x == b.velocity.x &&
                   a.velocity.y == b.velocity.y &&
                   a.velocity.z == b.velocity.z &&
                   a.radius == b.radius &&
                   a.density == b.density &&
                   a.temperature == b.temperature &&
                   a.fuel == b.fuel &&
                   a.falloff == b.falloff &&
                   a.velocity_coupling == b.velocity_coupling;
        };
        for (auto& system : particle_systems) {
            if (!system.runtime) continue;
            auto& sources = system.runtime->flowSources();
            for (std::size_t flow_i = 0; flow_i < sources.size(); ++flow_i) {
                auto& source = sources[flow_i];
                const std::string track_name =
                    "Simulation Flow " +
                    std::to_string(source.timeline_uid);
                auto track_it = timeline.tracks.find(track_name);
                if (track_it == timeline.tracks.end()) continue;

                std::map<int, RayTrophiSim::SimulationFlowSourceDesc::Keyframe>
                    synchronized;
                for (const Keyframe& marker : track_it->second.keyframes) {
                    if (!marker.has_emitter) continue;
                    RayTrophiSim::SimulationFlowSourceDesc::Keyframe key;
                    if (!source.keyframes.empty()) {
                        auto nearest = source.keyframes.lower_bound(marker.frame);
                        if (nearest == source.keyframes.end()) {
                            key = source.keyframes.rbegin()->second;
                        } else if (nearest == source.keyframes.begin()) {
                            key = nearest->second;
                        } else {
                            auto previous = std::prev(nearest);
                            key = (marker.frame - previous->first <=
                                   nearest->first - marker.frame)
                                ? previous->second : nearest->second;
                        }
                    }
                    const EmitterKeyframe& emitter = marker.emitter;
                    key.has_enabled = emitter.has_enabled;
                    key.has_position = emitter.has_position;
                    key.has_velocity = emitter.has_velocity;
                    key.has_radius = emitter.has_radius;
                    key.has_density = emitter.has_density_rate;
                    key.has_temperature = emitter.has_temperature;
                    key.has_fuel = emitter.has_fuel_rate;
                    if (emitter.has_enabled) {
                        key.enabled = emitter.enabled;
                    }
                    if (emitter.has_position) {
                        key.position = emitter.position;
                    }
                    if (emitter.has_velocity) {
                        key.velocity = emitter.velocity;
                    }
                    if (emitter.has_radius) {
                        key.radius = emitter.radius;
                    }
                    if (emitter.has_density_rate) {
                        key.density = emitter.density_rate;
                    }
                    if (emitter.has_temperature) {
                        key.temperature = emitter.temperature;
                    }
                    if (emitter.has_fuel_rate) {
                        key.fuel = emitter.fuel_rate;
                    }
                    synchronized[marker.frame] = key;
                }
                bool same = source.keyframes.size() == synchronized.size();
                if (same) {
                    auto a = source.keyframes.begin();
                    auto b = synchronized.begin();
                    for (; a != source.keyframes.end(); ++a, ++b) {
                        if (a->first != b->first ||
                            !keyEqual(a->second, b->second)) {
                            same = false;
                            break;
                        }
                    }
                }
                if (!same) {
                    source.keyframes = std::move(synchronized);
                    keys_changed = true;
                }
            }
        }
        if (keys_changed && !preserve_cache) {
            clearSimFrameCache();
            force_simulation_render_sync_ = true;
        }
    }

    void updateSimulationTimeline(int tl_frame, bool playing, float realtime_dt, float fps, bool live_mode,
                                  bool ui_editing = false) {
        if (tl_frame < 0) tl_frame = 0;
        const bool preserve_timeline_range_edit = timeline_range_edit_grace_ > 0;
        publishCompletedSdfBakes();
        syncFlowSourceKeysFromTimeline(preserve_timeline_range_edit);
        if (preserve_timeline_range_edit) {
            // Rebase after async SDF publication and flow-key synchronization,
            // both of which run above and may otherwise expose a benign delta on
            // the frame after the End field commits.
            last_sim_config_sig_ = computeSimConfigSignature();
            last_fluid_coupling_sig_ = computeFluidCouplingSignature();
            sim_rewind_request_ = false;
            --timeline_range_edit_grace_;
        }
        simulation_render_updated = false;
        if (preserve_script_simulation_preview_) {
            preserve_script_simulation_preview_ = false;
            last_sim_config_sig_ = computeSimConfigSignature();
            last_fluid_coupling_sig_ = computeFluidCouplingSignature();
            force_simulation_render_sync_ = false;
            sim_cache_valid_ = false;
            sim_timeline_frame_ = tl_frame;
            rigid_timeline_frame_ = tl_frame;
            syncSimulationRenderVolumes();
            simulation_render_updated = true;
            return;
        }
        const bool force_resync = force_simulation_render_sync_;
        if (simulation_key_authoring_mode_ && !playing) {
            // Key editing and simulation playback are intentionally separated:
            // scrubbing only chooses the frame to author. Keep the last solved
            // gas image and never auto-restore/resimulate while staging values.
            force_simulation_render_sync_ = false;
            return;
        }

        // Catch a user moving a rigid's source object (gizmo) while the timeline is
        // IDLE — sitting on its current baked frame, not playing/scrubbing. Only
        // then is a geometry-generation bump a user edit rather than the sim's own
        // per-step churn. Refreshes rb.initial_pivot so the signature below sees the
        // move WITHOUT scanning world.objects every tick.
        if (!playing && !force_resync && sim_timeline_frame_ == tl_frame) {
            refreshRigidRestPosesOnUserEdit();
        }

        // Auto-invalidate the in-memory bake cache when the simulation SETUP
        // changes (add/remove of any sim element, rigid-body param edit, …) so a
        // stale cache is never replayed. Replaces the old manual-reset workflow.
        // ★★★ THE FRAME RATE IS A PHYSICS PARAMETER HERE, and it was the one
        // config input with no cache key. Every system steps once per timeline
        // frame at fixed_dt = 1/fps, so changing fps changes dt — and dt is the
        // single largest term in what a FLIP liquid actually does. A bake made
        // at 24 fps is not the same simulation at 96.
        //
        // ★★ It is also the exact reason a "raise the fps and see" test comes
        // back meaningless: the cache is keyed by FRAME INDEX, so the old frames
        // simply replay four times faster and NOTHING re-simulates. The user
        // sees a fast playback, concludes the change did nothing, and a correct
        // diagnosis gets thrown away on the strength of a test that never ran.
        // That is worse than a wrong answer — it is a wrong answer that looks
        // like evidence.
        //
        // Handled here rather than inside computeSimConfigSignature() because
        // fps arrives as an argument, not as scene state: the signature is a
        // const method over the scene and cannot see it.
        const float effective_fps = (fps > 1.0f) ? fps : 24.0f;
        const bool fps_changed =
            last_sim_bake_fps_ > 0.0f &&
            std::fabs(effective_fps - last_sim_bake_fps_) > 1.0e-4f;
        last_sim_bake_fps_ = effective_fps;

        // Auto-invalidate the in-memory bake cache when the simulation SETUP
        // changes (add/remove of any sim element, rigid-body param edit, …) so a
        // stale cache is never replayed.
        const uint64_t cfg_sig = computeSimConfigSignature();
        if (cfg_sig != last_sim_config_sig_ || fps_changed) {
            // Settle-gate: defer the (expensive) cache drop until the edit finishes.
            // The signature changes on EVERY drag tick, so committing immediately
            // would restart the bake from frame 0 each frame of a slider drag —
            // never progressing. ui_editing is true while a widget is held; we keep
            // showing the current cache until the user lets go, then commit once.
            if (!ui_editing) {
                syncRigidBodyProxyColliders();
                // Decide whether the change actually touches a FLUID bake. A far or
                // static rigid that doesn't overlap any fluid domain only needs the
                // cheap rigid re-sim — the expensive fluid cache survives. Recompute
                // the coupling signature AFTER the proxy-collider sync.
                const uint64_t fluid_sig = computeFluidCouplingSignature();
                // ★ A frame-rate change always affects the fluid: dt is in every
                // term of the step. It cannot show up in the coupling signature,
                // which only knows about scene contents, so it is OR-ed in here —
                // otherwise the drop would run and spare the one cache that most
                // needed dropping.
                const bool fluid_affected =
                    (fluid_sig != last_fluid_coupling_sig_) || fps_changed;
                last_sim_config_sig_ = computeSimConfigSignature();
                last_fluid_coupling_sig_ = fluid_sig;
                if (fluid_affected) {
                    // Viewing frame N of a changed sim needs a fresh deterministic
                    // bake of 0..N. Auto-resimming up to a high PARKED frame on every
                    // edit is costly, so instead rewind to frame 0 (cheap — one seed),
                    // drop the RAM + disk caches, and ask the UI to move the playhead
                    // to start. The user plays forward to re-bake (and re-bakes to
                    // disk) when satisfied — the cost is opt-in, not automatic.
                    resetSimulationToStart(/*clear_cache=*/true, /*capture_frame=*/true);
                    sim_cache_valid_ = false;   // on-disk bake is stale too
                    sim_timeline_frame_ = 0;
                    rigid_timeline_frame_ = 0;
                    sim_rewind_request_ = true;
                    // Show frame 0 now and skip this tick's bake/scrub — tl_frame
                    // still holds the OLD parked value, so falling through would
                    // catch up 0→N, exactly the cost we are avoiding. The playhead
                    // moves to start next tick once the UI consumes the request.
                    syncSimulationRenderVolumes();
                    simulation_render_updated = true;
                    return;
                } else {
                    // Non-coupling rigid edit/move: keep the fluid cache but drop the
                    // now-stale rigid AND soft caches so the changed body re-bakes on
                    // next play. (soft_frame_cache_ was previously left intact here,
                    // so adding a second body froze the first at its last cached
                    // deform while only the newest body re-simulated — the bug.)
                    rigid_frame_cache_.clear();
                    soft_frame_cache_.clear();
                    rigid_timeline_frame_ = -1;  // rigid re-bakes/replays from frame 0
                }
            }
        }

        // Live Update: free-run whenever the timeline is not actively playing.
        if (live_mode && !playing) {
            sim_timeline_frame_ = -1;  // detached from the baked timeline
            rigid_timeline_frame_ = -1;
            syncSimulationWorld();
            // ★ The sim's own frame counter advances once per UI frame here, so
            // it runs at 60-144 Hz while the playhead stands still. Anything that
            // evaluates a keyframe against it burns through an authored curve in
            // a second or two — a hose keyed to pour from frame 18 to 150 empties
            // itself almost instantly, and a source keyed to fade out is already
            // past its last key. Keys belong to the PLAYHEAD, so publish it.
            simulation_world.setTimelineFrame(tl_frame);
            simulation_world.stepOnce(realtime_dt);
            processFractureImpacts();  // live preview: shatter on impact
            emitCombustionStructuralImpulses(realtime_dt);  // fire -> blast
            processStructuralImpulseEvents();  // ...and by blast overpressure
            syncSimulationRenderVolumes();
            return;
        }

        // Timeline-driven deterministic bake / scrub.
        const float fixed_dt = (fps > 1.0f) ? (1.0f / fps) : (1.0f / 24.0f);
        constexpr int kMaxStepsPerTick = 8;  // spread big jumps across UI ticks
        bool changed = false;

        // Disk-cache fast path: a project loaded with a valid bake never
        // resimulates — every frame (including 0 and loop-backs) is streamed from
        // disk. Clamp the request into the baked range so scrubbing past the ends
        // holds the first/last baked frame instead of falling through to a live
        // resim that would fight the cache.
        if (sim_cache_valid_) {
            int want = tl_frame;
            if (want < sim_cache_start_frame_) want = sim_cache_start_frame_;
            if (want > sim_cache_end_frame_)   want = sim_cache_end_frame_;
            bool cache_frame_ready = (want == sim_timeline_frame_);
            if (want != sim_timeline_frame_ || force_resync) {
                if (restoreSimFrameFromDisk(want, fixed_dt)) {
                    sim_timeline_frame_ = want;
                    cache_frame_ready = true;
                    changed = true;
                }
            }
            if (cache_frame_ready && rigid_timeline_frame_ != want) {
                syncRigidToFrame(want, fixed_dt, kMaxStepsPerTick);
                changed = true;
            }
            // ★★★ `|| force_resync`: a LOOK edit changes nothing about the
            // simulation, so `changed` stays false and the request would be
            // dropped on the floor — with the flag then cleared by the next
            // sync that happens for some other reason. That is the "picking a
            // material does nothing until I rewind, and sometimes I have to
            // Reset" report: the resync only landed when the timeline happened
            // to move. A resync request IS the change; honour it.
            if (changed || force_resync) syncSimulationRenderVolumes();
            return;
        }

        if (force_resync && !playing && restoreSimFrame(tl_frame, fixed_dt)) {
            sim_timeline_frame_ = tl_frame;
            syncRigidToFrame(tl_frame, fixed_dt, kMaxStepsPerTick);
            changed = true;
        }

        // Fresh bake on first entry. On a playback loop-back (tl_frame jumps
        // below our baked frame), do NOT drop the whole cache. Rigid bodies are
        // not frame-cached yet, so some rewinds still need a deterministic resim
        // from frame 0, but the grid/fluid cache remains useful for non-rigid
        // frames and for the next scrub/play pass.
        if (sim_timeline_frame_ < 0) {
            if (restoreSimFrame(tl_frame, fixed_dt)) {
                sim_timeline_frame_ = tl_frame;
                syncRigidToFrame(tl_frame, fixed_dt, kMaxStepsPerTick);
            } else {
                resetSimulationToStart(false, false);
                sim_timeline_frame_ = 0;
            }
            changed = true;
        } else if (playing && tl_frame < sim_timeline_frame_) {
            if (restoreSimFrame(tl_frame, fixed_dt)) {
                sim_timeline_frame_ = tl_frame;
                syncRigidToFrame(tl_frame, fixed_dt, kMaxStepsPerTick);
                changed = true;
            } else {
                resetSimulationToStart(false, false);
                sim_timeline_frame_ = 0;
                changed = true;
            }
        }

        if (tl_frame != sim_timeline_frame_) {
            if (restoreSimFrame(tl_frame, fixed_dt)) {
                sim_timeline_frame_ = tl_frame;
                syncRigidToFrame(tl_frame, fixed_dt, kMaxStepsPerTick);
                changed = true;
            } else {
                // Uncached: rewind to nearest cached <= target, then resim (capped).
                if (tl_frame < sim_timeline_frame_) {
                    const int nearest = nearestCachedSimFrameAtOrBelow(tl_frame);
                    if (nearest >= 0 && restoreSimFrame(nearest, fixed_dt)) {
                        sim_timeline_frame_ = nearest;
                        syncRigidToFrame(nearest, fixed_dt, kMaxStepsPerTick);
                    } else {
                        resetSimulationToStart(false, false);
                        sim_timeline_frame_ = 0;
                    }
                    changed = true;
                }
                // Resume soft bodies from the (cached) frame we're stepping FROM, so
                // crossing the cache boundary continues the cloth/soft motion instead
                // of rebuilding it at rest and re-animating from the start.
                soft_resume_frame_ = sim_timeline_frame_;
                soft_resume_dt_ = fixed_dt;
                int steps = 0;
                while (sim_timeline_frame_ < tl_frame && steps < kMaxStepsPerTick) {
                    // Re-pose keyframed sim-source objects (e.g. moving colliders)
                    // for the frame we are about to step INTO, so the solid mask
                    // tracks the animated geometry instead of freezing at one pose.
                    applySimSourceObjectPosesForFrame(sim_timeline_frame_ + 1);
                    syncSimulationWorld();
                    // Keys are evaluated against the frame being stepped INTO —
                    // the same one the source poses above were just set to. The
                    // world's own frame_ counter is not that number: it only ever
                    // gets re-anchored by resetTime, so it drifts across scrubs.
                    simulation_world.setTimelineFrame(sim_timeline_frame_ + 1);
                    simulation_world.stepOnce(fixed_dt);
                    processFractureImpacts();  // shatter on impact above threshold
                    emitCombustionStructuralImpulses(fixed_dt);  // fire -> blast
                    processStructuralImpulseEvents();  // ...and by blast overpressure
                    ++sim_timeline_frame_;
                    rigid_timeline_frame_ = sim_timeline_frame_;
                    captureSimFrame(sim_timeline_frame_);
                    ++steps;
                    changed = true;
                }
                soft_resume_frame_ = -1;  // one-shot: only the boundary step resumes
                // If the bake hasn't caught up to the displayed frame this tick,
                // restore the playhead pose so the viewport collider doesn't lag
                // behind the timeline while the remaining frames bake.
                if (sim_timeline_frame_ != tl_frame) {
                    applySimSourceObjectPosesForFrame(tl_frame);
                }
            }
        }

        if (sim_timeline_frame_ >= 0 && rigid_timeline_frame_ != sim_timeline_frame_) {
            syncRigidToFrame(sim_timeline_frame_, fixed_dt, kMaxStepsPerTick);
            changed = true;
        }

        // Only touch the renderer when something actually changed; otherwise the
        // timeline is frozen and the path tracer is allowed to converge + idle.
        //
        // ★★★ A pending resync counts as changed. Without it a look edit on a
        // PAUSED timeline was silently discarded whenever the frame could not be
        // restored from the RAM cache (freshly loaded project, or a frame the
        // bake never captured): nothing moved, so `changed` was false, so the
        // rebuild never ran and the request evaporated. The user's workaround
        // was to rewind or Reset, i.e. to force something else to move.
        if (changed || force_resync) {
            syncSimulationRenderVolumes();
        }
    }

    // Deterministic per-frame simulation driver for the SEQUENCE RENDER worker.
    // Unlike updateSimulationTimeline (capped at kMaxStepsPerTick to keep the UI
    // responsive across ticks), this drives the sim to EXACTLY tl_frame in one
    // blocking call — during a sequence render the worker owns the timeline and
    // there is no UI tick to spread the work across, so the first rendered frame
    // may need an unbounded bake from 0..start_frame. After stepping it rebuilds
    // the SurfaceSDF volumes AND the discrete particle / foam render instances so
    // splat / foam / SurfaceSDF all appear in the rendered frame (the viewport
    // gets these for free from updateSimulationTimeline + syncParticleRenderInstances;
    // render_Animation previously did neither).
    //
    // MUST be called on the render worker thread ONLY. While a sequence render is
    // active the UI's updateSimulationTimeline + syncParticleRenderInstances are
    // gated off (render_owns_timeline / skip_backend_for_anim), so the worker is
    // the single owner of sim state + the render bridge groups — no concurrent
    // writes. The particle/foam bridge self-flags g_scene_geometry_generation /
    // g_optix_rebuild_pending / g_gpu_refit_pending on structural / motion change;
    // the caller consumes those to drive the backend AS rebuild before tracing.
    //
    // cache_frames=false (the sequence-render default): do NOT accumulate the
    // per-frame snapshot cache. A sequence walks frames forward exactly once and
    // never scrubs back, but captureSimFrame deep-copies the FULL grid + the
    // entire FluidParticles / FoamParticles SoA for every frame — O(N) per frame
    // in copy cost and O(N × frames) in resident memory. On a long filling fluid
    // that ballooning cache is what makes the sequence "start fast then crawl".
    // Forward stepping works straight off the LIVE sim state, so the cache buys
    // nothing here; we drop it and only fall back to a reset+resim when the
    // target is BEHIND the live frame (which a forward sequence never hits).
    void bakeSimulationForRenderFrame(int tl_frame, float fps, bool enable_rt_geometry = true,
                                      bool cache_frames = false) {
        if (tl_frame < 0) tl_frame = 0;
        const float fixed_dt = (fps > 1.0f) ? (1.0f / fps) : (1.0f / 24.0f);

        // Release any cache the viewport left behind so a long sequence doesn't
        // sit on (and keep growing) hundreds of full-state snapshots.
        if (!cache_frames) {
            clearSimFrameCache();
        } else if (restoreSimFrame(tl_frame, fixed_dt)) {
            // Exact cached frame → restore and done.
            sim_timeline_frame_ = tl_frame;
            syncRigidToFrame(tl_frame, fixed_dt, tl_frame + 1);
            syncSimulationRenderVolumes();
            syncParticleRenderInstances(enable_rt_geometry);
            return;
        }

        // Rewind to the nearest cached frame <= target (or reset to 0), then
        // resimulate forward UNCAPPED to the exact target frame. With caching
        // off the live state is already correct for forward steps, so the rewind
        // only triggers on a genuine backward jump (reset + resim from 0).
        if (sim_timeline_frame_ < 0 || tl_frame < sim_timeline_frame_) {
            const int nearest = cache_frames ? nearestCachedSimFrameAtOrBelow(tl_frame) : -1;
            if (nearest >= 0 && restoreSimFrame(nearest, fixed_dt)) {
                sim_timeline_frame_ = nearest;
                syncRigidToFrame(nearest, fixed_dt, nearest + 1);
            } else {
                resetSimulationToStart();
                sim_timeline_frame_ = 0;
            }
        }
        // Resume soft bodies from the cached frame we're stepping FROM (same fix as
        // the interactive path): rebuilt-from-rest soft bodies would otherwise restart
        // their motion when the bake resumes past the cache.
        soft_resume_frame_ = sim_timeline_frame_;
        soft_resume_dt_ = fixed_dt;
        while (sim_timeline_frame_ < tl_frame) {
            // Re-pose keyframed sim-source objects for the frame being stepped
            // into. Matters for the first rendered frame (bakes 0..start_frame in
            // one call) and any backward-jump reset+resim — without it a moving
            // collider stays frozen at the render frame's pose for the whole bake.
            applySimSourceObjectPosesForFrame(sim_timeline_frame_ + 1);
            syncSimulationWorld();
            simulation_world.setTimelineFrame(sim_timeline_frame_ + 1);
            simulation_world.stepOnce(fixed_dt);
            processFractureImpacts();  // shatter on impact above threshold
            emitCombustionStructuralImpulses(fixed_dt);  // fire -> blast
            processStructuralImpulseEvents();  // ...and by blast overpressure
            ++sim_timeline_frame_;
            rigid_timeline_frame_ = sim_timeline_frame_;
            if (cache_frames) captureSimFrame(sim_timeline_frame_);
        }
        soft_resume_frame_ = -1;  // one-shot: only the boundary step resumes

        // SurfaceSDF volumes (level-set → NanoVDB) + discrete particle / foam
        // render instances. Order matches the viewport's per-tick drive.
        syncSimulationRenderVolumes();
        syncParticleRenderInstances(enable_rt_geometry);
    }

    // ── VDB export ───────────────────────────────────────────────────────────
    // Write a grid domain's current state (density/temperature/fuel/flame) to a
    // .vdb file. World placement is baked into the grid transform (origin+voxel).
    bool exportDomainVDB(std::size_t system_index, std::size_t domain_index, const std::string& filepath) {
        if (system_index >= particle_systems.size()) return false;
        auto& sys = particle_systems[system_index];
        if (!sys.runtime) return false;
        const auto& states = sys.runtime->gridDomainStates();
        if (domain_index >= states.size()) return false;
        const auto& g = states[domain_index].grid;
        if (g.nx <= 0 || g.ny <= 0 || g.nz <= 0) return false;
        return VDBVolumeManager::exportDenseGridToVDB(
            filepath, g.nx, g.ny, g.nz, g.voxel_size,
            g.origin.x, g.origin.y, g.origin.z,
            g.density.empty() ? nullptr : g.density.data(),
            g.temperature.empty() ? nullptr : g.temperature.data(),
            g.fuel.empty() ? nullptr : g.fuel.data(),
            g.interaction.empty() ? nullptr : g.interaction.data());
    }

    // Deterministic bake from frame 0; writes frames [start,end] as base_####.vdb.
    // Blocking (explicit user action). Returns the number of files written. Leaves
    // the simulation back in free-run.
    int exportDomainVDBSequence(std::size_t system_index, std::size_t domain_index,
                                const std::string& directory, const std::string& base,
                                int start_frame, int end_frame, float fps) {
        if (system_index >= particle_systems.size() || end_frame < start_frame) return 0;
        const float dt = (fps > 1.0f) ? (1.0f / fps) : (1.0f / 24.0f);
        resetSimulationToStart();
        sim_timeline_frame_ = 0;
        rigid_timeline_frame_ = 0;
        int written = 0;
        for (int f = 0; f <= end_frame; ++f) {
            if (f > 0) {
                syncSimulationWorld();
                simulation_world.setTimelineFrame(f);
                simulation_world.stepOnce(dt);
                sim_timeline_frame_ = f;
                rigid_timeline_frame_ = f;
            }
            if (f >= start_frame) {
                std::string num = std::to_string(f);
                while (num.size() < 4) num = "0" + num;
                const std::string path = directory + "/" + base + "_" + num + ".vdb";
                if (exportDomainVDB(system_index, domain_index, path)) ++written;
            }
        }
        sim_timeline_frame_ = -1;
        rigid_timeline_frame_ = -1;
        syncSimulationRenderVolumes();
        return written;
    }

    // ── On-disk bake (render-only point cache) ───────────────────────────────
    // Deterministic config-signature of a system, used to detect a stale bake on
    // load. Intentionally hashes only AUTHORED, stable fields (source identity,
    // counts, resolution, physics, emitter params, collider material). Resolved
    // geometry (object-bound bounds / OBB / sphere / capsule) is EXCLUDED — it is
    // re-derived live from the source object each step, so it drifts between save
    // and load and would otherwise cause false "bake outdated" invalidations.
    uint64_t computeSystemConfigHash(const ParticleSystemObject& sys) const {
        uint64_t h = 1469598103934665603ull; // FNV-1a offset basis
        auto B = [&h](const void* d, size_t n) {
            const uint8_t* p = static_cast<const uint8_t*>(d);
            for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
        };
        auto I = [&](int64_t v) { B(&v, sizeof(v)); };
        auto S = [&](const std::string& s) { B(s.data(), s.size()); uint32_t n = (uint32_t)s.size(); B(&n, sizeof(n)); };

        if (!sys.runtime) return h;
        auto& rt = *sys.runtime;
        // STRUCTURAL IDENTITY ONLY. The hash must match between bake time (live
        // config) and load time (config parsed back from the project file), so it
        // hashes ONLY fields that the project serializer round-trips losslessly:
        // enum modes, source names, and counts. Floats (gravity, viscosity, …),
        // emitter seeds, and resolution are deliberately EXCLUDED — they are
        // either not all serialized or drift (Adaptive auto-resize), which would
        // make the hash mismatch and the cache silently never bind on reload.
        // This still catches the meaningful changes (Gas↔Fluid, collider type,
        // physics mode, add/remove of any domain/emitter/collider/flow source).
        // A pure parameter tweak won't invalidate — acceptable; the user re-bakes.
        I((int64_t)rt.physicsSettings().mode);
        const auto& doms = rt.gridDomains();
        I((int64_t)doms.size());
        for (const auto& d : doms) {
            I((int64_t)d.type); I((int64_t)d.source_mode); I((int64_t)d.boundary_mode);
            S(d.source_name);
        }
        const auto& ems = rt.emitters();
        I((int64_t)ems.size());
        for (const auto& e : ems) {
            I((int64_t)e.source_mode); S(e.source_name);
        }
        const auto& cols = rt.colliders();
        I((int64_t)cols.size());
        for (const auto& c : cols) {
            I((int64_t)c.source_mode); S(c.source_name);
        }
        const auto& fss = rt.flowSources();
        I((int64_t)fss.size());
        for (const auto& fsd : fss) {
            I((int64_t)fsd.source_mode); S(fsd.source_name);
        }
        return h;
    }

    // Deterministically bake frames [start,end] (re-simulated from 0) to a disk
    // point cache: one binary file per (system, frame) + a manifest carrying the
    // per-system config signatures. Render-only — see SimCache.h. BLOCKING: walks
    // the whole sim on the calling thread (an explicit user action; the UI should
    // run it on a worker). Binds the cache so scrubbing serves from disk at once.
    // Blocking convenience wrapper (runs the cooperative bake to completion on
    // the calling thread without yielding). The interactive UI uses the
    // begin/tick/cancel state machine below instead so it never freezes.
    bool bakeSimulationToDisk(const std::string& cache_dir, int start_frame, int end_frame, float fps) {
        if (!beginSimulationDiskBake(cache_dir, start_frame, end_frame, fps)) return false;
        while (tickSimulationDiskBake(1.0e9)) { /* run to completion */ }
        return sim_cache_valid_;
    }

    // ── Cooperative disk bake: begin / tick / cancel ─────────────────────────
    // Start a frame-driven bake. Does the one-time setup (clear folder, snapshot
    // authored config hashes, reset + synchronize, write frame 0 if in range) and
    // arms the state machine. Returns false if a bake is already running or the
    // request is invalid. Drive it each UI tick with tickSimulationDiskBake.
    bool beginSimulationDiskBake(const std::string& cache_dir, int start_frame, int end_frame, float fps) {
        if (sim_bake_active_) return false;                                  // already baking
        // Bake needs SOMETHING to cache: a particle/fluid/gas system OR a rigid/
        // soft body. A cloth-only scene has no particle_systems but still bakes its
        // soft deformation — previously this guard bailed on empty particle_systems
        // so the Bodies-panel bake button did nothing without a fluid domain.
        if (cache_dir.empty() || end_frame < start_frame ||
            (particle_systems.empty() && rigid_bodies.empty() && soft_weld_cache_.empty())) return false;

        sim_bake_dir_    = cache_dir;
        sim_bake_start_  = start_frame;
        sim_bake_end_    = end_frame;
        sim_bake_fps_    = fps;
        sim_bake_dt_     = (fps > 1.0f) ? (1.0f / fps) : (1.0f / 24.0f);
        sim_bake_cancel_ = false;
        sim_bake_ok_     = true;

        RayTrophiSim::SimCache::clearCache(cache_dir);  // fresh folder for this bake

        // Snapshot config hashes from the AUTHORED config BEFORE simulating.
        // Adaptive domains auto-resize their resolution while the sim runs, and
        // bounds get resolved live — computing the hash post-bake would capture
        // those derived values and never match the freshly-loaded (authored) hash
        // on reload, so the cache would silently never bind.
        sim_bake_hashes_.clear();
        sim_bake_hashes_.reserve(particle_systems.size());
        for (const auto& sys : particle_systems) {
            sim_bake_hashes_.emplace_back(sys.id, computeSystemConfigHash(sys));
        }

        resetSimulationToStart();
        // resetGridDomainStates() leaves states DEFAULT-constructed (type = Gas,
        // empty grid). Synchronize now so frame 0 carries the correct domain type
        // + grid metadata — otherwise a reloaded bake restores a "gas" frame 0.
        for (auto& sys : particle_systems) {
            if (sys.runtime) sys.runtime->synchronizeGridDomainsNow();
        }
        sim_timeline_frame_ = 0;
        rigid_timeline_frame_ = 0;
        sim_bake_cur_ = 0;
        if (0 >= start_frame) {                 // frame 0 has no step; write if in range
            if (!writeAllSystemsBakeFrame_(0)) sim_bake_ok_ = false;
        }
        sim_bake_active_ = true;
        return true;
    }

    // Advance the active bake for up to budget_ms of wall time, then yield. Steps
    // the sim + writes each frame in range. Returns true while the bake is still
    // running (call again next tick), false once it finished or was cancelled (in
    // which case it has already written the manifest + bound the cache, or cleared
    // a partial bake on cancel, and refreshed the render volumes).
    bool tickSimulationDiskBake(double budget_ms) {
        if (!sim_bake_active_) return false;
        if (sim_bake_cancel_) { finalizeSimulationDiskBake_(true); return false; }

        const auto t0 = std::chrono::steady_clock::now();
        while (sim_bake_cur_ < sim_bake_end_) {
            ++sim_bake_cur_;
            // Re-pose keyframed sim-source objects (e.g. animated colliders) for the
            // frame we are about to step INTO, so the solid mask tracks the moving
            // geometry instead of freezing at the reset pose — same as the live
            // timeline + sequence-render drivers. Also drops the moved objects from
            // the surface-cache epoch memo so their next resolve rebuilds from the
            // new world verts (static colliders stay memoized and cheap).
            applySimSourceObjectPosesForFrame(sim_bake_cur_);
            syncSimulationWorld();
            simulation_world.setTimelineFrame(sim_bake_cur_);
            simulation_world.stepOnce(sim_bake_dt_);
            sim_timeline_frame_ = sim_bake_cur_;
            rigid_timeline_frame_ = sim_bake_cur_;
            if (sim_bake_cur_ >= sim_bake_start_) {
                if (!writeAllSystemsBakeFrame_(sim_bake_cur_)) sim_bake_ok_ = false;
            }
            if (sim_bake_cancel_) { finalizeSimulationDiskBake_(true); return false; }
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            if (elapsed >= budget_ms) break;     // hand the frame back to the UI
        }
        if (sim_bake_cur_ >= sim_bake_end_) {
            finalizeSimulationDiskBake_(false);
            return false;
        }
        return true;
    }

    void cancelSimulationDiskBake() { if (sim_bake_active_) sim_bake_cancel_ = true; }

    bool  isSimulationBaking()  const { return sim_bake_active_; }
    int   simBakeCurrentFrame() const { return sim_bake_cur_; }
    int   simBakeStartFrame()   const { return sim_bake_start_; }
    int   simBakeEndFrame()     const { return sim_bake_end_; }
    float simBakeProgress()     const {
        if (sim_bake_end_ <= 0) return sim_bake_active_ ? 0.0f : 1.0f;
        return std::clamp(static_cast<float>(sim_bake_cur_) / static_cast<float>(sim_bake_end_), 0.0f, 1.0f);
    }

private:
    // Write every particle system's current grid-domain state for frame f to the
    // active bake folder. Returns false if any system's write failed.
    bool writeAllSystemsBakeFrame_(int f) {
        bool ok = true;
        for (auto& sys : particle_systems) {
            if (!sys.runtime) continue;
            // Burn/heat damage is captured in the SAME call as the grid states:
            // it is the one thing in this cache that the sim cannot re-derive on
            // load, so a frame written without it bakes a fire with nothing burnt.
            const auto msf = sys.runtime->captureMaterialStateFieldsForCache(
                simulation_world.compute());
            if (!RayTrophiSim::SimCache::writeSystemFrame(
                    sim_bake_dir_, sys.id, f, sys.runtime->gridDomainStates(), msf)) {
                ok = false;
            }
        }
        if (!writeSoftBodiesBakeFrame_(f)) ok = false;
        if (!writeRigidBodiesBakeFrame_(f)) ok = false;
        return ok;
    }

    // ★ Rigid motion belongs in the bake exactly like the grid domains do.
    //
    // It used to be the one moving thing in the scene with no on-disk frame: the
    // disk replay path restored the grid/soft frames and then RE-SIMULATED Jolt
    // to catch up (syncRigidToFrame -> advanceRigidTimelineToFrame). Two things
    // then go wrong every play/reset:
    //   • the re-sim is capped at max_steps per UI tick, so the rigid timeline
    //     lags the fluid frame being displayed — anything parented to a body
    //     (a flame on a falling match) reads a DIFFERENT frame's pose than the
    //     fire it is supposed to be lighting;
    //   • Jolt re-simulated against restored-but-not-resumable fluid state does
    //     not reproduce the run that was baked, so the result changes run to run.
    // Writing the frame makes rigid replay a lookup, same as everything else.
    bool writeRigidBodiesBakeFrame_(int f) {
        if (!rigid_body_system || rigid_bodies.empty()) return true;
        std::vector<RayTrophiSim::RigidBodyFrameState> bodies;
        rigid_body_system->captureFrameState(bodies);
        if (bodies.empty()) return true;
        return RayTrophiSim::SimCache::writeRigidFrame(sim_bake_dir_, f, bodies);
    }

    // Write every soft body's deformed world vertices for frame f (alongside the
    // fluid frames, same folder). No-op (success) when there are no soft bodies.
    bool writeSoftBodiesBakeFrame_(int f) {
        if (soft_weld_cache_.empty()) return true;
        std::map<std::string, std::vector<Vec3>> snap;
        snapshotSoftBodies(snap);
        if (snap.empty()) return true;
        std::vector<RayTrophiSim::SimCache::SoftBodyFrame> bodies;
        bodies.reserve(snap.size());
        for (auto& kv : snap) {
            RayTrophiSim::SimCache::SoftBodyFrame b;
            b.name = kv.first;
            b.vertices = std::move(kv.second);
            bodies.push_back(std::move(b));
        }
        return RayTrophiSim::SimCache::writeSoftFrame(sim_bake_dir_, f, bodies);
    }

    // End the active bake: on success write the manifest + bind the cache so
    // scrubbing streams from disk; on cancel drop the partial folder so a
    // half-bake is never bound. Either way return to free-run + refresh volumes.
    void finalizeSimulationDiskBake_(bool cancelled) {
        if (!cancelled) {
            RayTrophiSim::SimCache::Manifest m;
            m.version = RayTrophiSim::SimCache::kVersion;
            m.start_frame = sim_bake_start_;
            m.end_frame = sim_bake_end_;
            m.fps = sim_bake_fps_;
            for (const auto& sys : particle_systems) {
                RayTrophiSim::SimCache::SystemManifest sm;
                sm.id = sys.id;
                sm.config_hash = 0;
                for (const auto& kv : sim_bake_hashes_) {
                    if (kv.first == sys.id) { sm.config_hash = kv.second; break; }
                }
                sm.domain_count = sys.runtime ? (int)sys.runtime->gridDomainStates().size() : 0;
                m.systems.push_back(sm);
            }
            if (!RayTrophiSim::SimCache::writeManifest(sim_bake_dir_, m)) sim_bake_ok_ = false;

            sim_cache_dir_ = sim_bake_dir_;
            sim_cache_valid_ = sim_bake_ok_;
            sim_cache_start_frame_ = sim_bake_start_;
            sim_cache_end_frame_ = sim_bake_end_;
            if (sim_bake_ok_) {
                sim_cache_valid_system_ids_.clear();
                for (const auto& sys : particle_systems) {
                    sim_cache_valid_system_ids_.insert(sys.id);
                }
                clearSimFrameCache();
            }
        } else {
            RayTrophiSim::SimCache::clearCache(sim_bake_dir_);  // drop the half-bake
            sim_bake_ok_ = false;
            sim_cache_valid_system_ids_.clear();
        }

        sim_timeline_frame_ = -1;       // back to free-run; disk now serves restores
        rigid_timeline_frame_ = -1;
        syncSimulationRenderVolumes();
        sim_bake_active_ = false;
        sim_bake_hashes_.clear();
    }

public:

    // Validate a cache folder against the CURRENT systems' config and bind it so
    // restoreSimFrame streams from disk. Returns false (and leaves the cache
    // unbound) on missing/old manifest or any per-system config-hash mismatch —
    // the caller can then surface a "bake outdated, re-bake" hint. Called by the
    // project loader after particle_systems are restored.
    bool setSimDiskCache(const std::string& cache_dir) {
        sim_cache_valid_ = false;
        sim_cache_dir_.clear();
        sim_cache_valid_system_ids_.clear();

        // Sync signatures to loaded baseline scene to prevent false invalidations on first tick
        last_sim_config_sig_ = computeSimConfigSignature();
        last_fluid_coupling_sig_ = computeFluidCouplingSignature();

        SCENE_LOG_INFO("[SimDiskCache] Attempting to bind cache directory: " + cache_dir);

        RayTrophiSim::SimCache::Manifest m;
        if (!RayTrophiSim::SimCache::readManifest(cache_dir, m)) {
            SCENE_LOG_WARN("[SimDiskCache] Failed to read manifest.json from: " + cache_dir);
            return false;
        }
        if (m.version != RayTrophiSim::SimCache::kVersion) {
            SCENE_LOG_WARN("[SimDiskCache] Manifest version mismatch (cache: " + 
                           std::to_string(m.version) + ", expected: " + 
                           std::to_string(RayTrophiSim::SimCache::kVersion) + ")");
            return false;
        }

        SCENE_LOG_INFO("[SimDiskCache] Manifest loaded successfully. Range: [" + 
                       std::to_string(m.start_frame) + ", " + std::to_string(m.end_frame) + 
                       "] at " + std::to_string(m.fps) + " FPS. Systems in manifest: " + 
                       std::to_string(m.systems.size()));

        for (const auto& sys : particle_systems) {
            const uint64_t want = computeSystemConfigHash(sys);
            char hex_want[32];
            std::snprintf(hex_want, sizeof(hex_want), "0x%016llx", static_cast<unsigned long long>(want));

            bool found = false;
            bool matched = false;
            for (const auto& sm : m.systems) {
                if (sm.id == sys.id) {
                    found = true;
                    if (sm.config_hash == want) {
                        matched = true;
                    } else {
                        char hex_got[32];
                        std::snprintf(hex_got, sizeof(hex_got), "0x%016llx", static_cast<unsigned long long>(sm.config_hash));
                        SCENE_LOG_WARN("[SimDiskCache] System ID " + std::to_string(sys.id) + 
                                       " config hash mismatch. Scene wants: " + hex_want + 
                                       ", Cache has: " + hex_got);
                    }
                    break;
                }
            }
            if (!found) {
                SCENE_LOG_WARN("[SimDiskCache] System ID " + std::to_string(sys.id) + 
                               " not found in cache manifest.");
            }

            if (matched) {
                // Check if the frame file actually exists on disk!
                if (RayTrophiSim::SimCache::frameExists(cache_dir, sys.id, m.start_frame)) {
                    sim_cache_valid_system_ids_.insert(sys.id);
                    SCENE_LOG_INFO("[SimDiskCache] System ID " + std::to_string(sys.id) + 
                                   " successfully validated and bound to cache.");
                } else {
                    SCENE_LOG_WARN("[SimDiskCache] System ID " + std::to_string(sys.id) + 
                                   " matched hash but frame files are missing on disk.");
                }
            }
        }

        bool has_any_cache = !sim_cache_valid_system_ids_.empty() || 
                             RayTrophiSim::SimCache::softFrameExists(cache_dir, m.start_frame);

        if (!has_any_cache) {
            SCENE_LOG_WARN("[SimDiskCache] No valid fluid or soft frame files found in cache directory.");
            return false;
        }

        sim_cache_dir_ = cache_dir;
        sim_cache_start_frame_ = m.start_frame;
        sim_cache_end_frame_ = m.end_frame;
        sim_cache_valid_ = true;
        SCENE_LOG_INFO("[SimDiskCache] Cache successfully bound. sim_cache_valid_ = true");
        return true;
    }

    void clearSimDiskCacheBinding() {
        sim_cache_valid_ = false;
        sim_cache_dir_.clear();
        sim_cache_valid_system_ids_.clear();
    }

    bool hasValidParticleSimDiskCache() const {
        if (!sim_cache_valid_ || sim_cache_dir_.empty()) return false;
        if (sim_cache_valid_system_ids_.empty()) return false;
        uint32_t first_id = *sim_cache_valid_system_ids_.begin();
        return RayTrophiSim::SimCache::frameExists(sim_cache_dir_, first_id, sim_cache_start_frame_);
    }

    bool hasValidSoftSimDiskCache() const {
        if (!sim_cache_valid_ || sim_cache_dir_.empty()) return false;
        return RayTrophiSim::SimCache::softFrameExists(sim_cache_dir_, sim_cache_start_frame_);
    }

    void clearParticleSimDiskCache() {
        if (sim_cache_dir_.empty()) return;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(sim_cache_dir_, ec)) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("sys", 0) == 0 && entry.path().extension() == ".rtfc") {
                std::filesystem::remove(entry.path(), ec);
            }
        }
        sim_cache_valid_system_ids_.clear();
        if (!hasValidSoftSimDiskCache()) {
            RayTrophiSim::SimCache::clearCache(sim_cache_dir_);
            sim_cache_valid_ = false;
            sim_cache_dir_.clear();
        }
    }

    void clearSoftSimDiskCache() {
        if (sim_cache_dir_.empty()) return;
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator(sim_cache_dir_, ec)) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("soft", 0) == 0 && entry.path().extension() == ".rtfc") {
                std::filesystem::remove(entry.path(), ec);
            }
        }
        if (sim_cache_valid_system_ids_.empty()) {
            RayTrophiSim::SimCache::clearCache(sim_cache_dir_);
            sim_cache_valid_ = false;
            sim_cache_dir_.clear();
        }
    }

    bool hasValidSimDiskCache() const { return sim_cache_valid_; }
    const std::string& simDiskCacheDir() const { return sim_cache_dir_; }

    // Canonical cache-folder location for a project file: "<dir>/<stem>.simcache"
    // (project name without extension), e.g. scene.rtproj → scene.simcache.
    static std::string simCacheDirForProject(const std::string& project_path) {
        if (project_path.empty()) return std::string();
        std::filesystem::path p(project_path);
        return (p.parent_path() / (p.stem().string() + ".simcache")).string();
    }

    // Gas/volumetric grid disk cache directory. Mirrors simCacheDirForProject
    // but uses ".volcache" to keep fluid particle cache and gas grid cache
    // separate. Used by Cinema profile and manual disk-bake workflows.
    static std::string volCacheDirForProject(const std::string& project_path) {
        if (project_path.empty()) return std::string();
        std::filesystem::path p(project_path);
        return (p.parent_path() / (p.stem().string() + ".volcache")).string();
    }

private:
    // Implemented in FluidDomainRenderLifecycle.cpp so representation ownership
    // does not grow this already-large scene header further.
    bool hasAuthoritativeGridFluidDomain(const std::string& name) const;
    void retireDomainSurfaceRepresentation(ParticleSystemObject& system,
                                           std::size_t domain_index);

    void invalidateSimulationRenderBindings(ParticleSystemObject& system) {
        // Invalidates NanoVDB host/GPU bindings so the next syncSimulationRenderVolumes
        // re-registers + re-uploads density from the restored/reset sim state.
        // Does NOT trigger a full TLAS rebuild — the VDBVolume objects and their TLAS
        // instances remain valid. Only the NanoVDB buffer contents change, which is
        // handled cheaply by re-registering via registerOrUpdateLiveVolume and setting
        // g_gas_volumes_dirty (SSBO re-sync). Setting rebuild pending here caused a full
        // GPU TLAS rebuild on every cached frame restore (restoreSimFrame), i.e. every
        // single frame during timeline playback.
        //
        // ★DO NOT "optimise" the unbind away. Dropping the id to -1 is what makes the
        // next registerOrUpdateLiveVolume call take the prev_id < 0 branch, and THAT is
        // what forces do_update unconditionally — a guaranteed fresh upload for every
        // restored frame. Replacing it with an upload-signature reset alone looks
        // equivalent but is not: it leaves the upload subject to the density_ptr_override
        // / stride conditions, and the surface then failed to appear even on the first
        // cached frame (tried 2026-07-31, reverted).
        auto& mgr = VDBVolumeManager::getInstance();
        for (std::size_t d = 0; d < system.domain_vdb_ids.size(); ++d) {
            if (system.domain_vdb_ids[d] >= 0) {
                mgr.unloadVDB(system.domain_vdb_ids[d]);
                system.domain_vdb_ids[d] = -1;
                g_gas_volumes_dirty = true;
            }
            if (d < system.domain_volumes.size() && system.domain_volumes[d]) {
                system.domain_volumes[d]->setVDBVolumeID(-1);
                // Keep visible=true so the TLAS customIndex→SSBO slot mapping stays
                // intact. The volume will re-upload density on next sync without
                // triggering a became_visible rebuild cycle.
                //
                // ★ visible=true alone did NOT keep the mapping intact: with the id
                // at -1 the volume reads as !isLoaded(), and updateGeometry drops
                // unloaded volumes from the TLAS entirely. Any rebuild landing in
                // this window — switching to Vulkan RT performs one — deleted the
                // liquid surface permanently. Mark the rebind as expected so the
                // slot survives the window it was always assumed to survive.
                system.domain_volumes[d]->awaiting_live_rebind = true;
            }
            if (d < system.domain_vdb_upload_signatures.size()) {
                system.domain_vdb_upload_signatures[d] = 0;
            }
        }
        g_gas_volumes_dirty = true;
    }

    void removeDomainVolume(ParticleSystemObject& system, std::size_t d) {
        auto& mgr = VDBVolumeManager::getInstance();
        if (d < system.domain_vdb_ids.size() && system.domain_vdb_ids[d] >= 0) {
            mgr.unloadVDB(system.domain_vdb_ids[d]);
            system.domain_vdb_ids[d] = -1;
        }
        if (d < system.domain_volumes.size() && system.domain_volumes[d]) {
            auto vol = system.domain_volumes[d];
            removeVDBVolume(vol);
            auto it = std::find(world.objects.begin(), world.objects.end(),
                                std::static_pointer_cast<Hittable>(vol));
            if (it != world.objects.end()) {
                world.objects.erase(it);
            }
            system.domain_volumes[d].reset();
            g_geometry_dirty = true;
            g_vulkan_rebuild_pending = true;
            g_optix_rebuild_pending = true;
            g_gas_volumes_dirty = true;
            // Drop the now-erased volume node from the CPU BVH too, else the CPU
            // reference render keeps hitting the stale/dangling volume AABB and the
            // domain shows black (e.g. when a fluid domain switches to Particles
            // mode, which tears the volume down). Symmetric with the create path.
            g_bvh_rebuild_pending = true;
        }
        if (d < system.domain_sdf_signatures.size()) {
            system.domain_sdf_signatures[d] = 0;
        }
        if (d < system.domain_vdb_upload_signatures.size()) {
            system.domain_vdb_upload_signatures[d] = 0;
        }
    }

    void removeFoamDomainVolume(ParticleSystemObject& system, std::size_t d) {
        auto& mgr = VDBVolumeManager::getInstance();
        if (d < system.domain_foam_vdb_ids.size() && system.domain_foam_vdb_ids[d] >= 0) {
            mgr.unloadVDB(system.domain_foam_vdb_ids[d]);
            system.domain_foam_vdb_ids[d] = -1;
        }
        if (d < system.domain_foam_volumes.size() && system.domain_foam_volumes[d]) {
            auto vol = system.domain_foam_volumes[d];
            removeVDBVolume(vol);
            auto it = std::find(world.objects.begin(), world.objects.end(),
                                std::static_pointer_cast<Hittable>(vol));
            if (it != world.objects.end()) world.objects.erase(it);
            system.domain_foam_volumes[d].reset();
            g_geometry_dirty = true;
            g_vulkan_rebuild_pending = true;
            g_optix_rebuild_pending = true;
            g_gas_volumes_dirty = true;
            g_bvh_rebuild_pending = true;  // drop stale node from CPU BVH (see removeDomainVolume)
        }
    }

    void destroyDomainVolumes(ParticleSystemObject& system) {
        for (std::size_t d = 0; d < system.domain_volumes.size(); ++d) {
            removeDomainVolume(system, d);
        }
        for (std::size_t d = 0; d < system.domain_foam_volumes.size(); ++d) {
            removeFoamDomainVolume(system, d);
        }
        system.domain_vdb_ids.clear();
        system.domain_volumes.clear();
        system.domain_foam_vdb_ids.clear();
        system.domain_foam_volumes.clear();
        system.domain_foam_density.clear();
        // Surface-route render artifacts share the per-domain lifetime.
        system.domain_sdf_buffers.clear();
        system.domain_uvw_buffers.clear();
        system.domain_sdf_stats.clear();
        system.domain_sdf_signatures.clear();
        system.domain_vdb_upload_signatures.clear();
        system.domain_last_fluid_render_mode.clear();
        system.domain_last_tuned_shader.clear();
    }

    bool buildFluidDensityVolume(RayTrophiSim::Fluid::FluidObject& obj,
                                 FluidRenderBinding& binding,
                                 int& active_cells) {
        obj.ensureGrid();
        const auto& grid = obj.grid;
        active_cells = 0;
        if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
            grid.voxel_size <= 0.0f || obj.particles.empty()) {
            binding.density.clear();
            return false;
        }

        const std::size_t cell_count = grid.getCellCount();
        binding.density.assign(cell_count, 0.0f);
        const float inv_h = 1.0f / grid.voxel_size;
        const float per_particle_density =
            1.0f / static_cast<float>(std::max(1, obj.params.particles_per_cell));

        for (const Vec3& p : obj.particles.position) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                continue;
            }

            const Vec3 local = (p - grid.origin) * inv_h - Vec3(0.5f, 0.5f, 0.5f);
            const int i0 = static_cast<int>(std::floor(local.x));
            const int j0 = static_cast<int>(std::floor(local.y));
            const int k0 = static_cast<int>(std::floor(local.z));
            const float fx = local.x - static_cast<float>(i0);
            const float fy = local.y - static_cast<float>(j0);
            const float fz = local.z - static_cast<float>(k0);

            for (int dz = 0; dz <= 1; ++dz) {
                const int k = k0 + dz;
                if (k < 0 || k >= grid.nz) continue;
                const float wz = dz ? fz : (1.0f - fz);
                for (int dy = 0; dy <= 1; ++dy) {
                    const int j = j0 + dy;
                    if (j < 0 || j >= grid.ny) continue;
                    const float wy = dy ? fy : (1.0f - fy);
                    for (int dx = 0; dx <= 1; ++dx) {
                        const int i = i0 + dx;
                        if (i < 0 || i >= grid.nx) continue;
                        const float wx = dx ? fx : (1.0f - fx);
                        binding.density[grid.cellIndex(i, j, k)] +=
                            per_particle_density * wx * wy * wz;
                    }
                }
            }
        }

        for (float d : binding.density) {
            if (d > 1e-5f) {
                ++active_cells;
            }
        }
        return active_cells > 0;
    }

    // Phase 2: rebuild the narrow-band SDF from the live particles and write
    // a "density proxy" channel that fades across the surface band, so the
    // existing volume backend produces a recognizable liquid shape even
    // before the dedicated isosurface render mode (Phase 3) is wired up.
    // After Phase 3 lands the GPU side will read `obj.sdf` directly.
    bool buildFluidSurfaceVolume(RayTrophiSim::Fluid::FluidObject& obj,
                                 FluidRenderBinding& binding,
                                 int& active_cells) {
        obj.ensureGrid();
        const auto& grid = obj.grid;
        active_cells = 0;
        const bool bad_grid = (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
                               grid.voxel_size <= 0.0f);
        const bool no_particles = obj.particles.empty();
        // Which of the three causes stopped the surface — the caller's gate can
        // only report "built=0", and the three have completely different fixes.
        SCENE_LOG_ON_CHANGE("fluidsurf.cause." + obj.name,
            (bad_grid ? 1 : 0) + (no_particles ? 2 : 0),
            std::string("[VolumeGate 1a] fluid '") + obj.name + "' surface input: grid=" +
            std::to_string(grid.nx) + "x" + std::to_string(grid.ny) + "x" +
            std::to_string(grid.nz) + " voxel=" + std::to_string(grid.voxel_size) +
            " particles=" + std::to_string(obj.particles.size()) +
            (bad_grid ? " -> BAD GRID" : (no_particles ? " -> NO PARTICLES" : " -> ok")));
        if (bad_grid || no_particles) {
            binding.density.clear();
            obj.sdf.clear();
            return false;
        }

        const bool built = RayTrophiSim::Fluid::buildLevelSet(
            obj.particles, grid, obj.level_set_params, obj.sdf, &obj.level_set_stats);
        SCENE_LOG_ON_CHANGE("fluidsurf.levelset." + obj.name, built ? 1 : 0,
            std::string("[VolumeGate 1b] fluid '") + obj.name + "' buildLevelSet " +
            (built ? "succeeded again" : "FAILED with " +
                     std::to_string(obj.particles.size()) + " particles"));
        if (!built) {
            binding.density.clear();
            return false;
        }

        // SDF may be refined above the sim grid (surface_resolution_multiplier),
        // so size the proxy from the SDF buffer itself, not the sim cell count.
        const std::size_t cell_count = obj.sdf.size();
        binding.density.assign(cell_count, 0.0f);
        // Surface-centred density proxy (matches the SimulationGridDomain path):
        //   density = clamp(0.5 - 0.5 * phi / grad_width, 0, 1)
        // phi=0 (surface) -> 0.5 (shader iso threshold); symmetric ramp keeps a
        // valid gradient (surface normal) across the whole band for both thin
        // and thick fluid. See the domain-path comment for the matte-pool fix.
        // grad_width stays PHYSICAL (sim voxel) so the band is invariant to the
        // surface refinement — phi is a physical distance regardless of m.
        const float voxel = grid.voxel_size;
        const float grad_width = std::max(1.0f, obj.surface_band_voxels) * voxel;
        const float inv_w = 0.5f / grad_width;
        std::size_t live = 0;
        for (std::size_t i = 0; i < cell_count; ++i) {
            const float phi = obj.sdf[i];
            float d = 0.5f - phi * inv_w;
            if (d < 0.0f) d = 0.0f;
            if (d > 1.0f) d = 1.0f;
            binding.density[i] = d;
            if (d > 1e-5f) ++live;
        }
        active_cells = static_cast<int>(live);
        return live > 0;
    }

    void syncFluidRenderVolumes(VDBVolumeManager& mgr, int frame, bool force_sync) {
        std::unordered_set<uint32_t> alive_ids;
        for (auto& obj : fluid_objects) {
            // Modern liquid domains keep this object only for legacy API/editor
            // identity. Its particles/render mode are not authoritative and
            // must never publish a second SDF beside the grid-domain route.
            if (hasAuthoritativeGridFluidDomain(obj.name)) {
                destroyFluidRenderVolume(obj.id);
                continue;
            }
            // Volume + SurfaceSDF routes share this density / level-set volume.
            // Particles route is handled exclusively by ParticleRenderBridge —
            // tear any prior density binding down so the two routes never both
            // contribute scene geometry at the same time.
            const bool wants_volume_route =
                obj.render_mode == RayTrophiSim::Fluid::FluidRenderMode::Volume ||
                obj.render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
            if (!wants_volume_route) {
                destroyFluidRenderVolume(obj.id);
                continue;
            }
            alive_ids.insert(obj.id);
            FluidRenderBinding& binding = fluid_render_bindings[obj.id];
            int active_cells = 0;
            const bool is_surface_route =
                obj.render_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
            const bool built = is_surface_route
                ? buildFluidSurfaceVolume(obj, binding, active_cells)
                : buildFluidDensityVolume(obj, binding, active_cells);
            const bool renderable = obj.visible && obj.enabled && built;

            // GATE 1 of 4 — the producer refused to build a surface at all.
            SCENE_LOG_ON_CHANGE("fluidvol.build." + obj.name, renderable ? 1 : 0,
                std::string("[VolumeGate 1/4] fluid '") + obj.name + "' surface " +
                (renderable ? "BUILT again"
                            : std::string("NOT built (visible=") +
                              (obj.visible ? "1" : "0") + " enabled=" +
                              (obj.enabled ? "1" : "0") + " built=" +
                              (built ? "1" : "0") + " cells=" +
                              std::to_string(active_cells) + ")"));

            if (!renderable) {
                destroyFluidRenderVolume(obj.id);
                continue;
            }

            const auto& grid = obj.grid;
            const long long cells =
                static_cast<long long>(grid.nx) * grid.ny * grid.nz;
            int stride = 1;
            if (cells >= 160LL * 160 * 160) stride = 3;
            else if (cells >= 104LL * 104 * 104) stride = 2;

            const bool do_update =
                force_sync || binding.vdb_id < 0 || ((frame % stride) == 0);
            if (do_update) {
                // SurfaceSDF may be refined above the sim grid — upload at its
                // effective resolution (same origin/extent, finer voxels).
                const auto& ls = obj.level_set_stats;
                const bool refined_upload =
                    is_surface_route && ls.eff_nx > 0 &&
                    static_cast<std::size_t>(ls.eff_nx) *
                    static_cast<std::size_t>(ls.eff_ny) *
                    static_cast<std::size_t>(ls.eff_nz) == binding.density.size();
                const int   up_nx    = refined_upload ? ls.eff_nx : grid.nx;
                const int   up_ny    = refined_upload ? ls.eff_ny : grid.ny;
                const int   up_nz    = refined_upload ? ls.eff_nz : grid.nz;
                const float up_voxel = refined_upload ? ls.eff_voxel : grid.voxel_size;
                // ★★ Was this volume without a live grid until now?
                //
                // The rebuild flags below fire only on `created` — the one frame
                // the VDBVolume OBJECT is made. But the object can outlive its
                // GRID: a backend switch drops the live volume registration while
                // the object stays in world.objects, and the SurfaceSDF is
                // rebuilt continuously anyway. On the frames in between, the
                // volume has a TLAS slot but no content, and the SSBO publish
                // writes it as an inactive (invisible) slot.
                //
                // Nothing then republishes when the grid comes back, because the
                // per-frame publishes are gated on animation flags a regenerating
                // fluid never raises — so the surface renders on the rebuild
                // frame and disappears on the next publish, permanently. Treat
                // "grid (re)appeared" exactly like "object created".
                const bool live_grid_returned = (binding.vdb_id < 0);
                binding.vdb_id = mgr.registerOrUpdateLiveVolume(
                    binding.vdb_id,
                    obj.name + " [Fluid NanoVDB]",
                    up_nx, up_ny, up_nz,
                    up_voxel,
                    binding.density.data(),
                    nullptr,
                    nullptr);
                simulation_render_updated = true;
                g_gas_volumes_dirty = true;
                // GATE 2 of 4 — registration itself failed, so the object keeps a
                // stale id the manager no longer knows about.
                SCENE_LOG_ON_CHANGE("fluidvol.register." + obj.name,
                    binding.vdb_id >= 0 ? 1 : 0,
                    std::string("[VolumeGate 2/4] fluid '") + obj.name +
                    (binding.vdb_id >= 0
                        ? std::string("' registered live grid id=") + std::to_string(binding.vdb_id)
                        : std::string("' registerOrUpdateLiveVolume FAILED (id=-1)")));
                // The grid just came back on an existing volume. Only a Vulkan
                // geometry rebuild runs updateGeometry + syncVDBVolumesToGPU as
                // one publication, so it is the only thing that can turn the
                // slot back on; g_gas_volumes_dirty above drives the legacy gas
                // path and never touches this SSBO.
                if (live_grid_returned && binding.vdb_id >= 0) {
                    g_geometry_dirty = true;
                    g_vulkan_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                }
            }

            if (binding.vdb_id < 0) {
                continue;
            }

            const Vec3 world_min = grid.origin;
            const Vec3 world_max = grid.origin +
                Vec3(static_cast<float>(grid.nx) * grid.voxel_size,
                     static_cast<float>(grid.ny) * grid.voxel_size,
                     static_cast<float>(grid.nz) * grid.voxel_size);

            bool created = false;
            if (!binding.volume) {
                auto vol = std::make_shared<VDBVolume>();
                vol->transient = true;
                vol->name = obj.name + " [Fluid NanoVDB]";
                if (!obj.shader) obj.shader = VolumeShader::createSmokePreset();
                vol->setShader(obj.shader);
                binding.volume = vol;
                addVDBVolume(vol);
                world.objects.push_back(vol);
                created = true;
                binding.last_render_mode = -1;  // force preset (re)tune below
            }

            // (Re)tune the volume shader when the render mode changes or the
            // binding has just been created. The density-proxy used by
            // SurfaceSDF mode is a sharp 0..1 band (0..voxel wide), so it
            // needs a tighter density multiplier than the diffuse splatted
            // density of Volume mode.
            const int cur_mode = static_cast<int>(obj.render_mode);
            if (cur_mode != binding.last_render_mode && binding.volume) {
                if (!obj.shader) obj.shader = VolumeShader::createSmokePreset();
                binding.volume->setShader(obj.shader);
                auto shader = obj.shader;
                if (shader) {
                    if (is_surface_route) {
                        shader->name = "Liquid Surface (SDF Proxy)";
                        shader->density.multiplier = 12.0f;        // band is 0..1, push it opaque
                        shader->density.cutoff_threshold = 0.05f;
                        shader->scattering.color = Vec3(0.45f, 0.70f, 0.92f);
                        shader->scattering.coefficient = 2.5f;
                        shader->scattering.anisotropy = 0.2f;
                        shader->absorption.color = Vec3(0.08f, 0.18f, 0.32f);
                        shader->absorption.coefficient = 0.6f;
                        shader->emission.mode = VolumeEmissionMode::None;
                    } else {
                        shader->name = "Liquid Preview";
                        shader->density.multiplier = 1.6f;
                        shader->density.cutoff_threshold = 0.01f;
                        shader->scattering.color = Vec3(0.62f, 0.78f, 0.92f);
                        shader->scattering.coefficient = 1.1f;
                        shader->scattering.anisotropy = 0.0f;
                        shader->absorption.color = Vec3(0.0f, 0.0f, 0.0f);
                        shader->absorption.coefficient = 0.04f;
                        shader->emission.mode = VolumeEmissionMode::None;
                    }
                    binding.volume->setShader(shader);
                }
                binding.last_render_mode = cur_mode;
            }

            binding.volume->visible = true;
            binding.volume->render_as_isosurface = is_surface_route;
            if (is_surface_route) {
                binding.volume->render_isosurface_ior = obj.surface_ior;
                binding.volume->render_isosurface_roughness = obj.surface_roughness;
                binding.volume->render_isosurface_foam = obj.surface_foam;
            }
            binding.volume->bindLiveVolume(binding.vdb_id, grid.voxel_size, world_min, world_max);

            if (created) {
                g_geometry_dirty = true;
                g_vulkan_rebuild_pending = true;
                g_optix_rebuild_pending = true;
                g_gas_volumes_dirty = true;
            }
        }

        std::vector<uint32_t> stale;
        stale.reserve(fluid_render_bindings.size());
        for (const auto& kv : fluid_render_bindings) {
            if (alive_ids.find(kv.first) == alive_ids.end()) {
                stale.push_back(kv.first);
            }
        }
        for (uint32_t id : stale) {
            destroyFluidRenderVolume(id);
        }
    }

    void destroyFluidRenderVolume(uint32_t id) {
        auto it = fluid_render_bindings.find(id);
        if (it == fluid_render_bindings.end()) {
            return;
        }

        auto& mgr = VDBVolumeManager::getInstance();
        FluidRenderBinding& binding = it->second;
        // ★ Only a binding that actually HELD something is a scene change.
        //
        // syncFluidRenderVolumes reaches this through
        // `fluid_render_bindings[obj.id]`, and operator[] INSERTS. An empty
        // domain therefore ran insert -> build fails -> destroy every frame, and
        // the unconditional dirty flags below turned that into a full TLAS
        // rebuild on every single frame for as long as the domain stayed empty.
        // Nothing was torn down; there was nothing there to tear down.
        const bool had_content = (binding.vdb_id >= 0) || static_cast<bool>(binding.volume);
        if (binding.vdb_id >= 0) {
            mgr.unloadVDB(binding.vdb_id);
            binding.vdb_id = -1;
        }
        if (binding.volume) {
            auto vol = binding.volume;
            removeVDBVolume(vol);
            auto obj_it = std::find(world.objects.begin(), world.objects.end(),
                                    std::static_pointer_cast<Hittable>(vol));
            if (obj_it != world.objects.end()) {
                world.objects.erase(obj_it);
            }
            binding.volume.reset();
        }
        fluid_render_bindings.erase(it);
        if (!had_content) {
            return;
        }
        g_geometry_dirty = true;
        g_vulkan_rebuild_pending = true;
        g_optix_rebuild_pending = true;
        g_gas_volumes_dirty = true;
    }

    void destroyAllFluidRenderVolumes() {
        std::vector<uint32_t> ids;
        ids.reserve(fluid_render_bindings.size());
        for (const auto& kv : fluid_render_bindings) {
            ids.push_back(kv.first);
        }
        for (uint32_t id : ids) {
            destroyFluidRenderVolume(id);
        }
    }

public:

    ParticleSystemObject* activeParticleSystemObject() {
        if (active_particle_system_index < 0 ||
            active_particle_system_index >= static_cast<int>(particle_systems.size())) {
            return nullptr;
        }
        return &particle_systems[static_cast<std::size_t>(active_particle_system_index)];
    }

    const ParticleSystemObject* activeParticleSystemObject() const {
        if (active_particle_system_index < 0 ||
            active_particle_system_index >= static_cast<int>(particle_systems.size())) {
            return nullptr;
        }
        return &particle_systems[static_cast<std::size_t>(active_particle_system_index)];
    }

    std::shared_ptr<RayTrophiSim::ParticleSimulationSystem> activeParticleRuntime() const {
        if (const auto* active_system = activeParticleSystemObject()) {
            return active_system->runtime;
        }
        return nullptr;
    }

    // Runtime is now authoritative; these legacy round-trip hooks are retained as
    // no-ops so existing UI/serialization call sites keep compiling. Per-system
    // edits flow straight into the owning runtime, so no copy-back is needed.
    void syncActiveParticleSystemObjectFromRuntime() {}
    void applyActiveParticleSystemObjectToRuntime() {}

    // Push an object's enabled/visible flags down into its runtime.
    static void applyParticleSystemEnabledState(ParticleSystemObject& system) {
        if (system.runtime) {
            system.runtime->setEnabled(system.enabled && system.visible);
        }
    }

    ParticleSystemObject& addParticleSystemObject(const std::string& requested_name = "Particle System") {
        ParticleSystemObject system;
        system.id = next_particle_system_id++;
        system.name = requested_name.empty() ? "Particle System" : requested_name;
        if (system.name == "Particle System") {
            system.name += " " + std::to_string(system.id);
        }
        system.runtime = createParticleRuntime();
        applyParticleSystemEnabledState(system);
        particle_systems.push_back(std::move(system));
        active_particle_system_index = static_cast<int>(particle_systems.size()) - 1;
        syncRigidBodyProxyColliders();
        return particle_systems.back();
    }

    ParticleSystemObject& ensureActiveParticleSystemObject() {
        if (!activeParticleSystemObject()) {
            return addParticleSystemObject();
        }
        return *activeParticleSystemObject();
    }

    // One-click behaviour presets. Each builds a fully-configured system (physics +
    // emitter/domain + RT render look) so the user does not have to dial in the many
    // particle/domain/shader knobs by hand. 0=Campfire, 1=Explosion, 2=Smoke.
    // Explosion is the general airburst; GroundBurst is the dirt-and-debris
    // ground detonation that climbs; Fireball is the slow fuel-rich mushroom.
    enum class ParticleSystemPreset {
        Campfire = 0,
        Explosion = 1,
        Smoke = 2,
        GroundBurst = 3,
        Fireball = 4,
        Flamethrower = 5,
        BurningFuelSpill = 6,
        IgnitedFuelJet = 7
    };

    ParticleSystemObject& addParticleSystemPreset(
        ParticleSystemPreset preset);

    bool setActiveParticleSystemObject(std::size_t index) {
        if (index >= particle_systems.size()) {
            return false;
        }
        // Selection only: every system keeps simulating regardless of which is active.
        active_particle_system_index = static_cast<int>(index);
        return true;
    }

    void unregisterParticleRuntime(ParticleSystemObject& system) {
        destroyDomainVolumes(system);
        destroyParticleRenderGroup(system);
        // Cache is indexed per-system; system add/remove invalidates it.
        clearSimFrameCache();
        sim_timeline_frame_ = -1;
        rigid_timeline_frame_ = -1;
        if (system.runtime) {
            system.runtime->releaseComputeResources(simulation_world.compute());
            simulation_world.removeSystem(system.runtime.get());
            system.runtime.reset();
        }
    }

    bool removeParticleSystemObject(std::size_t index) {
        if (index >= particle_systems.size()) {
            return false;
        }

        unregisterParticleRuntime(particle_systems[index]);
        particle_systems.erase(particle_systems.begin() + static_cast<std::ptrdiff_t>(index));
        if (particle_systems.empty()) {
            active_particle_system_index = -1;
        } else if (active_particle_system_index >= static_cast<int>(particle_systems.size())) {
            active_particle_system_index = static_cast<int>(particle_systems.size()) - 1;
        } else if (static_cast<int>(index) < active_particle_system_index) {
            --active_particle_system_index;
        }
        return true;
    }

    void clearParticleSystemObjects() {
        for (auto& system : particle_systems) {
            unregisterParticleRuntime(system);
        }
        particle_systems.clear();
        active_particle_system_index = -1;
        next_particle_system_id = 1;
    }

    bool anyParticleRuntimeEnabled() const {
        // UI hot-path guard only: do not dereference runtime shared_ptrs here.
        // Runtime lists can be re-bound by the simulation/render bridges later
        // in the frame; the actual timeline/update path validates each runtime.
        return !particle_systems.empty();
    }

    bool anySimulationRuntimeEnabled() const {
        // Structural presence check only. The simulation update path performs
        // per-object enabled/runtime checks; this UI predicate must not walk or
        // dereference scene-owned runtime objects while transient NanoVDB/live
        // volume bindings may be mutating adjacent scene containers.
        return !particle_systems.empty() || !fluid_objects.empty() || !gas_volumes.empty() ||
               !rigid_bodies.empty();
    }

    // True only if node_name is actually referenced as a sim source (particle
    // emitter/collider/grid-domain/flow-source or rigid body). Used to gate
    // per-frame gizmo bounds refresh so it doesn't force a full surface-mesh
    // rebuild for large meshes that have nothing to do with simulation.
    bool isObjectUsedAsSimSource(const std::string& node_name) const {
        if (node_name.empty()) return false;
        for (const auto& rb : rigid_bodies) {
            if (rb.source_name == node_name) return true;
        }
        for (const auto& system : particle_systems) {
            if (!system.runtime) continue;
            for (const auto& emitter : system.runtime->emitters()) {
                if (emitter.source_mode == RayTrophiSim::ParticleEmitterSourceMode::ObjectOrigin &&
                    emitter.source_name == node_name) return true;
            }
            for (const auto& collider : system.runtime->colliders()) {
                if (collider.source_name == node_name) return true;
            }
            for (const auto& domain : system.runtime->gridDomains()) {
                if (domain.source_mode == RayTrophiSim::SimulationGridDomainSourceMode::ObjectBounds &&
                    domain.source_name == node_name) return true;
            }
            for (const auto& source : system.runtime->flowSources()) {
                if ((source.source_mode == RayTrophiSim::SimulationFlowSourceMode::ObjectBounds ||
                     source.source_mode == RayTrophiSim::SimulationFlowSourceMode::MeshSurface) &&
                    source.source_name == node_name) return true;
            }
        }
        return false;
    }

    bool hasLiveSimulationObject(const std::string& node_name) const {
        if (node_name.empty() || isEditorPendingDeleteObjectName(node_name)) {
            return false;
        }
        for (const auto& obj : world.objects) {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                if (tri->getNodeName() == node_name) return true;
            } else if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                // Flat (direct SoA) mesh: no per-face facades but it IS a live sim-eligible object.
                // Without this gate the whole collider/OBB/surface-cache resolution returned null for
                // a flat mesh, so a flat STATIC body's collision shape was never built and cloth /
                // particles / rigid bodies fell straight through it (collision "didn't see" flat).
                if (tm->nodeName == node_name) return true;
            }
        }
        return false;
    }

    size_t removeParticleBindingsForObjectName(const std::string& node_name) {
        if (node_name.empty()) {
            return 0;
        }

        size_t removed = 0;
        auto pruneVectors = [&](auto& emitters, auto& colliders) {
            const auto emitter_before = emitters.size();
            emitters.erase(
                std::remove_if(emitters.begin(), emitters.end(),
                    [&](const RayTrophiSim::ParticleEmitterDesc& emitter) {
                        return emitter.source_mode == RayTrophiSim::ParticleEmitterSourceMode::ObjectOrigin &&
                               emitter.source_name == node_name;
                    }),
                emitters.end());
            removed += emitter_before - emitters.size();

            const auto collider_before = colliders.size();
            colliders.erase(
                std::remove_if(colliders.begin(), colliders.end(),
                    [&](const RayTrophiSim::ParticleColliderDesc& collider) {
                        return (collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectAABB ||
                                collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectOBB ||
                                collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::Sphere ||
                                collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::Capsule) &&
                               collider.source_name == node_name;
                    }),
                colliders.end());
            removed += collider_before - colliders.size();

        };

        for (auto& system : particle_systems) {
            if (system.runtime) {
                pruneVectors(system.runtime->emitters(), system.runtime->colliders());
                auto& domains = system.runtime->gridDomains();
                const auto domain_before = domains.size();
                domains.erase(
                    std::remove_if(domains.begin(), domains.end(),
                        [&](const RayTrophiSim::SimulationGridDomainDesc& domain) {
                            return domain.source_mode == RayTrophiSim::SimulationGridDomainSourceMode::ObjectBounds &&
                                   domain.source_name == node_name;
                        }),
                    domains.end());
                removed += domain_before - domains.size();
                auto& flow_sources = system.runtime->flowSources();
                const auto flow_before = flow_sources.size();
                flow_sources.erase(
                    std::remove_if(flow_sources.begin(), flow_sources.end(),
                        [&](const RayTrophiSim::SimulationFlowSourceDesc& source) {
                            return (source.source_mode == RayTrophiSim::SimulationFlowSourceMode::ObjectBounds ||
                                    source.source_mode == RayTrophiSim::SimulationFlowSourceMode::MeshSurface) &&
                                   source.source_name == node_name;
                        }),
                    flow_sources.end());
                removed += flow_before - flow_sources.size();
            }
        }
        return removed;
    }

    size_t pruneInvalidParticleObjectBindings() {
        if (particle_systems.empty()) {
            return 0;
        }

        size_t removed = 0;
        auto pruneVectors = [&](auto& emitters, auto& colliders) {
            const auto emitter_before = emitters.size();
            emitters.erase(
                std::remove_if(emitters.begin(), emitters.end(),
                    [&](const RayTrophiSim::ParticleEmitterDesc& emitter) {
                        return emitter.source_mode == RayTrophiSim::ParticleEmitterSourceMode::ObjectOrigin &&
                               !hasLiveSimulationObject(emitter.source_name);
                    }),
                emitters.end());
            removed += emitter_before - emitters.size();

            const auto collider_before = colliders.size();
            colliders.erase(
                std::remove_if(colliders.begin(), colliders.end(),
                    [&](const RayTrophiSim::ParticleColliderDesc& collider) {
                        return (collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectAABB ||
                                collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectOBB ||
                                collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::Sphere ||
                                collider.source_mode == RayTrophiSim::ParticleColliderSourceMode::Capsule) &&
                               !collider.source_name.empty() &&
                               !hasLiveSimulationObject(collider.source_name);
                    }),
                colliders.end());
            removed += collider_before - colliders.size();
        };

        for (auto& system : particle_systems) {
            if (system.runtime) {
                pruneVectors(system.runtime->emitters(), system.runtime->colliders());
                auto& domains = system.runtime->gridDomains();
                const auto domain_before = domains.size();
                domains.erase(
                    std::remove_if(domains.begin(), domains.end(),
                        [&](const RayTrophiSim::SimulationGridDomainDesc& domain) {
                            return domain.source_mode == RayTrophiSim::SimulationGridDomainSourceMode::ObjectBounds &&
                                   !domain.source_name.empty() &&
                                   !hasLiveSimulationObject(domain.source_name);
                        }),
                    domains.end());
                removed += domain_before - domains.size();
                auto& flow_sources = system.runtime->flowSources();
                const auto flow_before = flow_sources.size();
                flow_sources.erase(
                    std::remove_if(flow_sources.begin(), flow_sources.end(),
                        [&](const RayTrophiSim::SimulationFlowSourceDesc& source) {
                            return (source.source_mode == RayTrophiSim::SimulationFlowSourceMode::ObjectBounds ||
                                    source.source_mode == RayTrophiSim::SimulationFlowSourceMode::MeshSurface) &&
                                   !source.source_name.empty() &&
                                   !hasLiveSimulationObject(source.source_name);
                        }),
                    flow_sources.end());
                removed += flow_before - flow_sources.size();
            }
        }
        return removed;
    }

    void invalidateSurfaceMeshCache(const std::string& node_name = std::string()) const {
        if (node_name.empty()) {
            surface_mesh_cache.clear();
            simulation_local_bounds_.clear();
            last_sim_pose_applied_.clear();  // drop stale sim-pose memo on full reset/reload
        } else {
            surface_mesh_cache.erase(node_name);
            simulation_local_bounds_.erase(node_name);
            last_sim_pose_applied_.erase(node_name);
        }
        ++surface_mesh_cache_version;
    }

    // Drop ONLY the per-epoch rebuild memo for one node (not the cache entry, not
    // the version) so the next bounds/OBB resolve rebuilds from current world
    // verts. Cheap — lets the collider/bounds gizmos track an object that moved
    // without bumping the geometry generation (manual gizmo drag mid-edit).
    void refreshSimSourceGizmoBounds(const std::string& node_name) const {
        if (!node_name.empty()) surface_cache_epoch_done_.erase(node_name);
    }

    const RayTrophiSim::SurfaceMeshCache* getSurfaceMeshCacheForObject(const std::string& node_name,
                                                                       bool refresh = true) const {
        if (node_name.empty() || !hasLiveSimulationObject(node_name)) {
            return nullptr;
        }

        auto existing = surface_mesh_cache.find(node_name);
        if (!refresh && existing != surface_mesh_cache.end()) {
            return &existing->second;
        }

        // Per-epoch memo: skip the rebuild if this object was already refreshed in
        // the current geometry epoch. A new epoch (gizmo/proc edit bumps
        // g_scene_geometry_generation; keyframe re-pose erases posed names from the
        // set, see applySimSourceObjectPosesForFrame) drops the memo so the rescan
        // runs exactly once after anything that can move world vertices.
        if (refresh) {
            const uint64_t gen = g_scene_geometry_generation.load(std::memory_order_acquire);
            if (gen != surface_cache_epoch_gen_) {
                surface_cache_epoch_gen_ = gen;
                surface_cache_epoch_done_.clear();
            }
            if (existing != surface_mesh_cache.end() &&
                !existing->second.empty() &&
                surface_cache_epoch_done_.find(node_name) != surface_cache_epoch_done_.end()) {
                return &existing->second; // already rebuilt this epoch
            }
        }

        std::vector<std::shared_ptr<Triangle>> triangles;
        for (const auto& obj : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->getNodeName() == node_name) {
                triangles.push_back(tri);
            }
        }

        if (triangles.empty()) {
            auto cache_it = base_mesh_cache.find(node_name);
            if (cache_it != base_mesh_cache.end()) {
                triangles = cache_it->second;
            }
        }

        if (triangles.empty()) {
            // Flat (direct SoA) mesh: no per-face facades. Build the surface cache straight from the
            // mesh SoA so collider/OBB resolution (rigid body creation, particle colliders) works on
            // a flat mesh exactly like a facade-backed one.
            for (const auto& obj : world.objects) {
                auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj);
                if (!tm || tm->nodeName != node_name || !tm->geometry) continue;
                DNA::GeometryDetail* g = tm->geometry.get();
                const Vec3* P = g->get_attribute_data<Vec3>("P");
                const Vec3* P_orig = g->get_attribute_data<Vec3>("P_orig");
                const Vec2* uv = g->get_attribute_data<Vec2>("uv");
                const uint16_t* mat = g->get_attribute_data<uint16_t>("materialID");
                // GPU transform edits intentionally do not rebake P every frame.
                // Build the simulation surface in current world space from the
                // authoritative local P_orig + final matrix when available;
                // otherwise the SDF cook sees the pre-rotate/pre-scale P and its
                // later world_to_local conversion produces an offset grid.
                std::vector<Vec3> transformed_positions;
                if (P_orig && tm->transform) {
                    const std::size_t vertex_count = g->get_vertex_count();
                    transformed_positions.resize(vertex_count);
                    const Matrix4x4 final_transform = tm->transform->getFinal();
                    for (std::size_t i = 0; i < vertex_count; ++i) {
                        transformed_positions[i] =
                            final_transform.transform_point(P_orig[i]);
                    }
                    P = transformed_positions.data();
                }
                auto& fcache = surface_mesh_cache[node_name];
                fcache = RayTrophiSim::SurfaceMeshCache::buildFromSoA(
                    node_name, P, uv, mat, g->indices.data(), g->indices.size(),
                    surface_mesh_cache_version);
                if (refresh && !fcache.empty()) surface_cache_epoch_done_.insert(node_name);
                return fcache.empty() ? nullptr : &fcache;
            }
        }

        if (triangles.empty()) {
            surface_mesh_cache.erase(node_name);
            return nullptr;
        }

        auto& cache = surface_mesh_cache[node_name];
        cache = RayTrophiSim::SurfaceMeshCache::build(node_name, triangles, surface_mesh_cache_version);
        if (refresh && !cache.empty()) surface_cache_epoch_done_.insert(node_name);
        return cache.empty() ? nullptr : &cache;
    }

    bool resolveLightweightObjectOBBForSimulation(
        const std::string& node_name,
        RayTrophiSim::ParticleColliderOBB& out_obb) const {
        if (node_name.empty()) return false;

        Matrix4x4 world_matrix = Matrix4x4::identity();
        std::vector<std::shared_ptr<Triangle>> source_triangles;
        const TriangleMesh* flat_mesh = nullptr;
        const uint64_t generation =
            g_scene_geometry_generation.load(std::memory_order_acquire);
        auto& cached = simulation_local_bounds_[node_name];
        const bool rebuild_local_bounds =
            !cached.valid || cached.geometry_generation != generation;

        for (const auto& obj : world.objects) {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                if (tri->getNodeName() == node_name) {
                    if (source_triangles.empty()) world_matrix = tri->getTransformMatrix();
                    source_triangles.push_back(tri);
                    if (!rebuild_local_bounds) break;
                }
            }
        }
        if (source_triangles.empty()) for (const auto& obj : world.objects) {
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                if (inst->node_name == node_name && inst->source_triangles && !inst->source_triangles->empty()) {
                    world_matrix = inst->transform;
                    if (rebuild_local_bounds) source_triangles.assign(inst->source_triangles->begin(), inst->source_triangles->end());
                    else source_triangles.push_back(inst->source_triangles->front());
                    break;
                }
            }
            if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                if (tm->nodeName == node_name && tm->geometry) {
                    flat_mesh = tm.get();
                    world_matrix = tm->transform ? tm->transform->getFinal() : Matrix4x4::identity();
                    break;
                }
            }
        }
        if (source_triangles.empty() && !flat_mesh) return false;

        if (rebuild_local_bounds) {
            Vec3 local_min(std::numeric_limits<float>::max());
            Vec3 local_max(-std::numeric_limits<float>::max());
            bool have_point = false;
            auto includePoint = [&](const Vec3& p) {
                local_min = Vec3::min(local_min, p);
                local_max = Vec3::max(local_max, p);
                have_point = true;
            };
            if (!source_triangles.empty()) {
                for (const auto& tri : source_triangles) {
                    if (!tri) continue;
                    includePoint(tri->getOriginalVertexPosition(0));
                    includePoint(tri->getOriginalVertexPosition(1));
                    includePoint(tri->getOriginalVertexPosition(2));
                }
            } else {
                DNA::GeometryDetail* geometry = flat_mesh->geometry.get();
                // Flat-mesh P is world-baked (see the surface-cache path above).
                // Feeding it into local bounds and then applying world_matrix
                // rotates/scales the collider twice. Prefer the authored local
                // P_orig; older/procedural meshes without it are unbaked through
                // inverse(world) before entering the local-bounds cache.
                const Vec3* positions = geometry
                    ? geometry->get_attribute_data<Vec3>("P_orig") : nullptr;
                const Vec3* world_positions = geometry
                    ? geometry->get_attribute_data<Vec3>("P") : nullptr;
                const bool unbake_world_positions =
                    positions == nullptr && world_positions != nullptr;
                const Matrix4x4 world_to_local =
                    unbake_world_positions
                        ? world_matrix.inverse()
                        : Matrix4x4::identity();
                const Vec3* source_positions =
                    positions ? positions : world_positions;
                if (source_positions) {
                    for (uint32_t index : geometry->indices) {
                        Vec3 point = source_positions[index];
                        if (unbake_world_positions) {
                            point = world_to_local.transform_point(point);
                        }
                        includePoint(point);
                    }
                }
            }
            if (!have_point) {
                simulation_local_bounds_.erase(node_name);
                return false;
            }
            cached.min = local_min;
            cached.max = local_max;
            cached.geometry_generation = generation;
            cached.valid = true;
        }

        out_obb.local_bounds_min = cached.min;
        out_obb.local_bounds_max = cached.max;
        out_obb.local_to_world = world_matrix;
        return true;
    }

    // ★ Bounds come from the SAME authority as the OBB: the live world-space
    // surface cache. The "lightweight" local-bounds × transform-handle shortcut
    // that used to short-circuit this is gone — see resolveObjectOBBForSimulation
    // for why it silently regressed rotated colliders.
    // Current world matrix of a scene node, for parenting a simulation flow
    // source to an object.
    //
    // ★ This deliberately returns the object's TRANSFORM and not a box derived
    // from its geometry — the opposite choice from resolveObjectOBBForSimulation
    // below, and for a different job. That function needs an oriented box that
    // always matches the RENDERED triangles (see its regression guard). A parent
    // needs a stable frame with a fixed origin: an offset like "the match's tip"
    // must stay put in local space, and a mesh-derived frame would drift with
    // topology and give no usable pivot at all.
    // ★ PERF: the LOOKUP is memoized, the MATRIX never is. A parented emitter is
    // resolved several times per step (motion sampling + frame resolve, per
    // source, plus the viewport gizmo), and each miss used to be a full linear
    // scan of world.objects with a dynamic_pointer_cast per entry. Caching the
    // matrix instead would be wrong — the whole point is that the parent MOVES,
    // including under physics — so we cache only the object handle and re-read
    // its live transform every call. A stale/renamed/deleted entry is detected
    // by the expiry + name re-check below and falls back to a fresh scan.
    bool resolveObjectTransformForSimulation(const std::string& node_name,
                                             Matrix4x4& out_matrix) const {
        if (node_name.empty()) return false;

        // ★★ A Jolt-driven object's motion is NOT in its transform handle.
        //
        // RigidBodySystem deliberately BAKES the rigid delta into the mesh
        // vertices instead of moving the handle, because moving the handle of an
        // imported/non-TRS mesh corrupted it in the renderer. So for a falling
        // rigid body `transform->getFinal()` stays frozen at the SPAWN pose, and
        // a parented emitter read from it burns at the spot the object STARTED
        // — the object visibly falls while its flame stays behind.
        //
        // ★ Do NOT reach for last_written_pivot to fix that. It is the delta
        // composed onto initial_pivot, and initial_pivot is getPivotMatrix() —
        // the authored PIVOT POINT, not the object's world transform. It is
        // identity for objects that were never given a pivot, which drops the
        // parented source at the WORLD ORIGIN and lets it fall from there: a
        // flame burning in the middle of the floor, forever, no matter where the
        // object it is parented to actually is. (The gizmo hit this same trap.)
        //
        // The frozen handle IS the spawn pose, so the live world pose is simply
        // the rigid delta applied to it. Resolve the handle first, then compose.
        const Matrix4x4* rigid_delta = nullptr;
        for (const auto& rb : rigid_bodies) {
            if (rb.source_name != node_name) continue;
            if (rb.has_written) rigid_delta = &rb.last_rigid_delta;
            break;
        }
        auto finish = [&](bool found) {
            if (found && rigid_delta) out_matrix = (*rigid_delta) * out_matrix;
            return found;
        };

        auto readMatrix = [&](const std::shared_ptr<Hittable>& obj) -> bool {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                if (tri->getNodeName() != node_name) return false;
                out_matrix = tri->getTransformMatrix();
                return true;
            }
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                if (inst->node_name != node_name) return false;
                out_matrix = inst->transform;
                return true;
            }
            if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                if (tm->nodeName != node_name) return false;
                out_matrix = tm->transform ? tm->transform->getFinal()
                                           : Matrix4x4::identity();
                return true;
            }
            return false;
        };

        const auto cached = simulation_transform_lookup_.find(node_name);
        if (cached != simulation_transform_lookup_.end()) {
            if (auto obj = cached->second.lock()) {
                if (readMatrix(obj)) return finish(true);   // still the right object
            }
            simulation_transform_lookup_.erase(cached);  // gone or renamed
        }

        for (const auto& obj : world.objects) {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                if (tri->getNodeName() == node_name) {
                    out_matrix = tri->getTransformMatrix();
                    simulation_transform_lookup_[node_name] = obj;
                    return finish(true);
                }
            }
        }
        for (const auto& obj : world.objects) {
            if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
                if (inst->node_name == node_name) {
                    out_matrix = inst->transform;
                    simulation_transform_lookup_[node_name] = obj;
                    return finish(true);
                }
            }
            if (auto tm = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                if (tm->nodeName == node_name) {
                    out_matrix = tm->transform ? tm->transform->getFinal()
                                               : Matrix4x4::identity();
                    simulation_transform_lookup_[node_name] = obj;
                    return finish(true);
                }
            }
        }
        return false;
    }

    bool resolveObjectBoundsForSimulation(const std::string& node_name, Vec3& out_min, Vec3& out_max) const {
        const auto* surface_cache = getSurfaceMeshCacheForObject(node_name);
        if (!surface_cache) {
            return false;
        }
        out_min = surface_cache->bounds_min;
        out_max = surface_cache->bounds_max;
        return true;
    }

    // ★ REGRESSION GUARD — do NOT put a transform-handle shortcut in front of
    // this again.
    //
    // A "lightweight" path (local bounds from getOriginalVertexPosition × the
    // transform handle) was added later and short-circuited this function. It
    // reintroduced the exact bug the world-vertex derivation below was written to
    // fix: the collider box came out ground-aligned, so it was correct only while
    // the object was parallel to the ground and wrong (huge) at steep angles.
    // Symptom: "collider obje rotate olunca particle collider'ı yanlış görüyor".
    //
    // The reason the shortcut cannot work is stated below: live vertex positions
    // are rotated by paths that leave the transform handle out of sync, so
    // handle × original verts and the rendered geometry disagree. The surface
    // cache is the only representation that is always current (for flat meshes it
    // rebuilds world positions from P_orig × getFinal() itself).
    //
    // Cost is covered by the per-epoch memo in getSurfaceMeshCacheForObject: an
    // object rebuilds at most once per geometry epoch, which is what actually
    // removed the resolver cost — not the shortcut.
    bool resolveObjectOBBForSimulation(const std::string& node_name,
                                       RayTrophiSim::ParticleColliderOBB& out_obb) const {
        const auto* surface_cache = getSurfaceMeshCacheForObject(node_name);
        if (!surface_cache) {
            return false;
        }

        // Derive the oriented box DIRECTLY from current world-space vertices, so it
        // always matches the rendered geometry under any rotation/scale. We do NOT
        // trust the transform handle / original verts here: in practice the live
        // vertex positions can be rotated by paths that leave getTransformMatrix()
        // out of sync, which produced a ground-aligned (huge at steep angles) box.
        std::vector<Vec3> world_verts;
        bool have_frame = false;
        Vec3 frame_a, frame_b, frame_c;

        auto collectTriangle = [&](const RayTrophiSim::SurfaceMeshTriangle& tri) {
            const Vec3 a = tri.p0;
            const Vec3 b = tri.p1;
            const Vec3 c = tri.p2;
            world_verts.push_back(a);
            world_verts.push_back(b);
            world_verts.push_back(c);

            // Use the first non-degenerate triangle to define the box orientation.
            if (!have_frame) {
                const Vec3 e1 = b - a;
                const Vec3 e2 = c - a;
                if (e1.length() > 1e-5f && Vec3::cross(e1, e2).length() > 1e-8f) {
                    frame_a = a;
                    frame_b = b;
                    frame_c = c;
                    have_frame = true;
                }
            }
        };

        for (const auto& tri : surface_cache->triangles) {
            collectTriangle(tri);
        }

        if (world_verts.empty() || !have_frame) {
            return false;
        }

        // Build an orthonormal frame from the chosen triangle. Hand-rolled
        // normalization because Vec3::normalize() zeroes sub-millimetre vectors,
        // which kills tight triangle edges/normals.
        const auto unit = [](const Vec3& v, const Vec3& fallback) {
            const float len = v.length();
            return len > 1e-8f ? v * (1.0f / len) : fallback;
        };
        const Vec3 e1 = frame_b - frame_a;
        const Vec3 e2 = frame_c - frame_a;
        const Vec3 axis_n = unit(Vec3::cross(e1, e2), Vec3(0.0f, 1.0f, 0.0f)); // normal
        const Vec3 axis_b = unit(Vec3::cross(axis_n, unit(e1, Vec3(1.0f, 0.0f, 0.0f))), Vec3(0.0f, 0.0f, 1.0f));
        const Vec3 axis_t = Vec3::cross(axis_b, axis_n); // guaranteed orthonormal, in-plane

        Vec3 centroid(0.0f, 0.0f, 0.0f);
        for (const auto& v : world_verts) {
            centroid = centroid + v;
        }
        centroid = centroid * (1.0f / static_cast<float>(world_verts.size()));

        Vec3 min_bound(std::numeric_limits<float>::max());
        Vec3 max_bound(-std::numeric_limits<float>::max());
        for (const auto& v : world_verts) {
            const Vec3 d = v - centroid;
            const Vec3 local(Vec3::dot(d, axis_t), Vec3::dot(d, axis_b), Vec3::dot(d, axis_n));
            min_bound = Vec3::min(min_bound, local);
            max_bound = Vec3::max(max_bound, local);
        }

        // local_to_world: columns = (axis_t, axis_b, axis_n), translation = centroid.
        // world = centroid + x*axis_t + y*axis_b + z*axis_n.
        Matrix4x4 m = Matrix4x4::identity();
        m.m[0][0] = axis_t.x; m.m[1][0] = axis_t.y; m.m[2][0] = axis_t.z;
        m.m[0][1] = axis_b.x; m.m[1][1] = axis_b.y; m.m[2][1] = axis_b.z;
        m.m[0][2] = axis_n.x; m.m[1][2] = axis_n.y; m.m[2][2] = axis_n.z;
        m.m[0][3] = centroid.x; m.m[1][3] = centroid.y; m.m[2][3] = centroid.z;

        out_obb.local_bounds_min = min_bound;
        out_obb.local_bounds_max = max_bound;
        out_obb.local_to_world = m;
        return true;
    }

    bool sampleObjectSurfaceForSimulation(const std::string& node_name,
                                          uint32_t seed,
                                          RayTrophiSim::ParticleSurfaceSample& out_sample) const {
        const auto* surface_cache = getSurfaceMeshCacheForObject(node_name);
        if (!surface_cache) {
            return false;
        }

        RayTrophiSim::SurfaceMeshSample sample;
        if (!surface_cache->sample(seed, sample)) {
            return false;
        }
        out_sample.position = sample.position;
        out_sample.normal = sample.normal;
        return true;
    }

    // Wire scene-aware resolvers (object/force-field bound emitters & colliders)
    // into a runtime. Called once per runtime at creation; every system shares the
    // same resolution logic, keyed by emitter/collider source_name.
    void configureParticleRuntime(RayTrophiSim::ParticleSimulationSystem& runtime) {
        runtime.setEmitterSourceResolver(
                [this](const RayTrophiSim::ParticleEmitterDesc& emitter, Vec3& out_position, Vec3& out_direction) {
                    out_direction = emitter.direction;

                    if (emitter.source_mode == RayTrophiSim::ParticleEmitterSourceMode::ForceFieldOrigin) {
                        auto field = findForceFieldByName(emitter.source_name);
                        if (!field) {
                            return false;
                        }
                        out_position = field->position + emitter.local_offset;
                        if (out_direction.length() < 1e-5f) {
                            out_direction = field->direction.length() > 1e-5f ? field->direction : field->axis;
                        }
                        return true;
                    }

                    if (emitter.source_mode == RayTrophiSim::ParticleEmitterSourceMode::ObjectOrigin) {
                        Vec3 bounds_min;
                        Vec3 bounds_max;
                        if (!resolveObjectBoundsForSimulation(
                                emitter.source_name, bounds_min, bounds_max)) {
                            return false;
                        }
                        // Resolve from the current world bounds on every spawn
                        // tick. A cached mesh centroid can describe the import
                        // pose when the object is moved through its transform
                        // handle without rebaking vertex positions.
                        out_position = (bounds_min + bounds_max) * 0.5f +
                                       emitter.local_offset;
                        return true;
                    }

                    out_position = emitter.point + emitter.local_offset;
                    return true;
                });
            runtime.setEmitterBoundsResolver(
                [this](const RayTrophiSim::ParticleEmitterDesc& emitter, Vec3& out_min, Vec3& out_max) {
                    if (emitter.source_mode != RayTrophiSim::ParticleEmitterSourceMode::ObjectOrigin) {
                        return false;
                    }
                    return resolveObjectBoundsForSimulation(emitter.source_name, out_min, out_max);
                });
            runtime.setEmitterSurfaceSampler(
                [this](const RayTrophiSim::ParticleEmitterDesc& emitter,
                       uint32_t seed,
                       RayTrophiSim::ParticleSurfaceSample& out_sample) {
                    if (emitter.source_mode != RayTrophiSim::ParticleEmitterSourceMode::ObjectOrigin) {
                        return false;
                    }
                    return sampleObjectSurfaceForSimulation(emitter.source_name, seed, out_sample);
                });
            runtime.setColliderBoundsResolver(
                [this](const RayTrophiSim::ParticleColliderDesc& collider, Vec3& out_min, Vec3& out_max) {
                    if ((collider.source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectAABB &&
                         collider.source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectOBB &&
                         collider.source_mode != RayTrophiSim::ParticleColliderSourceMode::Sphere &&
                         collider.source_mode != RayTrophiSim::ParticleColliderSourceMode::Capsule) ||
                        collider.source_name.empty()) {
                        return false;
                    }
                    return resolveObjectBoundsForSimulation(collider.source_name, out_min, out_max);
                });
            runtime.setColliderOBBResolver(
                [this](const RayTrophiSim::ParticleColliderDesc& collider,
                       RayTrophiSim::ParticleColliderOBB& out_obb) {
                    if ((collider.source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectOBB &&
                         collider.source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF &&
                         collider.source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectConvexDecomp &&
                         collider.source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectMeshBVH) ||
                        collider.source_name.empty()) {
                        return false;
                    }
                    return resolveObjectOBBForSimulation(collider.source_name, out_obb);
                });
            runtime.setColliderMeshResolver(
                [this](const RayTrophiSim::ParticleColliderDesc& collider,
                       std::vector<RayTrophiSim::SurfaceMeshTriangle>& out_triangles,
                       uint64_t& out_version) {
                    if (collider.source_name.empty()) return false;
                    const auto* surface_cache = getSurfaceMeshCacheForObject(collider.source_name);
                    if (!surface_cache) return false;
                    out_triangles = surface_cache->triangles;
                    out_version = surface_cache->version;
                    return true;
                });
            runtime.setGridDomainBoundsResolver(
                [this](const RayTrophiSim::SimulationGridDomainDesc& domain, Vec3& out_min, Vec3& out_max) {
                    if (domain.source_mode != RayTrophiSim::SimulationGridDomainSourceMode::ObjectBounds ||
                        domain.source_name.empty()) {
                        return false;
                    }
                    return resolveObjectBoundsForSimulation(domain.source_name, out_min, out_max);
                });
            runtime.setFlowSourceBoundsResolver(
                [this](const RayTrophiSim::SimulationFlowSourceDesc& source, Vec3& out_min, Vec3& out_max) {
                    if ((source.source_mode != RayTrophiSim::SimulationFlowSourceMode::ObjectBounds &&
                         source.source_mode != RayTrophiSim::SimulationFlowSourceMode::MeshSurface) ||
                        source.source_name.empty()) {
                        return false;
                    }
                    return resolveObjectBoundsForSimulation(source.source_name, out_min, out_max);
                });
            runtime.setFlowSourceTransformResolver(
                [this](const std::string& node_name, Matrix4x4& out_matrix) {
                    return resolveObjectTransformForSimulation(node_name, out_matrix);
                });
            runtime.setFlowSourceSurfaceSampler(
                [this](const RayTrophiSim::SimulationFlowSourceDesc& source,
                       uint32_t seed,
                       RayTrophiSim::ParticleSurfaceSample& out_sample) {
                    if (source.source_mode != RayTrophiSim::SimulationFlowSourceMode::MeshSurface ||
                        source.source_name.empty()) {
                        return false;
                    }
                    return sampleObjectSurfaceForSimulation(source.source_name, seed, out_sample);
                });
    }

    // Create, configure, and register a new runtime with the SimulationWorld so it
    // simulates concurrently with all other particle systems.
    std::shared_ptr<RayTrophiSim::ParticleSimulationSystem> createParticleRuntime() {
        syncSimulationWorld();
        auto runtime = std::make_shared<RayTrophiSim::ParticleSimulationSystem>();
        configureParticleRuntime(*runtime);
        simulation_world.addSystem(runtime);
        return runtime;
    }

    // Returns the active system's runtime, creating an active system if none exists.
    RayTrophiSim::ParticleSimulationSystem& ensureParticleSimulationSystem() {
        return *ensureActiveParticleSystemObject().runtime;
    }

    // Active system's runtime (the one the Simulation panel edits), or null.
    std::shared_ptr<RayTrophiSim::ParticleSimulationSystem> getParticleSimulationSystem() const {
        return activeParticleRuntime();
    }

    std::size_t spawnParticle(const RayTrophiSim::ParticleSpawnDesc& desc) {
        ensureActiveParticleSystemObject();
        return ensureParticleSimulationSystem().spawn(desc);
    }

    RayTrophiSim::ParticleEmitterDesc& addParticleEmitter(const RayTrophiSim::ParticleEmitterDesc& desc) {
        ensureActiveParticleSystemObject();
        auto& emitter = ensureParticleSimulationSystem().addEmitter(desc);
        syncActiveParticleSystemObjectFromRuntime();
        return emitter;
    }

    RayTrophiSim::ParticleEmitterDesc& addParticleEmitterFromForceField(const std::shared_ptr<Physics::ForceField>& field) {
        RayTrophiSim::ParticleEmitterDesc desc;
        desc.name = field ? field->name + " Emitter" : "Force Field Emitter";
        desc.source_mode = RayTrophiSim::ParticleEmitterSourceMode::ForceFieldOrigin;
        desc.source_name = field ? field->name : std::string();
        desc.point = field ? field->position : Vec3(0.0f, 1.0f, 0.0f);
        desc.direction = Vec3(0.0f, 1.0f, 0.0f);
        desc.rate_per_second = 48.0f;
        desc.speed = 2.5f;
        desc.spread = 0.45f;
        desc.lifetime_seconds = 5.0f;
        desc.seed = static_cast<uint32_t>(force_field_manager.force_fields.size() * 131u + 17u);
        return addParticleEmitter(desc);
    }

    RayTrophiSim::ParticleEmitterDesc& addParticleEmitterFromObject(const std::string& node_name) {
        RayTrophiSim::ParticleEmitterDesc desc;
        desc.name = node_name.empty() ? "Object Emitter" : node_name + " Emitter";
        desc.source_mode = RayTrophiSim::ParticleEmitterSourceMode::ObjectOrigin;
        desc.spawn_mode = RayTrophiSim::ParticleEmitterSpawnMode::MeshSurface;
        desc.source_name = node_name;
        desc.direction = Vec3(0.0f, 1.0f, 0.0f);
        desc.rate_per_second = 32.0f;
        desc.speed = 1.8f;
        desc.spread = 0.5f;
        desc.lifetime_seconds = 4.0f;
        desc.seed = static_cast<uint32_t>(node_name.size() * 97u + 29u);
        return addParticleEmitter(desc);
    }

    void clearParticleEmitters() {
        if (auto runtime = activeParticleRuntime()) {
            runtime->clearEmitters();
        }
    }

    RayTrophiSim::ParticleColliderDesc& addParticleCollider(const RayTrophiSim::ParticleColliderDesc& desc) {
        ensureActiveParticleSystemObject();
        auto& collider = ensureParticleSimulationSystem().addCollider(desc);
        syncActiveParticleSystemObjectFromRuntime();
        syncRigidBodyProxyColliders();
        invalidateRigidBodySimulationCache();
        return collider;
    }

    bool fitParticleColliderToObjectBounds(RayTrophiSim::ParticleColliderDesc& desc,
                                           const std::string& node_name,
                                           bool bind_to_object) const {
        Vec3 min_bound;
        Vec3 max_bound;
        if (!resolveObjectBoundsForSimulation(node_name, min_bound, max_bound)) {
            return false;
        }

        const Vec3 mn = Vec3::min(min_bound, max_bound);
        const Vec3 mx = Vec3::max(min_bound, max_bound);
        const Vec3 center = (mn + mx) * 0.5f;
        const Vec3 extent = mx - mn;

        if (bind_to_object) {
            desc.source_name = node_name;
        } else {
            desc.source_name.clear();
        }
        if (!node_name.empty()) {
            desc.name = node_name + (bind_to_object ? " Collider" : " Proxy Collider");
        }

        if (desc.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectAABB ||
            desc.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectOBB ||
            desc.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF ||
            desc.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectConvexDecomp ||
            desc.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectMeshBVH) {
            desc.bounds_min = mn;
            desc.bounds_max = mx;
            return true;
        }

        if (desc.source_mode == RayTrophiSim::ParticleColliderSourceMode::PlaneY) {
            desc.plane_y = mn.y;
            return true;
        }

        if (desc.source_mode == RayTrophiSim::ParticleColliderSourceMode::Sphere) {
            desc.sphere_center = center;
            desc.sphere_radius = std::max(0.001f, extent.length() * 0.5f);
            return true;
        }

        if (desc.source_mode == RayTrophiSim::ParticleColliderSourceMode::Capsule) {
            const float min_side = std::min({ extent.x, extent.y, extent.z });
            desc.capsule_radius = std::max(0.001f, min_side * 0.5f);
            if (extent.x >= extent.y && extent.x >= extent.z) {
                desc.capsule_start = Vec3(mn.x, center.y, center.z);
                desc.capsule_end = Vec3(mx.x, center.y, center.z);
            } else if (extent.y >= extent.x && extent.y >= extent.z) {
                desc.capsule_start = Vec3(center.x, mn.y, center.z);
                desc.capsule_end = Vec3(center.x, mx.y, center.z);
            } else {
                desc.capsule_start = Vec3(center.x, center.y, mn.z);
                desc.capsule_end = Vec3(center.x, center.y, mx.z);
            }
            return true;
        }

        return false;
    }

    // target_runtime: which system's collider receives the rebuilt SDF. Defaults
    // to the ACTIVE system (correct for UI edits, where the edited system is the
    // active one). The LOAD path MUST pass the specific system being deserialized:
    // during load the loaded system is neither active nor pushed into
    // particle_systems yet (active_particle_system_index is assigned only after
    // every system is read), so an active-system lookup would attach the voxel
    // SDF to the wrong system — or none — and the collider would silently fail to
    // block fluid after reload. (The SDF voxel grid itself is intentionally not
    // serialized; it is deterministically rebuilt here from the source mesh.)
    bool rebuildSDFColliderAsync(RayTrophiSim::ParticleColliderDesc& desc,
                                 std::shared_ptr<RayTrophiSim::ParticleSimulationSystem> target_runtime = nullptr,
                                 bool coalesce = false) {
        if (desc.source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF ||
            desc.source_name.empty()) {
            return false;
        }
        if (!desc.sdf_bake_serial) desc.sdf_bake_serial = std::make_shared<std::atomic<uint64_t>>(0u);
        if (!desc.sdf_bake_busy) desc.sdf_bake_busy = std::make_shared<std::atomic<bool>>(false);
        auto bake_serial = desc.sdf_bake_serial;
        auto bake_busy = desc.sdf_bake_busy;
        if (coalesce) {
            bool expected = false;
            if (!bake_busy->compare_exchange_strong(expected, true, std::memory_order_acq_rel))
                return false;
        } else {
            bake_busy->store(true, std::memory_order_release);
        }
        const uint64_t request_serial = bake_serial->fetch_add(1u, std::memory_order_acq_rel) + 1u;

        std::string node_name = desc.source_name;
        int res_mode = desc.sdf_resolution_mode;
        
        int N = 64;
        if (res_mode == 0) N = 32;
        else if (res_mode == 1) N = 64;
        else if (res_mode == 2) N = 128;

        const auto* surface_cache = getSurfaceMeshCacheForObject(node_name);
        if (!surface_cache || surface_cache->empty()) {
            bake_busy->store(false, std::memory_order_release);
            return false;
        }

        RayTrophiSim::ParticleColliderOBB obb;
        if (!resolveObjectOBBForSimulation(node_name, obb)) {
            bake_busy->store(false, std::memory_order_release);
            return false;
        }

        Matrix4x4 world_to_local = obb.local_to_world.inverse();

        auto triangles = surface_cache->triangles;
        Vec3 bmin(std::numeric_limits<float>::max());
        Vec3 bmax(-std::numeric_limits<float>::max());
        for (auto& tri : triangles) {
            tri.p0 = world_to_local.transform_point(tri.p0);
            tri.p1 = world_to_local.transform_point(tri.p1);
            tri.p2 = world_to_local.transform_point(tri.p2);
            tri.normal = world_to_local.transform_vector(tri.normal);
            const float nlen = tri.normal.length();
            if (nlen > 1e-6f) {
                tri.normal = tri.normal * (1.0f / nlen);
            } else {
                tri.normal = Vec3(0.0f, 1.0f, 0.0f);
            }
            bmin = Vec3::min(bmin, tri.p0);
            bmin = Vec3::min(bmin, tri.p1);
            bmin = Vec3::min(bmin, tri.p2);
            bmax = Vec3::max(bmax, tri.p0);
            bmax = Vec3::max(bmax, tri.p1);
            bmax = Vec3::max(bmax, tri.p2);
        }

        auto result_vec = std::make_shared<std::vector<float>>();

        g_active_sdf_bakes.fetch_add(1, std::memory_order_acquire);

        std::thread([this, node_name, triangles, bmin, bmax, N, result_vec, target_runtime,
                     bake_serial, bake_busy, request_serial]() {
            Vec3 size = bmax - bmin;
            Vec3 pad = size * 0.15f;
            Vec3 origin = bmin - pad;
            Vec3 extents = size + pad * 2.0f;

            int nx = N;
            int ny = N;
            int nz = N;
            result_vec->resize(static_cast<std::size_t>(nx * ny * nz), 0.0f);

            // Build a triangle BVH over ALL local-space triangles (no stride
            // decimation) so the distance field is exact and the cook costs
            // O(cells · log tris) instead of the old O(cells · tris) brute force.
            ColliderMeshBVH bvh;
            {
                std::vector<ColliderMeshBVH::Triangle> bvh_tris;
                bvh_tris.reserve(triangles.size());
                for (const auto& tri : triangles) bvh_tris.push_back({ tri.p0, tri.p1, tri.p2 });
                bvh.build(std::move(bvh_tris));
            }

            // Inside/outside by ray-parity vote over three non-axis-aligned probe
            // directions (each ~unit length). Robust to a single ray grazing a
            // shared edge, and far more reliable than the old single-nearest-
            // triangle normal dot, which flipped sign on edges / thin features.
            const Vec3 probe_dirs[3] = {
                Vec3(0.5060f,  0.7071f, 0.4943f),
                Vec3(-0.3651f, 0.5345f, 0.7625f),
                Vec3(0.8112f, -0.2701f, 0.5184f)
            };

            float* out = result_vec->data();
            const float step_x = extents.x / nx, step_y = extents.y / ny, step_z = extents.z / nz;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
            for (int k = 0; k < nz; ++k) {
                if (g_cancel_sdf_bakes.load(std::memory_order_relaxed)) {
                    continue;
                }
                for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const Vec3 cell_p = origin + Vec3((i + 0.5f) * step_x,
                                                      (j + 0.5f) * step_y,
                                                      (k + 0.5f) * step_z);
                    Vec3 closest;
                    float dist = std::sqrt(bvh.closestDistanceSquared(cell_p, closest));
                    int inside_votes = 0;
                    for (int d = 0; d < 3; ++d) {
                        if (bvh.countRayHits(cell_p, probe_dirs[d]) & 1) ++inside_votes;
                    }
                    if (inside_votes >= 2) dist = -dist;
                    out[static_cast<std::size_t>(k * (nx * ny) + j * nx + i)] = dist;
                }
            }

            if (g_cancel_sdf_bakes.load(std::memory_order_relaxed)) {
                if (bake_serial->load(std::memory_order_acquire) == request_serial)
                    bake_busy->store(false, std::memory_order_release);
                g_active_sdf_bakes.fetch_sub(1, std::memory_order_release);
                return;
            }

            // A later manual/automatic request owns publication. Discarding an
            // old result prevents a slow 128^3 cook restoring stale geometry.
            if (bake_serial->load(std::memory_order_acquire) != request_serial) {
                g_active_sdf_bakes.fetch_sub(1, std::memory_order_release);
                return;
            }

            // Never mutate the live collider from this detached worker. Timeline
            // restore/step reads it on the main thread and concurrent shared_ptr /
            // dimension writes are a real data race. Queue an immutable result;
            // updateSimulationTimeline publishes it at its safe tick boundary.
            {
                std::lock_guard<std::mutex> lock(pending_sdf_bakes_mutex_);
                pending_sdf_bakes_.push_back(PendingSdfBake{
                    node_name, target_runtime, result_vec, origin, extents,
                    nx, ny, nz, bake_serial, bake_busy, request_serial
                });
            }
            g_active_sdf_bakes.fetch_sub(1, std::memory_order_release);
        }).detach();
        return true;
    }

    void publishCompletedSdfBakes() {
        std::vector<PendingSdfBake> ready;
        {
            std::lock_guard<std::mutex> lock(pending_sdf_bakes_mutex_);
            ready.swap(pending_sdf_bakes_);
        }
        for (auto& bake : ready) {
            if (!bake.serial || bake.serial->load(std::memory_order_acquire) !=
                    bake.request_serial) {
                continue;
            }
            auto p_sys = bake.runtime ? bake.runtime : getParticleSimulationSystem();
            if (p_sys) {
                for (auto& coll : p_sys->colliders()) {
                    if (coll.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF &&
                        coll.source_name == bake.node_name) {
                        coll.sdf_grid_data = std::move(bake.grid);
                        coll.sdf_origin = bake.origin;
                        coll.sdf_extents = bake.extents;
                        coll.sdf_nx = bake.nx;
                        coll.sdf_ny = bake.ny;
                        coll.sdf_nz = bake.nz;
                        break;
                    }
                }
            }
            if (bake.busy) bake.busy->store(false, std::memory_order_release);
        }
    }

    RayTrophiSim::ParticleColliderDesc& addParticleProxyColliderFromObject(const std::string& node_name) {
        RayTrophiSim::ParticleColliderDesc desc;
        desc.name = node_name.empty() ? "Proxy Collider" : node_name + " Proxy Collider";
        desc.restitution = 0.25f;
        desc.friction = 0.15f;
        desc.thickness = 0.02f;

        Vec3 min_bound;
        Vec3 max_bound;
        if (resolveObjectBoundsForSimulation(node_name, min_bound, max_bound)) {
            const Vec3 extent = Vec3::max(min_bound, max_bound) - Vec3::min(min_bound, max_bound);
            const float min_side = std::max(0.001f, std::min({ extent.x, extent.y, extent.z }));
            const float max_side = std::max({ extent.x, extent.y, extent.z });
            const float mid_side = extent.x + extent.y + extent.z - min_side - max_side;
            if (max_side <= min_side * 1.35f) {
                desc.source_mode = RayTrophiSim::ParticleColliderSourceMode::Sphere;
            } else if (max_side >= std::max(0.001f, mid_side) * 1.75f) {
                desc.source_mode = RayTrophiSim::ParticleColliderSourceMode::Capsule;
            } else {
                desc.source_mode = RayTrophiSim::ParticleColliderSourceMode::ObjectOBB;
            }
            fitParticleColliderToObjectBounds(desc, node_name, desc.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectOBB);
        }

        return addParticleCollider(desc);
    }

    RayTrophiSim::SimulationGridDomainDesc& addSimulationGridDomain(const RayTrophiSim::SimulationGridDomainDesc& desc) {
        ensureActiveParticleSystemObject();
        auto& domain = ensureParticleSimulationSystem().addGridDomain(desc);
        syncActiveParticleSystemObjectFromRuntime();
        return domain;
    }

    bool removeSimulationGridDomain(
        std::size_t system_index,
        std::size_t domain_index,
        const std::function<void()>& after_render_detach = {}) {
        if (system_index >= particle_systems.size()) return false;
        auto& system = particle_systems[system_index];
        if (!system.runtime ||
            domain_index >= system.runtime->gridDomains().size()) {
            return false;
        }

        // First detach every RT/NanoVDB consumer while the compute buffers are
        // still valid. The caller drains in-flight rendering before entering.
        removeDomainVolume(system, domain_index);
        removeFoamDomainVolume(system, domain_index);

        auto erase_index = [domain_index](auto& values) {
            if (domain_index < values.size()) {
                values.erase(values.begin() +
                    static_cast<std::ptrdiff_t>(domain_index));
            }
        };
        erase_index(system.domain_vdb_ids);
        erase_index(system.domain_volumes);
        erase_index(system.domain_foam_vdb_ids);
        erase_index(system.domain_foam_volumes);
        erase_index(system.domain_foam_density);
        erase_index(system.domain_sdf_buffers);
        erase_index(system.domain_uvw_buffers);
        erase_index(system.domain_sdf_stats);
        erase_index(system.domain_sdf_signatures);
        erase_index(system.domain_vdb_upload_signatures);
        erase_index(system.domain_last_fluid_render_mode);
        erase_index(system.domain_last_tuned_shader);

        // A live Vulkan-RT volume can read the solver's dense compute buffers
        // through published device addresses. Give the renderer a two-phase
        // deletion boundary: the volume/world entries above are already gone,
        // but the owning compute buffers below are still alive. Vulkan can now
        // rebuild its TLAS/volume table without the domain before those buffers
        // are physically destroyed.
        if (after_render_detach) {
            after_render_detach();
        }

        // Only after all render consumers are detached may the indexed Vulkan
        // storage buffers be destroyed and the solver vectors compacted.
        const bool removed = system.runtime->removeGridDomain(
            domain_index, &simulation_world.compute());
        if (removed) {
            clearSimFrameCache();
            force_simulation_render_sync_ = true;
            g_gas_volumes_dirty = true;
            g_geometry_dirty = true;
            g_vulkan_rebuild_pending = true;
            g_optix_rebuild_pending = true;
            g_bvh_rebuild_pending = true;
        }
        return removed;
    }

    RayTrophiSim::SimulationGridDomainDesc& addSimulationGridDomainFromObject(const std::string& node_name) {
        RayTrophiSim::SimulationGridDomainDesc desc;
        desc.name = node_name.empty() ? "Grid Domain" : node_name + " Domain";
        desc.source_mode = RayTrophiSim::SimulationGridDomainSourceMode::ManualBox;
        desc.source_name.clear();
        // Same rule as the panel's "Add Grid Domain": start on the fastest solver
        // the machine actually has, not on the descriptor's compatibility default.
        desc.backend = RayTrophiSim::defaultSimulationDomainBackend();
        Vec3 min_bound;
        Vec3 max_bound;
        if (resolveObjectBoundsForSimulation(node_name, min_bound, max_bound)) {
            desc.bounds_min = min_bound;
            desc.bounds_max = max_bound;
            const Vec3 extent = Vec3::max(min_bound, max_bound) - Vec3::min(min_bound, max_bound);
            const float max_extent = std::max({ extent.x, extent.y, extent.z, 0.001f });
            const int max_res = std::max({ desc.resolution_x, desc.resolution_y, desc.resolution_z, 1 });
            desc.voxel_size = max_extent / static_cast<float>(max_res);
        }
        return addSimulationGridDomain(desc);
    }

    void clearSimulationGridDomains() {
        if (auto runtime = activeParticleRuntime()) {
            runtime->clearGridDomains();
        }
    }

    RayTrophiSim::SimulationFlowSourceDesc& addSimulationFlowSource(
        const RayTrophiSim::SimulationFlowSourceDesc& desc) {
        ensureActiveParticleSystemObject();
        auto& source = ensureParticleSimulationSystem().addFlowSource(desc);
        syncActiveParticleSystemObjectFromRuntime();
        return source;
    }

    RayTrophiSim::SimulationFlowSourceDesc& addSimulationFlowSourceFromObject(
        const std::string& node_name,
        int domain_index) {
        RayTrophiSim::SimulationFlowSourceDesc desc;
        desc.name = node_name.empty() ? "Object Flow Source" : node_name + " Flow";
        desc.source_mode = RayTrophiSim::SimulationFlowSourceMode::ObjectBounds;
        desc.source_name = node_name;
        desc.domain_index = std::max(0, domain_index);
        desc.density = 2.0f;
        desc.temperature = 0.6f;
        desc.fuel = 0.0f;
        desc.velocity = Vec3(0.0f, 1.0f, 0.0f);

        Vec3 min_bound;
        Vec3 max_bound;
        if (resolveObjectBoundsForSimulation(node_name, min_bound, max_bound)) {
            const Vec3 mn = Vec3::min(min_bound, max_bound);
            const Vec3 mx = Vec3::max(min_bound, max_bound);
            desc.position = (mn + mx) * 0.5f;
            desc.radius = std::max(0.05f, (mx - mn).length() * 0.25f);
        }

        return addSimulationFlowSource(desc);
    }

    void clearSimulationFlowSources() {
        if (auto runtime = activeParticleRuntime()) {
            runtime->clearFlowSources();
        }
    }

    RayTrophiSim::ParticleColliderDesc& addParticleColliderFromObject(const std::string& node_name) {
        RayTrophiSim::ParticleColliderDesc desc;
        desc.name = node_name.empty() ? "Object Collider" : node_name + " Collider";
        desc.source_mode = RayTrophiSim::ParticleColliderSourceMode::ObjectOBB;
        desc.source_name = node_name;
        desc.restitution = 0.25f;
        desc.friction = 0.15f;
        desc.thickness = 0.02f;
        return addParticleCollider(desc);
    }

    RayTrophiSim::ParticleColliderDesc& addParticleSphereColliderFromObject(const std::string& node_name) {
        RayTrophiSim::ParticleColliderDesc desc;
        desc.name = node_name.empty() ? "Sphere Collider" : node_name + " Sphere Collider";
        desc.source_mode = RayTrophiSim::ParticleColliderSourceMode::Sphere;
        desc.source_name = node_name;
        desc.restitution = 0.25f;
        desc.friction = 0.15f;
        desc.thickness = 0.02f;

        Vec3 min_bound;
        Vec3 max_bound;
        if (resolveObjectBoundsForSimulation(node_name, min_bound, max_bound)) {
            const Vec3 mn = Vec3::min(min_bound, max_bound);
            const Vec3 mx = Vec3::max(min_bound, max_bound);
            desc.sphere_center = (mn + mx) * 0.5f;
            desc.sphere_radius = std::max(0.001f, (mx - mn).length() * 0.5f);
        }

        return addParticleCollider(desc);
    }

    RayTrophiSim::ParticleColliderDesc& addParticleCapsuleColliderFromObject(const std::string& node_name) {
        RayTrophiSim::ParticleColliderDesc desc;
        desc.name = node_name.empty() ? "Capsule Collider" : node_name + " Capsule Collider";
        desc.source_mode = RayTrophiSim::ParticleColliderSourceMode::Capsule;
        desc.source_name = node_name;
        desc.restitution = 0.25f;
        desc.friction = 0.15f;
        desc.thickness = 0.02f;

        Vec3 min_bound;
        Vec3 max_bound;
        if (resolveObjectBoundsForSimulation(node_name, min_bound, max_bound)) {
            const Vec3 mn = Vec3::min(min_bound, max_bound);
            const Vec3 mx = Vec3::max(min_bound, max_bound);
            const Vec3 center = (mn + mx) * 0.5f;
            const Vec3 extent = mx - mn;
            const float min_side = std::min({ extent.x, extent.y, extent.z });
            desc.capsule_radius = std::max(0.001f, min_side * 0.5f);
            if (extent.x >= extent.y && extent.x >= extent.z) {
                desc.capsule_start = Vec3(mn.x, center.y, center.z);
                desc.capsule_end = Vec3(mx.x, center.y, center.z);
            } else if (extent.y >= extent.x && extent.y >= extent.z) {
                desc.capsule_start = Vec3(center.x, mn.y, center.z);
                desc.capsule_end = Vec3(center.x, mx.y, center.z);
            } else {
                desc.capsule_start = Vec3(center.x, center.y, mn.z);
                desc.capsule_end = Vec3(center.x, center.y, mx.z);
            }
        }

        return addParticleCollider(desc);
    }

    void clearParticleColliders() {
        ensureActiveParticleSystemObject();
        auto& runtime = ensureParticleSimulationSystem();
        runtime.clearColliders();
        syncActiveParticleSystemObjectFromRuntime();
        syncRigidBodyProxyColliders();
        invalidateRigidBodySimulationCache();
    }

    void spawnDebugParticleBurst(const Vec3& center,
                                 int count = 64,
                                 float radius = 0.15f,
                                 float speed = 2.0f,
                                 float lifetime_seconds = 4.0f) {
        auto& particles = ensureParticleSimulationSystem();
        ensureActiveParticleSystemObject();
        const int safe_count = std::clamp(count, 1, 4096);
        particles.reserve(particles.capacity() + static_cast<std::size_t>(safe_count));

        constexpr float two_pi = 6.28318530718f;
        for (int i = 0; i < safe_count; ++i) {
            const float t = safe_count > 1 ? static_cast<float>(i) / static_cast<float>(safe_count - 1) : 0.0f;
            const float angle = two_pi * t * 2.61803398875f;
            const float ring = radius * (0.35f + 0.65f * std::sqrt(t));
            const Vec3 offset(std::cos(angle) * ring, 0.0f, std::sin(angle) * ring);
            Vec3 velocity(std::cos(angle) * speed, speed * (0.35f + 0.65f * (1.0f - t)), std::sin(angle) * speed);

            RayTrophiSim::ParticleSpawnDesc desc;
            desc.position = center + offset;
            desc.velocity = velocity;
            desc.lifetime_seconds = lifetime_seconds;
            desc.mass = 1.0f;
            particles.spawn(desc);
        }
    }

    void clearParticles() {
        if (auto runtime = activeParticleRuntime()) {
            runtime->releaseComputeResources(simulation_world.compute());
            runtime->clear();
        }
    }

    void updateParticleSimulation(float dt) {
        syncSimulationWorld();
        simulation_world.stepOnce(dt);
        syncSimulationRenderVolumes();
    }

    void updateSimulation(float dt) {
        syncSimulationWorld();
        simulation_world.stepOnce(dt);
        syncSimulationRenderVolumes();
    }

    // =========================================================================
    // Force Fields (Universal Physics System)
    // =========================================================================
    Physics::ForceFieldManager force_field_manager;
    RayTrophiSim::SimulationWorld simulation_world;

    // Live dense gas fields publish raw Vulkan buffer device addresses through
    // VDBVolumeManager so Vulkan RT can ray-march the simulation without a host
    // copy. Those addresses borrow the lifetime of the current simulation
    // compute backend. Clear them BEFORE replacing/destroying that backend;
    // otherwise a Vulkan -> OptiX/CPU -> Vulkan round-trip can republish a freed
    // VkBuffer address in the volume SSBO and the first RT traversal causes a
    // device loss/TDR. Host OpenVDB/NanoVDB data and stable volume ids remain
    // intact, so the non-Vulkan path keeps rendering and the next Vulkan solve
    // naturally publishes fresh addresses.
    void invalidateSimulationDenseGpuAddresses() {
        auto& manager = VDBVolumeManager::getInstance();
        for (auto& system : particle_systems) {
            for (const int volume_id : system.domain_vdb_ids) {
                if (volume_id >= 0) {
                    manager.clearLiveDenseGpuFields(volume_id);
                }
            }
        }
        force_simulation_render_sync_ = true;
        g_gas_volumes_dirty = true;
    }

    void syncSimulationWorld() {
        simulation_world.setForceFieldManager(&force_field_manager);

        // GPU_Compute (value 1) = auto-select: CUDA preferred, Vulkan fallback.
        // GPU_Vulkan  (value 3) = force Vulkan regardless (for explicit testing).
        bool auto_gpu_requested    = g_sim_use_gpu_solver;
        bool vulkan_only_requested = false;
        for (const auto& system : particle_systems) {
            if (system.runtime) {
                for (const auto& domain : system.runtime->gridDomains()) {
                    if (domain.backend == RayTrophiSim::SimulationDomainBackend::GPU_Compute)
                        auto_gpu_requested = true;
                    if (domain.backend == RayTrophiSim::SimulationDomainBackend::GPU_Vulkan)
                        vulkan_only_requested = true;
                }
            }
        }

        auto& compute = simulation_world.compute();

        if (vulkan_only_requested) {
            // Explicit Vulkan-only path (testing / non-NVIDIA systems)
            if (compute.backendType() != RayTrophiSim::ComputeBackendType::VulkanCompute) {
                invalidateSimulationDenseGpuAddresses();
                auto vk_backend =
                    RayTrophiSim::createVulkanSimulationComputeBackend(g_vulkan_sim_compute_ctx);
                g_hasVulkanComputeSim = (vk_backend != nullptr);
                compute.setBackend(std::move(vk_backend));
            }
        } else if (auto_gpu_requested || vulkan_only_requested) {
            // Auto: try CUDA first, fall back to Vulkan
            if (compute.backendType() == RayTrophiSim::ComputeBackendType::CUDA) {
                // Already on CUDA — keep it
            } else {
                // GPU_Compute is CUDA-preferred even if an earlier selection
                // left this context on Vulkan.
                invalidateSimulationDenseGpuAddresses();
                auto cuda_backend = RayTrophiSim::createCudaSimulationComputeBackend();
                if (cuda_backend) {
                    compute.setBackend(std::move(cuda_backend));
                } else {
                    // CUDA unavailable — try Vulkan
                    auto vk_backend =
                        RayTrophiSim::createVulkanSimulationComputeBackend(g_vulkan_sim_compute_ctx);
                    g_hasVulkanComputeSim = (vk_backend != nullptr);
                    compute.setBackend(std::move(vk_backend));
                }
            }
        } else if (compute.backendType() != RayTrophiSim::ComputeBackendType::CPU) {
            invalidateSimulationDenseGpuAddresses();
            compute.setBackend(nullptr);
        }
    }

    RayTrophiSim::SimulationWorld& getSimulationWorld() {
        syncSimulationWorld();
        return simulation_world;
    }

    const RayTrophiSim::SimulationWorld& getSimulationWorld() const {
        return simulation_world;
    }

    void refreshSimulationForceFieldSnapshot() {
        syncSimulationWorld();
        simulation_world.refreshForceFieldSnapshot();
    }
    
    // Add a force field to the scene
    int addForceField(std::shared_ptr<Physics::ForceField> field) {
        syncSimulationWorld();
        return force_field_manager.addForceField(field);
    }
    
    // Remove a force field from the scene
    bool removeForceField(std::shared_ptr<Physics::ForceField> field) {
        syncSimulationWorld();
        return force_field_manager.removeForceField(field);
    }
    
    // Find force field by name
    std::shared_ptr<Physics::ForceField> findForceFieldByName(const std::string& name) const {
        return force_field_manager.findByName(name);
    }
    
    // Evaluate all force fields at a position (for physics simulations)
    Vec3 evaluateForceFieldsAt(const Vec3& world_pos, float time, 
                               const Vec3& velocity) const {
        return force_field_manager.evaluateAt(world_pos, time, velocity);
    }
    
    // Evaluate all force fields at a position (simplified - no velocity)
    Vec3 evaluateForceFieldsAt(const Vec3& world_pos, float time) const {
        return force_field_manager.evaluateAt(world_pos, time, Vec3(0,0,0));
    }
    
    // Evaluate force fields for specific system type
    Vec3 evaluateForceFieldsForGas(const Vec3& world_pos, float time, 
                                   const Vec3& velocity) const {
        return force_field_manager.evaluateAtFiltered(world_pos, time, velocity, true, false, false, false);
    }

    // =========================================================================
    // Clear all scene data
    // =========================================================================
    void clear() {
        // Cancel and wait for all active background SDF bakes before destroying the scene objects
        g_cancel_sdf_bakes.store(true, std::memory_order_release);
        while (g_active_sdf_bakes.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
        {
            std::lock_guard<std::mutex> lock(pending_sdf_bakes_mutex_);
            pending_sdf_bakes_.clear();
        }
        g_cancel_sdf_bakes.store(false, std::memory_order_release); // reset for subsequent projects

        syncSimulationWorld();
        world.clear();
        lights.clear();
        cameras.clear();
        animationDataList.clear();
        boneData.clear();              // Clear bone hierarchy
        timeline.clear();              // Clear keyframes
        ui_settings_json_str = "";     // Clear UI settings string
        load_counter = 0;              // Reset load counter
        
        base_mesh_cache.clear();
        mesh_modifiers.clear();
        mesh_paint_texture_sets.clear();
        mesh_paint_layer_stacks.clear();
        object_groups.clear();
        surface_cache_epoch_done_.clear();
        simulation_local_bounds_.clear();
        last_sim_pose_applied_.clear();
        soft_weld_cache_.clear();
        rigid_bake_cache_.clear();
        melt_weld_cache_.clear();
        melt_displaced_.clear();
        melt_sdf_refresh_stamp_.clear();
        melt_applied_stamp_.clear();
        fracture_summary_tick_ = 0;
        editor_pending_delete_object_names.clear();
        
        // Clear per-model animator caches BEFORE clearing the vector
        for (auto& ctx : importedModelContexts) {
            if (ctx.animator) {
                ctx.animator->clear();
            }
            if (ctx.graph) {
                ctx.graph.reset();
            }
            if (ctx.runtimeGraph) {
                ctx.runtimeGraph.reset();
            }
            ctx.members.clear();
        }
        importedModelContexts.clear(); // Clear model contexts (releases aiScene memory)
        
        vdb_volumes.clear();           // Clear VDB volumes
        gas_volumes.clear();           // Clear gas volumes
        gas_simulation_system.reset();

        // ── CRITICAL ORDER: release render resources BEFORE destroying the
        //    objects they reference. releaseSimulationRenderVolumes() iterates
        //    fluid_objects and particle_systems to tear down InstanceManager
        //    groups, VDB volumes, and FluidRenderBindings. If we clear the
        //    vectors first, those render resources leak or hold dangling refs
        //    into freed FluidObject memory, corrupting the heap on teardown.
        releaseSimulationRenderVolumes();
        clearSimFrameCache();
        sim_timeline_frame_ = -1;
        rigid_timeline_frame_ = -1;
        // Abandon any in-flight cooperative disk bake — its tick references the
        // particle_systems we are about to clear, so it must not run again.
        sim_bake_active_ = false;
        sim_bake_cancel_ = false;
        sim_bake_hashes_.clear();
        // Drop any disk bake-cache binding from the previous project so a freshly
        // loaded scene without a cache doesn't keep streaming the old one.
        sim_cache_valid_ = false;
        sim_cache_dir_.clear();
        sim_cache_valid_system_ids_.clear();

        // Detach the FluidSimulationSystem's raw pointer to fluid_objects
        // BEFORE clearing the vector. This prevents any stale-pointer access
        // during destruction ordering or if a system dtor triggers a step.
        if (fluid_simulation_system) {
            fluid_simulation_system->setObjects(nullptr);
        }
        // Detach the RigidBodySystem from rigid_bodies before clearing it.
        if (rigid_body_system) {
            rigid_body_system->setBodies(nullptr);
        }

        // Release particle system compute resources before clearing.
        for (auto& system : particle_systems) {
            if (system.runtime) {
                system.runtime->releaseComputeResources(simulation_world.compute());
            }
        }

        // Clear simulation_world systems BEFORE destroying the objects those
        // systems reference (fluid_objects, particle_systems). The systems
        // hold raw pointers into these vectors; releasing the shared_ptrs
        // first ensures no system dtor can accidentally dereference them.
        simulation_world.clearSystems();
        simulation_world.resetTime();

        // NOW safe to destroy the actual data vectors.
        fluid_objects.clear();
        fluid_simulation_system.reset();
        rigid_bodies.clear();
        rigid_body_system.reset();
        next_fluid_object_id = 1;
        active_fluid_object_index = -1;
        particle_systems.clear();
        active_particle_system_index = -1;
        next_particle_system_id = 1;
        force_field_manager.clear();   // Clear force fields
        syncSimulationWorld();
        camera = nullptr;
        active_camera_index = 0;
        bvh = nullptr;
        
        // Clear paint data
        mesh_paint_texture_sets.clear();
        mesh_paint_layer_stacks.clear();
        editor_pending_delete_object_names.clear();
        object_groups.clear();
        mesh_modifiers.clear();
        base_mesh_cache.clear();
        soft_weld_cache_.clear();   // holds shared_ptr<Triangle> into the old scene
        rigid_bake_cache_.clear();  // ditto (rigid render-bake rest cache)
        melt_weld_cache_.clear();   // ditto (Phase 6c melt slump rest cache)
        melt_displaced_.clear();
        melt_sdf_refresh_stamp_.clear();
        melt_applied_stamp_.clear();
        fracture_summary_tick_ = 0;
        soft_frame_cache_.clear();
        invalidateSurfaceMeshCache();

        // Reset Post-Processing to defaults
        color_processor = ColorProcessor();

        initialized = false;
    }
};
