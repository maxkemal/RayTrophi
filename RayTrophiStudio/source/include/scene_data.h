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

// Global atomics to track and cancel active background SDF bakes during scene destruction/clearance.
inline std::atomic<bool> g_cancel_sdf_bakes{false};
inline std::atomic<int> g_active_sdf_bakes{0};
#include "Fluid/FluidObject.h"
#include "Fluid/FluidSimulationSystem.h"
#include "RigidBodySystem.h"
#include "Core/RenderStateManager.h"
#include "globals.h"
#include "SurfaceMeshCache.h"
#include "ColliderMeshBVH.h"
#include "MeshModifiers.h"
#include "GeometryNodesV2.h"
#include "MaterialNodesV2.h"
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
    void makeFractureGroupBreakable(const std::string& group,
                                    const std::vector<std::string>& shard_nodes,
                                    float threshold) {
        if (shard_nodes.empty()) return;
        ensureRigidBodySystem();
        for (const auto& node : shard_nodes) {
            RayTrophiSim::RigidBodyObject* rb = addRigidBodyForObject(node, /*dynamic=*/false);
            if (!rb) continue;
            rb->setBreakable(true);
            rb->broken = false;
            rb->setFractureGroup(group);
            rb->setBreakImpulse(threshold);
            rb->shape = RayTrophiSim::RigidBodyShape::Mesh;   // convex hull when broken
            rb->motion_type = RayTrophiSim::RigidBodyMotionType::Static;
            rb->dynamic = false;
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

    // After a sim step: read the contact events and shatter any breakable group
    // whose shard took an impact above its threshold. No-op without breakables.
    void processFractureImpacts() {
        if (!rigid_body_system || !rigid_body_system->contactEventsEnabled()) return;
        const auto& events = rigid_body_system->contactEvents();
        if (events.empty()) return;
        auto findBreakable = [&](const std::string& src) -> RayTrophiSim::RigidBodyObject* {
            if (src.empty()) return nullptr;
            for (auto& rb : rigid_bodies)
                if (rb.getBreakable() && !rb.broken && rb.source_name == src) return &rb;
            return nullptr;
        };
        for (const auto& ev : events) {
            RayTrophiSim::RigidBodyObject* hit = findBreakable(ev.source_a);
            Vec3 dir = ev.normal;
            if (!hit) { hit = findBreakable(ev.source_b); dir = ev.normal * -1.0f; }
            if (!hit || ev.impulse < hit->getBreakImpulse()) continue;
            const float strength = std::max(2.0f, ev.impulse * 0.5f);
            breakFractureGroup(hit->getFractureGroup(), ev.point, dir, strength);
        }
    }

    // Reset every breakable group back to intact (Static, unbroken) — called on a
    // rewind to frame 0 so a replay re-derives the shatter deterministically.
    void resetFractureToIntact() {
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
    // bug). Indexed [frame] -> per-system {SoA, alive}. Cleared with the fluid cache.
    std::map<int, std::vector<std::pair<RayTrophiSim::ParticleSoABuffers, std::size_t>>> particle_frame_cache_;
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
        if (g_optix_rebuild_in_progress.load() || g_viewport_rebuild_in_progress.load()) {
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
            for (std::size_t d = states.size(); d < system.domain_vdb_ids.size(); ++d) {
                removeDomainVolume(system, d);
            }
            system.domain_vdb_ids.resize(states.size(), -1);
            system.domain_volumes.resize(states.size());
            system.domain_sdf_buffers.resize(states.size());
            system.domain_sdf_stats.resize(states.size());
            system.domain_sdf_signatures.resize(states.size(), 0);
            system.domain_vdb_upload_signatures.resize(states.size(), 0);
            system.domain_last_fluid_render_mode.resize(states.size(), -1);
            // Reused below to carry foam into the fluid-surface volume's
            // temperature channel (single-volume whitewater compositing).
            system.domain_foam_density.resize(states.size());

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
                const RayTrophiSim::Fluid::FluidRenderMode fluid_mode =
                    (is_fluid_domain && d < domains.size())
                        ? domains[d].fluid_render_mode
                        : RayTrophiSim::Fluid::FluidRenderMode::Volume;
                const bool fluid_skip_volume =
                    is_fluid_domain &&
                    fluid_mode == RayTrophiSim::Fluid::FluidRenderMode::Particles;
                const bool fluid_surface_route =
                    is_fluid_domain &&
                    fluid_mode == RayTrophiSim::Fluid::FluidRenderMode::SurfaceSDF;
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
                    domain_render_enabled && system.visible && state.valid && has_density &&
                    state.grid.nx > 0 && !fluid_skip_volume &&
                    // Volume mode needs density splatted; SurfaceSDF rebuilds
                    // its own density-proxy from particles, so it only needs
                    // particles to be present.
                    (fluid_surface_route
                        ? !state.particles.empty()
                        : state.active_density_cells > 0);

                if (!renderable) {
                    // Keep the VDB registered and visible so the TLAS customIndex→SSBO
                    // slot mapping stays valid. Setting visible=false here causes a
                    // became_visible rebuild on the very next renderable frame, and
                    // unloading the VDB causes the same full-rebuild cycle every time
                    // the fluid transitions between empty/non-empty (e.g. on timeline
                    // loop or when density briefly hits 0 at simulation start).
                    // Just signal an SSBO re-sync; the last-uploaded density stays
                    // registered (shows a momentary stale frame, then updates naturally).
                    if (system.domain_vdb_ids[d] >= 0) {
                        g_gas_volumes_dirty = true;
                    }
                    // Particles/splat-sphere mode: the kept-alive volume must NOT
                    // occlude on the CPU, or its AABB masks the splat spheres inside
                    // the domain (only wall-adjacent ones, hit before the box entry,
                    // survived). GPU keeps using it via the SSBO (visible stays true).
                    if (fluid_skip_volume && d < system.domain_volumes.size() &&
                        system.domain_volumes[d]) {
                        system.domain_volumes[d]->cpu_render_skip = true;
                    }
                    continue;
                }

                // Each domain owns its volume shader (created lazily, editable in
                // the domain panel, serialized with the domain).
                if (d < domains.size() && !domains[d].shader) {
                    domains[d].shader = VolumeShader::createSmokePreset();
                }
                std::shared_ptr<VolumeShader> domain_shader =
                    (d < domains.size()) ? domains[d].shader : nullptr;

                // Re-tune the per-domain shader ONLY when the fluid render mode
                // crosses a boundary (or on first sight). Otherwise the user's
                // live UI edits would be stomped by the preset every frame.
                if (is_fluid_domain && domain_shader && d < system.domain_last_fluid_render_mode.size()) {
                    const int cur_mode = static_cast<int>(fluid_mode);
                    if (cur_mode != system.domain_last_fluid_render_mode[d]) {
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
                        if (system.domain_last_fluid_render_mode[d] != -1) {
                            removeDomainVolume(system, d);
                            if (d < system.domain_sdf_buffers.size()) {
                                system.domain_sdf_buffers[d].clear();
                                system.domain_sdf_buffers[d].shrink_to_fit();
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
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(lsp.narrow_band_voxels));
                    sdf_sig = hash_combine_local(sdf_sig, static_cast<uint64_t>(lsp.smoothing_iterations));
                    sdf_sig = hash_combine_local(sdf_sig, static_cast<uint64_t>(lsp.surface_resolution_multiplier));
                    sdf_sig = hash_combine_local(sdf_sig, lsp.anisotropy_enabled ? 1ull : 0ull);
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(lsp.anisotropy_radius_voxels));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(lsp.anisotropy_max_stretch));
                    sdf_sig = hash_combine_local(sdf_sig, static_cast<uint64_t>(lsp.anisotropy_neighbor_min));
                    sdf_sig = hash_combine_local(sdf_sig, quantize_local(lsp.position_smoothing));
                    for (const auto& p : state.particles.position) {
                        sdf_sig = hash_combine_local(sdf_sig, quantize_local(p.x));
                        sdf_sig = hash_combine_local(sdf_sig, quantize_local(p.y));
                        sdf_sig = hash_combine_local(sdf_sig, quantize_local(p.z));
                    }

                    const bool needs_sdf_rebuild =
                        force_sync ||
                        d >= system.domain_sdf_signatures.size() ||
                        system.domain_sdf_signatures[d] != sdf_sig ||
                        sdf_buf.empty();
                    if (needs_sdf_rebuild) {
                        RayTrophiSim::Fluid::buildLevelSet(
                            state.particles, state.grid,
                            lsp, sdf_buf, &sdf_stats);
                        if (d < system.domain_sdf_signatures.size()) {
                            system.domain_sdf_signatures[d] = sdf_sig;
                        }
                        surface_sdf_changed = true;
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
                const bool do_update = force_sync || (prev_id < 0) ||
                    (density_ptr_override
                        ? (upload_changed && ((frame % stride) == 0))
                        : ((frame % stride) == 0));
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
                    system.domain_vdb_ids[d] = mgr.registerOrUpdateLiveVolume(
                        prev_id,
                        volume_name,
                        up_nx, up_ny, up_nz,
                        up_voxel,
                        density_ptr,
                        up_temp,
                        nullptr);
                    if (density_ptr_override &&
                        d < system.domain_vdb_upload_signatures.size() &&
                        system.domain_vdb_ids[d] >= 0) {
                        system.domain_vdb_upload_signatures[d] = upload_sig;
                    }
                    simulation_render_updated = true;
                    // The host/GPU NanoVDB grid changed: force the backend volume
                    // table re-sync. OptiX stores device pointers in that table,
                    // so a live grid reallocation must refresh it before launch.
                    g_gas_volumes_dirty = true;
                }
                const int id = system.domain_vdb_ids[d];
                if (id < 0) {
                    continue;
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
                vol->render_as_isosurface = fluid_surface_route;
                if (fluid_surface_route && d < domains.size()) {
                    vol->render_isosurface_ior = domains[d].fluid_surface_ior;
                    vol->render_isosurface_roughness = domains[d].fluid_surface_roughness;
                    vol->render_isosurface_foam = domains[d].fluid_surface_foam;
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
            }
            h = mix(h, s.runtime->flowSources().size());
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
            }
            h = mix(h, s.runtime->flowSources().size());
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
                h = mix(h, qf(rb.getBreakImpulse()));
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
        for (auto& obj : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (!tri) continue;
            const std::string& nn = tri->getNodeName();
            if (nn.empty()) continue;
            for (std::size_t i = 0; i < posed_names.size(); ++i) {
                if (posed_names[i] != nn) continue;
                if (Transform* th = tri->getTransformPtr()) th->setPivotMatrix(posed_mats[i]);
                break;
            }
        }
        // These objects just moved without bumping g_scene_geometry_generation, so
        // drop them from the surface-cache epoch memo — their next resolve must
        // rebuild from the new world verts. Static (un-posed) objects keep their
        // memo and stay cheap.
        for (const auto& name : posed_names) surface_cache_epoch_done_.erase(name);
    }

    bool hasSimFrame(int frame) const {
        return sim_frame_cache_.find(frame) != sim_frame_cache_.end();
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
                bytes += snap.first.position_x.size() * 80;  // ~20 float columns + flags
        for (const auto& f : rigid_frame_cache_)
            bytes += f.second.size() * sizeof(RayTrophiSim::RigidBodyFrameState);
        return bytes;
    }
    int cachedSimFrameCount() const { return static_cast<int>(sim_frame_cache_.size()); }

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

            auto it = soft_weld_cache_.find(node);
            const bool have_rest = (it != soft_weld_cache_.end() &&
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
            soft_weld_cache_[node] = std::move(cache);
            return ok;
        }

        std::size_t current_tri_count = 0;
        for (const auto& obj : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->getNodeName() == node) current_tri_count++;
        }

        auto it = soft_weld_cache_.find(node);
        const bool have_rest = (it != soft_weld_cache_.end() &&
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
        soft_weld_cache_[node] = std::move(cache);
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
        SoftWeldCache& cache = it->second;
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
            if (system.runtime) entry.push_back(system.runtime->gridDomainStates());
            else entry.emplace_back();
        }
        // Capture the discrete particle SoA in the SAME pass so a cached-frame
        // replay restores the actual particles (grid states alone left them empty).
        auto& psnap = particle_frame_cache_[frame];
        psnap.clear();
        psnap.reserve(particle_systems.size());
        for (auto& system : particle_systems) {
            if (system.runtime) psnap.emplace_back(system.runtime->buffers(), system.runtime->aliveCount());
            else psnap.emplace_back();
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
        if (it == rigid_frame_cache_.end()) return false;
        if (!rigid_body_system->restoreFrameState(it->second)) return false;
        rigid_timeline_frame_ = frame;
        return true;
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
        // Rigid bodies aren't frame-cached yet (Faz 5); the start is the only frame
        // we can faithfully reconstruct, so reset them to their initial pose there.
        // (A loop-back / scrub to frame 0 thus puts a fallen body back at the top.)
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
            for (std::size_t i = 0; i < particle_systems.size(); ++i) {
                if (particle_systems[i].runtime) {
                    particle_systems[i].runtime->setGridDomainStates(it->second[i]);
                    if (have_particles) {
                        particle_systems[i].runtime->restoreSoA(pit->second[i].first, pit->second[i].second);
                    }
                    invalidateSimulationRenderBindings(particle_systems[i]);
                }
            }
            simulation_world.resetTime(static_cast<float>(frame) * fixed_dt, frame);
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
                if (RayTrophiSim::SimCache::readSystemFrame(
                        sim_cache_dir_, particle_systems[i].id, frame, loaded)) {
                    particle_systems[i].runtime->setGridDomainStates(loaded);
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
        simulation_world.resetTime(static_cast<float>(frame) * fixed_dt, frame);
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
                // Re-arm standing-tank (FillLevel) fluid seeds and synchronize the
                // domain states NOW, so frame 0 already carries the full tank +
                // correct grid metadata — exactly what the disk bake does
                // (bakeSimulationToDisk). Without this the interactive play path
                // started frame 0 EMPTY (clear() wiped the seeded particles and the
                // seed wasn't re-applied until the first step), so a fill domain
                // only refilled a couple of frames in and its SurfaceSDF volume
                // wasn't built/raymarched until ~frame 2 (the reported bug).
                for (auto& dom : system.runtime->gridDomains()) {
                    if (dom.type == RayTrophiSim::SimulationDomainType::Fluid &&
                        dom.fluid_seed_mode == RayTrophiSim::FluidSeedMode::FillLevel) {
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
    void updateSimulationTimeline(int tl_frame, bool playing, float realtime_dt, float fps, bool live_mode,
                                  bool ui_editing = false) {
        if (tl_frame < 0) tl_frame = 0;
        simulation_render_updated = false;
        const bool force_resync = force_simulation_render_sync_;

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
        const uint64_t cfg_sig = computeSimConfigSignature();
        if (cfg_sig != last_sim_config_sig_) {
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
                const bool fluid_affected = (fluid_sig != last_fluid_coupling_sig_);
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
            simulation_world.stepOnce(realtime_dt);
            processFractureImpacts();  // live preview: shatter on impact
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
            if (changed) syncSimulationRenderVolumes();
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
                    simulation_world.stepOnce(fixed_dt);
                    processFractureImpacts();  // shatter on impact above threshold
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
        if (changed) {
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
            simulation_world.stepOnce(fixed_dt);
            processFractureImpacts();  // shatter on impact above threshold
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
            if (!RayTrophiSim::SimCache::writeSystemFrame(
                    sim_bake_dir_, sys.id, f, sys.runtime->gridDomainStates())) {
                ok = false;
            }
        }
        if (!writeSoftBodiesBakeFrame_(f)) ok = false;
        return ok;
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

private:
    void invalidateSimulationRenderBindings(ParticleSystemObject& system) {
        // Invalidates NanoVDB host/GPU bindings so the next syncSimulationRenderVolumes
        // re-registers + re-uploads density from the restored/reset sim state.
        // Does NOT trigger a full TLAS rebuild — the VDBVolume objects and their TLAS
        // instances remain valid. Only the NanoVDB buffer contents change, which is
        // handled cheaply by re-registering via registerOrUpdateLiveVolume and setting
        // g_gas_volumes_dirty (SSBO re-sync). Setting rebuild pending here caused a full
        // GPU TLAS rebuild on every cached frame restore (restoreSimFrame), i.e. every
        // single frame during timeline playback.
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
        system.domain_sdf_stats.clear();
        system.domain_sdf_signatures.clear();
        system.domain_vdb_upload_signatures.clear();
        system.domain_last_fluid_render_mode.clear();
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
        if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0 ||
            grid.voxel_size <= 0.0f || obj.particles.empty()) {
            binding.density.clear();
            obj.sdf.clear();
            return false;
        }

        const bool built = RayTrophiSim::Fluid::buildLevelSet(
            obj.particles, grid, obj.level_set_params, obj.sdf, &obj.level_set_stats);
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
                vol->setShader(VolumeShader::createSmokePreset());
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
                auto shader = binding.volume->getShader();
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
    enum class ParticleSystemPreset { Campfire = 0, Explosion = 1, Smoke = 2 };

    ParticleSystemObject& addParticleSystemPreset(ParticleSystemPreset preset) {
        const char* preset_name = "Particle System";
        switch (preset) {
            case ParticleSystemPreset::Campfire:  preset_name = "Campfire";  break;
            case ParticleSystemPreset::Explosion: preset_name = "Explosion"; break;
            case ParticleSystemPreset::Smoke:     preset_name = "Smoke";     break;
        }
        // Apply to the ACTIVE system (create one only if none exists) instead of
        // spawning a brand-new system on every click — consecutive preset presses
        // would otherwise pile up systems that all simulate + refit at once. To get
        // a separate effect, click "Add Particle System" first, then a preset.
        ParticleSystemObject& sys = ensureActiveParticleSystemObject();
        auto rt = sys.runtime;
        if (!rt) return sys;

        // Clean slate so the preset fully defines the system (drop old render group,
        // domain volumes, emitters/colliders/domains/flow + any live particles).
        destroyParticleRenderGroup(sys);
        destroyDomainVolumes(sys);
        rt->clear();
        rt->clearEmitters();
        rt->clearColliders();
        rt->clearGridDomains();
        rt->clearFlowSources();
        rt->resetGridDomainStates();
        sys.render = ParticleRenderSettings{};
        sys.name = preset_name;

        switch (preset) {
            case ParticleSystemPreset::Campfire: {
                rt->applyPhysicsModePreset(RayTrophiSim::ParticlePhysicsMode::Spark);
                rt->applyQualityModePreset(RayTrophiSim::ParticleQualityMode::Realtime);
                rt->setGravity(Vec3(0.0f, -1.6f, 0.0f));   // gentle: sparks rise then drift down
                rt->setLinearDrag(0.5f);
                RayTrophiSim::ParticleEmitterDesc e;
                e.name = "Campfire Emitter";
                e.point = Vec3(0.0f, 0.1f, 0.0f);
                e.direction = Vec3(0.0f, 1.0f, 0.0f);
                e.rate_per_second = 70.0f;
                e.speed = 1.6f;
                e.spread = 0.35f;
                e.lifetime_seconds = 1.4f;
                e.start_size = 0.08f;  e.end_size = 0.01f;  e.size_jitter = 0.5f;
                e.start_opacity = 1.0f; e.end_opacity = 0.0f;
                e.start_color = Vec3(1.0f, 0.8f, 0.35f); e.end_color = Vec3(0.9f, 0.15f, 0.03f);
                e.angular_velocity = 2.0f; e.angular_jitter = 3.0f;
                rt->addEmitter(e);
                sys.blend_mode = ParticleBlendMode::Additive;
                sys.render.render_in_raytrace = true;
                sys.render.shape = ParticleRenderShape::Sphere;
                sys.render.emissive = true;
                sys.render.base_color = Vec3(1.0f, 0.75f, 0.3f);
                sys.render.color_end  = Vec3(1.0f, 0.2f, 0.05f);
                sys.render.color_buckets = 10;
                sys.render.emission_strength = 8.0f;
                break;
            }
            case ParticleSystemPreset::Explosion: {
                rt->applyPhysicsModePreset(RayTrophiSim::ParticlePhysicsMode::Spark);
                rt->applyQualityModePreset(RayTrophiSim::ParticleQualityMode::Realtime);
                rt->setGravity(Vec3(0.0f, -9.81f, 0.0f));
                rt->setLinearDrag(0.12f);
                rt->setCollisionPlane(0.0f, true, 0.3f);   // debris bounces on the ground
                RayTrophiSim::ParticleEmitterDesc e;
                e.name = "Explosion Burst";
                e.point = Vec3(0.0f, 1.0f, 0.0f);
                e.direction = Vec3(0.0f, 1.0f, 0.0f);
                e.rate_per_second = 0.0f;
                e.burst_count = 400;
                e.speed = 6.0f;
                e.spread = 3.0f;          // near-omnidirectional
                e.lifetime_seconds = 2.0f;
                e.start_size = 0.1f;  e.end_size = 0.04f;  e.size_jitter = 0.6f;
                e.start_opacity = 1.0f; e.end_opacity = 0.0f;
                e.start_color = Vec3(1.0f, 0.9f, 0.5f); e.end_color = Vec3(0.3f, 0.08f, 0.02f);
                e.angular_velocity = 4.0f; e.angular_jitter = 8.0f;
                rt->addEmitter(e);
                sys.blend_mode = ParticleBlendMode::Additive;
                sys.render.render_in_raytrace = true;
                sys.render.shape = ParticleRenderShape::Tetra;  // chunky debris (or set SceneMeshes)
                sys.render.emissive = true;
                sys.render.base_color = Vec3(1.0f, 0.8f, 0.4f);
                sys.render.color_end  = Vec3(0.25f, 0.06f, 0.02f);
                sys.render.color_buckets = 8;
                sys.render.emission_strength = 6.0f;
                break;
            }
            case ParticleSystemPreset::Smoke: {
                rt->applyPhysicsModePreset(RayTrophiSim::ParticlePhysicsMode::Gas);
                rt->applyQualityModePreset(RayTrophiSim::ParticleQualityMode::Preview);
                RayTrophiSim::SimulationGridDomainDesc dom;
                dom.name = "Smoke Domain";
                dom.bounds_min = Vec3(-2.0f, 0.0f, -2.0f);
                dom.bounds_max = Vec3(2.0f, 5.0f, 2.0f);
                dom.resolution_x = dom.resolution_y = dom.resolution_z = 64;
                dom.fire_enabled = false;                       // smoke only, no combustion
                dom.shader = VolumeShader::createSmokePreset();
                rt->addGridDomain(dom);
                RayTrophiSim::SimulationFlowSourceDesc fs;
                fs.name = "Smoke Source";
                fs.position = Vec3(0.0f, 0.3f, 0.0f);
                fs.velocity = Vec3(0.0f, 1.5f, 0.0f);
                fs.radius = 0.35f;
                fs.density = 1.0f;
                fs.temperature = 0.4f;
                rt->addFlowSource(fs);
                sys.render.render_in_raytrace = false;          // volumetric, drawn by the VDB bridge
                break;
            }
        }

        applyParticleSystemEnabledState(sys);
        return sys;
    }

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
                if (source.source_mode == RayTrophiSim::SimulationFlowSourceMode::ObjectBounds &&
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
                            return source.source_mode == RayTrophiSim::SimulationFlowSourceMode::ObjectBounds &&
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
                            return source.source_mode == RayTrophiSim::SimulationFlowSourceMode::ObjectBounds &&
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
            last_sim_pose_applied_.clear();  // drop stale sim-pose memo on full reset/reload
        } else {
            surface_mesh_cache.erase(node_name);
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
                const Vec3* P  = g->get_attribute_data<Vec3>("P");       // world-baked
                const Vec2* uv = g->get_attribute_data<Vec2>("uv");
                const uint16_t* mat = g->get_attribute_data<uint16_t>("materialID");
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

    bool resolveObjectBoundsForSimulation(const std::string& node_name, Vec3& out_min, Vec3& out_max) const {
        const auto* surface_cache = getSurfaceMeshCacheForObject(node_name);
        if (!surface_cache) {
            return false;
        }
        out_min = surface_cache->bounds_min;
        out_max = surface_cache->bounds_max;
        return true;
    }

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
                        const auto* surface_cache = getSurfaceMeshCacheForObject(emitter.source_name);
                        if (!surface_cache) {
                            return false;
                        }
                        out_position = surface_cache->centroid + emitter.local_offset;
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
                    if (source.source_mode != RayTrophiSim::SimulationFlowSourceMode::ObjectBounds ||
                        source.source_name.empty()) {
                        return false;
                    }
                    return resolveObjectBoundsForSimulation(source.source_name, out_min, out_max);
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
    void rebuildSDFColliderAsync(RayTrophiSim::ParticleColliderDesc& desc,
                                 std::shared_ptr<RayTrophiSim::ParticleSimulationSystem> target_runtime = nullptr) {
        if (desc.source_mode != RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF ||
            desc.source_name.empty()) {
            return;
        }

        std::string node_name = desc.source_name;
        int res_mode = desc.sdf_resolution_mode;
        
        int N = 64;
        if (res_mode == 0) N = 32;
        else if (res_mode == 1) N = 64;
        else if (res_mode == 2) N = 128;

        const auto* surface_cache = getSurfaceMeshCacheForObject(node_name);
        if (!surface_cache || surface_cache->empty()) {
            return;
        }

        RayTrophiSim::ParticleColliderOBB obb;
        if (!resolveObjectOBBForSimulation(node_name, obb)) {
            return;
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

        std::thread([this, node_name, triangles, bmin, bmax, N, result_vec, target_runtime]() {
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
                g_active_sdf_bakes.fetch_sub(1, std::memory_order_release);
                return;
            }

            auto p_sys = target_runtime ? target_runtime : this->getParticleSimulationSystem();
            if (p_sys) {
                auto& list = p_sys->colliders();
                for (auto& coll : list) {
                    if (coll.source_mode == RayTrophiSim::ParticleColliderSourceMode::ObjectMeshSDF &&
                        coll.source_name == node_name) {
                        coll.sdf_grid_data = result_vec;
                        coll.sdf_origin = origin;
                        coll.sdf_extents = extents;
                        coll.sdf_nx = nx;
                        coll.sdf_ny = ny;
                        coll.sdf_nz = nz;
                        break;
                    }
                }
            }
            g_active_sdf_bakes.fetch_sub(1, std::memory_order_release);
        }).detach();
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

    RayTrophiSim::SimulationGridDomainDesc& addSimulationGridDomainFromObject(const std::string& node_name) {
        RayTrophiSim::SimulationGridDomainDesc desc;
        desc.name = node_name.empty() ? "Grid Domain" : node_name + " Domain";
        desc.source_mode = RayTrophiSim::SimulationGridDomainSourceMode::ManualBox;
        desc.source_name.clear();
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

        if (vulkan_only_requested && !auto_gpu_requested) {
            // Explicit Vulkan-only path (testing / non-NVIDIA systems)
            if (compute.backendType() != RayTrophiSim::ComputeBackendType::VulkanCompute) {
                auto vk_backend =
                    RayTrophiSim::createVulkanSimulationComputeBackend(g_vulkan_sim_compute_ctx);
                g_hasVulkanComputeSim = (vk_backend != nullptr);
                compute.setBackend(std::move(vk_backend));
            }
        } else if (auto_gpu_requested || vulkan_only_requested) {
            // Auto: try CUDA first, fall back to Vulkan
            if (compute.backendType() == RayTrophiSim::ComputeBackendType::CUDA) {
                // Already on CUDA — keep it
            } else if (compute.backendType() == RayTrophiSim::ComputeBackendType::VulkanCompute) {
                // Already on Vulkan — keep it
            } else {
                // Try CUDA
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
        last_sim_pose_applied_.clear();
        soft_weld_cache_.clear();
        rigid_bake_cache_.clear();
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
        soft_frame_cache_.clear();
        invalidateSurfaceMeshCache();

        // Reset Post-Processing to defaults
        color_processor = ColorProcessor();

        initialized = false;
    }
};

