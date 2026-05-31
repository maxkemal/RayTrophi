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
#include "SimulationSystems.h"
#include "SimulationWorld.h"
#include "SimulationComputeVulkanContext.h"
#include <thread>
#include "Fluid/FluidObject.h"
#include "Fluid/FluidSimulationSystem.h"
#include "globals.h"
#include "SurfaceMeshCache.h"
#include "MeshModifiers.h"
#include "Paint/PaintTextureSet.h"
#include "Paint/PaintLayerStack.h"

#include <functional>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <iterator>
#include <utility>
#include <unordered_map>
#include <map>
#include <unordered_set>

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
    
    // Non-destructive Modeling Cache
    std::unordered_map<std::string, std::vector<std::shared_ptr<Triangle>>> base_mesh_cache; // nodeName -> list of Triangles
    mutable std::unordered_map<std::string, RayTrophiSim::SurfaceMeshCache> surface_mesh_cache; // shared wet/particle/collider surface cache
    mutable uint64_t surface_mesh_cache_version = 1;
    std::unordered_map<std::string, MeshModifiers::ModifierStack> mesh_modifiers;          // nodeName -> Modifier Stack
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
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            return tri && isEditorPendingDeleteObjectName(tri->nodeName);
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
    int sim_timeline_frame_ = -1;
    static constexpr int kMaxCachedSimFrames = 600;

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
                if (fluid_surface_route && d < system.domain_sdf_buffers.size()) {
                    auto& sdf_buf = system.domain_sdf_buffers[d];
                    auto& sdf_stats = system.domain_sdf_stats[d];
                    RayTrophiSim::Fluid::buildLevelSet(
                        state.particles, state.grid,
                        domains[d].fluid_level_set_params, sdf_buf, &sdf_stats);
                    // SDF may be refined above the sim grid (surface_resolution_
                    // multiplier), so size the proxy loop from the buffer itself,
                    // not the sim grid cell count.
                    const std::size_t cells = sdf_buf.size();
                    if (cells > 0) {
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
                const bool do_update = force_sync || (prev_id < 0) || ((frame % stride) == 0);
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
                    system.domain_vdb_ids[d] = mgr.registerOrUpdateLiveVolume(
                        prev_id,
                        volume_name,
                        up_nx, up_ny, up_nz,
                        up_voxel,
                        density_ptr,
                        up_temp,
                        nullptr);
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
                // Mark the volume's render route every frame — the SDF proxy
                // density is the same NanoVDB channel either way; the shader
                // picks "fog raymarch" vs "isosurface walk + refraction" based
                // on render_as_isosurface (mapped to source_type=4 downstream).
                vol->render_as_isosurface = fluid_surface_route;
                if (fluid_surface_route && d < domains.size()) {
                    vol->render_isosurface_ior = domains[d].fluid_surface_ior;
                    vol->render_isosurface_roughness = domains[d].fluid_surface_roughness;
                    vol->render_isosurface_foam = domains[d].fluid_surface_foam;
                }
                if (domain_shader) {
                    vol->setShader(domain_shader);  // pick up live shader edits
                }
                vol->bindLiveVolume(id, state.grid.voxel_size, world_min, world_max);

                if (created) {
                    // New hittable added to world.objects: rebuild GPU TLAS so it
                    // gets primary-ray hits and re-sync volume buffers.
                    // (No CPU BVH flag: live volumes are not CPU-sampleable.)
                    // became_visible is NOT treated as a structural change — the TLAS
                    // instance was already registered on 'created', so showing it again
                    // only requires an SSBO update (g_gas_volumes_dirty) to re-activate
                    // the slot. Triggering a full rebuild on became_visible caused a
                    // rebuild every time density went 0→non-zero (e.g. on timeline loop).
                    g_geometry_dirty = true;
                    g_vulkan_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    g_gas_volumes_dirty = true;
                } else if (became_visible) {
                    g_gas_volumes_dirty = true;
                }
            }
        }

        syncFluidRenderVolumes(mgr, frame, force_sync);
        syncFluidFoamVolumes(mgr, frame, force_sync);

        force_simulation_render_sync_ = false;
    }

    // Whitewater "Volume" render mode was REMOVED (it splatted foam into a white
    // scattering NanoVDB that fought the fluid surface volume on the Vulkan
    // integrator → black cube). Foam now renders as Surface (metaball mesh, see
    // syncFluidFoamRenderInstances) or Spheres. This routine only TEARS DOWN any
    // foam volumes left over from an older session/save so they stop rendering.
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
    }

    bool restoreSimFrame(int frame, float fixed_dt = 1.0f / 24.0f) {
        auto it = sim_frame_cache_.find(frame);
        if (it == sim_frame_cache_.end() || it->second.size() != particle_systems.size()) {
            return false;
        }
        for (std::size_t i = 0; i < particle_systems.size(); ++i) {
            if (particle_systems[i].runtime) {
                particle_systems[i].runtime->setGridDomainStates(it->second[i]);
                invalidateSimulationRenderBindings(particle_systems[i]);
            }
        }
        simulation_world.resetTime(static_cast<float>(frame) * fixed_dt, frame);
        return true;
    }

    void resetSimulationToStart() {
        clearSimFrameCache();
        for (auto& system : particle_systems) {
            if (system.runtime) {
                invalidateSimulationRenderBindings(system);
                system.runtime->resetGridDomainStates();
                system.runtime->clear();  // particles back to empty for a deterministic bake
            }
        }
        simulation_world.resetTime(0.0f, 0);
        captureSimFrame(0);
    }

    // Return to interactive free-run preview (default mode).
    void resetSimulation() {
        resetSimulationToStart();
        sim_timeline_frame_ = -1;
        syncSimulationRenderVolumes();
    }

    void requestSimulationTimelineRenderResync() {
        force_simulation_render_sync_ = true;
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
    void updateSimulationTimeline(int tl_frame, bool playing, float realtime_dt, float fps, bool live_mode) {
        if (tl_frame < 0) tl_frame = 0;
        simulation_render_updated = false;
        const bool force_resync = force_simulation_render_sync_;

        // Live Update: free-run whenever the timeline is not actively playing.
        if (live_mode && !playing) {
            sim_timeline_frame_ = -1;  // detached from the baked timeline
            syncSimulationWorld();
            simulation_world.stepOnce(realtime_dt);
            syncSimulationRenderVolumes();
            return;
        }

        // Timeline-driven deterministic bake / scrub.
        const float fixed_dt = (fps > 1.0f) ? (1.0f / fps) : (1.0f / 24.0f);
        constexpr int kMaxStepsPerTick = 8;  // spread big jumps across UI ticks
        bool changed = false;

        if (force_resync && !playing && restoreSimFrame(tl_frame, fixed_dt)) {
            sim_timeline_frame_ = tl_frame;
            changed = true;
        }

        // Fresh bake on first entry, or when playback loops back before our frame.
        if (sim_timeline_frame_ < 0 || (playing && tl_frame < sim_timeline_frame_)) {
            resetSimulationToStart();
            sim_timeline_frame_ = 0;
            changed = true;
        }

        if (tl_frame != sim_timeline_frame_) {
            if (restoreSimFrame(tl_frame, fixed_dt)) {
                sim_timeline_frame_ = tl_frame;
                changed = true;
            } else {
                // Uncached: rewind to nearest cached <= target, then resim (capped).
                if (tl_frame < sim_timeline_frame_) {
                    const int nearest = nearestCachedSimFrameAtOrBelow(tl_frame);
                    if (nearest >= 0 && restoreSimFrame(nearest, fixed_dt)) {
                        sim_timeline_frame_ = nearest;
                    } else {
                        resetSimulationToStart();
                        sim_timeline_frame_ = 0;
                    }
                    changed = true;
                }
                int steps = 0;
                while (sim_timeline_frame_ < tl_frame && steps < kMaxStepsPerTick) {
                    syncSimulationWorld();
                    simulation_world.stepOnce(fixed_dt);
                    ++sim_timeline_frame_;
                    captureSimFrame(sim_timeline_frame_);
                    ++steps;
                    changed = true;
                }
            }
        }

        // Only touch the renderer when something actually changed; otherwise the
        // timeline is frozen and the path tracer is allowed to converge + idle.
        if (changed) {
            syncSimulationRenderVolumes();
        }
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
        int written = 0;
        for (int f = 0; f <= end_frame; ++f) {
            if (f > 0) {
                syncSimulationWorld();
                simulation_world.stepOnce(dt);
                sim_timeline_frame_ = f;
            }
            if (f >= start_frame) {
                std::string num = std::to_string(f);
                while (num.size() < 4) num = "0" + num;
                const std::string path = directory + "/" + base + "_" + num + ".vdb";
                if (exportDomainVDB(system_index, domain_index, path)) ++written;
            }
        }
        sim_timeline_frame_ = -1;
        syncSimulationRenderVolumes();
        return written;
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
        return !particle_systems.empty() || !fluid_objects.empty() || !gas_volumes.empty();
    }

    bool hasLiveSimulationObject(const std::string& node_name) const {
        if (node_name.empty() || isEditorPendingDeleteObjectName(node_name)) {
            return false;
        }
        for (const auto& obj : world.objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
            if (tri && tri->getNodeName() == node_name) {
                return true;
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
        } else {
            surface_mesh_cache.erase(node_name);
        }
        ++surface_mesh_cache_version;
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
            surface_mesh_cache.erase(node_name);
            return nullptr;
        }

        auto& cache = surface_mesh_cache[node_name];
        cache = RayTrophiSim::SurfaceMeshCache::build(node_name, triangles, surface_mesh_cache_version);
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

    void rebuildSDFColliderAsync(RayTrophiSim::ParticleColliderDesc& desc) {
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

        std::thread([this, node_name, triangles, bmin, bmax, N, result_vec]() {
            Vec3 size = bmax - bmin;
            Vec3 pad = size * 0.15f;
            Vec3 origin = bmin - pad;
            Vec3 extents = size + pad * 2.0f;

            int nx = N;
            int ny = N;
            int nz = N;
            result_vec->resize(static_cast<std::size_t>(nx * ny * nz), 0.0f);

            auto pointTriangleDistanceSquared = [](const Vec3& p, const Vec3& a, const Vec3& b, const Vec3& c, Vec3& out_closest) -> float {
                Vec3 ab = b - a;
                Vec3 ac = c - a;
                Vec3 ap = p - a;
                float d1 = Vec3::dot(ab, ap);
                float d2 = Vec3::dot(ac, ap);
                if (d1 <= 0.0f && d2 <= 0.0f) {
                    out_closest = a;
                    return (p - a).length_squared();
                }
                Vec3 bp = p - b;
                float d3 = Vec3::dot(ab, bp);
                float d4 = Vec3::dot(ac, bp);
                if (d3 >= 0.0f && d4 <= d3) {
                    out_closest = b;
                    return (p - b).length_squared();
                }
                float vc = d1 * d4 - d3 * d2;
                if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
                    float v = d1 / (d1 - d3);
                    out_closest = a + ab * v;
                    return (p - out_closest).length_squared();
                }
                Vec3 cp = p - c;
                float d5 = Vec3::dot(ab, cp);
                float d6 = Vec3::dot(ac, cp);
                if (d6 >= 0.0f && d5 <= d6) {
                    out_closest = c;
                    return (p - c).length_squared();
                }
                float vb = d5 * d2 - d1 * d6;
                if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
                    float w = d2 / (d2 - d6);
                    out_closest = a + ac * w;
                    return (p - out_closest).length_squared();
                }
                float va = d3 * d6 - d5 * d4;
                if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
                    float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                    out_closest = b + (c - b) * w;
                    return (p - out_closest).length_squared();
                }
                float denom = 1.0f / (va + vb + vc);
                float v = vb * denom;
                float w = vc * denom;
                out_closest = a + ab * v + ac * w;
                return (p - out_closest).length_squared();
            };

            for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                Vec3 cell_p = origin + Vec3(
                    (i + 0.5f) * (extents.x / nx),
                    (j + 0.5f) * (extents.y / ny),
                    (k + 0.5f) * (extents.z / nz)
                );

                float min_dist_sq = std::numeric_limits<float>::max();
                Vec3 best_closest(0.0f);
                Vec3 best_normal(0.0f, 1.0f, 0.0f);

                std::size_t stride = triangles.size() > 5000 ? (triangles.size() / 5000 + 1) : 1;
                for (std::size_t t = 0; t < triangles.size(); t += stride) {
                    const auto& tri = triangles[t];
                    Vec3 closest;
                    float d_sq = pointTriangleDistanceSquared(cell_p, tri.p0, tri.p1, tri.p2, closest);
                    if (d_sq < min_dist_sq) {
                        min_dist_sq = d_sq;
                        best_closest = closest;
                        best_normal = tri.normal;
                    }
                }

                float dist = std::sqrt(min_dist_sq);
                Vec3 delta = cell_p - best_closest;
                if (Vec3::dot(delta, best_normal) < 0.0f) {
                    dist = -dist;
                }

                result_vec->at(static_cast<std::size_t>(k * (nx * ny) + j * nx + i)) = dist;
            }

            auto p_sys = this->getParticleSimulationSystem();
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
        if (auto runtime = activeParticleRuntime()) {
            runtime->clearColliders();
        }
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

    bool sim_compute_selftest_done_ = false;
    void syncSimulationWorld() {
        simulation_world.setForceFieldManager(&force_field_manager);
        // Phase 2: validate the CUDA compute pipeline once (logs OK/FAILED). The
        // global compute backend is not switched yet — the GPU solver path lands
        // in Phase 3.
        if (!sim_compute_selftest_done_) {
            sim_compute_selftest_done_ = true;
            RayTrophiSim::selfTestCudaSimulationCompute();
        }

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
        syncSimulationWorld();
        world.clear();
        lights.clear();
        cameras.clear();
        animationDataList.clear();
        boneData.clear();              // Clear bone hierarchy
        timeline.clear();              // Clear keyframes
        ui_settings_json_str = "";     // Clear UI settings string
        load_counter = 0;              // Reset load counter
        
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

        // Detach the FluidSimulationSystem's raw pointer to fluid_objects
        // BEFORE clearing the vector. This prevents any stale-pointer access
        // during destruction ordering or if a system dtor triggers a step.
        if (fluid_simulation_system) {
            fluid_simulation_system->setObjects(nullptr);
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
        invalidateSurfaceMeshCache();

        // Reset Post-Processing to defaults
        color_processor = ColorProcessor();

        initialized = false;
    }
};

