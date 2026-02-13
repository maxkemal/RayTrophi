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

#include <string>

namespace AnimationGraph {
    class AnimationNodeGraph;
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
    // UI Settings Serialization Helper
    std::string ui_settings_json_str;  // JSON string storing UI settings
    int load_counter = 0;              // Incremented when a project is loaded
    // =========================================================================
    // Core Geometry
    // =========================================================================
    HittableList world;                                    // All renderable objects
    std::shared_ptr<Hittable> bvh;                         // Acceleration structure
    
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
        std::shared_ptr<class AssimpLoader> loader; // Keep loader alive (owns aiScene)
        std::string importName;
        bool hasAnimation = false;                  // True if this model has animation data
        Matrix4x4 globalInverseTransform;           // Matrix to correct FBX axis/scale (from Root node)
        
        // --- Multi-Animator Logic ---
        std::shared_ptr<class AnimationController> animator;  // Per-model animator state
        std::shared_ptr<AnimationGraph::AnimationNodeGraph> graph; // Per-model AnimGraph
        bool useAnimGraph = false;                  // Toggle between Controller and Node Graph
        bool useRootMotion = false;                 // Move object transform with character
        bool visible = true;                        // Visibility toggle for the whole model
        
        // Link to scene world objects (Triangles/Meshes) belonging to this model
        // This allows applying root motion to the correct TransformHandle
        std::vector<std::shared_ptr<class Hittable>> members; 
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
    
    // Add a gas volume to the scene
    void addGasVolume(std::shared_ptr<GasVolume> gas) {
        if (gas) {
            static int gas_id_counter = 0;
            gas->id = gas_id_counter++;
            
            // LINK TO FORCE FIELDS: Critical for simulation to respond to fields
            gas->getSimulator().setExternalForceFieldManager(&this->force_field_manager);
            
            gas_volumes.push_back(gas);
        }
    }
    
    // Remove a gas volume from the scene
    bool removeGasVolume(std::shared_ptr<GasVolume> gas) {
        auto it = std::find(gas_volumes.begin(), gas_volumes.end(), gas);
        if (it != gas_volumes.end()) {
            gas_volumes.erase(it);
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
        for (auto& gas : gas_volumes) {
            if (gas) {
                // Keep linkage updated (useful if manager pointer ever changes or during reload)
                gas->getSimulator().setExternalForceFieldManager(&this->force_field_manager);
                gas->update(dt);
            }
        }
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
    // Force Fields (Universal Physics System)
    // =========================================================================
    Physics::ForceFieldManager force_field_manager;
    
    // Add a force field to the scene
    int addForceField(std::shared_ptr<Physics::ForceField> field) {
        return force_field_manager.addForceField(field);
    }
    
    // Remove a force field from the scene
    bool removeForceField(std::shared_ptr<Physics::ForceField> field) {
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
            ctx.members.clear();
        }
        importedModelContexts.clear(); // Clear model contexts (releases aiScene memory)
        
        vdb_volumes.clear();           // Clear VDB volumes
        gas_volumes.clear();           // Clear gas volumes
        force_field_manager.clear();   // Clear force fields
        camera = nullptr;
        active_camera_index = 0;
        bvh = nullptr;
        initialized = false;
    }
};

