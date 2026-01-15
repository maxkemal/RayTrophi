#pragma once
#include <HittableList.h>
#include <AssimpLoader.h>
#include "KeyframeSystem.h"
#include "VDBVolume.h"

#include <string>

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
    std::vector<AnimationData> animationDataList;          // File-based animations
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
        bool hasAnimation = false; // True if this model has animation data
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
    // Clear all scene data
    // =========================================================================
    void clear() {
        world.clear();
        lights.clear();
        cameras.clear();
        animationDataList.clear();
        boneData.clear();              // Clear bone hierarchy
        timeline.clear();              // Clear keyframes
        importedModelContexts.clear(); // Clear model contexts (releases aiScene memory)
        vdb_volumes.clear();           // Clear VDB volumes
        camera = nullptr;
        active_camera_index = 0;
        bvh = nullptr;
        initialized = false;
    }
};
