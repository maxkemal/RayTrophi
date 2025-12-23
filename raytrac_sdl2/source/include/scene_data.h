#pragma once
#include <HittableList.h>
#include <AssimpLoader.h>
#include <AnimatedObject.h>
#include "KeyframeSystem.h"

struct SceneData {
    HittableList world;
    std::shared_ptr<Hittable> bvh;
    std::vector<AnimationData> animationDataList;
    std::vector<std::shared_ptr<AnimatedObject>> animatedObjects;
    
    // Multi-camera support
    std::vector<std::shared_ptr<Camera>> cameras;  // All cameras in scene
    size_t active_camera_index = 0;                 // Index of currently active camera
    
    // Convenience accessor for active camera (for backward compatibility)
    std::shared_ptr<Camera> camera;  // Points to active camera
    
    std::vector<std::shared_ptr<Light>> lights;
    Vec3 background_color = Vec3(0.2, 0.2, 0.2);
    bool initialized = false;
    BoneData boneData;
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
    
    void clear() {
        world.clear();
        lights.clear();
        cameras.clear();
        animatedObjects.clear();
        animationDataList.clear();
        timeline.clear();  // Clear keyframes
        camera = nullptr;
        active_camera_index = 0;
        bvh = nullptr;
        initialized = false;
    }
};
