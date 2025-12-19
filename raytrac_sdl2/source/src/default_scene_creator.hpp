#ifndef DEFAULT_SCENE_CREATOR_HPP
#define DEFAULT_SCENE_CREATOR_HPP

#include "scene_data.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Texture.h"
#include "Triangle.h" 
#include "PointLight.h"
#include "Camera.h"
#include "Transform.h"
#include "Renderer.h"
#include <filesystem>
#include <string>

// Helper function to create a default Blender-like startup scene
inline void createDefaultScene(SceneData& scene, Renderer& renderer, OptixWrapper* optix_gpu) {
    if (scene.initialized && (!scene.world.objects.empty() || !scene.lights.empty())) {
         return; // Scene already has content
    }
    
    // Check for default asset file
    std::string defaultAssetPath = "assets/default.glb";
    bool found = false;
    try {
        if (std::filesystem::exists(defaultAssetPath)) found = true;
        else if (std::filesystem::exists("../assets/default.glb")) {
            defaultAssetPath = "../assets/default.glb";
            found = true;
        }
        else if (std::filesystem::exists("default.glb")) {
            defaultAssetPath = "default.glb";
            found = true;
        }
    } catch (const std::exception& e) {
        SCENE_LOG_WARN("Filesystem check failed: " + std::string(e.what()));
        found = false;
    }

    if (found) {
        SCENE_LOG_INFO("Loading Default Scene Asset: " + defaultAssetPath);
        // Load the GLB/GLTF file
        renderer.create_scene(scene, optix_gpu, defaultAssetPath, nullptr);
        
        // If camera loaded from GLB is null, create a default one
        if (!scene.camera) {
             Vec3 lookfrom(7.0f, 5.0f, 7.0f);
             Vec3 lookat(0.0f, 0.0f, 0.0f);
             Vec3 vup(0.0f, 1.0f, 0.0f);
             float dist_to_focus = (lookfrom - lookat).length();
             float ar = 16.0f/9.0f; 
             scene.camera = std::make_shared<Camera>(lookfrom, lookat, vup, 40.0f, ar, 0.0f, dist_to_focus, 0);
        }
        
        // If no lights, add a default point light
        if (scene.lights.empty()) {
             auto light = std::make_shared<PointLight>(Vec3(4.0f, 4.0f, 4.0f), Vec3(50.0f, 50.0f, 50.0f), 0.2f);
             scene.lights.push_back(light);
        }
        
        scene.initialized = true;
        return;
    }

    // Fallback: Create Cube programmatically if file not found
    // Fallback: Default scene with NO OBJECTS (Empty), just light and camera
    SCENE_LOG_INFO("Default asset not found. Creating Empty Scene (Light + Camera only).");

    // 2. Create Light (Point Light)
    // PointLight(pos, intensity, radius)
    auto light = std::make_shared<PointLight>(Vec3(4.0f, 4.0f, 4.0f), Vec3(50.0f, 50.0f, 50.0f), 0.2f);
    scene.lights.push_back(light);
    
    // 3. Setup Camera (if not already set)
    if (!scene.camera) {
        Vec3 lookfrom(7.0f, 5.0f, 7.0f); // Isometric-ish view
        Vec3 lookat(0.0f, 0.0f, 0.0f);
        Vec3 vup(0.0f, 1.0f, 0.0f);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.0f;
        float vfov = 40.0f;
        
        // Ensure aspect_ratio global is available or passed. 
        // Main.cpp defines aspect_ratio global. We might need extern or pass it.
        // Assuming extern declaration in globals.h or similar if included via scene_data.h helpers.
        // If not, use generic 16:9
        float ar = 16.0f/9.0f;
        
        scene.camera = std::make_shared<Camera>(lookfrom, lookat, vup, vfov, ar, aperture, dist_to_focus, 0);
    }

    scene.initialized = true;
    SCENE_LOG_INFO("Created Default Empty Scene (Light + Camera)");
}

#endif
