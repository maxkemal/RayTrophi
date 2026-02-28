/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          default_scene_creator.hpp
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef DEFAULT_SCENE_CREATOR_HPP
#define DEFAULT_SCENE_CREATOR_HPP

#include "scene_data.h"
#include "Triangle.h" 
#include "Vec2.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "Camera.h"
#include "Renderer.h"
#include "World.h"
#include "MaterialManager.h"
#include "Material.h"
#include "PrincipledBSDF.h"
#include "material_gpu.h"
#include <vector_types.h> // For make_float3
#include <filesystem>
#include <string>


// Helper function to create a default Blender-like startup scene
inline void createDefaultScene(SceneData& scene, Renderer& renderer, Backend::IBackend* backend) {
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
        renderer.create_scene(scene, backend, defaultAssetPath, nullptr);
        
        // ONLY create default camera if NO cameras were loaded from file
        // The cameras list is the source of truth - if it has cameras, don't create another
        if (scene.cameras.empty()) {
             SCENE_LOG_INFO("No camera in GLB file, creating default camera");
             Vec3 lookfrom(7.0f, 5.0f, 7.0f);
             Vec3 lookat(0.0f, 0.0f, 0.0f);
             Vec3 vup(0.0f, 1.0f, 0.0f);
             float dist_to_focus = (lookfrom - lookat).length();
             float ar = 16.0f/9.0f; 
             auto default_cam = std::make_shared<Camera>(lookfrom, lookat, vup, 40.0f, ar, 0.0f, dist_to_focus, 0);
             default_cam->nodeName = "Default Camera";
             scene.addCamera(default_cam);
        } else {
             // Cameras loaded from file - ensure scene.camera pointer is synced
             if (!scene.camera && !scene.cameras.empty()) {
                 scene.setActiveCamera(0);
             }
             SCENE_LOG_INFO("Using camera from GLB file (total: " + std::to_string(scene.cameras.size()) + ")");
        }
        
        // SYNC NISHITA WITH SCENE SUN (Directional Light)
        for (const auto& light : scene.lights) {
            if (light->type() == LightType::Directional) {
                SCENE_LOG_INFO("Found Directional Light in default asset. Syncing Nishita Sun...");
                
                Vec3 lightDir = light->direction.normalize();
                Vec3 sunDir = -lightDir; 
                
                float elevRad = asinf(sunDir.y);
                float elevDeg = elevRad * 180.0f / 3.14159265f;
                
                float azimRad = atan2f(sunDir.x, sunDir.z); 
                float azimDeg = azimRad * 180.0f / 3.14159265f;
                if (azimDeg < 0.0f) azimDeg += 360.0f;
                
                NishitaSkyParams nishita = renderer.world.getNishitaParams();
                nishita.sun_elevation = elevDeg;
                nishita.sun_azimuth = azimDeg;
                nishita.sun_direction = make_float3(sunDir.x, sunDir.y, sunDir.z); // Positive Z logic
                
                renderer.world.setNishitaParams(nishita);
                break;
            }
        }
        
        scene.initialized = true;
        return;
    }
    

    // Fallback: Create Professional Empty Scene - Ground Plane + Nishita Sky
    SCENE_LOG_INFO("Default asset not found. Creating Professional Empty Scene.");
    
    // 0. Setup World Environment (Nishita Sky)
    renderer.world.setMode(WORLD_MODE_NISHITA);
    
    NishitaSkyParams nishita = renderer.world.getNishitaParams();
    nishita.sun_elevation = 15.0f; // Golden hour
    nishita.sun_azimuth = 170.0f;   // Matches camera angle
    nishita.sun_intensity = 10.0f;
    
    // CRITICAL: Must update sun_direction vector manually as it's used by renderers
    // Convert Elevation/Azimuth (Degrees) to Direction Vector (Y-up)
    float elevRad = nishita.sun_elevation * 3.14159265f / 180.0f;
    float azimRad = nishita.sun_azimuth * 3.14159265f / 180.0f;
    
    // Assuming Y-up, +Z forward (or standard math convention)
    // x = cos(elev) * sin(azim)
    // y = sin(elev)
    // z = -cos(elev) * cos(azim)
    float cosElev = cosf(elevRad);
    float sinElev = sinf(elevRad);
    float sinAzim = sinf(azimRad);
    float cosAzim = cosf(azimRad);
    
    // Adjust signs based on coordinate system if needed (Standard sky vector)
    // MATCHING UI LOGIC: Z is POSITIVE (cos * cos)
    Vec3 sunDir(
        cosElev * sinAzim,
        sinElev,
        cosElev * cosAzim
    );
    // Normalize just in case
    sunDir = sunDir.normalize();
    
    nishita.sun_direction = make_float3(sunDir.x, sunDir.y, sunDir.z);

    renderer.world.setNishitaParams(nishita);
    
    // 1. Create Ground Material (Needed for OptiX) - Use PrincipledBSDF
    auto ground_mat = std::make_shared<PrincipledBSDF>();
    ground_mat->gpuMaterial = std::make_shared<GpuMaterial>();
    // Set default values manually to match GpuMaterial layout
    ground_mat->gpuMaterial->albedo = make_float3(0.3f, 0.3f, 0.32f); // Neutral gray
    ground_mat->gpuMaterial->roughness = 0.8f;
    ground_mat->gpuMaterial->metallic = 0.0f;
    ground_mat->gpuMaterial->opacity = 1.0f;
    ground_mat->gpuMaterial->emission = make_float3(0.0f, 0.0f, 0.0f);
    ground_mat->gpuMaterial->transmission = 0.0f;
    ground_mat->gpuMaterial->ior = 1.45f;
    
    // Add to MaterialManager
    uint16_t groundMatId = MaterialManager::getInstance().addMaterial("Ground", ground_mat);

    // 2. Create Ground Plane (large flat surface)
    float planeSize = 50.0f;
    Vec3 p0(-planeSize, 0.0f, -planeSize);
    Vec3 p1( planeSize, 0.0f, -planeSize);
    Vec3 p2( planeSize, 0.0f,  planeSize);
    Vec3 p3(-planeSize, 0.0f,  planeSize);
    Vec3 normal(0.0f, 1.0f, 0.0f);
    
    // UV coordinates for ground plane
    Vec2 uv0(0.0f, 0.0f);
    Vec2 uv1(1.0f, 0.0f);
    Vec2 uv2(1.0f, 1.0f);
    Vec2 uv3(0.0f, 1.0f);
    
    // Use groundMatId
    auto tri1 = std::make_shared<Triangle>(p0, p1, p2, normal, normal, normal, uv0, uv1, uv2, groundMatId);
    auto tri2 = std::make_shared<Triangle>(p0, p2, p3, normal, normal, normal, uv0, uv2, uv3, groundMatId);
    tri1->nodeName = "Ground";
    tri2->nodeName = "Ground";
    scene.world.objects.push_back(tri1);
    scene.world.objects.push_back(tri2);
    
    // Note: Nishita sky already set up with correct sun_direction above (lines 92-126)
    // No need to override again here - it would lose the calculated sun_direction
    
    // 3. Create Directional Light matching sun direction (for sync to work)
    auto dirLight = std::make_shared<DirectionalLight>(
        -sunDir,  // Light direction is opposite of sun direction (light travels FROM sun)
        Vec3(1.0f, 0.98f, 0.95f),  // Warm white sun color
        10.0f    // Intensity
       
    );
    dirLight->nodeName = "Sun";
    scene.lights.push_back(dirLight);
    
    // Set fallback background color
    scene.background_color = Vec3(0.4f, 0.6f, 0.9f);
    
    // 3. Setup Camera - Looking towards sun and horizon (Landscape view)
    // Only create if NO cameras exist - cameras list is the source of truth
    if (scene.cameras.empty()) {
        // Sun is at Azimuth 45, Elevation 15.
        // Position camera "behind" the origin to look towards the sun (+X, +Z direction)
        Vec3 lookfrom(-12.0f, 3.0f, -12.0f); 
        Vec3 lookat(0.0f, 4.0f, 0.0f);     // Look slightly up/horizon
        Vec3 vup(0.0f, 1.0f, 0.0f);
        float dist_to_focus = (lookfrom - lookat).length();
        float aperture = 0.0f;
        float vfov = 50.0f;  // Wide FOV for landscape
        float ar = 16.0f/9.0f;
        
        auto default_cam = std::make_shared<Camera>(lookfrom, lookat, vup, vfov, ar, aperture, dist_to_focus, 0);
        default_cam->nodeName = "Default Camera";
        scene.addCamera(default_cam);
    }

    scene.initialized = true;
    SCENE_LOG_INFO("Created Professional Default Scene (Ground Plane + Nishita Sky)");
}

#endif

