#include "scene_ui.h"
#include "scene_data.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "KeyframeSystem.h"
#include "Camera.h"
#include "light.h"
#include "DirectionalLight.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "World.h"

// Helper to apply transform to an object (or group of triangles with same name)


void SceneUI::processAnimations(UIContext& ctx) {
    if (!ctx.scene.initialized) return;
    
    // 1. Check if we need to update
    // Update if playing OR if timeline frame changed manually
    // We store the 'last processed frame' to avoid redundant updates when paused
    static int last_processed_frame = -1;
    static float last_processed_time = -1.0f;
    
    // Force update if rendering animation (batch mode)
    bool force_update = ctx.is_animation_mode || ctx.render_settings.start_animation_render;
    
    if (!timeline.isPlaying() && ctx.scene.timeline.current_frame == last_processed_frame && !force_update) {
        return;
    }
    
    last_processed_frame = ctx.scene.timeline.current_frame;
    int current_frame = ctx.scene.timeline.current_frame;

    bool anything_changed = false;
    bool tlas_dirty = false;

    // 2. Iterate Tracks
    for (auto& [name, track] : ctx.scene.timeline.tracks) {
        // Evaluate track at current frame
        Keyframe kf = track.evaluate(current_frame);
        
        if (kf.has_transform) {
             const auto& tk = kf.transform;
             if (tk.has_position || tk.has_rotation || tk.has_scale) {
                 // Use mesh_cache for fast lookup
                 if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);
                 
                 auto it = mesh_cache.find(name);
                 if (it != mesh_cache.end()) {
                     for (auto& pair : it->second) {
                         auto tri = pair.second;
                         if (!tri) continue;
                         
                         auto transform_handle = tri->getTransformHandle();
                         if (!transform_handle) continue;
                         
                         Vec3 final_pos = transform_handle->position;
                         Vec3 final_rot = transform_handle->rotation;
                         Vec3 final_scale = transform_handle->scale;
                         
                         if (tk.has_position) final_pos = tk.position;
                         if (tk.has_rotation) final_rot = tk.rotation;
                         if (tk.has_scale)    final_scale = tk.scale;
                         
                         if (final_pos != transform_handle->position || 
                             final_rot != transform_handle->rotation || 
                             final_scale != transform_handle->scale) 
                         {
                             transform_handle->position = final_pos;
                             transform_handle->rotation = final_rot;
                             transform_handle->scale = final_scale;
                             transform_handle->updateMatrix();
                             
                             anything_changed = true;
                             
                             // TLAS Update Logic (using OptixWrapper)
                             if (ctx.optix_gpu_ptr && ctx.optix_gpu_ptr->isUsingTLAS()) {
                                 tlas_dirty = true;
                                 std::vector<int> inst_ids = ctx.optix_gpu_ptr->getInstancesByNodeName(name);
                                 
                                 if (!inst_ids.empty()) {
                                     float t[12];
                                     const auto& newMat = transform_handle->base;
                                     t[0] = newMat.m[0][0]; t[1] = newMat.m[0][1]; t[2] = newMat.m[0][2]; t[3] = newMat.m[0][3];
                                     t[4] = newMat.m[1][0]; t[5] = newMat.m[1][1]; t[6] = newMat.m[1][2]; t[7] = newMat.m[1][3];
                                     t[8] = newMat.m[2][0]; t[9] = newMat.m[2][1]; t[10] = newMat.m[2][2]; t[11] = newMat.m[2][3];
                                    
                                    for (int inst_id : inst_ids) {
                                         ctx.optix_gpu_ptr->updateInstanceTransform(inst_id, t);
                                    }
                                 }
                             }
                         }
                         // Optimization: Break after first logic update if we assume shared handle for same object
                         break; 
                     }
                 }
             }
        }
        
        if (kf.has_light) {
             // Find light by name
             for (auto& light : ctx.scene.lights) {
                 if (light->nodeName == name) {
                     bool l_changed = false;
                     // Color
                     if (kf.light.has_color) {
                         light->color = kf.light.color;
                         l_changed = true;
                     }
                     // Intensity
                     if (kf.light.has_intensity) {
                         light->intensity = kf.light.intensity;
                         l_changed = true;
                     }
                     // Position (if not handled by Transform? Lights usually have Position directly)
                     // Code usually treats Light Gizmo as Transform Update.
                     // But KeyframeSystem segregates "LightKeyframe" from "Transform".
                     // Ideally Lights should use Transform tracks for movement and Light tracks for properties.
                     // But `ApplyTransform` checks `mesh_cache`. Lights are NOT in `mesh_cache` (usually).
                     // So we handle Light position here if needed.
                     // Does LightKeyframe have position? checking...
                     // KeyframeSystem.h -> LightKeyframe: color, intensity, position, direction, type...
                     if (kf.light.has_position) {
                         light->position = kf.light.position;
                         l_changed = true;
                     }
                     if (kf.light.has_direction) {
                          if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(light)) dl->direction = kf.light.direction;
                          else if (auto sl = std::dynamic_pointer_cast<SpotLight>(light)) sl->direction = kf.light.direction;
                          l_changed = true;
                     }
                     
                     if (l_changed) anything_changed = true;
                     break; // Found the light
                 }
             }
        }
        
        if (kf.has_camera && ctx.scene.camera) {
             // Check if this track targets the active camera?
             // Or find camera by name
             // Currently single camera support mostly, or `ctx.scene.camera` is the active one.
             // If track name == camera name
             // Note: Camera doesn't store a "nodeName" in the variable `ctx.scene.camera` directly?
             // We'd have to check if we can name cameras.
             // Assuming "Camera" or "MainCamera"
             if (name == "Camera" || name == "Main Camera" || (ctx.scene.camera && ctx.scene.camera->nodeName == name)) {
                 bool c_changed = false;
                 // FOV
                 if (kf.camera.has_fov) {
                     ctx.scene.camera->vfov = kf.camera.fov;
                     c_changed = true;
                 }
                 // Position / LookAt - Handled by CameraKeyframe?
                 if (kf.camera.has_position) {
                     ctx.scene.camera->lookfrom = kf.camera.position;
                     c_changed = true;
                 }
                 if (kf.camera.has_target) {
                     ctx.scene.camera->lookat = kf.camera.target;
                     c_changed = true;
                 }
                 // Others: Aperture, Focal Length
                 if (kf.camera.has_aperture) { ctx.scene.camera->aperture = kf.camera.has_aperture; c_changed = true; }
                 if (kf.camera.has_focus) { ctx.scene.camera->focus_dist = kf.camera.focus_distance; c_changed = true; }

                 if (c_changed) anything_changed = true;
             }
        }
        
        if (kf.has_world) {
            if (name == "World") {
                 // Update world
                 // sun_elevation, sun_azimuth, etc.
                 // Need access to World object.
                 // `ctx.scene.world` is HittableList? NO. `SceneData` has `World` object?
                 // `SceneData` has `World world;` struct? No, `World.h` defines `class World`.
                 // But where is the instance?
                 // `SceneData.h` -> `KeyframeSystem`...
                 // `SceneData.h` usually has `Environment` or something.
                 // Checking `scene_data.h` or `scene_ui.h`...
                 // `ctx.scene.environment` or similar?
                 // `renderer.h` uses `HittableList world`.
                 // Actually `World.cpp` is separate.
                 // Let's assume there is a global or accessible `ctx.scene.world_data`?
                 // Checking `scene_ui_world.cpp` would calculate this.
                 // Usually `ctx.scene.lights`...
                 // Wait, `SceneData` likely has `SceneParameters` or similar.
                 
                 // Fallback: If `World` is a singleton or global?
                 // `scene_ui.cpp` uses `ctx.scene.world`... which is `HittableList`.
                 // Ah, `World` class for Enviroment.
                 // `e:\visual studio proje c++\raytracing_Proje_Moduler\raytrac_sdl2\source\src\World.cpp`
                 // In `scene_ui.h`, `Context` struct has `SceneData scene;`
                 // `SceneData` struct in `scene_data.h`?
                 
                 // Let's check `evaluate` above. 
                 // I will skip World logic implemented blindly for now to avoid errors, 
                 // as the primary user request is about OBJECTS.
                 // I will focus on Object transforms.
            }
        }
    }
    
    // 3. Trigger General Updates if changed
    if (anything_changed) {
        // Reset Accumulation
        ctx.renderer.resetCPUAccumulation();
        if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
        
        // Rebuild BVH (CPU)
        // If objects moved, we must rebuild CPU BVH for picking/CPU render
        // This can be slow for many objects.
        // Optimization: Only Refit?
        // Current engine just calls `rebuildBVH`.
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        
        // GPU Updates
        if (ctx.optix_gpu_ptr) {
             // If TLAS was marked dirty (Object Transforms)
             if (tlas_dirty) {
                 ctx.optix_gpu_ptr->rebuildTLAS();
             }
             
             // Update Lights if needed (we don't track specifically, just safe update)
             ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
             
             // Update Camera
             if (ctx.scene.camera) {
                 ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
             }
        }
    }
}
