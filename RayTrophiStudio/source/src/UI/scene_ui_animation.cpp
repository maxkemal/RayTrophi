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

    // [RENDER-LOCK RACE FIX — root cause of "render hangs sometimes when
    // timeline is not at frame 0"] During an active sequence render the
    // worker thread (Renderer::render_Animation) is the sole owner of
    // animation state. It writes scene.timeline.current_frame each frame,
    // re-evaluates ALL keyframes, mutates Transform handles / lights /
    // camera, and submits backend AS rebuilds.
    //
    // This function used to use `force_update = ctx.is_animation_mode`
    // which is ALSO true during sequence render, so it ran every UI frame,
    // re-read the worker-written scene.timeline.current_frame, re-applied
    // every keyframe on the main thread, AND called
    // ctx.backend_ptr->rebuildAccelerationStructure() / setLights /
    // syncCamera against the same backend the worker was actively tracing
    // through. Destroying an AS while the GPU is mid-traceRays is
    // undefined behavior on Vulkan — the typical NVIDIA symptom is a
    // silent driver hang (no AV, no exception). The race window only
    // closes on frames where the AS rebuild happens to land between trace
    // submissions, which is why the render sometimes finishes and
    // sometimes locks up. OptiX is masked from this by CUDA stream
    // serialization.
    //
    // Skip entirely while the render thread owns the scene; the worker
    // handles all keyframe application internally and already keeps the
    // backend up to date.
    //
    // [GUARD WINDOW FIX] Check ui_ctx flags BEFORE animation_render_locked
    // — Main.cpp sets rendering_in_progress + is_animation_mode immediately
    // before std::thread().detach(), but animation_render_locked is set
    // INSIDE render_Animation on the worker thread, which can be tens of
    // milliseconds later (detached threads have no scheduler guarantee).
    // In that window the main thread may run processAnimations and race
    // the worker — the symptom is the FIRST FEW rendered frames using the
    // pre-render scrub pose because the main thread's keyframe apply +
    // updateInstanceTransforms runs concurrently with the worker's TLAS
    // refit.
    if (ctx.is_animation_mode && rendering_in_progress.load()) {
        return;
    }

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
                         
                         Transform* transform_handle = tri->getTransformPtr();
                         if (!transform_handle) continue;

                         Vec3 final_pos = transform_handle->position;
                         Vec3 final_rot = transform_handle->rotation;
                         Vec3 final_scale = transform_handle->scale;
                         
                         // ROOT MOTION CHECK: If this object is an animated character using root motion,
                         // we skip EVALUATING the position track so the animator can drive it.
                         bool skip_eval_pos = false;
                         for (const auto& mctx : ctx.scene.importedModelContexts) {
                             // Match either the model name exactly (root node) or a prefixed node name
                             bool nameMatch = (mctx.importName == name) || (name.find(mctx.importName + "_") == 0);
                             if (nameMatch && mctx.useRootMotion) {
                                 skip_eval_pos = true;
                                 break;
                             }
                         }

                         if (tk.has_position && !skip_eval_pos) final_pos = tk.position;
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
                             
                             // TLAS Update Logic.
                             //
                             // Just flag it dirty — the authoritative per-frame sync is
                             // updateInstanceTransforms(world.objects) below, which rebuilds every
                             // instance transform from its source handle (Triangle / HittableInstance
                             // / flat TriangleMesh) and commits the TLAS.
                             //
                             // [FLAT KEYFRAME TLAS-STALE FIX] We used to ALSO pre-poke each instance
                             // here via updateInstanceTransform(transform_handle->base). On Vulkan that
                             // wrote the new matrix straight into m_vkInstances *before*
                             // updateInstanceTransforms ran — and that function's commit gate is
                             // `updated != m_vkInstances`. Pre-poking made `updated` (recomputed from
                             // getFinal()) equal the already-mutated m_vkInstances, so the gate saw NO
                             // change and skipped commitTLAS: the BLAS was at the right pose but the TLAS
                             // instance was never refit. For a non-parented object base == getFinal(),
                             // so a plain keyframed flat mesh froze at frame 0 in Vulkan RT (parented
                             // rigs slipped through because base != getFinal()). Leaving m_vkInstances
                             // untouched lets the gate detect the real change and commit.
                             if (ctx.backend_ptr && ctx.backend_ptr->isUsingTLAS()) {
                                 tlas_dirty = true;
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
        } // End Camera block

        if (kf.has_emitter) {
            // Find emitter by track name
            // Track name format: "VolumeName_EmitterName_Index"
            for (auto& gas : ctx.scene.gas_volumes) {
                auto& emitters = gas->getSimulator().getEmitters();
                for (int i = 0; i < (int)emitters.size(); ++i) {
                    auto& e = emitters[i];
                    std::string expected_name = gas->getName() + "_" + e.name + "_" + std::to_string(e.uid);
                    if (expected_name == name) {
                        e.applyKeyframe(kf.emitter);
                        anything_changed = true;
                        break;
                    }
                }
            }
        }
    } // End Track Loop
    
    // 3. Trigger General Updates if changed
    if (anything_changed) {
        // Reset Accumulation
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
        
        // CPU picking/CPU-render BVH refit. A keyframe is a transform-only change (no topology),
        // so a full rebuild is never needed — refitBVH does an in-place Embree refit (facade verts
        // via the dirty pre-pass, flat/direct-SoA meshes self-baked from getFinal()*P_orig).
        //
        // [TIMELINE-PLAY COST] But the CPU Embree BVH is consumed per-frame ONLY when the CPU
        // reference renderer is what's drawing the viewport. During a GPU RT (OptiX / Vulkan) or
        // Solid raster session it's needed solely for picking — which tries GPU pick first and, on
        // the CPU fallback, independently syncs verts + brings a stale BVH current before querying.
        // So refitting EVERY keyframe there is pure overhead, heaviest on a dense FLAT mesh that
        // self-bakes all its verts each frame. Defer it (mark stale); the pick fallback / a backend
        // switch to CPU promotes it on demand. When CPU IS the viewport we must refit now — it's
        // what's on screen. The deferred refit stays correct: the flat self-bake is xform-gated (not
        // dirty-gated) and the facade BVH-buffer update is last_xform/vertexPositionsDirty-gated, and
        // the click-time pick sync writes the facade world verts regardless.
        extern bool g_solid_viewport_active;
        extern bool g_cpu_bvh_stale;
        if (ctx.backend_ptr != nullptr || g_solid_viewport_active) {
            g_cpu_bvh_stale = true;
        } else {
            ctx.renderer.refitBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        }
        
        // GPU Updates
        if (ctx.backend_ptr) {
             // If TLAS was marked dirty (Object Transforms)
             //
             // [VULKAN GEOMETRY-LOSS FIX] Don't call rebuildAccelerationStructure
             // here — on the Vulkan adapter it destroys all BLAS / TLAS /
             // m_vkInstances and DOES NOT rebuild (rebuild is deferred to
             // Main.cpp's pending block, which only fires when
             // g_vulkan_rebuild_pending is set). The result is a viewport
             // with zero geometry until something else triggers a full
             // rebuild. OptiX's same-named call does the rebuild inline,
             // masking the bug.
             //
             // Keyframe transform changes only need a TLAS refit — the
             // BLASes themselves are unchanged. updateInstanceTransforms
             // rebuilds m_vkInstances from scene.world.objects (current
             // transforms), waitIdles, then refits the TLAS in place.
             if (tlas_dirty) {
                 ctx.backend_ptr->updateInstanceTransforms(ctx.scene.world.objects);
             }

             // Update Lights if needed (we don't track specifically, just safe update)
             ctx.backend_ptr->setLights(ctx.scene.lights);

             // Update Camera
             if (ctx.scene.camera) {
                 ctx.backend_ptr->syncCamera(*ctx.scene.camera);
             }
        }
    }
}
