#include "SceneCommand.h"
#include "scene_ui.h"
#include "Renderer.h"
#include "globals.h"  // For SCENE_LOG_INFO
#include <algorithm>

// ============================================================================
// DELETE COMMAND IMPLEMENTATION
// ============================================================================

void DeleteObjectCommand::execute(UIContext& ctx) {
    // Redo: Remove objects again
    auto& objs = ctx.scene.world.objects;
    bool needs_update = false;
    
    // Efficiently remove all triangles in the deleted list
    // Note: We need to match by pointer or name. Pointers are preserved in deleted_triangles_
    // but the scene might have new pointers if we are not careful? 
    // Actually, Undo puts the SAME shared_ptrs back. So equality check works.
    
    auto new_end = std::remove_if(objs.begin(), objs.end(), [&](const std::shared_ptr<Hittable>& h){
        auto t = std::dynamic_pointer_cast<Triangle>(h);
        if (!t) return false;
        // Check if this triangle is in our deleted list
        for (auto& del : deleted_triangles_) {
            if (del == t) return true;
        }
        return false;
    });
    
    if (new_end != objs.end()) {
        objs.erase(new_end, objs.end());
        needs_update = true;
    }
    
    if (needs_update) {
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        ctx.renderer.resetCPUAccumulation();
        if (ctx.optix_gpu_ptr) ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
        
        // Rebuild UI cache since objects are gone
        // We can't easily call rebuildMeshCache since it is static/member of SceneUI. 
        // But ctx has no access to SceneUI methods directly unless we pass SceneUI... 
        // Wait, SceneUI::draw calls this. But SceneUI isn't passed here.
        // We usually rely on the check `if (ctx.scene.world.objects.size() != last_obj_count)` in drawSceneHierarchy
        // to rebuild the cache. So we just need to ensure size changes.
        
        ctx.start_render = true;
        SCENE_LOG_INFO("Redo: Deleted " + object_name_);
    }
}

void DeleteObjectCommand::undo(UIContext& ctx) {
    // Restore deleted objects
    ctx.scene.world.objects.insert(
        ctx.scene.world.objects.end(),
        deleted_triangles_.begin(),
        deleted_triangles_.end()
    );
    
    // Rebuild GPU structures
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    
    if (ctx.optix_gpu_ptr) {
        ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
    }
    
    SCENE_LOG_INFO("Undo: Restored " + object_name_);
    ctx.start_render = true;
}

// ============================================================================
// DUPLICATE COMMAND IMPLEMENTATION
// ============================================================================

void DuplicateObjectCommand::execute(UIContext& ctx) {
    // Redo: Add duplicates back
    bool needs_update = false;
    
    // Check if they are already there?
    // If we rely on shared_ptrs, we can just push them back.
    // But we should key by name if possible?
    
    // For Redo, we just re-insert the specific pointers we saved.
    // Check if first one is in scene.
    if (!new_triangles_.empty()) {
        bool exists = false;
        for (const auto& obj : ctx.scene.world.objects) {
            if (obj == new_triangles_[0]) { exists = true; break; }
        }
        
        if (!exists) {
            ctx.scene.world.objects.insert(ctx.scene.world.objects.end(), new_triangles_.begin(), new_triangles_.end());
            needs_update = true;
        }
    }
    
    if (needs_update) {
         ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
         ctx.renderer.resetCPUAccumulation();
         if (ctx.optix_gpu_ptr) ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
         ctx.start_render = true;
         SCENE_LOG_INFO("Redo: Duplicated " + source_name_ + " -> " + new_name_);
    }
}

void DuplicateObjectCommand::undo(UIContext& ctx) {
    // Remove duplicated objects
    auto& objs = ctx.scene.world.objects;
    auto new_end = std::remove_if(objs.begin(), objs.end(), 
        [this](const std::shared_ptr<Hittable>& h) {
             // Match by pointer for robustness
             for(auto& t : new_triangles_) if (std::dynamic_pointer_cast<Triangle>(h) == t) return true;
             return false;
        });
    
    if (new_end != objs.end()) {
        objs.erase(new_end, objs.end());
    }
    
    // Rebuild GPU structures
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    
    if (ctx.optix_gpu_ptr) {
        ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
    }
    
    SCENE_LOG_INFO("Undo: Removed duplicate " + new_name_);
    ctx.start_render = true;
}

// ============================================================================
// TRANSFORM COMMAND IMPLEMENTATION
// ============================================================================

void TransformCommand::applyState(UIContext& ctx, const TransformState& state) {
    // Find objects by name
    bool found = false;
    for (auto& obj : ctx.scene.world.objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri && tri->nodeName == object_name_) {
             if (auto t = tri->getTransformHandle()) {
                 t->setBase(state.matrix);
                 tri->updateTransformedVertices(); // Important
             }
            found = true;
        }
    }
    
    if (found) {
        // Rebuild GPU structures
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        ctx.renderer.resetCPUAccumulation();
        
        if (ctx.optix_gpu_ptr) {
            ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects); 
            ctx.optix_gpu_ptr->resetAccumulation();
        }
    }
}

void TransformCommand::execute(UIContext& ctx) {
    applyState(ctx, new_state_);
    SCENE_LOG_INFO("Redo: Transform " + object_name_);
    ctx.start_render = true; 
}

void TransformCommand::undo(UIContext& ctx) {
    applyState(ctx, old_state_);
    SCENE_LOG_INFO("Undo: Transform " + object_name_);
    ctx.start_render = true; 
}

// ============================================================================
// LIGHT COMMANDS IMPLEMENTATION
// ============================================================================

void TransformLightCommand::execute(UIContext& ctx) {
    new_state_.apply(*light_);
    if (ctx.optix_gpu_ptr) {
        ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
        ctx.optix_gpu_ptr->resetAccumulation();
    }
    SCENE_LOG_INFO("Redo: Transform Light");
    ctx.start_render = true; 
}

void TransformLightCommand::undo(UIContext& ctx) {
    old_state_.apply(*light_);
    if (ctx.optix_gpu_ptr) {
        ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
        ctx.optix_gpu_ptr->resetAccumulation();
    }
    SCENE_LOG_INFO("Undo: Transform Light");
    ctx.start_render = true; 
}

void DeleteLightCommand::execute(UIContext& ctx) {
    auto& lights = ctx.scene.lights;
    auto it = std::remove(lights.begin(), lights.end(), light_);
    if (it != lights.end()) {
        lights.erase(it, lights.end());
        if (ctx.optix_gpu_ptr) {
             ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
             ctx.optix_gpu_ptr->resetAccumulation();
        }
        ctx.start_render = true;
        SCENE_LOG_INFO("Redo: Delete Light");
    }
}

void DeleteLightCommand::undo(UIContext& ctx) {
    ctx.scene.lights.push_back(light_);
    if (ctx.optix_gpu_ptr) {
         ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
         ctx.optix_gpu_ptr->resetAccumulation();
    }
    SCENE_LOG_INFO("Undo: Restored Light");
    ctx.start_render = true;
}

void AddLightCommand::execute(UIContext& ctx) {
    // Check if already exists
    bool exists = false;
    for(auto& l : ctx.scene.lights) if(l == light_) exists = true;
    
    if (!exists) {
        ctx.scene.lights.push_back(light_);
        if (ctx.optix_gpu_ptr) {
             ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
             ctx.optix_gpu_ptr->resetAccumulation();
        }
        ctx.start_render = true;
        SCENE_LOG_INFO("Redo: Add Light");
    }
}

void AddLightCommand::undo(UIContext& ctx) {
    auto& lights = ctx.scene.lights;
    auto it = std::remove(lights.begin(), lights.end(), light_);
    if (it != lights.end()) {
        lights.erase(it, lights.end());
        if (ctx.optix_gpu_ptr) {
             ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
             ctx.optix_gpu_ptr->resetAccumulation();
        }
        ctx.start_render = true;
    }
    SCENE_LOG_INFO("Undo: Removed Light");
}
