#include "SceneCommand.h"
#include "scene_ui.h"
#include "Renderer.h"
#include "ProjectManager.h"
#include "globals.h"  // For SCENE_LOG_INFO
#include <algorithm>

// ============================================================================
// DELETE COMMAND IMPLEMENTATION
// ============================================================================

void DeleteObjectCommand::execute(UIContext& ctx) {
    // Redo: Remove objects again
    auto& objs = ctx.scene.world.objects;
    bool needs_update = false;
    
    // 1. Remove from Scene
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
    
    // 2. Sync with ProjectManager (Persistence)
    // Find which model this object belongs to and add to its deleted_objects list
    auto& proj = ProjectManager::getInstance().getProjectData();
    for (auto& model : proj.imported_models) {
        // Check if the deleted object is part of this model
        // We assume object_name_ is unique enough or we check if the name matches a known object in the model
        // Since we are deleting by name/group, we add the name to the list.
        bool belongs_to_model = false;
        for(const auto& inst : model.objects) {
            if(inst.node_name == object_name_) {
                belongs_to_model = true;
                break;
            }
        }
        
        if (belongs_to_model) {
            // Add to allow-list (deleted names) if not present
            if (std::find(model.deleted_objects.begin(), model.deleted_objects.end(), object_name_) 
                == model.deleted_objects.end()) {
                model.deleted_objects.push_back(object_name_);
            }
        }
    }
    
    if (needs_update) {
        // Defer BVH rebuild to Main.cpp async handler
        extern bool g_bvh_rebuild_pending;
        g_bvh_rebuild_pending = true;
        ctx.renderer.resetCPUAccumulation();
        
        // Incremental GPU update for TLAS mode
        if (ctx.optix_gpu_ptr && ctx.optix_gpu_ptr->isUsingTLAS()) {
            ctx.optix_gpu_ptr->hideInstancesByNodeName(object_name_);
            ctx.optix_gpu_ptr->rebuildTLAS();
        } else if (ctx.optix_gpu_ptr) {
            extern bool g_optix_rebuild_pending;
            g_optix_rebuild_pending = true;
        }
        
        ctx.start_render = true;
        ProjectManager::getInstance().markModified();
        SCENE_LOG_INFO("Redo: Deleted " + object_name_);
    }
}

void DeleteObjectCommand::undo(UIContext& ctx) {
    // 1. Restore objects to Scene
    ctx.scene.world.objects.insert(
        ctx.scene.world.objects.end(),
        deleted_triangles_.begin(),
        deleted_triangles_.end()
    );
    
    // 2. Sync with ProjectManager (Persistence)
    // Remove from deleted_objects list so it reappears on save/load
    auto& proj = ProjectManager::getInstance().getProjectData();
    for (auto& model : proj.imported_models) {
        auto it = std::find(model.deleted_objects.begin(), model.deleted_objects.end(), object_name_);
        if (it != model.deleted_objects.end()) {
            model.deleted_objects.erase(it);
        }
    }
    
    // Defer BVH rebuild to Main.cpp async handler
    extern bool g_bvh_rebuild_pending;
    g_bvh_rebuild_pending = true;
    ctx.renderer.resetCPUAccumulation();
    
    // Incremental GPU update for TLAS mode
    // Note: For undo of delete, we need FULL rebuild since we're adding triangles back
    // The cloneInstancesByNodeName won't work here because BLAS is gone
    if (ctx.optix_gpu_ptr) {
        extern bool g_optix_rebuild_pending;
        g_optix_rebuild_pending = true;
    }
    
    ProjectManager::getInstance().markModified();
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
         // Defer full rebuild to Main.cpp async handler
         extern bool g_bvh_rebuild_pending;
         g_bvh_rebuild_pending = true;
         extern bool g_optix_rebuild_pending;
         g_optix_rebuild_pending = true;
         ctx.renderer.resetCPUAccumulation();
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
    
    // Defer full rebuild to Main.cpp async handler
    extern bool g_bvh_rebuild_pending;
    g_bvh_rebuild_pending = true;
    extern bool g_optix_rebuild_pending;
    g_optix_rebuild_pending = true;
    ctx.renderer.resetCPUAccumulation();
    
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
        // TLAS MODE: Fast instance transform update (no BLAS rebuild!)
        if (ctx.optix_gpu_ptr && ctx.optix_gpu_ptr->isUsingTLAS()) {
            // Convert Matrix4x4 to 3x4 row-major float array
            float t[12];
            t[0] = state.matrix.m[0][0]; t[1] = state.matrix.m[0][1]; t[2] = state.matrix.m[0][2]; t[3] = state.matrix.m[0][3];
            t[4] = state.matrix.m[1][0]; t[5] = state.matrix.m[1][1]; t[6] = state.matrix.m[1][2]; t[7] = state.matrix.m[1][3];
            t[8] = state.matrix.m[2][0]; t[9] = state.matrix.m[2][1]; t[10] = state.matrix.m[2][2]; t[11] = state.matrix.m[2][3];
            
            // Update all instances matching this object name
            std::vector<int> inst_ids = ctx.optix_gpu_ptr->getInstancesByNodeName(object_name_);
            for (int inst_id : inst_ids) {
                ctx.optix_gpu_ptr->updateInstanceTransform(inst_id, t);
            }
            ctx.optix_gpu_ptr->rebuildTLAS();  // Fast TLAS update
            ctx.optix_gpu_ptr->resetAccumulation();
            // Mark CPU data as needing sync when switching to CPU mode
            extern bool g_cpu_sync_pending;
            g_cpu_sync_pending = true;
        } else if (ctx.optix_gpu_ptr) {
            // GAS MODE: Defer to async handler
            extern bool g_gpu_refit_pending;
            g_gpu_refit_pending = true;
        }
        
        // Defer CPU BVH rebuild to async handler (prevents UI freeze)
        extern bool g_bvh_rebuild_pending;
        g_bvh_rebuild_pending = true;
        ctx.renderer.resetCPUAccumulation();
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
