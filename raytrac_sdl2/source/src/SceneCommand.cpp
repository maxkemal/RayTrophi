#include "SceneCommand.h"
#include "scene_ui.h"
#include "Renderer.h"
#include "globals.h"  // For SCENE_LOG_INFO
#include <algorithm>

// ============================================================================
// DELETE COMMAND IMPLEMENTATION
// ============================================================================

void DeleteObjectCommand::execute(UIContext& ctx) {
    // Remove objects from scene (already done by caller, this is no-op for execute)
    // This is called when creating the command, not when executing
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
}

// ============================================================================
// DUPLICATE COMMAND IMPLEMENTATION
// ============================================================================

void DuplicateObjectCommand::execute(UIContext& ctx) {
    // Add duplicated objects (already done by caller, this is no-op for execute)
}

void DuplicateObjectCommand::undo(UIContext& ctx) {
    // Remove duplicated objects
    auto& objs = ctx.scene.world.objects;
    auto new_end = std::remove_if(objs.begin(), objs.end(), 
        [this](const std::shared_ptr<Hittable>& h) {
            auto tri = std::dynamic_pointer_cast<Triangle>(h);
            return tri && tri->nodeName == new_name_;
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
}

// ============================================================================
// TRANSFORM COMMAND IMPLEMENTATION
// ============================================================================

void TransformCommand::applyState(UIContext& ctx, const TransformState& state) {
    // Find objects by name
    bool found = false;
    // Iterate through all objects to find all parts of the group
    for (auto& obj : ctx.scene.world.objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri && tri->nodeName == object_name_) {
            // Apply transform
            tri->setBaseTransform(state.matrix);
            tri->updateTransformedVertices();
            found = true;
        }
    }
    
    if (found) {
        // Rebuild GPU structures
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        ctx.renderer.resetCPUAccumulation();
        
        if (ctx.optix_gpu_ptr) {
            // For pure transforms (no topology change), updateGeometry needs objects list
            ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects); 
            ctx.optix_gpu_ptr->resetAccumulation();
        }
    }
}

void TransformCommand::execute(UIContext& ctx) {
    applyState(ctx, new_state_);
    SCENE_LOG_INFO("Redo: Transform " + object_name_);
    ctx.start_render = true; // Trigger restart
}

void TransformCommand::undo(UIContext& ctx) {
    applyState(ctx, old_state_);
    SCENE_LOG_INFO("Undo: Transform " + object_name_);
    ctx.start_render = true; // Trigger restart
}

