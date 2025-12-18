
// ============================================================================
// TRANSFORM COMMAND IMPLEMENTATION
// ============================================================================

void TransformCommand::applyState(UIContext& ctx, const TransformState& state) {
    // Find objects by name
    bool found = false;
    for (auto& obj : ctx.scene.world.objects) {
        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
        if (tri && tri->nodeName == object_name_) {
            // Apply transform
            tri->setBase(state.position, state.rotation, state.scale);
            tri->updateTransformedVertices();
            found = true;
        }
    }
    
    if (found) {
        // Rebuild GPU structures
        // Optimization: For transform, we can use updateGeometry() if topology didn't change
        // But rebuildOptiXGeometry is safer and handles everything
        
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        // Note: resetCPUAccumulation and start_render setting is handled by caller or SceneUI usually,
        // but here we must ensure render restarts
        ctx.renderer.resetCPUAccumulation();
        
        if (ctx.optix_gpu_ptr) {
            // Use updateGeometry for transforms - it's faster!
            ctx.optix_gpu_ptr->updateGeometry(); 
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
