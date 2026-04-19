#include "SceneCommand.h"
#include "scene_ui.h"
#include "Renderer.h"
#include "ProjectManager.h"
#include "globals.h"  // For SCENE_LOG_INFO
#include "Paint/PaintLayerStack.h"
#include "Paint/PaintTextureSet.h"
#include "Paint/MeshPaintAdapter.h"
#include "HittableInstance.h"
#include "Backend/OptixBackend.h"
#include "Backend/VulkanBackend.h"
#include "Backend/IViewportBackend.h"
#include <algorithm>
#include <SceneSelection.h>

extern bool g_optix_rebuild_pending;
extern bool g_vulkan_rebuild_pending;
extern bool g_viewport_raster_rebuild_pending;
extern bool g_bvh_rebuild_pending;
extern std::unique_ptr<Backend::IBackend> g_backend;
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

namespace {
bool isVulkanBackend(Backend::IBackend* backend) {
    if (!backend) return false;
    Backend::BackendType type = Backend::BackendType::CPU_EMBREE;
    try {
        type = backend->getInfo().type;
    } catch (...) {
        return false;
    }
    return type == Backend::BackendType::VULKAN_RT || type == Backend::BackendType::VULKAN_COMPUTE;
}

bool isOptixBackend(Backend::IBackend* backend) {
    if (!backend) return false;
    Backend::BackendType type = Backend::BackendType::CPU_EMBREE;
    try {
        type = backend->getInfo().type;
    } catch (...) {
        return false;
    }
    return type == Backend::BackendType::OPTIX;
}

bool sceneCommandNodeMatches(const std::string& candidate, const std::string& target) {
    if (candidate.empty() || target.empty()) return false;
    return candidate == target || candidate.rfind(target + "_mat_", 0) == 0;
}

Backend::IBackend* getActiveRenderBackend(UIContext& ctx) {
    if (g_backend) {
        return g_backend.get();
    }
    if (ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) == nullptr) {
        return ctx.backend_ptr;
    }
    return nullptr;
}

Backend::IViewportBackend* getActiveViewportBackend(UIContext& ctx) {
    if (g_viewport_backend) {
        return g_viewport_backend.get();
    }
    return dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
}

void resetAccumulationForSceneMutation(UIContext& ctx) {
    Backend::IBackend* renderBackend = getActiveRenderBackend(ctx);
    Backend::IViewportBackend* viewportBackend = getActiveViewportBackend(ctx);

    if (renderBackend) {
        renderBackend->resetAccumulation();
    }
    if (viewportBackend && viewportBackend != renderBackend) {
        viewportBackend->resetAccumulation();
    }
}

void setSceneObjectVisibility(UIContext& ctx, const std::string& object_name, bool visible) {
    if (object_name.empty()) return;
    if (ctx.scene.isEditorPendingDeleteObjectName(object_name)) {
        visible = false;
    }

    if (Backend::IViewportBackend* viewportBackend = getActiveViewportBackend(ctx)) {
        viewportBackend->setVisibilityByNodeName(object_name, visible);
    }

    if (Backend::IBackend* renderBackend = getActiveRenderBackend(ctx)) {
        if (renderBackend != getActiveViewportBackend(ctx)) {
            renderBackend->setVisibilityByNodeName(object_name, visible);
        }
    }
}

void setSceneNodeLocalVisibility(UIContext& ctx, const std::string& object_name, bool visible) {
    if (object_name.empty()) return;
    if (ctx.scene.isEditorPendingDeleteObjectName(object_name)) {
        visible = false;
    }

    for (auto& obj : ctx.scene.world.objects) {
        if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
            if (sceneCommandNodeMatches(tri->getNodeName(), object_name)) {
                tri->visible = visible;
            }
            continue;
        }
        if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) {
            if (sceneCommandNodeMatches(inst->node_name, object_name)) {
                inst->visible = visible;
            }
        }
    }
}

bool activeSceneGpuRenderBackend(UIContext& ctx) {
    if (Backend::IBackend* renderBackend = getActiveRenderBackend(ctx)) {
        return isVulkanBackend(renderBackend) || isOptixBackend(renderBackend);
    }
    return false;
}

void scheduleSceneMutationRebuilds(UIContext& ctx, bool includeCpuBvh) {
    // Invalidate cached raster geometry for both Solid/Matcap/Preview backends.
    // Some CPU-side mesh edit paths only set rebuild-pending flags; without a
    // generation bump, Vulkan raster buildRasterGeometry() may early-out and keep
    // showing stale meshes until a later backend-specific edit forces a refresh.
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);

    if (includeCpuBvh) {
        g_bvh_rebuild_pending = true;
    }

    if (getActiveViewportBackend(ctx)) {
        g_viewport_raster_rebuild_pending = true;
    }

    if (Backend::IBackend* renderBackend = getActiveRenderBackend(ctx)) {
        if (isVulkanBackend(renderBackend)) {
            g_vulkan_rebuild_pending = true;
        } else if (isOptixBackend(renderBackend)) {
            g_optix_rebuild_pending = true;
        }
    } else if (ctx.backend_ptr) {
        if (isVulkanBackend(ctx.backend_ptr)) {
            g_vulkan_rebuild_pending = true;
        } else if (isOptixBackend(ctx.backend_ptr)) {
            g_optix_rebuild_pending = true;
        }
    }

    resetAccumulationForSceneMutation(ctx);
}

std::string getHittableNodeName(const std::shared_ptr<Hittable>& obj) {
    if (!obj) return "";
    if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) return tri->getNodeName();
    if (auto inst = std::dynamic_pointer_cast<HittableInstance>(obj)) return inst->node_name;
    return "";
}

std::vector<std::shared_ptr<Triangle>> evaluateMeshForNode(UIContext& ctx, const std::string& object_name) {
    auto base_it = ctx.scene.base_mesh_cache.find(object_name);
    if (base_it == ctx.scene.base_mesh_cache.end()) {
        return {};
    }

    auto stack_it = ctx.scene.mesh_modifiers.find(object_name);
    if (stack_it != ctx.scene.mesh_modifiers.end() && !stack_it->second.modifiers.empty()) {
        return stack_it->second.evaluate(base_it->second);
    }
    return base_it->second;
}

void replaceSceneObjectsForNode(UIContext& ctx,
                                const std::string& object_name,
                                const std::vector<std::shared_ptr<Triangle>>& mesh) {
    auto& objects = ctx.scene.world.objects;
    objects.erase(
        std::remove_if(objects.begin(), objects.end(), [&](const auto& obj) {
            return getHittableNodeName(obj) == object_name;
        }),
        objects.end());

    for (const auto& tri : mesh) {
        if (tri) {
            objects.push_back(tri);
        }
    }

    if (ctx.selection.selected.type == SelectableType::Object && ctx.selection.selected.object) {
        if (getHittableNodeName(ctx.selection.selected.object) == object_name && !mesh.empty()) {
            ctx.selection.selected.object = mesh.front();
        }
    }
}

void scheduleUvGeometrySync(UIContext& ctx) {
    ctx.renderer.resetCPUAccumulation();
    scheduleSceneMutationRebuilds(ctx, false);

    ProjectManager::getInstance().markModified();
    ctx.start_render = true;
}
}

// ============================================================================
// DELETE COMMAND IMPLEMENTATION
// ============================================================================

void DeleteObjectCommand::execute(UIContext& ctx) {
    bool needs_update = false;

    for (const auto& tri : deleted_triangles_) {
        if (tri) {
            tri->visible = false;
            ctx.scene.markObjectPendingDelete(tri->nodeName);
            needs_update = true;
        }
    }
    setSceneNodeLocalVisibility(ctx, object_name_, false);
    needs_update = true;

    if (needs_update) {
        ctx.scene.markObjectPendingDelete(object_name_);
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
        ctx.renderer.resetCPUAccumulation();
        setSceneObjectVisibility(ctx, object_name_, false);
        if (activeSceneGpuRenderBackend(ctx)) {
            extern bool g_cpu_sync_pending;
            g_cpu_sync_pending = true;
        }
        
        ctx.start_render = true;
        ProjectManager::getInstance().markModified();
        SCENE_LOG_INFO("Redo: Deleted " + object_name_);
    }
}

void DeleteObjectCommand::undo(UIContext& ctx) {
    bool needs_update = false;
    for (const auto& tri : deleted_triangles_) {
        if (tri) {
            const bool existsInScene =
                std::find(ctx.scene.world.objects.begin(), ctx.scene.world.objects.end(), tri) != ctx.scene.world.objects.end();
            if (!existsInScene) {
                ctx.scene.world.objects.push_back(tri);
            }
            tri->visible = true;
            needs_update = true;
        }
    }
    ctx.scene.restoreObjectPendingDelete(object_name_);
    setSceneNodeLocalVisibility(ctx, object_name_, true);
    needs_update = true;

    // 2. Sync with ProjectManager (Persistence)
    // Remove from deleted_objects list so it reappears on save/load
    auto& proj = ProjectManager::getInstance().getProjectData();
    for (auto& model : proj.imported_models) {
        auto it = std::find(model.deleted_objects.begin(), model.deleted_objects.end(), object_name_);
        if (it != model.deleted_objects.end()) {
            model.deleted_objects.erase(it);
        }
    }

    if (needs_update) {
        ctx.renderer.resetCPUAccumulation();
        setSceneObjectVisibility(ctx, object_name_, true);
        if (activeSceneGpuRenderBackend(ctx)) {
            extern bool g_cpu_sync_pending;
            g_cpu_sync_pending = true;
        }
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
         scheduleSceneMutationRebuilds(ctx, true);
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
    
    scheduleSceneMutationRebuilds(ctx, true);
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
        if (ctx.backend_ptr && ctx.backend_ptr->isUsingTLAS()) {
            if (isVulkanBackend(ctx.backend_ptr)) {
                // CRITICAL INVARIANT (Vulkan Undo/Redo):
                // Use updateObjectTransform() here. Do not redirect transform-only undo/redo
                // through full scene rebuild paths. That reintroduces frozen viewport /
                // ghosting issues until backend switch.
                ctx.backend_ptr->updateObjectTransform(object_name_, state.matrix);
                ctx.backend_ptr->resetAccumulation();
                extern bool g_cpu_sync_pending;
                g_cpu_sync_pending = true;
            } else {
                // Convert Matrix4x4 to 3x4 row-major float array
                float t[12];
                t[0] = state.matrix.m[0][0]; t[1] = state.matrix.m[0][1]; t[2] = state.matrix.m[0][2]; t[3] = state.matrix.m[0][3];
                t[4] = state.matrix.m[1][0]; t[5] = state.matrix.m[1][1]; t[6] = state.matrix.m[1][2]; t[7] = state.matrix.m[1][3];
                t[8] = state.matrix.m[2][0]; t[9] = state.matrix.m[2][1]; t[10] = state.matrix.m[2][2]; t[11] = state.matrix.m[2][3];
                
                // Update all instances matching this object name
                std::vector<int> inst_ids = ctx.backend_ptr->getInstancesByNodeName(object_name_);
                for (int inst_id : inst_ids) {
                    ctx.backend_ptr->updateInstanceTransform(inst_id, t);
                }
                ctx.backend_ptr->rebuildAccelerationStructure();  // Fast TLAS update
                ctx.backend_ptr->resetAccumulation();
                // Mark CPU data as needing sync when switching to CPU mode
                extern bool g_cpu_sync_pending;
                g_cpu_sync_pending = true;
            }
        } else if (ctx.backend_ptr) {
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
    if (ctx.backend_ptr) {
        ctx.backend_ptr->setLights(ctx.scene.lights);
        ctx.backend_ptr->resetAccumulation();
    }
    SCENE_LOG_INFO("Redo: Transform Light");
    ctx.start_render = true; 
}

void TransformLightCommand::undo(UIContext& ctx) {
    old_state_.apply(*light_);
    if (ctx.backend_ptr) {
        ctx.backend_ptr->setLights(ctx.scene.lights);
        ctx.backend_ptr->resetAccumulation();
    }
    SCENE_LOG_INFO("Undo: Transform Light");
    ctx.start_render = true; 
}

void DeleteLightCommand::execute(UIContext& ctx) {
    auto& lights = ctx.scene.lights;
    auto it = std::remove(lights.begin(), lights.end(), light_);
    if (it != lights.end()) {
        lights.erase(it, lights.end());
        if (ctx.backend_ptr) {
             ctx.backend_ptr->setLights(ctx.scene.lights);
             ctx.backend_ptr->resetAccumulation();
        }
        ctx.start_render = true;
        SCENE_LOG_INFO("Redo: Delete Light");
    }
}

void DeleteLightCommand::undo(UIContext& ctx) {
    ctx.scene.lights.push_back(light_);
    if (ctx.backend_ptr) {
         ctx.backend_ptr->setLights(ctx.scene.lights);
         ctx.backend_ptr->resetAccumulation();
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
        if (ctx.backend_ptr) {
             ctx.backend_ptr->setLights(ctx.scene.lights);
             ctx.backend_ptr->resetAccumulation();
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
        if (ctx.backend_ptr) {
             ctx.backend_ptr->setLights(ctx.scene.lights);
             ctx.backend_ptr->resetAccumulation();
        }
        ctx.start_render = true;
    }
    SCENE_LOG_INFO("Undo: Removed Light");
}

void PaintTextureCommand::applyPixels(UIContext& ctx, const std::vector<CompactVec4>& pixels) {
    if (!texture_ || pixels.empty()) {
        return;
    }

    texture_->pixels = pixels;
    if (texture_->isUploaded()) {
        texture_->updateGPU();
    } else {
        texture_->upload_to_gpu();
    }

    ctx.renderer.resetCPUAccumulation();
    if (ctx.backend_ptr) {
        ctx.renderer.updateBackendMaterials(ctx.scene);
        ctx.backend_ptr->resetAccumulation();
    }
    ctx.start_render = true;
    ProjectManager::getInstance().markModified();
}

void PaintTextureCommand::execute(UIContext& ctx) {
    applyPixels(ctx, after_pixels_);
    SCENE_LOG_INFO("Redo: Paint " + object_name_);
}

void PaintTextureCommand::undo(UIContext& ctx) {
    applyPixels(ctx, before_pixels_);
    SCENE_LOG_INFO("Undo: Paint " + object_name_);
}

// ============================================================================
// PAINT LAYER COMMAND IMPLEMENTATION
// ============================================================================

void PaintLayerCommand::applyPixels(UIContext& ctx, const std::vector<CompactVec4>& pixels) {
    // Find layer stack
    auto it = ctx.scene.mesh_paint_layer_stacks.find(layer_stack_key_);
    if (it == ctx.scene.mesh_paint_layer_stacks.end()) return;

    Paint::PaintLayerStack& stack = it->second;
    Paint::PaintLayerData* layer = stack.layerById(layer_id_);
    if (!layer) return;

    // Restore layer pixel data
    const size_t ch_idx = static_cast<size_t>(channel_);
    layer->channel_pixels[ch_idx] = pixels;

    // Recomposite the affected channel into the flat texture set
    auto tex_it = ctx.scene.mesh_paint_texture_sets.find(layer_stack_key_);
    if (tex_it != ctx.scene.mesh_paint_texture_sets.end()) {
        Paint::PaintTextureSet& tex_set = tex_it->second;
        if (tex_set.initialized) {
            stack.flattenChannelInto(channel_, tex_set);
        }
    }

    ctx.renderer.resetCPUAccumulation();
    if (ctx.backend_ptr) {
        ctx.renderer.updateBackendMaterials(ctx.scene);
        ctx.backend_ptr->resetAccumulation();
    }
    ctx.start_render = true;
    ProjectManager::getInstance().markModified();
}

void PaintLayerCommand::execute(UIContext& ctx) {
    applyPixels(ctx, after_pixels_);
    SCENE_LOG_INFO("Redo: Paint Layer " + object_name_);
}

void PaintLayerCommand::undo(UIContext& ctx) {
    applyPixels(ctx, before_pixels_);
    SCENE_LOG_INFO("Undo: Paint Layer " + object_name_);
}

void UVProjectionCommand::applyStates(UIContext& ctx, const std::vector<TriangleUVSetState>& states) {
    for (const auto& state : states) {
        if (!state.triangle) {
            continue;
        }
        state.triangle->setUVSetCoordinates(state.uv_set_index, state.uvs[0], state.uvs[1], state.uvs[2]);
        state.triangle->applyUVSet(state.uv_set_index);
    }

    const auto evaluated_mesh = evaluateMeshForNode(ctx, object_name_);
    replaceSceneObjectsForNode(ctx, object_name_, evaluated_mesh);
    scheduleUvGeometrySync(ctx);
}

void UVProjectionCommand::execute(UIContext& ctx) {
    applyStates(ctx, after_states_);
    SCENE_LOG_INFO("Redo: Project UVs: " + object_name_);
}

void UVProjectionCommand::undo(UIContext& ctx) {
    applyStates(ctx, before_states_);
    SCENE_LOG_INFO("Undo: Project UVs: " + object_name_);
}

void MeshEditCommand::applyStates(UIContext& ctx, const std::vector<MeshEditTriangleState>& states) {
    bool changed = false;
    for (const auto& state : states) {
        if (!state.triangle) {
            continue;
        }

        for (int corner = 0; corner < 3; ++corner) {
            state.triangle->setOriginalVertexPosition(corner, state.positions[corner]);
        }
        state.triangle->markAABBDirty();
        state.triangle->updateTransformedVertices();
        changed = true;
    }

    if (!changed) {
        return;
    }

    scheduleSceneMutationRebuilds(ctx, true);

    ctx.renderer.resetCPUAccumulation();
    ProjectManager::getInstance().markModified();
    ctx.start_render = true;
}

void MeshEditCommand::execute(UIContext& ctx) {
    applyStates(ctx, after_states_);
    SCENE_LOG_INFO("Redo: Edit Mesh " + object_name_);
}

void MeshEditCommand::undo(UIContext& ctx) {
    applyStates(ctx, before_states_);
    SCENE_LOG_INFO("Undo: Edit Mesh " + object_name_);
}

void ReplaceMeshGeometryCommand::applyMesh(
    UIContext& ctx,
    const std::vector<std::shared_ptr<Triangle>>& display_mesh,
    const std::vector<std::shared_ptr<Triangle>>& base_mesh,
    const MeshModifiers::ModifierStack& stack) {
    replaceSceneObjectsForNode(ctx, object_name_, display_mesh);
    ctx.scene.base_mesh_cache[object_name_] = base_mesh;
    ctx.scene.mesh_modifiers[object_name_] = stack;

    if (Backend::IViewportBackend* viewportBackend = getActiveViewportBackend(ctx)) {
        if (!viewportBackend->updateRasterMeshFromTriangles(object_name_, display_mesh)) {
            viewportBackend->buildRasterGeometry(ctx.scene.world.objects);
        }
        viewportBackend->resetAccumulation();
    }

    scheduleSceneMutationRebuilds(ctx, true);
    ProjectManager::getInstance().markModified();
    ctx.start_render = true;
}

void ReplaceMeshGeometryCommand::execute(UIContext& ctx) {
    applyMesh(ctx, after_display_mesh_, after_base_mesh_, after_stack_);
    SCENE_LOG_INFO("Redo: Replace Mesh Geometry " + object_name_);
}

void ReplaceMeshGeometryCommand::undo(UIContext& ctx) {
    applyMesh(ctx, before_display_mesh_, before_base_mesh_, before_stack_);
    SCENE_LOG_INFO("Undo: Replace Mesh Geometry " + object_name_);
}

void CompositeSceneCommand::execute(UIContext& ctx) {
    for (auto& command : commands_) {
        if (command) {
            command->execute(ctx);
        }
    }
    SCENE_LOG_INFO("Redo: " + description_);
}

void CompositeSceneCommand::undo(UIContext& ctx) {
    for (auto it = commands_.rbegin(); it != commands_.rend(); ++it) {
        if (*it) {
            (*it)->undo(ctx);
        }
    }
    SCENE_LOG_INFO("Undo: " + description_);
}
