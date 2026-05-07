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
#include <cmath>
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
             Transform* t = tri->getTransformPtr();
             if (t) {
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

namespace {

bool inferSquareDimensions(size_t pixel_count, int& width, int& height) {
    if (pixel_count == 0) {
        return false;
    }
    const int side = static_cast<int>(std::sqrt(static_cast<double>(pixel_count)) + 0.5);
    if (side > 0 && static_cast<size_t>(side) * static_cast<size_t>(side) == pixel_count) {
        width = side;
        height = side;
        return true;
    }
    return false;
}

bool findChangedRegion(const std::vector<CompactVec4>& before,
                       const std::vector<CompactVec4>& after,
                       int width,
                       int height,
                       int& x,
                       int& y,
                       int& w,
                       int& h) {
    if (width <= 0 || height <= 0 ||
        before.size() != after.size() ||
        before.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
        return false;
    }

    int min_x = width;
    int min_y = height;
    int max_x = -1;
    int max_y = -1;
    for (int py = 0; py < height; ++py) {
        const size_t row = static_cast<size_t>(py) * static_cast<size_t>(width);
        for (int px = 0; px < width; ++px) {
            const size_t index = row + static_cast<size_t>(px);
            const CompactVec4& a = before[index];
            const CompactVec4& b = after[index];
            if (a.r == b.r && a.g == b.g && a.b == b.b && a.a == b.a) {
                continue;
            }
            min_x = std::min(min_x, px);
            min_y = std::min(min_y, py);
            max_x = std::max(max_x, px);
            max_y = std::max(max_y, py);
        }
    }

    if (max_x < min_x || max_y < min_y) {
        return false;
    }
    x = min_x;
    y = min_y;
    w = max_x - min_x + 1;
    h = max_y - min_y + 1;
    return true;
}

bool findNonTransparentRegion(const std::vector<CompactVec4>& pixels,
                              int width,
                              int height,
                              int& x,
                              int& y,
                              int& w,
                              int& h) {
    if (width <= 0 || height <= 0 ||
        pixels.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
        return false;
    }

    int min_x = width;
    int min_y = height;
    int max_x = -1;
    int max_y = -1;
    for (int py = 0; py < height; ++py) {
        const size_t row = static_cast<size_t>(py) * static_cast<size_t>(width);
        for (int px = 0; px < width; ++px) {
            const CompactVec4& p = pixels[row + static_cast<size_t>(px)];
            if (p.a == 0) {
                continue;
            }
            min_x = std::min(min_x, px);
            min_y = std::min(min_y, py);
            max_x = std::max(max_x, px);
            max_y = std::max(max_y, py);
        }
    }

    if (max_x < min_x || max_y < min_y) {
        return false;
    }
    x = min_x;
    y = min_y;
    w = max_x - min_x + 1;
    h = max_y - min_y + 1;
    return true;
}

std::vector<CompactVec4> extractPixelRegion(const std::vector<CompactVec4>& src,
                                            int width,
                                            int height,
                                            int x,
                                            int y,
                                            int w,
                                            int h) {
    std::vector<CompactVec4> out;
    if (src.empty() || width <= 0 || height <= 0 || w <= 0 || h <= 0) {
        return out;
    }
    out.resize(static_cast<size_t>(w) * static_cast<size_t>(h));
    for (int row = 0; row < h; ++row) {
        const size_t src_offset = static_cast<size_t>(y + row) * static_cast<size_t>(width) + static_cast<size_t>(x);
        const size_t dst_offset = static_cast<size_t>(row) * static_cast<size_t>(w);
        std::copy_n(src.begin() + static_cast<std::ptrdiff_t>(src_offset), w, out.begin() + static_cast<std::ptrdiff_t>(dst_offset));
    }
    return out;
}

void applyPixelRegion(std::vector<CompactVec4>& dst,
                      const std::vector<CompactVec4>& region,
                      int width,
                      int height,
                      int x,
                      int y,
                      int w,
                      int h) {
    if (width <= 0 || height <= 0 || w <= 0 || h <= 0 || region.empty()) {
        return;
    }
    if (dst.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
        dst.assign(static_cast<size_t>(width) * static_cast<size_t>(height), CompactVec4(0, 0, 0, 0));
    }
    if (region.size() != static_cast<size_t>(w) * static_cast<size_t>(h)) {
        return;
    }
    for (int row = 0; row < h; ++row) {
        const size_t dst_offset = static_cast<size_t>(y + row) * static_cast<size_t>(width) + static_cast<size_t>(x);
        const size_t src_offset = static_cast<size_t>(row) * static_cast<size_t>(w);
        std::copy_n(region.begin() + static_cast<std::ptrdiff_t>(src_offset), w, dst.begin() + static_cast<std::ptrdiff_t>(dst_offset));
    }
}

} // namespace

PaintTextureCommand::PaintTextureCommand(const std::string& object_name,
                                         uint16_t material_id,
                                         const std::shared_ptr<Texture>& texture,
                                         std::vector<CompactVec4> before_pixels,
                                         std::vector<CompactVec4> after_pixels)
    : object_name_(object_name)
    , material_id_(material_id)
    , texture_(texture) {
    width_ = texture_ ? texture_->width : 0;
    height_ = texture_ ? texture_->height : 0;
    if (width_ <= 0 || height_ <= 0) {
        inferSquareDimensions(std::max(before_pixels.size(), after_pixels.size()), width_, height_);
    }

    int x = 0, y = 0, w = 0, h = 0;
    if (findChangedRegion(before_pixels, after_pixels, width_, height_, x, y, w, h)) {
        region_mode_ = true;
        region_x_ = x;
        region_y_ = y;
        region_w_ = w;
        region_h_ = h;
        before_pixels_ = extractPixelRegion(before_pixels, width_, height_, x, y, w, h);
        after_pixels_ = extractPixelRegion(after_pixels, width_, height_, x, y, w, h);
    } else {
        before_pixels_ = std::move(before_pixels);
        after_pixels_ = std::move(after_pixels);
    }
}

void PaintTextureCommand::applyPixels(UIContext& ctx, const std::vector<CompactVec4>& pixels) {
    if (!texture_ || pixels.empty()) {
        return;
    }

    if (region_mode_) {
        applyPixelRegion(texture_->pixels, pixels, width_, height_, region_x_, region_y_, region_w_, region_h_);
        if (texture_->isUploaded()) {
            texture_->updateGPURegion(region_x_, region_y_, region_w_, region_h_);
        } else {
            texture_->markVulkanDirtyRegion(region_x_, region_y_, region_w_, region_h_);
            texture_->upload_to_gpu();
        }
    } else {
        texture_->pixels = pixels;
        texture_->markVulkanDirtyFull();
        if (texture_->isUploaded()) {
            texture_->updateGPU();
        } else {
            texture_->upload_to_gpu();
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

PaintLayerCommand::PaintLayerCommand(const std::string& object_name,
                                     uint16_t material_id,
                                     const std::string& layer_stack_key,
                                     uint32_t layer_id,
                                     Paint::PaintChannel channel,
                                     std::vector<CompactVec4> before_pixels,
                                     std::vector<CompactVec4> after_pixels)
    : object_name_(object_name)
    , material_id_(material_id)
    , layer_stack_key_(layer_stack_key)
    , layer_id_(layer_id)
    , channel_(channel)
    , before_empty_(before_pixels.empty())
    , after_empty_(after_pixels.empty()) {
    inferSquareDimensions(std::max(before_pixels.size(), after_pixels.size()), width_, height_);

    int x = 0, y = 0, w = 0, h = 0;
    const bool can_region =
        (!before_empty_ && !after_empty_ &&
         findChangedRegion(before_pixels, after_pixels, width_, height_, x, y, w, h)) ||
        (before_empty_ && !after_empty_ &&
         findNonTransparentRegion(after_pixels, width_, height_, x, y, w, h));
    if (can_region) {
        region_mode_ = true;
        region_x_ = x;
        region_y_ = y;
        region_w_ = w;
        region_h_ = h;
        before_pixels_ = before_empty_
            ? std::vector<CompactVec4>{}
            : extractPixelRegion(before_pixels, width_, height_, x, y, w, h);
        after_pixels_ = extractPixelRegion(after_pixels, width_, height_, x, y, w, h);
    } else {
        before_pixels_ = std::move(before_pixels);
        after_pixels_ = std::move(after_pixels);
    }
}

void PaintLayerCommand::applyPixels(UIContext& ctx, const std::vector<CompactVec4>& pixels, bool empty_state) {
    // Find layer stack
    auto it = ctx.scene.mesh_paint_layer_stacks.find(layer_stack_key_);
    if (it == ctx.scene.mesh_paint_layer_stacks.end()) return;

    Paint::PaintLayerStack& stack = it->second;
    Paint::PaintLayerData* layer = stack.layerById(layer_id_);
    if (!layer) return;

    // Restore layer pixel data
    const size_t ch_idx = static_cast<size_t>(channel_);
    if (empty_state) {
        layer->channel_pixels[ch_idx].clear();
    } else if (region_mode_) {
        applyPixelRegion(layer->channel_pixels[ch_idx], pixels, width_, height_, region_x_, region_y_, region_w_, region_h_);
    } else {
        layer->channel_pixels[ch_idx] = pixels;
    }

    // Recomposite the affected channel into the flat texture set
    auto tex_it = ctx.scene.mesh_paint_texture_sets.find(layer_stack_key_);
    if (tex_it != ctx.scene.mesh_paint_texture_sets.end()) {
        Paint::PaintTextureSet& tex_set = tex_it->second;
        if (tex_set.initialized) {
            if (region_mode_ && !empty_state) {
                Paint::PaintDirtyRect dirty;
                dirty.expand(region_x_, region_y_, region_x_ + region_w_ - 1, region_y_ + region_h_ - 1);
                stack.flattenChannelRegionInto(channel_, tex_set, dirty);
            } else {
                stack.flattenChannelInto(channel_, tex_set);
            }
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
    applyPixels(ctx, after_pixels_, after_empty_);
    SCENE_LOG_INFO("Redo: Paint Layer " + object_name_);
}

void PaintLayerCommand::undo(UIContext& ctx) {
    applyPixels(ctx, before_pixels_, before_empty_);
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
