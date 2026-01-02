// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - SELECTION & INTERACTION
// ═══════════════════════════════════════════════════════════════════════════════
// This file handles Mouse picking, Marquee selection, and Delete operations.
// ═══════════════════════════════════════════════════════════════════════════════

#include "scene_ui.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "globals.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "scene_data.h"
#include <unordered_set>
#include <ProjectManager.h>
#include "WaterSystem.h"
#include "TerrainManager.h"



void SceneUI::handleMarqueeSelection(UIContext& ctx) {
    ImGuiIO& io = ImGui::GetIO();

    // Only handle when not interacting with UI windows and not using gizmo
    // WantCaptureMouse is true when mouse is over an interactive UI element (button, slider, etc.)
    // This is less restrictive than IsAnyItemHovered which blocks even when hovering inactive areas
    if (io.WantCaptureMouse || ImGuizmo::IsOver() || ImGuizmo::IsUsing()) {
        return;
    }

    // Start marquee on right mouse button down (or B key + left click for Blender style)
    bool start_marquee = ImGui::IsMouseClicked(ImGuiMouseButton_Right) && !io.KeyCtrl && !io.KeyShift;

    if (start_marquee && !is_marquee_selecting) {
        is_marquee_selecting = true;
        marquee_start = io.MousePos;
        marquee_end = io.MousePos;
    }

    // Update marquee while dragging
    if (is_marquee_selecting && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
        marquee_end = io.MousePos;
    }

    // Complete marquee on mouse release
    if (is_marquee_selecting && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
        is_marquee_selecting = false;

        // Normalize rectangle
        float x1 = fminf(marquee_start.x, marquee_end.x);
        float y1 = fminf(marquee_start.y, marquee_end.y);
        float x2 = fmaxf(marquee_start.x, marquee_end.x);
        float y2 = fmaxf(marquee_start.y, marquee_end.y);

        // Minimum size to prevent accidental selections
        if ((x2 - x1) < 10 || (y2 - y1) < 10) {
            return;
        }

        // Clear current selection if Ctrl is not held
        if (!io.KeyCtrl) {
            ctx.selection.clearSelection();
        }

        if (!ctx.scene.camera) return;

        Camera& cam = *ctx.scene.camera;
        float screen_w = io.DisplaySize.x;
        float screen_h = io.DisplaySize.y;

        // Camera basis vectors for projection
        Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
        Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
        Vec3 cam_up = cam_right.cross(cam_forward).normalize();
        float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
        float tan_half_fov = tanf(fov_rad * 0.5f);

        // Lambda to project 3D point to screen
        auto ProjectToScreen = [&](const Vec3& p) -> ImVec2 {
            Vec3 to_point = p - cam.lookfrom;
            float depth = to_point.dot(cam_forward);
            if (depth <= 0.01f) return ImVec2(-10000, -10000);

            float local_x = to_point.dot(cam_right);
            float local_y = to_point.dot(cam_up);

            float half_height = depth * tan_half_fov;
            float half_width = half_height * aspect_ratio;

            float ndc_x = local_x / half_width;
            float ndc_y = local_y / half_height;

            return ImVec2(
                (ndc_x * 0.5f + 0.5f) * screen_w,
                (0.5f - ndc_y * 0.5f) * screen_h
            );
            };

        // Check which objects are inside the marquee
        if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

        int skipped_procedural = 0;

        for (auto& [name, triangles] : mesh_cache) {
            if (triangles.empty()) continue;

            // IMPORTANT: Check if all triangles share the same TransformHandle
            // Procedural objects may have separate transforms per triangle
            auto firstHandle = triangles[0].second->getTransformHandle();
            bool all_same_transform = true;

            for (size_t i = 1; i < triangles.size() && all_same_transform; ++i) {
                auto handle = triangles[i].second->getTransformHandle();
                if (handle.get() != firstHandle.get()) {
                    all_same_transform = false;
                }
            }

            if (!all_same_transform) {
                // This object has mixed transforms - skip it
                // (would break if selected because transform would only affect some triangles)
                skipped_procedural++;
                continue;
            }

            // Calculate bounding box center for quick check
            Vec3 bb_min(1e10f, 1e10f, 1e10f);
            Vec3 bb_max(-1e10f, -1e10f, -1e10f);

            for (auto& pair : triangles) {
                auto& tri = pair.second;
                Vec3 v0 = tri->getV0();
                Vec3 v1 = tri->getV1();
                Vec3 v2 = tri->getV2();

                bb_min.x = fminf(bb_min.x, fminf(v0.x, fminf(v1.x, v2.x)));
                bb_min.y = fminf(bb_min.y, fminf(v0.y, fminf(v1.y, v2.y)));
                bb_min.z = fminf(bb_min.z, fminf(v0.z, fminf(v1.z, v2.z)));
                bb_max.x = fmaxf(bb_max.x, fmaxf(v0.x, fmaxf(v1.x, v2.x)));
                bb_max.y = fmaxf(bb_max.y, fmaxf(v0.y, fmaxf(v1.y, v2.y)));
                bb_max.z = fmaxf(bb_max.z, fmaxf(v0.z, fmaxf(v1.z, v2.z)));
            }

            Vec3 center = (bb_min + bb_max) * 0.5f;
            ImVec2 screenPos = ProjectToScreen(center);

            // Check if center is inside marquee
            if (screenPos.x >= x1 && screenPos.x <= x2 && screenPos.y >= y1 && screenPos.y <= y2) {
                SelectableItem item;
                item.type = SelectableType::Object;
                item.object = triangles[0].second;
                item.object_index = triangles[0].first;
                item.name = name;

                if (!ctx.selection.isSelected(item)) {
                    ctx.selection.addToSelection(item);
                }
            }
        }

        if (skipped_procedural > 0) {
            SCENE_LOG_WARN("Skipped " + std::to_string(skipped_procedural) + " objects with mixed transforms (use Ctrl+Click)");
            addViewportMessage("Skipped " + std::to_string(skipped_procedural) + " objects (Mixed Transforms)", 3.0f, ImVec4(1.0f, 0.8f, 0.2f, 1.0f));
        }

        // Also check lights
        for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
            auto& light = ctx.scene.lights[i];
            if (!light) continue;

            ImVec2 screenPos = ProjectToScreen(light->position);

            if (screenPos.x >= x1 && screenPos.x <= x2 && screenPos.y >= y1 && screenPos.y <= y2) {
                SelectableItem item;
                item.type = SelectableType::Light;
                item.light = light;
                item.light_index = (int)i;
                item.name = "Light_" + std::to_string(i);

                if (!ctx.selection.isSelected(item)) {
                    ctx.selection.addToSelection(item);
                }
            }
        }

        if (ctx.selection.multi_selection.size() > 0) {
            // [VERBOSE] SCENE_LOG_INFO("Marquee selected " + std::to_string(ctx.selection.multi_selection.size()) + " items");
        }
    }

    // Draw the marquee rectangle while selecting
    drawMarqueeRect();
}
bool SceneUI::deleteSelectedObject(UIContext& ctx)
{
    std::string deleted_name = ctx.selection.selected.name;
    if (deleted_name.empty()) return false;

    auto& objects = ctx.scene.world.objects;
    size_t removed_count = 0;

    objects.erase(
        std::remove_if(objects.begin(), objects.end(),
            [&deleted_name, &removed_count](const std::shared_ptr<Hittable>& obj)
            {
                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                if (tri && tri->nodeName == deleted_name) {
                    removed_count++;
                    return true;
                }
                return false;
            }),
        objects.end()
    );

    if (removed_count == 0) return false;

    // --- Project data bookkeeping ---
    auto& proj = g_ProjectManager.getProjectData();

    for (auto& model : proj.imported_models) {
        for (const auto& inst : model.objects) {
            if (inst.node_name == deleted_name) {
                if (std::find(model.deleted_objects.begin(),
                    model.deleted_objects.end(),
                    deleted_name) == model.deleted_objects.end()) {
                    model.deleted_objects.push_back(deleted_name);
                }
                break;
            }
        }
    }

    // Remove procedural objects
    auto& procs = proj.procedural_objects;
    procs.erase(
        std::remove_if(procs.begin(), procs.end(),
            [&deleted_name](const ProceduralObjectData& p) {
                return p.display_name == deleted_name;
            }),
        procs.end()
    );

    g_ProjectManager.markModified();

    // ═══════════════════════════════════════════════════════════════════════════
    // DEFERRED REBUILD - Set flags instead of immediate rebuild for faster UI
    // ═══════════════════════════════════════════════════════════════════════════
    g_mesh_cache_dirty = true;
    g_bvh_rebuild_pending = true;
    g_optix_rebuild_pending = true;
    
    // Auto-cleanup timeline tracks for deleted object
    auto it = ctx.scene.timeline.tracks.find(deleted_name);
    if (it != ctx.scene.timeline.tracks.end()) {
        ctx.scene.timeline.tracks.erase(it);
    }

    SCENE_LOG_INFO(
        "Deleted: " + deleted_name + " (" +
        std::to_string(removed_count) + " triangles)"
    );

    return true;
}

void SceneUI::handleDeleteShortcut(UIContext& ctx)
{
    if (!ImGui::IsKeyPressed(ImGuiKey_Delete) &&
        !ImGui::IsKeyPressed(ImGuiKey_X)) return;

    bool deleted = false;

    if (ctx.selection.selected.type == SelectableType::Light) {
        deleted = deleteSelectedLight(ctx);
    }
    else if (ctx.selection.selected.type == SelectableType::Object) {
        deleted = deleteSelectedObject(ctx);
    }

    if (deleted) {
        ctx.selection.clearSelection();
    }
}

void SceneUI::handleMouseSelection(UIContext& ctx) {
    // Only select if not interacting with UI or Gizmo
    if (ImGui::IsMouseClicked(0)) {

        // Ignore click if over UI elements (Window/Panel), Gizmo, or HUD overlay
        // WantCaptureMouse is true if mouse is over any ImGui window or interacting with it
        // hud_captured_mouse is true if HUD elements (Focus Ring, Zoom, Exposure) captured the click
        if (ImGui::GetIO().WantCaptureMouse || ImGuizmo::IsOver() || hud_captured_mouse) {
            // Reset HUD flag for next frame
            hud_captured_mouse = false;
            return;
        }

        // Check if Ctrl is held for multi-selection
        bool ctrl_held = ImGui::GetIO().KeyCtrl;

        int x, y;
        SDL_GetMouseState(&x, &y);

        float win_w = ImGui::GetIO().DisplaySize.x;
        float win_h = ImGui::GetIO().DisplaySize.y;

        float u = (float)x / win_w;
        float v = (float)y / win_h;
        v = 1.0f - v;

        if (ctx.scene.camera) {

            Ray r = ctx.scene.camera->get_ray(u, v);

            // Check for Light Selection first (Bounding Sphere Intersection)
            std::shared_ptr<Light> closest_light = nullptr;
            float closest_t = 1e9f;
            int closest_light_index = -1;

            for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
                auto& light = ctx.scene.lights[i];
                if (!light) continue;

                // Proxy Sphere at light position - smaller radius for precise selection
                Vec3 oc = r.origin - light->position;
                float radius = 0.2f;  // Reduced from 0.5 to not block nearby objects
                float a = r.direction.dot(r.direction);
                float half_b = oc.dot(r.direction);
                float c = oc.dot(oc) - radius * radius;
                float discriminant = half_b * half_b - a * c;

                if (discriminant > 0) {
                    float root = sqrt(discriminant);
                    float temp = (-half_b - root) / a;
                    if (temp < closest_t && temp > 0.001f) {
                        closest_t = temp;
                        closest_light = light;
                        closest_light_index = (int)i;
                    }
                    temp = (-half_b + root) / a;
                    if (temp < closest_t && temp > 0.001f) {
                        closest_t = temp;
                        closest_light = light;
                        closest_light_index = (int)i;
                    }
                }
            }

            // Check for Camera Selection (non-active cameras only)
            std::shared_ptr<Camera> closest_camera = nullptr;
            float closest_camera_t = closest_t;  // Must be closer than light

            for (size_t i = 0; i < ctx.scene.cameras.size(); ++i) {
                // Camera selection sphere
                auto& cam = ctx.scene.cameras[i]; // HATALI cam kullanımı için düzeltme
                if (!cam) continue;
                Vec3 oc = r.origin - cam->lookfrom;
                float radius = 0.6f;  // Increased from 0.4 to make selection easier
                float a = r.direction.dot(r.direction);
                float half_b = oc.dot(r.direction);
                float c = oc.dot(oc) - radius * radius;
                float discriminant = half_b * half_b - a * c;

                if (discriminant > 0) {
                    float root = sqrt(discriminant);
                    float temp = (-half_b - root) / a;
                    if (temp < closest_camera_t && temp > 0.001f) {
                        closest_camera_t = temp;
                        closest_camera = cam;
                    }
                    temp = (-half_b + root) / a;
                    if (temp < closest_camera_t && temp > 0.001f) {
                        closest_camera_t = temp;
                        closest_camera = cam;
                    }
                }
            }

            // Perform Linear Selection (Bypasses BVH for accuracy and avoids rebuilds)

            HitRecord rec;
            bool hit = false;
            float closest_so_far = 1e9f;
            HitRecord temp_rec;

            for (const auto& obj : ctx.scene.world.objects) {
                if (obj->hit(r, 0.001f, closest_so_far, temp_rec)) {
                    hit = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            // ═══════════════════════════════════════════════════════════
            // APPLY PICK FOCUS (Using Found Hits)
            // ═══════════════════════════════════════════════════════════
            if (is_picking_focus) {
                float min_dist = 1e9f;
                bool found_hit = false;

                // Check Object Hit
                if (hit && rec.t < min_dist) {
                    min_dist = rec.t;
                    found_hit = true;
                }
                // Check Light Hit
                if (closest_light && closest_t < min_dist) {
                    min_dist = closest_t;
                    found_hit = true;
                }
                // Check Camera Hit
                if (closest_camera && closest_camera_t < min_dist) {
                    min_dist = closest_camera_t;
                    found_hit = true;
                }

                if (found_hit) {
                    ctx.scene.camera->focus_dist = min_dist;
                    ctx.scene.camera->update_camera_vectors();
                    if (ctx.optix_gpu_ptr) {
                        ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                    SCENE_LOG_INFO(std::string("Pick Focus set to: ") + std::to_string(min_dist) + "m");
                }

                is_picking_focus = false;
                ctx.start_render = true;
                return;
            }

            if (hit && rec.triangle && (rec.t < closest_t)) {
                std::shared_ptr<Triangle> found_tri = nullptr;
                int index = -1;

                // Ensure cache is valid
                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

                bool found = false;
                for (auto& [name, list] : mesh_cache) {
                    for (auto& pair : list) {
                        if (pair.second.get() == rec.triangle) {
                            found_tri = pair.second;
                            index = pair.first;
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }

                if (found_tri) {
                    if (ctrl_held) {
                        // Multi-selection: Toggle object in selection list
                        SelectableItem item;
                        item.type = SelectableType::Object;
                        item.object = found_tri;
                        item.object_index = index;
                        item.name = found_tri->nodeName;

                        if (ctx.selection.isSelected(item)) {
                            ctx.selection.removeFromSelection(item);
                            // SCENE_LOG_INFO("Multi-select: Removed '" + item.name + "' (Total: " + std::to_string(ctx.selection.multi_selection.size()) + ")");
                        }
                        else {
                            ctx.selection.addToSelection(item);
                            // SCENE_LOG_INFO("Multi-select: Added '" + item.name + "' (Total: " + std::to_string(ctx.selection.multi_selection.size()) + ")");
                        }
                    }
                    else {
                        // Single selection: Replace selection
                        ctx.selection.selectObject(found_tri, index, found_tri->nodeName);
                        
                        // TERRAIN CONNECTION: Check if this is a terrain chunk
                        std::string tName = found_tri->nodeName;
                        if (tName.find("Terrain_") == 0) {
                            size_t chunkPos = tName.find("_Chunk");
                            if (chunkPos != std::string::npos) {
                                tName = tName.substr(0, chunkPos);
                            }
                            auto terrain = TerrainManager::getInstance().getTerrainByName(tName);
                            if (terrain) {
                                terrain_brush.active_terrain_id = terrain->id;
                                show_terrain_tab = true;
                                tab_to_focus = "Terrain";
                                SCENE_LOG_INFO("Terrain selected via viewport: " + tName);
                            }
                        }
                    }
                }
                else {
                    SCENE_LOG_WARN("Selection: Object found but not in cache.");
                }
            }
            else if (closest_camera && closest_camera_t < closest_t) {
                // Camera is closer than light
                if (ctrl_held) {
                    SelectableItem item;
                    item.type = SelectableType::Camera;
                    item.camera = closest_camera;
                    item.name = "Camera";

                    if (ctx.selection.isSelected(item)) {
                        ctx.selection.removeFromSelection(item);
                    }
                    else {
                        ctx.selection.addToSelection(item);
                    }
                }
                else {
                    ctx.selection.selectCamera(closest_camera);
                }
            }
            else if (closest_light) {
                if (ctrl_held) {
                    SelectableItem item;
                    item.type = SelectableType::Light;
                    item.light = closest_light;
                    item.light_index = closest_light_index;
                    item.name = "Light";

                    if (ctx.selection.isSelected(item)) {
                        ctx.selection.removeFromSelection(item);
                    }
                    else {
                        ctx.selection.addToSelection(item);
                    }
                }
                else {
                    ctx.selection.selectLight(closest_light);
                }
            }
            else {
                // Clicked on empty space - clear selection only if Ctrl is not held
                if (!ctrl_held) {
                    ctx.selection.clearSelection();
                }
            }
        }
    }
}

// ============================================================================
// Delete Operation (Shared by Menu and Key Shortcut)
// ============================================================================
// OPTIMIZED VERSION - O(n) instead of O(n²)
void SceneUI::triggerDelete(UIContext& ctx) {
    if (!ctx.selection.hasSelection()) return;

    // Collect all items to delete (supports multi-selection)
    std::vector<SelectableItem> items_to_delete = ctx.selection.multi_selection;

    // Build mesh cache once if needed
    if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

    // OPTIMIZATION: Collect ALL hittable pointers to delete into a single set for O(1) lookup
    // Using Hittable* (base class) for type safety
    std::unordered_set<Hittable*> objects_to_delete;
    std::vector<std::string> deleted_names;
    std::vector<std::pair<std::string, std::vector<std::shared_ptr<Triangle>>>> undo_data;

    // First pass: Collect all triangles to delete
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::Object && item.object) {
            std::string deleted_name = item.name;

            auto cache_it = mesh_cache.find(deleted_name);
            if (cache_it != mesh_cache.end()) {
                std::vector<std::shared_ptr<Triangle>> tris_for_undo;
                for (auto& pair : cache_it->second) {
                    objects_to_delete.insert(pair.second.get());
                    tris_for_undo.push_back(pair.second);
                }

                if (!tris_for_undo.empty()) {
                    deleted_names.push_back(deleted_name);
                    undo_data.push_back({ deleted_name, std::move(tris_for_undo) });
                }
            }
        }
    }

    // OPTIMIZATION: Single remove_if pass for ALL objects - O(n) instead of O(n²)
    // CRITICAL: Using raw pointer .get() instead of dynamic_pointer_cast for massive speedup
    // dynamic_pointer_cast does RTTI check on every call = very slow on 4M objects
    // We already know exact pointers from mesh_cache, so just compare raw pointers
    if (!objects_to_delete.empty()) {
        auto& objs = ctx.scene.world.objects;
        objs.erase(
            std::remove_if(objs.begin(), objs.end(), [&](const std::shared_ptr<Hittable>& h) {
                // Fast O(1) lookup - no RTTI, just pointer comparison with base class
                return objects_to_delete.count(h.get()) > 0;
                }),
            objs.end()
        );
    }

    // Track deletions in ProjectManager (batch update)
    auto& proj_data = g_ProjectManager.getProjectData();
    for (const auto& deleted_name : deleted_names) {
        // Check Imported Models
        bool found = false;
        for (auto& model : proj_data.imported_models) {
            std::string prefix = std::to_string(model.id) + "_";
            if (deleted_name.find(prefix) == 0) {
                model.deleted_objects.push_back(deleted_name);
                found = true;
                break;
            }
            for (const auto& obj_inst : model.objects) {
                if (obj_inst.node_name == deleted_name) {
                    model.deleted_objects.push_back(deleted_name);
                    found = true;
                    break;
                }
            }
            if (found) break;
        }

        // Check Procedural Objects
        if (!found) {
            auto it = std::remove_if(proj_data.procedural_objects.begin(), proj_data.procedural_objects.end(),
                [&](const ProceduralObjectData& p) { return p.display_name == deleted_name; });
            proj_data.procedural_objects.erase(it, proj_data.procedural_objects.end());
        }
        
        // Check Water Surfaces - remove from WaterManager if name matches
        auto& water_surfaces = WaterManager::getInstance().getWaterSurfaces();
        for (auto& surf : water_surfaces) {
            if (surf.name == deleted_name) {
                // Don't call removeWaterSurface here - triangles already removed above
                // Just mark for removal from water_surfaces vector
                surf.id = -1; // Mark for removal
                break;
            }
        }
    }
    
    // Remove marked water surfaces
    auto& water_surfaces = WaterManager::getInstance().getWaterSurfaces();
    water_surfaces.erase(
        std::remove_if(water_surfaces.begin(), water_surfaces.end(),
            [](const WaterSurface& s) { return s.id == -1; }),
        water_surfaces.end()
    );

    // Record undo commands
    for (auto& [name, tris] : undo_data) {
        history.record(std::make_unique<DeleteObjectCommand>(name, tris));
    }

    // Handle light deletions
    int deleted_lights = 0;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::Light && item.light) {
            auto& lights = ctx.scene.lights;
            auto it = std::find(lights.begin(), lights.end(), item.light);
            if (it != lights.end()) {
                history.record(std::make_unique<DeleteLightCommand>(item.light));
                lights.erase(it);
                deleted_lights++;
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Handle camera deletions
    // ═══════════════════════════════════════════════════════════════════════════
    int deleted_cameras = 0;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::Camera && item.camera) {
            // Safety checks are in removeCamera method
            if (ctx.scene.cameras.size() <= 1) {
                SCENE_LOG_WARN("Cannot delete the last camera in scene");
                continue;
            }

            // Check if this is the active camera
            if (item.camera == ctx.scene.camera) {
                SCENE_LOG_WARN("Cannot delete the active camera. Switch to another camera first.");
                continue;
            }

            // Remove camera
            if (ctx.scene.removeCamera(item.camera)) {
                deleted_cameras++;
                SCENE_LOG_INFO("Camera deleted successfully");
            }
        }
    }

    // Only rebuild once after all deletions are done
    int deleted_objects = static_cast<int>(deleted_names.size());
    if (deleted_objects > 0 || deleted_lights > 0 || deleted_cameras > 0) {
        ctx.selection.clearSelection();
        g_ProjectManager.markModified();

        // ═══════════════════════════════════════════════════════════════════════════
        // OPTIMIZED UPDATE - Manually remove from cache to avoid slow rebuild
        // ═══════════════════════════════════════════════════════════════════════════
        for (const auto& name : deleted_names) {
            // Remove from fast lookup cache
            mesh_cache.erase(name);
            
            // Remove from UI iteration cache
            auto it = std::remove_if(mesh_ui_cache.begin(), mesh_ui_cache.end(), 
                [&](const auto& pair){ return pair.first == name; });
            mesh_ui_cache.erase(it, mesh_ui_cache.end());
        }

        g_mesh_cache_dirty = true;  // Flag for Main.cpp / other systems
        mesh_cache_valid = true;    // KEEP TRUE: Our local cache is now up-to-date!
        
        if (deleted_objects > 0) {
            // ═══════════════════════════════════════════════════════════════════════
            // INCREMENTAL GPU UPDATE (TLAS mode) - Instant!
            // ═══════════════════════════════════════════════════════════════════════
            if (ctx.optix_gpu_ptr && ctx.optix_gpu_ptr->isUsingTLAS()) {
                // Fast path: Just hide instances by setting visibility_mask = 0
                for (const auto& name : deleted_names) {
                    ctx.optix_gpu_ptr->hideInstancesByNodeName(name);
                }
                // Single TLAS update after ALL hides complete (batched for efficiency)
                ctx.optix_gpu_ptr->rebuildTLAS();
            } else {
                // GAS mode fallback: Full rebuild required
                g_optix_rebuild_pending = true;
            }
            
            // CPU BVH still needs rebuild for picking (async)
            g_bvh_rebuild_pending = true;
        }

        // Update lights if any were deleted
        if (deleted_lights > 0 && ctx.optix_gpu_ptr) {
            ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
            ctx.optix_gpu_ptr->resetAccumulation();
        }

        // Update GPU camera if cameras changed
        if (deleted_cameras > 0 && ctx.optix_gpu_ptr && ctx.scene.camera) {
            ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
            ctx.optix_gpu_ptr->resetAccumulation();
        }
        
        // ═══════════════════════════════════════════════════════════════════════════
        // AUTO-CLEANUP TIMELINE TRACKS FOR DELETED ENTITIES
        // ═══════════════════════════════════════════════════════════════════════════
        // Remove timeline tracks for deleted objects/lights/cameras
        // This prevents orphan keyframes from cluttering the timeline
        for (const auto& deleted_name : deleted_names) {
            auto it = ctx.scene.timeline.tracks.find(deleted_name);
            if (it != ctx.scene.timeline.tracks.end()) {
                ctx.scene.timeline.tracks.erase(it);
            }
        }
        
        // Also clean up deleted lights and cameras from timeline
        for (const auto& item : items_to_delete) {
            if (item.type == SelectableType::Light && item.light) {
                auto it = ctx.scene.timeline.tracks.find(item.light->nodeName);
                if (it != ctx.scene.timeline.tracks.end()) {
                    ctx.scene.timeline.tracks.erase(it);
                }
            }
            else if (item.type == SelectableType::Camera && item.camera) {
                auto it = ctx.scene.timeline.tracks.find(item.camera->nodeName);
                if (it != ctx.scene.timeline.tracks.end()) {
                    ctx.scene.timeline.tracks.erase(it);
                }
            }
        }

        ctx.start_render = true;

        // [VERBOSE] SCENE_LOG_INFO("Deleted " + std::to_string(deleted_objects) + " objects, " + 
        //                std::to_string(deleted_lights) + " lights, " +
        //                std::to_string(deleted_cameras) + " cameras");
    }
}

// ============================================================================
// MARQUEE (BOX) SELECTION
// ============================================================================
void SceneUI::drawMarqueeRect() {
    if (!is_marquee_selecting) return;

    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    // Normalize rectangle (handle dragging in any direction)
    float x1 = fminf(marquee_start.x, marquee_end.x);
    float y1 = fminf(marquee_start.y, marquee_end.y);
    float x2 = fmaxf(marquee_start.x, marquee_end.x);
    float y2 = fmaxf(marquee_start.y, marquee_end.y);

    // Draw filled rect with transparency
    draw_list->AddRectFilled(ImVec2(x1, y1), ImVec2(x2, y2), IM_COL32(100, 150, 255, 40));
    // Draw border
    draw_list->AddRect(ImVec2(x1, y1), ImVec2(x2, y2), IM_COL32(100, 150, 255, 200), 0.0f, 0, 2.0f);
}
