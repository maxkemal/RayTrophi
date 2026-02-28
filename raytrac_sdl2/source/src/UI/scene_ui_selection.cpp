// ===============================================================================
// SCENE UI - SELECTION & INTERACTION
// ===============================================================================
// This file handles Mouse picking, Marquee selection, and Delete operations.
// ===============================================================================

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

            // Calculate bounding box center for quick check - USE CACHED BBOX!
            Vec3 bb_min, bb_max;
            auto bbox_it = bbox_cache.find(name);
            if (bbox_it != bbox_cache.end()) {
                bb_min = bbox_it->second.first;
                bb_max = bbox_it->second.second;
            } else {
                // Fallback: calculate if not in cache (shouldn't happen if cache is valid)
                bb_min = Vec3(1e10f, 1e10f, 1e10f);
                bb_max = Vec3(-1e10f, -1e10f, -1e10f);
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
            }

            Vec3 center = (bb_min + bb_max) * 0.5f;
            ImVec2 screenPos = ProjectToScreen(center);

            // Check if center OR any corner is inside marquee (catches partially visible objects)
            bool any_inside = false;
            
            // Check center first (quick test)
            if (screenPos.x >= x1 && screenPos.x <= x2 && screenPos.y >= y1 && screenPos.y <= y2) {
                any_inside = true;
            }
            
            // Check all 8 corners if center missed
            if (!any_inside) {
                Vec3 corners[8] = {
                    Vec3(bb_min.x, bb_min.y, bb_min.z), Vec3(bb_max.x, bb_min.y, bb_min.z),
                    Vec3(bb_min.x, bb_max.y, bb_min.z), Vec3(bb_max.x, bb_max.y, bb_min.z),
                    Vec3(bb_min.x, bb_min.y, bb_max.z), Vec3(bb_max.x, bb_min.y, bb_max.z),
                    Vec3(bb_min.x, bb_max.y, bb_max.z), Vec3(bb_max.x, bb_max.y, bb_max.z)
                };
                
                for (int c = 0; c < 8; ++c) {
                    ImVec2 sp = ProjectToScreen(corners[c]);
                    if (sp.x >= x1 && sp.x <= x2 && sp.y >= y1 && sp.y <= y2) {
                        any_inside = true;
                        break;
                    }
                }
            }
            
            if (any_inside) {
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

void SceneUI::handleDeleteShortcut(UIContext& ctx)
{
    if (!ImGui::IsKeyPressed(ImGuiKey_Delete) &&
        !ImGui::IsKeyPressed(ImGuiKey_X)) return;

    // Use the optimized multi-selection capable triggerDelete
    if (ctx.selection.hasSelection()) {
        triggerDelete(ctx);
    }
}

void SceneUI::handleMouseSelection(UIContext& ctx) {
    // Only select if not interacting with UI or Gizmo
    if (ImGui::IsMouseClicked(0)) {
        bool capture = ImGui::GetIO().WantCaptureMouse;
        bool gizmo_over = ImGuizmo::IsOver();
        
        //// Detailed logging for diagnostics
        //SCENE_LOG_INFO("Viewport click. Cache: " + std::string(mesh_cache_valid ? "OK" : "NO") + 
        //               ", ObjCount: " + std::to_string(ctx.scene.world.objects.size()) + 
        //               ", Capture=" + std::to_string(capture) + 
        //               ", GizmoOver=" + std::to_string(gizmo_over) + 
        //               ", HUD=" + std::to_string(hud_captured_mouse) +
        //               ", Dragging=" + std::to_string(is_dragging));
                       
        if (!is_dragging) {
            // Ignore click if over UI elements (Window/Panel), Gizmo, or HUD overlay
            if (capture || gizmo_over || hud_captured_mouse) {
                // Reset HUD flag for next frame
                hud_captured_mouse = false;
                return;
            }
        }

        // ===========================================================================
        // SELECTION CACHE SYNC (Now centralized in update(), but kept here as safety)
        // ===========================================================================
        if (!mesh_cache_valid) {
            rebuildMeshCache(ctx.scene.world.objects);
        }

        // ===========================================================================
        // GPU MODE VERTEX SYNC: In GPU TLAS mode, CPU vertices are NOT updated during
        // gizmo drag (only GPU transform is updated). We must sync before picking.
        // This is a one-time cost per click - acceptable for accurate selection.
        // ===========================================================================
        extern bool g_bvh_rebuild_pending;
        if (g_bvh_rebuild_pending && ctx.render_settings.use_optix) {
            // BVH is stale - transforms changed. Sync ALL objects with TransformHandle.
            int synced_objects = 0;
            for (auto& [name, triangles] : mesh_cache) {
                if (triangles.empty()) continue;
                auto first_tri = triangles[0].second;
                if (first_tri && first_tri->getTransformHandle()) {
                    for (auto& pair : triangles) {
                        pair.second->updateTransformedVertices();
                    }
                    synced_objects++;
                }
            }
            if (synced_objects > 0) {
              //  SCENE_LOG_INFO("GPU picking sync: updated " + std::to_string(synced_objects) + " objects");
            }
            // Clear the pending flag after full sync
            g_bvh_rebuild_pending = false;
        }

        // Also process lazy sync queue (for any stragglers)
        ensureCPUSyncForPicking(ctx);


        // ===========================================================================
        // SKINNED MESH FIX: When using GPU rendering with animations,
        // CPU vertices may be out of sync. Force a sync for picking accuracy.
        // ===========================================================================
        if (ctx.render_settings.use_optix && !ctx.scene.animationDataList.empty()) {
            // We have GPU rendering + animations: sync CPU vertices for skinned meshes
            // This is a one-time cost per click, acceptable for accurate selection
            bool synced_any = false;
            for (auto& obj : ctx.scene.world.objects) {
                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                if (tri && !tri->getVertexBoneWeights().empty()) {
                    // This is a skinned triangle - apply current bone matrices
                    if (!ctx.renderer.finalBoneMatrices.empty()) {
                        tri->apply_skinning(ctx.renderer.finalBoneMatrices);
                        synced_any = true;
                    }
                }
            }
            if (synced_any) {
                SCENE_LOG_INFO("Synced skinned mesh vertices for viewport selection");
            }
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
                auto& cam = ctx.scene.cameras[i];
                if (!cam || cam == ctx.scene.camera) continue;

                Vec3 oc = r.origin - cam->lookfrom;
                float radius = 0.6f;
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
                }
            }

            // Check for Force Field Selection (Bounding Sphere Intersection for icons)
            std::shared_ptr<Physics::ForceField> closest_force_field = nullptr;
            float closest_force_field_t = closest_camera_t; // Must be closer than lights/cameras

            for (size_t i = 0; i < ctx.scene.force_field_manager.force_fields.size(); ++i) {
                auto& field = ctx.scene.force_field_manager.force_fields[i];
                if (!field || !field->visible) continue;

                Vec3 oc = r.origin - field->position;
                float radius = 0.4f; // Interface icon size
                float a = r.direction.dot(r.direction);
                float half_b = oc.dot(r.direction);
                float c = oc.dot(oc) - radius * radius;
                float discriminant = half_b * half_b - a * c;

                if (discriminant > 0) {
                    float root = sqrt(discriminant);
                    float temp = (-half_b - root) / a;
                    if (temp < closest_force_field_t && temp > 0.001f) {
                        closest_force_field_t = temp;
                        closest_force_field = field;
                    }
                }
            }

            // ===========================================================================
            // SMART SELECTION: GPU picking (O(1)) with CPU fallback
            // GPU mode: Try pick buffer first, fall back to CPU linear scan
            // CPU mode: Linear scan through mesh_cache with updated vertices
            // ===========================================================================

            HitRecord rec;
            bool hit = false;
            float closest_so_far = 1e9f;
            HitRecord temp_rec;

            // GPU PICKING PATH: Use pick buffer for O(1) object selection
            // Skip GPU pick if rebuild is pending (pick buffer is stale after object delete/add)
            extern bool g_optix_rebuild_pending;
            bool gpu_pick_success = false;
            std::string gpu_picked_name;
            
            bool use_gpu = ctx.render_settings.use_optix;
            bool has_ptr = (ctx.backend_ptr != nullptr);
            bool rebuild_pending = g_optix_rebuild_pending;
            
            if (use_gpu && has_ptr && !rebuild_pending) {
                // Pass viewport dimensions for coordinate scaling
                int vp_w = static_cast<int>(win_w);
                int vp_h = static_cast<int>(win_h);
                int object_id = ctx.backend_ptr->getPickedObjectId(x, y, vp_w, vp_h);
                if (object_id >= 0) {
                    gpu_picked_name = ctx.backend_ptr->getPickedObjectName(x, y, vp_w, vp_h);
                    // Only mark as success if name found AND exists in mesh_cache
                    if (!gpu_picked_name.empty() && mesh_cache.find(gpu_picked_name) != mesh_cache.end()) {
                        // [FIX] Ignore ForceField visualization meshes
                        if (gpu_picked_name.find("ForceField") == std::string::npos && 
                            gpu_picked_name.find("Force Field") == std::string::npos) {
                            gpu_pick_success = true;
                        }
                       // SCENE_LOG_INFO("GPU Pick: " + gpu_picked_name);
                    }
                }
            }

            // =======================================================================
            // CPU BVH PICKING: Faster fallback for large scenes (e.g. 1.2M triangles)
            // =======================================================================
            if (!gpu_pick_success) {
                // First try the world BVH (if not being rebuilt)
                extern bool g_bvh_rebuild_pending;
                if (ctx.scene.bvh && !g_bvh_rebuild_pending) {
                    if (ctx.scene.bvh->hit(r, 0.001f, closest_so_far, temp_rec)) {
                        hit = true;
                        closest_so_far = temp_rec.t;
                        rec = temp_rec;
                    }
                }
                
                // Fallback to Linear Scan ONLY if scene is very small OR BVH missing/rebuilding
                // Note: Rebuilding 1.2M objects BVH is much faster than linear scanning them!
                if (!hit && (mesh_cache.size() < 1000 || !ctx.scene.bvh)) {
                    for (const auto& [name, triangles] : mesh_cache) {
                        // [FIX] Ignore ForceField gizmos
                        if (name.find("ForceField") != std::string::npos || 
                            name.find("Force Field") != std::string::npos) continue;

                        for (const auto& pair : triangles) {
                            if (pair.second->hit(r, 0.001f, closest_so_far, temp_rec)) {
                                hit = true;
                                closest_so_far = temp_rec.t;
                                rec = temp_rec;
                            }
                        }
                    }
                }
            }
            
            // Check Gas Volumes (not in main BVH, separate list)
            for (const auto& gas : ctx.scene.gas_volumes) {
                if (gas->hit(r, 0.001f, closest_so_far, temp_rec)) {
                    hit = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            // Check VDB Volumes (not in main BVH, separate list)
            for (const auto& vdb : ctx.scene.vdb_volumes) {
                if (vdb->hit(r, 0.001f, closest_so_far, temp_rec)) {
                    hit = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            // ===========================================================
            // APPLY PICK FOCUS (Using Found Hits)
            // ===========================================================
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
                // Check Force Field Hit
                if (closest_force_field && closest_force_field_t < min_dist) {
                    min_dist = closest_force_field_t;
                    found_hit = true;
                }

                if (found_hit) {
                    ctx.scene.camera->focus_dist = min_dist;
                    ctx.scene.camera->update_camera_vectors();
                    if (ctx.backend_ptr) {
                        ctx.backend_ptr->syncCamera(*ctx.scene.camera);
                        ctx.backend_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                   // SCENE_LOG_INFO(std::string("Pick Focus set to: ") + std::to_string(min_dist) + "m");
                }

                is_picking_focus = false;
                ctx.start_render = true;
                return;
            }

            // Priority Selection: ForceField > Camera > Light > Object (by distance)
            if (closest_force_field && closest_force_field_t < closest_so_far && closest_force_field_t < closest_camera_t && closest_force_field_t < closest_t) {
                if (ctrl_held) {
                    SelectableItem item;
                    item.type = SelectableType::ForceField;
                    item.force_field = closest_force_field;
                    item.name = closest_force_field->name;
                    if (ctx.selection.isSelected(item)) ctx.selection.removeFromSelection(item);
                    else ctx.selection.addToSelection(item);
                } else {
                    ctx.selection.selectForceField(closest_force_field);
                }
                return;
            }

            if (closest_camera && closest_camera_t < closest_so_far && closest_camera_t < closest_t) {
                // Camera is the closest thing clicked
                if (ctrl_held) {
                    SelectableItem item;
                    item.type = SelectableType::Camera;
                    item.camera = closest_camera;
                    item.name = closest_camera->nodeName.empty() ? "Camera" : closest_camera->nodeName;

                    if (ctx.selection.isSelected(item)) {
                        ctx.selection.removeFromSelection(item);
                    } else {
                        ctx.selection.addToSelection(item);
                    }
                } else {
                    ctx.selection.selectCamera(closest_camera);
                }
                return; // Camera selected, done
            }

            // ===========================================================================
            // GPU PICK SUCCESS PATH: Direct mesh selection from pick buffer result
            // ===========================================================================
            if (gpu_pick_success && !gpu_picked_name.empty()) {
                // Find the object in mesh_cache using GPU-provided name
                auto cache_it = mesh_cache.find(gpu_picked_name);
                if (cache_it != mesh_cache.end() && !cache_it->second.empty()) {
                    auto& first_tri = cache_it->second[0].second;
                    int index = cache_it->second[0].first;
                    
                    if (ctrl_held) {
                        SelectableItem item;
                        item.type = SelectableType::Object;
                        item.object = first_tri;
                        item.object_index = index;
                        item.name = first_tri->nodeName;
                        
                        if (ctx.selection.isSelected(item)) {
                            ctx.selection.removeFromSelection(item);
                        } else {
                            ctx.selection.addToSelection(item);
                        }
                    } else {
                        ctx.selection.selectObject(first_tri, index, first_tri->nodeName);
                        
                        // TERRAIN CONNECTION: Check if this is a terrain chunk
                        std::string tName = first_tri->nodeName;
                        if (tName.find("Terrain_") == 0) {
                            size_t chunkPos = tName.find("_Chunk");
                            if (chunkPos != std::string::npos) {
                                tName = tName.substr(0, chunkPos);
                            }
                            auto terrain = TerrainManager::getInstance().getTerrainByName(tName);
                            if (terrain) {
                                terrain_brush.active_terrain_id = terrain->id;
                                show_terrain_tab = true;
                                //SCENE_LOG_INFO("Terrain selected via GPU pick: " + tName);
                            }
                        }
                    }
                    return; // GPU pick selection done
                }
                // Note: If gpu_pick_success is true but cache lookup failed, we already
                // handled that above by setting gpu_pick_success = false before CPU scan
            }

            if (hit && (rec.t < closest_t)) {

                // --- VDB VOLUME SELECTION ---
                if (rec.vdb_volume) {
                    std::shared_ptr<VDBVolume> found_vdb = nullptr;
                    int index = -1;

                    // Find shared_ptr for this VDB
                    for (size_t i = 0; i < ctx.scene.vdb_volumes.size(); ++i) {
                        if (ctx.scene.vdb_volumes[i].get() == rec.vdb_volume) {
                            found_vdb = ctx.scene.vdb_volumes[i];
                            index = (int)i;
                            break;
                        }
                    }

                    if (found_vdb) {
                        if (ctrl_held) {
                            // Multi-selection
                            SelectableItem item;
                            item.type = SelectableType::VDBVolume;
                            item.vdb_volume = found_vdb;
                            item.vdb_index = index;
                            item.name = found_vdb->name;

                            if (ctx.selection.isSelected(item)) {
                                ctx.selection.removeFromSelection(item);
                            }
                            else {
                                ctx.selection.addToSelection(item);
                            }
                        }
                        else {
                            // Single selection
                            ctx.selection.selectVDBVolume(found_vdb, index, found_vdb->name);
                           // SCENE_LOG_INFO("Selected VDB Volume via viewport: " + found_vdb->name);
                        }
                    }
                }
                
                // --- GAS VOLUME SELECTION ---
                else if (rec.gas_volume) {
                    std::shared_ptr<GasVolume> found_gas = nullptr;
                    int index = -1;

                    // Find shared_ptr for this Gas Volume
                    for (size_t i = 0; i < ctx.scene.gas_volumes.size(); ++i) {
                        if (ctx.scene.gas_volumes[i].get() == rec.gas_volume) {
                            found_gas = ctx.scene.gas_volumes[i];
                            index = (int)i;
                            break;
                        }
                    }

                    if (found_gas) {
                        if (ctrl_held) {
                            // Multi-selection
                            SelectableItem item;
                            item.type = SelectableType::GasVolume;
                            item.gas_volume = found_gas;
                            item.vdb_index = index; // Reuse vdb_index for gas index
                            item.name = found_gas->name;

                            if (ctx.selection.isSelected(item)) {
                                ctx.selection.removeFromSelection(item);
                            }
                            else {
                                ctx.selection.addToSelection(item);
                            }
                        }
                        else {
                            // Single selection
                            ctx.selection.selectGasVolume(found_gas, index, found_gas->name);
                           // SCENE_LOG_INFO("Selected Gas Volume via viewport: " + found_gas->name);
                        }
                    }
                }

                // --- TRIANGLE SELECTION ---
                else if (rec.triangle) {
                    std::shared_ptr<Triangle> found_tri = nullptr;
                    int index = -1;

                    // Ensure cache is valid
                    if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

                    // Try fast O(1) lookup first with const-correctness
                    auto it = tri_to_index.find(rec.triangle);
                    if (it != tri_to_index.end()) {
                        index = it->second;
                        if (index >= 0 && index < (int)ctx.scene.world.objects.size()) {
                            found_tri = std::dynamic_pointer_cast<Triangle>(ctx.scene.world.objects[index]);
                        }
                    }

                    if (found_tri) {
                        if (ctrl_held) {
                            SelectableItem item;
                            item.type = SelectableType::Object;
                            item.object = found_tri;
                            item.object_index = index;
                            item.name = found_tri->nodeName;

                            if (ctx.selection.isSelected(item)) {
                                ctx.selection.removeFromSelection(item);
                            } else {
                                ctx.selection.addToSelection(item);
                            }
                        } else {
                            ctx.selection.selectObject(found_tri, index, found_tri->nodeName);
                            SCENE_LOG_INFO("Selected via CPU Viewport: " + found_tri->nodeName);

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
                                }
                            }
                        }
                    } else {
                        SCENE_LOG_WARN("Selection: Triangle hit but not found in cache. Forcing rebuild...");
                        rebuildMeshCache(ctx.scene.world.objects);
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
}

// ============================================================================
// Delete Operation (Shared by Menu and Key Shortcut)
// ============================================================================
// OPTIMIZED VERSION - O(n) instead of O(n²)
void SceneUI::triggerDelete(UIContext& ctx) {
    if (!ctx.selection.hasSelection()) return;

    // Collect all items to delete (supports multi-selection)
    std::vector<SelectableItem> items_to_delete = ctx.selection.multi_selection;

    // OPTIMIZATION: Collect ALL hittable pointers to delete into a single set for O(1) lookup
    // Using Hittable* (base class) for type safety
    std::unordered_set<Hittable*> objects_to_delete;
    std::vector<std::string> deleted_names;
    std::vector<std::pair<std::string, std::vector<std::shared_ptr<Triangle>>>> undo_data;

    // ===========================================================================
    // VDB VOLUME DELETION
    // ===========================================================================
    bool vdb_deleted = false;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::VDBVolume && item.vdb_volume) {
            // Unload GPU resources
            item.vdb_volume->unload();

            // Remove from Scene VDB list
            if (ctx.scene.removeVDBVolume(item.vdb_volume)) {
                vdb_deleted = true;
                SCENE_LOG_INFO("Deleted VDB Volume: " + item.vdb_volume->name);
            }

            // Also ensure it is removed from world.objects (Hittable list)
            // This is handled generically below if we add it to objects_to_delete, 
            // OR we can explicitly remove it here.
            // Since VDBVolume is a Hittable, forcing it into objects_to_delete is safest.
            objects_to_delete.insert(item.vdb_volume.get());
        }
    }

    if (vdb_deleted) {
        // Sync empty/reduced list to GPU immediately to prevent invalid memory access
        if (ctx.backend_ptr) {
            syncVDBVolumesToGPU(ctx);
            // Also reset accumulation as scene changed
            ctx.backend_ptr->resetAccumulation();
        }
        ctx.renderer.resetCPUAccumulation();
    }

    // ===========================================================================
    // GAS VOLUME DELETION
    // ===========================================================================
    bool gas_deleted = false;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::GasVolume && item.gas_volume) {
            // Unload GPU resources
            item.gas_volume->freeGPUResources();

            // Remove from Scene Gas list
            if (ctx.scene.removeGasVolume(item.gas_volume)) {
                gas_deleted = true;
                SCENE_LOG_INFO("Deleted Gas Volume: " + item.gas_volume->name);
            }

            // GasVolume is a Hittable, must be removed from world.objects
            objects_to_delete.insert(item.gas_volume.get());
        }
    }

    if (gas_deleted) {
        if (ctx.backend_ptr) {
            ctx.renderer.updateBackendGasVolumes(ctx.scene);
            ctx.backend_ptr->resetAccumulation();
        }
        ctx.renderer.resetCPUAccumulation();
    }

    // ===========================================================================
    // FORCE FIELD DELETION
    // ===========================================================================
    int ff_deleted_count = 0;
    for (const auto& item : items_to_delete) {
        if (item.type == SelectableType::ForceField && item.force_field) {
            if (ctx.scene.removeForceField(item.force_field)) {
                ff_deleted_count++;
                SCENE_LOG_INFO("Deleted Force Field: " + item.force_field->name);
            }
        }
    }

    if (ff_deleted_count > 0) {
        // Reset accumulation if needed (force fields might affect visual simulation)
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
    }

    // Build mesh cache once if needed
    if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

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

    // ===========================================================================
    // Handle camera deletions
    // ===========================================================================
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

        g_mesh_cache_dirty = true;  // Flag for Main.cpp / other systems

        // Update class tracker and FORCE FULL REBUILD
        last_scene_obj_count = ctx.scene.world.objects.size();
        invalidateCache();

        if (deleted_objects > 0) {
            // =======================================================================
            // INCREMENTAL GPU UPDATE (TLAS mode) - Instant!
            // =======================================================================
            if (ctx.backend_ptr && ctx.backend_ptr->isUsingTLAS()) {
                // Fast path: Just hide instances by setting visibility_mask = 0
                for (const auto& name : deleted_names) {
                    ctx.backend_ptr->hideInstancesByNodeName(name);
                }
                // Single TLAS update after ALL hides complete (batched for efficiency)
                ctx.backend_ptr->rebuildAccelerationStructure();
            }
            else {
                // GAS mode fallback: Full rebuild required
                g_optix_rebuild_pending = true;
            }

            // CPU BVH still needs rebuild for picking (async)
            g_bvh_rebuild_pending = true;
        }

        // Update lights if any were deleted
        if (deleted_lights > 0 && ctx.backend_ptr) {
            ctx.backend_ptr->setLights(ctx.scene.lights);
            ctx.backend_ptr->resetAccumulation();
        }

        // Update GPU camera if cameras changed
        if (deleted_cameras > 0 && ctx.backend_ptr && ctx.scene.camera) {
            ctx.backend_ptr->syncCamera(*ctx.scene.camera);
            ctx.backend_ptr->resetAccumulation();
        }

        // ===========================================================================
        // AUTO-CLEANUP TIMELINE TRACKS FOR DELETED ENTITIES
        // ===========================================================================
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

void SceneUI::triggerDuplicate(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;
    if (!sel.hasSelection()) return;

    // Build a list of objects to duplicate
    // If multi-selection exists, use it. Otherwise use the single active selection.
    std::vector<SelectableItem> itemsToDuplicate;
    if (sel.multi_selection.size() > 0) {
        itemsToDuplicate = sel.multi_selection;
    } else {
        itemsToDuplicate.push_back(sel.selected);
    }

    std::vector<std::shared_ptr<Hittable>> allNewTriangles;
    std::vector<SelectableItem> newSelectionList;
    
    // Temporary map for name uniqueness check
    if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);
    
    // Perform duplication for each item
    bool anyDuplicated = false;
    std::unordered_set<std::string> alreadyDuplicated; // Track duplicated source names
    std::unordered_set<std::string> assignedNames;   // Track names assigned to clones
    
    for (const auto& item : itemsToDuplicate) {
        if (item.type == SelectableType::Object && item.object) {
            
            // Use item.name as primary, fallback to nodeName
            std::string targetName = item.name;
            if (targetName.empty()) {
                targetName = item.object->nodeName;
            }
            if (targetName.empty()) targetName = "Unnamed";
            
            // Skip if this object name was already duplicated
            if (alreadyDuplicated.count(targetName) > 0) continue;
            alreadyDuplicated.insert(targetName);

            // Unique name generation
            std::string baseName = targetName;
            size_t lastUnderscore = baseName.rfind('_');
            if (lastUnderscore != std::string::npos) {
                std::string suffix = baseName.substr(lastUnderscore + 1);
                if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
                    baseName = baseName.substr(0, lastUnderscore);
                }
            }
            int copyNum = 1;
            std::string newName;
            do { 
                newName = baseName + "_" + std::to_string(copyNum++); 
            } while (mesh_cache.count(newName) > 0 || assignedNames.count(newName) > 0);
            
            assignedNames.insert(newName);

            // Create Unique Transform
            std::shared_ptr<Transform> newTransform = std::make_shared<Transform>();
            if (item.object->getTransformHandle()) {
                *newTransform = *item.object->getTransformHandle();
            }

            // Duplicate Triangles - search by targetName
            std::shared_ptr<Triangle> firstNewTri = nullptr;
            auto it = mesh_cache.find(targetName);
            if (it != mesh_cache.end()) {
                for (auto& pair : it->second) {
                    auto& oldTri = pair.second;
                    auto newTri = std::make_shared<Triangle>(*oldTri);
                    newTri->setTransformHandle(newTransform);
                    newTri->setNodeName(newName);
                    
                    allNewTriangles.push_back(newTri);
                    if (!firstNewTri) firstNewTri = newTri;
                }
            }
            
            if (firstNewTri) {
                SelectableItem newItem;
                newItem.type = SelectableType::Object;
                newItem.object = firstNewTri;
                newItem.object_index = (int)ctx.scene.world.objects.size() + (int)allNewTriangles.size() - 1;
                newItem.name = newName;
                
                if (auto th = firstNewTri->getTransformHandle()) {
                    newItem.position = Vec3(th->base.m[0][3], th->base.m[1][3], th->base.m[2][3]);
                } else {
                    newItem.position = Vec3(0, 0, 0);
                }
                newItem.rotation = Vec3(0, 0, 0); // Decomposed on next update if needed
                newItem.scale = Vec3(1, 1, 1);
                
                newSelectionList.push_back(newItem);
                anyDuplicated = true;
            }
        }
    }
    
    // VDB Duplication
    for (const auto& item : itemsToDuplicate) {
        if (item.type == SelectableType::VDBVolume && item.vdb_volume) {
            auto oldVDB = item.vdb_volume;
            auto newVDB = std::make_shared<VDBVolume>();
            
            newVDB->loadVDB(oldVDB->getFilePath());
            VDBVolumeManager::getInstance().uploadToGPU(newVDB->getVDBVolumeID());
            newVDB->setTransform(oldVDB->getTransform());
            
            auto oldShader = oldVDB->getShader();
            if (oldShader) {
                auto newShader = std::make_shared<VolumeShader>(*oldShader);
                newShader->name = oldShader->name + " (Copy)";
                newVDB->setShader(newShader);
            }
            
            newVDB->name = oldVDB->name + "_Copy";
            ctx.scene.addVDBVolume(newVDB);
            ctx.scene.world.add(newVDB);
            
            SelectableItem newSel;
            newSel.type = SelectableType::VDBVolume;
            newSel.vdb_volume = newVDB;
            newSel.vdb_index = (int)ctx.scene.vdb_volumes.size() - 1;
            newSel.name = newVDB->name;
            newSel.position = newVDB->getPosition();
            newSel.rotation = newVDB->getRotation();
            newSel.scale = newVDB->getScale();
            
            newSelectionList.push_back(newSel);
            anyDuplicated = true;
        }
    }

    // Light Duplication
    for (const auto& item : itemsToDuplicate) {
        if (item.type == SelectableType::Light && item.light) {
            std::shared_ptr<Light> newLight = nullptr;
            auto l = item.light;
            
            if (auto pl = std::dynamic_pointer_cast<PointLight>(l)) newLight = std::make_shared<PointLight>(*pl);
            else if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(l)) newLight = std::make_shared<DirectionalLight>(*dl);
            else if (auto sl = std::dynamic_pointer_cast<SpotLight>(l)) newLight = std::make_shared<SpotLight>(*sl);
            else if (auto al = std::dynamic_pointer_cast<AreaLight>(l)) newLight = std::make_shared<AreaLight>(*al);

            if (newLight) {
                std::string newName = l->nodeName + "_Copy";
                newLight->nodeName = newName;
                
                ctx.scene.lights.push_back(newLight);
                history.record(std::make_unique<AddLightCommand>(newLight));
                
                SelectableItem newSel;
                newSel.type = SelectableType::Light;
                newSel.light = newLight;
                newSel.light_index = (int)ctx.scene.lights.size() - 1;
                newSel.name = newName;
                newSel.position = newLight->position;
                newSel.rotation = Vec3(0, 0, 0);
                newSel.scale = Vec3(1, 1, 1);
                
                newSelectionList.push_back(newSel);
                anyDuplicated = true;
            }
        }
    }

    if (anyDuplicated) {
        if (!allNewTriangles.empty()) {
            ctx.scene.world.objects.insert(ctx.scene.world.objects.end(), allNewTriangles.begin(), allNewTriangles.end());
            rebuildMeshCache(ctx.scene.world.objects);
        }
        
        sel.clearSelection();
        for (const auto& newItem : newSelectionList) {
            sel.addToSelection(newItem);
        }
        
        extern bool g_optix_rebuild_pending;
        g_optix_rebuild_pending = true;
        extern bool g_bvh_rebuild_pending;
        g_bvh_rebuild_pending = true;
        
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            ctx.backend_ptr->setLights(ctx.scene.lights);
            ctx.backend_ptr->resetAccumulation();
        }
        
        ProjectManager::getInstance().markModified();
    }
}
