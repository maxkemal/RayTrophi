// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - HIERARCHY PANEL
// ═══════════════════════════════════════════════════════════════════════════════
// This file handles the Scene Hierarchy tree view (Outliner).
// ═══════════════════════════════════════════════════════════════════════════════

#include "scene_ui.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "imgui.h"
#include "scene_data.h"
#include "Triangle.h"
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"
#include "AreaLight.h"
#include "Volumetric.h"

#include <unordered_set>
#include "TerrainManager.h"

// ═══════════════════════════════════════════════════════════════════════════════
// SCENE HIERARCHY PANEL (Outliner)
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawSceneHierarchy(UIContext& ctx) {
    // Window creation logic removed - now embedded in tabs

    // Check if embedded in another window, if not create child
    // Since we are moving to tab, we don't need window creation here


    SceneSelection& sel = ctx.selection;

    // ─────────────────────────────────────────────────────────────────────────
    // DELETE LOGIC (Keyboard Shortcut)
    // ─────────────────────────────────────────────────────────────────────────
    // Only process when viewport has focus (not UI panels)
    if ((ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_X)) &&
        sel.hasSelection() && !ImGui::GetIO().WantCaptureKeyboard) {
        triggerDelete(ctx);
    }


    // ─────────────────────────────────────────────────────────────────────────
    // Scene Tree
    // ─────────────────────────────────────────────────────────────────────────
    float available_h = ImGui::GetContentRegionAvail().y;
    // Split: 25% for Tree, Rest for Properties (World/Material/etc)
    ImGui::BeginChild("HierarchyTree", ImVec2(0, available_h * 0.25f), true);

    // WORLD (Environment) - Selectable like any other object
    {
        bool world_selected = (sel.selected.type == SelectableType::World);
        ImGuiTreeNodeFlags world_flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (world_selected) world_flags |= ImGuiTreeNodeFlags_Selected;

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.7f, 1.0f, 1.0f));  // Pink
        ImGui::TreeNodeEx("[W] World", world_flags);
        ImGui::PopStyleColor();

        if (ImGui::IsItemClicked()) {
            sel.selectWorld();
        }
    }

    ImGui::Separator();

    // CAMERAS (Multi-camera support)
    if (!ctx.scene.cameras.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
        if (ImGui::TreeNode("Cameras")) {
            ImGui::PopStyleColor();

            for (size_t i = 0; i < ctx.scene.cameras.size(); i++) {
                ImGui::PushID((int)(1000 + i));  // Unique ID for each camera

                auto& cam = ctx.scene.cameras[i];
                if (!cam) { ImGui::PopID(); continue; }

                bool is_selected = (sel.selected.type == SelectableType::Camera &&
                    sel.selected.camera == cam);
                bool is_active = (i == ctx.scene.active_camera_index);

                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;

                // Active camera indicator
                std::string label = is_active ? "[>] Camera #" + std::to_string(i) + " (Active)"
                    : "[O] Camera #" + std::to_string(i);

                ImVec4 color = is_active ? ImVec4(0.3f, 1.0f, 0.5f, 1.0f) : ImVec4(0.5f, 0.7f, 1.0f, 1.0f);
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::TreeNodeEx(label.c_str(), flags);
                ImGui::PopStyleColor();

                if (ImGui::IsItemClicked()) {
                    sel.selectCamera(cam);
                }

                // Double-click to set as active
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                    ctx.scene.setActiveCamera(i);
                    if (ctx.optix_gpu_ptr && ctx.scene.camera) {
                        ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                    SCENE_LOG_INFO("Set Camera #" + std::to_string(i) + " as active");
                }

                ImGui::PopID();
            }
            ImGui::TreePop();
        }
        else {
            ImGui::PopStyleColor();
        }
    }
    else if (ctx.scene.camera) {
        // Fallback for single legacy camera
        bool is_selected = (sel.selected.type == SelectableType::Camera);
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;

        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
        ImGui::TreeNodeEx("[CAM] Camera", flags);
        ImGui::PopStyleColor();

        if (ImGui::IsItemClicked()) {
            sel.selectCamera(ctx.scene.camera);
        }
    }

    // LIGHTS
    if (!ctx.scene.lights.empty()) {
        if (ImGui::TreeNode("Lights")) {
            // SELECT ALL / SELECT NONE buttons
            if (ImGui::Button("Select All##lights")) {
                ctx.selection.clearSelection();
                for (size_t i = 0; i < ctx.scene.lights.size(); ++i) {
                    auto& light = ctx.scene.lights[i];
                    SelectableItem item;
                    item.type = SelectableType::Light;
                    item.light = light;
                    item.light_index = (int)i;
                    item.name = "Light_" + std::to_string(i);
                    ctx.selection.addToSelection(item);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Select None##lights")) {
                ctx.selection.clearSelection();
            }
            ImGui::SameLine();
            ImGui::Text("(%d lights)", (int)ctx.scene.lights.size());

            for (size_t i = 0; i < ctx.scene.lights.size(); i++) {
                ImGui::PushID((int)i);  // Unique ID for each light

                auto& light = ctx.scene.lights[i];
                bool is_selected = (sel.selected.type == SelectableType::Light &&
                    sel.selected.light_index == (int)i);

                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;

                // Icon based on light type
                const char* icon = "[*]";  // Point light default
                ImVec4 color = ImVec4(1.0f, 0.9f, 0.4f, 1.0f);  // Yellow for lights

                std::string light_type = "Light";
                switch (light->type()) {
                case LightType::Point: icon = "[*]"; light_type = "Point"; break;
                case LightType::Directional: icon = "[>]"; light_type = "Directional"; color = ImVec4(1.0f, 0.7f, 0.3f, 1.0f); break;
                case LightType::Spot: icon = "[V]"; light_type = "Spot"; break;
                case LightType::Area: icon = "[#]"; light_type = "Area"; break;
                }

                std::string label = std::string(icon) + " " + light_type + " " + std::to_string(i + 1);

                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::TreeNodeEx(label.c_str(), flags);
                ImGui::PopStyleColor();

                if (ImGui::IsItemClicked()) {
                    sel.selectLight(light, (int)i, label);
                }

                ImGui::PopID();  // End unique ID
            }
            ImGui::TreePop();
        }
    }
    // Check for scene changes to invalidate cache
    static size_t last_obj_count = 0;
    if (ctx.scene.world.objects.size() != last_obj_count) {
        mesh_cache_valid = false;
        last_obj_count = ctx.scene.world.objects.size();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // OBJECTS LIST (HIERARCHY)
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::TreeNode("Objects")) {

        // Ensure cache is valid
        if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

        // SELECT ALL / SELECT NONE buttons
        if (ImGui::Button("Select All##obj")) {
            ctx.selection.clearSelection();
            for (auto& [name, triangles] : mesh_cache) {
                if (triangles.empty()) continue;

                // Check if all triangles share same transform (skip procedurals)
                auto firstHandle = triangles[0].second->getTransformHandle();
                bool all_same = true;
                for (size_t i = 1; i < triangles.size() && all_same; ++i) {
                    auto h = triangles[i].second->getTransformHandle();
                    if (h.get() != firstHandle.get()) all_same = false;
                }

                if (all_same) {
                    SelectableItem item;
                    item.type = SelectableType::Object;
                    item.object = triangles[0].second;
                    item.object_index = triangles[0].first;
                    item.name = name;
                    ctx.selection.addToSelection(item);
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Select None##obj")) {
            ctx.selection.clearSelection();
        }
        ImGui::SameLine();
        ImGui::Text("(%d objects)", (int)mesh_ui_cache.size());

        static ImGuiTextFilter filter;
        filter.Draw("Filter##objects");

        ImGuiListClipper clipper;
        clipper.Begin((int)mesh_ui_cache.size());

        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
                if (i >= mesh_ui_cache.size()) break;

                auto& kv = mesh_ui_cache[i];
                const std::string& name = kv.first;

                // Simple filter check
                if (filter.IsActive() && !filter.PassFilter(name.c_str())) continue;

                bool is_selected = (sel.selected.type == SelectableType::Object &&
                    sel.selected.object &&
                    sel.selected.object->nodeName == name);

                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow;
                if (!is_selected) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                if (is_selected) flags |= ImGuiTreeNodeFlags_Selected | ImGuiTreeNodeFlags_DefaultOpen;

                ImGui::PushID(i);
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.8f, 0.8f, 0.85f, 1.0f));
                std::string displayName = name.empty() ? "Unnamed Object" : name;

                bool node_open = false;
                if (is_selected) {
                    node_open = ImGui::TreeNodeEx(displayName.c_str(), flags);
                }
                else {
                    ImGui::TreeNodeEx(displayName.c_str(), flags); // Leaf, no push
                }
                ImGui::PopStyleColor();

                if (ImGui::IsItemClicked()) {
                    if (!kv.second.empty()) {
                        auto& first_pair = kv.second[0];
                        sel.selectObject(first_pair.second, first_pair.first, name);
                        
                        // TERRAIN CONNECTION: Check if this is a terrain chunk
                        if (name.find("Terrain_") == 0) {
                            // Try to find corresponding TerrainObject
                            // Name format: "Terrain_X" or "Terrain_X_Chunk"
                            // Let's strip "_Chunk" if present
                            std::string tName = name;
                            size_t chunkPos = tName.find("_Chunk");
                            if (chunkPos != std::string::npos) {
                                tName = tName.substr(0, chunkPos);
                            }
                            
                            auto terrain = TerrainManager::getInstance().getTerrainByName(tName);
                            if (terrain) {
                                terrain_brush.active_terrain_id = terrain->id;
                                show_terrain_tab = true;
                                tab_to_focus = "Terrain";
                                SCENE_LOG_INFO("Terrain selected: " + tName);
                            }
                        }
                    }
                }

                if (node_open && is_selected) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
                    ImGui::Indent();

                    // --- In-Tree Properties ---
                    Vec3 pos = sel.selected.position;
                    // Position Control
                    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
                    if (ImGui::DragFloat3("##Pos", &pos.x, 0.1f)) {
                        Vec3 delta = pos - sel.selected.position;
                        sel.selected.position = pos;
                        if (!kv.second.empty()) {
                            auto tri = kv.second[0].second;
                            auto t_handle = tri->getTransformHandle();
                            if (t_handle) {
                                t_handle->base.m[0][3] = pos.x;
                                t_handle->base.m[1][3] = pos.y;
                                t_handle->base.m[2][3] = pos.z;
                            }
                            // Sync Updates
                            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                            ctx.renderer.resetCPUAccumulation();
                            if (ctx.optix_gpu_ptr) {
                                ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects);
                                ctx.optix_gpu_ptr->resetAccumulation();
                            }
                            sel.selected.has_cached_aabb = false;
                        }
                    }
                    ImGui::PopItemWidth();

                    if (!kv.second.empty()) {
                        int matID = kv.second[0].second->getMaterialID();
                        ImGui::PushItemWidth(100);
                        if (ImGui::InputInt("Mat ID", &matID)) {
                            for (auto& pair : kv.second) pair.second->setMaterialID(matID);
                            if (ctx.optix_gpu_ptr) {
                                ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects);
                                ctx.optix_gpu_ptr->resetAccumulation();
                            }
                            ctx.renderer.resetCPUAccumulation();
                        }
                        ImGui::PopItemWidth();
                    }

                    ImGui::Unindent();
                    ImGui::PopStyleColor();
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }
        ImGui::TreePop();
    }

    ImGui::EndChild(); // End HierarchyTree

    // ─────────────────────────────────────────────────────────────────────────
    // Selection Properties (Compact - Camera/Light only)
    // ─────────────────────────────────────────────────────────────────────────
    if (sel.hasSelection()) {
        if (ImGui::CollapsingHeader("Selection Properties", ImGuiTreeNodeFlags_DefaultOpen)) {

            // Header with type and name + delete button
            const char* typeIcon = "[?]";
            ImVec4 typeColor = ImVec4(1, 1, 1, 1);
            switch (sel.selected.type) {
            case SelectableType::Camera: typeIcon = "[CAM]"; typeColor = ImVec4(0.4f, 0.8f, 1.0f, 1.0f); break;
            case SelectableType::Light: typeIcon = "[*]"; typeColor = ImVec4(1.0f, 0.9f, 0.4f, 1.0f); break;
            case SelectableType::Object: typeIcon = "[M]"; typeColor = ImVec4(0.7f, 0.8f, 0.9f, 1.0f); break;
            case SelectableType::World: typeIcon = "[W]"; typeColor = ImVec4(1.0f, 0.7f, 1.0f, 1.0f); break;
            default: break;
            }
            ImGui::TextColored(typeColor, "%s %s", typeIcon, sel.selected.name.c_str());

            // Gizmo and Delete not applicable for World (environment can't be transformed or deleted)
            if (sel.selected.type != SelectableType::World) {
                ImGui::Checkbox("Gizmo", &sel.show_gizmo);
                ImGui::SameLine(ImGui::GetWindowWidth() - 55);
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
                if (ImGui::SmallButton("Del")) {
                    bool deleted = false;
                    if (sel.selected.type == SelectableType::Object && sel.selected.object) {
                        std::string targetName = sel.selected.object->nodeName;
                        auto& objs = ctx.scene.world.objects;
                        auto new_end = std::remove_if(objs.begin(), objs.end(),
                            [&](const std::shared_ptr<Hittable>& obj) {
                                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                return tri && tri->nodeName == targetName;
                            });
                        if (new_end != objs.end()) {
                            objs.erase(new_end, objs.end());
                            deleted = true;
                        }
                    }
                    else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
                        auto light_to_delete = sel.selected.light;
                        auto& lights = ctx.scene.lights;
                        auto it = std::find(lights.begin(), lights.end(), light_to_delete);
                        if (it != lights.end()) {
                            history.record(std::make_unique<DeleteLightCommand>(light_to_delete));
                            lights.erase(it);
                            deleted = true;
                        }
                    }
                    else if (sel.selected.type == SelectableType::Camera && sel.selected.camera) {
                        // Camera deletion with safety checks
                        if (ctx.scene.cameras.size() <= 1) {
                            SCENE_LOG_WARN("Cannot delete the last camera in scene");
                        }
                        else if (sel.selected.camera == ctx.scene.camera) {
                            SCENE_LOG_WARN("Cannot delete the active camera. Switch to another camera first.");
                        }
                        else {
                            if (ctx.scene.removeCamera(sel.selected.camera)) {
                                deleted = true;
                                SCENE_LOG_INFO("Camera deleted successfully");
                            }
                        }
                    }
                    if (deleted) {
                        sel.clearSelection();
                        invalidateCache();
                        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                        ctx.renderer.resetCPUAccumulation();
                        if (ctx.optix_gpu_ptr) {
                            ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                            ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                            if (ctx.scene.camera) ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                        is_bvh_dirty = false;
                    }
                    else {
                        sel.clearSelection();
                    }
                }
                ImGui::PopStyleColor();
            } // End World check block




            // Custom Keyframe Buttons Helper (Draws Diamond Shape Manually)
            auto KeyframeButton = [&](const char* id, bool keyed) -> bool {
                ImGui::PushID(id);
                float s = ImGui::GetFrameHeight();
                ImVec2 pos = ImGui::GetCursorScreenPos();
                bool clicked = ImGui::InvisibleButton("kbtn", ImVec2(s, s));

                ImU32 bg = keyed ? IM_COL32(255, 200, 0, 255) : IM_COL32(40, 40, 40, 255);
                ImU32 border = IM_COL32(180, 180, 180, 255);

                if (ImGui::IsItemHovered()) {
                    border = IM_COL32(255, 255, 255, 255);
                    bg = keyed ? IM_COL32(255, 220, 50, 255) : IM_COL32(70, 70, 70, 255);
                }

                ImDrawList* dl = ImGui::GetWindowDrawList();
                float cx = pos.x + s * 0.5f;
                float cy = pos.y + s * 0.5f;
                float r = s * 0.22f;

                ImVec2 p[4] = {
                    ImVec2(cx, cy - r),
                    ImVec2(cx + r, cy),
                    ImVec2(cx, cy + r),
                    ImVec2(cx - r, cy)
                };

                dl->AddQuadFilled(p[0], p[1], p[2], p[3], bg);
                dl->AddQuad(p[0], p[1], p[2], p[3], border, 1.0f);

                ImGui::PopID();
                return clicked;
                };

            // CAMERA PROPERTIES
            if (sel.selected.type == SelectableType::Camera && sel.selected.camera) {
                auto& cam = *sel.selected.camera;
                ImGui::Separator();

                // Helper lambda for camera per-property keyframes
                auto insertCamKey = [&](const std::string& prop_name,
                    bool key_pos, bool key_target, bool key_fov, bool key_focus, bool key_aperture) {
                        int current_frame = ctx.render_settings.animation_current_frame;
                        std::string cam_name = cam.nodeName.empty() ? "Camera" : cam.nodeName;
                        if (cam.nodeName.empty()) cam.nodeName = cam_name;

                        auto& track = ctx.scene.timeline.tracks[cam_name];

                        // TOGGLE BEHAVIOR: Check if property is already keyed, if so remove it
                        for (auto it = track.keyframes.begin(); it != track.keyframes.end(); ++it) {
                            if (it->frame == current_frame && it->has_camera) {
                                bool removed = false;
                                if (key_pos && it->camera.has_position) { it->camera.has_position = false; removed = true; }
                                if (key_target && it->camera.has_target) { it->camera.has_target = false; removed = true; }
                                if (key_fov && it->camera.has_fov) { it->camera.has_fov = false; removed = true; }
                                if (key_focus && it->camera.has_focus) { it->camera.has_focus = false; removed = true; }
                                if (key_aperture && it->camera.has_aperture) { it->camera.has_aperture = false; removed = true; }

                                if (removed) {
                                    // Check if keyframe is now empty
                                    bool hasAny = it->camera.has_position || it->camera.has_target ||
                                        it->camera.has_fov || it->camera.has_focus || it->camera.has_aperture;
                                    if (!hasAny) {
                                        it->has_camera = false;
                                        if (!it->has_transform && !it->has_light && !it->has_material && !it->has_world) {
                                            track.keyframes.erase(it);
                                        }
                                    }
                                    SCENE_LOG_INFO("Removed " + prop_name + " keyframe at frame " + std::to_string(current_frame));
                                    return;
                                }
                            }
                        }

                        // Not keyed - add new keyframe
                        Keyframe kf(current_frame);
                        kf.has_camera = true;
                        kf.camera.has_position = key_pos;
                        kf.camera.has_target = key_target;
                        kf.camera.has_fov = key_fov;
                        kf.camera.has_focus = key_focus;
                        kf.camera.has_aperture = key_aperture;
                        kf.camera.position = cam.lookfrom;
                        kf.camera.target = cam.lookat;
                        kf.camera.fov = (float)cam.vfov;
                        kf.camera.focus_distance = cam.focus_dist;
                        kf.camera.lens_radius = cam.lens_radius;

                        bool found = false;
                        for (auto& existing : track.keyframes) {
                            if (existing.frame == current_frame) {
                                existing.has_camera = true;
                                if (key_pos) { existing.camera.has_position = true; existing.camera.position = kf.camera.position; }
                                if (key_target) { existing.camera.has_target = true; existing.camera.target = kf.camera.target; }
                                if (key_fov) { existing.camera.has_fov = true; existing.camera.fov = kf.camera.fov; }
                                if (key_focus) { existing.camera.has_focus = true; existing.camera.focus_distance = kf.camera.focus_distance; }
                                if (key_aperture) { existing.camera.has_aperture = true; existing.camera.lens_radius = kf.camera.lens_radius; }
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            track.keyframes.push_back(kf);
                            std::sort(track.keyframes.begin(), track.keyframes.end(),
                                [](const Keyframe& a, const Keyframe& b) { return a.frame < b.frame; });
                        }
                        SCENE_LOG_INFO("Added " + prop_name + " keyframe at frame " + std::to_string(current_frame));
                    };

                // Check if camera property is keyed at current frame
                auto isCamKeyed = [&](bool check_pos, bool check_target, bool check_fov, bool check_focus, bool check_aperture) -> bool {
                    std::string cam_name = cam.nodeName.empty() ? "Camera" : cam.nodeName;
                    auto it = ctx.scene.timeline.tracks.find(cam_name);
                    if (it == ctx.scene.timeline.tracks.end()) return false;
                    int cf = ctx.render_settings.animation_current_frame;
                    for (auto& kf : it->second.keyframes) {
                        if (kf.frame == cf && kf.has_camera) {
                            if (check_pos && kf.camera.has_position) return true;
                            if (check_target && kf.camera.has_target) return true;
                            if (check_fov && kf.camera.has_fov) return true;
                            if (check_focus && kf.camera.has_focus) return true;
                            if (check_aperture && kf.camera.has_aperture) return true;
                        }
                    }
                    return false;
                    };

                // Position with ◇ key button
                Vec3 pos = cam.lookfrom;
                bool posKeyed = isCamKeyed(true, false, false, false, false);
                if (KeyframeButton("##CPos", posKeyed)) { insertCamKey("Position", true, false, false, false, false); }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(posKeyed ? "REMOVE Position key" : "ADD Position key");
                ImGui::SameLine();
                if (ImGui::DragFloat3("Position##cam", &pos.x, 0.1f)) {
                    Vec3 delta = pos - cam.lookfrom;
                    cam.lookfrom = pos;
                    cam.lookat = cam.lookat + delta;
                    cam.update_camera_vectors();
                    sel.selected.position = pos;
                    if (ctx.optix_gpu_ptr && g_hasOptix) {
                        ctx.optix_gpu_ptr->setCameraParams(cam);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                }

                // Target with ◇ key button
                Vec3 target = cam.lookat;
                bool targetKeyed = isCamKeyed(false, true, false, false, false);
                if (KeyframeButton("##CTgt", targetKeyed)) { insertCamKey("Target", false, true, false, false, false); }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(targetKeyed ? "REMOVE Target key" : "ADD Target key");
                ImGui::SameLine();
                if (ImGui::DragFloat3("Target", &target.x, 0.1f)) {
                    cam.lookat = target;
                    cam.update_camera_vectors();
                }

                ImGui::Separator();

                // ═══════════════════════════════════════════════════════════════════════════
                // CAMERA BODY - Sensor size affects crop factor
                // ═══════════════════════════════════════════════════════════════════════════
                static int selected_body = 0;

                if (ImGui::BeginCombo("Camera Body", CameraPresets::CAMERA_BODIES[selected_body].name)) {
                    for (size_t i = 0; i < CameraPresets::CAMERA_BODY_COUNT; ++i) {
                        bool is_selected = (selected_body == (int)i);
                        std::string label = std::string(CameraPresets::CAMERA_BODIES[i].name);
                        if (i > 0) {
                            label += " (" + std::string(CameraPresets::getSensorTypeName(CameraPresets::CAMERA_BODIES[i].sensor)) + ")";
                        }
                        if (ImGui::Selectable(label.c_str(), is_selected)) {
                            selected_body = (int)i;
                        }
                        if (is_selected) ImGui::SetItemDefaultFocus();
                        if (ImGui::IsItemHovered() && i > 0) {
                            ImGui::SetTooltip("%s - %s\nCrop: %.2fx",
                                CameraPresets::CAMERA_BODIES[i].brand,
                                CameraPresets::CAMERA_BODIES[i].description,
                                CameraPresets::CAMERA_BODIES[i].crop_factor);
                        }
                    }
                    ImGui::EndCombo();
                }
                if (selected_body > 0) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "%.1fx crop",
                        CameraPresets::CAMERA_BODIES[selected_body].crop_factor);
                }

                // ═══════════════════════════════════════════════════════════════════════════
                // LENS PRESETS - Photographer-friendly focal length selection
                // ═══════════════════════════════════════════════════════════════════════════
                struct LensPreset {
                    const char* name;
                    float focal_mm;
                    float fov;
                    const char* description;
                };
                static const LensPreset lens_presets[] = {
                    { "Custom",          0.0f,   0.0f, "Manual FOV control" },
                    { "16mm Ultra Wide", 16.0f,  97.0f, "Extreme wide angle - landscapes, architecture" },
                    { "24mm Wide",       24.0f,  84.0f, "Wide angle - interiors, groups" },
                    { "35mm Standard",   35.0f,  63.0f, "Standard wide - street, documentary" },
                    { "50mm Normal",     50.0f,  46.8f, "Human eye perspective - general use" },
                    { "85mm Portrait",   85.0f,  28.6f, "Portrait lens - headshots, bokeh" },
                    { "135mm Telephoto", 135.0f, 18.0f, "Telephoto - portrait, compression" },
                    { "200mm Tele",      200.0f, 12.0f, "Long telephoto - wildlife, sports" },
                };
                static int selected_lens = 0;

                float fov = (float)cam.vfov;

                // Find current preset based on FOV
                selected_lens = 0;
                for (int i = 1; i < IM_ARRAYSIZE(lens_presets); i++) {
                    if (std::abs(fov - lens_presets[i].fov) < 1.0f) {
                        selected_lens = i;
                        break;
                    }
                }

                if (ImGui::Combo("Lens Preset", &selected_lens,
                    [](void* data, int idx, const char** out_text) {
                        *out_text = ((LensPreset*)data)[idx].name;
                        return true;
                    }, (void*)lens_presets, IM_ARRAYSIZE(lens_presets)))
                {
                    if (selected_lens > 0) {
                        fov = lens_presets[selected_lens].fov;
                        cam.vfov = fov;
                        cam.fov = fov;
                        cam.update_camera_vectors();
                        if (ctx.optix_gpu_ptr && g_hasOptix) {
                            ctx.optix_gpu_ptr->setCameraParams(cam);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                    }
                }
                if (selected_lens > 0 && ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", lens_presets[selected_lens].description);
                }

                // FOV with ◇ key button
                bool fovKeyed = isCamKeyed(false, false, true, false, false);
                if (KeyframeButton("##CFOV", fovKeyed)) { insertCamKey("FOV", false, false, true, false, false); }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(fovKeyed ? "REMOVE FOV key" : "ADD FOV key");
                ImGui::SameLine();
                if (ImGui::SliderFloat("FOV", &fov, 10.0f, 120.0f)) {
                    cam.vfov = fov;
                    cam.fov = fov;
                    cam.update_camera_vectors();
                    selected_lens = 0; // Reset to Custom
                    if (ctx.optix_gpu_ptr && g_hasOptix) {
                        ctx.optix_gpu_ptr->setCameraParams(cam);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                }

                ImGui::Separator();

                // ═══════════════════════════════════════════════════════════════════════════
                // F-STOP PRESETS - Photographer-friendly aperture selection
                // ═══════════════════════════════════════════════════════════════════════════
                // Find current preset based on index or aperture match if index is default
                // Better to rely on index if it's set properly, or find match
                if (cam.fstop_preset_index == 0 && cam.aperture > 0.01f) {
                    for (size_t i = 1; i < CameraPresets::FSTOP_PRESET_COUNT; i++) {
                        if (std::abs(cam.aperture - CameraPresets::FSTOP_PRESETS[i].aperture_value) < 0.05f) {
                            cam.fstop_preset_index = (int)i;
                            break;
                        }
                    }
                }

                if (ImGui::BeginCombo("F-Stop", CameraPresets::FSTOP_PRESETS[cam.fstop_preset_index].name)) {
                    for (size_t i = 0; i < CameraPresets::FSTOP_PRESET_COUNT; ++i) {
                        bool is_selected = (cam.fstop_preset_index == (int)i);
                        if (ImGui::Selectable(CameraPresets::FSTOP_PRESETS[i].name, is_selected)) {
                            cam.fstop_preset_index = (int)i;

                            // Update physical aperture based on preset
                            if (cam.fstop_preset_index > 0) {
                                cam.aperture = CameraPresets::FSTOP_PRESETS[i].aperture_value;
                                cam.lens_radius = cam.aperture * 0.5f;
                            }
                            cam.update_camera_vectors();

                            // Trigger update
                            if (ctx.optix_gpu_ptr && g_hasOptix) {
                                ctx.optix_gpu_ptr->setCameraParams(cam);
                                ctx.optix_gpu_ptr->resetAccumulation();
                            }
                            ctx.renderer.resetCPUAccumulation();
                        }
                        if (is_selected) ImGui::SetItemDefaultFocus();
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", CameraPresets::FSTOP_PRESETS[i].description);
                    }
                    ImGui::EndCombo();
                }
                if (cam.fstop_preset_index > 0 && ImGui::IsItemHovered()) {
                    // Main combo tooltip if needed, or leave blank
                }

                // DOF Settings - Aperture with ◇ key button
                bool apKeyed = isCamKeyed(false, false, false, false, true);
                if (KeyframeButton("##CAp", apKeyed)) { insertCamKey("Aperture", false, false, false, false, true); }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(apKeyed ? "REMOVE Aperture key" : "ADD Aperture key");
                ImGui::SameLine();
                if (ImGui::SliderFloat("Aperture", &cam.aperture, 0.0f, 5.0f)) {
                    cam.fstop_preset_index = 0; // Reset to Custom
                    cam.lens_radius = cam.aperture * 0.5f;

                    if (ctx.optix_gpu_ptr && g_hasOptix) {
                        ctx.optix_gpu_ptr->setCameraParams(cam);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                }

                // Lens & Aperture Shape Settings
                ImGui::Separator();
                if (ImGui::TreeNodeEx("Lens & Bokeh", ImGuiTreeNodeFlags_DefaultOpen)) {
                    // 1. Blade Count (Bokeh Shape)
                    bool blades_changed = false;
                    if (ImGui::SliderInt("Aperture Blades", &cam.blade_count, 0, 9, "%d blades")) {
                        blades_changed = true;
                    }
                    if (cam.blade_count < 3) ImGui::TextDisabled("Shape: Round (Modern / Perfect)");
                    else ImGui::TextDisabled("Shape: %d-sided Polygon (Vintage / Cinema)", cam.blade_count);

                    if (blades_changed) {
                        if (ctx.optix_gpu_ptr && g_hasOptix) {
                            ctx.optix_gpu_ptr->setCameraParams(cam);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                        ctx.renderer.resetCPUAccumulation();
                    }

                    // 2. Physical Lens Settings (mm)
                    bool lens_changed = false;
                    if (ImGui::Checkbox("Physical Lens (mm)", &cam.use_physical_lens)) {
                        lens_changed = true;
                    }

                    if (cam.use_physical_lens) {
                        // Focal Length
                        if (ImGui::SliderFloat("Focal Length", &cam.focal_length_mm, 10.0f, 200.0f, "%.1f mm")) {
                            lens_changed = true;
                        }

                        // Sensor Size presets
                        const char* sensor_names[] = { "Full Frame (35mm)", "Super 35", "APS-C", "Micro 4/3", "IMAX 65mm" };
                        static int sensor_idx = 0;
                        // Simple sync check
                        if (std::abs(cam.sensor_width_mm - 36.0f) < 0.1f) sensor_idx = 0;
                        else if (std::abs(cam.sensor_width_mm - 24.89f) < 0.1f) sensor_idx = 1;

                        if (ImGui::Combo("Sensor Size", &sensor_idx, sensor_names, IM_ARRAYSIZE(sensor_names))) {
                            if (sensor_idx == 0) { cam.sensor_width_mm = 36.0f; cam.sensor_height_mm = 24.0f; }
                            else if (sensor_idx == 1) { cam.sensor_width_mm = 24.89f; cam.sensor_height_mm = 18.66f; } // Approx S35
                            else if (sensor_idx == 2) { cam.sensor_width_mm = 22.2f; cam.sensor_height_mm = 14.8f; } // Canon APS-C
                            else if (sensor_idx == 3) { cam.sensor_width_mm = 17.3f; cam.sensor_height_mm = 13.0f; } // MFT
                            else if (sensor_idx == 4) { cam.sensor_width_mm = 70.0f; cam.sensor_height_mm = 48.5f; } // IMAX
                            lens_changed = true;
                        }

                        ImGui::TextDisabled("Sensor: %.1fx%.1f mm", cam.sensor_width_mm, cam.sensor_height_mm);
                    }

                    // Motion Blur Toggle
                    ImGui::Separator();
                    bool mb_changed = false;
                    if (ImGui::Checkbox("Motion Blur (Camera)", &cam.enable_motion_blur)) {
                        mb_changed = true;
                    }
                    if (cam.enable_motion_blur) {
                        ImGui::SameLine();
                        ImGui::TextDisabled("(Uses Shutter Speed)");
                    }

                    if (mb_changed) {
                        if (ctx.optix_gpu_ptr && g_hasOptix) {
                            ctx.optix_gpu_ptr->setCameraParams(cam);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                        ctx.renderer.resetCPUAccumulation();
                    }

                    // Logic to update FOV from Lens settings
                    if (lens_changed && cam.use_physical_lens) {
                        // FOV = 2 * atan(SensorHeight / (2 * FocalLength))
                        // We use Sensor Height for VFOV to match standard vertical FOV usage in ray tracer
                        float vfov_rad = 2.0f * std::atan((cam.sensor_height_mm * 0.5f) / cam.focal_length_mm);
                        cam.vfov = vfov_rad * (180.0f / 3.14159265f);
                        cam.fov = cam.vfov;

                        cam.update_camera_vectors();
                        if (ctx.optix_gpu_ptr && g_hasOptix) {
                            ctx.optix_gpu_ptr->setCameraParams(cam);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                        ctx.renderer.resetCPUAccumulation();
                    }
                    ImGui::TreePop();
                }

                // Focus Distance with ◇ key button and Pick Focus button
                bool focKeyed = isCamKeyed(false, false, false, true, false);
                if (KeyframeButton("##CFoc", focKeyed)) { insertCamKey("Focus", false, false, false, true, false); }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(focKeyed ? "REMOVE Focus key" : "ADD Focus key");
                ImGui::SameLine();
                ImGui::DragFloat("Focus Dist", &cam.focus_dist, 0.1f, 0.01f, 100.0f);

                // Pick Focus mode - sets focus to clicked object distance
                ImGui::SameLine();
                if (is_picking_focus) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.8f, 0.4f, 0.1f, 1.0f));
                }
                if (ImGui::Button(is_picking_focus ? "Picking..." : "Pick")) {
                    is_picking_focus = !is_picking_focus;
                }
                if (is_picking_focus) {
                    ImGui::PopStyleColor();
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Click on an object in viewport to set focus distance (ignores selection)");
                }

                // Mouse Sensitivity
                ImGui::Spacing();
                if (ImGui::SliderFloat("Mouse Sensitivity", &ctx.mouse_sensitivity, 0.01f, 5.0f, "%.3f")) {
                    // Value updated directly via reference
                }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Camera rotation/panning speed");

                // ═══════════════════════════════════════════════════════════════════════════
                // EXPOSURE SETTINGS - Professional camera exposure controls
                // ═══════════════════════════════════════════════════════════════════════════
                ImGui::Separator();
                if (ImGui::CollapsingHeader("Exposure", ImGuiTreeNodeFlags_DefaultOpen)) {
                    bool exposure_changed = false;

                    // Auto Exposure Toggle
                    if (ImGui::Checkbox("Auto Exposure", &cam.auto_exposure)) {
                        exposure_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Automatically adjust exposure based on scene brightness");

                    ImGui::Spacing();

                    // ISO Preset
                    ImGui::PushItemWidth(180);
                    if (ImGui::BeginCombo("ISO", CameraPresets::ISO_PRESETS[cam.iso_preset_index].name)) {
                        for (size_t i = 0; i < CameraPresets::ISO_PRESET_COUNT; ++i) {
                            bool is_selected = (cam.iso_preset_index == (int)i);
                            if (ImGui::Selectable(CameraPresets::ISO_PRESETS[i].name, is_selected)) {
                                cam.iso_preset_index = (int)i;
                                exposure_changed = true;
                            }
                            if (is_selected) ImGui::SetItemDefaultFocus();
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("%s\nNoise: %.0f%%",
                                    CameraPresets::ISO_PRESETS[i].description,
                                    CameraPresets::ISO_PRESETS[i].noise_factor * 100.0f);
                            }
                        }
                        ImGui::EndCombo();
                    }

                    // Shutter Speed Preset  
                    if (ImGui::BeginCombo("Shutter", CameraPresets::SHUTTER_SPEED_PRESETS[cam.shutter_preset_index].name)) {
                        for (size_t i = 0; i < CameraPresets::SHUTTER_SPEED_PRESET_COUNT; ++i) {
                            bool is_selected = (cam.shutter_preset_index == (int)i);
                            if (ImGui::Selectable(CameraPresets::SHUTTER_SPEED_PRESETS[i].name, is_selected)) {
                                cam.shutter_preset_index = (int)i;
                                exposure_changed = true;
                            }
                            if (is_selected) ImGui::SetItemDefaultFocus();
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip("%s\nMotion Blur: %.0f%%",
                                    CameraPresets::SHUTTER_SPEED_PRESETS[i].description,
                                    CameraPresets::SHUTTER_SPEED_PRESETS[i].motion_blur_factor * 100.0f);
                            }
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::PopItemWidth();

                    // EV Compensation
                    ImGui::PushItemWidth(150);
                    if (ImGui::SliderFloat("EV Comp", &cam.ev_compensation, -2.0f, 2.0f, "%+.1f EV")) {
                        exposure_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Exposure Value compensation");
                    ImGui::PopItemWidth();

                    // Trigger Update if changed
                    if (exposure_changed) {
                        if (ctx.optix_gpu_ptr && g_hasOptix) {
                            ctx.optix_gpu_ptr->setCameraParams(cam);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                        ctx.renderer.resetCPUAccumulation();
                    }

                    // Calculate and show exposure info
                    float iso_mult = CameraPresets::ISO_PRESETS[cam.iso_preset_index].exposure_multiplier;
                    float shutter_time = CameraPresets::SHUTTER_SPEED_PRESETS[cam.shutter_preset_index].speed_seconds; // Corrected member name
                    float aperture_area = cam.aperture * cam.aperture;
                    cam.calculated_ev = log2f(100.0f / (iso_mult * shutter_time * aperture_area + 0.001f)) + cam.ev_compensation;

                    ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.7f, 1.0f), "Exposure: %.1f EV", cam.calculated_ev);

                    if (cam.auto_exposure) {
                        ImGui::SameLine();
                        ImGui::TextDisabled("(Auto Active)");
                    }
                }

                // ═══════════════════════════════════════════════════════════════════════════
                // OUTPUT ASPECT RATIO - Syncs with Final Render
                // ═══════════════════════════════════════════════════════════════════════════
                ImGui::Separator();
                if (ImGui::CollapsingHeader("Output Aspect Ratio")) {
                    ImGui::PushItemWidth(200);
                    if (ImGui::BeginCombo("Aspect##CamOutput",
                        CameraPresets::ASPECT_RATIOS[cam.output_aspect_index].name))
                    {
                        for (size_t i = 0; i < CameraPresets::ASPECT_RATIO_COUNT; ++i) {
                            bool is_selected = (cam.output_aspect_index == (int)i);
                            std::string label = std::string(CameraPresets::ASPECT_RATIOS[i].name) +
                                " - " + CameraPresets::ASPECT_RATIOS[i].usage;
                            if (ImGui::Selectable(label.c_str(), is_selected)) {
                                cam.output_aspect_index = (int)i;
                                // Sync with Final Render settings
                                ctx.render_settings.aspect_ratio_index = cam.output_aspect_index;
                                // Also sync to viewport guide letterbox
                                guide_settings.aspect_ratio_index = cam.output_aspect_index;
                            }
                            if (is_selected) ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    ImGui::PopItemWidth();

                    // Show calculated resolution preview
                    float ratio = CameraPresets::ASPECT_RATIOS[cam.output_aspect_index].ratio;
                    if (ratio > 0.01f) {
                        int base_h = ctx.render_settings.aspect_base_height;
                        int calc_w = (int)(base_h * ratio);
                        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f),
                            "Final Render: %d x %d (base %dp)", calc_w, base_h, base_h);
                    }
                    ImGui::TextDisabled("This syncs with Final Render aspect ratio");
                }


                // Set as Active Camera button
                bool is_active = (ctx.scene.camera.get() == &cam);
                if (!is_active) {
                    if (ImGui::Button("Set as Active Camera")) {
                        // Find camera index
                        for (size_t i = 0; i < ctx.scene.cameras.size(); ++i) {
                            if (ctx.scene.cameras[i].get() == &cam) {
                                ctx.scene.setActiveCamera(i);
                                if (ctx.optix_gpu_ptr) {
                                    ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                                    ctx.optix_gpu_ptr->resetAccumulation();
                                }
                                ctx.renderer.resetCPUAccumulation();
                                break;
                            }
                        }
                    }
                    ImGui::SameLine();
                }
                else {
                    ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1.0f), "[Active]");
                    ImGui::SameLine();
                }

                // Reset Camera button
                if (ImGui::Button("Reset Camera")) {
                    cam.reset();
                    sel.selected.position = cam.lookfrom;
                    if (ctx.optix_gpu_ptr && g_hasOptix) {
                        ctx.optix_gpu_ptr->setCameraParams(cam);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                    ctx.start_render = true;
                }

                // ═══════════════════════════════════════════════════════════════════════════
                // VIEWPORT GUIDES (Safe Areas, Letterbox, Grids)
                // ═══════════════════════════════════════════════════════════════════════════
                ImGui::Separator();
                if (ImGui::CollapsingHeader("Viewport Guides")) {
                    // Safe Areas
                    ImGui::Checkbox("Safe Areas", &guide_settings.show_safe_areas);
                    if (guide_settings.show_safe_areas) {
                        ImGui::Indent();
                        const char* safe_types[] = { "Both", "Title Safe", "Action Safe" };
                        ImGui::Combo("Type##SafeArea", &guide_settings.safe_area_type, safe_types, 3);
                        ImGui::Unindent();
                    }

                    // Letterbox / Aspect Ratio
                    ImGui::Checkbox("Aspect Ratio Overlay", &guide_settings.show_letterbox);
                    if (guide_settings.show_letterbox) {
                        ImGui::Indent();
                        if (ImGui::BeginCombo("Aspect##Letterbox",
                            CameraPresets::ASPECT_RATIOS[guide_settings.aspect_ratio_index].name))
                        {
                            for (size_t i = 0; i < CameraPresets::ASPECT_RATIO_COUNT; ++i) {
                                bool is_selected = (guide_settings.aspect_ratio_index == (int)i);
                                if (ImGui::Selectable(CameraPresets::ASPECT_RATIOS[i].name, is_selected)) {
                                    guide_settings.aspect_ratio_index = (int)i;
                                }
                                if (is_selected) ImGui::SetItemDefaultFocus();
                            }
                            ImGui::EndCombo();
                        }
                        ImGui::SliderFloat("Opacity##Letterbox", &guide_settings.letterbox_opacity, 0.0f, 1.0f);
                        ImGui::Unindent();
                    }

                    // Composition Grid
                    ImGui::Checkbox("Composition Grid", &guide_settings.show_grid);
                    if (guide_settings.show_grid) {
                        ImGui::Indent();
                        const char* grid_types[] = { "Rule of Thirds", "Golden Ratio", "Center Cross", "Diagonal" };
                        ImGui::Combo("Grid##Comp", &guide_settings.grid_type, grid_types, 4);
                        ImGui::Unindent();
                    }

                    // Center Crosshair
                    ImGui::Checkbox("Center Crosshair", &guide_settings.show_center);
                }
            }

            // LIGHT PROPERTIES
            else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
                auto& light = *sel.selected.light;
                bool light_changed = false;
                ImGui::Separator();

                // Helper lambda for light per-property keyframes
                auto insertLightKey = [&](const std::string& prop_name,
                    bool key_pos, bool key_color, bool key_intensity, bool key_dir) {
                        int current_frame = ctx.render_settings.animation_current_frame;
                        std::string light_name = light.nodeName.empty() ? "Light_" + std::to_string(sel.selected.light_index) : light.nodeName;
                        if (light.nodeName.empty()) light.nodeName = light_name;

                        auto& track = ctx.scene.timeline.tracks[light_name];

                        // TOGGLE BEHAVIOR: Check if property is already keyed, if so remove it
                        for (auto it = track.keyframes.begin(); it != track.keyframes.end(); ++it) {
                            if (it->frame == current_frame && it->has_light) {
                                bool removed = false;
                                if (key_pos && it->light.has_position) { it->light.has_position = false; removed = true; }
                                if (key_color && it->light.has_color) { it->light.has_color = false; removed = true; }
                                if (key_intensity && it->light.has_intensity) { it->light.has_intensity = false; removed = true; }
                                if (key_dir && it->light.has_direction) { it->light.has_direction = false; removed = true; }

                                if (removed) {
                                    // Check if keyframe is now empty
                                    bool hasAny = it->light.has_position || it->light.has_color ||
                                        it->light.has_intensity || it->light.has_direction;
                                    if (!hasAny) {
                                        it->has_light = false;
                                        if (!it->has_transform && !it->has_camera && !it->has_material && !it->has_world) {
                                            track.keyframes.erase(it);
                                        }
                                    }
                                    SCENE_LOG_INFO("Removed Light " + prop_name + " keyframe at frame " + std::to_string(current_frame));
                                    return;
                                }
                            }
                        }

                        // Not keyed - add new keyframe
                        Keyframe kf(current_frame);
                        kf.has_light = true;
                        kf.light.has_position = key_pos;
                        kf.light.has_color = key_color;
                        kf.light.has_intensity = key_intensity;
                        kf.light.has_direction = key_dir;
                        kf.light.position = light.position;
                        kf.light.color = light.color;
                        kf.light.intensity = light.intensity;
                        kf.light.direction = light.direction;

                        bool found = false;
                        for (auto& existing : track.keyframes) {
                            if (existing.frame == current_frame) {
                                existing.has_light = true;
                                if (key_pos) { existing.light.has_position = true; existing.light.position = kf.light.position; }
                                if (key_color) { existing.light.has_color = true; existing.light.color = kf.light.color; }
                                if (key_intensity) { existing.light.has_intensity = true; existing.light.intensity = kf.light.intensity; }
                                if (key_dir) { existing.light.has_direction = true; existing.light.direction = kf.light.direction; }
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            track.keyframes.push_back(kf);
                            std::sort(track.keyframes.begin(), track.keyframes.end(),
                                [](const Keyframe& a, const Keyframe& b) { return a.frame < b.frame; });
                        }
                        SCENE_LOG_INFO("Added Light " + prop_name + " keyframe at frame " + std::to_string(current_frame));
                    };

                // Check if property is keyed at current frame
                auto isLightKeyed = [&](bool check_pos, bool check_color, bool check_int, bool check_dir) -> bool {
                    std::string light_name = light.nodeName.empty() ? "Light_" + std::to_string(sel.selected.light_index) : light.nodeName;
                    auto it = ctx.scene.timeline.tracks.find(light_name);
                    if (it == ctx.scene.timeline.tracks.end()) return false;
                    int cf = ctx.render_settings.animation_current_frame;
                    for (auto& kf : it->second.keyframes) {
                        if (kf.frame == cf && kf.has_light) {
                            if (check_pos && kf.light.has_position) return true;
                            if (check_color && kf.light.has_color) return true;
                            if (check_int && kf.light.has_intensity) return true;
                            if (check_dir && kf.light.has_direction) return true;
                        }
                    }
                    return false;
                    };

                const char* lightTypes[] = { "Point", "Directional", "Spot", "Area" };
                int typeIdx = (int)light.type();
                if (typeIdx >= 0 && typeIdx < 4) ImGui::TextDisabled("Type: %s", lightTypes[typeIdx]);

                // Position with ◇ key button
                bool posKeyed = isLightKeyed(true, false, false, false);
                if (KeyframeButton("##LPos", posKeyed)) { insertLightKey("Position", true, false, false, false); }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(posKeyed ? "REMOVE Position key" : "ADD Position key");
                ImGui::SameLine();
                if (ImGui::DragFloat3("Position##light", &light.position.x, 0.1f)) {
                    sel.selected.position = light.position;
                    light_changed = true;
                }

                // Direction with ◇ key button (only for Directional/Spot)
                if (light.type() == LightType::Directional || light.type() == LightType::Spot) {
                    bool dirKeyed = isLightKeyed(false, false, false, true);
                    if (KeyframeButton("##LDir", dirKeyed)) { insertLightKey("Direction", false, false, false, true); }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip(dirKeyed ? "REMOVE Direction key" : "ADD Direction key");
                    ImGui::SameLine();
                    if (ImGui::DragFloat3("Direction", &light.direction.x, 0.01f)) light_changed = true;
                }

                // Color with ◇ key button
                bool colKeyed = isLightKeyed(false, true, false, false);
                if (KeyframeButton("##LCol", colKeyed)) { insertLightKey("Color", false, true, false, false); }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(colKeyed ? "REMOVE Color key" : "ADD Color key");
                ImGui::SameLine();
                if (ImGui::ColorEdit3("Color", &light.color.x)) light_changed = true;

                // Intensity with ◇ key button
                bool intKeyed = isLightKeyed(false, false, true, false);
                if (KeyframeButton("##LInt", intKeyed)) { insertLightKey("Intensity", false, false, true, false); }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(intKeyed ? "REMOVE Intensity key" : "ADD Intensity key");
                ImGui::SameLine();
                if (ImGui::DragFloat("Intensity", &light.intensity, 0.5f, 0.0f, 1000.0f)) light_changed = true;

                if (light.type() == LightType::Point || light.type() == LightType::Directional) {
                    if (ImGui::DragFloat("Radius", &light.radius, 0.01f, 0.01f, 100.0f)) light_changed = true;
                }

                if (auto sl = dynamic_cast<SpotLight*>(&light)) {
                    float angle = sl->getAngleDegrees();
                    if (ImGui::DragFloat("Cone Angle", &angle, 0.5f, 1.0f, 89.0f)) {
                        sl->setAngleDegrees(angle);
                        light_changed = true;
                    }
                }
                else if (auto al = dynamic_cast<AreaLight*>(&light)) {
                    if (ImGui::DragFloat("Width", &al->width, 0.05f, 0.01f, 100.0f)) {
                        al->u = al->u.normalize() * al->width;
                        light_changed = true;
                    }
                    if (ImGui::DragFloat("Height", &al->height, 0.05f, 0.01f, 100.0f)) {
                        al->v = al->v.normalize() * al->height;
                        light_changed = true;
                    }
                }

                if (light_changed && ctx.optix_gpu_ptr && g_hasOptix) {
                    ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
            }

            // WORLD PROPERTIES - Draw full World content when selected
            if (sel.selected.type == SelectableType::World) {
                ImGui::Separator();
                // Draw all World/Environment settings (Sky, Atmosphere, Clouds, etc.)
                drawWorldContent(ctx);
            }
        }


        // ─────────────────────────────────────────────────────────────────────────
        // MATERIAL EDITOR (Takes remaining space)
        // ─────────────────────────────────────────────────────────────────────────
        // Hide Material Editor when World is selected (World has no materials)
        if (sel.selected.type != SelectableType::World) {
            ImGui::Separator();
            if (ImGui::CollapsingHeader("Material Editor", ImGuiTreeNodeFlags_DefaultOpen)) {
                drawMaterialPanel(ctx);
            }
        }
    }
    // Light Gizmos and Transform Gizmos moved to main draw() loop for all tabs
}

