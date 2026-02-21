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
#include "VDBVolume.h"

#include <unordered_set>
#include "TerrainManager.h"
#include "ProjectManager.h"
#include "scene_ui_animgraph.hpp"

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
    
    // Safety clamp for hierarchy height
    if (hierarchy_panel_height > available_h - 100.0f) 
        hierarchy_panel_height = std::max(100.0f, available_h - 100.0f);

    ImGui::BeginChild("HierarchyTree", ImVec2(0, hierarchy_panel_height), true);

    // WORLD (Environment) - Selectable like any other object
    {
        ImGui::PushID("WorldItem");
        bool world_selected = (sel.selected.type == SelectableType::World);
        
        // Visibility toggle for world (background)
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        // We can use a property in render_settings if it exists, otherwise just a visual toggle
        if (ImGui::Button(ctx.render_settings.show_background ? "(o)" : "( )")) {
            ctx.render_settings.show_background = !ctx.render_settings.show_background;
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
        }
        ImGui::PopStyleColor();
        ImGui::SameLine();

        ImGuiTreeNodeFlags world_flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanAvailWidth;
        if (world_selected) world_flags |= ImGuiTreeNodeFlags_Selected;

        ImVec4 worldColor = ctx.render_settings.show_background ? ImVec4(1.0f, 0.7f, 1.0f, 1.0f) : ImVec4(0.5f, 0.35f, 0.5f, 1.0f);
        
        ImVec2 pos = ImGui::GetCursorScreenPos();
        UIWidgets::DrawIcon(UIWidgets::IconType::System, pos, 16, ImGui::ColorConvertFloat4ToU32(worldColor));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 20);

        ImGui::PushStyleColor(ImGuiCol_Text, worldColor);
        ImGui::TreeNodeEx("World / Environment", world_flags);
        ImGui::PopStyleColor();

        if (ImGui::IsItemClicked()) {
            sel.selectWorld();
        }

        if (ImGui::BeginPopupContextItem()) {
            if (ImGui::MenuItem("Toggle Background", nullptr, ctx.render_settings.show_background)) {
                ctx.render_settings.show_background = !ctx.render_settings.show_background;
                ctx.renderer.resetCPUAccumulation();
            }
            ImGui::EndPopup();
        }
        ImGui::PopID();
    }


    ImGui::Separator();

    // ─────────────────────────────────────────────────────────────────────────
    // CHARACTERS / MODELS (Multi-Model Management)
    // ─────────────────────────────────────────────────────────────────────────
    if (!ctx.scene.importedModelContexts.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.1f, 0.9f, 0.8f, 1.0f));
        bool models_expanded = ImGui::TreeNode("Characters / Models");
        ImGui::PopStyleColor();

        if (models_expanded) {
            for (size_t i = 0; i < ctx.scene.importedModelContexts.size(); i++) {
                auto& mctx = ctx.scene.importedModelContexts[i];
                ImGui::PushID((int)(5000 + i));

                // 1. Visibility Pill Toggle
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
                if (ImGui::Button(mctx.visible ? "(o)" : "( )")) {
                    mctx.visible = !mctx.visible;
                    // Apply visibility to all mesh members belonging to this model
                    for (auto& member : mctx.members) {
                        if (member) member->visible = mctx.visible;
                    }
                    ctx.renderer.resetCPUAccumulation();
                    if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                }
                ImGui::PopStyleColor();
                ImGui::SameLine();

                // 2. Icon & Label
                ImVec2 pos = ImGui::GetCursorScreenPos();
                UIWidgets::DrawIcon(UIWidgets::IconType::Timeline, pos, 16, mctx.visible ? 0xFFFFFFFF : 0x88FFFFFF);
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 20);

                bool is_selected = (g_animGraphUI.activeCharacter == mctx.importName);
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
                if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;

                if (ImGui::TreeNodeEx(mctx.importName.c_str(), flags)) {
                    // Quick Settings
                    ImGui::Checkbox("Use Root Motion", &mctx.useRootMotion);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Object pivot follows the character's root animation (walking, etc.)");
                    
                    if (ImGui::Button("Select for AnimGraph")) {
                        g_animGraphUI.activeCharacter = mctx.importName;
                        show_anim_graph = true;
                    }
                    
                    ImGui::TreePop();
                }

                if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
                    g_animGraphUI.activeCharacter = mctx.importName;
                }

                ImGui::PopID();
            }
            ImGui::TreePop();
        }
        ImGui::Separator();
    }

    // CAMERAS (Multi-camera support)
    if (!ctx.scene.cameras.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 0.8f, 1.0f, 1.0f));
        bool cameras_expanded = ImGui::TreeNode("Cameras");
        ImGui::PopStyleColor();

        if (cameras_expanded) {
            for (size_t i = 0; i < ctx.scene.cameras.size(); i++) {
                ImGui::PushID((int)(1000 + i));  // Unique ID for each camera

                auto& cam = ctx.scene.cameras[i];
                if (!cam) { ImGui::PopID(); continue; }

                // Visibility Toggle
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
                if (ImGui::Button(cam->visible ? "(o)" : "( )")) {
                    cam->visible = !cam->visible;
                    ProjectManager::getInstance().markModified();
                }
                ImGui::PopStyleColor();
                ImGui::SameLine();

                SelectableItem cam_item;
                cam_item.type = SelectableType::Camera;
                cam_item.camera = cam;
                bool is_selected = sel.isSelected(cam_item);
                bool is_active = (i == ctx.scene.active_camera_index);

                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanAvailWidth;
                if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;

                // Active camera indicator
                std::string label = is_active ? "Camera #" + std::to_string(i) + " (Active)"
                    : "Camera #" + std::to_string(i);

                ImVec4 color = is_active ? ImVec4(0.3f, 1.0f, 0.5f, 1.0f) : ImVec4(0.5f, 0.7f, 1.0f, 1.0f);
                if (!cam->visible) color.w = 0.5f; // Dim if hidden

                ImVec2 pos = ImGui::GetCursorScreenPos();
                UIWidgets::DrawIcon(UIWidgets::IconType::Camera, pos, 16, ImGui::ColorConvertFloat4ToU32(color));
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 20);

                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::TreeNodeEx(label.c_str(), flags);
                ImGui::PopStyleColor();

                if (ImGui::IsItemClicked()) {
                    sel.selectCamera(cam);
                }

                // Context Menu
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem("Set As Active", nullptr, is_active)) {
                        ctx.scene.setActiveCamera(i);
                        if (ctx.optix_gpu_ptr && ctx.scene.camera) {
                            ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                        ctx.renderer.resetCPUAccumulation();
                    }
                    if (ImGui::MenuItem("Toggle Visibility", nullptr, cam->visible)) {
                        cam->visible = !cam->visible;
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem("Delete", "Del")) {
                        // Deletion logic (copied from properties for now)
                        if (ctx.scene.cameras.size() > 1 && cam != ctx.scene.camera) {
                            ctx.scene.removeCamera(cam);
                            sel.clearSelection();
                        } else {
                            SCENE_LOG_WARN("Cannot delete active or last camera");
                        }
                    }
                    ImGui::EndPopup();
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

                auto light = ctx.scene.lights[i]; // Local copy to avoid invalidation on resize

                // Visibility Toggle
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
                if (ImGui::Button(light->visible ? "(o)" : "( )")) {
                    light->visible = !light->visible;
                    if (ctx.optix_gpu_ptr) {
                        ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                    ProjectManager::getInstance().markModified();
                }
                ImGui::PopStyleColor();
                ImGui::SameLine();

                SelectableItem light_item;
                light_item.type = SelectableType::Light;
                light_item.light = light;
                light_item.light_index = (int)i;
                bool is_selected = sel.isSelected(light_item);

                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanAvailWidth;
                if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;

                ImVec4 color = ImVec4(1.0f, 0.9f, 0.4f, 1.0f); // Default light color
                if (!light->visible) color.w = 0.5f;

                std::string light_type = "Light";
                UIWidgets::IconType icon_type = UIWidgets::IconType::Light;

                switch (light->type()) {
                    case LightType::Point:       light_type = "Point"; icon_type = UIWidgets::IconType::LightPoint; break;
                    case LightType::Directional: light_type = "Directional"; color = ImVec4(1.0f, 0.7f, 0.3f, 1.0f); icon_type = UIWidgets::IconType::LightDir; break;
                    case LightType::Spot:        light_type = "Spot"; icon_type = UIWidgets::IconType::LightSpot; break;
                    case LightType::Area:        light_type = "Area"; icon_type = UIWidgets::IconType::LightArea; break;
                }

                ImVec2 pos = ImGui::GetCursorScreenPos();
                UIWidgets::DrawIcon(icon_type, pos, 16.0f, ImGui::ColorConvertFloat4ToU32(color));
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 20);

                std::string label = light_type + " " + std::to_string(i + 1);

                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::TreeNodeEx(label.c_str(), flags);
                ImGui::PopStyleColor();

                if (ImGui::IsItemClicked()) {
                    bool ctrl_held = ImGui::GetIO().KeyCtrl;
                    if (ctrl_held) {
                        SelectableItem item;
                        item.type = SelectableType::Light;
                        item.light = light;
                        item.light_index = (int)i;
                        item.name = label;
                        
                        if (sel.isSelected(item)) {
                            sel.removeFromSelection(item);
                        } else {
                            sel.addToSelection(item);
                        }
                    } else {
                        sel.selectLight(light, (int)i, label);
                    }
                }

                // Context Menu
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem("Duplicate", "Shift+D")) {
                        if (!is_selected) sel.selectLight(light, (int)i, label);
                        triggerDuplicate(ctx);
                    }
                    if (ImGui::MenuItem("Delete", "Del")) {
                        if (!is_selected) sel.selectLight(light, (int)i, label);
                        triggerDelete(ctx);
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem("Toggle Visibility", nullptr, light->visible)) {
                        light->visible = !light->visible;
                        if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                        ctx.renderer.resetCPUAccumulation();
                    }
                    ImGui::EndPopup();
                }

                ImGui::PopID();  // End unique ID
            }
            ImGui::TreePop();
        }
    }

    
    // Check for scene changes to invalidate cache
    // Sync logic moved to SceneUI::draw() for centralized consistency

    // ─────────────────────────────────────────────────────────────────────────
    // VDB VOLUMES
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::TreeNode("VDB Volumes")) {
        
        for (size_t i = 0; i < ctx.scene.vdb_volumes.size(); ++i) {
            auto vdb = ctx.scene.vdb_volumes[i]; // Local copy to avoid invalidation on resize
            if (!vdb) continue;
            
            ImGui::PushID((int)i);

            // Visibility Toggle
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
            if (ImGui::Button(vdb->visible ? "(o)" : "( )")) {
                vdb->visible = !vdb->visible;
                if (ctx.optix_gpu_ptr) {
                    syncVDBVolumesToGPU(ctx);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
                ctx.renderer.resetCPUAccumulation();
                ProjectManager::getInstance().markModified();
            }
            ImGui::PopStyleColor();
            ImGui::SameLine();

            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanAvailWidth;
            
            SelectableItem vdb_item;
            vdb_item.type = SelectableType::VDBVolume;
            vdb_item.vdb_volume = vdb;
            vdb_item.vdb_index = (int)i;
            vdb_item.name = vdb->name;
            bool is_selected = sel.isSelected(vdb_item);
            
            if (is_selected) {
                flags |= ImGuiTreeNodeFlags_Selected;
            }
            
            std::string label = "[VOL] " + vdb->name;
            ImGui::PushStyleColor(ImGuiCol_Text, vdb->visible ? ImVec4(1.0f, 0.5f, 0.0f, 1.0f) : ImVec4(0.5f, 0.25f, 0.0f, 1.0f));
            ImGui::TreeNodeEx((void*)(intptr_t)vdb.get(), flags, label.c_str());
            ImGui::PopStyleColor();
            
            if (ImGui::IsItemClicked()) {
                bool multi_select = ImGui::GetIO().KeyCtrl;
                if (multi_select) {
                    SelectableItem item;
                    item.type = SelectableType::VDBVolume;
                    item.vdb_volume = vdb;
                    item.vdb_index = (int)i;
                    item.name = vdb->name;
                    
                    if (ctx.selection.isSelected(item))
                        ctx.selection.removeFromSelection(item);
                    else
                        ctx.selection.addToSelection(item);
                } else {
                    ctx.selection.selectVDBVolume(vdb, (int)i, vdb->name);
                    
                    // Auto-open VDB tab removed per user request
                }
            }

            // Context Menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Duplicate", "Shift+D")) {
                    if (!is_selected) ctx.selection.selectVDBVolume(vdb, (int)i, vdb->name);
                    triggerDuplicate(ctx);
                }
                if (ImGui::MenuItem("Delete", "Del")) {
                    if (!is_selected) ctx.selection.selectVDBVolume(vdb, (int)i, vdb->name);
                    triggerDelete(ctx);
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Toggle Visibility", nullptr, vdb->visible)) {
                    vdb->visible = !vdb->visible;
                    if (ctx.optix_gpu_ptr) {
                        syncVDBVolumesToGPU(ctx);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                }
                ImGui::EndPopup();
            }

            ImGui::PopID();
        }
        
        ImGui::TreePop();
    }


    // ─────────────────────────────────────────────────────────────────────────
    // GAS VOLUMES
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::TreeNode("Gas Volumes")) {
        
        for (size_t i = 0; i < ctx.scene.gas_volumes.size(); ++i) {
            auto& gas = ctx.scene.gas_volumes[i];
            if (!gas) continue;
            
            ImGui::PushID((int)i);

            // Visibility Toggle
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
            if (ImGui::Button(gas->visible ? "(o)" : "( )")) {
                gas->visible = !gas->visible;
                if (ctx.optix_gpu_ptr) {
                    ctx.renderer.updateOptiXGasVolumes(ctx.scene, ctx.optix_gpu_ptr);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
                ctx.renderer.resetCPUAccumulation();
                ProjectManager::getInstance().markModified();
            }
            ImGui::PopStyleColor();
            ImGui::SameLine();

            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanAvailWidth;
            
            SelectableItem gas_item;
            gas_item.type = SelectableType::GasVolume;
            gas_item.gas_volume = gas;
            gas_item.name = gas->name;
            bool is_selected = sel.isSelected(gas_item);
            
            if (is_selected) {
                flags |= ImGuiTreeNodeFlags_Selected;
            }
            
            std::string label = "[GAS] " + gas->name;
            ImGui::PushStyleColor(ImGuiCol_Text, gas->visible ? ImVec4(0.0f, 1.0f, 1.0f, 1.0f) : ImVec4(0.0f, 0.5f, 0.5f, 1.0f));
            ImGui::TreeNodeEx((void*)(intptr_t)gas.get(), flags, label.c_str());
            ImGui::PopStyleColor();
            
            if (ImGui::IsItemClicked()) {
                 ctx.selection.selectGasVolume(gas, (int)i, gas->name);
                 
                 // Auto-open Gas tab removed per user request
            }

            // Context Menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Delete", "Del")) {
                    if (!is_selected) ctx.selection.selectGasVolume(gas, (int)i, gas->name);
                    triggerDelete(ctx);
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Toggle Visibility", nullptr, gas->visible)) {
                    gas->visible = !gas->visible;
                    if (ctx.optix_gpu_ptr) {
                        ctx.renderer.updateOptiXGasVolumes(ctx.scene, ctx.optix_gpu_ptr);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                }
                ImGui::EndPopup();
            }

            ImGui::PopID();
        }
        
        ImGui::TreePop();
    }


    // ─────────────────────────────────────────────────────────────────────────
    // FORCE FIELDS
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::TreeNode("Force Fields")) {
        auto& fields = ctx.scene.force_field_manager.force_fields;
        for (size_t i = 0; i < fields.size(); ++i) {
            auto& ff = fields[i];
            if (!ff) continue;
            
            ImGui::PushID((int)(20000 + i));

            // Visibility Toggle
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
            if (ImGui::Button(ff->visible ? "(o)" : "( )")) {
                ff->visible = !ff->visible;
                ProjectManager::getInstance().markModified();
            }
            ImGui::PopStyleColor();
            ImGui::SameLine();

            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_SpanAvailWidth;
            
            SelectableItem ff_item;
            ff_item.type = SelectableType::ForceField;
            ff_item.force_field = ff;
            ff_item.name = ff->name;
            bool is_selected = sel.isSelected(ff_item);
            
            if (is_selected) flags |= ImGuiTreeNodeFlags_Selected;
            
            std::string label = "[FF] " + ff->name;
            ImGui::PushStyleColor(ImGuiCol_Text, ff->visible ? ImVec4(0.7f, 1.0f, 0.7f, 1.0f) : ImVec4(0.35f, 0.5f, 0.35f, 1.0f));
            ImGui::TreeNodeEx((void*)(intptr_t)ff.get(), flags, label.c_str());
            ImGui::PopStyleColor();
            
            if (ImGui::IsItemClicked()) {
                 ctx.selection.selectForceField(ff, (int)i, ff->name);
                 show_forcefield_tab = true;
            }

            // Context Menu
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Delete", "Del")) {
                    if (!is_selected) ctx.selection.selectForceField(ff, (int)i, ff->name);
                    triggerDelete(ctx);
                }
                ImGui::EndPopup();
            }

            ImGui::PopID();
        }
        ImGui::TreePop();
    }


    // ─────────────────────────────────────────────────────────────────────────
    // OBJECTS LIST (HIERARCHY)
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::TreeNode("Objects")) {

        // Ensure cache is valid
        if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

        // SELECT ALL / SELECT NONE / NEW GROUP buttons
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
        
        // NEW GROUP button - creates group from current selection
        if (ctx.selection.multi_selection.size() > 1) {
            if (ImGui::Button("New Group##obj")) {
                static int group_counter = 0;
                SceneData::SceneGroup new_group;
                new_group.name = "Group_" + std::to_string(++group_counter);
                
                for (const auto& item : ctx.selection.multi_selection) {
                    if (item.type == SelectableType::Object && !item.name.empty()) {
                        new_group.member_names.push_back(item.name);
                    }
                }
                
                if (!new_group.member_names.empty()) {
                    ctx.scene.object_groups.push_back(new_group);
                    SCENE_LOG_INFO("Created group '" + new_group.name + "' with " + 
                                   std::to_string(new_group.member_names.size()) + " objects");
                }
            }
            ImGui::SameLine();
        }
        
        ImGui::Text("(%d objects)", (int)mesh_ui_cache.size());

        // ─────────────────────────────────────────────────────────────────────
        // GROUPS SECTION
        // ─────────────────────────────────────────────────────────────────────
        if (!ctx.scene.object_groups.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.8f, 0.4f, 1.0f));
            for (size_t gi = 0; gi < ctx.scene.object_groups.size(); ++gi) {
                auto& grp = ctx.scene.object_groups[gi];
                ImGui::PushID((int)(10000 + gi));
                
                // Group Visibility Toggle
                bool group_visible = true;
                if (!grp.member_names.empty()) {
                    auto it = mesh_cache.find(grp.member_names[0]);
                    if (it != mesh_cache.end() && !it->second.empty()) group_visible = it->second[0].second->visible;
                }

                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
                if (ImGui::Button(group_visible ? "(o)" : "( )")) {
                    for (const auto& member : grp.member_names) {
                        auto it = mesh_cache.find(member);
                        if (it != mesh_cache.end()) {
                            for (auto& pair : it->second) pair.second->visible = !group_visible;
                        }
                    }
                    ctx.renderer.resetCPUAccumulation();
                }
                ImGui::PopStyleColor();
                ImGui::SameLine();

                ImGuiTreeNodeFlags grp_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
                if (grp.expanded) grp_flags |= ImGuiTreeNodeFlags_DefaultOpen;
                
                std::string grp_label = "[GRP] " + grp.name + " (" + 
                                        std::to_string(grp.member_names.size()) + ")";
                
                ImGui::PushStyleColor(ImGuiCol_Text, group_visible ? ImVec4(0.9f, 0.8f, 0.4f, 1.0f) : ImVec4(0.45f, 0.4f, 0.2f, 1.0f));
                bool grp_open = ImGui::TreeNodeEx(grp_label.c_str(), grp_flags);
                ImGui::PopStyleColor();

                grp.expanded = grp_open;
                
                // Click on group = select all members
                if (ImGui::IsItemClicked()) {
                    ctx.selection.clearSelection();
                    for (const auto& member_name : grp.member_names) {
                        auto cache_it = mesh_cache.find(member_name);
                        if (cache_it != mesh_cache.end() && !cache_it->second.empty()) {
                            SelectableItem item;
                            item.type = SelectableType::Object;
                            item.object = cache_it->second[0].second;
                            item.object_index = cache_it->second[0].first;
                            item.name = member_name;
                            ctx.selection.addToSelection(item);
                        }
                    }
                }
                
                // Right-click context menu for group
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem("Select Members")) {
                         ctx.selection.clearSelection();
                         for (const auto& member_name : grp.member_names) {
                             auto cache_it = mesh_cache.find(member_name);
                             if (cache_it != mesh_cache.end() && !cache_it->second.empty()) {
                                 SelectableItem item;
                                 item.type = SelectableType::Object;
                                 item.object = cache_it->second[0].second;
                                 item.object_index = cache_it->second[0].first;
                                 item.name = member_name;
                                 ctx.selection.addToSelection(item);
                             }
                         }
                    }
                    if (ImGui::MenuItem("Toggle Visibility", nullptr, group_visible)) {
                        for (const auto& member : grp.member_names) {
                            auto it = mesh_cache.find(member);
                            if (it != mesh_cache.end()) {
                                for (auto& pair : it->second) pair.second->visible = !group_visible;
                            }
                        }
                        ctx.renderer.resetCPUAccumulation();
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem("Delete Group")) {
                        ctx.scene.object_groups.erase(ctx.scene.object_groups.begin() + gi);
                        ImGui::EndPopup();
                        ImGui::PopID();
                        break; // Exit loop since vector modified
                    }
                    ImGui::EndPopup();
                }

                
                if (grp_open) {
                    // Show group members
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
                    for (const auto& member_name : grp.member_names) {
                        ImGuiTreeNodeFlags mem_flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                        
                        // Check if member is currently selected
                        bool mem_selected = false;
                        for (const auto& sel_item : ctx.selection.multi_selection) {
                            if (sel_item.name == member_name) {
                                mem_selected = true;
                                break;
                            }
                        }
                        if (mem_selected) mem_flags |= ImGuiTreeNodeFlags_Selected;
                        
                        ImGui::TreeNodeEx(member_name.c_str(), mem_flags);
                        
                        // Click on member = select just that member
                        if (ImGui::IsItemClicked()) {
                            auto cache_it = mesh_cache.find(member_name);
                            if (cache_it != mesh_cache.end() && !cache_it->second.empty()) {
                                bool ctrl = ImGui::GetIO().KeyCtrl;
                                SelectableItem item;
                                item.type = SelectableType::Object;
                                item.object = cache_it->second[0].second;
                                item.object_index = cache_it->second[0].first;
                                item.name = member_name;
                                
                                if (ctrl) {
                                    sel.isSelected(item) ? sel.removeFromSelection(item) : sel.addToSelection(item);
                                } else {
                                    sel.selectObject(item.object, item.object_index, member_name);
                                }
                            }
                        }
                    }
                    ImGui::PopStyleColor();
                    ImGui::TreePop();
                }
                
                ImGui::PopID();
            }
            ImGui::PopStyleColor();
            ImGui::Separator();
        }

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
                
                // Skip objects that are in a group (they're shown inside the group)
                bool in_group = false;
                for (const auto& grp : ctx.scene.object_groups) {
                    if (grp.contains(name)) {
                        in_group = true;
                        break;
                    }
                }
                if (in_group) continue;

                SelectableItem obj_item;
                obj_item.type = SelectableType::Object;
                obj_item.name = name;
                if (!kv.second.empty()) {
                    obj_item.object = kv.second[0].second;
                    obj_item.object_index = kv.second[0].first;
                }
                bool is_selected = sel.isSelected(obj_item);
                
                // Check if this object has skinning data
                bool has_skinning = false;
                if (!kv.second.empty()) {
                    has_skinning = kv.second[0].second->hasSkinData();
                }
                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
                // Always allow tree expansion for skinned meshes
                if (!is_selected && !has_skinning) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                if (is_selected) flags |= ImGuiTreeNodeFlags_Selected | ImGuiTreeNodeFlags_DefaultOpen;

                // OBJECT ITEM START
                ImGui::PushID(i);

                // Visibility Toggle (Mesh level)
                bool all_visible = true;
                if (!kv.second.empty()) all_visible = kv.second[0].second->visible;

                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
                if (ImGui::Button(all_visible ? "(o)" : "( )")) {
                    bool new_visible = !all_visible;
                    for (auto& pair : kv.second) pair.second->visible = new_visible;
                    
                    if (ctx.optix_gpu_ptr) {
                        ctx.optix_gpu_ptr->setVisibilityByNodeName(name, new_visible);
                    }
                    
                    extern bool g_bvh_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    
                    ProjectManager::getInstance().markModified();
                    ctx.renderer.resetCPUAccumulation();
                }
                ImGui::PopStyleColor();
                ImGui::SameLine();
              
                // Check if this object has skinning data
             
                if (!kv.second.empty()) {
                    has_skinning = kv.second[0].second->hasSkinData();
                }

             
                // Always allow tree expansion for skinned meshes
                if (!is_selected && !has_skinning) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                if (is_selected) flags |= ImGuiTreeNodeFlags_Selected | ImGuiTreeNodeFlags_DefaultOpen;

                // Different color/icon for skinned meshes
                ImVec4 textColor = has_skinning ? ImVec4(0.9f, 0.7f, 0.5f, 1.0f) : ImVec4(0.8f, 0.8f, 0.85f, 1.0f);
                if (!all_visible) textColor.w = 0.5f;

                ImVec2 pos = ImGui::GetCursorScreenPos();
                UIWidgets::DrawIcon(has_skinning ? UIWidgets::IconType::Physics : UIWidgets::IconType::Mesh, 
                    pos, 16, ImGui::ColorConvertFloat4ToU32(textColor));
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 20);

                ImGui::PushStyleColor(ImGuiCol_Text, textColor);
                
                std::string displayName = (name.empty() ? "Unnamed Object" : name);

                bool node_open = false;
                if (is_selected || has_skinning) {
                    node_open = ImGui::TreeNodeEx(displayName.c_str(), flags);
                }
                else {
                    ImGui::TreeNodeEx(displayName.c_str(), flags); // Leaf, no push
                }
                ImGui::PopStyleColor();

                if (ImGui::IsItemClicked()) {
                    if (!kv.second.empty()) {
                        auto& first_pair = kv.second[0];
                        bool ctrl_held = ImGui::GetIO().KeyCtrl;
                        
                        if (ctrl_held) {
                            // Ctrl+Click: Toggle selection (add/remove from multi-selection)
                            SelectableItem item;
                            item.type = SelectableType::Object;
                            item.object = first_pair.second;
                            item.object_index = first_pair.first;
                            item.name = name;
                            
                            if (sel.isSelected(item)) {
                                sel.removeFromSelection(item);
                            } else {
                                sel.addToSelection(item);
                            }
                        } else {
                            // Normal click: Single selection (replaces current)
                            sel.selectObject(first_pair.second, first_pair.first, name);
                        }
                        
                        // TERRAIN CONNECTION: Check if this is a terrain chunk
                        if (name.find("Terrain_") == 0) {
                            std::string tName = name;
                            size_t chunkPos = tName.find("_Chunk");
                            if (chunkPos != std::string::npos) tName = tName.substr(0, chunkPos);
                            
                            auto terrain = TerrainManager::getInstance().getTerrainByName(tName);
                            if (terrain) {
                                terrain_brush.active_terrain_id = terrain->id;
                                SCENE_LOG_INFO("Terrain selected: " + tName);
                            }
                        }
                    }
                }

                // Context Menu
                if (ImGui::BeginPopupContextItem()) {
                    if (ImGui::MenuItem("Duplicate", "Shift+D")) {
                        if (!is_selected && !kv.second.empty()) sel.selectObject(kv.second[0].second, kv.second[0].first, name);
                        triggerDuplicate(ctx);
                    }
                    if (ImGui::MenuItem("Delete", "Del")) {
                        if (!is_selected && !kv.second.empty()) sel.selectObject(kv.second[0].second, kv.second[0].first, name);
                        triggerDelete(ctx);
                    }
                    ImGui::Separator();
                    if (ImGui::MenuItem("Toggle Visibility", nullptr, all_visible)) {
                        bool new_visible = !all_visible;
                        for (auto& pair : kv.second) pair.second->visible = new_visible;
                        if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->setVisibilityByNodeName(name, new_visible);
                        ctx.renderer.resetCPUAccumulation();
                    }
                    ImGui::EndPopup();
                }

                if (node_open) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
                    ImGui::Indent();
                    
                    // Identify model context for prefix-based filtering
                    SceneData::ImportedModelContext* model_ctx = nullptr;
                    for (auto& mctx : ctx.scene.importedModelContexts) {
                        if (!mctx.importName.empty() && name.find(mctx.importName + "_") == 0) {
                            model_ctx = &mctx;
                            break;
                        }
                    }
                    std::string prefix = (model_ctx ? model_ctx->importName : "") + "_";

                    // === SKINNING INFO (for skinned meshes) ===
                    if (has_skinning) {
                        // Bones sub-tree
                        bool has_bones = false;
                        for (const auto& [bn, bi] : ctx.scene.boneData.boneNameToIndex) {
                            if (bn.find(prefix) == 0) { has_bones = true; break; }
                        }

                        if (has_bones) {
                            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.9f, 0.6f, 0.3f, 1.0f));
                            if (ImGui::TreeNodeEx("Bones##skinned", ImGuiTreeNodeFlags_SpanAvailWidth)) {
                                ImGui::PopStyleColor();
                                
                                // Show only bones for this model
                                for (const auto& [boneName, boneIdx] : ctx.scene.boneData.boneNameToIndex) {
                                    if (boneName.find(prefix) == 0) {
                                        ImGui::BulletText("%d: %s", boneIdx, boneName.c_str());
                                    }
                                }
                                ImGui::TreePop();
                            } else {
                                ImGui::PopStyleColor();
                            }
                        }
                        
                        // Animations sub-tree
                        if (model_ctx && model_ctx->animator) {
                            auto& clips = model_ctx->animator->getAllClips();
                            if (!clips.empty()) {
                                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.6f, 1.0f));
                                if (ImGui::TreeNodeEx("Animations##skinned", ImGuiTreeNodeFlags_SpanAvailWidth)) {
                                    ImGui::PopStyleColor();
                                    
                                    std::string currentClip = model_ctx->animator->getCurrentClipName();
                                    for (size_t ai = 0; ai < clips.size(); ++ai) {
                                        auto& clip = clips[ai];
                                        bool is_active = (clip.name == currentClip);
                                        
                                        ImGui::PushID((int)ai);
                                        if (is_active) ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.8f, 0.0f, 1.0f));
                                        
                                        if (ImGui::Selectable(clip.name.c_str(), is_active)) {
                                            model_ctx->animator->play(clip.name, 0.3f);
                                        }
                                        
                                        if (is_active) ImGui::PopStyleColor();
                                        ImGui::PopID();
                                    }
                                    ImGui::TreePop();
                                } else {
                                    ImGui::PopStyleColor();
                                }
                            }
                        }
                        
                        ImGui::Separator();
                    }

                    // --- In-Tree Properties (for selected objects) ---
                    if (is_selected) {
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
                                ProjectManager::getInstance().markModified();
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
    
    // --- VERTICAL RESIZE SPLITTER ---
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1, 1, 1, 0.05f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.1f, 0.9f, 0.8f, 0.4f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.1f, 0.9f, 0.8f, 0.6f));
    
    ImGui::Button("##H_Splitter", ImVec2(-1, 5)); // Thin horizontal bar
    if (ImGui::IsItemActive()) {
        hierarchy_panel_height += ImGui::GetIO().MouseDelta.y;
    }
    if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar();
    ImGui::Spacing();

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
            case SelectableType::VDBVolume: typeIcon = "[VOL]"; typeColor = ImVec4(1.0f, 0.5f, 0.0f, 1.0f); break;
            case SelectableType::GasVolume: typeIcon = "[GAS]"; typeColor = ImVec4(0.0f, 1.0f, 1.0f, 1.0f); break;
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
                    else if (sel.selected.type == SelectableType::GasVolume && sel.selected.gas_volume) {
                        if (ctx.scene.removeGasVolume(sel.selected.gas_volume)) {
                            deleted = true;
                            SCENE_LOG_INFO("Gas Volume deleted successfully");
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
                        ProjectManager::getInstance().markModified();
                    }
                    else {
                        sel.clearSelection();
                    }
                }
                ImGui::PopStyleColor();
                
                // ─────────────────────────────────────────────────────────────────
                // MANUAL TRANSFORM CONTROLS (POS / ROT / SCALE)
                // ─────────────────────────────────────────────────────────────────
                if (sel.selected.type == SelectableType::Object || sel.selected.type == SelectableType::Light || 
                    sel.selected.type == SelectableType::Camera || sel.selected.type == SelectableType::VDBVolume ||
                    sel.selected.type == SelectableType::GasVolume || sel.selected.type == SelectableType::ForceField ||
                    sel.selected.type == SelectableType::CameraTarget) 
                {
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.8f, 1.0f), "Transform");
                    
                    static bool lock_scale = false;
                    
                    bool transform_changed = false;
                    ImGui::PushItemWidth(std::max(100.0f, ImGui::GetContentRegionAvail().x - 60.0f));
                    
                    if (ImGui::DragFloat3("Pos", &sel.selected.position.x, 0.1f)) transform_changed = true;
                    if (ImGui::DragFloat3("Rot", &sel.selected.rotation.x, 1.0f)) transform_changed = true;
                    
                    Vec3 old_scale = sel.selected.scale;
                    if (ImGui::DragFloat3("Scl", &sel.selected.scale.x, 0.05f)) {
                        transform_changed = true;
                        
                        if (lock_scale) {
                            float ratio = 1.0f;
                            float delta = 0.0f;
                            
                            if (sel.selected.scale.x != old_scale.x) {
                                if (old_scale.x != 0.0f) ratio = sel.selected.scale.x / old_scale.x;
                                else delta = sel.selected.scale.x - old_scale.x;
                            } else if (sel.selected.scale.y != old_scale.y) {
                                if (old_scale.y != 0.0f) ratio = sel.selected.scale.y / old_scale.y;
                                else delta = sel.selected.scale.y - old_scale.y;
                            } else if (sel.selected.scale.z != old_scale.z) {
                                if (old_scale.z != 0.0f) ratio = sel.selected.scale.z / old_scale.z;
                                else delta = sel.selected.scale.z - old_scale.z;
                            }
                            
                            if (ratio != 1.0f) {
                                sel.selected.scale = old_scale * ratio;
                            } else if (delta != 0.0f) {
                                sel.selected.scale.x = old_scale.x + delta;
                                sel.selected.scale.y = old_scale.y + delta;
                                sel.selected.scale.z = old_scale.z + delta;
                            }
                        }
                    }
                    
                    ImGui::PopItemWidth();
                    ImGui::SameLine();
                    
                    // Simple lock toggle button
                    ImGui::PushStyleColor(ImGuiCol_Button, lock_scale ? ImVec4(0.3f, 0.6f, 0.3f, 1.0f) : ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
                    if (ImGui::Button(lock_scale ? " L " : " U ")) {
                        lock_scale = !lock_scale;
                    }
                    ImGui::PopStyleColor();
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Uniform Scale Lock (L=Locked, U=Unlocked)");
                    
                    if (transform_changed) {
                        for (auto& item : sel.multi_selection) {
                            // Apply same modified absolute transform
                            item.position = sel.selected.position;
                            item.rotation = sel.selected.rotation;
                            item.scale = sel.selected.scale;

                            if (item.type == SelectableType::Object && item.object) {
                                auto transform = item.object->getTransformHandle();
                                if (transform) {
                                    transform->setBase(Matrix4x4::fromTRS(item.position, item.rotation, item.scale));
                                    item.object->updateTransformedVertices();
                                }
                            } else if (item.type == SelectableType::Light && item.light) {
                                item.light->position = item.position;
                            } else if (item.type == SelectableType::Camera && item.camera) {
                                Vec3 delta = item.position - item.camera->lookfrom;
                                item.camera->lookfrom = item.position;
                                item.camera->lookat = item.camera->lookat + delta;
                                item.camera->update_camera_vectors();
                            } else if (item.type == SelectableType::CameraTarget && item.camera) {
                                item.camera->lookat = item.position;
                                item.camera->update_camera_vectors();
                            } else if (item.type == SelectableType::VDBVolume && item.vdb_volume) {
                                item.vdb_volume->setPosition(item.position);
                                item.vdb_volume->setRotation(item.rotation);
                                item.vdb_volume->setScale(item.scale);
                            } else if (item.type == SelectableType::GasVolume && item.gas_volume) {
                                item.gas_volume->setPosition(item.position);
                                item.gas_volume->setRotation(item.rotation);
                                item.gas_volume->setScale(item.scale);
                            } else if (item.type == SelectableType::ForceField && item.force_field) {
                                item.force_field->position = item.position;
                                item.force_field->rotation = item.rotation;
                                item.force_field->scale = item.scale;
                            }
                        }

                        // Trigger global visual updates
                        extern bool g_bvh_rebuild_pending;
                        if (sel.selected.type == SelectableType::Object) {
                            g_bvh_rebuild_pending = true;
                            sel.selected.has_cached_aabb = false;
                            if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects);
                        } else if (sel.selected.type == SelectableType::Light) {
                            if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                        } else if (sel.selected.type == SelectableType::Camera || sel.selected.type == SelectableType::CameraTarget) {
                            if (ctx.optix_gpu_ptr && sel.selected.camera) ctx.optix_gpu_ptr->setCameraParams(*sel.selected.camera);
                        } else if (sel.selected.type == SelectableType::VDBVolume) {
                            SceneUI::syncVDBVolumesToGPU(ctx);
                        } else if (sel.selected.type == SelectableType::GasVolume) {
                            ctx.renderer.updateOptiXGasVolumes(ctx.scene, ctx.optix_gpu_ptr);
                        }

                        if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                        ctx.renderer.resetCPUAccumulation();
                        ProjectManager::getInstance().markModified();
                    }
                }
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

                // Position with Smart Slider
                Vec3 pos = cam.lookfrom;
                bool posKeyed = isCamKeyed(true, false, false, false, false);
                // Note: DrawSmartFloat handles 1 float, here we have 3 (DragFloat3). 
                // Currently DrawSmartFloat does not support Float3. We will keep DragFloat3 for vectors for now
                // BUT we can use DrawLCDSlider logic if we wanted to overload it, or just keep DragFloat3 as it is standard for vectors.
                // Reverting Position change plan as DrawSmartFloat is for single floats.
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
                // CAMERA MODE SELECTOR (TOP - Controls what's visible below)
                // ═══════════════════════════════════════════════════════════════════════════
                // Mode determines complexity: Auto (simple), Pro (full control), Cinema (effects)
                {
                    ImGui::TextColored(ImVec4(0.9f, 0.8f, 0.4f, 1.0f), "Camera Mode");
                    
                    // Styled mode buttons (like radio buttons but prettier)
                    float btn_width = (ImGui::GetContentRegionAvail().x - 8) / 3.0f;
                    int current_mode = static_cast<int>(cam.camera_mode);
                    
                    auto ModeButton = [&](const char* label, int mode, ImVec4 color) -> bool {
                        bool selected = (current_mode == mode);
                        if (selected) {
                            ImGui::PushStyleColor(ImGuiCol_Button, color);
                            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(color.x + 0.1f, color.y + 0.1f, color.z + 0.1f, 1.0f));
                        } else {
                            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.25f, 1.0f));
                            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.35f, 1.0f));
                        }
                        bool clicked = ImGui::Button(label, ImVec2(btn_width, 0));
                        ImGui::PopStyleColor(2);
                        return clicked;
                    };
                    
                    if (ModeButton("Auto", 0, ImVec4(0.2f, 0.6f, 0.3f, 1.0f))) {
                        cam.camera_mode = CameraMode::Auto;
                        cam.auto_exposure = true;
                        cam.enable_chromatic_aberration = false;
                        cam.enable_vignetting = false;
                        cam.enable_camera_shake = false;
                        if (ctx.optix_gpu_ptr && g_hasOptix) {
                            ctx.optix_gpu_ptr->setCameraParams(cam);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                        ctx.renderer.resetCPUAccumulation();
                        ProjectManager::getInstance().markModified();
                    }
                    ImGui::SameLine(0, 4);
                    if (ModeButton("Pro", 1, ImVec4(0.3f, 0.5f, 0.8f, 1.0f))) {
                        cam.camera_mode = CameraMode::Pro;
                        cam.auto_exposure = false;
                        if (ctx.optix_gpu_ptr && g_hasOptix) {
                            ctx.optix_gpu_ptr->setCameraParams(cam);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                        ctx.renderer.resetCPUAccumulation();
                        ProjectManager::getInstance().markModified();
                    }
                    ImGui::SameLine(0, 4);
                    if (ModeButton("Cinema", 2, ImVec4(0.7f, 0.4f, 0.2f, 1.0f))) {
                        cam.camera_mode = CameraMode::Cinema;
                        cam.auto_exposure = false;
                        cam.enable_vignetting = true;
                        cam.vignetting_amount = 0.3f;
                        if (ctx.optix_gpu_ptr && g_hasOptix) {
                            ctx.optix_gpu_ptr->setCameraParams(cam);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                        ctx.renderer.resetCPUAccumulation();
                        ProjectManager::getInstance().markModified();
                    }
                    
                    // Mode description
                    const char* mode_desc[] = {
                        "Simple controls, automatic settings",
                        "Full manual control for professionals", 
                        "Cinematic effects & lens simulation"
                    };
                    ImGui::TextDisabled("%s", mode_desc[current_mode]);
                }
                
                ImGui::Separator();
                ImGui::Spacing();
                
                // ── DISTINCT INPUT FIELD STYLING FOR CAMERA PROPERTIES ────────────────
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.15f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.18f, 0.20f, 0.25f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.22f, 0.25f, 0.30f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImVec4(0.4f, 0.6f, 0.8f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4(0.5f, 0.7f, 0.9f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.35f, 0.38f, 0.45f, 0.8f));
                ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
                ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);

                // ═══════════════════════════════════════════════════════════════════════════
                // PRO & CINEMA MODE: Full Camera Controls
                // ═══════════════════════════════════════════════════════════════════════════
                // Show professional controls only in Pro and Cinema modes
                bool show_pro_controls = (cam.camera_mode == CameraMode::Pro || cam.camera_mode == CameraMode::Cinema);
                
                if (show_pro_controls) {
                // ═══════════════════════════════════════════════════════════════════════════
                // CAMERA BODY - Sensor size affects crop factor and ISO limits
                // ═══════════════════════════════════════════════════════════════════════════
                // Use member index instead of static local
                int selected_body = cam.body_preset_index; 
                // Safety check
                if (selected_body < 0 || selected_body >= (int)CameraPresets::CAMERA_BODY_COUNT) selected_body = 0;

                if (ImGui::BeginCombo("Camera Body", CameraPresets::CAMERA_BODIES[selected_body].name)) {
                    for (size_t i = 0; i < CameraPresets::CAMERA_BODY_COUNT; ++i) {
                        bool is_selected = (selected_body == (int)i);
                        std::string label = std::string(CameraPresets::CAMERA_BODIES[i].name);
                        if (i > 0) {
                            label += " (" + std::string(CameraPresets::getSensorTypeName(CameraPresets::CAMERA_BODIES[i].sensor)) + ")";
                        }
                        if (ImGui::Selectable(label.c_str(), is_selected)) {
                            cam.body_preset_index = (int)i;
                            
                            // Apply Body Specs (Sensor Dimensions & ISO Limits)
                            if (i > 0) {
                                const auto& body = CameraPresets::CAMERA_BODIES[i];
                                cam.sensor_width_mm = body.sensor_width_mm;
                                cam.sensor_height_mm = body.sensor_height_mm;
                                
                                // Clamp ISO to camera capabilities
                                if (cam.iso < body.min_iso) cam.iso = body.min_iso;
                                if (cam.iso > body.max_iso) cam.iso = body.max_iso;
                                
                                // Recalculate FOV based on new Sensor Height (Crop Factor)
                                if (cam.use_physical_lens) {
                                    float vfov_rad = 2.0f * std::atan((cam.sensor_height_mm * 0.5f) / cam.focal_length_mm);
                                    cam.vfov = vfov_rad * (180.0f / 3.14159265f);
                                    cam.fov = cam.vfov;
                                }
                                cam.update_camera_vectors(); 
                                
                                cam.markDirty();
                            }
                        }
                        if (is_selected) ImGui::SetItemDefaultFocus();
                        if (ImGui::IsItemHovered() && i > 0) {
                            ImGui::SetTooltip("%s - %s\nCrop: %.2fx\nISO: %d - %d",
                                CameraPresets::CAMERA_BODIES[i].brand,
                                CameraPresets::CAMERA_BODIES[i].description,
                                CameraPresets::CAMERA_BODIES[i].crop_factor,
                                CameraPresets::CAMERA_BODIES[i].min_iso,
                                CameraPresets::CAMERA_BODIES[i].max_iso);
                        }
                    }
                    ImGui::EndCombo();
                }
                
                // Show current specs
                ImGui::TextDisabled("Sensor: %.1f x %.1f mm", cam.sensor_width_mm, cam.sensor_height_mm);
                if (cam.body_preset_index > 0) {
                   ImGui::TextDisabled("Native ISO: %d - %d", 
                       CameraPresets::CAMERA_BODIES[cam.body_preset_index].min_iso,
                       CameraPresets::CAMERA_BODIES[cam.body_preset_index].max_iso);
                }
                
                ImGui::SameLine();
                if (ImGui::Button("Reset to Custom")) {
                     cam.body_preset_index = 0; // Custom
                }
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "%.1fx crop",
                        CameraPresets::CAMERA_BODIES[selected_body].crop_factor);
                

                // ═══════════════════════════════════════════════════════════════════════════
                // LENS PRESETS - Photographer-friendly focal length selection
                // ═══════════════════════════════════════════════════════════════════════════
                int selected_lens = cam.lens_preset_index;
                if (selected_lens < 0 || selected_lens >= (int)CameraPresets::LENS_PRESET_COUNT) selected_lens = 0;

                if (ImGui::BeginCombo("Lens Preset", CameraPresets::LENS_PRESETS[selected_lens].name)) {
                     for (int i = 0; i < (int)CameraPresets::LENS_PRESET_COUNT; i++) {
                         const auto& preset = CameraPresets::LENS_PRESETS[i];
                         
                         char label[128];
                         if (preset.is_zoom) {
                             snprintf(label, sizeof(label), "%s (%.0f-%.0fmm)", preset.name, preset.min_mm, preset.max_mm);
                         } else {
                             snprintf(label, sizeof(label), "%s (%.0fmm)", preset.name, preset.focal_mm);
                         }
                         
                         bool is_selected = (selected_lens == i);
                         if (ImGui::Selectable(label, is_selected)) {
                             cam.lens_preset_index = i;
                             selected_lens = i;
                             
                             if (i > 0) {
                                cam.vfov = preset.fov_deg;
                                cam.fov = preset.fov_deg;
                                cam.focal_length_mm = preset.focal_mm;
                             }
                             cam.update_camera_vectors();
                             cam.markDirty();
                         }
                         if (is_selected) ImGui::SetItemDefaultFocus();
                         
                         if (ImGui::IsItemHovered() && i > 0) {
                             if (preset.is_zoom)
                                ImGui::SetTooltip("%s\nZoom Range: %.0f-%.0fmm\nMax Aperture: f/%.1f\n%s", 
                                    preset.brand, preset.min_mm, preset.max_mm, preset.max_aperture, preset.description);
                             else
                                ImGui::SetTooltip("%s\nFocal Length: %.0fmm\nMax Aperture: f/%.1f\n%s", 
                                    preset.brand, preset.focal_mm, preset.max_aperture, preset.description);
                         }
                     }
                     ImGui::EndCombo();
                }

                if (selected_lens > 0) {
                    float current_fov = (float)cam.vfov;
                    float current_mm = 24.0f / (2.0f * tanf(current_fov * 0.5f * 3.14159f / 180.0f));
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.8f, 1.0f), "%.0fmm", current_mm);
                }
                float fov = (float)cam.vfov;
                // FOV with ◇ key button
                // FOV with Smart Slider
                bool fovKeyed = isCamKeyed(false, false, true, false, false);
                if (SceneUI::DrawSmartFloat("fov", "FOV", &fov, 10.0f, 120.0f, "%.1f", fovKeyed,
                    [&](){ insertCamKey("FOV", false, false, true, false, false); }, 16)) {
                    cam.vfov = fov;
                    cam.fov = fov;
                    cam.update_camera_vectors();
                    selected_lens = 0; // Reset to Custom
                    cam.markDirty();
                }
                

                

                ImGui::Separator();

                // ═══════════════════════════════════════════════════════════════════════════
                // F-STOP PRESETS - Photographer-friendly aperture selection
                // ═══════════════════════════════════════════════════════════════════════════
                // ═══════════════════════════════════════════════════════════════════════════
                // F-STOP SELECTION (Dynamic based on Lens Body)
                // ═══════════════════════════════════════════════════════════════════════════

                // 1. Determine Limits from selected lens
                float limit_min_f = 0.5f;    // Widest (Smallest number) - Custom default
                float limit_max_f = 128.0f;  // Tightest (Largest number) - Custom default

                if (cam.lens_preset_index > 0 && cam.lens_preset_index < (int)CameraPresets::LENS_PRESET_COUNT) {
                    limit_min_f = CameraPresets::LENS_PRESETS[cam.lens_preset_index].max_aperture;
                    limit_max_f = CameraPresets::LENS_PRESETS[cam.lens_preset_index].min_aperture;
                }

                // Auto-fix if current aperture is out of bounds (optional, or just clamp display)
                // ...

                // 2. Combo Box (Presets Filtering)
                if (ImGui::BeginCombo("F-Stop Preset", CameraPresets::FSTOP_PRESETS[cam.fstop_preset_index].name)) {
                    for (size_t i = 0; i < CameraPresets::FSTOP_PRESET_COUNT; ++i) {
                        float f_val = CameraPresets::FSTOP_PRESETS[i].f_number;
                        
                        // Filter: Skip if out of lens range (Always show Custom=0)
                        if (i > 0 && (f_val < limit_min_f - 0.01f || f_val > limit_max_f + 0.01f)) {
                            continue;
                        }

                        bool is_selected = (cam.fstop_preset_index == (int)i);
                        if (ImGui::Selectable(CameraPresets::FSTOP_PRESETS[i].name, is_selected)) {
                            cam.fstop_preset_index = (int)i;

                            // Apply Preset
                            if (cam.fstop_preset_index > 0) {
                                // Calculate aperture diameter: D = f / N
                                // Ensure focal length is updated
                                float f_mm = (cam.focal_length_mm > 1.0f) ? cam.focal_length_mm : 50.0f;
                                // Revert to using the preset's calibrated aperture_value
                                cam.aperture = CameraPresets::FSTOP_PRESETS[i].aperture_value;
                                cam.lens_radius = cam.aperture * 0.5f;
                            }
                            cam.update_camera_vectors();
                            
                            cam.markDirty();
                        }
                        if (is_selected) ImGui::SetItemDefaultFocus();
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s (f/%.1f)", CameraPresets::FSTOP_PRESETS[i].description, f_val);
                    }
                    ImGui::EndCombo();
                }

                // 3. Manual F-Stop Slider
                bool apKeyed = isCamKeyed(false, false, false, false, true);
                if (KeyframeButton("##CAp", apKeyed)) { insertCamKey("Aperture", false, false, false, false, true); }
                if (ImGui::IsItemHovered()) ImGui::SetTooltip(apKeyed ? "REMOVE Aperture key" : "ADD Aperture key");
                ImGui::SameLine();
                
                // Calculate current f-stop from physical aperture
                float calc_f_mm = (cam.focal_length_mm > 1.0f) ? cam.focal_length_mm : 50.0f;
                float current_f = (cam.aperture > 0.001f) ? (calc_f_mm / cam.aperture) : limit_max_f;
                
                // Clamp for display safely
                current_f = std::max(limit_min_f, std::min(current_f, limit_max_f));

                if (SceneUI::DrawSmartFloat("fstop", "F-Stop", &current_f, limit_min_f, limit_max_f, "f/%.2f", apKeyed,
                    [&](){ insertCamKey("Aperture", false, false, false, false, true); }, 16)) {
                    cam.fstop_preset_index = 0; // Reset to Custom
                    // Scale calculated aperture to match preset units (approx 0.01 scale factor)
                    cam.aperture = (calc_f_mm / current_f) * 0.01f; 
                    cam.lens_radius = cam.aperture * 0.5f;

                    cam.markDirty();
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
                        cam.markDirty();
                    }

                    // 2. Physical Lens Settings (mm)
                    bool lens_changed = false;
                    if (ImGui::Checkbox("Physical Lens (mm)", &cam.use_physical_lens)) {
                        lens_changed = true;
                    }

                    if (cam.use_physical_lens) {
                        // Focal Length
                        if (SceneUI::DrawSmartFloat("flen", "Focal Len", &cam.focal_length_mm, 10.0f, 200.0f, "%.1f mm", false, nullptr, 12)) {
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
                        cam.markDirty();
                    }

                    // Logic to update FOV from Lens settings
                    if (lens_changed && cam.use_physical_lens) {
                        // FOV = 2 * atan(SensorHeight / (2 * FocalLength))
                        // We use Sensor Height for VFOV to match standard vertical FOV usage in ray tracer
                        float vfov_rad = 2.0f * std::atan((cam.sensor_height_mm * 0.5f) / cam.focal_length_mm);
                        cam.vfov = vfov_rad * (180.0f / 3.14159265f);
                        cam.fov = cam.vfov;

                        cam.update_camera_vectors();
                        cam.markDirty();
                    }
                    ImGui::TreePop();
                }

                // Focus Distance with ◇ key button and Pick Focus button
                bool focKeyed = isCamKeyed(false, false, false, true, false);
                if (SceneUI::DrawSmartFloat("dist", "Focus Dist", &cam.focus_dist, 0.1f, 100.0f, "%.2f", focKeyed,
                    [&](){ insertCamKey("Focus", false, false, false, true, false); }, 16)) {
                    // Changed
                }             // Pick Focus mode - sets focus to clicked object distance
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
                if (SceneUI::DrawSmartFloat("msens", "Mouse Sens", &ctx.render_settings.mouse_sensitivity, 0.01f, 5.0f, "%.3f", false, nullptr, 12)) {
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

                    // ISO Settings (Dynamic based on Body)
                    int iso_min = 50;
                    int iso_max = 819200;
                    if (cam.body_preset_index > 0 && cam.body_preset_index < (int)CameraPresets::CAMERA_BODY_COUNT) {
                        iso_min = CameraPresets::CAMERA_BODIES[cam.body_preset_index].min_iso;
                        iso_max = CameraPresets::CAMERA_BODIES[cam.body_preset_index].max_iso;
                    }

                    ImGui::PushItemWidth(120);
                    // Find current preset name or show "Custom" if index is invalid/-1
                    const char* iso_preview = (cam.iso_preset_index >= 0 && cam.iso_preset_index < (int)CameraPresets::ISO_PRESET_COUNT) 
                         ? CameraPresets::ISO_PRESETS[cam.iso_preset_index].name 
                         : "Custom";

                    if (ImGui::BeginCombo("ISO", iso_preview)) {
                        for (size_t i = 0; i < CameraPresets::ISO_PRESET_COUNT; ++i) {
                            int p_iso = CameraPresets::ISO_PRESETS[i].iso_value;
                            // Filter based on body limits
                            if (p_iso < iso_min || p_iso > iso_max) continue;

                            bool is_selected = (cam.iso_preset_index == (int)i);
                            if (ImGui::Selectable(CameraPresets::ISO_PRESETS[i].name, is_selected)) {
                                cam.iso_preset_index = (int)i;
                                cam.iso = p_iso;
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
                    ImGui::SameLine();
                    
                    // Manual ISO Drag
                    // Clamp visual range but allow manual input
                    int iso_val = cam.iso;
                    if (ImGui::DragInt("##ISOVal", &iso_val, 10.0f, iso_min, iso_max, "ISO %d")) {
                         // Clamp result
                         if (iso_val < iso_min) iso_val = iso_min;
                         if (iso_val > iso_max) iso_val = iso_max;
                         cam.iso = iso_val;
                         
                         exposure_changed = true;
                         // Try to match preset, else set to custom (-1)
                         cam.iso_preset_index = -1; 
                         for (size_t i=0; i<CameraPresets::ISO_PRESET_COUNT; ++i) {
                             if (CameraPresets::ISO_PRESETS[i].iso_value == cam.iso) {
                                 cam.iso_preset_index = (int)i;
                                 break;
                             }
                         }
                    }
                    ImGui::PopItemWidth();

                    // Shutter Speed Preset  
                    ImGui::PushItemWidth(180);
                    if (ImGui::BeginCombo("Shutter", CameraPresets::SHUTTER_SPEED_PRESETS[cam.shutter_preset_index].name)) {
                        for (size_t i = 0; i < CameraPresets::SHUTTER_SPEED_PRESET_COUNT; ++i) {
                            bool is_selected = (cam.shutter_preset_index == (int)i);
                            if (ImGui::Selectable(CameraPresets::SHUTTER_SPEED_PRESETS[i].name, is_selected)) {
                                cam.shutter_preset_index = (int)i;
                                cam.shutter_speed = 1.0f / CameraPresets::SHUTTER_SPEED_PRESETS[i].speed_seconds; // Optional sync
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
                    // Use literal ISO value (cam.iso) for EV calc, not just preset index multiplier
                    float iso_mult = (float)cam.iso / 100.0f; 
                    float shutter_time = CameraPresets::SHUTTER_SPEED_PRESETS[cam.shutter_preset_index].speed_seconds; 
                    float aperture_area = cam.aperture * cam.aperture;
                    cam.calculated_ev = log2f(100.0f / (iso_mult * shutter_time * aperture_area + 0.0001f)) + cam.ev_compensation;

                    ImGui::TextColored(ImVec4(0.6f, 0.9f, 0.7f, 1.0f), "Exposure: %.1f EV", cam.calculated_ev);

                    if (cam.auto_exposure) {
                        ImGui::SameLine();
                        ImGui::TextDisabled("(Auto Active)");
                    }
                }
                
                } // END: if (show_pro_controls) - Pro & Cinema mode controls
                
                // ═══════════════════════════════════════════════════════════════════════════
                // AUTO MODE: Simplified Controls (Only FOV and Focus)
                // ═══════════════════════════════════════════════════════════════════════════
                if (cam.camera_mode == CameraMode::Auto) {
                    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.5f, 1.0f), "Basic Settings");
                    ImGui::Spacing();
                    
                    // Simple FOV slider
                    float fov = (float)cam.vfov;
                    if (ImGui::SliderFloat("Field of View", &fov, 20.0f, 100.0f, "%.0f°")) {
                        cam.vfov = fov;
                        cam.fov = fov;
                        cam.update_camera_vectors();
                        if (ctx.optix_gpu_ptr && g_hasOptix) {
                            ctx.optix_gpu_ptr->setCameraParams(cam);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Wider = More visible area\nNarrower = Zoom in effect");
                    
                    // Simple Focus Distance
                    if (ImGui::SliderFloat("Focus Distance", &cam.focus_dist, 0.5f, 50.0f, "%.1f m")) {
                        if (ctx.optix_gpu_ptr && g_hasOptix) {
                            ctx.optix_gpu_ptr->setCameraParams(cam);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Distance to the focused object");
                    
                    // Simple Depth of Field
                    float dof_strength = 1.0f - std::min(1.0f, cam.aperture * 20.0f);  // Invert for intuitive slider
                    if (ImGui::SliderFloat("Background Blur", &dof_strength, 0.0f, 1.0f, "%.2f")) {
                        cam.aperture = (1.0f - dof_strength) * 0.05f;
                        cam.lens_radius = cam.aperture * 0.5f;
                        if (ctx.optix_gpu_ptr && g_hasOptix) {
                            ctx.optix_gpu_ptr->setCameraParams(cam);
                            ctx.optix_gpu_ptr->resetAccumulation();
                        }
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("0 = Sharp everywhere\n1 = Blurry background (Portrait mode)");
                    
                    ImGui::Spacing();
                    ImGui::TextDisabled("Switch to Pro mode for full control");
                }
                
                // ═══════════════════════════════════════════════════════════════════════════
                // CINEMA MODE EFFECTS (Only visible in Cinema mode)
                // ═══════════════════════════════════════════════════════════════════════════
                if (cam.camera_mode == CameraMode::Cinema) {
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.3f, 1.0f), "Cinema Effects");
                    ImGui::Spacing();
                        
                        bool cinema_changed = false;
                        
                        // Lens Quality slider
                        cinema_changed |= ImGui::SliderFloat("Lens Quality", &cam.lens_quality, 0.0f, 1.0f, "%.2f");
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                            "0.0 = Vintage/Budget lens (more aberrations)\n"
                            "1.0 = Perfect optical design (minimal aberrations)");
                        
                        // Auto-calculate toggle
                        cinema_changed |= ImGui::Checkbox("Auto Lens Characteristics", &cam.auto_lens_characteristics);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                            "Calculate vignetting, CA, and distortion\n"
                            "from lens quality, focal length, and aperture");
                        
                        // If auto is on, recalculate and show the values read-only
                        if (cam.auto_lens_characteristics) {
                            cam.calculateLensCharacteristics();
                            ImGui::TextDisabled("Vignette: %.1f%% | CA: %.3f | Dist: %.2f",
                                cam.vignetting_amount * 100.0f,
                                cam.chromatic_aberration,
                                cam.distortion);
                        } else {
                            // Manual controls
                            ImGui::Spacing();
                            ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.3f, 1.0f), "Manual Overrides");
                            
                            // Chromatic Aberration
                            cinema_changed |= ImGui::Checkbox("Chromatic Aberration##CA", &cam.enable_chromatic_aberration);
                            if (cam.enable_chromatic_aberration) {
                                ImGui::Indent();
                                cinema_changed |= ImGui::SliderFloat("Amount##CA", &cam.chromatic_aberration, 0.0f, 0.05f, "%.3f");
                                ImGui::Unindent();
                            }
                            
                            // Vignetting
                            cinema_changed |= ImGui::Checkbox("Vignetting##Vig", &cam.enable_vignetting);
                            if (cam.enable_vignetting) {
                                ImGui::Indent();
                                cinema_changed |= ImGui::SliderFloat("Amount##Vig", &cam.vignetting_amount, 0.0f, 1.0f, "%.2f");
                                cinema_changed |= ImGui::SliderFloat("Falloff##Vig", &cam.vignetting_falloff, 1.0f, 4.0f, "%.1f");
                                ImGui::Unindent();
                            }
                        }
                        
                        ImGui::Spacing();
                        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Camera Motion");
                        
                        // Camera Shake
                        cinema_changed |= ImGui::Checkbox("Camera Shake (Handheld)##Shake", &cam.enable_camera_shake);
                        if (cam.enable_camera_shake) {
                            ImGui::Indent();
                            cinema_changed |= ImGui::SliderFloat("Intensity##Shake", &cam.shake_intensity, 0.0f, 1.0f, "%.2f");
                            cinema_changed |= ImGui::SliderFloat("Frequency##Shake", &cam.shake_frequency, 2.0f, 15.0f, "%.1f Hz");
                            
                            const char* skill_names[] = { "Amateur", "Intermediate", "Professional", "Expert" };
                            int skill = static_cast<int>(cam.operator_skill);
                            if (ImGui::Combo("Operator##Skill", &skill, skill_names, IM_ARRAYSIZE(skill_names))) {
                                cam.operator_skill = static_cast<Camera::OperatorSkill>(skill);
                                cinema_changed = true;
                            }
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                                "Amateur: Maximum shake, unsteady hands\n"
                                "Intermediate: Moderate shake\n"
                                "Professional: Minimal shake, trained operator\n"
                                "Expert: Almost no shake, documentary/cinema pro");
                            
                            // IBIS (In-Body Image Stabilization)
                            cinema_changed |= ImGui::Checkbox("IBIS##Stab", &cam.ibis_enabled);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                                "In-Body Image Stabilization (IBIS)\n\n"
                                "Simulates sensor-shift stabilization found in\n"
                                "modern mirrorless cameras (Sony, Canon, etc.)\n\n"
                                "Reduces shake by compensating sensor movement.\n"
                                "Higher stops = more stabilization.\n\n"
                                "Typical values:\n"
                                "  3-4 stops: Basic IBIS\n"
                                "  5-6 stops: Advanced (Sony A7)\n"
                                "  7-8 stops: Professional (Olympus OM-1)");
                            if (cam.ibis_enabled) {
                                ImGui::SameLine();
                                ImGui::PushItemWidth(60);
                                cinema_changed |= ImGui::DragFloat("##IBISStops", &cam.ibis_effectiveness, 0.5f, 1.0f, 8.0f, "%.1f stops");
                                ImGui::PopItemWidth();
                            }
                            
                            ImGui::Spacing();
                            
                            // Focus Drift
                            cinema_changed |= ImGui::Checkbox("Focus Drift##FocDrift", &cam.enable_focus_drift);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                                "When camera shakes, focus plane also moves.\n"
                                "This simulates real handheld focus breathing.\n\n"
                                "More noticeable with shallow DOF (wide aperture).");
                            if (cam.enable_focus_drift) {
                                ImGui::SameLine();
                                ImGui::PushItemWidth(60);
                                cinema_changed |= ImGui::DragFloat("##FocDriftAmt", &cam.focus_drift_amount, 0.01f, 0.0f, 0.5f, "%.2f m");
                                ImGui::PopItemWidth();
                            }
                            
                            ImGui::Unindent();
                        }
                        
                        if (cinema_changed) {
                            if (ctx.optix_gpu_ptr && g_hasOptix) {
                                ctx.optix_gpu_ptr->setCameraParams(cam);
                                ctx.optix_gpu_ptr->resetAccumulation();
                            }
                            ctx.renderer.resetCPUAccumulation();
                            ProjectManager::getInstance().markModified();
                        }
                } // END: Cinema mode effects

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
                
                // Pop camera properties styling
                ImGui::PopStyleVar(2);
                ImGui::PopStyleColor(6);
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

                // Intensity with Smart Slider
                bool intKeyed = isLightKeyed(false, false, true, false);
                if (SceneUI::DrawSmartFloat("lint", "Intensity", &light.intensity, 0.5f, 1000.0f, "%.1f", intKeyed,
                    [&](){ insertLightKey("Intensity", false, false, true, false); }, 16)) {
                    light_changed = true;
                }

                if (light.type() == LightType::Point || light.type() == LightType::Directional) {
                    if (SceneUI::DrawSmartFloat("lrad", "Radius", &light.radius, 0.01f, 100.0f, "%.2f", false, nullptr, 16)) light_changed = true;
                }

                if (auto sl = dynamic_cast<SpotLight*>(&light)) {
                    float angle = sl->getAngleDegrees();
                    if (SceneUI::DrawSmartFloat("lcne", "Cone Angle", &angle, 1.0f, 89.0f, "%.1f", false, nullptr, 16)) {
                        sl->setAngleDegrees(angle);
                        light_changed = true;
                    }
                }
                else if (auto al = dynamic_cast<AreaLight*>(&light)) {
                    if (SceneUI::DrawSmartFloat("awid", "Width", &al->width, 0.01f, 100.0f, "%.2f", false, nullptr, 16)) {
                        al->u = al->u.normalize() * al->width;
                        light_changed = true;
                    }
                    if (SceneUI::DrawSmartFloat("ahei", "Height", &al->height, 0.01f, 100.0f, "%.2f", false, nullptr, 16)) {
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
        // MATERIAL EDITOR (Only for Objects - not Camera, Light, or World)
        // ─────────────────────────────────────────────────────────────────────────
        if (sel.selected.type == SelectableType::Object) {
            ImGui::Separator();
            if (ImGui::CollapsingHeader("Material Editor", ImGuiTreeNodeFlags_DefaultOpen)) {
                drawMaterialPanel(ctx);
            }
        }
    }
    // Light Gizmos and Transform Gizmos moved to main draw() loop for all tabs
}

