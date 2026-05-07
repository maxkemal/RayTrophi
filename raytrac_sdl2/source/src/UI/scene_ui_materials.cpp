// ===============================================================================
// SCENE UI - MATERIAL EDITOR
// ===============================================================================
// This file handles the Material properties panel.
// ===============================================================================

#include "scene_ui.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "imgui.h"
#include "MaterialManager.h"
#include "PBRMaterialSnapshot.h"
#include "PrincipledBSDF.h"
#include "Volumetric.h"
#include "scene_data.h"
#include "VDBVolumeManager.h"
#include "HittableInstance.h"
#include "Texture.h"
#include "Triangle.h"
#include "TextureCompressionCache.h"
#include "TerrainManager.h"
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>

// Static editor instance
static bool showMatNodeEditor = false;

// Cache tag map: key = "texname|type_int", value = "BC7"/"BC5"/"BC4"/"" (empty = no managed cache).
// Populated lazily per texture slot; cleared when a bake completes.
static std::unordered_map<std::string, std::string> s_cacheTagMap;

void invalidateTextureCacheTagCache() {
    s_cacheTagMap.clear();
}

void projectObjectUVsFromView(UIContext& ctx, SceneUI& ui, const std::string& nodeName);

void SceneUI::resetMaterialUI() {
    showMatNodeEditor = false;
    uv_workflow_cache_dirty = true;
}

// ===============================================================================
// MATERIAL & TEXTURE EDITOR PANEL
// ===============================================================================
// ===============================================================================
// IMPLEMENTATION: manageTextureGraveyard
// ===============================================================================
extern bool g_optix_rebuild_pending; // Ensure this is available

void SceneUI::manageTextureGraveyard() {
    // Only clear invalid textures when NO rebuild is pending
    if (!g_optix_rebuild_pending && !texture_graveyard.empty()) {
        texture_graveyard.clear();
        // SCENE_LOG_INFO("Texture graveyard cleared. Old textures released.");
    }
}

void SceneUI::drawMaterialPanel(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;

    UIWidgets::PushControlSurfaceStyle(ImVec4(0.98f, 0.72f, 0.42f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 14.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 14.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_GrabRounding, 14.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, 14.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(8.0f, 6.0f));
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.10f, 0.115f, 0.14f, 0.94f));
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.13f, 0.145f, 0.17f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.16f, 0.18f, 0.215f, 0.99f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.19f, 0.215f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.20f, 0.16f, 0.11f, 0.96f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.26f, 0.20f, 0.13f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.30f, 0.24f, 0.16f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImVec4(1.0f, 0.78f, 0.48f, 0.95f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4(1.0f, 0.88f, 0.62f, 1.0f));

    // Only show for selected objects
    if (sel.selected.type != SelectableType::Object || !sel.selected.object) {
        ImGui::TextDisabled("Select an object to edit materials");
        ImGui::PopStyleColor(9);
        ImGui::PopStyleVar(6);
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    std::string obj_name = sel.selected.name;
    if (obj_name.empty()) {
        ImGui::TextDisabled("Unnamed Object");
        ImGui::PopStyleColor(9);
        ImGui::PopStyleVar(6);
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    // Ensure mesh cache is valid to find all triangles of this object
    if (!mesh_cache_valid) {
        rebuildMeshCache(ctx.scene.world.objects);
    }

    auto cache_it = mesh_cache.find(obj_name);
    if (cache_it == mesh_cache.end()) {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Object mesh data not found in cache.");
        ImGui::PopStyleColor(9);
        ImGui::PopStyleVar(6);
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    // �������������������������������������������������������������������������
    // 1. GET MATERIAL SLOTS FROM CACHE (O(1) lookup, not O(N) triangle scan!)
    // �������������������������������������������������������������������������
    // Use cached material slots instead of scanning all triangles every frame
    auto slots_it = material_slots_cache.find(obj_name);
    if (slots_it == material_slots_cache.end()) {
        ImGui::TextDisabled("Material data not cached. Cache may need rebuild.");
        ImGui::PopStyleColor(9);
        ImGui::PopStyleVar(6);
        UIWidgets::PopControlSurfaceStyle();
        return;
    }
    
    const std::vector<uint16_t>& used_material_ids = slots_it->second;

    if (used_material_ids.empty()) {
        ImGui::TextDisabled("No geometry/materials found.");
        ImGui::PopStyleColor(9);
        ImGui::PopStyleVar(6);
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    // �������������������������������������������������������������������������
    // 2. SLOT SELECTION UI
    // �������������������������������������������������������������������������
    static int active_slot_index = 0;

    // Safety: Reset slot index if out of bounds (e.g. material removed or selection changed)
    if (active_slot_index >= (int)used_material_ids.size()) {
        active_slot_index = 0;
    }

    // Reset slot index when selection changes (rudimentary check by name change)
    static std::string last_selected_obj_name = "";
    if (last_selected_obj_name != obj_name) {
        active_slot_index = 0;
        last_selected_obj_name = obj_name;
    }

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.11f, 0.13f, 0.16f, 0.94f));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 12.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    ImGui::BeginChild("MaterialHeaderCard", ImVec2(0, 64), true);
    ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.52f, 1.0f), "%s", obj_name.c_str());
    ImGui::TextDisabled("Slot assignment and material editing for the active object.");
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();

    if (UIWidgets::BeginSection("Material Slot Stack", ImVec4(1.0f, 0.72f, 0.42f, 1.0f))) {
        if (ImGui::BeginListBox("##MaterialSlots", ImVec2(-FLT_MIN, 5 * ImGui::GetTextLineHeightWithSpacing()))) {
            for (int i = 0; i < (int)used_material_ids.size(); i++) {
                uint16_t mat_id = used_material_ids[i];
                Material* mat = MaterialManager::getInstance().getMaterial(mat_id);

                std::string slot_label;
                if (mat) {
                    slot_label = std::string("Slot ") + std::to_string(i) + ": " + mat->materialName;
                }
                else {
                    slot_label = std::string("Slot ") + std::to_string(i) + ": [Null]";
                }

                const bool is_selected = (active_slot_index == i);
                if (ImGui::Selectable(slot_label.c_str(), is_selected)) {
                    active_slot_index = i;
                }

                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndListBox();
        }
        UIWidgets::EndSection();
    }
    // Current pointer to the active material
    uint16_t active_mat_id = (active_slot_index < (int)used_material_ids.size()) ? used_material_ids[active_slot_index] : MaterialManager::INVALID_MATERIAL_ID;
    Material* active_mat_ptr = MaterialManager::getInstance().getMaterial(active_mat_id);
    std::string current_mat_name = active_mat_ptr ? active_mat_ptr->materialName : "None";
    static std::string last_logged_slot_debug_obj_name = "";
    if (last_logged_slot_debug_obj_name != obj_name) {
        last_logged_slot_debug_obj_name = obj_name;

    }

    // �������������������������������������������������������������������������
    // 3. ASSIGN MATERIAL TO ACTIVE SLOT
    // �������������������������������������������������������������������������
    ImGui::Separator();

    // --- MATERIAL KEYFRAME BUTTON ---

    // Custom Keyframe Button Helper (Diamond Shape)
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

        ImVec2 p[4] = { ImVec2(cx, cy - r), ImVec2(cx + r, cy), ImVec2(cx, cy + r), ImVec2(cx - r, cy) };
        dl->AddQuadFilled(p[0], p[1], p[2], p[3], bg);
        dl->AddQuad(p[0], p[1], p[2], p[3], border, 1.0f);

        ImGui::PopID();
        return clicked;
        };

    // Check if material property is keyed
    auto isMatKeyed = [&](uint16_t mat_id, bool check_alb, bool check_opac, bool check_rgh, bool check_met, bool check_ems, bool check_trn, bool check_ior, bool check_spec) {
        std::string obj_n = sel.selected.name;
        auto it = ctx.scene.timeline.tracks.find(obj_n);
        if (it == ctx.scene.timeline.tracks.end()) return false;

        int cf = ctx.render_settings.animation_current_frame;
        for (auto& kf : it->second.keyframes) {
            if (kf.frame == cf && kf.has_material && kf.material.material_id == mat_id) {
                if (check_alb && kf.material.has_albedo) return true;
                if (check_opac && kf.material.has_opacity) return true; // NEW
                if (check_rgh && kf.material.has_roughness) return true;
                if (check_met && kf.material.has_metallic) return true;
                if (check_ems && kf.material.has_emission) return true;
                if (check_trn && kf.material.has_transmission) return true;
                if (check_ior && kf.material.has_ior) return true;
                if (check_spec && kf.material.has_specular) return true;
            }
        }
        return false;
        };

    // Insert Material Property Key (with toggle behavior)
    auto insertMatPropKey = [&](Material* mat, uint16_t mat_id, bool k_alb, bool k_opac, bool k_rgh, bool k_met, bool k_ems, bool k_trn, bool k_ior, bool k_spec) {
        int current_frame = ctx.render_settings.animation_current_frame;
        std::string obj_n = sel.selected.name;

        auto& track = ctx.scene.timeline.tracks[obj_n];

        // TOGGLE BEHAVIOR: Check if property is already keyed, if so remove it
        for (auto it = track.keyframes.begin(); it != track.keyframes.end(); ++it) {
            if (it->frame == current_frame && it->has_material && it->material.material_id == mat_id) {
                bool removed = false;
                if (k_alb && it->material.has_albedo) { it->material.has_albedo = false; removed = true; }
                if (k_opac && it->material.has_opacity) { it->material.has_opacity = false; removed = true; }
                if (k_rgh && it->material.has_roughness) { it->material.has_roughness = false; removed = true; }
                if (k_met && it->material.has_metallic) { it->material.has_metallic = false; removed = true; }
                if (k_ems && it->material.has_emission) { it->material.has_emission = false; removed = true; }
                if (k_trn && it->material.has_transmission) { it->material.has_transmission = false; removed = true; }
                if (k_ior && it->material.has_ior) { it->material.has_ior = false; removed = true; }
                if (k_spec && it->material.has_specular) { it->material.has_specular = false; removed = true; }

                if (removed) {
                    // Check if keyframe is now empty
                    bool hasAny = it->material.has_albedo || it->material.has_opacity || it->material.has_roughness ||
                        it->material.has_metallic || it->material.has_emission || it->material.has_transmission ||
                        it->material.has_ior || it->material.has_specular;
                    if (!hasAny) {
                        it->has_material = false;
                        if (!it->has_transform && !it->has_camera && !it->has_light && !it->has_world) {
                            track.keyframes.erase(it);
                        }
                    }
                    SCENE_LOG_INFO("Removed material property keyframe.");
                    return;
                }
            }
        }

        // Not keyed - add new keyframe
        if (mat && mat->gpuMaterial) {
            Keyframe kf(current_frame);
            kf.has_material = true;
            kf.material = MaterialKeyframe(*mat->gpuMaterial);
            kf.material.material_id = mat_id;

            // Set Flags
            kf.material.has_albedo = k_alb;
            kf.material.has_opacity = k_opac;
            kf.material.has_roughness = k_rgh;
            kf.material.has_metallic = k_met;
            kf.material.has_emission = k_ems;
            kf.material.has_transmission = k_trn;
            kf.material.has_ior = k_ior;
            kf.material.has_specular = k_spec;

            bool found = false;
            for (auto& existing : track.keyframes) {
                if (existing.frame == current_frame) {
                    existing.has_material = true;
                    // Merge Flags & Values
                    if (existing.material.material_id == mat_id) {
                        // Same material slot, merge fields
                        if (k_alb) { existing.material.has_albedo = true; existing.material.albedo = kf.material.albedo; }
                        if (k_opac) { existing.material.has_opacity = true; existing.material.opacity = kf.material.opacity; }
                        if (k_rgh) { existing.material.has_roughness = true; existing.material.roughness = kf.material.roughness; }
                        if (k_met) { existing.material.has_metallic = true; existing.material.metallic = kf.material.metallic; }
                        if (k_ems) { existing.material.has_emission = true; existing.material.emission = kf.material.emission; existing.material.emission_strength = kf.material.emission_strength; }
                        if (k_trn) { existing.material.has_transmission = true; existing.material.transmission = kf.material.transmission; }
                        if (k_ior) { existing.material.has_ior = true; existing.material.ior = kf.material.ior; }
                        if (k_spec) { existing.material.has_specular = true; existing.material.specular = kf.material.specular; }
                    }
                    else {
                        existing.material = kf.material;
                    }
                    found = true;
                    break;
                }
            }
            if (!found) {
                track.addKeyframe(kf);
            }
            SCENE_LOG_INFO("Added material property keyframe.");
        }
        };

    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.52f, 1.0f), "Slot Assignment & Actions");
    if (UIWidgets::IconActionButton("MaterialKeyAll", UIWidgets::IconType::AddKey, "",
        false, ImVec4(1.0f, 0.72f, 0.42f, 1.0f), ImVec2(42.0f, 38.0f),
        "Insert a material keyframe for the active slot at the current frame.")) {
        // Insert material keyframe at current frame
        int current_frame = ctx.render_settings.animation_current_frame;

        // Get GpuMaterial from active material  
        if (active_mat_ptr && active_mat_ptr->gpuMaterial) {
            // Create keyframe with material data
            Keyframe kf(current_frame);
            kf.has_material = true;
            kf.material = MaterialKeyframe(*active_mat_ptr->gpuMaterial);
            kf.material.material_id = active_mat_id;  // Store which material slot this is

            // Add to timeline
            auto& track = ctx.scene.timeline.tracks[obj_name];

            // Check if keyframe exists at this frame
            Keyframe* existing = track.getKeyframeAt(current_frame);
            if (existing) {
                existing->has_material = true;
                existing->material = kf.material;
            }
            else {
                track.addKeyframe(kf);
            }

            SCENE_LOG_INFO("Material keyframe inserted for '" + obj_name + "' at frame " + std::to_string(current_frame));
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Active Slot");

    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 110); // Space for +New buttons (adjusted for +S and +V)
    if (ImGui::BeginCombo("##SlotAssignment", current_mat_name.c_str())) {
        auto& mgr = MaterialManager::getInstance();
        const auto& all_materials = mgr.getAllMaterials();

        for (size_t i = 0; i < all_materials.size(); i++) {
            if (!all_materials[i]) continue;

            uint16_t mat_id = (uint16_t)i;
            bool is_selected = (mat_id == active_mat_id);
            std::string label = all_materials[i]->materialName;
            if (label.empty()) label = "Mat #" + std::to_string(i);

            ImGui::PushID((int)mat_id);
            if (ImGui::Selectable(label.c_str(), is_selected)) {
                if ((uint16_t)i != active_mat_id) {
                    // REPLACE MATERIAL LOGIC:
                    // Find all triangles that WERE using 'active_mat_id' (the old material for this slot)
                    // and update them to use 'i' (the new material).
                    // This preserves other slots.

                    int count_replaced = 0;
                    for (auto& pair : cache_it->second) {
                        if (pair.second->getMaterialID() == active_mat_id) {
                            pair.second->setMaterialID((uint16_t)i);
                            count_replaced++;
                        }
                    }

                    // UPDATE CACHE IN-PLACE to prevent UI reversion
                    if (active_slot_index < (int)slots_it->second.size()) {
                        slots_it->second[active_slot_index] = (uint16_t)i;
                    }

                    SCENE_LOG_INFO("Replaced material in Slot " + std::to_string(active_slot_index) +
                        " (ID: " + std::to_string(active_mat_id) + " -> " + std::to_string(i) +
                        "). Triangles updated: " + std::to_string(count_replaced));

                    // SYNC WITH TERRAIN MANAGER
                    // If this object is a terrain, update the redundant material_id in TerrainObject
                    // so it persists across rebuilds (erosion, resizing, etc.)
                   if (!cache_it->second.empty()) {
                        auto first_tri = cache_it->second[0].second;
                        if (first_tri && first_tri->terrain_id != -1) {
                            TerrainObject* terrain = TerrainManager::getInstance().getTerrain(first_tri->terrain_id);
                            if (terrain) {
                                terrain->material_id = (uint16_t)i;
                                // Also update name suffix if strictly following naming convention? Not needed.
                                SCENE_LOG_INFO("Synced TerrainObject Material ID for: " + terrain->name);
                            }
                        }
                    }

                    // Trigger Updates (Optimized)
                    ctx.renderer.resetCPUAccumulation();
                    
                    // Update bindings on GPU (Fast Path) - CRITICAL FIX
                    ctx.renderer.updateMeshMaterialBinding(ctx.scene, obj_name, active_mat_id, (uint16_t)i);

                    if (ctx.backend_ptr) {
                        ctx.renderer.updateBackendMaterial(ctx.scene, static_cast<uint16_t>(i));
                        ctx.backend_ptr->resetAccumulation();
                    }
                    g_ProjectManager.markModified();
                }
            }
            if (is_selected) ImGui::SetItemDefaultFocus();
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    // �������������������������������������������������������������������������
    // SHORTCUTS (New Surface / Volume) - affecting Active Slot
    // �������������������������������������������������������������������������
    UIWidgets::Divider();
    if (ImGui::Button("+ Surface")) {
        ImGui::OpenPopup("NewSurfPopup");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Create a new surface material and assign it to the active slot.");
    }

    ImGui::SameLine();
    if (ImGui::Button("+ Volume")) {
        ImGui::OpenPopup("NewVolPopup");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Create a new volumetric material and assign it to the active slot.");
    }
    

    if (ImGui::BeginPopup("NewSurfPopup")) {
        ImGui::Text("Create New Surface");
        if (ImGui::Selectable("Create & Assign")) {
            auto& mgr = MaterialManager::getInstance();
            auto new_mat = std::make_shared<PrincipledBSDF>(Vec3(0.8f), 0.5f, 0.0f);
            std::string name = "Surface_" + std::to_string(mgr.getMaterialCount());
            new_mat->materialName = name;
            uint16_t new_id = mgr.addMaterial(name, new_mat);

            // Assign to current slot triangles
            for (auto& pair : cache_it->second) {
                if (pair.second->getMaterialID() == active_mat_id) {
                    pair.second->setMaterialID(new_id);
                }
            }

            // SYNC WITH TERRAIN MANAGER
            if (!cache_it->second.empty()) {
                auto first_tri = cache_it->second[0].second;
                if (first_tri && first_tri->terrain_id != -1) {
                    TerrainObject* terrain = TerrainManager::getInstance().getTerrain(first_tri->terrain_id);
                    if (terrain) {
                        terrain->material_id = new_id;
                        SCENE_LOG_INFO("Synced TerrainObject Material ID (New Surf) for: " + terrain->name);
                    }
                }
            }
            
            // UPDATE CACHE IN-PLACE
            if (active_slot_index < (int)slots_it->second.size()) {
                slots_it->second[active_slot_index] = new_id;
            }
            // Trigger updates (Optimized)
            ctx.renderer.resetCPUAccumulation();
            
            // Update bindings on GPU (Fast Path)
            ctx.renderer.updateMeshMaterialBinding(ctx.scene, obj_name, active_mat_id, new_id);

            if (ctx.backend_ptr) {
                ctx.renderer.updateBackendMaterial(ctx.scene, new_id);
                ctx.backend_ptr->resetAccumulation();
            }
            g_ProjectManager.markModified();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("NewVolPopup")) {
        ImGui::Text("Create New Volumetric");
        if (ImGui::Selectable("Create & Assign")) {
            auto& mgr = MaterialManager::getInstance();
            auto perlin = std::make_shared<Perlin>();
            auto new_mat = std::make_shared<Volumetric>(
                Vec3(0.8f),     // albedo
                1.0f,            // density
                0.1f,            // absorption
                0.5f,            // scattering
                Vec3(0.0f),     // emission
                perlin          // noise
            );
            std::string name = "Volume_" + std::to_string(mgr.getMaterialCount());
            new_mat->materialName = name;
            uint16_t new_id = mgr.addMaterial(name, new_mat);

            // Assign to current slot triangles
            for (auto& pair : cache_it->second) {
                if (pair.second->getMaterialID() == active_mat_id) {
                    pair.second->setMaterialID(new_id);
                }
            }

            // SYNC WITH TERRAIN MANAGER
            if (!cache_it->second.empty()) {
                auto first_tri = cache_it->second[0].second;
                if (first_tri && first_tri->terrain_id != -1) {
                    TerrainObject* terrain = TerrainManager::getInstance().getTerrain(first_tri->terrain_id);
                    if (terrain) {
                        terrain->material_id = new_id;
                        SCENE_LOG_INFO("Synced TerrainObject Material ID (New Vol) for: " + terrain->name);
                    }
                }
            }
            
            // UPDATE CACHE IN-PLACE
            if (active_slot_index < (int)slots_it->second.size()) {
                slots_it->second[active_slot_index] = new_id;
            }
            // Trigger updates (Optimized)
            ctx.renderer.resetCPUAccumulation();
            
            // Update bindings on GPU (Fast Path)
            ctx.renderer.updateMeshMaterialBinding(ctx.scene, obj_name, active_mat_id, new_id);

            if (ctx.backend_ptr) {
                ctx.renderer.updateBackendMaterial(ctx.scene, new_id);
                ctx.backend_ptr->resetAccumulation();
            }
            g_ProjectManager.markModified();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // �������������������������������������������������������������������������
    // MATERIAL EDITOR (Context-Aware for Active Slot)
    // �������������������������������������������������������������������������
    // Re-fetch active material in case it changed
    active_mat_id = used_material_ids[active_slot_index];
    // NOTE: If we just replaced the material, the used_material_ids list is somewhat stale 
    // until next frame, but since we updated the triangles, the 'active_mat_id' locally 
    // typically refers to the old ID unless found again. 
    // Ideally we should update the 'used_material_ids' vector immediately, but for now 
    // we can re-query the triangles or just wait for next frame UI refresh. 
    // To be safe, let's just use the pointer from the manager for the *assigned* ID.
    // However, we just changed the ID in the triangles. The 'used_material_ids' array 
    // still holds the OLD ID at index 'active_slot_index'.
    // We should fix this visual glitch by simply returning if we made a heavy change, 
    // or by updating the local array.

    // Let's rely on the frame refresh for perfect consistency, but try to show 
    // the potentially new material if we can.

    active_mat_ptr = MaterialManager::getInstance().getMaterial(active_mat_id);
    if (!active_mat_ptr) {
        ImGui::TextDisabled("Active material could not be resolved.");
        ImGui::PopStyleColor(9);
        ImGui::PopStyleVar(6);
        UIWidgets::PopControlSurfaceStyle();
        return;
    }

    // Check material type
    Volumetric* vol_mat = dynamic_cast<Volumetric*>(active_mat_ptr);
    PrincipledBSDF* pbsdf = dynamic_cast<PrincipledBSDF*>(active_mat_ptr);

    bool material_changed = false;
    if (vol_mat) {
        if (UIWidgets::BeginSection("Volume Source", ImVec4(0.82f, 0.66f, 1.0f, 1.0f))) {
            if (vol_mat->hasVDBVolume()) {
                 VDBVolumeData* vdb = VDBVolumeManager::getInstance().getVolume(vol_mat->getVDBVolumeID());
                 if (vdb) {
                     ImGui::Text("Loaded VDB: %s", vdb->name.c_str());
                     ImGui::TextDisabled("%s", vdb->filepath.c_str());
                     
                     if (ImGui::Button("Clear VDB")) {
                         vol_mat->setVDBVolumeID(-1);
                         vol_mat->setDensitySource(0); // Switch back to procedural
                         material_changed = true;
                     }

                     ImGui::Spacing();
                     ImGui::TextDisabled("Bounds: (%.1f, %.1f, %.1f) - (%.1f, %.1f, %.1f)", 
                         vdb->bbox_min[0], vdb->bbox_min[1], vdb->bbox_min[2],
                         vdb->bbox_max[0], vdb->bbox_max[1], vdb->bbox_max[2]);
                         
                     ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                     if (ImGui::Button("Fit Mesh to VDB")) {
                     // Get bounding box size and center
                     Vec3 min_pt(vdb->bbox_min[0], vdb->bbox_min[1], vdb->bbox_min[2]);
                     Vec3 max_pt(vdb->bbox_max[0], vdb->bbox_max[1], vdb->bbox_max[2]);
                     
                     Vec3 size = max_pt - min_pt;
                     Vec3 center = min_pt + size * 0.5f;
                     
                     // Update selected object using generic TransformHandle interface
                     Transform* transform_handle = sel.selected.object->getTransformPtr();
                     
                     if (transform_handle) {
                         // Assumes object is a unit cube originally (-0.5 to 0.5 or similar unit size)
                         
                         Matrix4x4 S = Matrix4x4::scaling(size);
                         Matrix4x4 T = Matrix4x4::translation(center);
                         
                         // Apply Scale then Translation
                         Matrix4x4 new_transform = T * S;
                         
                         transform_handle->setPivotMatrix(new_transform);
                         
                         // CRITICAL: A mesh is composed of multiple triangles sharing the same TransformHandle.
                         // We must update the vertex cache for ALL of them, otherwise the CPU BVH and OptiX GAS
                         // will be inconsistent (only 1 triangle updated).
                         for (auto& obj : ctx.scene.world.objects) {
                             if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                                 // Check if they share the exact same transform handle instance
                                 if (tri->getTransformPtr() == transform_handle) {
                                     tri->updateTransformedVertices();
                                 }
                             }
                         }
                         
                         g_ProjectManager.markModified();
                         
                         // Trigger BVH rebuild (CPU)
                         ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                         ctx.renderer.resetCPUAccumulation();
                         
                         if (ctx.backend_ptr) {
                             // Rebuild Geometry (GAS) because vertices changed
                             ctx.renderer.rebuildBackendGeometry(ctx.scene);
                             ctx.backend_ptr->resetAccumulation();
                         }
                         addViewportMessage("Mesh fitted to VDB bounds", 3.0f, ImVec4(0.5f, 1.0f, 0.5f, 1.0f));
                     } else {
                         addViewportMessage("Selection does not support transforms", 3.0f, ImVec4(1.0f, 0.5f, 0.5f, 1.0f));
                     }
                     }
                     ImGui::PopStyleColor();

                 } else {
                     ImGui::TextColored(ImVec4(1,0,0,1), "Invalid VDB ID");
                     if (ImGui::Button("Reset")) {
                         vol_mat->setVDBVolumeID(-1);
                         vol_mat->setDensitySource(0);
                         material_changed = true;
                     }
                 }
            } else {
                 if (ImGui::Button("Load VDB File")) {
                     std::string path = SceneUI::openFileDialogW(L"OpenVDB/NanoVDB Files\0*.vdb;*.nvdb\0", "", "");
                     if (!path.empty()) {
                          int id = VDBVolumeManager::getInstance().loadVDB(path);
                          if (id >= 0) {
                              VDBVolumeManager::getInstance().uploadToGPU(id);
                              vol_mat->setVDBVolumeID(id);
                              vol_mat->setDensitySource(1); // Enable VDB density
                              material_changed = true;
                          }
                     }
                 }
                 ImGui::SameLine();
                 ImGui::TextDisabled("(Supports .vdb)");
            }
            UIWidgets::EndSection();
        }

        bool changed = false;
        if (UIWidgets::BeginSection("Volumetric Properties", ImVec4(0.76f, 0.60f, 1.0f, 1.0f))) {

        // �� DISTINCT INPUT FIELD STYLING FOR VOLUMETRIC PANEL (For Colors only) ����������������
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.15f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.18f, 0.20f, 0.25f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.22f, 0.25f, 0.30f, 1.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 3));

        // --- Reuse Volumetric UI Logic ---
        Vec3 alb = vol_mat->getAlbedo();
        if (ImGui::ColorEdit3("Scattering Color", &alb.x)) { vol_mat->setAlbedo(alb); changed = true; }

        // Pop styling for ColorEdit
        ImGui::PopStyleVar(3);
        ImGui::PopStyleColor(3);

        ImGui::Spacing();

        // LCD Sliders for Volumetric Params
        float dens = (float)vol_mat->getDensity();
        if (SceneUI::DrawSmartFloat("dens", "Density", &dens, 0.0f, 10.0f, "%.2f", false, nullptr, 16)) { 
            vol_mat->setDensity(dens); changed = true; 
        }

        float scatt = (float)vol_mat->getScattering();
        if (SceneUI::DrawSmartFloat("scatt", "Scatter", &scatt, 0.0f, 5.0f, "%.2f", false, nullptr, 16)) {
            vol_mat->setScattering(scatt); changed = true; 
        }

        float abs = (float)vol_mat->getAbsorption();
        if (SceneUI::DrawSmartFloat("abs", "Absorp", &abs, 0.0f, 2.0f, "%.2f", false, nullptr, 16)) {
            vol_mat->setAbsorption(abs); changed = true; 
        }

        float g = (float)vol_mat->getG();
        if (SceneUI::DrawSmartFloat("aniso", "Aniso", &g, -0.99f, 0.99f, "%.2f", false, nullptr, 16)) {
            vol_mat->setG(g); changed = true; 
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Phase Function Anisotropy (G)\n>0 Forward, <0 Backward");

        // Emission Color with Strength Control                
        Vec3 emis = vol_mat->getEmissionColor();
        float max_e = emis.x;
        if (emis.y > max_e) max_e = emis.y;
        if (emis.z > max_e) max_e = emis.z;
        float strength = (max_e > 1.0f) ? max_e : 1.0f;
        if (max_e < 0.001f) strength = 1.0f; 

        Vec3 normalized_emis = (max_e > 0.001f) ? emis * (1.0f / strength) : emis;
        bool em_changed = false;

        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.15f, 1.0f));
        if (ImGui::ColorEdit3("Emission Color", &normalized_emis.x)) em_changed = true;
        ImGui::PopStyleColor();

        // Use LCD slider for emission strength
        if (SceneUI::DrawSmartFloat("emstr", "EmStr", &strength, 0.0f, 100.0f, "%.1f", false, nullptr, 16)) em_changed = true;
        
        if (em_changed) {
            vol_mat->setEmissionColor(normalized_emis * strength);
            changed = true;
        }

        float n_scale = vol_mat->getNoiseScale();
        if (SceneUI::DrawSmartFloat("nscale", "Scale", &n_scale, 0.01f, 100.0f, "%.2f", false, nullptr, 16)) {
            vol_mat->setNoiseScale(n_scale); 
            changed = true;
        }

        float void_t = vol_mat->getVoidThreshold();
        if (SceneUI::DrawSmartFloat("void", "Void", &void_t, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) {
            vol_mat->setVoidThreshold(void_t); 
            changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Controls empty space amount (Higher = More Voids)");

        // Ray Marching Quality Settings
        ImGui::Separator();
        ImGui::TextDisabled("Ray Marching Quality:");
        
        float step_size = vol_mat->getStepSize();
        if (SceneUI::DrawSmartFloat("step", "Step", &step_size, 0.001f, 1.0f, "%.4f", false, nullptr, 16)) {
            vol_mat->setStepSize(step_size); 
            changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Smaller values = Higher Quality (Slower)");
        
        float max_steps_f = (float)vol_mat->getMaxSteps(); // LCD expects float
        if (SceneUI::DrawSmartFloat("mstep", "MaxStp", &max_steps_f, 1.0f, 1000.0f, "%.0f", false, nullptr, 16)) {
             vol_mat->setMaxSteps((int)max_steps_f);
             changed = true;
        }

        // ===================================================================
        // MULTI-SCATTERING CONTROLS
        // ===================================================================
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Multi-Scattering");

        float multi_scatter = vol_mat->getMultiScatter();
        if (SceneUI::DrawSmartFloat("mscat", "MScat", &multi_scatter, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) {
            vol_mat->setMultiScatter(multi_scatter); changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Controls multi-scattering brightness (0=single scatter, 1=full multi-scatter)");

        float g_back = vol_mat->getGBack();
        if (SceneUI::DrawSmartFloat("gback", "GBack", &g_back, -0.99f, 0.0f, "%.2f", false, nullptr, 16)) {
            vol_mat->setGBack(g_back); changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Backward scattering anisotropy for silver lining effect");

        float lobe_mix = vol_mat->getLobeMix();
        if (SceneUI::DrawSmartFloat("lmix", "LobeMx", &lobe_mix, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) {
            vol_mat->setLobeMix(lobe_mix); changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Forward/Backward lobe blend (1.0=all forward, 0.0=all backward)");

        float light_steps_f = (float)vol_mat->getLightSteps();
        if (SceneUI::DrawSmartFloat("lstep", "ShdStp", &light_steps_f, 0.0f, 8.0f, "%.0f", false, nullptr, 8)) {
            vol_mat->setLightSteps((int)light_steps_f); changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Light march steps for self-shadowing (0=disabled, 4-8 recommended)");

        float shadow_str = vol_mat->getShadowStrength();
        if (SceneUI::DrawSmartFloat("shstr", "ShdStr", &shadow_str, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) {
            vol_mat->setShadowStrength(shadow_str); changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Self-shadow intensity (0=no shadows, 1=full shadows)");

        if (changed) {
            // OPTIMIZED: Material property change - No geometry rebuild needed!
            // Only reset accumulation and update GPU material buffers
            ctx.renderer.resetCPUAccumulation();
            if (ctx.backend_ptr) {
                ctx.renderer.updateBackendMaterial(ctx.scene, active_mat_id);
                ctx.backend_ptr->resetAccumulation();
            }
        }
        UIWidgets::EndSection();
        }

    }

    else if (pbsdf) {
        drawPrincipledBSDFEditor(pbsdf, active_mat_id, ctx);
    }
    else {
        ImGui::TextDisabled("Unknown material type.");
    }

    ImGui::PopStyleColor(9);
    ImGui::PopStyleVar(6);
    UIWidgets::PopControlSurfaceStyle();
}

// ===============================================================================
// REUSABLE MATERIAL EDITOR WIDGET
// ===============================================================================
void SceneUI::drawPrincipledBSDFEditor(PrincipledBSDF* pbsdf, uint16_t mat_id, UIContext& ctx) {
    if (!pbsdf) return;
    
    bool changed = false;
    bool texture_changed = false;
    std::string obj_n = ctx.selection.selected.name;

    // Helper functions re-implemented locally to Context
    auto SyncGpuMaterial = [&](PrincipledBSDF* mat) -> void {
        if (!mat->gpuMaterial) mat->gpuMaterial = std::make_shared<GpuMaterial>();
        const PBRMaterialSnapshot snapshot = capturePBRMaterialSnapshot(*mat);
        applyPBRMaterialSnapshotToGpuMaterial(snapshot, *mat->gpuMaterial);
    };

    auto UpdateTriangleTextureBundle = [&](std::shared_ptr<Triangle> target_tri, PrincipledBSDF* mat) {
        OptixGeometryData::TextureBundle bundle = {};

        auto SetupTex = [&](std::shared_ptr<Texture>& tex, cudaTextureObject_t& out_tex, int& out_has) {
            if (tex && tex->is_loaded()) {
                tex->upload_to_gpu();
                out_tex = tex->get_cuda_texture();
                out_has = 1;
            }
            else {
                out_tex = 0;
                out_has = 0;
            }
        };

        SetupTex(mat->albedoProperty.texture, bundle.albedo_tex, bundle.has_albedo_tex);
        SetupTex(mat->normalProperty.texture, bundle.normal_tex, bundle.has_normal_tex);
        SetupTex(mat->roughnessProperty.texture, bundle.roughness_tex, bundle.has_roughness_tex);
        SetupTex(mat->metallicProperty.texture, bundle.metallic_tex, bundle.has_metallic_tex);
        SetupTex(mat->emissionProperty.texture, bundle.emission_tex, bundle.has_emission_tex);
        SetupTex(mat->opacityProperty.texture, bundle.opacity_tex, bundle.has_opacity_tex);

        target_tri->textureBundle = bundle;
    };

    auto TriggerMaterialUpdate = [&](bool needs_texture_update) {
        SyncGpuMaterial(pbsdf);

        if (needs_texture_update) {
            // Update all triangles using this material ID
            for (auto& obj : ctx.scene.world.objects) {
                auto t = std::dynamic_pointer_cast<Triangle>(obj);
                if (t && t->getMaterialID() == mat_id) {
                    UpdateTriangleTextureBundle(t, pbsdf);
                }
            }
        }

        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            ctx.renderer.updateBackendMaterial(ctx.scene, mat_id);
            ctx.backend_ptr->resetAccumulation();
        }
    };

    auto rebuildUvWorkflowPreviewEntries = [&]() {
        uv_workflow_preview_entries.clear();
        if (uv_workflow_cached_triangles.empty()) {
            return;
        }

        const size_t max_preview_tris = 300;
        const size_t step = uv_workflow_cached_triangles.size() > max_preview_tris
            ? static_cast<size_t>(std::ceil(static_cast<double>(uv_workflow_cached_triangles.size()) / static_cast<double>(max_preview_tris)))
            : 1;

        uv_workflow_preview_entries.reserve(std::min(max_preview_tris, uv_workflow_cached_triangles.size()));
        for (size_t tri_index = 0; tri_index < uv_workflow_cached_triangles.size(); tri_index += step) {
            const auto& tri = uv_workflow_cached_triangles[tri_index];
            auto [uv0, uv1, uv2] = tri->getUVSetCoordinates(static_cast<size_t>(uv_workflow_cached_uv_set));
            uv_workflow_preview_entries.push_back({ { uv0, uv1, uv2 } });
        }
    };

    auto ensureUvWorkflowCache = [&]() {
        size_t object_triangle_count = 0;
        auto mesh_it = mesh_cache.find(obj_n);
        if (mesh_it != mesh_cache.end()) {
            object_triangle_count = mesh_it->second.size();
        }

        const bool needs_rebuild =
            uv_workflow_cache_dirty ||
            uv_workflow_cached_object_name != obj_n ||
            uv_workflow_cached_material_id != mat_id ||
            uv_workflow_cached_uv_set != std::max(0, pbsdf->selected_uv_set) ||
            uv_workflow_cached_object_triangle_count != object_triangle_count;

        if (!needs_rebuild) {
            return;
        }

        uv_workflow_cached_object_name = obj_n;
        uv_workflow_cached_material_id = mat_id;
        uv_workflow_cached_uv_set = std::max(0, pbsdf->selected_uv_set);
        uv_workflow_cached_object_triangle_count = object_triangle_count;
        uv_workflow_cached_max_uv_sets = 1;
        uv_workflow_cached_triangles.clear();
        uv_workflow_preview_entries.clear();

        if (mesh_it == mesh_cache.end()) {
            uv_workflow_cache_dirty = false;
            return;
        }

        uv_workflow_cached_triangles.reserve(mesh_it->second.size());
        for (const auto& entry : mesh_it->second) {
            const auto& tri = entry.second;
            if (!tri || tri->getMaterialID() != mat_id) {
                continue;
            }
            uv_workflow_cached_max_uv_sets = std::max(uv_workflow_cached_max_uv_sets, static_cast<int>(tri->getUVSetCount()));
            uv_workflow_cached_triangles.push_back(tri);
        }

        rebuildUvWorkflowPreviewEntries();

        uv_workflow_cache_dirty = false;
    };

    auto refreshGeometryAfterUvChange = [&](bool invalidate_triangle_cache = false) {
        ctx.renderer.resetCPUAccumulation();
        if (ctx.backend_ptr) {
            ctx.renderer.rebuildBackendGeometry(ctx.scene);
            ctx.backend_ptr->resetAccumulation();
        }
        if (invalidate_triangle_cache) {
            uv_workflow_cache_dirty = true;
        } else {
            rebuildUvWorkflowPreviewEntries();
        }
        g_ProjectManager.markModified();
    };

    auto applyUvEditToSelectedSet = [&](const std::function<void(std::array<Vec2, 3>&)>& edit_fn) {
        const int selected_uv_set = std::max(0, pbsdf->selected_uv_set);
        ensureUvWorkflowCache();
        for (const auto& tri : uv_workflow_cached_triangles) {
            auto [uv0, uv1, uv2] = tri->getUVSetCoordinates(static_cast<size_t>(selected_uv_set));
            std::array<Vec2, 3> uv_data = { uv0, uv1, uv2 };
            edit_fn(uv_data);
            tri->setUVSetCoordinates(static_cast<size_t>(selected_uv_set), uv_data[0], uv_data[1], uv_data[2]);
            tri->applyUVSet(static_cast<size_t>(selected_uv_set));
        }
        refreshGeometryAfterUvChange();
    };

    auto LoadTextureFromDialog = [&](TextureType type, const std::string& initialDir = "", const std::string& defaultFile = "") -> std::shared_ptr<Texture> {
        std::string path = SceneUI::openFileDialogW(L"Image Files\0*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.exr;*.hdr\0", initialDir, defaultFile);
        if (path.empty()) return nullptr;

        auto tex = std::make_shared<Texture>(path, type);
        if (tex && tex->is_loaded()) {
            tex->upload_to_gpu();
            return tex;
        }
        SCENE_LOG_WARN("Failed to load texture: " + path);
        // Assuming addViewportMessage is available (member function) or global? It is member.
        // We are in SceneUI member function, so we can call addViewportMessage directly? YES.
        return nullptr; 
    };
    
    // Keyframe helpers from drawMaterialPanel (simplified or copied)
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

        ImVec2 p[4] = { ImVec2(cx, cy - r), ImVec2(cx + r, cy), ImVec2(cx, cy + r), ImVec2(cx - r, cy) };
        dl->AddQuadFilled(p[0], p[1], p[2], p[3], bg);
        dl->AddQuad(p[0], p[1], p[2], p[3], border, 1.0f);

        ImGui::PopID();
        return clicked;
    };

    // Need to copy/reimpl isMatKeyed and insertMatPropKey since they were lambdas in drawMaterialPanel
    // Or we can just adapt the code to run directly.
    // For brevity and reuse, I'll adapt the logic inline or re-define lambdas.
    
    auto isMatKeyed = [&](uint16_t mid, bool check_alb, bool check_opac, bool check_rgh, bool check_met, bool check_ems, bool check_trn, bool check_ior, bool check_spec) {
       if (obj_n.empty()) return false;
       auto it = ctx.scene.timeline.tracks.find(obj_n);
       if (it == ctx.scene.timeline.tracks.end()) return false;

       int cf = ctx.render_settings.animation_current_frame;
       for (auto& kf : it->second.keyframes) {
           if (kf.frame == cf && kf.has_material && kf.material.material_id == mid) {
               if (check_alb && kf.material.has_albedo) return true;
               if (check_opac && kf.material.has_opacity) return true; 
               if (check_rgh && kf.material.has_roughness) return true;
               if (check_met && kf.material.has_metallic) return true;
               if (check_ems && kf.material.has_emission) return true;
               if (check_trn && kf.material.has_transmission) return true;
               if (check_ior && kf.material.has_ior) return true;
               if (check_spec && kf.material.has_specular) return true;
           }
       }
       return false;
    };

    auto insertMatPropKey = [&](PrincipledBSDF* mat, uint16_t mid, bool k_alb, bool k_opac, bool k_rgh, bool k_met, bool k_ems, bool k_trn, bool k_ior, bool k_spec) {
        if (obj_n.empty()) return;
        int current_frame = ctx.render_settings.animation_current_frame;
        auto& track = ctx.scene.timeline.tracks[obj_n];

        // TOGGLE BEHAVIOR
        for (auto it = track.keyframes.begin(); it != track.keyframes.end(); ++it) {
            if (it->frame == current_frame && it->has_material && it->material.material_id == mid) {
                bool removed = false;
                if (k_alb && it->material.has_albedo) { it->material.has_albedo = false; removed = true; }
                if (k_opac && it->material.has_opacity) { it->material.has_opacity = false; removed = true; }
                if (k_rgh && it->material.has_roughness) { it->material.has_roughness = false; removed = true; }
                if (k_met && it->material.has_metallic) { it->material.has_metallic = false; removed = true; }
                if (k_ems && it->material.has_emission) { it->material.has_emission = false; removed = true; }
                if (k_trn && it->material.has_transmission) { it->material.has_transmission = false; removed = true; }
                if (k_ior && it->material.has_ior) { it->material.has_ior = false; removed = true; }
                if (k_spec && it->material.has_specular) { it->material.has_specular = false; removed = true; }

                if (removed) {
                    bool hasAny = it->material.has_albedo || it->material.has_opacity || it->material.has_roughness ||
                        it->material.has_metallic || it->material.has_emission || it->material.has_transmission ||
                        it->material.has_ior || it->material.has_specular;
                    if (!hasAny) {
                        it->has_material = false;
                        if (!it->has_transform && !it->has_camera && !it->has_light && !it->has_world) {
                            track.keyframes.erase(it);
                        }
                    }
                    SCENE_LOG_INFO("Removed material property keyframe.");
                    return;
                }
            }
        }

        if (mat && mat->gpuMaterial) {
            Keyframe kf(current_frame);
            kf.has_material = true;
            kf.material = MaterialKeyframe(*mat->gpuMaterial);
            kf.material.material_id = mid;
            kf.material.has_albedo = k_alb;
            kf.material.has_opacity = k_opac;
            kf.material.has_roughness = k_rgh;
            kf.material.has_metallic = k_met;
            kf.material.has_emission = k_ems;
            kf.material.has_transmission = k_trn;
            kf.material.has_ior = k_ior;
            kf.material.has_specular = k_spec;

            bool found = false;
            for (auto& existing : track.keyframes) {
                if (existing.frame == current_frame) {
                    existing.has_material = true;
                    if (existing.material.material_id == mid) {
                        if (k_alb) { existing.material.has_albedo = true; existing.material.albedo = kf.material.albedo; }
                        if (k_opac) { existing.material.has_opacity = true; existing.material.opacity = kf.material.opacity; }
                        if (k_rgh) { existing.material.has_roughness = true; existing.material.roughness = kf.material.roughness; }
                        if (k_met) { existing.material.has_metallic = true; existing.material.metallic = kf.material.metallic; }
                        if (k_ems) { existing.material.has_emission = true; existing.material.emission = kf.material.emission; existing.material.emission_strength = kf.material.emission_strength; }
                        if (k_trn) { existing.material.has_transmission = true; existing.material.transmission = kf.material.transmission; }
                        if (k_ior) { existing.material.has_ior = true; existing.material.ior = kf.material.ior; }
                        if (k_spec) { existing.material.has_specular = true; existing.material.specular = kf.material.specular; }
                    } else {
                        existing.material = kf.material;
                    }
                    found = true;
                    break;
                }
            }
            if (!found) track.addKeyframe(kf);
            SCENE_LOG_INFO("Added material property keyframe.");
        }
    };

    auto getPaintOverrideInfo = [&](const std::shared_ptr<Texture>& tex_ref, Paint::PaintChannel channel) -> std::pair<bool, std::string> {
        if (!tex_ref) {
            return { false, "" };
        }

        for (const auto& entry : ctx.scene.mesh_paint_texture_sets) {
            const Paint::PaintTextureSet& set = entry.second;
            if (set.material_id != mat_id) {
                continue;
            }

            std::shared_ptr<Texture> override_tex = set.getTexture(channel);
            if (override_tex && override_tex == tex_ref) {
                if (set.wasSeededFromExisting(channel) && !set.getSourceTextureName(channel).empty()) {
                    return { true, set.getSourceTextureName(channel) };
                }
                return { true, "" };
            }
        }

        return { false, "" };
    };

    // Style
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.12f, 0.12f, 0.15f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.18f, 0.20f, 0.25f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0.22f, 0.25f, 0.30f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImVec4(0.35f, 0.65f, 0.45f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImVec4(0.45f, 0.75f, 0.55f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.35f, 0.40f, 0.38f, 0.8f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(6, 3));

    if (UIWidgets::BeginSection("UV Workflow", ImVec4(0.54f, 0.86f, 1.0f, 1.0f))) {
        ImGui::TextWrapped("Choose the UV set here and run repair/projection tools before painting.");

        ImGui::BeginDisabled(obj_n.empty());
        if (UIWidgets::SecondaryButton("Project UVs from View", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
            ImGui::OpenPopup("MaterialUVProjectConfirm");
        }
        if (history.getUndoDescription().find("Project UVs") != std::string::npos) {
            ImVec2 btn_size = ImVec2(UIWidgets::GetInspectorActionWidth(), 0);
            if (UIWidgets::SecondaryButton("Reset / Undo Projection", btn_size)) {
                if (history.undo(ctx)) {
                    rebuildMeshCache(ctx.scene.world.objects);
                    ctx.selection.updatePositionFromSelection();
                    ctx.selection.selected.has_cached_aabb = false;
                    uv_workflow_cache_dirty = true;
                }
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Revert UV coordinates to previous state.");
        }
        if (ImGui::BeginPopupModal("MaterialUVProjectConfirm", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Warning: This will overwrite the active UV set for this object.\nExisting tiling textures might change.\nAre you sure?");
            ImGui::Separator();
            if (ImGui::Button("Yes, Project UVs", ImVec2(120, 0))) {
                projectObjectUVsFromView(ctx, *this, obj_n);
                uv_workflow_cache_dirty = true;
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel", ImVec2(120, 0))) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        ImGui::EndDisabled();

        ensureUvWorkflowCache();
        int max_uv_sets = uv_workflow_cached_max_uv_sets;
        if (max_uv_sets > 1) {
            std::vector<std::string> uv_labels;
            std::vector<const char*> uv_label_ptrs;
            uv_labels.reserve(max_uv_sets);
            uv_label_ptrs.reserve(max_uv_sets);
            for (int uv_index = 0; uv_index < max_uv_sets; ++uv_index) {
                uv_labels.push_back("UV Set " + std::to_string(uv_index));
                uv_label_ptrs.push_back(uv_labels.back().c_str());
            }

            int selected_uv_set = std::clamp(pbsdf->selected_uv_set, 0, max_uv_sets - 1);
            if (ImGui::Combo("UV Set", &selected_uv_set, uv_label_ptrs.data(), max_uv_sets)) {
                pbsdf->selected_uv_set = selected_uv_set;
                uv_workflow_cached_uv_set = selected_uv_set;
                for (const auto& tri : uv_workflow_cached_triangles) {
                    tri->applyUVSet(static_cast<size_t>(selected_uv_set));
                }
                refreshGeometryAfterUvChange();
            }
            ImGui::TextDisabled("Textures and mesh paint use the selected UV set.");
        } else {
            pbsdf->selected_uv_set = 0;
            ImGui::TextDisabled("UV Set: UV 0");
        }

        ImGui::Spacing();
        ImGui::SeparatorText("Material UV Transform");
        {
            auto& uvTransform = pbsdf->textureTransform;
            static bool lockUvScaleAxes = true;
            static bool lockUvOffsetAxes = false;
            float uvScale[2] = {
                static_cast<float>(uvTransform.scale.u),
                static_cast<float>(uvTransform.scale.v)
            };
            float uvOffset[2] = {
                static_cast<float>(uvTransform.translation.u),
                static_cast<float>(uvTransform.translation.v)
            };
            float uvRotation = uvTransform.rotation_degrees;
            int wrapMode = static_cast<int>(uvTransform.wrapMode);
            const char* wrapLabels[] = { "Repeat", "Mirror", "Clamp", "Planar", "Cubic" };

            ImGui::Checkbox("Lock Scale XY", &lockUvScaleAxes);
            if (ImGui::DragFloat2("Scale XY", uvScale, 0.01f, 0.001f, 64.0f, "%.3f")) {
                if (lockUvScaleAxes) {
                    const float lockedScale = std::max(0.001f, uvScale[0]);
                    uvScale[0] = lockedScale;
                    uvScale[1] = lockedScale;
                }
                uvTransform.scale = Vec2(
                    std::max(0.001f, uvScale[0]),
                    std::max(0.001f, uvScale[1]));
                changed = true;
            }
            ImGui::Checkbox("Lock Offset XY", &lockUvOffsetAxes);
            if (ImGui::DragFloat2("Offset XY", uvOffset, 0.01f, -100.0f, 100.0f, "%.3f")) {
                if (lockUvOffsetAxes) {
                    uvOffset[1] = uvOffset[0];
                }
                uvTransform.translation = Vec2(uvOffset[0], uvOffset[1]);
                changed = true;
            }
            if (ImGui::DragFloat("Rotation", &uvRotation, 0.25f, -360.0f, 360.0f, "%.1f deg")) {
                uvTransform.rotation_degrees = uvRotation;
                changed = true;
            }
            if (ImGui::Combo("Wrap", &wrapMode, wrapLabels, IM_ARRAYSIZE(wrapLabels))) {
                uvTransform.wrapMode = static_cast<WrapMode>(wrapMode);
                changed = true;
            }

            if (UIWidgets::SecondaryButton("Reset UV Transform", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
                uvTransform.scale = Vec2(1.0, 1.0);
                uvTransform.translation = Vec2(0.0, 0.0);
                uvTransform.rotation_degrees = 0.0f;
                uvTransform.wrapMode = WrapMode::Repeat;
                changed = true;
            }
            ImGui::TextDisabled("Scale/Offset/Rotation affect texture sampling, not stored mesh UVs.");
        }

        const float uv_tool_w = (UIWidgets::GetInspectorActionWidth() - ImGui::GetStyle().ItemSpacing.x) * 0.5f;
        if (UIWidgets::SecondaryButton("Normalize UVs", ImVec2(uv_tool_w, 0))) {
            const int selected_uv_set = std::max(0, pbsdf->selected_uv_set);
            double min_u = std::numeric_limits<double>::max();
            double min_v = std::numeric_limits<double>::max();
            double max_u = std::numeric_limits<double>::lowest();
            double max_v = std::numeric_limits<double>::lowest();
            for (const auto& tri : uv_workflow_cached_triangles) {
                auto [uv0, uv1, uv2] = tri->getUVSetCoordinates(static_cast<size_t>(selected_uv_set));
                const Vec2 uvs[3] = { uv0, uv1, uv2 };
                for (const Vec2& uv : uvs) {
                    min_u = std::min(min_u, static_cast<double>(uv.u));
                    min_v = std::min(min_v, static_cast<double>(uv.v));
                    max_u = std::max(max_u, static_cast<double>(uv.u));
                    max_v = std::max(max_v, static_cast<double>(uv.v));
                }
            }
            const double span_u = std::max(1e-8, max_u - min_u);
            const double span_v = std::max(1e-8, max_v - min_v);
            applyUvEditToSelectedSet([&](std::array<Vec2, 3>& uv_data) {
                for (Vec2& uv : uv_data) {
                    uv.u = (uv.u - min_u) / span_u;
                    uv.v = (uv.v - min_v) / span_v;
                }
            });
        }
        ImGui::SameLine();
        if (UIWidgets::SecondaryButton("Flip U", ImVec2(uv_tool_w, 0))) {
            applyUvEditToSelectedSet([&](std::array<Vec2, 3>& uv_data) {
                for (Vec2& uv : uv_data) uv.u = 1.0 - uv.u;
            });
        }

        if (UIWidgets::SecondaryButton("Flip V", ImVec2(uv_tool_w, 0))) {
            applyUvEditToSelectedSet([&](std::array<Vec2, 3>& uv_data) {
                for (Vec2& uv : uv_data) uv.v = 1.0 - uv.v;
            });
        }
        ImGui::SameLine();
        if (UIWidgets::SecondaryButton("Swap U/V", ImVec2(uv_tool_w, 0))) {
            applyUvEditToSelectedSet([&](std::array<Vec2, 3>& uv_data) {
                for (Vec2& uv : uv_data) std::swap(uv.u, uv.v);
            });
        }

        ImGui::BeginChild("##uv_workflow_panel", ImVec2(0, 250.0f), true, ImGuiWindowFlags_NoScrollbar);
        {
            ImGui::Spacing();
            const float preview_size = std::min(220.0f, ImGui::GetContentRegionAvail().x);
            const ImVec2 preview_min = ImGui::GetCursorScreenPos();
            const ImVec2 preview_max(preview_min.x + preview_size, preview_min.y + preview_size);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(preview_min, preview_max, IM_COL32(18, 22, 28, 255), 4.0f);
            dl->AddRect(preview_min, preview_max, IM_COL32(90, 110, 130, 220), 4.0f);

            for (int grid = 1; grid < 4; ++grid) {
                const float t = static_cast<float>(grid) / 4.0f;
                const float x = preview_min.x + preview_size * t;
                const float y = preview_min.y + preview_size * t;
                dl->AddLine(ImVec2(x, preview_min.y), ImVec2(x, preview_max.y), IM_COL32(55, 65, 78, 160), 1.0f);
                dl->AddLine(ImVec2(preview_min.x, y), ImVec2(preview_max.x, y), IM_COL32(55, 65, 78, 160), 1.0f);
            }

            const int selected_uv_set = std::max(0, pbsdf->selected_uv_set);

            auto toPreviewPoint = [&](const Vec2& uv) {
                return ImVec2(
                    preview_min.x + static_cast<float>(uv.u) * preview_size,
                    preview_max.y - static_cast<float>(uv.v) * preview_size);
            };

            for (const auto& entry : uv_workflow_preview_entries) {
                const ImVec2 p0 = toPreviewPoint(entry.uvs[0]);
                const ImVec2 p1 = toPreviewPoint(entry.uvs[1]);
                const ImVec2 p2 = toPreviewPoint(entry.uvs[2]);
                dl->AddTriangle(p0, p1, p2, IM_COL32(110, 200, 255, 180), 1.0f);
            }

            ImGui::InvisibleButton("##uv_preview", ImVec2(preview_size, preview_size));
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("UV Preview\nActive Set: %d\nTriangles: %d", selected_uv_set, static_cast<int>(uv_workflow_cached_triangles.size()));
            }
        }
        ImGui::EndChild();
        UIWidgets::EndSection();
    }

    if (UIWidgets::BeginSection("Surface Core", ImVec4(1.0f, 0.72f, 0.42f, 1.0f))) {
        Vec3 albedo = pbsdf->albedoProperty.color;
        float albedo_arr[3] = { (float)albedo.x, (float)albedo.y, (float)albedo.z };
        bool albKeyed = isMatKeyed(mat_id, true, false, false, false, false, false, false, false);
        if (KeyframeButton("##MAlb", albKeyed)) { insertMatPropKey(pbsdf, mat_id, true, false, false, false, false, false, false, false); }
        ImGui::SameLine();
        if (ImGui::ColorEdit3("Base Color", albedo_arr)) {
            pbsdf->albedoProperty.color = Vec3(albedo_arr[0], albedo_arr[1], albedo_arr[2]);
            changed = true;
        }

        float metallic = pbsdf->metallicProperty.intensity;
        bool metKeyed = isMatKeyed(mat_id, false, false, false, true, false, false, false, false);
        if (SceneUI::DrawSmartFloat("met", "Metal", &metallic, 0.0f, 1.0f, "%.3f", metKeyed,
            [&](){ insertMatPropKey(pbsdf, mat_id, false, false, false, true, false, false, false, false); }, 16)) {
            pbsdf->metallicProperty.intensity = metallic;
            changed = true;
        }

        float roughness = (float)pbsdf->roughnessProperty.color.x;
        bool rghKeyed = isMatKeyed(mat_id, false, false, true, false, false, false, false, false);
        if (SceneUI::DrawSmartFloat("rgh", "Rough", &roughness, 0.0f, 1.0f, "%.3f", rghKeyed,
            [&](){ insertMatPropKey(pbsdf, mat_id, false, false, true, false, false, false, false, false); }, 16)) {
            pbsdf->roughnessProperty.color = Vec3(roughness);
            changed = true;
        }

        float ior = pbsdf->ior;
        bool iorKeyed = isMatKeyed(mat_id, false, false, false, false, false, false, true, false);
        if (SceneUI::DrawSmartFloat("ior", "IOR", &ior, 1.0f, 3.0f, "%.3f", iorKeyed,
            [&](){ insertMatPropKey(pbsdf, mat_id, false, false, false, false, false, false, true, false); }, 16)) {
            pbsdf->ior = ior;
            changed = true;
        }

        float transmission = pbsdf->transmission;
        bool trnKeyed = isMatKeyed(mat_id, false, false, false, false, false, true, false, false);
        if (SceneUI::DrawSmartFloat("trans", "Trans", &transmission, 0.0f, 1.0f, "%.3f", trnKeyed,
            [&](){ insertMatPropKey(pbsdf, mat_id, false, false, false, false, false, true, false, false); }, 16)) {
            pbsdf->setTransmission(transmission, pbsdf->ior);
            changed = true;
        }

        float specular = pbsdf->specularProperty.intensity;
        bool specKeyed = isMatKeyed(mat_id, false, false, false, false, false, false, false, true);
        if (SceneUI::DrawSmartFloat("spec", "Spec", &specular, 0.0f, 1.0f, "%.3f", specKeyed,
            [&](){ insertMatPropKey(pbsdf, mat_id, false, false, false, false, false, false, false, true); }, 16)) {
            pbsdf->specularProperty.intensity = specular;
            changed = true;
        }

        float opacity = pbsdf->opacityProperty.alpha;
        bool opKeyed = isMatKeyed(mat_id, false, true, false, false, false, false, false, false);
        if (SceneUI::DrawSmartFloat("opac", "Alpha", &opacity, 0.0f, 1.0f, "%.3f", opKeyed,
            [&](){ insertMatPropKey(pbsdf, mat_id, false, true, false, false, false, false, false, false); }, 16)) {
            pbsdf->opacityProperty.alpha = opacity;
            changed = true;
        }

        Vec3 emission = pbsdf->emissionProperty.color;
        float emission_arr[3] = { (float)emission.x, (float)emission.y, (float)emission.z };
        bool emsKeyed = isMatKeyed(mat_id, false, false, false, false, true, false, false, false);
        if (KeyframeButton("##MEms", emsKeyed)) { insertMatPropKey(pbsdf, mat_id, false, false, false, false, true, false, false, false); }
        ImGui::SameLine();
        if (ImGui::ColorEdit3("Emission", emission_arr)) {
            pbsdf->emissionProperty.color = Vec3(emission_arr[0], emission_arr[1], emission_arr[2]);
            changed = true;
        }

        float emissionStr = pbsdf->emissionProperty.intensity;
        if (SceneUI::DrawSmartFloat("mems", "EmStr", &emissionStr, 0.0f, 1000.0f, "%.1f", false, nullptr, 12)) {
            pbsdf->emissionProperty.intensity = emissionStr;
            changed = true;
        }
        UIWidgets::EndSection();
    }
    if (UIWidgets::BeginSection("Subsurface Scattering", ImVec4(1.0f, 0.7f, 0.4f, 1.0f))) {
        float sss_amount = pbsdf->subsurface;
        if (SceneUI::DrawSmartFloat("sss", "Subsurf", &sss_amount, 0.0f, 1.0f, "%.3f", false, nullptr, 12)) {
            pbsdf->subsurface = sss_amount;
            changed = true;
        }
        UIWidgets::HelpMarker("Amount of subsurface scattering (0=off, 1=full)");
        
        Vec3 sss_color = pbsdf->subsurfaceColor;
        float sss_color_arr[3] = {(float)sss_color.x, (float)sss_color.y, (float)sss_color.z};
        if (ImGui::ColorEdit3("SSS Color", sss_color_arr)) {
            pbsdf->subsurfaceColor = Vec3(sss_color_arr[0], sss_color_arr[1], sss_color_arr[2]);
            changed = true;
        }
        
        Vec3 sss_radius = pbsdf->subsurfaceRadius;
        float sss_radius_arr[3] = {(float)sss_radius.x, (float)sss_radius.y, (float)sss_radius.z};
        if (ImGui::DragFloat3("Radius (RGB)", sss_radius_arr, 0.01f, 0.001f, 10.0f, "%.3f")) {
            pbsdf->subsurfaceRadius = Vec3(sss_radius_arr[0], sss_radius_arr[1], sss_radius_arr[2]);
            changed = true;
        }
        
        float sss_scale = pbsdf->subsurfaceScale;
        if (SceneUI::DrawSmartFloat("sscl", "Scale", &sss_scale, 0.001f, 2.0f, "%.4f", false, nullptr, 12)) {
            pbsdf->subsurfaceScale = sss_scale;
            changed = true;
        }
        
        float sss_aniso = pbsdf->subsurfaceAnisotropy;
        if (SceneUI::DrawSmartFloat("ssani", "Aniso", &sss_aniso, -0.9f, 0.9f, "%.2f", false, nullptr, 12)) {
            pbsdf->subsurfaceAnisotropy = sss_aniso;
            changed = true;
        }
        
        // SSS Index of Refraction (controls internal refraction/exit behavior)
        float sss_ior = pbsdf->subsurfaceIOR;
        if (SceneUI::DrawSmartFloat("ssior", "SSS IOR", &sss_ior, 1.0f, 3.0f, "%.3f", false, nullptr, 12)) {
            pbsdf->subsurfaceIOR = sss_ior;
            changed = true;
        }

        // Random-walk controls: enable and max steps
        bool sss_rw = pbsdf->useRandomWalkSSS;
        if (ImGui::Checkbox("Enable Random-Walk", &sss_rw)) {
            pbsdf->useRandomWalkSSS = sss_rw;
            changed = true;
        }
        int sss_steps = pbsdf->sssMaxSteps;
        if (ImGui::SliderInt("Max Steps", &sss_steps, 1, 32)) {
            pbsdf->sssMaxSteps = sss_steps;
            changed = true;
        }
        
        ImGui::Separator();
        ImGui::Text("Presets:");
        if (ImGui::SmallButton("Skin")) {
            pbsdf->subsurface = 0.3f;
            pbsdf->subsurfaceColor = Vec3(1.0f, 0.8f, 0.6f);
            pbsdf->subsurfaceRadius = Vec3(1.0f, 0.2f, 0.1f);
            pbsdf->subsurfaceScale = 0.05f;
            pbsdf->subsurfaceAnisotropy = 0.0f;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Wax")) {
            pbsdf->subsurface = 0.5f;
            pbsdf->subsurfaceColor = Vec3(1.0f, 0.9f, 0.6f);
            pbsdf->subsurfaceRadius = Vec3(0.3f, 0.3f, 0.2f);
            pbsdf->subsurfaceScale = 0.1f;
            pbsdf->subsurfaceAnisotropy = 0.0f;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Milk")) {
            pbsdf->subsurface = 0.8f;
            pbsdf->subsurfaceColor = Vec3(1.0f, 1.0f, 1.0f);
            pbsdf->subsurfaceRadius = Vec3(0.5f, 0.5f, 0.5f);
            pbsdf->subsurfaceScale = 0.2f;
            pbsdf->subsurfaceAnisotropy = 0.8f;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Jade")) {
            pbsdf->subsurface = 0.4f;
            pbsdf->subsurfaceColor = Vec3(0.3f, 0.8f, 0.4f);
            pbsdf->subsurfaceRadius = Vec3(0.2f, 0.5f, 0.2f);
            pbsdf->subsurfaceScale = 0.05f;
            pbsdf->subsurfaceAnisotropy = 0.3f;
            changed = true;
        }
        
        UIWidgets::EndSection();
    }

    if (UIWidgets::BeginSection("Clear Coat", ImVec4(0.8f, 0.2f, 0.8f, 1.0f))) {
        float cc_amount = pbsdf->clearcoat;
        if (SceneUI::DrawSmartFloat("cc", "ClearCt", &cc_amount, 0.0f, 1.0f, "%.3f", false, nullptr, 12)) {
            pbsdf->clearcoat = cc_amount;
            changed = true;
        }
        
        float cc_roughness = pbsdf->clearcoatRoughness;
        if (SceneUI::DrawSmartFloat("ccr", "CCRough", &cc_roughness, 0.0f, 1.0f, "%.3f", false, nullptr, 12)) {
            pbsdf->clearcoatRoughness = cc_roughness;
            changed = true;
        }
        
        ImGui::Separator();
        if (ImGui::SmallButton("Car Paint")) {
            pbsdf->clearcoat = 1.0f;
            pbsdf->clearcoatRoughness = 0.03f;
            changed = true;
        }
        UIWidgets::EndSection();
    }

    if (UIWidgets::BeginSection("Translucent", ImVec4(0.5f, 0.9f, 0.9f, 1.0f))) {
        float trans = pbsdf->translucent;
        if (SceneUI::DrawSmartFloat("trns", "Transl", &trans, 0.0f, 1.0f, "%.3f", false, nullptr, 12)) {
            pbsdf->translucent = trans;
            changed = true;
        }
        UIWidgets::EndSection();
    }

    // Procedural Detail — world-space color/dirt/roughness + optional UV tile-break
    // Shared by all three backends (Vulkan/GLSL, OptiX/CUDA, CPU).
    if (UIWidgets::BeginSection("Procedural Detail", ImVec4(0.65f, 0.55f, 0.40f, 1.0f))) {
        float det_str = pbsdf->micro_detail_strength;
        if (SceneUI::DrawSmartFloat("pd_str", "Strength", &det_str, 0.0f, 1.0f, "%.3f", false, nullptr, 12)) {
            pbsdf->micro_detail_strength = det_str;
            changed = true;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("World-space color variation + dirt + roughness.\nDoes NOT warp UV — texture stays intact.\n0 = off, 0.2–0.5 = subtle, 1.0 = strong");

        float det_sc = pbsdf->micro_detail_scale;
        if (SceneUI::DrawSmartFloat("pd_sc", "Scale", &det_sc, 0.1f, 10.0f, "%.2f", false, nullptr, 12)) {
            pbsdf->micro_detail_scale = det_sc;
            changed = true;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("World-space frequency\n1–2 = large patches\n4–8 = fine grain");

        ImGui::Separator();

        float tb_str = pbsdf->tile_break_strength;
        if (SceneUI::DrawSmartFloat("pd_tb", "Tile Break", &tb_str, 0.0f, 0.35f, "%.3f", false, nullptr, 12)) {
            pbsdf->tile_break_strength = tb_str;
            changed = true;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("UV warp to break repeating tiling seams.\nLeave at 0 for unique/non-tiling albedo maps.\n0.05–0.15 = subtle, 0.25–0.35 = strong");

        ImGui::Separator();
        if (ImGui::SmallButton("Worn Stone")) {
            pbsdf->micro_detail_strength = 0.45f;
            pbsdf->micro_detail_scale    = 3.5f;
            pbsdf->tile_break_strength   = 0.0f;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Dusty")) {
            pbsdf->micro_detail_strength = 0.30f;
            pbsdf->micro_detail_scale    = 2.0f;
            pbsdf->tile_break_strength   = 0.0f;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Tiled Tex")) {
            pbsdf->micro_detail_strength = 0.20f;
            pbsdf->micro_detail_scale    = 2.5f;
            pbsdf->tile_break_strength   = 0.15f;
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Off")) {
            pbsdf->micro_detail_strength = 0.0f;
            pbsdf->tile_break_strength   = 0.0f;
            changed = true;
        }
        UIWidgets::EndSection();
    }

    if (UIWidgets::BeginSection("Texture Maps", ImVec4(0.4f, 1.0f, 0.6f, 1.0f))) {
        static std::shared_ptr<Texture> texture_clipboard = nullptr;

        auto DrawTextureSlot = [&](const char* label, std::shared_ptr<Texture>& tex_ref, TextureType type, Paint::PaintChannel paint_channel) {
            ImGui::PushID(label);
            
            // 1. Label
            ImGui::AlignTextToFramePadding();
            ImGui::Text("%s", label);
            
            // 2. Setup Right Alignment for Buttons
            float buttonsWidth = 85.0f; 
            float availWidth = ImGui::GetContentRegionAvail().x;
            float rightEdge = ImGui::GetWindowContentRegionMax().x;

            // 3. Texture Info / Short Name
            ImGui::SameLine();
            
            // Move cursor to allow space for label but don't overlap buttons
            // Calculate max width for name
            float cursorX = ImGui::GetCursorPosX();
            float nameMaxWidth = rightEdge - cursorX - buttonsWidth - 10.0f; 
            
            if (tex_ref && tex_ref->is_loaded()) {
                std::string name = tex_ref->name;
                std::string shortName = std::filesystem::path(name).filename().string();
                const auto [is_paint_override, source_name] = getPaintOverrideInfo(tex_ref, paint_channel);
                
                // Truncate if long 
                if (shortName.length() > 20) {
                     shortName = shortName.substr(0, 8) + "..." + shortName.substr(shortName.length()-5);
                }

                ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "%s", shortName.c_str());
                if (ImGui::IsItemHovered()) {
                    if (is_paint_override && !source_name.empty()) {
                        ImGui::SetTooltip("%s\n(%dx%d)\nPaint Override\nSeeded from: %s",
                            name.c_str(), tex_ref->width, tex_ref->height, source_name.c_str());
                    } else if (is_paint_override) {
                        ImGui::SetTooltip("%s\n(%dx%d)\nPaint Override",
                            name.c_str(), tex_ref->width, tex_ref->height);
                    } else {
                        ImGui::SetTooltip("%s\n(%dx%d)", name.c_str(), tex_ref->width, tex_ref->height);
                    }
                }

                if (is_paint_override) {
                    ImGui::SameLine(0, 6);
                    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.30f, 1.0f), "[Paint]");
                    if (ImGui::IsItemHovered()) {
                        if (!source_name.empty()) {
                            ImGui::SetTooltip("Paint override active\nSeeded from: %s", source_name.c_str());
                        } else {
                            ImGui::SetTooltip("Paint override active");
                        }
                    }
                }

                if (type != TextureType::Unknown && !tex_ref->name.empty()) {
                    const std::string cacheKey = tex_ref->name + "|" + std::to_string(static_cast<int>(type));
                    auto it = s_cacheTagMap.find(cacheKey);
                    if (it == s_cacheTagMap.end()) {
                        it = s_cacheTagMap.emplace(cacheKey, queryManagedCacheTag(*tex_ref, type)).first;
                    }
                    if (!it->second.empty()) {
                        ImGui::SameLine(0, 6);
                        ImGui::TextColored(ImVec4(0.3f, 0.7f, 1.0f, 1.0f), "[%s]", it->second.c_str());
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("GPU cache: %s (managed)\nReload project to activate.", it->second.c_str());
                    }
                }

                // 4. Buttons (Right Aligned)
                ImGui::SameLine();
                if (ImGui::GetCursorPosX() < (rightEdge - buttonsWidth)) {
                    ImGui::SetCursorPosX(rightEdge - buttonsWidth);
                }
                
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));
                
                // X Button
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.2f, 0.2f, 1.0f));
                if (ImGui::SmallButton("X")) {
                    if (tex_ref) texture_graveyard.push_back(tex_ref); // Safety
                    tex_ref = nullptr;
                    texture_changed = true;
                }
                ImGui::PopStyleColor();
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove");

                ImGui::SameLine(0, 4);
                // C Button
                if (ImGui::SmallButton("C")) texture_clipboard = tex_ref;
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Copy");
            
                ImGui::PopStyleVar();
            }
            else {
                ImGui::TextDisabled("-");
                
                // 4. Buttons (Right Aligned even if empty)
                ImGui::SameLine();
                if (ImGui::GetCursorPosX() < (rightEdge - buttonsWidth)) {
                    ImGui::SetCursorPosX(rightEdge - buttonsWidth);
                }

                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));
                
                // P Button
                if (texture_clipboard) {
                    if (ImGui::SmallButton("P")) {
                        tex_ref = texture_clipboard;
                        texture_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Paste");
                    ImGui::SameLine(0, 4);
                }
                
                ImGui::PopStyleVar();
            }

            // Load Button (Always at end)
            ImGui::SameLine(0, 4);
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(4, 2));
            if (ImGui::SmallButton("+")) {
                std::string initialDir = "";
                std::string defaultFile = "";

                if (tex_ref && !tex_ref->name.empty()) {
                    std::string fullPath = tex_ref->name;
                    size_t lastSlash = fullPath.find_last_of("/\\");
                    if (lastSlash != std::string::npos) {
                        initialDir = fullPath.substr(0, lastSlash);
                        defaultFile = fullPath.substr(lastSlash + 1);
                    }
                }
                else if (texture_clipboard && !texture_clipboard->name.empty()) {
                    std::string fullPath = texture_clipboard->name;
                    size_t lastSlash = fullPath.find_last_of("/\\");
                    if (lastSlash != std::string::npos) initialDir = fullPath.substr(0, lastSlash);
                }

                auto new_tex = LoadTextureFromDialog(type, initialDir, defaultFile);
                if (new_tex) {
                    if (tex_ref) texture_graveyard.push_back(tex_ref); // Safety
                    tex_ref = new_tex;
                    texture_changed = true;
                }
            }
            ImGui::PopStyleVar();
            
            ImGui::PopID();
        };

        DrawTextureSlot("Albedo", pbsdf->albedoProperty.texture, TextureType::Albedo, Paint::PaintChannel::BaseColor);
        DrawTextureSlot("Normal", pbsdf->normalProperty.texture, TextureType::Normal, Paint::PaintChannel::Normal);
        float normal_strength = pbsdf->get_normal_strength();
        if (SceneUI::DrawSmartFloat("nrmstr", "Normal Strength", &normal_strength, 0.0f, 8.0f, "%.2f", false, nullptr, 12)) {
            pbsdf->set_normal_strength(normal_strength);
            pbsdf->normalProperty.intensity = normal_strength;
            changed = true;
        }
        DrawTextureSlot("Roughness", pbsdf->roughnessProperty.texture, TextureType::Roughness, Paint::PaintChannel::Roughness);
        DrawTextureSlot("Metallic", pbsdf->metallicProperty.texture, TextureType::Metallic, Paint::PaintChannel::Metallic);
        DrawTextureSlot("Emission", pbsdf->emissionProperty.texture, TextureType::Emission, Paint::PaintChannel::Emission);
        DrawTextureSlot("Transmission", pbsdf->transmissionProperty.texture, TextureType::Transmission, Paint::PaintChannel::Transmission);
        DrawTextureSlot("Height", pbsdf->heightProperty.texture, TextureType::Unknown, Paint::PaintChannel::Mask);
        DrawTextureSlot("Opacity", pbsdf->opacityProperty.texture, TextureType::Opacity, Paint::PaintChannel::Opacity);

        UIWidgets::EndSection();
    }

    if (changed || texture_changed) {
        TriggerMaterialUpdate(texture_changed);
        g_ProjectManager.markModified();
    }

    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(6);
}
