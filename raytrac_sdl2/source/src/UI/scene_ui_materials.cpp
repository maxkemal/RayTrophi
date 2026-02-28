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
#include "PrincipledBSDF.h"
#include "Volumetric.h"
#include "scene_data.h"
#include "VDBVolumeManager.h"
#include "HittableInstance.h"
#include "Texture.h" 
#include "Triangle.h"
#include "TerrainManager.h"

// Static editor instance
static bool showMatNodeEditor = false;

void SceneUI::resetMaterialUI() {
    showMatNodeEditor = false;
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

    // Only show for selected objects
    if (sel.selected.type != SelectableType::Object || !sel.selected.object) {
        ImGui::TextDisabled("Select an object to edit materials");
        return;
    }

    std::string obj_name = sel.selected.name;
    if (obj_name.empty()) {
        ImGui::TextDisabled("Unnamed Object");
        return;
    }

    // Ensure mesh cache is valid to find all triangles of this object
    if (!mesh_cache_valid) {
        rebuildMeshCache(ctx.scene.world.objects);
    }

    auto cache_it = mesh_cache.find(obj_name);
    if (cache_it == mesh_cache.end()) {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Object mesh data not found in cache.");
        return;
    }

    // ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
    // 1. GET MATERIAL SLOTS FROM CACHE (O(1) lookup, not O(N) triangle scan!)
    // ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
    // Use cached material slots instead of scanning all triangles every frame
    auto slots_it = material_slots_cache.find(obj_name);
    if (slots_it == material_slots_cache.end()) {
        ImGui::TextDisabled("Material data not cached. Cache may need rebuild.");
        return;
    }
    
    const std::vector<uint16_t>& used_material_ids = slots_it->second;

    if (used_material_ids.empty()) {
        ImGui::TextDisabled("No geometry/materials found.");
        return;
    }

    // ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
    // 2. SLOT SELECTION UI
    // ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
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

    ImGui::TextColored(ImVec4(0.9f, 0.7f, 0.3f, 1.0f), "Material Slots");

    // Draw the Slot List
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

            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndListBox();

    }
    // Current pointer to the active material
    uint16_t active_mat_id = (active_slot_index < (int)used_material_ids.size()) ? used_material_ids[active_slot_index] : MaterialManager::INVALID_MATERIAL_ID;
    Material* active_mat_ptr = MaterialManager::getInstance().getMaterial(active_mat_id);
    std::string current_mat_name = active_mat_ptr ? active_mat_ptr->materialName : "None";

    // ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
    // 3. ASSIGN MATERIAL TO ACTIVE SLOT
    // ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
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

    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.5f, 0.2f, 1.0f));
    if (ImGui::SmallButton("Key Mat")) {
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
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Insert Material Keyframe at current frame (keys all material properties)");
    }
    ImGui::SameLine();
    ImGui::Text("Active Slot Assignment:");

    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - 110); // Space for +New buttons (adjusted for +S and +V)
    if (ImGui::BeginCombo("##SlotAssignment", current_mat_name.c_str())) {
        auto& mgr = MaterialManager::getInstance();
        const auto& all_materials = mgr.getAllMaterials();

        for (size_t i = 0; i < all_materials.size(); i++) {
            if (!all_materials[i]) continue;

            bool is_selected = ((uint16_t)i == active_mat_id);
            std::string label = all_materials[i]->materialName;
            if (label.empty()) label = "Mat #" + std::to_string(i);

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
                    ctx.renderer.updateMeshMaterialBinding(obj_name, active_mat_id, (uint16_t)i);

                    if (ctx.backend_ptr) {
                        ctx.renderer.updateBackendMaterials(ctx.scene);
                        ctx.backend_ptr->resetAccumulation();
                    }
                    g_ProjectManager.markModified();
                }
            }
            if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    // ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
    // SHORTCUTS (New Surface / Volume) - affecting Active Slot
    // ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
    ImGui::SameLine();
    if (ImGui::Button("+S", ImVec2(40, 0))) {
        ImGui::OpenPopup("NewSurfPopup");
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Create New Surface Material");

    ImGui::SameLine();
    if (ImGui::Button("+V", ImVec2(40, 0))) {
        ImGui::OpenPopup("NewVolPopup");
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Create New Volumetric Material");

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
            ctx.renderer.updateMeshMaterialBinding(obj_name, active_mat_id, new_id);

            if (ctx.backend_ptr) {
                ctx.renderer.updateBackendMaterials(ctx.scene);
                ctx.backend_ptr->resetAccumulation();
            }
            g_ProjectManager.markModified();
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
            ctx.renderer.updateMeshMaterialBinding(obj_name, active_mat_id, new_id);

            if (ctx.backend_ptr) {
                ctx.renderer.updateBackendMaterials(ctx.scene);
                ctx.backend_ptr->resetAccumulation();
            }
            g_ProjectManager.markModified();
        }
        ImGui::EndPopup();
    }

    // ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
    // MATERIAL EDITOR (Context-Aware for Active Slot)
    // ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
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
    if (!active_mat_ptr) return;

    // Check material type
    Volumetric* vol_mat = dynamic_cast<Volumetric*>(active_mat_ptr);
    PrincipledBSDF* pbsdf = dynamic_cast<PrincipledBSDF*>(active_mat_ptr);

    ImGui::Separator();
    bool material_changed = false;
    if (vol_mat) {
        ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), "[Volumetric Properties]");

        // --- VDB LOAD UI ---
        ImGui::Separator();
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
                     auto transform_handle = sel.selected.object->getTransformHandle();
                     
                     if (transform_handle) {
                         // Assumes object is a unit cube originally (-0.5 to 0.5 or similar unit size)
                         
                         Matrix4x4 S = Matrix4x4::scaling(size);
                         Matrix4x4 T = Matrix4x4::translation(center);
                         
                         // Apply Scale then Translation
                         Matrix4x4 new_transform = T * S;
                         
                         transform_handle->setBase(new_transform);
                         
                         // CRITICAL: A mesh is composed of multiple triangles sharing the same TransformHandle.
                         // We must update the vertex cache for ALL of them, otherwise the CPU BVH and OptiX GAS
                         // will be inconsistent (only 1 triangle updated).
                         for (auto& obj : ctx.scene.world.objects) {
                             if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                                 // Check if they share the exact same transform handle instance
                                 if (tri->getTransformHandle() == transform_handle) {
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
        ImGui::Separator();

        bool changed = false;

        // ¦¦ DISTINCT INPUT FIELD STYLING FOR VOLUMETRIC PANEL (For Colors only) ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
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
                ctx.renderer.updateBackendMaterials(ctx.scene);
                ctx.backend_ptr->resetAccumulation();
            }
        }

    }

    else if (pbsdf) {
        drawPrincipledBSDFEditor(pbsdf, active_mat_id, ctx);
    }
    else {
        ImGui::TextDisabled("Unknown material type.");
    }
}

// ===============================================================================
// REUSABLE MATERIAL EDITOR WIDGET
// ===============================================================================
void SceneUI::drawPrincipledBSDFEditor(PrincipledBSDF* pbsdf, uint16_t mat_id, UIContext& ctx) {
    if (!pbsdf) return;
    
    bool changed = false;
    bool texture_changed = false;

    // Helper functions re-implemented locally to Context
    auto SyncGpuMaterial = [&](PrincipledBSDF* mat) -> void {
        if (!mat->gpuMaterial) mat->gpuMaterial = std::make_shared<GpuMaterial>();

        Vec3 alb = mat->albedoProperty.color;
        mat->gpuMaterial->albedo = make_float3((float)alb.x, (float)alb.y, (float)alb.z);
        mat->gpuMaterial->roughness = (float)mat->roughnessProperty.color.x;
        mat->gpuMaterial->metallic = (float)mat->metallicProperty.intensity;

        Vec3 em = mat->emissionProperty.color;
        float emStr = mat->emissionProperty.intensity;
        mat->gpuMaterial->emission = make_float3((float)em.x * emStr, (float)em.y * emStr, (float)em.z * emStr);

        mat->gpuMaterial->ior = mat->ior;
        mat->gpuMaterial->transmission = mat->transmission;
        mat->gpuMaterial->opacity = mat->opacityProperty.alpha;
        
        // SSS (Random Walk)
        mat->gpuMaterial->subsurface = mat->subsurface;
        Vec3 sssColor = mat->subsurfaceColor;
        mat->gpuMaterial->subsurface_color = make_float3((float)sssColor.x, (float)sssColor.y, (float)sssColor.z);
        Vec3 sssRadius = mat->subsurfaceRadius;
        mat->gpuMaterial->subsurface_radius = make_float3((float)sssRadius.x, (float)sssRadius.y, (float)sssRadius.z);
        mat->gpuMaterial->subsurface_scale = mat->subsurfaceScale;
        mat->gpuMaterial->subsurface_anisotropy = mat->subsurfaceAnisotropy;
        mat->gpuMaterial->subsurface_ior = mat->subsurfaceIOR;
        
        // Clear Coat
        mat->gpuMaterial->clearcoat = mat->clearcoat;
        mat->gpuMaterial->clearcoat_roughness = mat->clearcoatRoughness;
        
        // Translucent
        mat->gpuMaterial->translucent = mat->translucent;
        
        // Anisotropic
        mat->gpuMaterial->anisotropic = mat->anisotropic;

        // Sheen (Water flags)
        mat->gpuMaterial->sheen = mat->sheen;
        mat->gpuMaterial->sheen_tint = mat->sheen_tint;
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
            ctx.renderer.updateBackendMaterials(ctx.scene);
            ctx.backend_ptr->resetAccumulation();
        }
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
    // We need access to ctx.selection.selected.name which is available via ctx
    std::string obj_n = ctx.selection.selected.name;
    
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


    // Scalar Properties
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

    ImGui::Separator();
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

    if (UIWidgets::BeginSection("Texture Maps", ImVec4(0.4f, 1.0f, 0.6f, 1.0f))) {
        static std::shared_ptr<Texture> texture_clipboard = nullptr;

        auto DrawTextureSlot = [&](const char* label, std::shared_ptr<Texture>& tex_ref, TextureType type) {
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
                
                // Truncate if long 
                if (shortName.length() > 20) {
                     shortName = shortName.substr(0, 8) + "..." + shortName.substr(shortName.length()-5);
                }

                ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "%s", shortName.c_str());
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n(%dx%d)", name.c_str(), tex_ref->width, tex_ref->height);
                
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

        DrawTextureSlot("Albedo", pbsdf->albedoProperty.texture, TextureType::Albedo);
        DrawTextureSlot("Normal", pbsdf->normalProperty.texture, TextureType::Normal);
        DrawTextureSlot("Roughness", pbsdf->roughnessProperty.texture, TextureType::Roughness);
        DrawTextureSlot("Metallic", pbsdf->metallicProperty.texture, TextureType::Metallic);
        DrawTextureSlot("Emission", pbsdf->emissionProperty.texture, TextureType::Emission);
        DrawTextureSlot("Opacity", pbsdf->opacityProperty.texture, TextureType::Opacity);

        UIWidgets::EndSection();
    }

    if (changed || texture_changed) {
        TriggerMaterialUpdate(texture_changed);
        g_ProjectManager.markModified();
    }

    ImGui::PopStyleVar(3);
    ImGui::PopStyleColor(6);
}
