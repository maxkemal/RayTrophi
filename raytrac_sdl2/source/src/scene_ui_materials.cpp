// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - MATERIAL EDITOR
// ═══════════════════════════════════════════════════════════════════════════════
// This file handles the Material properties panel.
// ═══════════════════════════════════════════════════════════════════════════════

#include "scene_ui.h"
#include "imgui.h"
#include "MaterialManager.h"
#include "PrincipledBSDF.h"
#include "Volumetric.h"
#include "scene_data.h"
#include <ProjectManager.h>

// ═════════════════════════════════════════════════════════════════════════════
// MANUEL TAŞIMA TALİMATI (MANUAL TRANSFER INSTRUCTIONS):
// ═════════════════════════════════════════════════════════════════════════════
// Lütfen aşağıdaki fonksiyonu `scene_ui.cpp` dosyasından buraya Kes/Yapıştır yapın:
// Please Cut/Paste the following function from `scene_ui.cpp` to here:
//
// 1. void SceneUI::drawMaterialPanel(UIContext& ctx)
//    (Tahmini Satırlar / Approx Lines: ~4600 - 5461)
//
// ═════════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════════
// MATERIAL & TEXTURE EDITOR PANEL
// ═══════════════════════════════════════════════════════════════════════════════
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

    // ─────────────────────────────────────────────────────────────────────────
    // 1. SCAN FOR USED MATERIALS (SLOTS)
    // ─────────────────────────────────────────────────────────────────────────
    // We need to know which unique material IDs are used by this object's triangles.
    // We'll store them in a list preserving order if possible (or just sorted/unique).
    std::vector<uint16_t> used_material_ids;

    // Simple scan: Iterate all triangles in the cache for this object
    // Note: For very high-poly objects, this might be slow every frame. 
    // Optimization: Cache this list in SceneSelection or similar if needed.
    for (const auto& pair : cache_it->second) {
        std::shared_ptr<Triangle> tri = pair.second;
        if (tri) {
            uint16_t mid = tri->getMaterialID();
            bool found = false;
            for (uint16_t existing_id : used_material_ids) {
                if (existing_id == mid) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                used_material_ids.push_back(mid);
            }
        }
    }

    // Sort for consistent display order
    // std::sort(used_material_ids.begin(), used_material_ids.end()); 
    // (Optional: Sorting might re-order slots unexpectedly if ids change, but keeps it stable)

    if (used_material_ids.empty()) {
        ImGui::TextDisabled("No geometry/materials found.");
        return;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. SLOT SELECTION UI
    // ─────────────────────────────────────────────────────────────────────────
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
    uint16_t active_mat_id = used_material_ids[active_slot_index];
    Material* active_mat_ptr = MaterialManager::getInstance().getMaterial(active_mat_id);
    std::string current_mat_name = active_mat_ptr ? active_mat_ptr->materialName : "None";

    // ─────────────────────────────────────────────────────────────────────────
    // 3. ASSIGN MATERIAL TO ACTIVE SLOT
    // ─────────────────────────────────────────────────────────────────────────
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

                    SCENE_LOG_INFO("Replaced material in Slot " + std::to_string(active_slot_index) +
                        " (ID: " + std::to_string(active_mat_id) + " -> " + std::to_string(i) +
                        "). Triangles updated: " + std::to_string(count_replaced));

                    // Trigger Rebuilds
                    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                    ctx.renderer.resetCPUAccumulation();
                    if (ctx.optix_gpu_ptr) {
                        ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                    g_ProjectManager.markModified();
                }
            }
            if (is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    // ─────────────────────────────────────────────────────────────────────────
    // SHORTCUTS (New Surface / Volume) - affecting Active Slot
    // ─────────────────────────────────────────────────────────────────────────
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
            // Trigger updates
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) {
                ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                ctx.optix_gpu_ptr->resetAccumulation();
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
                1.0,            // density
                0.1,            // absorption
                0.5,            // scattering
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
            // Trigger updates
            ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) {
                ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
            g_ProjectManager.markModified();
        }
        ImGui::EndPopup();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // MATERIAL EDITOR (Context-Aware for Active Slot)
    // ─────────────────────────────────────────────────────────────────────────
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

    if (vol_mat) {
        ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), "[Volumetric Properties]");

        bool changed = false;

        // --- Reuse Volumetric UI Logic ---
        Vec3 alb = vol_mat->getAlbedo();
        if (ImGui::ColorEdit3("Scattering Color", &alb.x)) { vol_mat->setAlbedo(alb); changed = true; }

        float dens = (float)vol_mat->getDensity();
        if (ImGui::SliderFloat("Density", &dens, 0.0f, 10.0f)) { vol_mat->setDensity(dens); changed = true; }

        float scatt = (float)vol_mat->getScattering();
        if (ImGui::SliderFloat("Scattering", &scatt, 0.0f, 5.0f)) { vol_mat->setScattering(scatt); changed = true; }

        float abs = (float)vol_mat->getAbsorption();
        if (ImGui::SliderFloat("Absorption", &abs, 0.0f, 2.0f)) { vol_mat->setAbsorption(abs); changed = true; }

        float g = (float)vol_mat->getG();
        if (ImGui::SliderFloat("Anisotropy", &g, -0.99f, 0.99f)) { vol_mat->setG(g); changed = true; }

        // Emission Color with Strength Control                
        Vec3 emis = vol_mat->getEmissionColor();
        // Emissive Strength Logic (Infer max component as strength)
        float max_e = emis.x;
        if (emis.y > max_e) max_e = emis.y;
        if (emis.z > max_e) max_e = emis.z;
        float strength = (max_e > 1.0f) ? max_e : 1.0f;
        if (max_e < 0.001f) strength = 1.0f; // Default for black

        Vec3 normalized_emis = (max_e > 0.001f) ? emis * (1.0f / strength) : emis;

        if (ImGui::ColorEdit3("Emission Color", &normalized_emis.x)) {
            vol_mat->setEmissionColor(normalized_emis * strength);
            changed = true;
        }
        if (ImGui::DragFloat("Emission Strength", &strength, 0.1f, 0.0f, 1000.0f)) {
            vol_mat->setEmissionColor(normalized_emis * strength);
            changed = true;
        }

        float n_scale = vol_mat->getNoiseScale();
        if (ImGui::DragFloat("Noise Scale", &n_scale, 0.01f, 0.01f, 100.0f)) {
            vol_mat->setNoiseScale(n_scale);
            changed = true;
        }

        float void_t = vol_mat->getVoidThreshold();
        if (ImGui::SliderFloat("Void Threshold", &void_t, 0.0f, 1.0f)) {
            vol_mat->setVoidThreshold(void_t);
            changed = true;
        }
        UIWidgets::HelpMarker("Controls empty space amount (Higher = More Voids)");

        // Ray Marching Quality Settings
        float step_size = vol_mat->getStepSize();
        if (ImGui::DragFloat("Step Size (Quality)", &step_size, 0.001f, 0.001f, 1.0f, "%.4f")) {
            vol_mat->setStepSize(step_size);
            changed = true;
        }
        UIWidgets::HelpMarker("Smaller values = Higher Quality (Slower)");

        int max_steps = vol_mat->getMaxSteps();
        if (ImGui::DragInt("Max Steps", &max_steps, 1, 1, 1000)) {
            vol_mat->setMaxSteps(max_steps);
            changed = true;
        }

        // ═══════════════════════════════════════════════════════════════════
        // MULTI-SCATTERING CONTROLS (NEW)
        // ═══════════════════════════════════════════════════════════════════
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f), "Multi-Scattering");

        float multi_scatter = vol_mat->getMultiScatter();
        if (ImGui::SliderFloat("Multi-Scatter##vol", &multi_scatter, 0.0f, 1.0f)) {
            vol_mat->setMultiScatter(multi_scatter);
            changed = true;
        }
        UIWidgets::HelpMarker("Controls multi-scattering brightness (0=single scatter, 1=full multi-scatter)");

        float g_back = vol_mat->getGBack();
        if (ImGui::SliderFloat("Backward G##vol", &g_back, -0.99f, 0.0f)) {
            vol_mat->setGBack(g_back);
            changed = true;
        }
        UIWidgets::HelpMarker("Backward scattering anisotropy for silver lining effect");

        float lobe_mix = vol_mat->getLobeMix();
        if (ImGui::SliderFloat("Lobe Mix##vol", &lobe_mix, 0.0f, 1.0f)) {
            vol_mat->setLobeMix(lobe_mix);
            changed = true;
        }
        UIWidgets::HelpMarker("Forward/Backward lobe blend (1.0=all forward, 0.0=all backward)");

        int light_steps = vol_mat->getLightSteps();
        if (ImGui::SliderInt("Shadow Steps##vol", &light_steps, 0, 8)) {
            vol_mat->setLightSteps(light_steps);
            changed = true;
        }
        UIWidgets::HelpMarker("Light march steps for self-shadowing (0=disabled, 4-8 recommended)");

        float shadow_str = vol_mat->getShadowStrength();
        if (ImGui::SliderFloat("Shadow Strength##vol", &shadow_str, 0.0f, 1.0f)) {
            vol_mat->setShadowStrength(shadow_str);
            changed = true;
        }
        UIWidgets::HelpMarker("Self-shadow intensity (0=no shadows, 1=full shadows)");

        if (changed) {
            // OPTIMIZED: Material property change - No geometry rebuild needed!
            // Only reset accumulation and update GPU material buffers
            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) {
                ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
        }

    }
    else if (pbsdf) {
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.6f, 1.0f), "[Surface Properties]");

        bool changed = false;
        bool texture_changed = false;

        // ─────────────────────────────────────────────────────────────────────
        // HELPER LAMBDAS (Re-introduced for Texture Editing)
        // ─────────────────────────────────────────────────────────────────────

        // 1. Sync GPU Material
        auto SyncGpuMaterial = [](PrincipledBSDF* mat) {
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
            };

        // 2. Update Texture Bundle for a Triangle
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

        // 3. Trigger Update
        auto TriggerMaterialUpdate = [&](bool needs_texture_update) {
            SyncGpuMaterial(pbsdf);

            if (needs_texture_update) {
                // Update all triangles using this material ID
                // We don't have the scan readily available unless we re-scan or pass used_material_ids
                // But we can iterate all objects effectively.
                for (auto& obj : ctx.scene.world.objects) {
                    auto t = std::dynamic_pointer_cast<Triangle>(obj);
                    if (t && t->getMaterialID() == active_mat_id) {
                        UpdateTriangleTextureBundle(t, pbsdf);
                    }
                }
            }

            ctx.renderer.resetCPUAccumulation();
            if (ctx.optix_gpu_ptr) {
                // OPTIMIZED: Material property change - use fast update path
                // Full rebuild only needed for texture changes (needs_texture_update=true)
                if (needs_texture_update) {
                    ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                }
                else {
                    ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                }
                ctx.optix_gpu_ptr->resetAccumulation();
            }
            };

        // 4. Load Texture Dialog
        auto LoadTextureFromDialog = [&](TextureType type, const std::string& initialDir = "", const std::string& defaultFile = "") -> std::shared_ptr<Texture> {

            std::string path = SceneUI::openFileDialogW(L"Image Files\0*.png;*.jpg;*.jpeg;*.tga;*.bmp;*.exr;*.hdr\0", initialDir, defaultFile);
            if (path.empty()) return nullptr;

            auto tex = std::make_shared<Texture>(path, type);
            if (tex && tex->is_loaded()) {
                tex->upload_to_gpu();
                SCENE_LOG_INFO("Loaded texture: " + path);
                return tex;
            }
            SCENE_LOG_WARN("Failed to load texture: " + path);
            return nullptr;
            };

        // ─── PROPERTIES UI ───────────────────────────────────────────────────

        // Base Color (Albedo)
        Vec3 albedo = pbsdf->albedoProperty.color;
        float albedo_arr[3] = { (float)albedo.x, (float)albedo.y, (float)albedo.z };
        bool albKeyed = isMatKeyed(active_mat_id, true, false, false, false, false, false, false, false);
        if (KeyframeButton("##MAlb", albKeyed)) { insertMatPropKey(active_mat_ptr, active_mat_id, true, false, false, false, false, false, false, false); }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(albKeyed ? "REMOVE Base Color key" : "ADD Base Color key");
        ImGui::SameLine();
        if (ImGui::ColorEdit3("Base Color", albedo_arr)) {
            pbsdf->albedoProperty.color = Vec3(albedo_arr[0], albedo_arr[1], albedo_arr[2]);
            changed = true;
        }

        // Metallic
        float metallic = pbsdf->metallicProperty.intensity;
        bool metKeyed = isMatKeyed(active_mat_id, false, false, false, true, false, false, false, false);
        if (KeyframeButton("##MMet", metKeyed)) { insertMatPropKey(active_mat_ptr, active_mat_id, false, false, false, true, false, false, false, false); }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(metKeyed ? "REMOVE Metallic key" : "ADD Metallic key");
        ImGui::SameLine();
        if (ImGui::SliderFloat("Metallic", &metallic, 0.0f, 1.0f, "%.3f")) {
            pbsdf->metallicProperty.intensity = metallic;
            changed = true;
        }

        // Roughness
        float roughness = (float)pbsdf->roughnessProperty.color.x;
        bool rghKeyed = isMatKeyed(active_mat_id, false, false, true, false, false, false, false, false);
        if (KeyframeButton("##MRgh", rghKeyed)) { insertMatPropKey(active_mat_ptr, active_mat_id, false, false, true, false, false, false, false, false); }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(rghKeyed ? "REMOVE Roughness key" : "ADD Roughness key");
        ImGui::SameLine();
        if (ImGui::SliderFloat("Roughness", &roughness, 0.0f, 1.0f, "%.3f")) {
            pbsdf->roughnessProperty.color = Vec3(roughness);
            changed = true;
        }

        // IOR
        float ior = pbsdf->ior;
        bool iorKeyed = isMatKeyed(active_mat_id, false, false, false, false, false, false, true, false);
        if (KeyframeButton("##MIOR", iorKeyed)) { insertMatPropKey(active_mat_ptr, active_mat_id, false, false, false, false, false, false, true, false); }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(iorKeyed ? "REMOVE IOR key" : "ADD IOR key");
        ImGui::SameLine();
        if (ImGui::SliderFloat("IOR", &ior, 1.0f, 3.0f, "%.3f")) {
            pbsdf->ior = ior;
            changed = true;
        }

        // Transmission
        float transmission = pbsdf->transmission;
        bool trnKeyed = isMatKeyed(active_mat_id, false, false, false, false, false, true, false, false);
        if (KeyframeButton("##MTrn", trnKeyed)) { insertMatPropKey(active_mat_ptr, active_mat_id, false, false, false, false, false, true, false, false); }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(trnKeyed ? "REMOVE Transmission key" : "ADD Transmission key");
        ImGui::SameLine();
        if (ImGui::SliderFloat("Transmission", &transmission, 0.0f, 1.0f, "%.3f")) {
            pbsdf->setTransmission(transmission, pbsdf->ior);
            changed = true;
        }

        // Specular
        float specular = pbsdf->specularProperty.intensity;
        bool specKeyed = isMatKeyed(active_mat_id, false, false, false, false, false, false, false, true);
        if (KeyframeButton("##MSpec", specKeyed)) { insertMatPropKey(active_mat_ptr, active_mat_id, false, false, false, false, false, false, false, true); }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(specKeyed ? "REMOVE Specular key" : "ADD Specular key");
        ImGui::SameLine();
        if (ImGui::SliderFloat("Specular", &specular, 0.0f, 1.0f, "%.3f")) {
            pbsdf->specularProperty.intensity = specular;
            changed = true;
        }

        // Opacity
        float opacity = pbsdf->opacityProperty.alpha;
        bool opKeyed = isMatKeyed(active_mat_id, false, true, false, false, false, false, false, false);
        if (KeyframeButton("##MOpa", opKeyed)) { insertMatPropKey(active_mat_ptr, active_mat_id, false, true, false, false, false, false, false, false); }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(opKeyed ? "REMOVE Opacity key" : "ADD Opacity key");
        ImGui::SameLine();
        if (ImGui::SliderFloat("Opacity", &opacity, 0.0f, 1.0f, "%.3f")) {
            pbsdf->opacityProperty.alpha = opacity;
            changed = true;
        }

        // Emission
        Vec3 emission = pbsdf->emissionProperty.color;
        float emission_arr[3] = { (float)emission.x, (float)emission.y, (float)emission.z };
        bool emsKeyed = isMatKeyed(active_mat_id, false, false, false, false, true, false, false, false);
        if (KeyframeButton("##MEms", emsKeyed)) { insertMatPropKey(active_mat_ptr, active_mat_id, false, false, false, false, true, false, false, false); }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(emsKeyed ? "REMOVE Emission key" : "ADD Emission key");
        ImGui::SameLine();
        if (ImGui::ColorEdit3("Emission", emission_arr)) {
            pbsdf->emissionProperty.color = Vec3(emission_arr[0], emission_arr[1], emission_arr[2]);
            changed = true;
        }

        float emissionStr = pbsdf->emissionProperty.intensity;
        if (ImGui::DragFloat("Emission Strength", &emissionStr, 0.1f, 0.0f, 1000.0f)) {
            pbsdf->emissionProperty.intensity = emissionStr;
            changed = true;
        }

        // ─── TEXTURES UI ─────────────────────────────────────────────────────
        if (ImGui::TreeNodeEx("Texture Maps", ImGuiTreeNodeFlags_DefaultOpen)) {

            // Texture Clipboard (Static to persist across frames/slots)
            static std::shared_ptr<Texture> texture_clipboard = nullptr;

            auto DrawTextureSlot = [&](const char* label, std::shared_ptr<Texture>& tex_ref, TextureType type) {
                ImGui::PushID(label);

                ImGui::Text("%s:", label);
                ImGui::SameLine(100);

                if (tex_ref && tex_ref->is_loaded()) {
                    // Show texture info
                    std::string name = tex_ref->name;
                    std::string shortName = name;
                    size_t lastSlash = name.find_last_of("/\\");
                    if (lastSlash != std::string::npos) shortName = name.substr(lastSlash + 1);
                    if (shortName.length() > 20) shortName = "..." + shortName.substr(shortName.length() - 17);

                    ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "%s", shortName.c_str());
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s\n(%dx%d)", name.c_str(), tex_ref->width, tex_ref->height);

                    ImGui::SameLine();
                    ImGui::TextDisabled("(%dx%d)", tex_ref->width, tex_ref->height);

                    // Clear button
                    ImGui::SameLine();
                    if (ImGui::SmallButton("X##Clear")) {
                        tex_ref = nullptr;
                        texture_changed = true;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove Texture");

                    // Copy Button
                    ImGui::SameLine();
                    if (ImGui::Button("C##Copy")) {
                        texture_clipboard = tex_ref;
                    }
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Copy Texture to Clipboard");

                }
                else {
                    ImGui::TextDisabled("[None]");

                    // Paste Button (Only if clipboard has content)
                    if (texture_clipboard) {
                        ImGui::SameLine();
                        if (ImGui::Button("P##Paste")) {
                            tex_ref = texture_clipboard;
                            texture_changed = true;
                        }
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Paste Texture from Clipboard");
                    }
                }

                // Load button
                ImGui::SameLine();
                if (ImGui::SmallButton("Load...")) {
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
                        // Optional: Start from clipboard's location if slot is empty
                        std::string fullPath = texture_clipboard->name;
                        size_t lastSlash = fullPath.find_last_of("/\\");
                        if (lastSlash != std::string::npos) {
                            initialDir = fullPath.substr(0, lastSlash);
                        }
                    }

                    auto new_tex = LoadTextureFromDialog(type, initialDir, defaultFile);
                    if (new_tex) {
                        tex_ref = new_tex;
                        texture_changed = true;
                    }
                }
                ImGui::PopID();
                };

            DrawTextureSlot("Albedo", pbsdf->albedoProperty.texture, TextureType::Albedo);
            DrawTextureSlot("Normal", pbsdf->normalProperty.texture, TextureType::Normal);
            DrawTextureSlot("Roughness", pbsdf->roughnessProperty.texture, TextureType::Roughness);
            DrawTextureSlot("Metallic", pbsdf->metallicProperty.texture, TextureType::Metallic);
            DrawTextureSlot("Emission", pbsdf->emissionProperty.texture, TextureType::Emission);
            DrawTextureSlot("Opacity", pbsdf->opacityProperty.texture, TextureType::Opacity);

            ImGui::TreePop();
        }

        if (changed || texture_changed) {
            TriggerMaterialUpdate(texture_changed);
            g_ProjectManager.markModified();
        }
    }
    else {
        ImGui::TextDisabled("Unknown material type.");
    }
}
