/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_terrain.hpp
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef SCENE_UI_TERRAIN_HPP
#define SCENE_UI_TERRAIN_HPP

#include "scene_ui.h"
#include "SceneSelection.h"
#include "MaterialManager.h" // Added for material selection
#include "OptixWrapper.h"
#include "OptixAccelManager.h"
#include <TerrainManager.h>
#include "PrincipledBSDF.h" // For layer texture editing
#include <set>
#include <random>
#include <algorithm>
#include "InstanceManager.h"
#include "InstanceGroup.h"
#include <thread>
#include <chrono>

// ===============================================================================
// ===============================================================================
// TERRAIN PANEL UI
// ===============================================================================

// Texture Graveyard: Holds textures that are replaced until a rebuild confirms they are safe to delete
static std::vector<std::shared_ptr<Texture>> texture_graveyard;

static void ManageTextureGraveyard() {
    // Only clear invalid textures when NO rebuild is pending
    // This assumes that if rebuild_pending is false, the last rebuild has definitely finished
    // and the SBT has been updated to point to new textures.
    // CAUTION: This requires g_optix_rebuild_pending to be managed reliably in the main loop.
    
    if (!g_optix_rebuild_pending && !texture_graveyard.empty()) {
        texture_graveyard.clear();
        // SCENE_LOG_INFO("Texture graveyard cleared. Old textures released.");
    }
}

void SceneUI::drawTerrainPanel(UIContext& ctx) {
    // SCENE_UI_TERRAIN.HPP is included in Main.cpp, so static vector is shared? 
    // Wait, if included multiple times, static vector duplicate?
    // It is included in Main.cpp usually. If also in SceneUI.cpp, we have problem.
    // Assuming single TU or safe.
    ManageTextureGraveyard();
    
    // -----------------------------------------------------------------------------
    // 1. TERRAIN MANAGEMENT (Create / Import)
    // -----------------------------------------------------------------------------
    if (UIWidgets::BeginSection("Terrain Management", ImVec4(0.4f, 0.8f, 0.5f, 1.0f), true)) {
        static int new_res = 128;
        static float new_size = 100.0f;

        // Creation Params
        ImGui::TextDisabled("Create New Terrain:");
        ImGui::PushItemWidth(160.0f);
        ImGui::DragInt("Resolution", &new_res, 1, 64, 4096);
        ImGui::DragFloat("Size (m)", &new_size, 10.0f, 10.0f, 100000.0f);
        ImGui::PopItemWidth();

        if (UIWidgets::PrimaryButton("Create Grid Terrain", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
            auto t = TerrainManager::getInstance().createTerrain(ctx.scene, new_res, new_size);
            if (t) {
                terrain_brush.active_terrain_id = t->id;
                // Open node graph for advanced editing immediately after creation
                show_terrain_graph = true;
                // Ensure graph is visible: close timeline and set bottom panel height
                show_animation_panel = false;
                bottom_panel_height = 320.0f;
                // If terrain has generated mesh triangles, select one to sync viewport selection
                if (!t->mesh_triangles.empty() && ctx.selection.hasSelection() == false) {
                    auto tri = t->mesh_triangles.front();
                    if (tri) ctx.selection.selectObject(tri, -1, tri->nodeName);
                }
                SCENE_LOG_INFO("Terrain created: " + t->name);
                ctx.renderer.resetCPUAccumulation();
                g_bvh_rebuild_pending = true;
                g_optix_rebuild_pending = true;
            }
        }

        ImGui::SameLine();
        UIWidgets::HelpMarker("Use the node-based terrain graph (bottom panel) to build complex, procedural terrain structures and advanced edits.");

        ImGui::Spacing();
        ImGui::Separator();
        
        // ---------------- TERRAIN LIST ----------------
        ImGui::TextDisabled("Terrain List:");
        auto& terrains = TerrainManager::getInstance().getTerrains();

        // Cleanup: If terrain triangles were removed from the scene by other code,
        // remove stale terrain entries so the list stays in sync with the viewport.
        std::vector<int> stale_ids;
        for (auto& tt : terrains) {
            bool found = false;
            for (auto& tri_ptr : tt.mesh_triangles) {
                if (std::find(ctx.scene.world.objects.begin(), ctx.scene.world.objects.end(), tri_ptr) != ctx.scene.world.objects.end()) {
                    found = true; break;
                }
            }
            if (!found) stale_ids.push_back(tt.id);
        }
        for (int sid : stale_ids) {
            if (ctx.optix_gpu_ptr && g_hasCUDA) cudaDeviceSynchronize();
            TerrainManager::getInstance().removeTerrain(ctx.scene, sid);
            if (terrain_brush.active_terrain_id == sid) terrain_brush.active_terrain_id = -1;
            SCENE_LOG_INFO("Removed stale terrain: " + std::to_string(sid));
            ctx.renderer.resetCPUAccumulation();
            g_bvh_rebuild_pending = true; g_optix_rebuild_pending = true;
        }
        
        if (terrains.empty()) {
            ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "No terrains created.");
        }
        else {
             // Create a scrollable list with fixed height
             if (ImGui::BeginChild("TerrainList", ImVec2(0, 150), true)) {
                 for (size_t ti = 0; ti < terrains.size(); ++ti) {
                     auto& t = terrains[ti];
                     bool is_selected = (terrain_brush.active_terrain_id == t.id);
                     std::string label = t.name + " (ID: " + std::to_string(t.id) + ")";
                     
                     if (ImGui::Selectable(label.c_str(), is_selected)) {
                         terrain_brush.active_terrain_id = t.id;
                         // Select a representative triangle so the viewport selection matches the list
                         if (!t.mesh_triangles.empty()) {
                             auto tri = t.mesh_triangles.front();
                             if (tri) ctx.selection.selectObject(tri, -1, tri->nodeName);
                         }
                         SCENE_LOG_INFO("Active terrain switched to: " + t.name);
                         // Trigger rebuild for highlighting or gizmos if necessary
                         ctx.renderer.resetCPUAccumulation(); 
                     }
                     
                     if (is_selected) {
                         ImGui::SetItemDefaultFocus();
                     }

                     // Right-click context: delete single terrain
                     std::string popup_id = "terrain_ctx_" + std::to_string(t.id);
                     if (ImGui::BeginPopupContextItem(popup_id.c_str())) {
                         if (ImGui::MenuItem("Delete")) {
                             if (ctx.optix_gpu_ptr && g_hasCUDA) cudaDeviceSynchronize();
                             TerrainManager::getInstance().removeTerrain(ctx.scene, t.id);
                             if (terrain_brush.active_terrain_id == t.id) terrain_brush.active_terrain_id = -1;
                             SCENE_LOG_INFO("Terrain deleted from list: " + t.name);
                             ctx.renderer.resetCPUAccumulation();
                             g_bvh_rebuild_pending = true; g_optix_rebuild_pending = true;
                             ImGui::EndPopup();
                             break; // terrains vector changed, break out of loop
                         }
                         ImGui::EndPopup();
                     }
                 }
             }
             ImGui::EndChild();
        }

        ImGui::Spacing();
        ImGui::Separator();

        if (UIWidgets::DangerButton("Clear All Terrains", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
            if (ctx.optix_gpu_ptr && g_hasCUDA) cudaDeviceSynchronize();
            TerrainManager::getInstance().removeAllTerrains(ctx.scene);
            terrain_brush.active_terrain_id = -1;
            SCENE_LOG_INFO("All terrains cleared.");
            ctx.renderer.resetCPUAccumulation();
            g_bvh_rebuild_pending = true;
            g_optix_rebuild_pending = true;
        }
        UIWidgets::EndSection();
    }

    ImGui::Spacing();

    // -----------------------------------------------------------------------------
    // 2. MESH QUALITY SETTINGS
    // -----------------------------------------------------------------------------
    if (terrain_brush.active_terrain_id != -1) {
        auto* t = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
        if (t) {
            if (UIWidgets::BeginSection("Mesh Quality", ImVec4(0.6f, 1.0f, 0.8f, 1.0f), true)) {
                // Normal Quality Dropdown
                // Normal Quality Dropdown
                const char* normalQualityItems[] = { "Fast (4-neighbor)", "Sobel (8-neighbor)", "High Quality" };
                int nq = (int)t->normal_quality;
                ImGui::SetNextItemWidth(160.0f);
                if (ImGui::Combo("Normal Quality", &nq, normalQualityItems, IM_ARRAYSIZE(normalQualityItems))) {
                    t->normal_quality = (NormalQuality)nq;
                    t->dirty_mesh = true;
                    TerrainManager::getInstance().updateTerrainMesh(t);
                    ctx.renderer.resetCPUAccumulation();
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                }
                ImGui::SameLine(); UIWidgets::HelpMarker("Fast: Simple 4-neighbor average\nSobel: Smooth 8-neighbor filter (recommended)\nHigh Quality: Enhanced edge detection");
                
                // Normal Strength Slider
                ImGui::PushItemWidth(160.0f);
                if (SceneUI::DrawSmartFloat("nstr", "Normal Strength", &t->normal_strength, 0.1f, 10.0f, "%.2f", false, nullptr, 16)) {
                    t->dirty_mesh = true;
                    TerrainManager::getInstance().updateTerrainMesh(t);
                    ctx.renderer.resetCPUAccumulation();
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                }
                ImGui::PopItemWidth();
                ImGui::SameLine(); UIWidgets::HelpMarker("Intensity of surface detail lighting (bumpiness).");
                
                // Dirty Sectors Info
                int dirtySectors = t->dirty_region.countDirtySectors();
                if (dirtySectors > 0) {
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Pending Sectors: %d/256", dirtySectors);
                }
                UIWidgets::EndSection();
            }
            
            ImGui::Spacing();
            
            ImGui::Spacing();

            // 3. LAYER MANAGEMENT
            if (UIWidgets::BeginSection("Materials & Layers", ImVec4(0.5f, 0.8f, 1.0f, 1.0f), true)) {
                ImGui::SameLine(); UIWidgets::HelpMarker("Manages which materials (grass, rock, snow, etc.) are used at different heights and slopes.");

            if (t->layers.empty()) {
                if (UIWidgets::PrimaryButton("Initialize Layers", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
                    TerrainManager::getInstance().initLayers(t);
                    SCENE_LOG_INFO("Terrain layers initialized for: " + t->name);
                    if (ctx.optix_gpu_ptr) {
                        ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                        ctx.optix_gpu_ptr->resetAccumulation();
                        ctx.renderer.resetCPUAccumulation();
                    }
                }
            }
            else {
                // Layer Editors
                static const char* autoLayerNames[4] = {"Grass", "Rock", "Snow", "Flow"};
                static const Vec3 autoLayerColors[4] = {
                    Vec3(0.3f, 0.5f, 0.2f),  // Grass
                    Vec3(0.4f, 0.4f, 0.4f),  // Rock
                    Vec3(0.9f, 0.9f, 0.95f), // Snow
                    Vec3(0.5f, 0.35f, 0.2f)  // Flow (Riverbeds)
                };
                
                for (int i = 0; i < 4; i++) {
                    ImGui::PushID(i);
                    std::string layerName = "";
                    ImVec4 layerColor;
                    if (i == 0) { layerName = "Layer 0 (Grass/Flat)"; layerColor = ImVec4(0.5, 0.8, 0.5, 1); }
                    else if (i == 1) { layerName = "Layer 1 (Rock/Slope)"; layerColor = ImVec4(0.6, 0.6, 0.6, 1); }
                    else if (i == 2) { layerName = "Layer 2 (Snow/Peak)"; layerColor = ImVec4(0.9, 0.9, 1.0, 1); }
                    else { layerName = "Layer 3 (Flow/River)"; layerColor = ImVec4(0.5, 0.7, 1.0, 1); }

                    ImGui::TextColored(layerColor, "%s", layerName.c_str());

                    // Material Selector
                    std::string currentMatName = "None";
                    if (t->layers[i]) currentMatName = t->layers[i]->materialName;
                    else currentMatName = "[None]";

                    ImGui::SetNextItemWidth(160.0f);
                    if (ImGui::BeginCombo("Material", currentMatName.c_str())) {
                        auto& materials = MaterialManager::getInstance().getAllMaterials();
                        for (auto& mat : materials) {
                            bool is_selected = (t->layers[i] == mat);
                            if (ImGui::Selectable(mat->materialName.c_str(), is_selected)) {
                                t->layers[i] = mat;
                                if (ctx.optix_gpu_ptr) {
                                    ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                                    ctx.optix_gpu_ptr->resetAccumulation();
                                    ctx.renderer.resetCPUAccumulation();
                                }
                            }
                            if (is_selected) ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndCombo();
                    }
                    
                    // Auto-create material if slot is empty
                    if (!t->layers[i]) {
                        ImGui::SameLine();
                        if (ImGui::SmallButton("+New")) {
                            std::string matName = "Terrain_" + std::string(autoLayerNames[i]);
                            auto newMat = std::make_shared<PrincipledBSDF>(autoLayerColors[i], 0.8f, 0.0f);
                            newMat->materialName = matName;
                            MaterialManager::getInstance().addMaterial(matName, newMat);
                            t->layers[i] = newMat;
                            if (ctx.optix_gpu_ptr) {
                                ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                                ctx.optix_gpu_ptr->resetAccumulation();
                                ctx.renderer.resetCPUAccumulation();
                            }
                            SCENE_LOG_INFO("Created terrain layer material: " + matName);
                        }
                    }

                    // UV Scale
                        ImGui::PushItemWidth(160.0f);
                        if (SceneUI::DrawSmartFloat("uvs", "UV Tile Scale", &t->layer_uv_scales[i], 0.1f, 1000.0f, "%.1f", false, nullptr, 12)) {
                            if (ctx.optix_gpu_ptr) {
                                ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                                ctx.optix_gpu_ptr->resetAccumulation();
                                ctx.renderer.resetCPUAccumulation();
                            }
                        }
                        ImGui::PopItemWidth();
                    
                    // Fill Button
                     ImGui::SameLine();
                     if (ImGui::SmallButton("Fill Mask")) {
                        if (t->splatMap && t->splatMap->is_loaded()) {
                            auto& pixels = t->splatMap->pixels; // vector<CompactVec4>
                            for (auto& p : pixels) {
                                if (i == 0) p.r = 255;
                                else if (i == 1) p.g = 255;
                                else if (i == 2) p.b = 255;
                                else if (i == 3) p.a = 255;
                            }
                            t->splatMap->updateGPU(); // Helper needed? or manually?
                            // Texture::updateGPU() exists in Texture.h (I saw it).
                            t->splatMap->updateGPU();
                            if (ctx.optix_gpu_ptr) ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                            if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                            SCENE_LOG_INFO("Filled mask for Layer " + std::to_string(i));
                        }
                     }

                    // Inline Material Editing (Textures)
                    if (t->layers[i]) {
                        auto pMat = std::dynamic_pointer_cast<PrincipledBSDF>(t->layers[i]);
                        if (pMat) {
                            ImGui::Indent();
                            
                            // Helper lambda for texture slot
                            auto DrawTextureSlot = [&](const char* label, std::shared_ptr<Texture>& texSlot, bool isNormal = false) {
                                // 1. Label column (Fixed width)
                                ImGui::Text("%s:", label); 
                                ImGui::SameLine(80); // Fixed start for value

                                // 2. Value (Filename only)
                                std::string dispName = "[None]";
                                std::string tooltipPath = "";
                                if (texSlot && !texSlot->name.empty()) {
                                    dispName = std::filesystem::path(texSlot->name).filename().string();
                                    tooltipPath = texSlot->name;
                                }
                                
                                // Truncate if too long to fit before buttons
                                float availWidth = ImGui::GetContentRegionAvail().x;
                                float buttonsWidth = 70.0f; // Space for Load + X
                                float textWidth = availWidth - buttonsWidth;
                                
                                ImGui::TextDisabled("%s", dispName.c_str());

                                if (!tooltipPath.empty() && ImGui::IsItemHovered()) {
                                    ImGui::SetTooltip("%s", tooltipPath.c_str());
                                }

                                // 3. Buttons (Right Aligned)
                                ImGui::SameLine();
                                float xTarget = ImGui::GetWindowContentRegionMax().x - buttonsWidth;
                                if (xTarget > ImGui::GetCursorPosX()) {
                                    ImGui::SetCursorPosX(xTarget);
                                }

                                if (ImGui::SmallButton((std::string("Load##") + label).c_str())) {
                                    std::string path = SceneUI::openFileDialogW(L"Image Files\0*.png;*.jpg;*.jpeg;*.bmp\0");
                                    if (!path.empty()) {
                                        // CRITICAL: Stop rendering before modifying GPU resources to prevent "illegal memory access"
                                        ctx.renderer.stopRendering();
                                        // Wait a tiny bit to ensure kernel finished
                                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                                                        if (g_hasCUDA) cudaDeviceSynchronize(); // Force GPU idle
                                        
                                        TextureType type = isNormal ? TextureType::Normal : TextureType::Albedo;
                                        if (std::string(label) == "Roughness") type = TextureType::Roughness;
                                        // GRAVEYARD: Keep old texture alive until rebuild completes
                                        if (texSlot) {
                                            texture_graveyard.push_back(texSlot);
                                        }
                                        
                                        texSlot = std::make_shared<Texture>(path, type);
                                        
                                        // Update Material properties (scalars)
                                        if (ctx.optix_gpu_ptr) {
                                            ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                                        }

                                        // CRITICAL: Request Geometry/SBT Rebuild to update texture handles in SBT
                                        // g_optix_rebuild_pending = true; // Fast sync handles this now

                                        ctx.renderer.resetCPUAccumulation();
                                        if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                                        
                                        ctx.renderer.resumeRendering();
                                    }
                                }
                                
                                if (texSlot) {
                                    ImGui::SameLine();
                                    if (ImGui::SmallButton((std::string("X##") + label).c_str())) {
                                        ctx.renderer.stopRendering();
                                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                                        // GRAVEYARD: Keep old texture alive until rebuild completes
                                        // This prevents GPU accessing destroyed texture handle before SBT update
                                        if (texSlot) {
                                            texture_graveyard.push_back(texSlot);
                                        }

                                                        if (g_hasCUDA) cudaDeviceSynchronize(); // Force GPU idle

                                        texSlot = nullptr;

                                        // Update Material properties (scalars)
                                        if (ctx.optix_gpu_ptr) {
                                            ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                                        }

                                        // CRITICAL: Request Geometry/SBT Rebuild to update texture handles in SBT
                                        // The main loop will handle this, ensuring synchronization.
                                        // g_optix_rebuild_pending = true; // Fast sync handles this now

                                        ctx.renderer.resetCPUAccumulation();
                                        if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                                        
                                        ctx.renderer.resumeRendering();
                                    }
                                }
                            };

                            DrawTextureSlot("Albedo", pMat->albedoProperty.texture);
                            DrawTextureSlot("Normal", pMat->normalProperty.texture, true);
                            DrawTextureSlot("Roughness", pMat->roughnessProperty.texture); 
                            
                            ImGui::Unindent();
                        }
                    }
                    
                    // FOLIAGE UI INTEGRATION (Disabled - Moved to Central Section)
                    if (false);
                    
                    ImGui::Separator();
                    ImGui::PopID();
                }
            }
            UIWidgets::EndSection();
        }

            ImGui::Spacing();

            // 4. CENTRALIZED FOLIAGE SYSTEM
            if (UIWidgets::BeginSection("Foliage System", ImVec4(0.3f, 0.9f, 0.4f, 1.0f), true)) {
                ImGui::SameLine(); UIWidgets::HelpMarker("Allows you to place trees, rocks, grass, etc., on the terrain via brushes or procedural scatter.");
            
            InstanceManager& im = InstanceManager::getInstance();
            static char newFolGroupName[64] = "New Foliage Layer";
            ImGui::SetNextItemWidth(160.0f);
            ImGui::InputText("##NewFolName", newFolGroupName, 64);
            ImGui::SameLine();
            if (UIWidgets::PrimaryButton("Create Layer", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 30))) {
                std::string gName = std::string(newFolGroupName);
                if (gName.find("Foliage") == std::string::npos && gName.find("Scatter") == std::string::npos) 
                    gName = "Foliage_" + gName;
                
                // Ensure unique name
                auto& existingGroups = im.getGroups();
                int suffix = 1;
                std::string finalName = gName;
                bool exists = true;
                while (exists) {
                    exists = false;
                    for (const auto& grp : existingGroups) {
                        if (grp.name == finalName) {
                            exists = true;
                            finalName = gName + "_" + std::to_string(suffix++);
                            break;
                        }
                    }
                }
                
                int newId = im.createGroup(finalName, "", std::vector<std::shared_ptr<Triangle>>{});
                InstanceGroup* g = im.getGroup(newId);
                if (g) g->brush_settings.splat_map_channel = -1; 
                SCENE_LOG_INFO("Created Foliage Layer: " + finalName);
            }
            
            ImGui::Spacing();
            ImGui::TextDisabled("Existing Layers:");
            ImGui::Separator();
            
            int fol_remove_id = -1;
            auto& groups = im.getGroups();
            for (size_t g_idx = 0; g_idx < groups.size(); g_idx++) {
                auto& group = groups[g_idx];
                if (group.name.find("Foliage") == std::string::npos && group.name.find("Scatter") == std::string::npos) continue; 
                
                ImGui::PushID((int)group.id);
                // Header with icon placeholder spacing
                std::string headerLabel = "   " + group.name + " (" + std::to_string(group.instances.size()) + ")###Header" + std::to_string(group.id);
                if (ImGui::TreeNode(headerLabel.c_str())) {
                    ImVec2 hp = ImGui::GetItemRectMin();
                    UIWidgets::DrawIcon(UIWidgets::IconType::Scene, ImVec2(hp.x + 10, hp.y + 2), 14, 0xFFBBBBBB);
                    
                    // RENAME and IDENTITY
                    char renameBuf[64];
                    strncpy(renameBuf, group.name.c_str(), 64);
                    ImGui::SetNextItemWidth(200.0f);
                    if (ImGui::InputText("Layer Name##Rename", renameBuf, 64)) {
                        group.name = std::string(renameBuf);
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("(ID: %d)", group.id);
                    
                    // SOURCES
                     ImGui::Separator();
                     ImGui::Text("   Source Meshes:");
                     {
                         ImVec2 cp = ImGui::GetItemRectMin();
                         UIWidgets::DrawIcon(UIWidgets::IconType::Mesh, ImVec2(cp.x + 4, cp.y + 1), 12, 0xFFBBBBBB);
                     }
                     ImGui::SameLine(); UIWidgets::HelpMarker("Select the 3D models to be used in this layer. If multiple models are added, they will be distributed randomly.");
                     if (UIWidgets::SecondaryButton("+ Add Selected", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 0))) {
                         if (ctx.selection.hasSelection()) {
                             std::string n = ctx.selection.selected.name;
                             std::vector<std::shared_ptr<Triangle>> tris;
                             for (auto& obj : ctx.scene.world.objects) {
                                 auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                 if (tri && tri->getNodeName() == n) tris.push_back(tri);
                             }
                             if (!tris.empty()) {
                                 group.sources.emplace_back(n, tris);
                             } else {
                                  SCENE_LOG_WARN("Selection invalid.");
                             }
                        }
                     }
                     
                     ImGui::SameLine();
                     if (UIWidgets::SecondaryButton("Pick from List", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 0))) {
                         ImGui::OpenPopup("ObjPicker");
                     }
                     
                     if (ImGui::BeginPopup("ObjPicker")) {
                         static char filter[64] = "";
                         ImGui::SetNextItemWidth(160.0f);
                         ImGui::InputText("Filter", filter, 64);
                         ImGui::Separator();
                         
                         ImGui::BeginChild("ObjList", ImVec2(300, 250));
                         // Sort or just list? List is fine.
                         // Use set to dedup names if multiple objects share name?
                         // Scene objects usually unique per mesh instance?
                         // We want to select by "Node Name".
                         std::set<std::string> listed_names;
                         
                         // OPTIMIZATION: Use mesh_cache instead of iterating all objects
                         // mesh_cache keys are unique node names (std::map<std::string, ...>)
                         
                         if (mesh_cache.empty()) {
                             rebuildMeshCache(ctx.scene.world.objects);
                         }

                         for (const auto& [name, tris_list] : mesh_cache) {
                             if (name.empty() || name.find("_inst_") == 0) continue;
                             if (filter[0] != '\0' && name.find(filter) == std::string::npos) continue;
                             
                             if (ImGui::Selectable(name.c_str())) {
                                 // Add all triangles associated with this name
                                 std::vector<std::shared_ptr<Triangle>> source_tris;
                                 source_tris.reserve(tris_list.size());
                                 
                                 for (const auto& pair : tris_list) {
                                     source_tris.push_back(pair.second);
                                 }
                                 
                                 if (!source_tris.empty()) {
                                     group.sources.emplace_back(name, source_tris);
                                 }
                                 ImGui::CloseCurrentPopup();
                             }
                         }
                         ImGui::EndChild();
                         ImGui::EndPopup();
                     }
                     
                     // List Sources
                     for (int s_i=0; s_i<group.sources.size(); s_i++) {
                         ImGui::PushID(s_i);
                         auto& src = group.sources[s_i];
                         ImGui::Text("%s (W: %.1f)", src.name.c_str(), src.weight);
                         ImGui::SameLine();
                         if (ImGui::SmallButton("Edit")) ImGui::OpenPopup("SrcEdit");
                         
                         if (ImGui::BeginPopup("SrcEdit")) {
                             ImGui::PushItemWidth(160.0f);
                             ImGui::DragFloat("Weight", &src.weight, 0.1f);
                             ImGui::DragFloatRange2("Scale", &src.settings.scale_min, &src.settings.scale_max, 0.01f, 0.001f, 1000.0f);
                             ImGui::DragFloatRange2("Y-Off", &src.settings.y_offset_min, &src.settings.y_offset_max, 0.01f);
                             ImGui::PopItemWidth();
                             ImGui::EndPopup();
                         }
                         
                         ImGui::SameLine();
                         if (ImGui::SmallButton("X")) {
                             group.sources.erase(group.sources.begin() + s_i);
                             s_i--;
                         }
                         ImGui::PopID();
                     }
                     
                     ImGui::Separator();
                     ImGui::Separator();
                     // ACTIONS
                     ImGui::Text("   Scatter Settings (Global):");
                     {
                         ImVec2 cp = ImGui::GetItemRectMin();
                         UIWidgets::DrawIcon(UIWidgets::IconType::World, ImVec2(cp.x + 4, cp.y + 1), 12, 0xFFBBBBBB);
                     }
                     ImGui::PushItemWidth(160.0f);
                     ImGui::DragInt("Target Count", &group.brush_settings.target_count, 100, 1, 10000000);
                     if (ImGui::IsItemHovered()) ImGui::SetTooltip("Total number of instances to spread across the whole terrain.");
                     ImGui::InputInt("Seed (Randomness)", &group.brush_settings.seed);
                     ImGui::DragFloat("Min. Distance", &group.brush_settings.min_distance, 0.1f, 0.0f, 50.0f);
                     if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum distance between instances (m). Prevents overlap.");
                     ImGui::PopItemWidth();
                     
                     // Helper helper for brush state
                     bool is_active_group = (foliage_brush.active_group_id == group.id);
                     bool is_painting = foliage_brush.enabled && is_active_group && foliage_brush.mode == 0;
                     bool is_erasing = foliage_brush.enabled && is_active_group && foliage_brush.mode == 1;

                     // Paint Button
                     if (is_painting) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                     if (ImGui::Button("Paint (Add)", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 30))) {
                         if (is_painting) {
                             foliage_brush.enabled = false;
                             foliage_brush.active_group_id = -1;
                         } else {
                             foliage_brush.enabled = true;
                             foliage_brush.active_group_id = group.id;
                             foliage_brush.mode = 0; // ADD
                         }
                     }
                     if (is_painting) ImGui::PopStyleColor();
                     
                     ImGui::SameLine();
                     
                     // Erase Button
                     if (is_erasing) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
                     if (ImGui::Button("Erase (Remove)", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 30))) {
                         if (is_erasing) {
                             foliage_brush.enabled = false;
                             foliage_brush.active_group_id = -1;
                         } else {
                             foliage_brush.enabled = true;
                             foliage_brush.active_group_id = group.id;
                             foliage_brush.mode = 1; // REMOVE
                         }
                     }
                     if (is_erasing) ImGui::PopStyleColor();

                     // Show Brush Settings if active
                     if (is_active_group && foliage_brush.enabled) {
                        ImGui::SameLine();
                        ImGui::TextDisabled("(Active)");
                        ImGui::PushItemWidth(160.0f);
                        ImGui::DragFloat("radius##br", &foliage_brush.radius, 0.1f, 0.1f, 100.0f, "%.1f m");
                        if (is_painting) {
                            ImGui::DragInt("density##br", &foliage_brush.density, 1, 1, 20);
                        }
                        ImGui::PopItemWidth();
                        ImGui::Checkbox("Lazy Update", &foliage_brush.lazy_update);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Update scene only on mouse release (Better performance for large terrains)");
                     }
                     
                     ImGui::Separator();

                         UIWidgets::ColoredHeader("      Placement Rules", ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
                         {
                             ImVec2 cp = ImGui::GetItemRectMin();
                             UIWidgets::DrawIcon(UIWidgets::IconType::System, ImVec2(cp.x + 12, cp.y + 2), 14, 0xFFBBBBBB);
                         }
                        ImGui::SameLine(); UIWidgets::HelpMarker("Configures the growth logic of the foliage. Plants can be filtered by altitude, slope, and terrain shape.");
                        
                        ImGui::PushItemWidth(160.0f);
                        // Height Range
                        ImGui::DragFloatRange2("Altitude Range", &group.brush_settings.height_min, &group.brush_settings.height_max, 1.0f, -1000.0f, 5000.0f, "Min: %.1f m", "Max: %.1f m");
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Restricts placement within a specific sea-level altitude range.");
                        
                        // Slope Limit
                        ImGui::DragFloat("Slope Limit", &group.brush_settings.slope_max, 1.0f, 0.0f, 90.0f, "%.1f deg");
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Maximum steepness for placement.\n0 = Only flats\n90 = Everywhere including vertical cliffs.");
                        
                        // Slope Direction (Exposure/Aspect)
                        ImGui::TextDisabled("Aspect Filter (Biology):");
                        ImGui::SameLine(); UIWidgets::HelpMarker("Simulates plant exposure to sunlight. You can place more vegetation on North (shaded) or South (sunny) facing slopes.");

                        if (SceneUI::DrawSmartFloat("sd_angle", "Target Angle", &group.brush_settings.slope_direction_angle, 0.0f, 360.0f, "%.0f deg", false, nullptr, 16)) {
                            while(group.brush_settings.slope_direction_angle < 0) group.brush_settings.slope_direction_angle += 360;
                            while(group.brush_settings.slope_direction_angle >= 360) group.brush_settings.slope_direction_angle -= 360;
                        }
                        
                        // Small Visual Compass Indicator
                        ImGui::SameLine();
                        ImDrawList* drawList = ImGui::GetWindowDrawList();
                        ImVec2 cp = ImGui::GetCursorScreenPos();
                        float c_size = 10.0f;
                        cp.x += c_size + 5;
                        cp.y += 10.0f;
                        drawList->AddCircleFilled(cp, c_size + 2, IM_COL32(40, 40, 40, 255));
                        drawList->AddCircle(cp, c_size, IM_COL32(200, 200, 200, 255), 16);
                        
                        float s_rad = group.brush_settings.slope_direction_angle * 0.0174533f;
                        ImVec2 s_needle(cp.x + sinf(s_rad) * c_size, cp.y - cosf(s_rad) * c_size);
                        drawList->AddLine(cp, s_needle, IM_COL32(255, 80, 80, 255), 2.0f);
                        drawList->AddText(ImVec2(cp.x - 3, cp.y - c_size - 14), IM_COL32(255, 255, 255, 150), "N");
                        ImGui::Dummy(ImVec2(c_size * 2 + 20, 1)); 
                        
                        SceneUI::DrawSmartFloat("sd_inf", "Influence", &group.brush_settings.slope_direction_influence, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Determines how strictly the target angle is followed.\n0 = All sides\n1 = Strict directional placement.");
                        ImGui::PopItemWidth();
                        
                        // Curvature (Flow/Ridge detection)
                        ImGui::TextDisabled("Curvature (Shape) Filter:");
                        ImGui::SameLine(); UIWidgets::HelpMarker("Identifies terrain features like Ridges (peaks), Gullies (valleys), or Flats. Essential for realistic ecosystem distribution.");
                        
                        ImGui::SetNextItemWidth(160.0f);
                        ImGui::DragInt("Detail Scale", &group.brush_settings.curvature_step, 0.1f, 1, 30, "%d px");
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Sampling distance for feature detection.\nSmall = Sharp ridges/cracks\nLarge = Broad mountain caps / river beds");
                        
                        // Line 1: Ridges
                        ImGui::Checkbox("Ridges", &group.brush_settings.allow_ridges);
                        if (group.brush_settings.allow_ridges) {
                            ImGui::SameLine();
                            UIWidgets::HelpMarker("Convex areas like mountain crests and hill tops.");
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(80);
                            SceneUI::DrawSmartFloat("##RThresh", "", &group.brush_settings.curvature_min, -200.0f, -0.01f, "< %.2f", false, nullptr, 0); 
                        }

                        // Line 2: Gullies
                        ImGui::Checkbox("Gullies", &group.brush_settings.allow_gullies);
                        if (group.brush_settings.allow_gullies) {
                            ImGui::SameLine();
                            UIWidgets::HelpMarker("Concave areas like valleys, river beds and depressions.");
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(80);
                            SceneUI::DrawSmartFloat("##GThresh", "", &group.brush_settings.curvature_max, 0.01f, 200.0f, "> %.2f", false, nullptr, 0);
                        }

                        // Line 3: Flats (Middle)
                        ImGui::Checkbox("Flats", &group.brush_settings.allow_flats);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Stable areas that are neither crests nor troughs.");
                        
                        // Splat Map Channel
                        const char* channels[] = { "None", "Red (Grass/Flat)", "Green (Slope/Rock)", "Blue (Height/Snow)", "Alpha (Flow Map)" };
                        int current_idx = group.brush_settings.splat_map_channel + 1; // Map -1 to 0, 0 to 1, etc.
                        ImGui::PushItemWidth(160.0f);
                        if (ImGui::Combo("Mask Channel", &current_idx, channels, 5)) {
                            group.brush_settings.splat_map_channel = current_idx - 1;
                        }
                        ImGui::SameLine(); UIWidgets::HelpMarker("Restricts placement based on terrain paint data.");
                        ImGui::PopItemWidth();

                        if (group.brush_settings.splat_map_channel != -1) {
                             if (!t->splatMap || !t->splatMap->is_loaded()) {
                                 ImGui::TextColored(ImVec4(1,0,0,1), "No Splat Map Loaded!");
                             }
                        }
                        
                        // Exclusion Mask UI
                        int ex_idx = group.brush_settings.exclusion_channel + 1;
                        ImGui::PushItemWidth(160.0f);
                        if (ImGui::Combo("Exclude Channel", &ex_idx, channels, 5)) {
                            group.brush_settings.exclusion_channel = ex_idx - 1;
                        }
                        ImGui::SameLine(); UIWidgets::HelpMarker("Placement is prevented where this channel is painted (e.g., roads).");
                        ImGui::PopItemWidth();
                        if (group.brush_settings.exclusion_channel != -1) {
                             ImGui::SameLine();
                             ImGui::SetNextItemWidth(80);
                             ImGui::DragFloat("Threshold##Ex", &group.brush_settings.exclusion_threshold, 0.05f, 0.0f, 1.0f);
                        }

                        if (UIWidgets::PrimaryButton("Scatter Procedurally", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 30))) {
                        // Clear existing instances to ensure fresh generation (Replace vs Append)
                        group.clearInstances();

                        int count = group.brush_settings.target_count;
                        std::mt19937 rng(group.brush_settings.seed);
                        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                        int spawned = 0;
                        
                        // Overlap Check (Simple Grid)
                        float min_dist_sq = group.brush_settings.min_distance * group.brush_settings.min_distance;
                        bool check_overlap = group.brush_settings.min_distance > 0.01f;
                        std::map<std::pair<int, int>, std::vector<Vec3>> grid;
                        float cell_size = group.brush_settings.min_distance > 0.1f ? group.brush_settings.min_distance : 1.0f;
                        
                        // New: Safety Counter
                        int attempts = 0;
                        int max_attempts = count * 100;

                        // Pre-calculate slope factors same as AutoMask
                        float scale = t->heightmap.scale_xz;
                        float hmCellSizeX = scale / (float)(std::max(1, t->heightmap.width - 1));
                        float hmCellSizeZ = scale / (float)(std::max(1, t->heightmap.height - 1));

                        while (spawned < count && attempts < max_attempts) {
                            attempts++;
                            
                            float r1 = dist(rng);
                            float r2 = dist(rng);
                            
                            // Splat Map Mask Check (Importance Sampling Rejection)
                            if (group.brush_settings.splat_map_channel >= 0 && t->splatMap && t->splatMap->is_loaded() && !t->splatMap->pixels.empty()) {
                                float u = r1;
                                float v = r2;
                                // Sample splat map
                                Vec3 maskVal = t->splatMap->get_color(u, 1.0f - v); // UV flip?
                                float val = 0.0f;
                                if (group.brush_settings.splat_map_channel == 0) val = maskVal.x;
                                else if (group.brush_settings.splat_map_channel == 1) val = maskVal.y;
                                else if (group.brush_settings.splat_map_channel == 2) val = maskVal.z;
                                else if (group.brush_settings.splat_map_channel == 3) {
                                    // Manual Alpha Access
                                    int texW = t->splatMap->width;
                                    int texH = t->splatMap->height;
                                    int tx = (int)(u * texW) % texW;
                                    int ty = (int)((1.0f - v) * texH) % texH;
                                    int pIdx = ty * texW + tx;
                                    if(pIdx < t->splatMap->pixels.size()) {
                                        val = t->splatMap->pixels[pIdx].a / 255.0f;
                                    }
                                }
                                
                                if (val < 0.2f) continue; // Skip if mask is black
                                if (dist(rng) > val) continue; // Probabilistic skip
                            }

                            // Exclusion Check
                            if (group.brush_settings.exclusion_channel >= 0 && t->splatMap && t->splatMap->is_loaded() && !t->splatMap->pixels.empty()) {
                                float u = r1;
                                float v = r2;
                                int texW = t->splatMap->width;
                                int texH = t->splatMap->height;
                                int tx = (int)(u * texW) % texW;
                                int ty = (int)((1.0f - v) * texH) % texH;
                                int pIdx = ty * texW + tx;
                                
                                float exVal = 0.0f;
                                if (pIdx < t->splatMap->pixels.size()) {
                                     const auto& p = t->splatMap->pixels[pIdx];
                                     if(group.brush_settings.exclusion_channel == 0) exVal = p.r / 255.0f;
                                     else if(group.brush_settings.exclusion_channel == 1) exVal = p.g / 255.0f;
                                     else if(group.brush_settings.exclusion_channel == 2) exVal = p.b / 255.0f;
                                     else if(group.brush_settings.exclusion_channel == 3) exVal = p.a / 255.0f;
                                }
                                
                                if (exVal > group.brush_settings.exclusion_threshold) continue; // Excluded!
                            }
                            
                            // 2. Coordinate Mapping & Height Sampling
                            float tx = r1 * t->heightmap.scale_xz;
                            float tz = r2 * t->heightmap.scale_xz;
                            
                            // Bilinear Height Sampling
                            float grid_x = r1 * (t->heightmap.width - 1);
                            float grid_z = r2 * (t->heightmap.height - 1);
                            int x0 = (int)grid_x;
                            int z0 = (int)grid_z;
                            int x1 = std::min(x0 + 1, t->heightmap.width - 1);
                            int z1 = std::min(z0 + 1, t->heightmap.height - 1);
                            float fx = grid_x - x0;
                            float fz = grid_z - z0;
                            
                            float h00 = t->heightmap.getHeight(x0, z0);
                            float h10 = t->heightmap.getHeight(x1, z0);
                            float h01 = t->heightmap.getHeight(x0, z1);
                            float h11 = t->heightmap.getHeight(x1, z1);
                            
                            float h_interp = (h00 * (1.0f - fx) + h10 * fx) * (1.0f - fz) + 
                                             (h01 * (1.0f - fx) + h11 * fx) * fz;
                                             
                            // 3. Height Check (Local to Terrain)
                            // Fix: Use local height so scattering works correctly even if terrain is moved
                            float checkHeight = h_interp; 
                            
                            if (checkHeight < group.brush_settings.height_min || checkHeight > group.brush_settings.height_max) {
                                continue;
                            }

                            // 4. Slope Check (vs Max Slope)
                            // Approximate slope at this location using same neighbors
                            // Need 4 neighbors from grid.
                            // We are inside grid quad (x0, z0)
                            // Slope ~ max derivative.
                            float dX = (h10 - h00) / hmCellSizeX; // Slope across X in this cell
                            float dZ = (h01 - h00) / hmCellSizeZ; // Slope across Z in this cell
                            // This is a rough approximation for random point.
                            // Better: Interpolated normals?
                            // Let's use simple finite diff at nearest integer for speed/robustness match with AutoMask
                            // Clamp sx, sz to valid range for step size
                            int step = group.brush_settings.curvature_step;
                            if (step < 1) step = 1;
                            
                            // Ensure terrain is large enough for step
                            if (t->heightmap.width <= step * 2 || t->heightmap.height <= step * 2) continue;

                            int sx = (int)(grid_x + 0.5f);
                            int sz = (int)(grid_z + 0.5f);
                            sx = std::clamp(sx, step, t->heightmap.width - 1 - step);
                            sz = std::clamp(sz, step, t->heightmap.height - 1 - step);
                            
                             // Sample neighbors with step (Use normalized data for scale-invariant curvature)
                             // Laplacian = (L + R + U + D) - 4*C
                             float hl = t->heightmap.data[sz * t->heightmap.width + (sx - step)];
                             float hr = t->heightmap.data[sz * t->heightmap.width + (sx + step)];
                             float hu = t->heightmap.data[(sz - step) * t->heightmap.width + sx];
                             float hd = t->heightmap.data[(sz + step) * t->heightmap.width + sx];
                             float h_center_norm = t->heightmap.data[sz * t->heightmap.width + sx];

                             float laplacian_norm = (hl + hr + hu + hd) - 4.0f * h_center_norm;
                             // Sensitivity Adjustment: Dividing by step size ensures that macro features 
                             // don't produce extreme Laplacian values compared to micro features.
                             float laplacian = (laplacian_norm / (float)(step * step)) * 1000.0f; 
                             
                             // 4. Slope calculation still needs world heights for degrees
                             float hl_world = hl * t->heightmap.scale_y;
                             float hr_world = hr * t->heightmap.scale_y;
                             float hu_world = hu * t->heightmap.scale_y;
                             float hd_world = hd * t->heightmap.scale_y;
                            
                            float slopeRunX = 2.0f * hmCellSizeX * step; // Adjusted for step
                            float slopeRunZ = 2.0f * hmCellSizeZ * step; // Adjusted for step
                            
                             float dX_raw = (hr_world - hl_world) / slopeRunX;
                             float dZ_raw = (hd_world - hu_world) / slopeRunZ;
                            
                            float dX_central = fabsf(dX_raw);
                            float dZ_central = fabsf(dZ_raw);
                            float slopeRunComp = sqrtf(dX_central*dX_central + dZ_central*dZ_central);
                            float slopeDeg = atan(slopeRunComp) * 57.2958f;
                            
                            if (slopeDeg > group.brush_settings.slope_max) {
                                continue;
                            }

                            // 4.5. Slope Direction (Exposure/Aspect) Check
                            if (group.brush_settings.slope_direction_influence > 0.01f && slopeDeg > 2.0f) {
                                // Direction vector in XZ plane (DOWNHILL): (-dX_raw, -dZ_raw)
                                // atan2(x, z) gives angle where 0 is North/Z+
                                float lookRad = atan2f(-dX_raw, -dZ_raw); 
                                float lookDeg = lookRad * 57.2958f;
                                if (lookDeg < 0) lookDeg += 360.0f;
                                
                                float targetDeg = group.brush_settings.slope_direction_angle;
                                float diff = fabsf(lookDeg - targetDeg);
                                if (diff > 180.0f) diff = 360.0f - diff;
                                
                                // Exponential/Cosine falloff for natural transition
                                float dirWeight = std::max(0.0f, cosf(diff * 0.0174533f)); 
                                // Mix based on influence
                                float finalDirProb = (1.0f - group.brush_settings.slope_direction_influence) + (group.brush_settings.slope_direction_influence * dirWeight);
                                
                                if (dist(rng) > finalDirProb) continue;
                            }
                            
                             // Laplacian already calculated above using normalized data
                            
                            bool is_ridge = laplacian < group.brush_settings.curvature_min;
                            bool is_gully = laplacian > group.brush_settings.curvature_max;
                            bool is_flat = !is_ridge && !is_gully;

                            if (is_ridge && !group.brush_settings.allow_ridges) continue;
                            if (is_gully && !group.brush_settings.allow_gullies) continue;
                            if (is_flat && !group.brush_settings.allow_flats) continue;

                            // Local Position (Y is now precise interpolated height)
                            Vec3 localPos(tx, h_interp, tz);
                            Vec3 worldPos = localPos;
                            
                            // Apply Full World Transform (Fix for Rotation/Scale issues)
                            if (t->transform) {
                                t->transform->updateFinal(); // Ensure matrix is current
                                worldPos = t->transform->final.transform_point(localPos); 
                            }


                            if (check_overlap) {
                                  // Grid check uses World or Local? 
                                  // Using Local aligned grid is safer for checking relative spacing
                                  // But if we want consistent world spacing... use World.
                                  // Let's use WorldPos for the grid to be safe
                                  int cx = (int)std::floor(worldPos.x / cell_size);
                                  int cz = (int)std::floor(worldPos.z / cell_size);
                                  bool collision = false;
                                  
                                  // Check 3x3
                                  for (int dx = -1; dx <= 1; dx++) {
                                      for (int dz = -1; dz <= 1; dz++) {
                                           auto it = grid.find({cx+dx, cz+dz});
                                           if (it != grid.end()) {
                                               for (const auto& p : it->second) {
                                                   if ((p - worldPos).length_squared() < min_dist_sq) {
                                                       collision = true; break;
                                                   }
                                               }
                                           }
                                           if (collision) break;
                                      }
                                      if (collision) break;
                                  }
                                  if (collision) continue;
                                  
                                  grid[{cx, cz}].push_back(worldPos);
                             }
                            
                            // Generate Instance (World Space)
                            InstanceTransform inst = group.generateRandomTransform(worldPos, Vec3(0, 1, 0));
                            group.addInstance(inst);
                            spawned++;
                        }
                        SceneUI::syncInstancesToScene(ctx, group, false);
                        g_optix_rebuild_pending = true;
                        ctx.renderer.resetCPUAccumulation();
                     }
                     ImGui::SameLine();
                         if (UIWidgets::DangerButton("Clear Instances", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 0))) {
                             group.clearInstances();
                             SceneUI::syncInstancesToScene(ctx, group, true);
                             g_optix_rebuild_pending = true;
                             ctx.renderer.resetCPUAccumulation();
                         }
                     ImGui::SameLine();
                     if (ImGui::Button("Delete Layer", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 0))) {
                         fol_remove_id = group.id;
                     }
                    
                    ImGui::Separator();
                    if (ImGui::TreeNode("Wind Animation")) {
                        bool enabled = group.wind_settings.enabled;
                        if (ImGui::Checkbox("Enable Wind", &enabled)) {
                            group.wind_settings.enabled = enabled;
                            if (!enabled) {
                                // Restore Rest Pose to remove bent state
                                if (!group.initial_instances.empty() && group.initial_instances.size() == group.instances.size()) {
                                    group.instances = group.initial_instances;
                                }
                            }
                            group.gpu_dirty = true; 
                            
                            // Force Update
                            ctx.renderer.resetCPUAccumulation();
                            if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                            g_optix_rebuild_pending = true; // Must ensure this is handled in Main loop!
                        }
                        
                        if (group.wind_settings.enabled) {
                            bool changed = false;
                            changed |= ImGui::DragFloat("Speed", &group.wind_settings.speed, 0.05f, 0.0f, 10.0f);
                            changed |= ImGui::DragFloat("Strength", &group.wind_settings.strength, 0.1f, 0.0f, 45.0f, "%.1f deg");
                            changed |= ImGui::DragFloat("Turbulence", &group.wind_settings.turbulence, 0.05f, 0.0f, 5.0f);
                            changed |= ImGui::DragFloat("Wave Size", &group.wind_settings.wave_size, 1.0f, 1.0f, 500.0f);
                            
                            // Direction Logic
                            float current_angle = atan2(group.wind_settings.direction.z, group.wind_settings.direction.x) * 57.2958f;
                            if (ImGui::DragFloat("Direction", &current_angle, 1.0f, -180.0f, 180.0f, "%.0f deg")) {
                                float rad = current_angle / 57.2958f;
                                group.wind_settings.direction = Vec3(cos(rad), 0, sin(rad));
                                changed = true;
                            }
                            
                            if (changed) {
                                // Real-time preview requires reset
                                ctx.renderer.resetCPUAccumulation();
                                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                                
                                // Since logic is time-based, just setting dirty might not be enough if time is paused.
                                // But Renderer::updateWind reads these values every frame.
                            }
                        }
                        ImGui::TreePop(); 
                    }
                    
                    ImGui::TreePop(); 
                }
                ImGui::PopID();
            }
            
            if (fol_remove_id != -1) {
                 // Correct logic: use deleteGroup(id)
                 InstanceGroup* g = im.getGroup(fol_remove_id);
                 if (g) {
                    // Safety: If deleting the active brush group, disable brush first
                    if (foliage_brush.active_group_id == fol_remove_id) {
                        foliage_brush.active_group_id = -1;
                        foliage_brush.enabled = false;
                    }

                    // CRITICAL: Stop rendering to prevent crash when modifying object list
                    ctx.renderer.stopRendering();
                    // Small sleep to ensure render threads exit loop
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));

                    SceneUI::syncInstancesToScene(ctx, *g, true); // Clear from scene
                    im.deleteGroup(fol_remove_id);
                    
                    ctx.renderer.resumeRendering();

                }
            }
            ImGui::Unindent();
            UIWidgets::EndSection();
        }

            // 5. PROCEDURAL TOOLS & MASKS
            if (UIWidgets::BeginSection("Procedural Generators", ImVec4(0.8f, 0.4f, 1.0f, 1.0f), true)) {
                ImGui::Indent();

                UIWidgets::ColoredHeader("Auto-Mask Generation", ImVec4(0.7f, 0.9f, 0.8f, 1.0f));
                ImGui::PushItemWidth(160.0f);
                SceneUI::DrawSmartFloat("mhmin", "Height Start", &t->am_height_min, 0.0f, 500.0f, "%.1f", false, nullptr, 12);
                SceneUI::DrawSmartFloat("mhmax", "Height End", &t->am_height_max, 0.0f, 500.0f, "%.1f", false, nullptr, 12);
                SceneUI::DrawSmartFloat("mslope", "Slope Steep", &t->am_slope, 1.0f, 20.0f, "%.1f", false, nullptr, 12);
                SceneUI::DrawSmartFloat("mflow", "Flow Thresh", &t->am_flow_threshold, 1.0f, 500.0f, "%.0f", false, nullptr, 12);
                ImGui::PopItemWidth();

                if (UIWidgets::PrimaryButton("Generate Mask", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
                    TerrainManager::getInstance().autoMask(t, 0.0f, 0.0f, t->am_height_min, t->am_height_max, t->am_slope);
                    ctx.renderer.resetCPUAccumulation();
                    if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                    if (ctx.optix_gpu_ptr) ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                    SCENE_LOG_INFO("Auto-mask generated for: " + t->name);
                }

                if (ImGui::Button("Import Splat Map", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
                    std::string path = SceneUI::openFileDialogW(L"Image Files\0*.png;*.jpg;*.jpeg;*.bmp\0All Files\0*.*\0");
                    if (!path.empty()) {
                        TerrainManager::getInstance().importSplatMap(t, path);
                        ctx.renderer.resetCPUAccumulation();
                        if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                        if (ctx.optix_gpu_ptr) ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                    }
                }

                ImGui::Spacing();
                UIWidgets::ColoredHeader("Flow Baking & Export", ImVec4(0.8f, 0.7f, 0.6f, 1.0f));
                ImGui::PushItemWidth(160.0f);
                static float bake_flow_threshold = 25.0f;
                ImGui::DragFloat("Flow Thresh", &bake_flow_threshold, 1.0f, 1.0f, 500.0f, "%.0f");
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum flow accumulation to be written to Alpha.\nIncrease this to ignore flat areas/rain and keep only rivers.");
                ImGui::PopItemWidth();

                if (UIWidgets::PrimaryButton("Bake Flow to Alpha", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.6f, 30))) {
                    if (t->splatMap && t->splatMap->is_loaded() && !t->flowMap.empty()) {
                        int w = t->heightmap.width;
                        int h = t->heightmap.height;
                        int sw = t->splatMap->width;
                        int sh = t->splatMap->height;

                        for (int y = 0; y < sh; y++) {
                            for (int x = 0; x < sw; x++) {
                                float u = (float)x / (float)(sw > 1 ? sw - 1 : 1);
                                float v = (float)y / (float)(sh > 1 ? sh - 1 : 1);
                                int fx = std::clamp((int)(u * (w-1) + 0.5f), 0, w-1);
                                int fy = std::clamp((int)(v * (h-1) + 0.5f), 0, h-1);
                                
                                float flowVal = t->flowMap[fy * w + fx];
                                float flowNorm = fmaxf(0.0f, flowVal - t->am_flow_threshold);
                                float final_A = 1.0f - expf(-flowNorm * 0.4f);
                                final_A = std::clamp(final_A, 0.0f, 0.98f);
                                
                                t->splatMap->pixels[y * sw + x].a = (uint8_t)(final_A * 255.0f);
                            }
                        }
                        t->splatMap->updateGPU();
                        ctx.renderer.resetCPUAccumulation();
                        if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                        SCENE_LOG_INFO("Flow baked to Splat Alpha using engine-side FlowMap.");
                    } else {
                        SCENE_LOG_WARN("Please run Erosion first to generate a Flow Map.");
                    }
                }

                ImGui::SameLine();
                if (ImGui::Button("Export Splat Map", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.6f, 30))) {
                    if (t->splatMap && !t->splatMap->pixels.empty()) {
                        std::string path = SceneUI::saveFileDialogW(L"PNG Files\0*.png\0", L"png");
                        if (!path.empty()) {
                            TerrainManager::getInstance().exportSplatMap(t, path);
                            SCENE_LOG_INFO("Splat map exported to: " + path);
                        }
                    }
                }

                ImGui::Unindent();
                UIWidgets::EndSection();
            
            // ===============================================================
            // ROCK HARDNESS (for realistic erosion)
            // ===============================================================
            UIWidgets::ColoredHeader("      Rock Hardness", ImVec4(0.7f, 0.5f, 0.3f, 1.0f));
            
            static float defaultHardness = 0.3f;
            ImGui::PushItemWidth(160.0f);
            if (SceneUI::DrawSmartFloat("hard", "Def Hardness", &defaultHardness, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) {}
            ImGui::PopItemWidth();
            UIWidgets::HelpMarker("0 = Soft (sand/soil), 1 = Hard (bedrock)");
            
            bool hasHardness = !t->hardnessMap.empty();
            if (hasHardness) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Hardness Map: Active (%dx%d)", 
                    t->heightmap.width, t->heightmap.height);
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Hardness Map: Not initialized");
            }
            
            if (ImGui::Button("Init Hardness Map", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 30))) {
                TerrainManager::getInstance().initHardnessMap(t, defaultHardness);
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
            }
            ImGui::SameLine();
            if (ImGui::Button("Auto-Generate", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 30))) {
                TerrainManager::getInstance().autoGenerateHardness(t, 0.7f, 0.15f);
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
            }
            ImGui::Unindent(12.0f);
            UIWidgets::EndSection();
        }
 
            // 6. TIMELINE & EXPORT
            if (UIWidgets::BeginSection("Timeline & Export", ImVec4(0.7f, 0.4f, 0.8f, 1.0f), true)) {
                ImGui::Indent();
                
                // Timeline Integration
                UIWidgets::ColoredHeader("Timeline & Keyframes", ImVec4(0.7f, 0.4f, 0.8f, 1.0f));
                
                int current_frame = ctx.scene.timeline.current_frame;
                ImGui::Text("Current Frame: %d", current_frame);
                
                std::string trackName = t->name.empty() ? "Terrain" : t->name;
                bool hasTrack = ctx.scene.timeline.tracks.find(trackName) != ctx.scene.timeline.tracks.end();
                
                if (ImGui::Button("Capture State Keyframe", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
                    // Get or create track
                    ObjectAnimationTrack& track = ctx.scene.timeline.tracks[trackName]; // Creates if not exists
                    track.object_name = trackName;
                    track.object_index = -1; 
                    
                    TerrainManager::getInstance().captureKeyframeToTrack(t, track, current_frame);
                }
                
                if (hasTrack) {
                    ImGui::SameLine();
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Track Exists");
                    
                    // Show keyframe count for this track
                    int kf_count = (int)ctx.scene.timeline.tracks[trackName].keyframes.size();
                    ImGui::SameLine();
                    ImGui::TextDisabled("(%d keys)", kf_count);
                } else {
                    ImGui::SameLine();
                    ImGui::TextDisabled("(No Track)");
                }

                // Heightmap Export
                ImGui::Separator();
                if (ImGui::Button("Export Heightmap (16-bit RAW)", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
                    std::string path = SceneUI::saveFileDialogW(L"RAW Files\0*.raw\0", L"raw");
                    if (!path.empty()) {
                        TerrainManager::getInstance().exportHeightmap(t, path);
                    }
                }
                ImGui::Unindent();
                UIWidgets::EndSection();
            } 
    } 

    // -----------------------------------------------------------------------------
    // 8. SCULPTING & PAINTING
    // -----------------------------------------------------------------------------
    if (terrain_brush.active_terrain_id != -1) {
        if (UIWidgets::BeginSection("      Sculpting & Painting Tools (EXPERIMENTAL)", ImVec4(1.0f, 0.7f, 0.4f, 1.0f), true)) {
            ImVec2 hp = ImGui::GetItemRectMin();
            UIWidgets::DrawIcon(UIWidgets::IconType::Magnet, ImVec2(hp.x + 8, hp.y + 4), 16, 0xFFBBBBBB);
            ImGui::Indent(12.0f);
                    ImGui::SameLine(); UIWidgets::HelpMarker("Use these tools to manually shape the terrain geometry or paint texture layers and physical properties.");

                    ImGui::Checkbox("Enable Brush", &terrain_brush.enabled);
                    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Activates the mouse brush in the viewport.");

                    if (terrain_brush.enabled) {
                        // Active Terrain Selector
                        auto& terrains = TerrainManager::getInstance().getTerrains();
                        if (terrains.empty()) {
                            ImGui::TextColored(ImVec4(1, 0, 0, 1), "No active terrain found.");
                        }
                        else {
                            std::string current_name = "None";
                            if (terrain_brush.active_terrain_id != -1) {
                                auto* t = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
                                if (t) current_name = t->name + " (ID: " + std::to_string(t->id) + ")";
                            }

                            ImGui::PushItemWidth(160.0f);
                            if (ImGui::BeginCombo("Target Terrain", current_name.c_str())) {
                                for (auto& t : terrains) {
                                    bool is_selected = (t.id == terrain_brush.active_terrain_id);
                                    std::string label = "> " + t.name + " (ID: " + std::to_string(t.id) + ")";
                                    if (ImGui::Selectable(label.c_str(), is_selected)) {
                                        terrain_brush.active_terrain_id = t.id;
                                    }
                                    if (is_selected) ImGui::SetItemDefaultFocus();
                                }
                                ImGui::EndCombo();
                            }
                            ImGui::PopItemWidth();

                            ImGui::Separator();
                            ImGui::TextDisabled("Brush Mode:");

                             // Brush Modes with Symbols
                            ImGui::RadioButton("Raise", &terrain_brush.mode, 0); ImGui::SameLine();
                            // If shift is held, small icon-like letters
                            ImGui::RadioButton("Lower", &terrain_brush.mode, 1); ImGui::SameLine();
                            ImGui::RadioButton("Flatten", &terrain_brush.mode, 2); ImGui::SameLine();
                            ImGui::RadioButton("Smooth", &terrain_brush.mode, 3);
                            
                            ImGui::RadioButton("Stamp", &terrain_brush.mode, 4); ImGui::SameLine();
                            ImGui::RadioButton("Paint Layers", &terrain_brush.mode, 5);
                            
                            ImGui::RadioButton("Paint Hard", &terrain_brush.mode, 6); ImGui::SameLine();
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Increases erosion resistance (Rock).");
                            
                            ImGui::RadioButton("Paint Soft", &terrain_brush.mode, 7);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Decreases erosion resistance (Soil/Sand).");

                            ImGui::Spacing();

                            if (terrain_brush.mode == 2) { // Flatten Settings
                                ImGui::Indent();
                                ImGui::Checkbox("Use Fixed Height", &terrain_brush.use_fixed_height);
                                if (terrain_brush.use_fixed_height) {
                                    ImGui::PushItemWidth(160.0f);
                                    ImGui::DragFloat("Altitude", &terrain_brush.flatten_target, 0.1f, -1000.0f, 5000.0f, "%.1f m");
                                    ImGui::PopItemWidth();
                                } else {
                                    ImGui::TextDisabled("(Sampled from initial click)");
                                }
                                ImGui::Unindent();
                            }
                            else if (terrain_brush.mode == 4) { // Stamp Settings
                                ImGui::Indent();
                                ImGui::PushItemWidth(160.0f);
                                ImGui::SliderFloat("Rotation", &terrain_brush.stamp_rotation, 0.0f, 360.0f, "%.0f deg");
                                ImGui::PopItemWidth();
                                if (UIWidgets::SecondaryButton("Load Stamp Texture...", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.8f, 0))) {
                                    std::string path = SceneUI::openFileDialogW(L"Image Files\0*.png;*.jpg;*.jpeg;*.bmp\0");
                                    if (!path.empty()) {
                                        terrain_brush.stamp_texture = std::make_shared<Texture>(path, TextureType::Albedo);
                                    }
                                }
                                if (terrain_brush.stamp_texture) {
                                    ImGui::TextDisabled("Stamp: %s", std::filesystem::path(terrain_brush.stamp_texture->name).filename().string().c_str());
                                }
                                ImGui::Unindent();
                            }
                            else if (terrain_brush.mode == 5) { // Paint Settings
                                ImGui::Indent();
                                const char* channels[] = { "Layer 0 (Grass/Flat)", "Layer 1 (Rock/Slope)", "Layer 2 (Snow/Peak)", "Layer 3 (Custom Alpha)" };
                                ImGui::PushItemWidth(200.0f);
                                ImGui::Combo("Splat Channel", &terrain_brush.paint_channel, channels, IM_ARRAYSIZE(channels));
                                ImGui::PopItemWidth();
                                ImGui::Unindent();
                            }

                            ImGui::Separator();
                            ImGui::PushItemWidth(160.0f);
                            ImGui::SliderFloat("Radius", &terrain_brush.radius, 1.0f, 200.0f, "%.1f m");
                            ImGui::SliderFloat("Strength", &terrain_brush.strength, 0.01f, 10.0f, "%.2f");
                            ImGui::PopItemWidth();

                            ImGui::Checkbox("Show Viewport Circle", &terrain_brush.show_preview);

                            ImGui::Separator();
                            ImGui::TextDisabled("Hotkeys:");
                            ImGui::Text("  LMB: Apply Brush");
                            ImGui::Text("  Shift + LMB: Alternate Mode");
                        }
                    }

                    ImGui::Unindent();
                UIWidgets::EndSection();
            }
        }
    }
}

// ===============================================================================
// TERRAIN INTERACTION (Viewport)
// ===============================================================================

void SceneUI::handleTerrainBrush(UIContext& ctx) {
    if (!terrain_brush.enabled || terrain_brush.active_terrain_id == -1) return;
    
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return; // UI interaction
    
    int x, y;
    Uint32 buttons = SDL_GetMouseState(&x, &y);
    bool is_left_down = (buttons & SDL_BUTTON(SDL_BUTTON_LEFT));
    
    float win_w = io.DisplaySize.x;
    float win_h = io.DisplaySize.y;
    float u = (float)x / win_w;
    float v = (float)(win_h - y) / win_h;
    
    if (!ctx.scene.camera) return;
    Ray r = ctx.scene.camera->get_ray(u, v);
    
    HitRecord rec;
    auto* terrain = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
    if (!terrain) return;
    
    // Check intersection with terrain mesh
    // USE TERRAIN-ONLY RAYCAST (Prevents hitting foliage/instances)
    float t_hit = 0.0f;
    Vec3 normal_hit;
    
    if (TerrainManager::getInstance().intersectRay(terrain, r, t_hit, normal_hit)) {
        // Construct Hit Point
        Vec3 hitPoint = r.at(t_hit);
        
        // For compatibility with rest of code (though we ignore material check now as intersectRay ONLY checks this terrain)
        bool is_terrain = true;
        
        if (is_terrain) {
            // hitPoint is set above
            
            // DRAW PREVIEW
            if (terrain_brush.show_preview) {
                ImDrawList* dl = ImGui::GetForegroundDrawList();
                int segments = 32;
                ImVec4 color = (terrain_brush.mode == 5) ? ImVec4(1,1,0,0.8f) : ImVec4(1, 0.4f, 0.2f, 0.8f); // Yellow for paint, Orange for sculpt
                ImU32 col = ImGui::ColorConvertFloat4ToU32(color);
                
                for (int i = 0; i < segments; i++) {
                    float theta = (float)i / segments * 6.28318f;
                    float theta2 = (float)(i + 1) / segments * 6.28318f;
                    
                    Vec3 p1 = hitPoint + Vec3(cos(theta) * terrain_brush.radius, 0.1f, sin(theta) * terrain_brush.radius);
                    Vec3 p2 = hitPoint + Vec3(cos(theta2) * terrain_brush.radius, 0.1f, sin(theta2) * terrain_brush.radius);

                    auto Project = [&](Vec3 p) -> ImVec2 {
                        Camera& cam = *ctx.scene.camera;
                        Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
                        Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
                        Vec3 cam_up = cam_right.cross(cam_forward).normalize();
                        float fov_rad = cam.vfov * 3.14159f / 180.0f;
                         
                        Vec3 to_p = p - cam.lookfrom;
                        float depth = to_p.dot(cam_forward);
                        if (depth <= 0.1f) return ImVec2(-100,-100);
                        
                        float h_dim = depth * tan(fov_rad/2);
                        float w_dim = h_dim * (win_w/win_h);
                        
                        float lx = to_p.dot(cam_right);
                        float ly = to_p.dot(cam_up);
                        
                        float ndc_x = lx / w_dim;
                        float ndc_y = ly / h_dim; // Corrected sign for Y
                        
                        return ImVec2((ndc_x * 0.5f + 0.5f) * win_w, (0.5f - ndc_y * 0.5f) * win_h);
                    };
                    
                    ImVec2 s1 = Project(p1);
                    ImVec2 s2 = Project(p2);
                    
                    if (s1.x > -50 && s1.x < win_w + 50) {
                         dl->AddLine(s1, s2, col, 2.0f);
                    }
                }
            }
            
            // APPLY
            if (is_left_down) {
                 float dt = 1.0f / 60.0f;
                 
                 float targetH = terrain_brush.flatten_target;
                 if (terrain_brush.mode == 2 && !terrain_brush.use_fixed_height) {
                     targetH = hitPoint.y;
                 }

                 if (terrain_brush.mode == 5) {
                     // PAINT SPLAT
                     TerrainManager::getInstance().paintSplatMap(
                         terrain, hitPoint, terrain_brush.paint_channel, terrain_brush.radius, terrain_brush.strength, dt
                     );
                     // Paint updates CPU texture, updateSplatMapTexture() already syncs to GPU
                     if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                     ctx.renderer.resetCPUAccumulation();
                     // NOTE: No g_optix_rebuild_pending needed - texture is already updated via updateGPU()
                 } 
                 else if (terrain_brush.mode == 6 || terrain_brush.mode == 7) {
                     // PAINT HARDNESS (6=increase, 7=decrease)
                     bool increase = (terrain_brush.mode == 6);
                     TerrainManager::getInstance().paintHardness(
                         terrain, hitPoint, terrain_brush.radius, terrain_brush.strength, dt, increase
                     );
                     // Hardness map is CPU-only, no GPU rebuild needed
                     // Just reset accumulation for visual feedback (brush preview update)
                     if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                     ctx.renderer.resetCPUAccumulation();
                 }
                 else {
                     // SCULPT
                     TerrainManager::getInstance().sculpt(
                         terrain, 
                         hitPoint, 
                         terrain_brush.mode, 
                         terrain_brush.radius, 
                         terrain_brush.strength, 
                         dt,
                         targetH,
                         terrain_brush.stamp_texture,
                         terrain_brush.stamp_rotation
                     );
                     
                     if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                     ctx.renderer.resetCPUAccumulation();
                     
                     if (ctx.optix_gpu_ptr && g_hasOptix && ctx.render_settings.use_optix) {
                         // OPTIMIZATION: Only update the terrain mesh BLAS
                         // Does NOT trigger full scene rebuild, just BLAS upload + TLAS refit
                         ctx.optix_gpu_ptr->updateTerrainBLASPartial(terrain->name, terrain);
                     }
                     terrain->dirty_region.clear();
                     g_bvh_rebuild_pending = true;
                 }
            }
        }
    }
}



#endif

