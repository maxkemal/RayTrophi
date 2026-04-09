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
#include "Backend/VulkanBackend.h"
#include <TerrainManager.h>
#include "PrincipledBSDF.h" // For layer texture editing
#include <set>
#include <random>
#include <algorithm>
#include "InstanceManager.h"
#include "InstanceGroup.h"
#include "RiverSpline.h"
#include <thread>
#include <chrono>

extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern std::unique_ptr<Backend::IBackend> g_backend;
extern bool g_bvh_rebuild_pending;
extern bool g_optix_rebuild_pending;
extern bool g_vulkan_rebuild_pending;
extern bool g_viewport_raster_rebuild_pending;

// ===============================================================================
// ===============================================================================
// TERRAIN PANEL UI
// ===============================================================================

// Texture Graveyard: MOVED TO SceneUI MEMBER
// static std::vector<std::shared_ptr<Texture>> texture_graveyard; // Deprecated

static void ManageTextureGraveyard() {
    // Deprecated - Use SceneUI::manageTextureGraveyard()
}

static Backend::IBackend* GetTerrainRenderBackend(UIContext& ctx) {
    if (g_backend) {
        return g_backend.get();
    }
    if (ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) == nullptr) {
        return ctx.backend_ptr;
    }
    return nullptr;
}

static Backend::IViewportBackend* GetTerrainViewportBackend(UIContext& ctx) {
    if (g_viewport_backend) {
        return g_viewport_backend.get();
    }
    return dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
}

static bool TerrainRenderBackendIsVulkan(UIContext& ctx) {
    return dynamic_cast<Backend::VulkanBackendAdapter*>(GetTerrainRenderBackend(ctx)) != nullptr;
}

static void ScheduleTerrainTopologyRebuild(UIContext& ctx, bool include_cpu_bvh = true) {
    if (include_cpu_bvh) {
        g_bvh_rebuild_pending = true;
    }
    g_optix_rebuild_pending = true;
    g_viewport_raster_rebuild_pending = true;
    if (TerrainRenderBackendIsVulkan(ctx)) {
        g_vulkan_rebuild_pending = true;
    }
    
    extern bool g_geometry_dirty;
    extern std::atomic<uint64_t> g_scene_geometry_generation;
    extern std::atomic<bool> g_needs_optix_sync;
    extern bool g_mesh_cache_dirty;
    
    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);
    g_mesh_cache_dirty = true;
}

static void ResetTerrainBackendAccumulation(UIContext& ctx) {
    Backend::IBackend* renderBackend = GetTerrainRenderBackend(ctx);
    Backend::IViewportBackend* viewportBackend = GetTerrainViewportBackend(ctx);
    if (renderBackend) {
        renderBackend->resetAccumulation();
    }
    if (viewportBackend && viewportBackend != renderBackend) {
        viewportBackend->resetAccumulation();
    }
}

static void SyncTerrainMaterialState(UIContext& ctx) {
    ctx.renderer.updateBackendMaterials(ctx.scene);
    ResetTerrainBackendAccumulation(ctx);
}

static bool BuildTerrainDirtyRasterPatchData(
    TerrainObject* terrain,
    std::vector<size_t>& dirtyTriangleIndices,
    std::vector<std::pair<int, std::shared_ptr<Triangle>>>& meshEntries) {
    dirtyTriangleIndices.clear();
    meshEntries.clear();

    if (!terrain || terrain->mesh_triangles.empty()) {
        return false;
    }

    meshEntries.reserve(terrain->mesh_triangles.size());
    for (size_t i = 0; i < terrain->mesh_triangles.size(); ++i) {
        meshEntries.emplace_back(static_cast<int>(i), terrain->mesh_triangles[i]);
    }

    if (!terrain->dirty_region.has_any_dirty) {
        return false;
    }

    const int w = terrain->heightmap.width;
    const int h = terrain->heightmap.height;
    if (w <= 1 || h <= 1) {
        return false;
    }

    const int sector_w = std::max(1, w / DirtyRegion::SECTOR_GRID_SIZE);
    const int sector_h = std::max(1, h / DirtyRegion::SECTOR_GRID_SIZE);

    dirtyTriangleIndices.reserve(terrain->dirty_region.countDirtySectors() * 32);
    for (int sy = 0; sy < DirtyRegion::SECTOR_GRID_SIZE; ++sy) {
        for (int sx = 0; sx < DirtyRegion::SECTOR_GRID_SIZE; ++sx) {
            if (!terrain->dirty_region.sectors[sx][sy]) continue;

            const int startX = sx * sector_w;
            const int startZ = sy * sector_h;
            const int endX = std::min(startX + sector_w, w - 1);
            const int endZ = std::min(startZ + sector_h, h - 1);

            for (int z = startZ; z < endZ; ++z) {
                for (int x = startX; x < endX; ++x) {
                    const size_t tri_idx = (static_cast<size_t>(z) * static_cast<size_t>(w - 1) +
                                            static_cast<size_t>(x)) * 2ull;
                    if (tri_idx < terrain->mesh_triangles.size()) {
                        dirtyTriangleIndices.push_back(tri_idx);
                    }
                    if (tri_idx + 1 < terrain->mesh_triangles.size()) {
                        dirtyTriangleIndices.push_back(tri_idx + 1);
                    }
                }
            }
        }
    }

    std::sort(dirtyTriangleIndices.begin(), dirtyTriangleIndices.end());
    dirtyTriangleIndices.erase(
        std::unique(dirtyTriangleIndices.begin(), dirtyTriangleIndices.end()),
        dirtyTriangleIndices.end());

    return !dirtyTriangleIndices.empty();
}

void SceneUI::drawTerrainPanel(UIContext& ctx) {
    UIWidgets::PushControlSurfaceStyle(ImVec4(0.48f, 0.86f, 0.58f, 1.0f));
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
                // Ensure graph is visible, focused, and large enough.
                show_animation_panel = false;
                show_scene_log = false;
                show_anim_graph = false;
                show_asset_browser = false;
                preferred_bottom_panel_height = 420.0f;
                bottom_panel_height = preferred_bottom_panel_height;
                focus_bottom_panel_next_frame = true;
                // If terrain has generated mesh triangles, select one to sync viewport selection
                if (!t->mesh_triangles.empty() && ctx.selection.hasSelection() == false) {
                    auto tri = t->mesh_triangles.front();
                    if (tri) ctx.selection.selectObject(tri, -1, tri->nodeName);
                }
                SCENE_LOG_INFO("Terrain created: " + t->name);
                ctx.renderer.resetCPUAccumulation();
                ScheduleTerrainTopologyRebuild(ctx);
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
            ScheduleTerrainTopologyRebuild(ctx);
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
                             ScheduleTerrainTopologyRebuild(ctx);
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
            ScheduleTerrainTopologyRebuild(ctx);
        }
        UIWidgets::EndSection();
    }

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
                    ScheduleTerrainTopologyRebuild(ctx);
                    ResetTerrainBackendAccumulation(ctx);
                }
                ImGui::SameLine(); UIWidgets::HelpMarker("Fast: Simple 4-neighbor average\nSobel: Smooth 8-neighbor filter (recommended)\nHigh Quality: Enhanced edge detection");
                
                // Normal Strength Slider
                ImGui::PushItemWidth(160.0f);
                if (SceneUI::DrawSmartFloat("nstr", "Normal Strength", &t->normal_strength, 0.1f, 10.0f, "%.2f", false, nullptr, 16)) {
                    t->dirty_mesh = true;
                    TerrainManager::getInstance().updateTerrainMesh(t);
                    ctx.renderer.resetCPUAccumulation();
                    ScheduleTerrainTopologyRebuild(ctx);
                    ResetTerrainBackendAccumulation(ctx);
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

            // 3. LAYER MANAGEMENT
            if (UIWidgets::BeginSection("Materials & Layers", ImVec4(0.5f, 0.8f, 1.0f, 1.0f), true)) {
                ImGui::SameLine(); UIWidgets::HelpMarker("Manages which materials (grass, rock, snow, etc.) are used at different heights and slopes.");
            if (t->layers.empty()) {
                if (UIWidgets::PrimaryButton("Initialize Layers", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
                    TerrainManager::getInstance().initLayers(t);
                    SCENE_LOG_INFO("Terrain layers initialized for: " + t->name);
                    ctx.renderer.resetCPUAccumulation();
                    if (GetTerrainRenderBackend(ctx)) {
                        ctx.renderer.rebuildBackendGeometry(ctx.scene);
                        ResetTerrainBackendAccumulation(ctx);
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

                    // Use a plain collapsing header here because the embedded material
                    // editor already manages its own section stack and styling.
                    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(layerColor.x * 0.22f, layerColor.y * 0.22f, layerColor.z * 0.22f, 0.92f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(layerColor.x * 0.30f, layerColor.y * 0.30f, layerColor.z * 0.30f, 0.98f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(layerColor.x * 0.36f, layerColor.y * 0.36f, layerColor.z * 0.36f, 1.0f));
                    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(layerColor.x, layerColor.y, layerColor.z, 0.22f));
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 10.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 7.0f));
                    ImGui::SetNextItemOpen(this->terrain_layer_open[i], ImGuiCond_Always);
                    bool opened = ImGui::CollapsingHeader(layerName.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
                    this->terrain_layer_open[i] = opened;
                    ImGui::PopStyleVar(3);
                    ImGui::PopStyleColor(4);

                    if (opened) {

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
                                ctx.renderer.resetCPUAccumulation();
                                if (GetTerrainRenderBackend(ctx) || GetTerrainViewportBackend(ctx)) {
                                    SyncTerrainMaterialState(ctx);
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
                            ctx.renderer.resetCPUAccumulation();
                            if (GetTerrainRenderBackend(ctx) || GetTerrainViewportBackend(ctx)) {
                                SyncTerrainMaterialState(ctx);
                            }
                            SCENE_LOG_INFO("Created terrain layer material: " + matName);
                        }
                    }

                    // UV Scale
                        ImGui::PushItemWidth(160.0f);
                        if (SceneUI::DrawSmartFloat("uvs", "UV Tile Scale", &t->layer_uv_scales[i], 0.1f, 1000.0f, "%.1f", false, nullptr, 12)) {
                            ctx.renderer.resetCPUAccumulation();
                            if (GetTerrainRenderBackend(ctx) || GetTerrainViewportBackend(ctx)) {
                                SyncTerrainMaterialState(ctx);
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
                            if (GetTerrainRenderBackend(ctx) || GetTerrainViewportBackend(ctx)) {
                                SyncTerrainMaterialState(ctx);
                            }
                            SCENE_LOG_INFO("Filled mask for Layer " + std::to_string(i));
                        }
                     }

                    // Inline Material Editing (Full PrincipledBSDF)
                    if (t->layers[i]) {
                        auto pMat = std::dynamic_pointer_cast<PrincipledBSDF>(t->layers[i]);
                        if (pMat) {
                            // Get Material ID for keyframing/updates
                            uint16_t matID = MaterialManager::getInstance().getMaterialID(pMat->materialName);

                            // Call the reusable editor (member of SceneUI)
                            drawPrincipledBSDFEditor(pMat.get(), matID, ctx);
                        }
                    }
                    
                    // FOLIAGE UI INTEGRATION (Disabled - Moved to Central Section)
                    if (false);
                    
                    ImGui::Separator();
                }
                ImGui::PopID();
            }
            UIWidgets::EndSection();
            } // Materials & Layers section end
        } // if (t) - terrain sections 2-3 end
    } // if (active_terrain_id) - terrain sections 2-3 end
    } // terrain sections 2-3 wrapper end

    // ─────────────────────────────────────────────────────────────────────────
    // 4. FOLIAGE & SCATTER SYSTEM
    //    Accessible with or without terrain.
    //    When no terrain is active, only Mesh Surface scatter is available.
    //    Terrain-specific controls (splat map, altitude, slope) auto-hide.
    // ─────────────────────────────────────────────────────────────────────────
    {
    TerrainObject* t = (terrain_brush.active_terrain_id != -1)
        ? TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id)
        : nullptr;
    (void)t; // used by terrain scatter path below; may be nullptr for mesh-only scenes

            // Respect persisted open/closed state for foliage section
            ImGui::SetNextItemOpen(this->foliage_section_open, ImGuiCond_Always);
            bool fol_opened = UIWidgets::BeginSection("Foliage System", ImVec4(0.3f, 0.9f, 0.4f, 1.0f), true);
            this->foliage_section_open = fol_opened;
            if (fol_opened) {
                ImGui::SameLine(); UIWidgets::HelpMarker(
                    "Place trees, rocks, grass etc. via brush or procedural scatter.\n"
                    "Works on Terrain or any Mesh Surface (see Target Surface per layer).\n"
                    "Without an active terrain: use Target Type = Mesh Surface.");
                // ── Note when no terrain is active ────────────────────────────────────
                if (!t) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.85f, 0.3f, 1.0f));
                    ImGui::TextWrapped("No terrain active.  Scatter layers work with Target Type = Mesh Surface. "
                                      "Terrain-specific controls (splat map, altitude, curvature) are hidden.");
                    ImGui::PopStyleColor();
                    ImGui::Spacing();
                }
            
            InstanceManager& im = InstanceManager::getInstance();
            static char newFolGroupName[512] = "New Foliage Layer";
            ImGui::SetNextItemWidth(160.0f);
            ImGui::InputText("##NewFolName", newFolGroupName, 512);
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
                    char renameBuf[512];
                    snprintf(renameBuf, sizeof(renameBuf), "%s", group.name.c_str());
                    ImGui::SetNextItemWidth(200.0f);
                    if (ImGui::InputText("Layer Name##Rename", renameBuf, 512)) {
                        std::string newName = std::string(renameBuf);
                        // Preserve implicit type prefix so the layer stays visible in the Foliage list
                        std::string prefix = "";
                        if (group.name.find("Foliage") != std::string::npos) prefix = "Foliage_";
                        else if (group.name.find("Scatter") != std::string::npos) prefix = "Scatter_";

                        if (!prefix.empty()) {
                            if (newName.find("Foliage") == std::string::npos && newName.find("Scatter") == std::string::npos) {
                                newName = prefix + newName;
                            }
                        }

                        group.name = newName;
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
                     if (UIWidgets::SecondaryButton("Add Selected Object", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 0))) {
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

                     // ─── TARGET SURFACE ────────────────────────────────────────────────
                     ImGui::Text("   Target Surface:");
                     {
                         ImVec2 cp2 = ImGui::GetItemRectMin();
                         UIWidgets::DrawIcon(UIWidgets::IconType::World, ImVec2(cp2.x + 4, cp2.y + 1), 12, 0xFF88DDFF);
                     }
                     ImGui::SameLine(); UIWidgets::HelpMarker(
                         "Scatter on terrain heightmap or on any mesh surface.\n"
                         "  Terrain     - uses heightmap, splat map masks, slope/altitude filters.\n"
                         "  Mesh Surface - scatters on triangle faces; normal_influence controls\n"
                         "                orientation (0=upright for buildings, 1=normal-aligned for foliage).");
                     {
                         ImGui::PushItemWidth(160.0f);
                         const char* tgt_items[] = { "Terrain", "Mesh Surface" };
                         int tgt_idx = (int)group.target_type;
                         if (ImGui::Combo("Target Type##surf", &tgt_idx, tgt_items, 2))
                             group.target_type = (InstanceGroup::TargetType)tgt_idx;
                         ImGui::PopItemWidth();

                         if (group.target_type == InstanceGroup::TargetType::MESH) {
                             // Surface mesh picker
                             ImGui::PushItemWidth(200.0f);
                             const char* surf_preview = group.target_node_name.empty()
                                 ? "-- Pick Surface --" : group.target_node_name.c_str();
                             if (ImGui::BeginCombo("Surface Mesh##tgt", surf_preview)) {
                                 if (mesh_cache.empty()) rebuildMeshCache(ctx.scene.world.objects);
                                 static char surf_filter[64] = "";
                                 ImGui::SetNextItemWidth(180.0f);
                                 ImGui::InputText("Filter##sf", surf_filter, 64);
                                 ImGui::Separator();
                                 ImGui::BeginChild("SurfaceList", ImVec2(300, 200));
                                 for (const auto& [mname, tris_list] : mesh_cache) {
                                     if (mname.empty() || mname.find("_inst_") == 0) continue;
                                     if (surf_filter[0] != '\0' && mname.find(surf_filter) == std::string::npos) continue;
                                     bool is_sel = (group.target_node_name == mname);
                                     if (ImGui::Selectable(mname.c_str(), is_sel))
                                         group.target_node_name = mname;
                                 }
                                 ImGui::EndChild();
                                 ImGui::EndCombo();
                             }
                             ImGui::PopItemWidth();

                             // Normal influence slider (key parameter for foliage vs upright objects)
                             ImGui::PushItemWidth(160.0f);
                             ImGui::DragFloat("Normal Influence##ni", &group.brush_settings.normal_influence, 0.01f, 0.0f, 1.0f, "%.2f");
                             ImGui::PopItemWidth();
                             if (ImGui::IsItemHovered()) ImGui::SetTooltip(
                                 "Controls how much the instance aligns to the surface normal.\n"
                                 "  0.0 = always world-up     (buildings, large rocks)\n"
                                 "  1.0 = full surface normal  (foliage, grass, ground cover)\n"
                                 "  0.3-0.6 = natural blend    (medium rocks, rubble)");
                             ImGui::Checkbox("Align to Normal##an", &group.brush_settings.align_to_normal);
                             if (ImGui::IsItemHovered()) ImGui::SetTooltip("Enable normal-based orientation (uses Normal Influence above).");
                         }
                     }
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
                        
                        // Splat Map / Exclusion Mask (terrain only)
                        if (t) {
                        const char* channels[] = { "None", "Red (Grass/Flat)", "Green (Slope/Rock)", "Blue (Height/Snow)", "Alpha (Flow Map)" };
                        int current_idx = group.brush_settings.splat_map_channel + 1;
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
                        } // if (t) - terrain splat/exclusion end

                        if (UIWidgets::PrimaryButton("Scatter", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.32f, 30))) {
                        group.clearInstances();

                        if (group.target_type == InstanceGroup::TargetType::MESH && !group.target_node_name.empty()) {
                        // ─── MESH SURFACE SCATTER ────────────────────────────────────────────
                        if (mesh_cache.empty()) rebuildMeshCache(ctx.scene.world.objects);
                        std::vector<std::shared_ptr<Triangle>> surf_tris;
                        {
                            auto surf_it = mesh_cache.find(group.target_node_name);
                            if (surf_it != mesh_cache.end()) {
                                surf_tris.reserve(surf_it->second.size());
                                for (const auto& p : surf_it->second) surf_tris.push_back(p.second);
                            }
                        }
                        if (surf_tris.empty()) {
                            SCENE_LOG_WARN("Mesh Scatter: Target mesh not found or empty: " + group.target_node_name);
                        } else {
                            MeshSurfaceSampler mss;
                            mss.build(surf_tris);

                            int   count       = group.brush_settings.target_count;
                            std::mt19937 rng(group.brush_settings.seed);
                            int   spawned     = 0;
                            int   attempts    = 0;
                            const int max_attempts = count * 50;

                            float min_dist_sq  = group.brush_settings.min_distance * group.brush_settings.min_distance;
                            bool  check_overlap = group.brush_settings.min_distance > 0.01f;
                            float cell_size    = group.brush_settings.min_distance > 0.1f ? group.brush_settings.min_distance : 1.0f;
                            std::map<std::pair<int,int>, std::vector<Vec3>> grid;

                            while (spawned < count && attempts < max_attempts) {
                                attempts++;
                                MeshSurfaceSampler::Sample s = mss.sample(rng);

                                // Slope check (angle between face normal and world-up)
                                float slope_deg = acosf(std::clamp(s.normal.y, -1.0f, 1.0f)) * 57.2958f;
                                if (slope_deg > group.brush_settings.slope_max) continue;

                                // Height filter
                                if (s.position.y < group.brush_settings.height_min ||
                                    s.position.y > group.brush_settings.height_max) continue;

                                // Min-distance overlap check
                                if (check_overlap) {
                                    int cx = (int)std::floor(s.position.x / cell_size);
                                    int cz = (int)std::floor(s.position.z / cell_size);
                                    bool collision = false;
                                    for (int ddx = -1; ddx <= 1 && !collision; ++ddx)
                                        for (int ddz = -1; ddz <= 1 && !collision; ++ddz) {
                                            auto git = grid.find({cx+ddx, cz+ddz});
                                            if (git != grid.end())
                                                for (const auto& gp : git->second)
                                                    if ((gp - s.position).length_squared() < min_dist_sq) { collision = true; break; }
                                        }
                                    if (collision) continue;
                                    grid[{cx, cz}].push_back(s.position);
                                }

                                // generateRandomTransform already handles align_to_normal + normal_influence blend
                                InstanceTransform inst = group.generateRandomTransform(s.position, s.normal);
                                group.addInstance(inst);
                                spawned++;
                            }
                            SCENE_LOG_INFO("Mesh scatter: " + std::to_string(spawned) + "/" +
                                           std::to_string(count) + " instances on '" + group.target_node_name + "'.");
                        }
                        } else {
                        // ─── TERRAIN SCATTER (original path) ─────────────────────────────────
                        if (!t) {
                            SCENE_LOG_WARN("Scatter: No active terrain. Set Target Type to 'Mesh Surface' to scatter on a mesh.");
                        } else {
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
                                float val = 0.0f;
                                int texW = t->splatMap->width;
                                int texH = t->splatMap->height;
                                int sampleX = std::clamp((int)(u * (float)(texW - 1)), 0, texW - 1);
                                int sampleY = std::clamp((int)((1.0f - v) * (float)(texH - 1)), 0, texH - 1);
                                int pIdx = sampleY * texW + sampleX;
                                if (pIdx < (int)t->splatMap->pixels.size()) {
                                    const auto& p = t->splatMap->pixels[pIdx];
                                    if (group.brush_settings.splat_map_channel == 0) val = p.r / 255.0f;
                                    else if (group.brush_settings.splat_map_channel == 1) val = p.g / 255.0f;
                                    else if (group.brush_settings.splat_map_channel == 2) val = p.b / 255.0f;
                                    else if (group.brush_settings.splat_map_channel == 3) val = p.a / 255.0f;
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
                        } // if (t) end - terrain scatter body
                        } // terrain scatter else end
                        SceneUI::syncInstancesToScene(ctx, group, false);
                        // Trigger appropriate rebuilds depending on active render backend
                        const bool hasVulkanViewportPath = TerrainRenderBackendIsVulkan(ctx) || (g_viewport_backend != nullptr);
                        if (ctx.render_settings.use_optix) {
                            g_optix_rebuild_pending = true;
                        } else if (ctx.render_settings.use_vulkan) {
                            g_vulkan_rebuild_pending = true;
                        } else {
                            g_bvh_rebuild_pending = true;
                        }

                        // Ensure CPU BVH + backend geometry are rebuilt so Solid/raster backends see new instances
                        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                        ctx.renderer.resetCPUAccumulation();
                        if (Backend::IBackend* renderBackend = GetTerrainRenderBackend(ctx)) {
                            ctx.renderer.rebuildBackendGeometry(ctx.scene);
                            renderBackend->resetAccumulation();
                        }
                        ResetTerrainBackendAccumulation(ctx);
                        if (hasVulkanViewportPath && ctx.backend_ptr) {
                            ctx.renderer.updateBackendMaterials(ctx.scene);
                            ResetTerrainBackendAccumulation(ctx);
                            // Also request an immediate raster geometry rebuild on the viewport backend
                            Backend::IViewportBackend* rasterBackend = g_viewport_backend.get();
                            if (!rasterBackend) rasterBackend = dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
                            if (rasterBackend) {
                                rasterBackend->buildRasterGeometry(ctx.scene.world.objects);
                                rasterBackend->resetAccumulation();
                            } else {
                                g_viewport_raster_rebuild_pending = true;
                            }
                            if (TerrainRenderBackendIsVulkan(ctx)) g_vulkan_rebuild_pending = true;
                            else if (GetTerrainRenderBackend(ctx)) g_optix_rebuild_pending = true;
                        }
                        // Notify geometry generation change for viewport sync
                        g_geometry_dirty = true;
                        g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
                        ctx.renderer.resetCPUAccumulation();
                        ctx.start_render = true;
                     }
                     ImGui::SameLine();
                     if (UIWidgets::SecondaryButton("Clear", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.32f, 30))) {
                         group.clearInstances();
                         SceneUI::syncInstancesToScene(ctx, group, true);
                         const bool hasVulkanViewportPath_clear = TerrainRenderBackendIsVulkan(ctx) || (g_viewport_backend != nullptr);
                         if (ctx.render_settings.use_optix) {
                             g_optix_rebuild_pending = true;
                         } else if (ctx.render_settings.use_vulkan) {
                             g_vulkan_rebuild_pending = true;
                         } else {
                             g_bvh_rebuild_pending = true;
                         }
                         ResetTerrainBackendAccumulation(ctx);
                         if (hasVulkanViewportPath_clear && ctx.backend_ptr) {
                             ctx.renderer.updateBackendMaterials(ctx.scene);
                             ResetTerrainBackendAccumulation(ctx);
                            Backend::IViewportBackend* rasterBackend_clear = g_viewport_backend.get();
                            if (!rasterBackend_clear) rasterBackend_clear = dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
                            if (rasterBackend_clear) {
                                rasterBackend_clear->buildRasterGeometry(ctx.scene.world.objects);
                                rasterBackend_clear->resetAccumulation();
                            } else {
                                g_viewport_raster_rebuild_pending = true;
                            }
                            if (TerrainRenderBackendIsVulkan(ctx)) g_vulkan_rebuild_pending = true;
                            else if (GetTerrainRenderBackend(ctx)) g_optix_rebuild_pending = true;
                         }
                         // Ensure BVH/backend geometry rebuilt for solid view
                         ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                         if (Backend::IBackend* renderBackend_clear = GetTerrainRenderBackend(ctx)) {
                             ctx.renderer.rebuildBackendGeometry(ctx.scene);
                             renderBackend_clear->resetAccumulation();
                         }
                         g_geometry_dirty = true;
                         g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
                         ctx.renderer.resetCPUAccumulation();
                         ctx.start_render = true;
                     }
                     ImGui::SameLine();
                     if (UIWidgets::DangerButton("Delete", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.32f, 30))) {
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
                            ResetTerrainBackendAccumulation(ctx);
                            g_optix_rebuild_pending = true; // Must ensure this is handled in Main loop!
                        }
                        
                        if (group.wind_settings.enabled) {
                            bool changed = false;
                            changed |= ImGui::DragFloat("Speed", &group.wind_settings.speed, 0.05f, 0.0f, 10.0f);
                            changed |= ImGui::DragFloat("Strength", &group.wind_settings.strength, 0.1f, 0.0f, 45.0f, "%.1f deg");
                            changed |= ImGui::DragFloat("Turbulence", &group.wind_settings.turbulence, 0.05f, 0.0f, 5.0f);
                            changed |= ImGui::DragFloat("Wave Size", &group.wind_settings.wave_size, 1.0f, 1.0f, 500.0f);
                            changed |= ImGui::Checkbox("Use Source Profiles", &group.wind_settings.use_source_profiles);
                            changed |= ImGui::Checkbox("Allow CUDA Deform", &group.wind_settings.allow_gpu_deform);
                            changed |= ImGui::DragFloat("CUDA Max Distance", &group.wind_settings.gpu_deform_max_distance, 0.5f, 1.0f, 500.0f, "%.1f m");
                            changed |= ImGui::DragInt("CUDA Max Instances", &group.wind_settings.gpu_deform_max_instances, 1.0f, 1, 10000);
                            
                            // Direction Logic
                            float current_angle = atan2(group.wind_settings.direction.z, group.wind_settings.direction.x) * 57.2958f;
                            if (ImGui::DragFloat("Direction", &current_angle, 1.0f, -180.0f, 180.0f, "%.0f deg")) {
                                float rad = current_angle / 57.2958f;
                                group.wind_settings.direction = Vec3(cos(rad), 0, sin(rad));
                                changed = true;
                            }
                            
                            if (group.wind_settings.use_source_profiles && !group.sources.empty()) {
                                ImGui::SeparatorText("Source Wind Profiles");
                                for (int s_i = 0; s_i < group.sources.size(); ++s_i) {
                                    auto& src = group.sources[s_i];
                                    ImGui::PushID(("windsrc" + std::to_string(s_i)).c_str());
                                    if (ImGui::TreeNode(src.name.c_str())) {
                                        changed |= ImGui::DragFloat("Strength Scale", &src.settings.wind_strength_scale, 0.05f, 0.0f, 3.0f, "%.2f");
                                        changed |= ImGui::DragFloat("Speed Scale", &src.settings.wind_speed_scale, 0.05f, 0.1f, 3.0f, "%.2f");
                                        changed |= ImGui::DragFloat("Turbulence Scale", &src.settings.wind_turbulence_scale, 0.05f, 0.0f, 3.0f, "%.2f");
                                        changed |= ImGui::DragFloat("Bend Limit Scale", &src.settings.wind_bend_limit_scale, 0.05f, 0.1f, 3.0f, "%.2f");
                                        changed |= ImGui::DragFloat("Phase Offset", &src.settings.wind_phase_offset, 0.05f, -10.0f, 10.0f, "%.2f");
                                        ImGui::TreePop();
                                    }
                                    ImGui::PopID();
                                }
                            }

                            if (changed) {
                                // Real-time preview requires reset
                                ctx.renderer.resetCPUAccumulation();
                                ResetTerrainBackendAccumulation(ctx);
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

                    // Trigger backend updates so Solid/Vulkan/Optix reflect deletion
                    const bool hasVulkanViewportPath_del = TerrainRenderBackendIsVulkan(ctx) || (g_viewport_backend != nullptr);
                    if (ctx.render_settings.use_optix) {
                        g_optix_rebuild_pending = true;
                    } else if (ctx.render_settings.use_vulkan) {
                        g_vulkan_rebuild_pending = true;
                    } else {
                        g_bvh_rebuild_pending = true;
                    }
                    ResetTerrainBackendAccumulation(ctx);
                    if (hasVulkanViewportPath_del && ctx.backend_ptr) {
                        ctx.renderer.updateBackendMaterials(ctx.scene);
                        ResetTerrainBackendAccumulation(ctx);
                        Backend::IViewportBackend* rasterBackend_del = g_viewport_backend.get();
                        if (!rasterBackend_del) rasterBackend_del = dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
                        if (rasterBackend_del) {
                            rasterBackend_del->buildRasterGeometry(ctx.scene.world.objects);
                            rasterBackend_del->resetAccumulation();
                        } else {
                            g_viewport_raster_rebuild_pending = true;
                        }
                        if (TerrainRenderBackendIsVulkan(ctx)) g_vulkan_rebuild_pending = true;
                        else if (GetTerrainRenderBackend(ctx)) g_optix_rebuild_pending = true;
                    }

                    // Rebuild BVH and backend geometry so Solid (raster/CPU) updates immediately
                    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                    ctx.renderer.resetCPUAccumulation();
                    if (Backend::IBackend* renderBackend_del = GetTerrainRenderBackend(ctx)) {
                        ctx.renderer.rebuildBackendGeometry(ctx.scene);
                        renderBackend_del->resetAccumulation();
                    }
                    g_geometry_dirty = true;
                    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
                    ctx.start_render = true;

                    ctx.renderer.resumeRendering();

                }
            }
            UIWidgets::EndSection();
        } // if (fol_opened) end
    } // standalone foliage block end

    // ─── Sections 5-6: terrain-specific procedural tools & timeline ──────────
    if (terrain_brush.active_terrain_id != -1) {
        auto* t = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
        if (t) {
            if (UIWidgets::BeginSection("River Terrain Carve", ImVec4(0.42f, 0.76f, 1.0f, 1.0f), true)) {
                auto& riverMgr = RiverManager::getInstance();
                RiverSpline* selectedRiver = riverMgr.getRiver(riverMgr.editingRiverId);

                ImGui::PushID("RiverTerrainCarve");
                if (selectedRiver) {
                    ImGui::TextColored(ImVec4(0.62f, 0.86f, 1.0f, 1.0f), "Active River: %s", selectedRiver->name.c_str());
                    ImGui::SameLine();
                    ImGui::TextDisabled("(%d points)", (int)selectedRiver->controlPointCount());
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.48f, 1.0f), "No river selected for carving.");
                    ImGui::TextDisabled("Create or select a river before carving terrain.");
                }

                ImGui::Checkbox("Auto-Carve on Move", &riverMgr.autoCarveOnMove);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Automatically updates the active terrain while moving river points.");
                }

                SceneUI::DrawSmartFloat("cdm", "Depth Multiplier", &riverMgr.carveDepthMult, 0.1f, 3.0f, "%.1f", false, nullptr, 16);
                SceneUI::DrawSmartFloat("csm", "Smoothness", &riverMgr.carveSmoothness, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                ImGui::Checkbox("Apply Post-Erosion", &riverMgr.carveAutoPostErosion);
                if (riverMgr.carveAutoPostErosion) {
                    ImGui::SliderInt("Erosion Iterations", &riverMgr.carveErosionIterations, 5, 30);
                }

                ImGui::Separator();
                ImGui::TextColored(ImVec4(0.55f, 0.92f, 0.76f, 1.0f), "Natural Riverbed");

                ImGui::Checkbox("Edge Noise", &riverMgr.carveEnableNoise);
                if (riverMgr.carveEnableNoise) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cns", "Noise Scale", &riverMgr.carveNoiseScale, 0.05f, 0.5f, "%.2f", false, nullptr, 16);
                    SceneUI::DrawSmartFloat("cnst", "Noise Strength", &riverMgr.carveNoiseStrength, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }

                ImGui::Checkbox("Deep Pools", &riverMgr.carveEnableDeepPools);
                if (riverMgr.carveEnableDeepPools) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cpf", "Pool Frequency", &riverMgr.carvePoolFrequency, 0.0f, 0.5f, "%.2f", false, nullptr, 16);
                    SceneUI::DrawSmartFloat("cpdm", "Pool Depth Mult", &riverMgr.carvePoolDepthMult, 1.0f, 3.0f, "%.1f", false, nullptr, 16);
                    ImGui::Unindent();
                }

                ImGui::Checkbox("Riffles", &riverMgr.carveEnableRiffles);
                if (riverMgr.carveEnableRiffles) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("crf", "Riffle Frequency", &riverMgr.carveRiffleFrequency, 0.0f, 0.5f, "%.2f", false, nullptr, 16);
                    SceneUI::DrawSmartFloat("crdm", "Riffle Depth Mult", &riverMgr.carveRiffleDepthMult, 0.1f, 0.8f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }

                ImGui::Checkbox("Asymmetric Banks", &riverMgr.carveEnableAsymmetry);
                if (riverMgr.carveEnableAsymmetry) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cas", "Asymmetry Strength", &riverMgr.carveAsymmetryStrength, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }

                ImGui::Checkbox("Point Bars", &riverMgr.carveEnablePointBars);
                if (riverMgr.carveEnablePointBars) {
                    ImGui::Indent();
                    SceneUI::DrawSmartFloat("cpbs", "Point Bar Strength", &riverMgr.carvePointBarStrength, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
                    ImGui::Unindent();
                }

                const bool hasActiveTerrain = TerrainManager::getInstance().hasActiveTerrain();
                const bool canCarve = hasActiveTerrain && selectedRiver && selectedRiver->spline.pointCount() >= 2;
                if (!canCarve) {
                    ImGui::BeginDisabled();
                }

                auto backupTerrainIfNeeded = [&]() {
                    auto& tm = TerrainManager::getInstance();
                    if (!riverMgr.hasTerrainBackup && !tm.getTerrains().empty()) {
                        auto& terrainRef = tm.getTerrains()[0];
                        riverMgr.terrainBackupData = terrainRef.heightmap.data;
                        riverMgr.terrainBackupWidth = terrainRef.heightmap.width;
                        riverMgr.terrainBackupHeight = terrainRef.heightmap.height;
                        riverMgr.hasTerrainBackup = true;
                    }
                };

                auto sampleRiverSpline = [&](std::vector<Vec3>& samplePoints,
                                             std::vector<float>& sampleWidths,
                                             std::vector<float>& sampleDepths) {
                    const int numSamples = selectedRiver->lengthSubdivisions * 3;
                    samplePoints.reserve(numSamples + 1);
                    sampleWidths.reserve(numSamples + 1);
                    sampleDepths.reserve(numSamples + 1);
                    for (int i = 0; i <= numSamples; ++i) {
                        const float tt = (float)i / (float)numSamples;
                        samplePoints.push_back(selectedRiver->spline.samplePosition(tt));
                        sampleWidths.push_back(selectedRiver->spline.sampleUserData1(tt));
                        sampleDepths.push_back(selectedRiver->spline.sampleUserData2(tt) * riverMgr.carveDepthMult);
                    }
                };

                auto applyPostErosion = [&]() {
                    if (!riverMgr.carveAutoPostErosion) return;
                    auto& terrains = TerrainManager::getInstance().getTerrains();
                    if (!terrains.empty()) {
                        ThermalErosionParams ep;
                        ep.iterations = riverMgr.carveErosionIterations;
                        ep.talusAngle = 0.3f;
                        ep.erosionAmount = 0.4f;
                        TerrainManager::getInstance().thermalErosion(&terrains[0], ep);
                    }
                };

                auto finalizeCarve = [&]() {
                    selectedRiver->needsRebuild = true;
                    riverMgr.generateMesh(selectedRiver, ctx.scene);
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    ctx.renderer.resetCPUAccumulation();
                    ProjectManager::getInstance().markModified();
                };

                if (UIWidgets::PrimaryButton("Carve River Bed", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
                    backupTerrainIfNeeded();
                    std::vector<Vec3> samplePoints;
                    std::vector<float> sampleWidths;
                    std::vector<float> sampleDepths;
                    sampleRiverSpline(samplePoints, sampleWidths, sampleDepths);
                    TerrainManager::getInstance().carveRiverBed(
                        -1,
                        samplePoints,
                        sampleWidths,
                        sampleDepths,
                        riverMgr.carveSmoothness,
                        ctx.scene
                    );
                    applyPostErosion();
                    finalizeCarve();
                }

                if (UIWidgets::SecondaryButton("Carve Natural Riverbed", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
                    backupTerrainIfNeeded();
                    std::vector<Vec3> samplePoints;
                    std::vector<float> sampleWidths;
                    std::vector<float> sampleDepths;
                    sampleRiverSpline(samplePoints, sampleWidths, sampleDepths);

                    TerrainManager::NaturalCarveParams np;
                    np.enableNoise = riverMgr.carveEnableNoise;
                    np.noiseScale = riverMgr.carveNoiseScale;
                    np.noiseStrength = riverMgr.carveNoiseStrength;
                    np.enableDeepPools = riverMgr.carveEnableDeepPools;
                    np.poolFrequency = riverMgr.carvePoolFrequency;
                    np.poolDepthMult = riverMgr.carvePoolDepthMult;
                    np.enableRiffles = riverMgr.carveEnableRiffles;
                    np.riffleFrequency = riverMgr.carveRiffleFrequency;
                    np.riffleDepthMult = riverMgr.carveRiffleDepthMult;
                    np.enableAsymmetry = riverMgr.carveEnableAsymmetry;
                    np.asymmetryStrength = riverMgr.carveAsymmetryStrength;
                    np.enablePointBars = riverMgr.carveEnablePointBars;
                    np.pointBarStrength = riverMgr.carvePointBarStrength;

                    TerrainManager::getInstance().carveRiverBedNatural(
                        -1,
                        samplePoints,
                        sampleWidths,
                        sampleDepths,
                        riverMgr.carveSmoothness,
                        np,
                        ctx.scene
                    );
                    applyPostErosion();
                    finalizeCarve();
                }

                if (!canCarve) {
                    ImGui::EndDisabled();
                }

                ImGui::Separator();
                if (riverMgr.hasTerrainBackup) {
                    ImGui::TextColored(ImVec4(0.48f, 1.0f, 0.72f, 1.0f), "Terrain backup available");
                    if (UIWidgets::DangerButton("Reset / Undo Carve", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.62f, 0))) {
                        auto& tm = TerrainManager::getInstance();
                        if (!tm.getTerrains().empty()) {
                            auto& terrainRef = tm.getTerrains()[0];
                            if (terrainRef.heightmap.data.size() == riverMgr.terrainBackupData.size()) {
                                terrainRef.heightmap.data = riverMgr.terrainBackupData;
                                tm.updateTerrainMesh(&terrainRef);
                                g_bvh_rebuild_pending = true;
                                g_optix_rebuild_pending = true;
                                ctx.renderer.resetCPUAccumulation();
                                ProjectManager::getInstance().markModified();
                                SCENE_LOG_INFO("Terrain restored from backup.");
                            }
                        }
                    }
                    ImGui::SameLine();
                    if (UIWidgets::SecondaryButton("Commit", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.34f, 0))) {
                        riverMgr.hasTerrainBackup = false;
                        riverMgr.terrainBackupData.clear();
                        SCENE_LOG_INFO("Carve changes committed.");
                    }
                } else {
                    ImGui::TextDisabled("Backup is created automatically on first carve.");
                }
                ImGui::PopID();
                UIWidgets::EndSection();
            }

            // 5. PROCEDURAL TOOLS & MASKS
            if (UIWidgets::BeginSection("Procedural Generators", ImVec4(0.8f, 0.4f, 1.0f, 1.0f), true)) {
                ImGui::TextWrapped("Terrain mask generation and splat import/export moved to the shared Paint panel.");
                ImGui::BulletText("Use Paint Mode for terrain layers, masks, and splat map IO.");
                ImGui::BulletText("This terrain tab now keeps only terrain setup and overview tools.");

                UIWidgets::EndSection();
            }
        
            // 6. TIMELINE & EXPORT
            if (UIWidgets::BeginSection("Timeline & Export", ImVec4(0.7f, 0.4f, 0.8f, 1.0f), true)) {
                
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
                    ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Track Status: Active");
                    
                    // Show keyframe count for this track
                    int kf_count = (int)ctx.scene.timeline.tracks[trackName].keyframes.size();
                    ImGui::SameLine();
                    ImGui::TextDisabled("(%d total keyframes)", kf_count);
                } else {
                    ImGui::TextDisabled("Track Status: No Track Found");
                }

                // Heightmap Export
                ImGui::Separator();
                if (ImGui::Button("Export Heightmap (16-bit RAW)", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
                    std::string path = SceneUI::saveFileDialogW(L"RAW Files\0*.raw\0", L"raw");
                    if (!path.empty()) {
                        TerrainManager::getInstance().exportHeightmap(t, path);
                    }
                }
                UIWidgets::EndSection();
            }
        } // if (t) end - sections 5-6
    } // if (active_terrain_id) end - sections 5-6

    // -----------------------------------------------------------------------------
    // 8. SCULPT WORKFLOW
    // -----------------------------------------------------------------------------
    if (terrain_brush.active_terrain_id != -1) {
        if (UIWidgets::BeginSection("Terrain Sculpt Workflow", ImVec4(1.0f, 0.7f, 0.4f, 1.0f), true)) {
            ImVec2 hp = ImGui::GetItemRectMin();
            UIWidgets::DrawIcon(UIWidgets::IconType::Magnet, ImVec2(hp.x + 8, hp.y + 4), 16, 0xFFBBBBBB);
            UIWidgets::HelpMarker("Terrain sculpt controls moved to the shared Sculpt panel so mesh and terrain use one workflow.");

            auto* activeTerrain = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
            if (activeTerrain) {
                ImGui::Text("Active Terrain: %s", activeTerrain->name.c_str());
            } else {
                ImGui::TextDisabled("No active terrain selected.");
            }
            ImGui::Spacing();
            ImGui::BulletText("Use the Modifiers & Sculpt tab to sculpt terrain.");
            ImGui::BulletText("Terrain-specific brush controls now appear in the shared Sculpt dock.");
            ImGui::BulletText("Mask and terrain paint workflows can be consolidated next.");
            ImGui::TextDisabled("This panel now acts as a terrain workflow guide.");

            UIWidgets::EndSection();
        }
    }
    UIWidgets::PopControlSurfaceStyle();
}
// ===============================================================================
// TERRAIN INTERACTION (Viewport)
// ===============================================================================

void SceneUI::handleTerrainBrush(UIContext& ctx) {
    const bool sculpt_proxy_active =
        terrain_sculpt_proxy_active &&
        sculpt_mode_state.enabled &&
        mesh_workspace_mode == MeshWorkspaceMode::Sculpt;
    if ((!terrain_brush.enabled && !sculpt_proxy_active) || terrain_brush.active_terrain_id == -1) return;
    
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) return; // UI interaction
    
    int x, y;
    Uint32 buttons = SDL_GetMouseState(&x, &y);
    bool is_left_down = (buttons & SDL_BUTTON(SDL_BUTTON_LEFT));
    auto* terrain = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
    if (!terrain) return;
    static bool was_left_down = false;
    static int last_brush_mouse_x = std::numeric_limits<int>::min();
    static int last_brush_mouse_y = std::numeric_limits<int>::min();
    static bool pending_geometry_commit = false;
    static bool pending_vulkan_material_sync = false;
    static float last_vulkan_paint_sync_time = -1000.0f;
    static float last_live_geometry_sync_time = -1000.0f;

    const bool mouse_moved_since_last_brush_frame =
        (x != last_brush_mouse_x) || (y != last_brush_mouse_y);
    last_brush_mouse_x = x;
    last_brush_mouse_y = y;
    if (mouse_moved_since_last_brush_frame && terrain_brush.show_preview) {
        ctx.start_render = true;
    }

    auto syncTerrainVulkanViewport = [&](Backend::IViewportBackend* vkBackend,
                                         bool includeRtGeometry,
                                         bool allowFullRasterFallback) -> bool {
        if (!vkBackend || !terrain) return false;

        bool handled = false;
        if (includeRtGeometry) {
            if (auto* vkRtBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(vkBackend)) {
                handled = vkRtBackend->updateTerrainBLASPartial(terrain->name, terrain);
            }
        }

        std::vector<size_t> dirtyTriangleIndices;
        std::vector<std::pair<int, std::shared_ptr<Triangle>>> meshEntries;
        const bool hasDirtyPatchData = BuildTerrainDirtyRasterPatchData(
            terrain,
            dirtyTriangleIndices,
            meshEntries);

        if (hasDirtyPatchData &&
            vkBackend->patchRasterMeshTriangles(terrain->name, dirtyTriangleIndices, meshEntries)) {
            handled = true;
        } else if (vkBackend->updateRasterMeshFromTriangles(terrain->name, terrain->mesh_triangles)) {
            handled = true;
        } else if (allowFullRasterFallback) {
            vkBackend->buildRasterGeometry(ctx.scene.world.objects);
            handled = true;
        }

        vkBackend->resetAccumulation();
        return handled;
    };

    auto commitTerrainStroke = [&]() {
        const bool hasVulkanViewportPath =
            TerrainRenderBackendIsVulkan(ctx) ||
            (g_viewport_backend != nullptr);
        const bool renderedViewportActive = (ctx.backend_ptr == g_backend.get());

        if (pending_vulkan_material_sync && hasVulkanViewportPath && ctx.backend_ptr) {
            ctx.renderer.updateBackendMaterials(ctx.scene);
            ResetTerrainBackendAccumulation(ctx);
        }

        if (pending_geometry_commit) {
            bool handled = false;
            bool renderedBackendHandled = !renderedViewportActive;
            if (hasVulkanViewportPath && ctx.backend_ptr) {
                if (auto* vkRtBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get())) {
                    handled = vkRtBackend->updateTerrainBLASPartial(terrain->name, terrain);
                    if (handled) {
                        vkRtBackend->resetAccumulation();
                    }
                    if (renderedViewportActive) {
                        renderedBackendHandled = handled;
                    }
                }

                Backend::IViewportBackend* rasterViewportBackend = g_viewport_backend.get();
                if (!rasterViewportBackend) {
                    rasterViewportBackend = dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
                }
                if (rasterViewportBackend && !renderedViewportActive) {
                    if (!terrain->dirty_region.has_any_dirty ||
                        syncTerrainVulkanViewport(rasterViewportBackend, false, true)) {
                        handled = true;
                    }
                }
                if (!renderedBackendHandled) {
                    g_viewport_raster_rebuild_pending = true;
                    if (TerrainRenderBackendIsVulkan(ctx)) {
                        g_vulkan_rebuild_pending = true;
                    }
                }
            } else if (ctx.render_settings.use_optix) {
                g_optix_rebuild_pending = true;
            } else {
                g_bvh_rebuild_pending = true;
            }

            if (renderedViewportActive) {
                if (ctx.render_settings.use_optix) {
                    g_optix_rebuild_pending = true;
                } else if (ctx.render_settings.use_vulkan) {
                    g_vulkan_rebuild_pending = true;
                } else {
                    g_bvh_rebuild_pending = true;
                }
                ctx.start_render = true;
            }
            ctx.renderer.resetCPUAccumulation();
            terrain->dirty_region.clear();
        }

        pending_geometry_commit = false;
        pending_vulkan_material_sync = false;
    };

    if (!is_left_down && was_left_down) {
        commitTerrainStroke();
    }
    was_left_down = is_left_down;
    
    float win_w = io.DisplaySize.x;
    float win_h = io.DisplaySize.y;
    float u = (float)x / win_w;
    float v = (float)(win_h - y) / win_h;
    
    if (!ctx.scene.camera) return;
    Ray r = ctx.scene.camera->get_ray(u, v);
    
    HitRecord rec;
    
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
                ImU32 innerCol = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 0.35f));

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
                    float ndc_y = ly / h_dim;
                    return ImVec2((ndc_x * 0.5f + 0.5f) * win_w, (0.5f - ndc_y * 0.5f) * win_h);
                };
                
                for (int i = 0; i < segments; i++) {
                    float theta = (float)i / segments * 6.28318f;
                    float theta2 = (float)(i + 1) / segments * 6.28318f;
                    
                    Vec3 p1 = hitPoint + Vec3(cos(theta) * terrain_brush.radius, 0.1f, sin(theta) * terrain_brush.radius);
                    Vec3 p2 = hitPoint + Vec3(cos(theta2) * terrain_brush.radius, 0.1f, sin(theta2) * terrain_brush.radius);
                    
                    ImVec2 s1 = Project(p1);
                    ImVec2 s2 = Project(p2);
                    
                    if (s1.x > -50 && s1.x < win_w + 50) {
                         dl->AddLine(s1, s2, col, 2.0f);
                    }
                }

                for (int ring = 0; ring < 2; ++ring) {
                    float ringScale = (ring == 0) ? 0.5f : std::pow(0.5f, 1.0f / std::max(terrain_brush.curve, 0.25f));
                    float ringRadius = terrain_brush.radius * ringScale;
                    for (int i = 0; i < segments; i++) {
                        float theta = (float)i / segments * 6.28318f;
                        float theta2 = (float)(i + 1) / segments * 6.28318f;
                        Vec3 p1 = hitPoint + Vec3(cos(theta) * ringRadius, 0.12f, sin(theta) * ringRadius);
                        Vec3 p2 = hitPoint + Vec3(cos(theta2) * ringRadius, 0.12f, sin(theta2) * ringRadius);
                        ImVec2 s1 = Project(p1);
                        ImVec2 s2 = Project(p2);
                        if (s1.x > -50 && s1.x < win_w + 50) {
                            dl->AddLine(s1, s2, innerCol, 1.0f);
                        }
                    }
                }

                if (terrain_brush.mode == 4) {
                    float rad = terrain_brush.stamp_rotation * 3.14159f / 180.0f;
                    float c = cosf(rad);
                    float s = sinf(rad);
                    Vec3 corners[4] = {
                        Vec3(-terrain_brush.radius, 0.15f, -terrain_brush.radius),
                        Vec3( terrain_brush.radius, 0.15f, -terrain_brush.radius),
                        Vec3( terrain_brush.radius, 0.15f,  terrain_brush.radius),
                        Vec3(-terrain_brush.radius, 0.15f,  terrain_brush.radius)
                    };
                    ImVec2 screenCorners[4];
                    for (int i = 0; i < 4; ++i) {
                        Vec3 local = corners[i];
                        Vec3 rotated(local.x * c - local.z * s, local.y, local.x * s + local.z * c);
                        screenCorners[i] = Project(hitPoint + rotated);
                    }
                    for (int i = 0; i < 4; ++i) {
                        ImVec2 a = screenCorners[i];
                        ImVec2 b = screenCorners[(i + 1) % 4];
                        if (a.x > -100 && a.x < win_w + 100) {
                            dl->AddLine(a, b, innerCol, 1.5f);
                        }
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
                     ResetTerrainBackendAccumulation(ctx);
                     const bool hasVulkanViewportPath =
                         TerrainRenderBackendIsVulkan(ctx) ||
                         (g_viewport_backend != nullptr);
                     if (hasVulkanViewportPath && ctx.backend_ptr) {
                         pending_vulkan_material_sync = true;
                         float now = (float)ImGui::GetTime();
                         if (now - last_vulkan_paint_sync_time > 0.08f) {
                             ctx.renderer.updateBackendMaterials(ctx.scene);
                             ResetTerrainBackendAccumulation(ctx);
                             last_vulkan_paint_sync_time = now;
                             pending_vulkan_material_sync = false;
                         }
                     }
                     ctx.renderer.resetCPUAccumulation();
                     ctx.start_render = true;
                 } 
                 else if (terrain_brush.mode == 6 || terrain_brush.mode == 7) {
                     // PAINT HARDNESS (6=increase, 7=decrease)
                     bool increase = (terrain_brush.mode == 6);
                     TerrainManager::getInstance().paintHardness(
                         terrain, hitPoint, terrain_brush.radius, terrain_brush.strength, dt, increase
                     );
                     // Hardness map is CPU-only, no GPU rebuild needed
                     // Just reset accumulation for visual feedback (brush preview update)
                     ResetTerrainBackendAccumulation(ctx);
                     ctx.renderer.resetCPUAccumulation();
                     ctx.start_render = true;
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
                         terrain_brush.curve,
                         targetH,
                         terrain_brush.stamp_texture,
                         terrain_brush.stamp_rotation,
                         false
                     );
                     
                     ResetTerrainBackendAccumulation(ctx);
                     ctx.renderer.resetCPUAccumulation();
                     
                     Backend::IViewportBackend* liveRasterViewportBackend = g_viewport_backend.get();
                     if (!liveRasterViewportBackend) {
                         liveRasterViewportBackend = dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
                     }

                     bool liveViewportUpdated = false;
                     if (liveRasterViewportBackend) {
                         const Backend::ViewportMode viewportMode = liveRasterViewportBackend->getViewportMode();
                         const bool interactiveRasterMode =
                             viewportMode == Backend::ViewportMode::Solid ||
                             viewportMode == Backend::ViewportMode::Matcap ||
                             viewportMode == Backend::ViewportMode::MaterialPreview;
                         liveViewportUpdated = syncTerrainVulkanViewport(
                             liveRasterViewportBackend,
                             false,
                             interactiveRasterMode);
                         if (liveViewportUpdated) {
                             terrain->dirty_region.clear();
                         }
                     }

                     const bool renderedViewportActive = (ctx.backend_ptr == g_backend.get());
                     bool liveRenderBackendUpdated = false;
                     if (renderedViewportActive) {
                         if (ctx.optix_gpu_ptr && g_hasOptix && ctx.render_settings.use_optix) {
                             // OptiX rendered viewport: push terrain BLAS updates immediately per dab.
                             liveRenderBackendUpdated =
                                 ctx.optix_gpu_ptr->updateTerrainBLASPartial(terrain->name, terrain);
                             if (!liveRenderBackendUpdated) {
                                 g_optix_rebuild_pending = true;
                             }
                         } else if (auto* vkRenderedBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(ctx.backend_ptr)) {
                             liveRenderBackendUpdated = vkRenderedBackend->updateTerrainBLASPartial(terrain->name, terrain);
                             if (liveRenderBackendUpdated) {
                                 vkRenderedBackend->resetAccumulation();
                             } else {
                                 g_vulkan_rebuild_pending = true;
                             }
                         } else {
                             // CPU rendered viewport: keep BVH fresh so terrain sculpt appears live.
                             g_cpu_bvh_refit_pending = true;
                             liveRenderBackendUpdated = true;
                         }
                     } else {
                         float now = (float)ImGui::GetTime();
                         if (now - last_live_geometry_sync_time > 0.08f) {
                             const bool hasVulkanViewportPath =
                                 TerrainRenderBackendIsVulkan(ctx) ||
                                 (g_viewport_backend != nullptr);
                             if (hasVulkanViewportPath) {
                                 bool updated = false;
                                 if (auto* vkRtBackend = dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get())) {
                                     updated = vkRtBackend->updateTerrainBLASPartial(terrain->name, terrain);
                                     if (updated) {
                                         vkRtBackend->resetAccumulation();
                                     }
                                 }
                                 if (liveViewportUpdated) {
                                     updated = true;
                                 } else {
                                     ResetTerrainBackendAccumulation(ctx);
                                 }
                                 if (!updated) {
                                     g_viewport_raster_rebuild_pending = true;
                                     if (TerrainRenderBackendIsVulkan(ctx)) {
                                         g_vulkan_rebuild_pending = true;
                                     }
                                 }
                             } else {
                                 g_cpu_bvh_refit_pending = true;
                             }
                             last_live_geometry_sync_time = now;
                         }
                     }
                     if (!liveViewportUpdated && liveRasterViewportBackend) {
                         g_viewport_raster_rebuild_pending = true;
                     }
                     if (renderedViewportActive && !liveRenderBackendUpdated) {
                         if (ctx.render_settings.use_optix) {
                             g_optix_rebuild_pending = true;
                         } else if (ctx.render_settings.use_vulkan) {
                             g_vulkan_rebuild_pending = true;
                         } else {
                             g_bvh_rebuild_pending = true;
                         }
                     }
                     ctx.start_render = true;
                     pending_geometry_commit = true;
                 }
            }
        }
    }
}



#endif

