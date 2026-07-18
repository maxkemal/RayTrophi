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
#include "FoliageAssetLibrary.h"
#include "RiverSpline.h"
#include <thread>
#include <chrono>
#include <unordered_set>
#include <atomic>
#include <cmath>

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
    Backend::IViewportBackend* viewportBackend = GetTerrainViewportBackend(ctx);
    Backend::IBackend* renderBackend = GetTerrainRenderBackend(ctx);

    struct TerrainSplatDirtySnapshot {
        Texture* tex = nullptr;
        bool full = false;
        int min_x = 0;
        int min_y = 0;
        int max_x = -1;
        int max_y = -1;
    };

    std::vector<TerrainSplatDirtySnapshot> dirtySnapshots;
    for (auto& terrain : TerrainManager::getInstance().getTerrains()) {
        Texture* tex = terrain.splatMap.get();
        if (!tex || !tex->vulkan_dirty) continue;
        dirtySnapshots.push_back({
            tex,
            tex->vulkan_dirty_full,
            tex->vulkan_dirty_min_x,
            tex->vulkan_dirty_min_y,
            tex->vulkan_dirty_max_x,
            tex->vulkan_dirty_max_y
        });
    }

    auto restoreDirtySnapshots = [&]() {
        for (const auto& snapshot : dirtySnapshots) {
            if (!snapshot.tex) continue;
            snapshot.tex->vulkan_dirty = true;
            snapshot.tex->vulkan_dirty_full = snapshot.full;
            snapshot.tex->vulkan_dirty_min_x = snapshot.min_x;
            snapshot.tex->vulkan_dirty_min_y = snapshot.min_y;
            snapshot.tex->vulkan_dirty_max_x = snapshot.max_x;
            snapshot.tex->vulkan_dirty_max_y = snapshot.max_y;
        }
    };

    if (renderBackend) {
        ctx.renderer.updateBackendMaterials(ctx.scene, renderBackend);
    }
    if (viewportBackend &&
        reinterpret_cast<const void*>(viewportBackend) != reinterpret_cast<const void*>(renderBackend)) {
        restoreDirtySnapshots();
        ctx.renderer.updateBackendMaterials(ctx.scene, viewportBackend);
    }
    ResetTerrainBackendAccumulation(ctx);
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
                // If terrain has a generated flat mesh, select a throwaway representative
                // facade (face 0) to sync viewport selection — same convention other flat
                // (direct SoA) meshes use for their selection UI handle.
                if (t->flatMesh && ctx.selection.hasSelection() == false) {
                    auto rep = std::make_shared<Triangle>(t->flatMesh, 0);
                    ctx.selection.selectObject(rep, -1, rep->getNodeName());
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
        //
        // This used to be a per-frame std::find() over world.objects for each
        // terrain triangle — O(mesh_triangles * world.objects), redone every
        // single frame the Terrain panel is visible. On a scene with a large
        // terrain and a nontrivial object count (scatter/foliage instances etc.)
        // that pegged the UI thread continuously (reported as the UI "freezing"
        // once a Hydraulic Erosion node existed — same panel, bigger scene).
        // Build an O(1)-lookup set once per actual scene-geometry change instead
        // (g_scene_geometry_generation is bumped by exactly the operations that
        // could make a terrain stale — object add/remove/rebuild), and reuse it
        // across frames until the next real change.
        extern std::atomic<uint64_t> g_scene_geometry_generation;
        static uint64_t s_stale_check_generation = static_cast<uint64_t>(-1);
        static std::unordered_set<Hittable*> s_world_objects_set;
        const uint64_t current_generation = g_scene_geometry_generation.load(std::memory_order_acquire);
        if (current_generation != s_stale_check_generation) {
            s_world_objects_set.clear();
            s_world_objects_set.reserve(ctx.scene.world.objects.size());
            for (auto& obj : ctx.scene.world.objects) s_world_objects_set.insert(obj.get());
            s_stale_check_generation = current_generation;
        }

        std::vector<int> stale_ids;
        for (auto& tt : terrains) {
            bool found = tt.flatMesh && s_world_objects_set.count(tt.flatMesh.get()) > 0;
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
                         // Select a throwaway representative facade so the viewport selection matches the list
                         if (t.flatMesh) {
                             auto rep = std::make_shared<Triangle>(t.flatMesh, 0);
                             ctx.selection.selectObject(rep, -1, rep->getNodeName());
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
                    float layer_fr = 10.0f;
                    if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
                        layer_fr = ThemeManager::instance().current().style.frameRounding;
                    }
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, layer_fr);
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
                            drawPrincipledBSDFEditor(pMat.get(), matID, ctx, false);
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
            if (UIWidgets::PrimaryButton("Create Layer", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 0))) {
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
                
                // Header is always colored in active theme accent color
                ImGui::PushStyleColor(ImGuiCol_Text, ThemeManager::instance().current().colors.accent);
                std::string headerLabel = "   " + group.name + " (" + std::to_string(group.instances.size()) + ")###Header" + std::to_string(group.id);
                bool opened = ImGui::TreeNode(headerLabel.c_str());
                ImGui::PopStyleColor();
                
                if (opened) {
                    ImVec2 hp = ImGui::GetItemRectMin();
                    UIWidgets::DrawIcon(UIWidgets::IconType::Scene, ImVec2(hp.x + 10, hp.y + 2), 14, 0xFFBBBBBB);
                    
                    auto BeginSubSection = [](const char* label, UIWidgets::IconType icon, const ImVec4& color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f)) {
                        ImGui::Spacing();
                        ImGui::BeginGroup();
                        ImGui::Indent(6.0f);
                        
                        ImGui::TextColored(color, "   %s", label);
                        ImVec2 cp = ImGui::GetItemRectMin();
                        UIWidgets::DrawIcon(icon, ImVec2(cp.x + 4, cp.y + 1), 12, ImGui::ColorConvertFloat4ToU32(color));
                        ImGui::Spacing();
                    };

                    auto EndSubSection = []() {
                        ImGui::Unindent(6.0f);
                        ImGui::EndGroup();
                        
                        ImVec2 min = ImGui::GetItemRectMin();
                        ImVec2 max = ImGui::GetItemRectMax();
                        min.x -= 4.0f;
                        min.y -= 4.0f;
                        max.x += 4.0f;
                        max.y += 4.0f;
                        
                        ImGui::GetWindowDrawList()->AddRect(min, max, ImGui::GetColorU32(ImGuiCol_Border, 0.40f), 4.0f);
                        ImGui::Spacing();
                    };

                    ImGui::Spacing();
                    ImGui::BeginGroup();
                    ImGui::Indent(8.0f);
                    
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
                    BeginSubSection("Source Meshes", UIWidgets::IconType::Mesh, ImVec4(0.8f, 0.8f, 0.8f, 0.9f));
                    #if 0 // Legacy source picker retained temporarily for migration/reference.
                    ImGui::SameLine(); UIWidgets::HelpMarker("Select the 3D models to be used in this layer. If multiple models are added, they will be distributed randomly.");
                     if (UIWidgets::SecondaryButton("Add Selected Object", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 0))) {
                         if (ctx.selection.hasSelection()) {
                             std::string n = ctx.selection.selected.name;
                             // Flat (SoA) objects live in world.objects as TriangleMesh, not
                             // per-face Triangle facades — a Triangle-only scan found nothing for
                             // them, and multi-material imports split into several sibling
                             // TriangleMesh sharing this nodeName. Materialize every face of every
                             // matching sibling instead (mirrors the Scatter Brush panel fix).
                             std::vector<std::shared_ptr<TriangleMesh>> flatMeshes;
                             std::vector<std::shared_ptr<Triangle>> legacyTriangles;
                             std::unordered_set<TriangleMesh*> seenMeshes;
                             for (auto& obj : ctx.scene.world.objects) {
                                 if (auto tmesh = std::dynamic_pointer_cast<TriangleMesh>(obj)) {
                                     if (tmesh->nodeName != n || !tmesh->geometry) continue;
                                     if (!seenMeshes.insert(tmesh.get()).second) continue;
                                     flatMeshes.push_back(tmesh);
                                 } else if (auto tri = std::dynamic_pointer_cast<Triangle>(obj)) {
                                     if (tri->getNodeName() == n) legacyTriangles.push_back(tri);
                                 }
                             }
                             if (!flatMeshes.empty()) {
                                 group.sources.emplace_back(n, flatMeshes);
                             } else if (!legacyTriangles.empty()) {
                                 group.sources.emplace_back(n, legacyTriangles);
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
                                 // tris_list (mesh_cache) holds only ONE representative single-face
                                 // facade per sibling TriangleMesh (the UI selection handle) —
                                 // materialize the full geometry of every sibling instead so the
                                 // foliage source is the whole object, not one triangle per material.
                                 std::vector<std::shared_ptr<TriangleMesh>> flatMeshes;
                                 std::vector<std::shared_ptr<Triangle>> legacyTriangles;
                                 std::unordered_set<TriangleMesh*> seenMeshes;
                                 for (const auto& pair : tris_list) {
                                     TriangleMesh* pm = pair.second ? pair.second->parentMesh.get() : nullptr;
                                     if (pm && pm->geometry) {
                                         if (!seenMeshes.insert(pm).second) continue;
                                         flatMeshes.push_back(pair.second->parentMesh);
                                     } else if (pair.second) {
                                         legacyTriangles.push_back(pair.second);
                                     }
                                 }

                                 if (!flatMeshes.empty()) {
                                     group.sources.emplace_back(name, flatMeshes);
                                 } else if (!legacyTriangles.empty()) {
                                     group.sources.emplace_back(name, legacyTriangles);
                                 }
                                 ImGui::CloseCurrentPopup();
                             }
                         }
                         ImGui::EndChild();
                         ImGui::EndPopup();
                     }

                      // List Sources in a clean aligned table (handles long name clipping automatically)
                      if (!group.sources.empty() && ImGui::BeginTable("##sources_table", 3, ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit)) {
                          ImGui::TableSetupColumn("##name", ImGuiTableColumnFlags_WidthStretch);
                          ImGui::TableSetupColumn("##edit", ImGuiTableColumnFlags_WidthFixed, 42.0f);
                          ImGui::TableSetupColumn("##delete", ImGuiTableColumnFlags_WidthFixed, 25.0f);

                          for (int s_i = 0; s_i < (int)group.sources.size(); s_i++) {
                              ImGui::PushID(s_i);
                              auto& src = group.sources[s_i];

                              ImGui::TableNextRow();
                              ImGui::TableNextColumn();
                              ImGui::Text("%s (W: %.1f)", src.name.c_str(), src.weight);

                              ImGui::TableNextColumn();
                              if (ImGui::SmallButton("Edit")) ImGui::OpenPopup("SrcEdit");

                              if (ImGui::BeginPopup("SrcEdit")) {
                                  ImGui::PushItemWidth(160.0f);
                                  ImGui::DragFloat("Weight", &src.weight, 0.1f);
                                  ImGui::DragFloatRange2("Scale", &src.settings.scale_min, &src.settings.scale_max, 0.01f, 0.001f, 1000.0f);
                                  ImGui::DragFloatRange2("Y-Off", &src.settings.y_offset_min, &src.settings.y_offset_max, 0.01f);
                                  ImGui::PopItemWidth();
                                  ImGui::EndPopup();
                              }

                              ImGui::TableNextColumn();
                              if (ImGui::SmallButton("X")) {
                                  group.sources.erase(group.sources.begin() + s_i);
                                  s_i--;
                              }

                              ImGui::PopID();
                          }
                          ImGui::EndTable();
                      }

                    #endif

                    ImGui::SameLine();
                    UIWidgets::HelpMarker(
                        "Add sources from biome recommendations, the Asset Library, or scene objects. "
                        "Terrain UI and Foliage nodes edit the same shared source list.");

                    auto hasSceneSource = [&](const std::string& sourceName) {
                        return std::any_of(group.sources.begin(), group.sources.end(),
                            [&](const ScatterSource& source) {
                                return source.asset_relative_path.empty() && source.name == sourceName;
                            });
                    };
                    auto hasLibrarySource = [&](const std::string& relativePath) {
                        return std::any_of(group.sources.begin(), group.sources.end(),
                            [&](const ScatterSource& source) {
                                return source.asset_relative_path == relativePath;
                            });
                    };
                    auto addSceneSource = [&](const std::string& sourceName) {
                        if (sourceName.empty() || hasSceneSource(sourceName)) return;
                        std::vector<std::shared_ptr<TriangleMesh>> flatMeshes;
                        std::vector<std::shared_ptr<Triangle>> legacyTriangles;
                        std::unordered_set<TriangleMesh*> seenMeshes;
                        for (auto& object : ctx.scene.world.objects) {
                            if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object)) {
                                if (mesh->nodeName != sourceName || !mesh->geometry) continue;
                                if (seenMeshes.insert(mesh.get()).second) flatMeshes.push_back(mesh);
                            } else if (auto triangle = std::dynamic_pointer_cast<Triangle>(object)) {
                                if (triangle->getNodeName() == sourceName) legacyTriangles.push_back(triangle);
                            }
                        }
                        if (!flatMeshes.empty()) group.sources.emplace_back(sourceName, flatMeshes);
                        else if (!legacyTriangles.empty()) group.sources.emplace_back(sourceName, legacyTriangles);
                        else return;
                        group.gpu_dirty = true;
                    };
                    auto findAssetRecord = [&](const std::string& relativePath) -> const AssetRecord* {
                        for (const auto& asset : FoliageAssets::catalog(false).getAssets()) {
                            if (asset.relative_entry_path.generic_string() == relativePath) return &asset;
                        }
                        return nullptr;
                    };
                    auto showAssetPreview = [&](const AssetRecord& asset) {
                        if (!ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) return;
                        ImGui::BeginTooltip();
                        ImGui::TextUnformatted(asset.name.c_str());
                        ImGui::TextDisabled("%s / %s", asset.category.c_str(), asset.subcategory.c_str());
                        SDL_Texture* previewTexture = nullptr;
                        int previewWidth = 0;
                        int previewHeight = 0;
                        if (asset.has_preview && ensureAssetBrowserThumbnailTexture(
                                ctx, asset.preview_path, previewTexture, previewWidth, previewHeight) &&
                            previewTexture && previewWidth > 0 && previewHeight > 0) {
                            const float maxPreview = 260.0f;
                            const float previewScale = (std::min)(maxPreview / previewWidth,
                                                                  maxPreview / previewHeight);
                            ImGui::Image((ImTextureID)previewTexture,
                                ImVec2(previewWidth * previewScale, previewHeight * previewScale));
                        } else {
                            ImGui::TextDisabled("No preview image");
                        }
                        ImGui::EndTooltip();
                    };
                    auto addLibrarySource = [&](const AssetRecord& asset) {
                        const std::string relativePath = asset.relative_entry_path.generic_string();
                        if (relativePath.empty() || hasLibrarySource(relativePath)) return;
                        ScatterSource source;
                        std::string error;
                        if (!FoliageAssets::loadScatterSource(relativePath, asset.name, 1.0f, source, &error)) {
                            if (!error.empty()) SCENE_LOG_WARN(error);
                            return;
                        }
                        const std::string layerType = group.name + " " +
                            group.brush_settings.density_mask_attribute;
                        FoliageAssets::configurePlacement(source,
                            FoliageAssets::defaultTargetHeight(layerType), 0.15f, false, 0.0f);
                        group.brush_settings.use_global_settings = false;
                        group.sources.push_back(std::move(source));
                        group.gpu_dirty = true;
                    };

                    if (UIWidgets::SecondaryButton("Add Source...",
                            ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 0))) {
                        ImGui::OpenPopup("FoliageSourcePicker");
                    }
                    ImGui::SameLine();
                    if (UIWidgets::SecondaryButton("Add Selected",
                            ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 0)) &&
                        ctx.selection.hasSelection()) {
                        addSceneSource(ctx.selection.selected.name);
                    }

                    ImGui::SetNextWindowSize(ImVec2(520.0f, 430.0f), ImGuiCond_Appearing);
                    if (ImGui::BeginPopup("FoliageSourcePicker")) {
                        static char pickerFilter[96] = "";
                        ImGui::SetNextItemWidth(-1.0f);
                        ImGui::InputTextWithHint("##FoliagePickerFilter", "Search sources...",
                                                 pickerFilter, sizeof(pickerFilter));
                        if (ImGui::BeginTabBar("##FoliageSourceTabs")) {
                            auto drawAssetCandidates = [&](const std::vector<const AssetRecord*>& candidates) {
                                ImGui::BeginChild("##AssetCandidates", ImVec2(0, 330.0f), true);
                                for (const AssetRecord* asset : candidates) {
                                    if (!asset) continue;
                                    const std::string relativePath = asset->relative_entry_path.generic_string();
                                    const bool alreadyAdded = hasLibrarySource(relativePath);
                                    const std::string label = alreadyAdded
                                        ? "[Added] " + asset->name : asset->name;
                                    if (ImGui::Selectable(label.c_str(), alreadyAdded) && !alreadyAdded) {
                                        addLibrarySource(*asset);
                                    }
                                    showAssetPreview(*asset);
                                }
                                ImGui::EndChild();
                            };

                            if (ImGui::BeginTabItem("Recommended")) {
                                const std::string layerType = group.name + " " +
                                    group.brush_settings.density_mask_attribute;
                                drawAssetCandidates(FoliageAssets::recommendedAssets(
                                    layerType, "Auto", pickerFilter));
                                ImGui::EndTabItem();
                            }
                            if (ImGui::BeginTabItem("Asset Library")) {
                                drawAssetCandidates(FoliageAssets::recommendedAssets(
                                    "", "Auto", pickerFilter));
                                ImGui::EndTabItem();
                            }
                            if (ImGui::BeginTabItem("Scene Objects")) {
                                if (mesh_cache.empty()) rebuildMeshCache(ctx.scene.world.objects);
                                ImGui::BeginChild("##SceneCandidates", ImVec2(0, 330.0f), true);
                                for (const auto& [sourceName, triangles] : mesh_cache) {
                                    (void)triangles;
                                    if (sourceName.empty() || sourceName.find("_inst_") == 0) continue;
                                    if (pickerFilter[0] != '\0' &&
                                        sourceName.find(pickerFilter) == std::string::npos) continue;
                                    const bool alreadyAdded = hasSceneSource(sourceName);
                                    const std::string label = alreadyAdded
                                        ? "[Added] " + sourceName : sourceName;
                                    if (ImGui::Selectable(label.c_str(), alreadyAdded) && !alreadyAdded) {
                                        addSceneSource(sourceName);
                                    }
                                }
                                ImGui::EndChild();
                                ImGui::EndTabItem();
                            }
                            ImGui::EndTabBar();
                        }
                        ImGui::EndPopup();
                    }

                    static char addedSourceFilter[80] = "";
                    ImGui::SetNextItemWidth(-1.0f);
                    ImGui::InputTextWithHint("##AddedSourceFilter", "Filter added sources...",
                                             addedSourceFilter, sizeof(addedSourceFilter));
                    int removeSourceIndex = -1;

                    auto drawSourceCategory = [&](const char* categoryLabel,
                                                  bool librarySources,
                                                  int sourceCount) {
                        if (sourceCount <= 0) return;
                        const std::string header = std::string(categoryLabel) + " (" +
                            std::to_string(sourceCount) + ")";
                        if (!ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) return;
                        ImGui::PushID(categoryLabel);
                        if (ImGui::BeginTable("##SourceCategory", 5,
                                ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_SizingFixedFit |
                                ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH)) {
                            ImGui::TableSetupColumn("##thumb", ImGuiTableColumnFlags_WidthFixed, 38.0f);
                            ImGui::TableSetupColumn("Source", ImGuiTableColumnFlags_WidthStretch);
                            ImGui::TableSetupColumn("Weight", ImGuiTableColumnFlags_WidthFixed, 64.0f);
                            ImGui::TableSetupColumn("##edit", ImGuiTableColumnFlags_WidthFixed, 42.0f);
                            ImGui::TableSetupColumn("##delete", ImGuiTableColumnFlags_WidthFixed, 25.0f);
                            for (int sourceIndex = 0;
                                 sourceIndex < static_cast<int>(group.sources.size()); ++sourceIndex) {
                                auto& source = group.sources[static_cast<size_t>(sourceIndex)];
                                const bool isLibrary = !source.asset_relative_path.empty();
                                if (isLibrary != librarySources) continue;
                                if (addedSourceFilter[0] != '\0' &&
                                    source.name.find(addedSourceFilter) == std::string::npos) continue;
                                const AssetRecord* asset = isLibrary
                                    ? findAssetRecord(source.asset_relative_path) : nullptr;
                                ImGui::PushID(sourceIndex);
                                ImGui::TableNextRow();
                                ImGui::TableNextColumn();
                                SDL_Texture* thumbnail = nullptr;
                                int thumbnailWidth = 0;
                                int thumbnailHeight = 0;
                                if (asset && asset->has_preview && ensureAssetBrowserThumbnailTexture(
                                        ctx, asset->preview_path, thumbnail,
                                        thumbnailWidth, thumbnailHeight) && thumbnail) {
                                    ImGui::Image((ImTextureID)thumbnail, ImVec2(32.0f, 32.0f));
                                    showAssetPreview(*asset);
                                } else {
                                    ImGui::TextDisabled(isLibrary ? "LIB" : "SCN");
                                }
                                ImGui::TableNextColumn();
                                ImGui::TextUnformatted(source.name.c_str());
                                if (asset) showAssetPreview(*asset);
                                ImGui::TableNextColumn();
                                ImGui::SetNextItemWidth(58.0f);
                                if (ImGui::DragFloat("##Weight", &source.weight,
                                                    0.05f, 0.0f, 100.0f, "%.2f")) {
                                    group.gpu_dirty = true;
                                }
                                ImGui::TableNextColumn();
                                if (ImGui::SmallButton("Edit")) ImGui::OpenPopup("SrcEdit");
                                if (ImGui::BeginPopup("SrcEdit")) {
                                    ImGui::PushItemWidth(180.0f);
                                    float targetHeight = 0.0f;
                                    float variation = 0.15f;
                                    if (source.has_local_bbox) {
                                        const float meshHeight = source.local_bbox.max.y -
                                            source.local_bbox.min.y;
                                        const float centerScale = (source.settings.scale_min +
                                            source.settings.scale_max) * 0.5f;
                                        targetHeight = meshHeight * centerScale;
                                        const float scaleSum = source.settings.scale_min +
                                            source.settings.scale_max;
                                        if (scaleSum > 1e-5f) {
                                            variation = (source.settings.scale_max -
                                                source.settings.scale_min) / scaleSum;
                                        }
                                    }
                                    bool placementChanged = ImGui::DragFloat("Target Height",
                                        &targetHeight, 0.1f, 0.01f, 10000.0f, "%.2f m");
                                    int variationPercent = static_cast<int>(
                                        std::round(variation * 100.0f));
                                    if (ImGui::SliderInt("Variation", &variationPercent,
                                                         0, 95, "%d%%")) {
                                        variation = variationPercent * 0.01f;
                                        placementChanged = true;
                                    }
                                    bool alignToNormal = source.settings.align_to_normal;
                                    if (ImGui::Checkbox("Follow Slope", &alignToNormal)) {
                                        placementChanged = true;
                                    }
                                    float normalInfluence = source.settings.normal_influence;
                                    if (alignToNormal && ImGui::SliderFloat("Normal Influence",
                                            &normalInfluence, 0.0f, 1.0f, "%.2f")) {
                                        placementChanged = true;
                                    }
                                    if (placementChanged) {
                                        FoliageAssets::configurePlacement(source, targetHeight,
                                            variation, alignToNormal, normalInfluence);
                                        group.brush_settings.use_global_settings = false;
                                        group.gpu_dirty = true;
                                    }
                                    if (ImGui::DragFloatRange2("Y Offset",
                                            &source.settings.y_offset_min,
                                            &source.settings.y_offset_max, 0.01f)) {
                                        group.gpu_dirty = true;
                                    }
                                    ImGui::PopItemWidth();
                                    ImGui::EndPopup();
                                }
                                ImGui::TableNextColumn();
                                if (ImGui::SmallButton("X")) removeSourceIndex = sourceIndex;
                                ImGui::PopID();
                            }
                            ImGui::EndTable();
                        }
                        ImGui::PopID();
                    };

                    int librarySourceCount = 0;
                    int sceneSourceCount = 0;
                    for (const auto& source : group.sources) {
                        if (source.asset_relative_path.empty()) ++sceneSourceCount;
                        else ++librarySourceCount;
                    }
                    drawSourceCategory("Asset Library", true, librarySourceCount);
                    drawSourceCategory("Scene Objects", false, sceneSourceCount);
                    if (group.sources.empty()) ImGui::TextDisabled("No sources added");
                    if (removeSourceIndex >= 0 &&
                        removeSourceIndex < static_cast<int>(group.sources.size())) {
                        group.sources.erase(group.sources.begin() + removeSourceIndex);
                        group.clearInstances();
                        group.source_bvh.reset();
                        group.source_triangles_ptr.reset();
                        group.blas_id = -1;
                        group.gpu_dirty = true;
                    }

                      EndSubSection();

                      // ─── TARGET SURFACE ────────────────────────────────────────────────
                      BeginSubSection("Target Surface", UIWidgets::IconType::World, ImVec4(0.5f, 0.8f, 1.0f, 0.9f));
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
                     EndSubSection();

                     // ACTIONS
                     BeginSubSection("Scatter Settings", UIWidgets::IconType::Settings, ImVec4(0.8f, 0.8f, 0.8f, 0.9f));
                     ImGui::PushItemWidth(160.0f);
                     ImGui::DragInt("Target Count", &group.brush_settings.target_count, 100, 1, 10000000);
                     if (ImGui::IsItemHovered()) ImGui::SetTooltip("Total number of instances to spread across the whole terrain.");
                     ImGui::InputInt("Seed (Randomness)", &group.brush_settings.seed);
                     ImGui::DragFloat("Min. Distance", &group.brush_settings.min_distance, 0.1f, 0.0f, 50.0f);
                     if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum distance between instances (m). Prevents overlap.");
                     ImGui::PopItemWidth();
                     EndSubSection();
                     
                     // Helper helper for brush state
                     bool is_active_group = (foliage_brush.active_group_id == group.id);
                     bool is_painting = foliage_brush.enabled && is_active_group && foliage_brush.mode == 0;
                     bool is_erasing = foliage_brush.enabled && is_active_group && foliage_brush.mode == 1;

                       // BRUSH TOOLS
                       BeginSubSection("Brush Tools", UIWidgets::IconType::Brush, ImVec4(0.4f, 0.9f, 0.5f, 0.9f));

                       // Paint Button
                       bool active_paint = is_painting;
                       if (UIWidgets::IconActionButton("paint_add", UIWidgets::IconType::PaintTool, "", active_paint, ImVec4(0.3f, 0.9f, 0.4f, 1.0f), ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 24.0f), "Paint Tool (Add Instances)")) {
                           if (is_painting) {
                               foliage_brush.enabled = false;
                               foliage_brush.active_group_id = -1;
                           } else {
                               foliage_brush.enabled = true;
                               foliage_brush.active_group_id = group.id;
                               foliage_brush.mode = 0; // ADD
                           }
                       }
                       
                       ImGui::SameLine();
                       
                       // Erase Button
                       bool active_erase = is_erasing;
                       if (UIWidgets::IconActionButton("paint_remove", UIWidgets::IconType::EraseTool, "", active_erase, ImVec4(0.9f, 0.3f, 0.3f, 1.0f), ImVec2(UIWidgets::GetInspectorActionWidth() * 0.48f, 24.0f), "Erase Tool (Remove Instances)")) {
                           if (is_erasing) {
                               foliage_brush.enabled = false;
                               foliage_brush.active_group_id = -1;
                           } else {
                               foliage_brush.enabled = true;
                               foliage_brush.active_group_id = group.id;
                               foliage_brush.mode = 1; // REMOVE
                           }
                       }

                        // Show Brush Settings if active
                        if (is_active_group && foliage_brush.enabled) {
                           ImGui::PushItemWidth(160.0f);
                           ImGui::DragFloat("radius##br", &foliage_brush.radius, 0.1f, 0.1f, 100.0f, "%.1f m");
                           if (is_painting) {
                               ImGui::DragInt("density##br", &foliage_brush.density, 1, 1, 20);
                           }
                           ImGui::PopItemWidth();
                           ImGui::Checkbox("Lazy Update", &foliage_brush.lazy_update);
                           if (ImGui::IsItemHovered()) ImGui::SetTooltip("Update scene only on mouse release (Better performance for large terrains)");
                        }
                        EndSubSection();

                      ImGui::Spacing();
                      BeginSubSection("Placement Rules", UIWidgets::IconType::System, ImVec4(0.4f, 0.7f, 1.0f, 0.9f));
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

                        // Named terrain fields are produced once by Terrain Analysis and
                        // reused by every foliage layer. Density and scale intentionally
                        // remain independent so a biome can occupy valleys while plant
                        // size follows wetness (or any other published field).
                        {
                            std::vector<std::string> terrainFields;
                            terrainFields.reserve(t->analysisFields.size());
                            for (const auto& [fieldName, fieldData] : t->analysisFields) {
                                if (fieldData && fieldData->size() == t->heightmap.data.size())
                                    terrainFields.push_back(fieldName);
                            }
                            std::sort(terrainFields.begin(), terrainFields.end());
                            auto fieldPicker = [&](const char* label, std::string& value) {
                                ImGui::SetNextItemWidth(160.0f);
                                const char* preview = value.empty() ? "<none>" : value.c_str();
                                if (ImGui::BeginCombo(label, preview)) {
                                    if (ImGui::Selectable("<none>", value.empty())) value.clear();
                                    for (const auto& fieldName : terrainFields) {
                                        const bool selected = value == fieldName;
                                        if (ImGui::Selectable(fieldName.c_str(), selected)) value = fieldName;
                                        if (selected) ImGui::SetItemDefaultFocus();
                                    }
                                    if (terrainFields.empty())
                                        ImGui::TextDisabled("Connect Terrain Analysis to Terrain Fields Output");
                                    ImGui::EndCombo();
                                }
                            };
                            fieldPicker("Density Field", group.brush_settings.density_mask_attribute);
                            fieldPicker("Exclusion Field", group.brush_settings.exclusion_mask_attribute);
                            if (group.brush_settings.exclusion_channel == -1 &&
                                !group.brush_settings.exclusion_mask_attribute.empty()) {
                                ImGui::SetNextItemWidth(160.0f);
                                ImGui::SliderFloat("Exclusion Threshold##Field", &group.brush_settings.exclusion_threshold,
                                                   0.0f, 1.0f, "%.2f");
                            }
                            fieldPicker("Scale Field", group.brush_settings.scale_mask_attribute);
                            if (!group.brush_settings.scale_mask_attribute.empty()) {
                                ImGui::SetNextItemWidth(160.0f);
                                ImGui::SliderFloat("Scale Field Influence", &group.brush_settings.scale_mask_influence, 0.0f, 1.0f);
                            }
                        }
                        } // if (t) - terrain splat/exclusion end
                        EndSubSection();

                        if (UIWidgets::PrimaryButton("Scatter", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.31f, 0))) {
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
                            // Shared with the Scatter panel's own "Scatter (Fill)" button
                            // (scene_ui_scatter.cpp) — one implementation, so Placement Rules AND
                            // the Faz 8b Field density/scale masks behave identically everywhere.
                            const int spawned = group.scatterFillMesh(surf_tris);
                            SCENE_LOG_INFO("Mesh scatter: " + std::to_string(spawned) + "/" +
                                           std::to_string(group.brush_settings.target_count) + " instances on '" + group.target_node_name + "'.");
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
                        auto sampleNamedTerrainField = [&](const std::string& fieldName, float u, float v) -> float {
                            if (fieldName.empty()) return 1.0f;
                            const auto fieldIt = t->analysisFields.find(fieldName);
                            if (fieldIt == t->analysisFields.end() || !fieldIt->second ||
                                fieldIt->second->size() != t->heightmap.data.size()) return -1.0f;
                            const auto& field = *fieldIt->second;
                            const float gx = u * (t->heightmap.width - 1);
                            const float gz = v * (t->heightmap.height - 1);
                            const int ix0 = static_cast<int>(gx), iz0 = static_cast<int>(gz);
                            const int ix1 = std::min(ix0 + 1, t->heightmap.width - 1);
                            const int iz1 = std::min(iz0 + 1, t->heightmap.height - 1);
                            const float ax = gx - ix0, az = gz - iz0;
                            auto sample = [&](int x, int z) { return field[static_cast<size_t>(z) * t->heightmap.width + x]; };
                            const float v0 = sample(ix0, iz0) * (1.0f - ax) + sample(ix1, iz0) * ax;
                            const float v1 = sample(ix0, iz1) * (1.0f - ax) + sample(ix1, iz1) * ax;
                            return std::clamp(v0 * (1.0f - az) + v1 * az, 0.0f, 1.0f);
                        };

                        while (spawned < count && attempts < max_attempts) {
                            attempts++;
                            
                            float r1 = dist(rng);
                            float r2 = dist(rng);

                            if (!group.brush_settings.density_mask_attribute.empty()) {
                                const float densityField = sampleNamedTerrainField(
                                    group.brush_settings.density_mask_attribute, r1, r2);
                                if (densityField >= 0.0f && dist(rng) > densityField) continue;
                            }
                            if (!group.brush_settings.exclusion_mask_attribute.empty()) {
                                const float exclusionField = sampleNamedTerrainField(
                                    group.brush_settings.exclusion_mask_attribute, r1, r2);
                                if (exclusionField >= group.brush_settings.exclusion_threshold) continue;
                            }
                            
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
                            if (!group.brush_settings.scale_mask_attribute.empty() &&
                                group.brush_settings.scale_mask_influence > 0.0f) {
                                const float fieldValue = sampleNamedTerrainField(
                                    group.brush_settings.scale_mask_attribute, r1, r2);
                                if (fieldValue >= 0.0f) {
                                    const float factor = 1.0f - group.brush_settings.scale_mask_influence * (1.0f - fieldValue);
                                    inst.scale = inst.scale * factor;
                                }
                            }
                            group.addInstance(inst);
                            spawned++;
                        }
                        } // if (t) end - terrain scatter body
                        } // terrain scatter else end
                        SceneUI::syncInstancesToScene(ctx, group, false);
                        // Trigger appropriate rebuilds depending on active render backend
                        const bool hasVulkanViewportPath = TerrainRenderBackendIsVulkan(ctx) || (g_viewport_backend != nullptr);
                        g_viewport_raster_rebuild_pending = true;
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
                     if (UIWidgets::SecondaryButton("Clear", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.31f, 0))) {
                         group.clearInstances();
                         SceneUI::syncInstancesToScene(ctx, group, true);
                         const bool hasVulkanViewportPath_clear = TerrainRenderBackendIsVulkan(ctx) || (g_viewport_backend != nullptr);
                         g_viewport_raster_rebuild_pending = true;
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
                     if (UIWidgets::DangerButton("Delete", ImVec2(UIWidgets::GetInspectorActionWidth() * 0.31f, 0))) {
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
                    
                    // End of Card Group and border drawing
                    ImGui::Unindent(8.0f);
                    ImGui::EndGroup();
                    
                    ImVec2 min = ImGui::GetItemRectMin();
                    ImVec2 max = ImGui::GetItemRectMax();
                    min.x -= 6.0f;
                    min.y -= 6.0f;
                    max.x += 6.0f;
                    max.y += 6.0f;

                    const auto& theme = ThemeManager::instance().current();
                    ImU32 borderColor = ImGui::ColorConvertFloat4ToU32(ImVec4(theme.colors.accent.x, theme.colors.accent.y, theme.colors.accent.z, 0.40f));

                    ImGui::GetWindowDrawList()->AddRect(min, max, borderColor, 4.0f);
                    ImGui::Spacing();
                    
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
                    g_viewport_raster_rebuild_pending = true;
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
                    g_vulkan_rebuild_pending = true;
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
                                g_vulkan_rebuild_pending = true;
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

    // Anchored stroke state variables
    static bool terrain_stroke_active = false;
    static Vec3 terrain_stroke_anchor_world(0.0f, 0.0f, 0.0f);
    static std::vector<float> terrain_stroke_backup_heightmap;
    static float terrain_stroke_current_radius = 0.0f;
    static float terrain_stroke_current_rotation = 0.0f;

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

        // Terrain triangles are registered in the raster backend with the node name
        // "<terrain->name>_Chunk" (set in TerrainManager/gridToFlatMesh), so we must
        // use that name when looking up the raster mesh — not bare terrain->name.
        const std::string terrainRasterNodeName = terrain->name + "_Chunk";

        // Flat (SoA) mesh: refit the raster vertex buffer straight from the mesh's
        // DNA SoA — same dirty-range-free, per-dab-realtime path regular mesh sculpt
        // already uses (see scene_ui_mesh_overlay.cpp's syncMeshRasterViewport /
        // updateRasterMeshFromMeshSoA). Falls through to a full buildRasterGeometry
        // only when the raster mesh isn't registered yet or topology changed (vertex
        // count mismatch) — buildRasterGeometry() itself early-outs on an unchanged
        // scene-geometry generation, which is fine here since a real topology change
        // already goes through rebuildTerrainMesh() (bumps that generation).
        if (terrain->flatMesh && vkBackend->updateRasterMeshFromMeshSoA(terrainRasterNodeName, terrain->flatMesh.get())) {
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
            SyncTerrainMaterialState(ctx);
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
                    // Only fall back to a full rebuild if the incremental BLAS refit above
                    // (renderedBackendHandled, from vkRtBackend->updateTerrainBLASPartial) didn't
                    // already handle it — this used to fire unconditionally, forcing a full
                    // rebuildAccelerationStructure()+updateGeometry() on every stroke commit
                    // (mouse-up) even when the cheap refit had already synced the terrain.
                    if (!renderedBackendHandled) {
                        g_vulkan_rebuild_pending = true;
                    }
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
        terrain_stroke_active = false;
        terrain_stroke_backup_heightmap.clear();
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
                
                Vec3 previewCenter = hitPoint;
                float previewRadius = terrain_brush.radius;
                float previewRotation = terrain_brush.stamp_rotation;
                
                if (terrain_brush.mode == 4 && terrain_brush.stroke_method == Paint::StrokeMethod::Anchored && terrain_stroke_active) {
                    previewCenter = terrain_stroke_anchor_world;
                    previewRadius = terrain_stroke_current_radius;
                    previewRotation = terrain_stroke_current_rotation;
                }

                for (int i = 0; i < segments; i++) {
                    float theta = (float)i / segments * 6.28318f;
                    float theta2 = (float)(i + 1) / segments * 6.28318f;
                    
                    Vec3 p1 = previewCenter + Vec3(cos(theta) * previewRadius, 0.1f, sin(theta) * previewRadius);
                    Vec3 p2 = previewCenter + Vec3(cos(theta2) * previewRadius, 0.1f, sin(theta2) * previewRadius);
                    
                    ImVec2 s1 = Project(p1);
                    ImVec2 s2 = Project(p2);
                    
                    if (s1.x > -50 && s1.x < win_w + 50) {
                         dl->AddLine(s1, s2, col, 2.0f);
                    }
                }

                for (int ring = 0; ring < 2; ++ring) {
                    float ringScale = (ring == 0) ? 0.5f : std::pow(0.5f, 1.0f / std::max(terrain_brush.curve, 0.25f));
                    float ringRadius = previewRadius * ringScale;
                    for (int i = 0; i < segments; i++) {
                        float theta = (float)i / segments * 6.28318f;
                        float theta2 = (float)(i + 1) / segments * 6.28318f;
                        Vec3 p1 = previewCenter + Vec3(cos(theta) * ringRadius, 0.12f, sin(theta) * ringRadius);
                        Vec3 p2 = previewCenter + Vec3(cos(theta2) * ringRadius, 0.12f, sin(theta2) * ringRadius);
                        ImVec2 s1 = Project(p1);
                        ImVec2 s2 = Project(p2);
                        if (s1.x > -50 && s1.x < win_w + 50) {
                            dl->AddLine(s1, s2, innerCol, 1.0f);
                        }
                    }
                }

                if (terrain_brush.mode == 4) {
                    float rad = previewRotation * 3.14159f / 180.0f;
                    float c = cosf(rad);
                    float s = sinf(rad);
                    Vec3 corners[4] = {
                        Vec3(-previewRadius, 0.15f, -previewRadius),
                        Vec3( previewRadius, 0.15f, -previewRadius),
                        Vec3( previewRadius, 0.15f,  previewRadius),
                        Vec3(-previewRadius, 0.15f,  previewRadius)
                    };
                    ImVec2 screenCorners[4];
                    for (int i = 0; i < 4; ++i) {
                        Vec3 local = corners[i];
                        Vec3 rotated(local.x * c - local.z * s, local.y, local.x * s + local.z * c);
                        screenCorners[i] = Project(previewCenter + rotated);
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
                     if (hasVulkanViewportPath) {
                         pending_vulkan_material_sync = true;
                         float now = (float)ImGui::GetTime();
                         const bool renderedVulkanRtActive =
                             TerrainRenderBackendIsVulkan(ctx) &&
                             dynamic_cast<Backend::VulkanBackendAdapter*>(GetTerrainRenderBackend(ctx)) != nullptr;
                         if (renderedVulkanRtActive || now - last_vulkan_paint_sync_time > 0.08f) {
                             SyncTerrainMaterialState(ctx);
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
                     if (terrain_brush.mode == 4 && terrain_brush.stroke_method == Paint::StrokeMethod::Anchored) {
                         if (!terrain_stroke_active) {
                             terrain_stroke_anchor_world = hitPoint;
                             terrain_stroke_backup_heightmap = terrain->heightmap.data;
                             terrain_stroke_active = true;
                             terrain_stroke_current_radius = 0.0f;
                             terrain_stroke_current_rotation = terrain_brush.stamp_rotation;
                         } else {
                             Vec3 drag_vector = hitPoint - terrain_stroke_anchor_world;
                             float drag_dist = sqrtf(drag_vector.x * drag_vector.x + drag_vector.z * drag_vector.z);
                             float drag_angle_rad = atan2f(drag_vector.z, drag_vector.x);
                             float drag_angle_deg = drag_angle_rad * 180.0f / 3.14159265f;
                             terrain_stroke_current_radius = drag_dist;
                             terrain_stroke_current_rotation = terrain_brush.stamp_rotation - drag_angle_deg;
                         }
                         
                         // Restore heightmap from backup before applying the new absolute state
                         terrain->heightmap.data = terrain_stroke_backup_heightmap;
                         
                         TerrainManager::getInstance().sculpt(
                             terrain, 
                             terrain_stroke_anchor_world, 
                             terrain_brush.mode, 
                             terrain_stroke_current_radius, 
                             terrain_brush.strength, 
                             1.0f, // dt = 1.0f for absolute strength
                             terrain_brush.curve,
                             targetH,
                             terrain_brush.stamp_texture,
                             terrain_stroke_current_rotation,
                             false
                         );
                     }
                     else {
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
                     }
                     
                     ResetTerrainBackendAccumulation(ctx);
                     ctx.renderer.resetCPUAccumulation();
                     
                     Backend::IViewportBackend* liveRasterViewportBackend = g_viewport_backend.get();
                     if (!liveRasterViewportBackend) {
                         liveRasterViewportBackend = dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr);
                     }

                     bool liveViewportUpdated = false;
                     bool liveViewportIsInteractiveRaster = false;
                     if (liveRasterViewportBackend) {
                         const Backend::ViewportMode viewportMode = liveRasterViewportBackend->getViewportMode();
                         const bool interactiveRasterMode =
                             viewportMode == Backend::ViewportMode::Solid ||
                             viewportMode == Backend::ViewportMode::Matcap ||
                             viewportMode == Backend::ViewportMode::MaterialPreview;
                         liveViewportIsInteractiveRaster = interactiveRasterMode;
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
                         // Throttle the synchronous BLAS+TLAS refit to a capped rate (same 0.08s /
                         // ~12Hz the non-active-render branch below already uses) instead of doing it
                         // on every UI frame during a drag (up to 100+ Hz). BLAS+TLAS update is a
                         // real GPU-synchronous cost proportional to terrain size, and re-doing it
                         // (plus resetAccumulation) far faster than a human can perceive is what made
                         // Vulkan RT feel like it was rebuilding on every single brush dab. The final
                         // state is always pushed on stroke commit (mouse-up) regardless.
                         float now = (float)ImGui::GetTime();
                         if (now - last_live_geometry_sync_time > 0.08f) {
                             if (ctx.optix_gpu_ptr && g_hasOptix && ctx.render_settings.use_optix) {
                                 // OptiX rendered viewport: push terrain BLAS updates per throttle tick.
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
                             last_live_geometry_sync_time = now;
                         } else {
                             // Skipped this frame purely due to the throttle — not stale, just not due
                             // yet — so don't request a full rebuild for it.
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
                     // Only request a raster rebuild when a raster (Solid/Matcap/MaterialPreview)
                     // view is actually visible. When the Rendered (RT) viewport is the active one
                     // and there's no separate dedicated raster panel, liveRasterViewportBackend
                     // aliases the SAME RT backend and updateRasterMeshFromMeshSoA() legitimately
                     // has nothing to do there — setting this flag anyway made every sculpt dab
                     // trip Main.cpp's g_viewport_raster_rebuild_pending handler (material buffer +
                     // hair re-upload) for a view that isn't even being shown, which is what made
                     // Vulkan RT feel like it was rebuilding on every stroke.
                     if (!liveViewportUpdated && liveRasterViewportBackend && liveViewportIsInteractiveRaster) {
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

