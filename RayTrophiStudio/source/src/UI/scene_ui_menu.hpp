/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_menu.hpp
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef SCENE_UI_MENU_HPP
#define SCENE_UI_MENU_HPP

// Implementation of SceneUI::drawMainMenuBar
// This file is included by scene_ui.cpp to reduce file size

#include "ProjectManager.h"
#include "scene_data.h"
#include "renderer.h"
#include "SceneSelection.h"
#include "OptixWrapper.h"
#include "MaterialManager.h"
#include "TerrainManager.h"
#include "TextureCompressionCache.h"
#include "scene_ui_animgraph.hpp"  // For AnimGraphUIState
#include "scene_ui_forcefield.hpp"
#include "SceneExporter.h" // For GLTF Export
#include "GasVolume.h"     // For Gas Volume creation
#include "HittableInstance.h"  // reorderInstanceTail below
#include <filesystem>
#include <thread>
#include <atomic>
#include <future>
#include <deque>
#include <algorithm>
#include <unordered_set>
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif
#include <scene_ui_gas.hpp>
#include "perlin.h"
#include "FractureGenerator.h"  // Convex Voronoi pre-fracture (destruction Faz 1)
#include "FractureSourceAdapter.h"
#include "MeshModifiers.h"      // facadesToFlatMesh (add-object-flat collapse)
#include "MeshEdit/SplineObjectService.h"
#include "TriangleMesh.h"       // complete type for shared_ptr<TriangleMesh>→shared_ptr<Hittable> upcast
#include "UI/TemplateHubUI.h"
#include "Template/UserTemplateManager.h"
// extern bool show_controls_window; // Assume defined elsewhere

extern bool g_vulkan_rebuild_pending;
extern bool g_viewport_raster_rebuild_pending;
extern std::unique_ptr<Backend::IBackend> g_backend;
// Defined in scene_ui_modifiers.cpp. Declared at FILE scope (NOT inside the anonymous namespace
// below) so it refers to the global symbol — an extern inside `namespace {}` gets internal linkage
// and would be "declared but not defined" (C7631).
extern bool g_dense_mesh_as_hittable;

namespace {
    bool s_show_save_user_template_modal = false;
    std::atomic<bool>  s_bakeRunning{false};
    std::atomic<int>   s_bakeDone{0};
    std::atomic<int>   s_bakeTotal{0};
    std::atomic<bool>  s_bakeNeedsHotReload{false};

    // Every CPU-side selection/hierarchy/picking scan assumes foliage/scatter instances
    // are a contiguous suffix of world.objects (selectable = size - InstanceManager::
    // getTotalInstanceCount(), see rebuildMeshCache/tri_to_index/viewport stats/selection
    // sync). Any code that push_back's a new object while that instance tail already
    // exists (mid-session "Add X" after a Geo-DAG Scatter/scatter-brush pass) breaks
    // that invariant — the new object gets miscounted as foliage and vanishes from the
    // hierarchy/picking even though it still renders (GPU paths iterate everything).
    // Restoring the invariant is a single stable_partition (non-instances first,
    // HittableInstance last) — cheap here since it only runs after a discrete "Add"
    // user action, never per-frame. See [[project_geo_dag_faz8b_fields]] for the
    // sibling fixes (asset import, Geo-DAG Evaluate) that hit the same bug class.
    void reorderInstanceTail(std::vector<std::shared_ptr<Hittable>>& objects) {
        std::stable_partition(objects.begin(), objects.end(),
            [](const std::shared_ptr<Hittable>& o) {
                return dynamic_cast<HittableInstance*>(o.get()) == nullptr;
            });
    }

    // Add-object-flat: collapse a freshly-added procedural primitive's standalone Triangle facades
    // (all sharing one node name) into ONE shared SoA TriangleMesh-as-Hittable — the add-object
    // equivalent of import-flat / apply-flat / Simple-subdivide-flat. Gated by
    // g_dense_mesh_as_hittable; flag OFF → no-op (facades stay, byte-for-byte old behaviour). All
    // procedural primitives are static (no skin), so no skin guard is needed. Runs right after the
    // facades are pushed, BEFORE rebuildMeshCache, so the editable cache / BVH / backends see the
    // flat mesh. facadesToFlatMesh copies the facades' shared Transform handle onto the TriangleMesh,
    // so the gizmo, procedural re-edit (name-keyed), and serialization (writeGeometryBinary's
    // TriangleMesh branch) keep working. Only standalone facades (parentMesh == null) with the given
    // name are collapsed, so it never disturbs already-flat meshes.
    void collapseProceduralAddToFlat(std::vector<std::shared_ptr<Hittable>>& objects,
                                     const std::string& name) {
        // Restore the instance-tail invariant FIRST — regardless of g_dense_mesh_as_hittable,
        // since the facades/objects were already push_back'd (at the true end, past any
        // existing instance tail) by the caller before this function runs.
        reorderInstanceTail(objects);
        if (!g_dense_mesh_as_hittable) return;
        std::vector<std::shared_ptr<Triangle>> facades;
        for (const auto& o : objects) {
            if (auto tri = std::dynamic_pointer_cast<Triangle>(o)) {
                if (!tri->parentMesh && tri->getNodeName() == name) facades.push_back(tri);
            }
        }
        if (facades.empty()) return;
        auto flat = MeshModifiers::facadesToFlatMesh(facades);
        if (!flat) return;
        std::vector<std::shared_ptr<Hittable>> rebuilt;
        rebuilt.reserve(objects.size());
        bool emitted = false;
        for (const auto& o : objects) {
            auto tri = std::dynamic_pointer_cast<Triangle>(o);
            if (tri && !tri->parentMesh && tri->getNodeName() == name) {
                if (!emitted) { rebuilt.push_back(flat); emitted = true; } // one TriangleMesh per group
            } else {
                rebuilt.push_back(o);
            }
        }
        objects = std::move(rebuilt);
    }

    // Launched in a detached thread. Iterates all materials and builds
    // compressed DDS cache for every eligible texture slot.
    void runTextureBakeThread() {
        const auto& mats = MaterialManager::getInstance().getAllMaterials();

        struct BakeSlot { std::shared_ptr<Texture> tex; TextureType type; bool srgb; };
        std::vector<BakeSlot> slots;
        slots.reserve(mats.size() * 5);

        // Deduplicate by (path, type): same texture shared across materials must only
        // be compressed once. Without this, two concurrent threads write the same DDS
        // simultaneously → file corruption + wasted work.
        std::unordered_set<std::string> seen;
        for (const auto& mat : mats) {
            if (!mat) continue;
            auto addSlot = [&](const MaterialProperty& prop, TextureType t, bool sg) {
                if (!prop.texture || prop.texture->name.empty() || prop.texture->pixels.empty()) return;
                const std::string key = prop.texture->name + '|' + std::to_string(static_cast<int>(t));
                if (!seen.insert(key).second) return; // already queued
                slots.push_back({prop.texture, t, sg});
            };
            addSlot(mat->albedoProperty,    TextureType::Albedo,    mat->albedoProperty.texture ? mat->albedoProperty.texture->is_srgb : false);
            addSlot(mat->normalProperty,    TextureType::Normal,    false);
            addSlot(mat->roughnessProperty, TextureType::Roughness, false);
            addSlot(mat->metallicProperty,  TextureType::Metallic,  false);
            addSlot(mat->specularProperty,  TextureType::Specular,  false);
            addSlot(mat->opacityProperty,   TextureType::Opacity,   false);
            if (mat->emissionProperty.texture && !mat->emissionProperty.texture->is_hdr)
                addSlot(mat->emissionProperty, TextureType::Emission, false);
        }

        // Sort small → large so progress advances quickly at first and large
        // textures are deferred to the end where the wait is more expected.
        std::sort(slots.begin(), slots.end(), [](const BakeSlot& a, const BakeSlot& b) {
            return (a.tex->width * a.tex->height) < (b.tex->width * b.tex->height);
        });

        s_bakeTotal.store(static_cast<int>(slots.size()), std::memory_order_release);
        s_bakeDone.store(0, std::memory_order_release);

        // 2 concurrent textures: each uses TEX_COMPRESS_PARALLEL internally,
        // so total threads ≈ 2 × core count — acceptable without severe contention.
        const size_t MAX_PARALLEL = 2;
        std::deque<std::future<void>> inflight;
        for (const auto& slot : slots) {
            // Drain oldest future when at capacity
            if (inflight.size() >= MAX_PARALLEL) {
                inflight.front().wait();
                inflight.pop_front();
            }
            inflight.push_back(std::async(std::launch::async, [slot]() {
                TextureCompressedCacheCandidate cand;
                tryBuildCompressedTextureCache(*slot.tex, slot.type, slot.srgb, cand);
                s_bakeDone.fetch_add(1, std::memory_order_release);
            }));
        }
        for (auto& f : inflight) f.wait();

        s_bakeNeedsHotReload.store(true, std::memory_order_release);
        s_bakeRunning.store(false, std::memory_order_release);
    }

    void openFolderInExplorer(const std::filesystem::path& folder) {
#ifdef _WIN32
        ShellExecuteW(nullptr, L"open", folder.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
        (void)folder;
#endif
    }

    bool sceneUiMenuRenderBackendIsVulkan() {
        return dynamic_cast<Backend::VulkanBackendAdapter*>(g_backend.get()) != nullptr;
    }
}



void SceneUI::drawMainMenuBar(UIContext& ctx)
{
    const auto& t = ThemeManager::instance().current();

    ImGui::PushStyleColor(ImGuiCol_MenuBarBg, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
    ImGui::PushStyleColor(ImGuiCol_PopupBg, ImVec4(t.colors.background.x, t.colors.background.y, t.colors.background.z, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.22f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.48f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.70f));
    ImGui::PushStyleColor(ImGuiCol_Border, t.colors.border);
    ImGui::PushStyleColor(ImGuiCol_Separator, t.colors.border);
    ImGui::PushStyleColor(ImGuiCol_Text, t.colors.text);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(14.0f, 11.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8.0f, 7.0f));
    float pop_round = 14.0f;
    float frame_round = 12.0f;
    float win_round = 14.0f;
    if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
        const auto& curTheme = ThemeManager::instance().current();
        pop_round = curTheme.style.popupRounding;
        frame_round = curTheme.style.frameRounding;
        win_round = curTheme.style.windowRounding;
    }
    ImGui::PushStyleVar(ImGuiStyleVar_PopupRounding, pop_round);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, frame_round);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, win_round);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(12.0f, 10.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_PopupBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign, ImVec2(0.0f, 0.5f));

    if (ImGui::BeginMainMenuBar())
    {
        ImDrawList* menu_dl = ImGui::GetWindowDrawList();
        const ImVec2 menu_min = ImGui::GetWindowPos();
        const ImVec2 menu_max(menu_min.x + ImGui::GetWindowWidth(), menu_min.y + ImGui::GetWindowHeight());
        // Keep the docked panels visually attached to the menu bar.
        // The previous +6 px reserve left a visible gap above the left panel.
        g_main_menu_reserved_height = (menu_max.y - menu_min.y) + 1.0f;
        const ImVec2 shell_min(menu_min.x + 6.0f, menu_min.y + 4.0f);
        const ImVec2 shell_max(menu_max.x - 6.0f, menu_max.y - 4.0f);
        
        // Floating shell background matching active theme background
        menu_dl->AddRectFilled(
            shell_min,
            shell_max,
            ImGui::ColorConvertFloat4ToU32(ImVec4(t.colors.background.x, t.colors.background.y, t.colors.background.z, 0.94f)),
            14.0f
        );
        // Floating shell border matching active theme border / faint accent
        menu_dl->AddRect(
            shell_min,
            shell_max,
            ImGui::ColorConvertFloat4ToU32(ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.14f)),
            14.0f,
            0,
            1.0f
        );
        menu_dl->AddLine(
            ImVec2(shell_min.x + 16.0f, shell_min.y + 1.0f),
            ImVec2(shell_max.x - 16.0f, shell_min.y + 1.0f),
            ImGui::ColorConvertFloat4ToU32(ImVec4(1.0f, 1.0f, 1.0f, 0.045f)),
            1.0f
        );
        menu_dl->AddLine(
            ImVec2(shell_min.x + 12.0f, shell_max.y - 1.0f),
            ImVec2(shell_max.x - 12.0f, shell_max.y - 1.0f),
            ImGui::ColorConvertFloat4ToU32(ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.08f)),
            1.0f
        );

        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + 12.0f);
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 2.0f);
        ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.22f));
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.48f));
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(t.colors.accent.x, t.colors.accent.y, t.colors.accent.z, 0.70f));
        ImGui::Selectable("RayTrophi", false, 0, ImVec2(90.0f, 0.0f));
        ImGui::PopStyleColor(3);
        ImGui::SameLine(0.0f, 14.0f);

        if (ImGui::BeginMenu("File"))
        {
            // ================================================================
            // NEW PROJECT
            // ================================================================
            if (ImGui::MenuItem("New Project", "Ctrl+N")) {
                 raytrophi::templates::TemplateHubUI::instance().show();
            }
            if (ImGui::MenuItem("Template Hub...", "Ctrl+Shift+T")) {
                 raytrophi::templates::TemplateHubUI::instance().show();
            }
            if (ImGui::MenuItem("Save Scene as Template...")) {
                 s_show_save_user_template_modal = true;
            }

            ImGui::Separator();

            // ================================================================
            // OPEN PROJECT (.rtp / .rts)
            // ================================================================
            if (ImGui::MenuItem("Open Project...", "Ctrl+O")) {
                tryOpen(ctx);
            }

            if (ImGui::BeginMenu("Project Save Options")) {
                auto& saveSettings = g_ProjectManager.save_settings;
                int textureMode = static_cast<int>(saveSettings.texture_storage_mode);

                ImGui::TextDisabled("Texture Storage");
                if (ImGui::RadioButton("Embedded In Project", textureMode == static_cast<int>(ProjectManager::TextureStorageMode::Embedded))) {
                    saveSettings.texture_storage_mode = ProjectManager::TextureStorageMode::Embedded;
                }
                if (ImGui::RadioButton("Project-local Copies", textureMode == static_cast<int>(ProjectManager::TextureStorageMode::ProjectLocal))) {
                    saveSettings.texture_storage_mode = ProjectManager::TextureStorageMode::ProjectLocal;
                }
                if (ImGui::RadioButton("Keep Original Paths", textureMode == static_cast<int>(ProjectManager::TextureStorageMode::KeepOriginalPaths))) {
                    saveSettings.texture_storage_mode = ProjectManager::TextureStorageMode::KeepOriginalPaths;
                }

                ImGui::Separator();
                ImGui::Checkbox("Embed Missing Only", &saveSettings.embed_missing_only);
                ImGui::Checkbox("Save Geometry To Project", &saveSettings.save_geometry);

                ImGui::Separator();
                ImGui::TextWrapped("Embedded keeps a self-contained project. Project-local copies writes textures next to the .rtp. Keep original paths stores external paths only.");
                ImGui::EndMenu();
            }

            if (ImGui::BeginMenu("Vulkan Texture Cache")) {
                const bool hasSavedProjectPath = !g_ProjectManager.getCurrentFilePath().empty();

                if (!hasSavedProjectPath) {
                    ImGui::TextDisabled("Save the project first to enable project-scoped cache.");
                    ImGui::Separator();
                }

                if (ImGui::MenuItem("Open Cache Folder", nullptr, false, hasSavedProjectPath)) {
                    const auto cacheDir = getProjectTextureCacheDirectory();
                    if (cacheDir) {
                        try {
                            std::filesystem::create_directories(*cacheDir);
                            openFolderInExplorer(*cacheDir);
                        } catch (const std::exception& e) {
                            SCENE_LOG_WARN(std::string("[VulkanTextureCache] Failed to open cache folder: ") + e.what());
                        }
                    } else {
                        SCENE_LOG_WARN("[VulkanTextureCache] No active saved project path is available.");
                    }
                }

                if (ImGui::MenuItem("Clear Project Cache", nullptr, false, hasSavedProjectPath)) {
                    std::string clearReason;
                    if (clearProjectTextureCache(&clearReason)) {
                        SCENE_LOG_INFO("[VulkanTextureCache] Project cache cleared.");
                    } else {
                        SCENE_LOG_WARN("[VulkanTextureCache] " + clearReason);
                    }
                }

                ImGui::Separator();
                {
                    const bool baking = s_bakeRunning.load(std::memory_order_acquire);
                    if (baking) {
                        const int done  = s_bakeDone.load(std::memory_order_acquire);
                        const int total = s_bakeTotal.load(std::memory_order_acquire);
                        if (total > 0) {
                            char buf[64];
                            snprintf(buf, sizeof(buf), "Baking... %d / %d", done, total);
                            ImGui::TextDisabled("%s", buf);
                            ImGui::ProgressBar(static_cast<float>(done) / static_cast<float>(total), ImVec2(-1.f, 0.f));
                        } else {
                            ImGui::TextDisabled("Baking...");
                        }
                    } else {
                        const bool canBake = hasSavedProjectPath &&
                                             !MaterialManager::getInstance().getAllMaterials().empty();
                        if (ImGui::MenuItem("Bake Texture Cache", nullptr, false, canBake)) {
                            s_bakeRunning.store(true, std::memory_order_release);
                            std::thread(runTextureBakeThread).detach();
                        }
                        if (!hasSavedProjectPath) {
                            ImGui::TextDisabled("Save the project first to enable baking.");
                        }
                    }
                }
                ImGui::EndMenu();
            }
            
            // ================================================================
            // SAVE PROJECT
            // ================================================================
            if (ImGui::MenuItem("Save Project", "Ctrl+S")) {
                 std::string current_path = g_ProjectManager.getCurrentFilePath();
                 
                 // Sync on MAIN thread
                 updateProjectFromScene(ctx);
                 
                 // Pause rendering
                 rendering_stopped_cpu = true;
                 rendering_stopped_gpu = true;

                 if (!current_path.empty()) {
                     bg_save_state = 1; // State: Saving
                     
                     SCENE_LOG_INFO("Starting background save...");
                     
                     std::thread save_thread([this, current_path, &ctx]() {
                         std::this_thread::sleep_for(std::chrono::milliseconds(50));
                         bool result = ProjectManager::getInstance().saveProject(current_path, ctx.scene, ctx.render_settings, ctx.renderer,
                            [this](int p, const std::string& s) {
                                // Background save - no progress bar needed
                            });
                         
                         if (result) {
                             // Save auxiliary settings (Terrain Node Graph + UI Settings) to separate file
                             try {
                                 std::string auxPath = current_path + ".aux.json";
                                 nlohmann::json rootJson;
                                 
                                 // 1. Terrain Node Graph (REMOVE: Now in main json via TerrainManager)
                                 // rootJson["terrain_graph"] = terrainNodeGraph.toJson();
                                 
                                 // 2. Viewport Settings
                                 rootJson["viewport_settings"] = {
                                     {"shading_mode", viewport_settings.shading_mode},
                                     {"show_gizmos", viewport_settings.show_gizmos},
                                     {"show_selection_outline", viewport_settings.show_selection_outline},
                                     {"show_camera_hud", viewport_settings.show_camera_hud},
                                     {"show_focus_ring", viewport_settings.show_focus_ring},
                                     {"show_zoom_ring", viewport_settings.show_zoom_ring}
                                 };
                                 
                                 // 3. Guide Settings
                                 rootJson["guide_settings"] = {
                                     {"show_safe_areas", guide_settings.show_safe_areas},
                                     {"safe_area_type", guide_settings.safe_area_type},
                                     {"show_letterbox", guide_settings.show_letterbox},
                                     {"aspect_ratio_index", guide_settings.aspect_ratio_index},
                                     {"show_grid", guide_settings.show_grid},
                                     {"grid_type", guide_settings.grid_type},
                                     {"show_center", guide_settings.show_center}
                                 };

                                 std::ofstream auxFile(auxPath);
                                 if (auxFile.is_open()) {
                                     auxFile << rootJson.dump(2);
                                     auxFile.close();
                                     SCENE_LOG_INFO("[Save] Auxiliary settings saved: " + auxPath);
                                     bg_save_state = 2; // State: Done
                                 } else {
                                     bg_save_state = 3; // State: Error
                                 }
                             } catch (const std::exception& e) {
                                 SCENE_LOG_WARN("[Save] Failed to save auxiliary settings: " + std::string(e.what()));
                                 bg_save_state = 3; // State: Error
                             }
                         } else {
                             bg_save_state = 3; // State: Error
                         }
                         
                         // Resume rendering
                         rendering_stopped_cpu = false;
                         rendering_stopped_gpu = false;
                     });
                     save_thread.detach();
                 } else {
                     // No path yet, prompt Save As
                     std::string filepath = saveFileDialogW(L"RayTrophi Project (.rtp)\0*.rtp\0", L"rtp");
                     if (!filepath.empty()) {
                         bg_save_state = 1; // State: Saving
                         
                         SCENE_LOG_INFO("Starting background save...");
                         
                         std::thread save_thread([this, filepath, &ctx]() {
                             std::this_thread::sleep_for(std::chrono::milliseconds(50));
                             bool result = ProjectManager::getInstance().saveProject(filepath, ctx.scene, ctx.render_settings, ctx.renderer,
                                [this](int p, const std::string& s) {});
                             
                             if (result) {
                                 // Save auxiliary settings (.aux.json)
                                 try {
                                     std::string auxPath = filepath + ".aux.json";
                                     nlohmann::json rootJson;
                                     
                                     // rootJson["terrain_graph"] = terrainNodeGraph.toJson();
                                     rootJson["viewport_settings"] = {
                                         {"shading_mode", viewport_settings.shading_mode},
                                         {"show_gizmos", viewport_settings.show_gizmos},
                                     {"show_selection_outline", viewport_settings.show_selection_outline},
                                         {"show_camera_hud", viewport_settings.show_camera_hud},
                                         {"show_focus_ring", viewport_settings.show_focus_ring},
                                         {"show_zoom_ring", viewport_settings.show_zoom_ring}
                                     };
                                     rootJson["guide_settings"] = {
                                         {"show_safe_areas", guide_settings.show_safe_areas},
                                         {"safe_area_type", guide_settings.safe_area_type},
                                         {"show_letterbox", guide_settings.show_letterbox},
                                         {"aspect_ratio_index", guide_settings.aspect_ratio_index},
                                         {"show_grid", guide_settings.show_grid},
                                         {"grid_type", guide_settings.grid_type},
                                         {"show_center", guide_settings.show_center}
                                     };

                                     std::ofstream auxFile(auxPath);
                                     if (auxFile.is_open()) {
                                         auxFile << rootJson.dump(2);
                                         auxFile.close();
                                         SCENE_LOG_INFO("[Save] Auxiliary settings saved: " + auxPath);
                                         bg_save_state = 2; // State: Done
                                     } else {
                                        bg_save_state = 3; // Error
                                     }
                                 } catch (const std::exception& e) {
                                     SCENE_LOG_WARN("[Save] Failed to save auxiliary settings: " + std::string(e.what()));
                                     bg_save_state = 3; // Error
                                 }
                             } else {
                                 bg_save_state = 3; // Error
                             }
                             
                             // Resume rendering
                             rendering_stopped_cpu = false;
                             rendering_stopped_gpu = false;
                         });
                         save_thread.detach();
                         SCENE_LOG_INFO("Project saved: " + filepath);
                     } else {
                         rendering_stopped_cpu = false;
                         rendering_stopped_gpu = false;
                     }
                 }
            }
            
            if (ImGui::MenuItem("Save Project As...", nullptr)) {
                 std::string filepath = saveFileDialogW(L"RayTrophi Project (.rtp)\0*.rtp\0", L"rtp");
                 if (!filepath.empty()) {
                     updateProjectFromScene(ctx);
                     
                     rendering_stopped_cpu = true;
                     rendering_stopped_gpu = true;

                     bg_save_state = 1; // State: Saving
                     
                     SCENE_LOG_INFO("Starting background save...");
                     
                     std::thread save_thread([this, filepath, &ctx]() {
                         std::this_thread::sleep_for(std::chrono::milliseconds(50));
                         bool result = ProjectManager::getInstance().saveProject(filepath, ctx.scene, ctx.render_settings, ctx.renderer,
                            [this](int p, const std::string& s) {});
                         
                         if (result) {
                             // Save auxiliary settings (.aux.json)
                             try {
                                 std::string auxPath = filepath + ".aux.json";
                                 nlohmann::json rootJson;
                                 
                                 rootJson["terrain_graph"] = terrainNodeGraph.toJson();
                                 rootJson["viewport_settings"] = {
                                     {"shading_mode", viewport_settings.shading_mode},
                                     {"show_gizmos", viewport_settings.show_gizmos},
                                     {"show_selection_outline", viewport_settings.show_selection_outline},
                                     {"show_camera_hud", viewport_settings.show_camera_hud},
                                     {"show_focus_ring", viewport_settings.show_focus_ring},
                                     {"show_zoom_ring", viewport_settings.show_zoom_ring}
                                 };
                                 rootJson["guide_settings"] = {
                                     {"show_safe_areas", guide_settings.show_safe_areas},
                                     {"safe_area_type", guide_settings.safe_area_type},
                                     {"show_letterbox", guide_settings.show_letterbox},
                                     {"aspect_ratio_index", guide_settings.aspect_ratio_index},
                                     {"show_grid", guide_settings.show_grid},
                                     {"grid_type", guide_settings.grid_type},
                                     {"show_center", guide_settings.show_center}
                                 };

                                 std::ofstream auxFile(auxPath);
                                 if (auxFile.is_open()) {
                                     auxFile << rootJson.dump(2);
                                     auxFile.close();
                                     SCENE_LOG_INFO("[Save] Auxiliary settings saved: " + auxPath);
                                     bg_save_state = 2; // Done
                                 } else {
                                     bg_save_state = 3; // Error
                                 }
                             } catch (const std::exception& e) {
                                 SCENE_LOG_WARN("[Save] Failed to save auxiliary settings: " + std::string(e.what()));
                                 bg_save_state = 3; // Error
                             }
                         } else {
                             bg_save_state = 3; // Error
                         }
                         
                         // Resume rendering
                         rendering_stopped_cpu = false;
                         rendering_stopped_gpu = false;
                     });
                     save_thread.detach();
                     SCENE_LOG_INFO("Project saved: " + filepath);
                 }
            }
            
            ImGui::Separator();

            // ================================================================
            // IMPORT MODEL (Adds to scene, doesn't clear)
            // ================================================================
            if (ImGui::MenuItem("Import Model...", "Ctrl+I")) {
#ifdef _WIN32
                std::string file = openFileDialogW(L"3D Files\0*.gltf;*.glb;*.fbx;*.obj\0All Files\0*.*\0");
                if (!file.empty()) {
                    rendering_stopped_cpu = true;
                    rendering_stopped_gpu = true;
                    
                    scene_loading = true;
                    scene_loading_done = false;
                    pending_project_ui_restore = false;
                    scene_loading_progress = 0;
                    setSceneLoadingStage("Importing model...");

                    std::thread loader_thread([this, file, &ctx]() {
                         int wait_count = 0;
                         while (rendering_in_progress && wait_count < 20) {
                             std::this_thread::sleep_for(std::chrono::milliseconds(100));
                             wait_count++;
                         }
                         
                         // Import WITHOUT clearing scene
                         bool success = ProjectManager::getInstance().importModel(file, ctx.scene, ctx.renderer, g_backend.get(),
                             [this](int p, const std::string& s) {
                                 scene_loading_progress = p; 
                                 setSceneLoadingStage(s);
                             }, false); // false = DO NOT REBUILD in thread (Crash fix)
                         
                         // Set flags to trigger rebuild on MAIN thread
                         if (success) {
                             g_needs_geometry_rebuild.store(true);
                             // Despite the legacy name, this flag triggers deferred backend sync
                             // for both OptiX and Vulkan in Main.cpp scene-load finalization.
                             g_needs_optix_sync.store(ctx.backend_ptr != nullptr);
                             extern bool g_geometry_dirty;
                             extern bool g_materials_dirty;
                             extern bool g_gas_volumes_dirty;
                             extern std::atomic<uint64_t> g_scene_geometry_generation;
                             g_geometry_dirty = true;
                             g_materials_dirty = true;
                             g_gas_volumes_dirty = true;
                             g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
                             // The raster viewport rebuild is NOT requested here: the
                             // scene_loading_done finalize block in Main.cpp already sets
                             // g_viewport_raster_rebuild_pending for every interactive
                             // shading mode, and scene_loading_done is set right below.
                         }

                         if(ctx.scene.camera) ctx.scene.camera->update_camera_vectors();
                         
                         // Wait for GPU to finish all pending operations
                         if (ctx.optix_gpu_ptr) {
                             if (g_hasCUDA) cudaDeviceSynchronize();
                         }
                         
                         scene_loading = false;
                         scene_loading_done = true;
                         pending_project_ui_restore = true;
                         // NOTE: ctx.start_render will be set by main loop when it sees scene_loading_done
                    });
                    loader_thread.detach();
                }
#endif
            }

            if (ImGui::MenuItem("Export Scene (.glb/.gltf)...", nullptr)) {
                 SceneExporter::getInstance().show_export_popup = true;
            }

            ImGui::Separator();
            
            // ================================================================
            // VDB VOLUME IMPORT (Industry-Standard Volumetrics)
            // ================================================================
            if (ImGui::BeginMenu("Import VDB")) {
                if (ImGui::MenuItem("VDB Volume (.vdb)", nullptr)) {
                    importVDBVolume(ctx);
                }
                if (ImGui::MenuItem("VDB Sequence (folder)...", nullptr)) {
                    importVDBSequence(ctx);
                }
                ImGui::EndMenu();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Save Render Image...", nullptr)) {
                ctx.render_settings.save_image_requested = true;
            }
            
            ImGui::Separator();
            
            // Show project info
            ImGui::TextDisabled("Project: %s", ProjectManager::getInstance().getProjectName().c_str());
            if (ProjectManager::getInstance().hasUnsavedChanges()) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1,0.5f,0,1), "*");
            }
            
            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Alt+F4")) {
                 tryExit();
            }
            rtui::drawMenuRegion("menu.file");
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Edit")) {
             const bool block_history_actions =
                 g_scene_loading_in_progress.load() ||
                 scene_loading.load() ||
                 rendering_in_progress.load() ||
                 ctx.render_settings.backend_changed;
             if (ImGui::MenuItem("Undo", "Ctrl+Z", false, history.canUndo() && !block_history_actions)) {
                 history.undo(ctx);
                 rebuildMeshCache(ctx.scene.world.objects);
                 mesh_cache_valid = false;
                 ctx.selection.updatePositionFromSelection();
                 ctx.selection.selected.has_cached_aabb = false;
                 g_ProjectManager.markModified();
            }
            if (ImGui::MenuItem("Redo", "Ctrl+Y", false, history.canRedo() && !block_history_actions)) {
                 history.redo(ctx);
                 rebuildMeshCache(ctx.scene.world.objects);
                 mesh_cache_valid = false;
                 ctx.selection.updatePositionFromSelection();
                 ctx.selection.selected.has_cached_aabb = false;
                 g_ProjectManager.markModified();
            }
            
            ImGui::Separator();
            
            if (ImGui::MenuItem("Delete Selected", "Del/X", false, ctx.selection.hasSelection())) {
                triggerDelete(ctx);
            }
            rtui::drawMenuRegion("menu.edit");
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Select")) {
            const bool edit_mode_active = mesh_overlay_settings.enabled &&
                                          mesh_overlay_settings.edit_mode &&
                                          ctx.selection.mesh_element_mode != MeshElementSelectMode::Object;

            if (ImGui::MenuItem("Select All", "A")) {
                if (edit_mode_active) {
                    selectAllMeshElements(ctx);
                } else {
                    selectAllObjects(ctx);
                }
            }
            if (ImGui::MenuItem("Select None", "Alt+A")) {
                if (edit_mode_active) {
                    clearEditableMeshSelection();
                } else {
                    ctx.selection.clearSelection();
                }
            }
            if (ImGui::MenuItem("Invert Selection", "Ctrl+I")) {
                if (edit_mode_active) {
                    invertMeshSelection(ctx);
                } else {
                    invertObjectSelection(ctx);
                }
            }

            ImGui::Separator();

            ImGui::TextDisabled("Selection Tool");
            bool isBox = (mesh_overlay_settings.selection_tool == 0);
            bool isLasso = (mesh_overlay_settings.selection_tool == 1);
            if (ImGui::RadioButton("Box Select (B)", isBox)) {
                mesh_overlay_settings.selection_tool = 0;
            }
            if (ImGui::RadioButton("Lasso Select (L)", isLasso)) {
                mesh_overlay_settings.selection_tool = 1;
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Object")) {
            bool hasSelection = (ctx.selection.selected.type == SelectableType::Object &&
                                 ctx.selection.selected.object != nullptr);
            const std::string selectedNodeName =
                hasSelection ? ctx.selection.selected.object->getNodeName() : std::string{};
            const std::string effectiveNodeName =
                !active_mesh_edit_object_name.empty() ? active_mesh_edit_object_name : selectedNodeName;
            
            const bool edit_mode_active = mesh_overlay_settings.enabled &&
                                          mesh_overlay_settings.edit_mode &&
                                          ctx.selection.mesh_element_mode != MeshElementSelectMode::Object;

            bool isMesh = false;
            if (edit_mode_active) {
                isMesh = !active_mesh_edit_object_name.empty();
            } else {
                for (const auto& item : ctx.selection.multi_selection) {
                    if (item.type == SelectableType::Object && item.object) {
                        isMesh = true;
                        break;
                    }
                }
            }

            const bool has_selected_faces = !edit_mode_active || !editable_mesh_cache.selection.face_ids.empty();

            if (ImGui::MenuItem("Shade Flat", nullptr, false, isMesh && has_selected_faces)) {
                if (edit_mode_active) {
                    applyShadingToSelectedFaces(ctx, true, false);
                } else {
                    for (const auto& item : ctx.selection.multi_selection) {
                        if (item.type == SelectableType::Object && item.object) {
                            std::string name = item.object->getNodeName();
                            auto& shading = ensureMeshShadingSettings(name);
                            shading.flat_shading = true;
                            shading.auto_smooth = false;
                            applyMeshShadingSettings(ctx, name);
                        }
                    }
                }
            }
            
            if (ImGui::MenuItem("Shade Smooth", nullptr, false, isMesh && has_selected_faces)) {
                if (edit_mode_active) {
                    applyShadingToSelectedFaces(ctx, false, false);
                } else {
                    for (const auto& item : ctx.selection.multi_selection) {
                        if (item.type == SelectableType::Object && item.object) {
                            std::string name = item.object->getNodeName();
                            auto& shading = ensureMeshShadingSettings(name);
                            shading.flat_shading = false;
                            shading.auto_smooth = false;
                            applyMeshShadingSettings(ctx, name);
                        }
                    }
                }
            }

            bool autoSmoothActive = false;
            if (edit_mode_active) {
                if (!active_mesh_edit_object_name.empty()) {
                    autoSmoothActive = ensureMeshShadingSettings(active_mesh_edit_object_name).auto_smooth;
                }
            } else if (hasSelection) {
                autoSmoothActive = ensureMeshShadingSettings(selectedNodeName).auto_smooth;
            }

            if (ImGui::MenuItem("Shade Auto Smooth", nullptr, autoSmoothActive, isMesh && has_selected_faces)) {
                if (edit_mode_active) {
                    applyShadingToSelectedFaces(ctx, false, true);
                } else {
                    for (const auto& item : ctx.selection.multi_selection) {
                        if (item.type == SelectableType::Object && item.object) {
                            std::string name = item.object->getNodeName();
                            auto& shading = ensureMeshShadingSettings(name);
                            shading.auto_smooth = !shading.auto_smooth;
                            if (shading.auto_smooth) {
                                shading.flat_shading = false;
                            }
                            applyMeshShadingSettings(ctx, name);
                        }
                    }
                }
            }

            ImGui::Separator();

            if (ImGui::BeginMenu("Normals", isMesh)) {
                if (ImGui::MenuItem("Flip", nullptr, false, isMesh)) {
                    if (edit_mode_active) {
                        flipSelectedMeshNormals(ctx);
                    } else {
                        std::vector<std::pair<std::shared_ptr<Triangle>, int>> targets;
                        for (const auto& item : ctx.selection.multi_selection) {
                            if (item.type == SelectableType::Object && item.object) {
                                targets.emplace_back(item.object, item.object_index);
                            }
                        }
                        auto origMulti = ctx.selection.multi_selection;
                        auto origSel = ctx.selection.selected;
                        for (const auto& target : targets) {
                            ctx.selection.selectObject(target.first, target.second, target.first->getNodeName());
                            flipSelectedMeshNormals(ctx);
                        }
                        ctx.selection.multi_selection = origMulti;
                        ctx.selection.selected = origSel;
                        ctx.selection.updatePositionFromSelection();
                    }
                }
                if (ImGui::MenuItem("Recalculate Outside", nullptr, false, isMesh)) {
                    if (edit_mode_active) {
                        recalculateMeshNormals(ctx, true);
                    } else {
                        std::vector<std::pair<std::shared_ptr<Triangle>, int>> targets;
                        for (const auto& item : ctx.selection.multi_selection) {
                            if (item.type == SelectableType::Object && item.object) {
                                targets.emplace_back(item.object, item.object_index);
                            }
                        }
                        auto origMulti = ctx.selection.multi_selection;
                        auto origSel = ctx.selection.selected;
                        for (const auto& target : targets) {
                            ctx.selection.selectObject(target.first, target.second, target.first->getNodeName());
                            recalculateMeshNormals(ctx, true);
                        }
                        ctx.selection.multi_selection = origMulti;
                        ctx.selection.selected = origSel;
                        ctx.selection.updatePositionFromSelection();
                    }
                }
                if (ImGui::MenuItem("Recalculate Inside", nullptr, false, isMesh)) {
                    if (edit_mode_active) {
                        recalculateMeshNormals(ctx, false);
                    } else {
                        std::vector<std::pair<std::shared_ptr<Triangle>, int>> targets;
                        for (const auto& item : ctx.selection.multi_selection) {
                            if (item.type == SelectableType::Object && item.object) {
                                targets.emplace_back(item.object, item.object_index);
                            }
                        }
                        auto origMulti = ctx.selection.multi_selection;
                        auto origSel = ctx.selection.selected;
                        for (const auto& target : targets) {
                            ctx.selection.selectObject(target.first, target.second, target.first->getNodeName());
                            recalculateMeshNormals(ctx, false);
                        }
                        ctx.selection.multi_selection = origMulti;
                        ctx.selection.selected = origSel;
                        ctx.selection.updatePositionFromSelection();
                    }
                }
                ImGui::EndMenu();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Duplicate Object", "Shift+D", false, ctx.selection.hasSelection())) {
                triggerDuplicate(ctx);
            }
            if (ImGui::MenuItem("Delete Object", "Del/X", false, ctx.selection.hasSelection())) {
                triggerDelete(ctx);
            }
            if (ImGui::MenuItem("Deselect All", nullptr, false, ctx.selection.hasSelection())) {
                ctx.selection.clearSelection();
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Add")) {
             if (ImGui::BeginMenu("Mesh")) {
                 if (ImGui::MenuItem("Plane")) {
                     addProceduralPlane(ctx);
                     g_ProjectManager.markModified();
                 }
                 if (ImGui::MenuItem("Cube")) {
                     addProceduralCube(ctx);
                     g_ProjectManager.markModified();
                 }
                 if (ImGui::MenuItem("UV Sphere")) {
                     addProceduralSphere(ctx);
                     g_ProjectManager.markModified();
                 }
                 if (ImGui::MenuItem("Cylinder")) {
                     addProceduralCylinder(ctx);
                     g_ProjectManager.markModified();
                 }
                 ImGui::Separator();
                 if (ImGui::MenuItem("Procedural Generator...")) {
                     show_procedural_generator = true;
                 }
                 ImGui::EndMenu();
             }

             if (ImGui::BeginMenu("2D Spline")) {
                 ImGui::TextDisabled("Authoring source only — no mesh is generated");
                 ImGui::Separator();
                 auto addSpline = [&](MeshEdit::SplinePrimitiveType type,
                                      const char* label, const char* name,
                                      MeshEdit::SplinePlane plane) {
                     if (ImGui::MenuItem(label)) {
                         MeshEdit::addSplinePrimitiveObject(
                             ctx, history, type, name, plane);
                         g_ProjectManager.markModified();
                         addViewportMessage(std::string("Added 2D Spline: ") + name);
                     }
                 };
                 // The engine is Y-up. Create the source in the canonical XZ
                 // ground/curve plane; users orient it with the normal object
                 // transform instead of duplicating Add entries for every plane.
                 if (ImGui::BeginMenu("Profile Plane (XZ / Y Up)")) {
                     addSpline(MeshEdit::SplinePrimitiveType::Circle, "Circle", "CircleSpline", MeshEdit::SplinePlane::XZ);
                     addSpline(MeshEdit::SplinePrimitiveType::Rectangle, "Rectangle", "RectangleSpline", MeshEdit::SplinePlane::XZ);
                     addSpline(MeshEdit::SplinePrimitiveType::OpenLine, "Open Line", "LineSpline", MeshEdit::SplinePlane::XZ);
                     addSpline(MeshEdit::SplinePrimitiveType::OpenArc, "Open Arc", "ArcSpline", MeshEdit::SplinePlane::XZ);
                     ImGui::EndMenu();
                 }
                 ImGui::EndMenu();
             }
             
             if (ImGui::BeginMenu("Light")) {
                 if (ImGui::MenuItem("Point Light")) {
                     auto l = std::make_shared<PointLight>(Vec3(0,5,0), Vec3(10,10,10), 0.1f);
                     l->nodeName = "Point_" + std::to_string(ctx.scene.lights.size() + 1);
                     ctx.scene.lights.push_back(l);
                     int new_index = (int)ctx.scene.lights.size() - 1;
                     ctx.selection.selectLight(l, new_index, l->nodeName);
                     history.record(std::make_unique<AddLightCommand>(l));
                     ctx.renderer.resetCPUAccumulation();
                     if (ctx.optix_gpu_ptr) { ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights); ctx.optix_gpu_ptr->resetAccumulation(); }
                     g_ProjectManager.markModified();
                     SCENE_LOG_INFO("Added Point Light");
                     addViewportMessage("Added Point Light");
                 }
                 if (ImGui::MenuItem("Directional Light")) {
                     auto l = std::make_shared<DirectionalLight>(Vec3(-1,-1,-0.5), Vec3(5,5,5), 0.1f);
                     l->nodeName = "Directional_" + std::to_string(ctx.scene.lights.size() + 1);
                     ctx.scene.lights.push_back(l);
                     int new_index = (int)ctx.scene.lights.size() - 1;
                     ctx.selection.selectLight(l, new_index, l->nodeName);
                     history.record(std::make_unique<AddLightCommand>(l));
                     ctx.renderer.resetCPUAccumulation();
                     if (ctx.optix_gpu_ptr) { ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights); ctx.optix_gpu_ptr->resetAccumulation(); }
                     g_ProjectManager.markModified();
                     SCENE_LOG_INFO("Added Directional Light");
                     addViewportMessage("Added Directional Light");
                 }
                 if (ImGui::MenuItem("Spot Light")) {
                     auto l = std::make_shared<SpotLight>(Vec3(0,5,0), Vec3(0,-1,0), Vec3(10,10,10), 45.0f, 60.0f);
                     l->nodeName = "Spot_" + std::to_string(ctx.scene.lights.size() + 1);
                     ctx.scene.lights.push_back(l);
                     int new_index = (int)ctx.scene.lights.size() - 1;
                     ctx.selection.selectLight(l, new_index, l->nodeName);
                     history.record(std::make_unique<AddLightCommand>(l));
                     ctx.renderer.resetCPUAccumulation();
                     if (ctx.optix_gpu_ptr) { ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights); ctx.optix_gpu_ptr->resetAccumulation(); }
                     g_ProjectManager.markModified();
                     SCENE_LOG_INFO("Added Spot Light");
                     addViewportMessage("Added Spot Light");
                 }
                 if (ImGui::MenuItem("Area Light")) {
                     auto l = std::make_shared<AreaLight>(Vec3(0,5,0), Vec3(1,0,0), Vec3(0,0,1), 2.0f, 2.0f, Vec3(10,10,10));
                     l->nodeName = "Area_" + std::to_string(ctx.scene.lights.size() + 1);
                     ctx.scene.lights.push_back(l);
                     int new_index = (int)ctx.scene.lights.size() - 1;
                     ctx.selection.selectLight(l, new_index, l->nodeName);
                     history.record(std::make_unique<AddLightCommand>(l));
                     ctx.renderer.resetCPUAccumulation();
                     if (ctx.optix_gpu_ptr) { ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights); ctx.optix_gpu_ptr->resetAccumulation(); }
                     g_ProjectManager.markModified();
                     SCENE_LOG_INFO("Added Area Light");
                     addViewportMessage("Added Area Light");
                 }
                 ImGui::EndMenu();
             }
             
             // Camera submenu
             if (ImGui::MenuItem("Camera")) {
                 // Create new camera at world center, looking at origin
                 Vec3 new_pos = Vec3(0, 2, 5);      // Standard position
                 Vec3 new_target = Vec3(0, 0, 0);   // Look at origin
                 
                 auto cam = std::make_shared<Camera>(
                     new_pos,           // lookfrom
                     new_target,        // lookat  
                     Vec3(0, 1, 0),     // vup
                     45.0,              // vfov
                     1.78f,             // aspect
                     0.0f,              // aperture (0 = no DOF)
                     10.0f,             // focus_dist
                     5
                 );
                 cam->nodeName = "Camera_" + std::to_string(ctx.scene.cameras.size() + 1);
                 
                 ctx.scene.addCamera(cam);
                 ctx.selection.selectCamera(cam);
                 
                 g_ProjectManager.markModified();
                 SCENE_LOG_INFO("Added Camera: " + cam->nodeName);
                 addViewportMessage("Added Camera: " + cam->nodeName);
             }
             
             ImGui::Separator();

             // NOTE: "Add > Gas Volume" was removed. The legacy GasVolume/GasSimulator
             // path is deprecated and its editing UI is gone, so creating one here only
             // produced an object that could no longer be configured. Author gas/smoke/
             // fire via the Simulations panel (grid domains + flow sources) instead.
             // GasVolume load/serialize/render is still kept for backward compatibility
             // with old projects.

             if (ImGui::MenuItem("Force Field")) {
                 auto ff = std::make_shared<Physics::ForceField>("Force Field " + std::to_string(ctx.scene.force_field_manager.force_fields.size() + 1));
                 ctx.scene.addForceField(ff);
                 ctx.selection.selectForceField(ff, -1, ff->name);
                 ForceFieldUI::selected_force_field = ff;
                 g_ProjectManager.markModified();
                 SCENE_LOG_INFO("Added Force Field: " + ff->name);
                 addViewportMessage("Added Force Field");
                 show_forcefield_tab = true;
             }

             ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Render")) {
             if (ImGui::MenuItem("Render Image", "F12")) {
                 // Same behavior as Start Final Render button
                 extern bool show_render_window;
                 show_render_window = true;
                 ctx.render_settings.is_final_render_mode = true;
                 ctx.render_settings.render_current_samples = 0;
                 ctx.render_settings.render_progress = 0.0f;
                 ctx.render_settings.is_rendering_active = true;
                 ctx.render_settings.is_render_paused = false;
                 ctx.start_render = true;
                 SCENE_LOG_INFO("Starting Final Render via Menu (F12)");
                 addViewportMessage("Starting Render...");
             }
             rtui::drawMenuRegion("menu.render");
             ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View"))
        {
            ImGui::MenuItem("Properties Panel", nullptr, &showSidePanel);
            ImGui::MenuItem("Bottom Panel", nullptr, &show_animation_panel);
            ImGui::MenuItem("Asset Browser", nullptr, &show_asset_browser);
            // Auto open log if check (optional)
            if (ImGui::MenuItem("Log Window", nullptr, &show_scene_log)) {
                 if (show_scene_log) show_animation_panel = false;
            }
            ImGui::MenuItem("Python Console", nullptr, &show_python_console);
            ImGui::MenuItem("Remote IPC Control", nullptr, &show_remote_ipc_panel);
            ImGui::MenuItem("Agent Chat", nullptr, &show_agent_chat);
            ImGui::Separator();
            // --- Dockable layout (modern movable/tabbable panels) ---
            if (ImGui::MenuItem("Dockable Layout", nullptr, &docking_enabled)) {
                if (docking_enabled) docking_layout_dirty = true; // (re)build default layout
            }
            if (ImGui::MenuItem("Reset Panel Layout", nullptr, false, docking_enabled)) {
                docking_layout_dirty = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Terrain Tab", nullptr, &show_terrain_tab)) { 
                if (show_terrain_tab) { tab_to_focus = "Terrain"; focus_properties_panel_next_frame = true; }
            }
            if (ImGui::MenuItem("Water Tab", nullptr, &show_water_tab)) { 
                if (show_water_tab) { tab_to_focus = "Water"; focus_properties_panel_next_frame = true; }
            }
            if (ImGui::MenuItem("Volumetrics Tab", nullptr, &show_volumetric_tab)) {
                if (show_volumetric_tab) { tab_to_focus = "Volumetric"; focus_properties_panel_next_frame = true; }
            }
            if (ImGui::MenuItem("Physics Tab", nullptr, &show_forcefield_tab)) {
                if (show_forcefield_tab) { tab_to_focus = "Simulation"; focus_properties_panel_next_frame = true; }
            }
            if (ImGui::MenuItem("World Tab", nullptr, &show_world_tab)) { 
                if (show_world_tab) { tab_to_focus = "World"; focus_properties_panel_next_frame = true; }
            }
            if (ImGui::MenuItem("Stylize Tab", nullptr, &show_stylize_tab)) {
                if (show_stylize_tab) { active_properties_tab = 12; focus_properties_panel_next_frame = true; }
            }
            if (ImGui::MenuItem("Hair & Fur Tab", nullptr, &show_hair_tab)) {
                if (show_hair_tab) { active_properties_tab = 8; focus_properties_panel_next_frame = true; }
            }
            if (ImGui::MenuItem("Scatter Tab", nullptr, &show_scatter_tab)) {
                if (show_scatter_tab) { tab_to_focus = "Scatter"; focus_properties_panel_next_frame = true; }
            }
            if (ImGui::MenuItem("Modifiers & Sculpt Tab", nullptr, &show_modifiers_tab)) {
                if (show_modifiers_tab) { tab_to_focus = "Modifiers"; focus_properties_panel_next_frame = true; }
            }
            if (ImGui::MenuItem("Paint Mode Tab", nullptr, &show_paint_tab)) {
                if (show_paint_tab) { tab_to_focus = "Paint"; focus_properties_panel_next_frame = true; }
            }
            if (ImGui::MenuItem("System Tab", nullptr, &show_system_tab)) {
                if (show_system_tab) { tab_to_focus = "System"; focus_properties_panel_next_frame = true; }
            }

            // Addon panels (rt.ui). Closing an addon window only hides it, so this
            // is how the user gets it back without reloading the addon. Panels
            // mounted into a region are listed read-only — their host tab owns them.
            {
                const std::vector<rtui::PanelInfo> addon_panels = rtui::listPanels();
                if (!addon_panels.empty()) {
                    ImGui::Separator();
                    if (ImGui::BeginMenu("Addon Panels")) {
                        for (const rtui::PanelInfo& panel : addon_panels) {
                            const std::string label =
                                panel.title.empty() ? ("Panel " + std::to_string(panel.id)) : panel.title;
                            bool visible = panel.visible;
                            if (panel.region.empty()) {
                                if (ImGui::MenuItem(label.c_str(), nullptr, &visible))
                                    rtui::setPanelVisible(panel.id, visible);
                            } else {
                                if (ImGui::MenuItem((label + "  [" + panel.region + "]").c_str(),
                                                    nullptr, &visible))
                                    rtui::setPanelVisible(panel.id, visible);
                            }
                        }
                        ImGui::EndMenu();
                    }
                }
            }

            rtui::drawMenuRegion("menu.view");
            ImGui::EndMenu();
        }
        

        if (ImGui::BeginMenu("Help"))
        {
            if (ImGui::MenuItem("Documentation / Manual (Web)", "F1")) {
#ifdef _WIN32
                 char buffer[MAX_PATH];
                 // Dosya yolunu oluştur (manual/index.html)
                 // Proje "raytrac_sdl2" klasöründe çalışıyorsa:
                 GetFullPathNameA("manual\\index.html", MAX_PATH, buffer, NULL);
                 ShellExecuteA(NULL, "open", buffer, NULL, NULL, SW_SHOWNORMAL);
#endif
            }
            ImGui::Separator();
            ImGui::MenuItem("Quick Guide & Shortcuts", nullptr, &show_controls_window);
            rtui::drawMenuRegion("menu.help");
            ImGui::EndMenu();
        }

        drawViewportControls(ctx);
        ImGui::EndMainMenuBar();
    } else {
        g_main_menu_reserved_height = (std::max)(g_main_menu_reserved_height, 30.0f);
    }
    ImGui::PopStyleVar(8);
    ImGui::PopStyleColor(8);

    // Bake progress — HUD message updated in-place each frame (no new entries, no flickering).
    {
        static bool s_bakeWasRunning = false;
        const bool bakeNow = s_bakeRunning.load(std::memory_order_acquire);

        if (bakeNow) {
            const int done  = s_bakeDone.load(std::memory_order_acquire);
            const int total = s_bakeTotal.load(std::memory_order_acquire);
            char buf[80];
            if (total > 0) snprintf(buf, sizeof(buf), "Baking textures: %d / %d", done, total);
            else           snprintf(buf, sizeof(buf), "Baking textures...");

            // Find existing bake message and update it in-place (text + timer reset).
            // This avoids adding a new entry every frame, which caused fade flicker.
            auto it = std::find_if(active_messages.begin(), active_messages.end(),
                [](const ViewportMessage& m) { return m.text.rfind("Baking", 0) == 0; });
            if (it != active_messages.end()) {
                it->text           = buf;
                it->time_remaining = 1.0f;
            } else {
                addViewportMessage(buf, 1.0f, ImVec4(1.f, 0.85f, 0.2f, 1.f));
            }

        } else if (s_bakeWasRunning) {
            // Remove the in-progress message immediately so it doesn't linger.
            active_messages.erase(
                std::remove_if(active_messages.begin(), active_messages.end(),
                    [](const ViewportMessage& m) { return m.text.rfind("Baking", 0) == 0; }),
                active_messages.end());

            s_bakeNeedsHotReload.store(false, std::memory_order_release);
            { extern void invalidateTextureCacheTagCache(); invalidateTextureCacheTagCache(); }
            const int total = s_bakeTotal.load(std::memory_order_acquire);
            char buf[128];
            snprintf(buf, sizeof(buf),
                "Texture bake complete (%d cached) — reload the project file to apply.", total);
            addViewportMessage(buf, 8.f, ImVec4(0.3f, 1.f, 0.4f, 1.f));
        }
        s_bakeWasRunning = bakeNow;
    }

    // Render Template Hub overlay if visible
    raytrophi::templates::TemplateHubUI::instance().render(ctx, *this, &history);

    if (s_show_save_user_template_modal) {
        static char t_name[128] = "";
        static char t_desc[256] = "";

        if (!ImGui::IsPopupOpen("Save Scene as Template##UserModal")) {
            ImGui::OpenPopup("Save Scene as Template##UserModal");
        }

        if (ImGui::BeginPopupModal("Save Scene as Template##UserModal", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("Save active scene setup as a custom user template:");
            ImGui::Spacing();

            ImGui::InputText("Template Title", t_name, sizeof(t_name));
            ImGui::InputText("Description", t_desc, sizeof(t_desc));
            ImGui::Spacing();

            if (UIWidgets::PrimaryButton("Save Template", ImVec2(140, 32))) {
                if (strlen(t_name) > 0) {
                    std::string name_str = t_name;
                    std::string desc_str = t_desc;
                    bool saved = raytrophi::templates::UserTemplateManager::instance().saveCurrentSceneAsTemplate(
                        name_str, desc_str, "user", ctx, *this, &history);
                    if (saved) {
                        std::string msg = "User template '" + name_str + "' saved successfully!";
                        addViewportMessage(msg, 4.0f, ImVec4(0.3f, 1.0f, 0.4f, 1.0f));

                        t_name[0] = '\0';
                        t_desc[0] = '\0';
                        s_show_save_user_template_modal = false;
                        ImGui::CloseCurrentPopup();
                    }
                }
            }
            ImGui::SameLine();
            if (UIWidgets::SecondaryButton("Cancel", ImVec2(100, 32))) {
                s_show_save_user_template_modal = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        } else {
            s_show_save_user_template_modal = false;
        }
    }
}

// ============================================================================
// Helper Functions for Menu Actions
// ============================================================================

// Update project data from current scene state (for saving)
void SceneUI::updateProjectFromScene(UIContext& ctx) {
    // Delegate to ProjectManager for robust sync (Deletions, Transforms, Procedurals)
    g_ProjectManager.syncProjectToScene(ctx.scene);
    
    // Serialize UI
    g_ProjectManager.getProjectData().ui_layout_data = serialize();
}

// Add a procedural plane to the scene
void SceneUI::addProceduralPlane(UIContext& ctx) {
    std::shared_ptr<Transform> t = std::make_shared<Transform>();
    t->setBase(Matrix4x4::translation(Vec3(0,0,0)));
    
    auto def_mat = std::make_shared<PrincipledBSDF>(); 
    auto gpu = std::make_shared<GpuMaterial>();
    gpu->albedo = make_float3(0.8f, 0.8f, 0.8f);
    gpu->roughness = 0.5f;
    gpu->metallic = 0.0f;
    gpu->transmission = 0.0f;
    gpu->emission = make_float3(0.0f, 0.0f, 0.0f);
    gpu->opacity = 1.0f;
    gpu->ior = 1.5f;
    gpu->anisotropic = 0.0f;
    gpu->clearcoat = 0.0f;
    def_mat->gpuMaterial = gpu;
    
    // Keep the material's own display name (read by the Properties panel) in sync with the
    // registry key (read by the Paint panel's reverse-lookup getMaterialName) — leaving
    // materialName empty made the same material show as "Mat #N" in Properties but the
    // registry key in Paint, looking like two different materials.
    def_mat->materialName = "Obj_" + std::to_string(g_ProjectManager.getProjectData().next_object_id) + "_Material";
    uint16_t mat_id = MaterialManager::getInstance().getOrCreateMaterialID(def_mat->materialName, def_mat);

    Vec3 v0(-1, 0, 1), v1(1, 0, 1), v2(1, 0, -1), v3(-1, 0, -1);
    Vec3 n(0, 1, 0);
    Vec2 t0(0, 0), t1(1, 0), t2(1, 1), t3(0, 1);
    
    // Generate unique name
    std::string name = "Plane_" + std::to_string(g_ProjectManager.getProjectData().next_object_id);
    
    auto tri1 = std::make_shared<Triangle>(v0, v1, v2, n, n, n, t0, t1, t2, mat_id);
    tri1->setTransformHandle(t); 
    tri1->setNodeName(name);
    tri1->update_bounding_box();
    
    auto tri2 = std::make_shared<Triangle>(v0, v2, v3, n, n, n, t0, t2, t3, mat_id);
    tri2->setTransformHandle(t); 
    tri2->setNodeName(name);
    tri2->update_bounding_box();

    ctx.scene.world.objects.push_back(tri1);
    ctx.scene.world.objects.push_back(tri2);
    
    // Track in project
    ProceduralObjectData proc;
    proc.id = g_ProjectManager.getProjectData().generateObjectId();
    proc.mesh_type = ProceduralMeshType::Plane;
    proc.display_name = name;
    proc.transform = t->base;
    proc.material_id = mat_id;
    g_ProjectManager.getProjectData().procedural_objects.push_back(proc);

    collapseProceduralAddToFlat(ctx.scene.world.objects, name);
    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (g_backend) ctx.renderer.rebuildBackendGeometry(ctx.scene);
    g_viewport_raster_rebuild_pending = true;
    if (sceneUiMenuRenderBackendIsVulkan()) g_vulkan_rebuild_pending = true;
    // Ensure Main loop sees geometry change and triggers necessary syncs
    extern bool g_geometry_dirty;
    extern std::atomic<uint64_t> g_scene_geometry_generation;
    extern std::atomic<bool> g_needs_optix_sync;
    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);
    
    SCENE_LOG_INFO("Added Plane: " + name);
    addViewportMessage("Added Plane: " + name);
}

// ── Destruction: convex Voronoi pre-fracture (Faz 1) ─────────────────────────

// Refresh every geometry consumer after shards are added/removed (mirrors the
// procedural-add tail so the change shows on all backends + the picking BVH).
// Member function because rebuildMeshCache is a SceneUI method.
void SceneUI::fractureRefreshScene_(UIContext& ctx) {
    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (g_backend) ctx.renderer.rebuildBackendGeometry(ctx.scene);
    g_viewport_raster_rebuild_pending = true;
    if (sceneUiMenuRenderBackendIsVulkan()) g_vulkan_rebuild_pending = true;
    extern bool g_geometry_dirty;
    extern std::atomic<uint64_t> g_scene_geometry_generation;
    extern std::atomic<bool> g_needs_optix_sync;
    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);
}

// Erase every Triangle in the scene whose node name is in `nodes`.
static void fractureEraseNodes_(UIContext& ctx, const std::unordered_set<std::string>& nodes) {
    auto& objs = ctx.scene.world.objects;
    objs.erase(std::remove_if(objs.begin(), objs.end(),
        [&](const std::shared_ptr<Hittable>& o) {
            auto tri = std::dynamic_pointer_cast<Triangle>(o);
            return tri && nodes.count(tri->getNodeName()) > 0;
        }), objs.end());

    // ★★ AND THE RIGID BODIES THAT POINTED AT THEM.
    //
    // This only ever erased scene geometry, so re-fracturing left the previous
    // run's breakable bodies alive in the scene, referring to nodes that no
    // longer exist. They were invisible and unhittable, but they still counted
    // as members of their fracture group — and nodeWorldCenter() answers (0,0,0)
    // for a node it cannot find, so every group's centre and AABB got dragged
    // toward the world origin by a crowd of phantom shards.
    //
    // The visible symptoms were all downstream of that: shard_count reporting
    // several times the number of shards actually requested, six clusters whose
    // centres all sat on top of each other instead of spread along the object,
    // and a pressure impulse computed from a projected area that belonged to no
    // real geometry. Worth stating plainly because none of those symptoms points
    // anywhere near this function.
    auto& bodies = ctx.scene.rigid_bodies;
    const std::size_t before = bodies.size();
    bodies.erase(std::remove_if(bodies.begin(), bodies.end(),
        [&](const RayTrophiSim::RigidBodyObject& rb) {
            return nodes.count(rb.source_name) > 0;
        }), bodies.end());
    if (bodies.size() != before && ctx.scene.rigid_body_system)
        ctx.scene.rigid_body_system->setBodies(&ctx.scene.rigid_bodies);
}

void SceneUI::fractureSelectedMesh(UIContext& ctx, const std::string& node,
                                   int site_count, uint32_t seed, int pattern,
                                   const SceneData::FractureRecipe* settings) {
    if (node.empty()) return;
    // ★ Two different callers, one parameter, and they must not be conflated:
    //   settings == nullptr        -> the panel drives it (the artist's knobs)
    //   settings, sites EMPTY      -> a script drives it (rt.physics.fracture_object)
    //   settings, sites NON-EMPTY  -> a saved fracture is being REPLAYED
    // Only the third case is a replay. A script asking for a thermal fracture
    // still wants the damage field sampled; treating it as a replay would hand
    // it an empty site list and it would cut nothing.
    const bool is_replay = settings && !settings->sites.empty();
    const int   cluster_count = settings ? settings->cluster_count : fracture_cluster_count;
    const bool  exact_surface = settings ? settings->exact_surface : fracture_exact_surface;
    const float preview_gap   = settings ? settings->preview_gap   : fracture_preview_gap;

    // Remove any shards from a previous fracture of this node (re-fracture).
    if (auto sit = fracture_shard_nodes_.find(node); sit != fracture_shard_nodes_.end()) {
        std::unordered_set<std::string> old(sit->second.begin(), sit->second.end());
        fractureEraseNodes_(ctx, old);
        sit->second.clear();
    }

    // Gather world-space source triangles. First time: collect from the live scene
    // and PARK the originals (kept alive, pulled out of world.objects). Re-fracture:
    // read from the parked copies (the originals are no longer in the scene).
    std::vector<RayTrophiSim::FractureInputTri> src;
    uint16_t src_mat = 0xFFFF;
    auto& parked = fracture_parked_originals_[node];
    if (parked.empty()) {
        RayTrophiSim::gatherFractureSource(
            ctx.scene.world.objects, node, src, src_mat);
        if (src.empty()) {
            fracture_parked_originals_.erase(node);
            addViewportMessage("Fracture: '" + node + "' has no geometry.");
            return;
        }
        std::vector<std::shared_ptr<Hittable>> keep;
        keep.reserve(ctx.scene.world.objects.size());
        for (auto& o : ctx.scene.world.objects) {
            if (RayTrophiSim::isFractureSourceObject(o, node)) {
                parked.push_back(o);
            } else {
                keep.push_back(o);
            }
        }
        ctx.scene.world.objects.swap(keep);
        // Tell the project saver this node is parked, not deleted.
        ctx.scene.fracture_parked_nodes.insert(node);
    } else {
        RayTrophiSim::gatherFractureSource(parked, node, src, src_mat);
    }
    if (src.empty()) {
        unfractureMesh(ctx, node);
        addViewportMessage("Fracture: '" + node + "' has no geometry.");
        return;
    }

    // Impact point / spread for the clustered pattern = source bbox centre + ~40%
    // of the diagonal (Faz 2 will feed the real contact point instead).
    Vec3 mn = src[0].a, mx = src[0].a;
    for (const auto& t : src) {
        for (const Vec3& p : {t.a, t.b, t.c}) {
            mn = Vec3((std::min)(mn.x, p.x), (std::min)(mn.y, p.y), (std::min)(mn.z, p.z));
            mx = Vec3((std::max)(mx.x, p.x), (std::max)(mx.y, p.y), (std::max)(mx.z, p.z));
        }
    }
    RayTrophiSim::FractureParams fp;
    fp.site_count = (std::max)(1, site_count);
    fp.seed = seed ? seed : 1u;
    fp.pattern = (pattern == 2) ? RayTrophiSim::FracturePattern::ThermalWeakened
               : (pattern == 1) ? RayTrophiSim::FracturePattern::ImpactClustered
                                : RayTrophiSim::FracturePattern::Uniform;
    fp.impact_point = (mn + mx) * 0.5f;
    fp.impact_radius = (std::max)(1.0e-3f, (mx - mn).length() * 0.4f);
    fp.cluster_count = (std::max)(1, cluster_count);
    fp.exact_surface = exact_surface;
    if (is_replay) fp.explicit_sites = settings->sites;

    // ── Feed the burn history into the seeding ────────────────────────────────
    // ★ The one place the material-transformation chain closes on itself: MSF
    // knows where this object has lost mass to pyrolysis, melting and transfer,
    // and that is exactly where it should come apart. Collected as a plain point
    // cloud so the destruction module never learns the MSF buffer layout.
    //
    // Reads the HOST mirror, so it is only as current as the last readback —
    // acceptable because fracture generation is an authoring action, not a
    // per-frame one, and the panel that triggers it keeps readbacks flowing.
    // ★ Skipped on replay, and that is the point of storing the sites. The damage
    // field this reads is gone by the time a project reopens, so a replay that
    // consulted it would seed from an unburnt object and cut somewhere else.
    if (!is_replay && fp.pattern == RayTrophiSim::FracturePattern::ThermalWeakened) {
        std::vector<RayTrophiSim::MaterialDamageSample> damage;
        for (auto& system : ctx.scene.particle_systems) {
            if (!system.runtime) continue;
            const auto& fields = system.runtime->materialStateFields();
            const auto it = fields.find(node);
            if (it == fields.end()) continue;
            RayTrophiSim::MaterialStateFieldSystem::collectDamageSamples(
                it->second, damage);
            break;
        }
        // Deliberately converted rather than shared: the destruction module
        // takes a geometric point cloud and must not include MaterialStateField.h.
        fp.damage_samples.reserve(damage.size());
        for (const auto& sample : damage)
            fp.damage_samples.push_back({sample.position, sample.weight});
        if (fp.damage_samples.empty()) {
            addViewportMessage("Thermal fracture: '" + node +
                               "' has no recorded burn damage yet - seeding uniformly.");
        }
    }

    std::vector<RayTrophiSim::FractureShard> shards;
    RayTrophiSim::FractureStats fstats;
    if (!RayTrophiSim::generateFracture(src, fp, shards, &fstats) || shards.empty()) {
        // Degenerate input (flat/thin mesh): restore the original so nothing is lost.
        unfractureMesh(ctx, node);
        addViewportMessage("Fracture failed: '" + node + "' is not a solid mesh.");
        return;
    }

    // Build a scene node per shard (identity transform → local == world rest).
    auto& shard_nodes = fracture_shard_nodes_[node];
    auto& shard_clusters = fracture_shard_clusters_[node];
    shard_nodes.clear();
    shard_clusters.clear();
    shard_nodes.reserve(shards.size());
    shard_clusters.reserve(shards.size());
    // Optional preview gap: pull each shard toward its own centroid so the cut
    // seams are visible before any physics moves the pieces (Faz 1 feedback). 0 =
    // perfect tiling (looks intact until Faz 2 separates them on impact).
    const float shrink = 1.0f - std::clamp(preview_gap, 0.0f, 0.9f);

    // ★ CUT FACES GET THEIR OWN MATERIAL, and this is not cosmetic.
    //
    // A fresh break is not burnt. Leave the interior on the source material and
    // it inherits the source's char mask, sampled at whatever UV the cut face
    // happens to carry — so the inside of a shard comes out painted with the
    // burn that broke it, which is exactly backwards. Created lazily, so a
    // convex-mode run (no interior faces distinguishable) registers nothing.
    uint16_t interior_mat = 0xFFFFu;
    const std::string interior_name = node + "_Interior";
    auto interiorMaterial = [&]() {
        if (interior_mat == 0xFFFFu) {
            // Reused by name across re-fractures. addUniqueMaterial would mint a
            // fresh slot on every Generate Shards press, and the registry is 16
            // bits wide — an artist iterating on a break would quietly spend it.
            interior_mat = MaterialManager::getInstance().getMaterialID(interior_name);
            if (interior_mat == MaterialManager::INVALID_MATERIAL_ID) {
                auto mat = std::make_shared<PrincipledBSDF>();
                interior_mat = MaterialManager::getInstance().addMaterial(
                    interior_name, mat);
            }
        }
        return interior_mat;
    };

    int idx = 0;
    for (const auto& shard : shards) {
        const std::string sname = node + "__shard_" + std::to_string(idx++);
        auto xf = std::make_shared<Transform>();
        xf->setBase(Matrix4x4::identity());
        const Vec3 cz = shard.centroid;
        for (const auto& st : shard.tris) {
            const Vec3 a = cz + (st.a - cz) * shrink;
            const Vec3 b = cz + (st.b - cz) * shrink;
            const Vec3 c = cz + (st.c - cz) * shrink;
            // Exterior faces carry the surface they were cut from: the source
            // UVs, interpolated across the cut, and that triangle's own material
            // (a multi-material asset does not collapse onto one slot). The
            // convex path leaves both at their defaults, so it falls back to the
            // single gathered material and the old placeholder UVs.
            //
            // Keyed off the MODE, not off the values: a convex-path shard also
            // has a material field and a UV field, both left at their defaults,
            // and reading those as real data would texture every hull shard with
            // texel (0,0) of material 0.
            const bool has_uv = fp.exact_surface && !st.interior;
            const uint16_t mat = st.interior ? interiorMaterial()
                               : (has_uv ? st.material : src_mat);
            auto tri = has_uv
                ? std::make_shared<Triangle>(a, b, c, st.n, st.n, st.n,
                                             st.ua, st.ub, st.uc, mat)
                : std::make_shared<Triangle>(a, b, c, st.n, st.n, st.n,
                                             Vec2(0, 0), Vec2(1, 0), Vec2(0, 1), mat);
            tri->setTransformHandle(xf);
            tri->setNodeName(sname);
            tri->update_bounding_box();
            ctx.scene.world.objects.push_back(tri);
        }
        shard_nodes.push_back(sname);
        shard_clusters.push_back(shard.cluster);
    }

    int cluster_total = 0;
    for (int c : shard_clusters) cluster_total = (std::max)(cluster_total, c + 1);

    // ── Record how to do this again ───────────────────────────────────────────
    // Written from what the run REPORTED, not from what it was asked for: the
    // scatter rejects candidates and tops up, so `fstats.sites` is the only
    // description of this cut that reproduces it. Rewritten on replay too, so a
    // recipe that survived a load is saved back out unchanged.
    {
        SceneData::FractureRecipe& recipe = ctx.scene.fracture_recipes[node];
        recipe.site_count    = fp.site_count;
        recipe.seed          = fp.seed;
        recipe.pattern       = pattern;
        recipe.cluster_count = fp.cluster_count;
        recipe.exact_surface = fp.exact_surface;
        recipe.preview_gap   = preview_gap;
        recipe.sites         = fstats.sites;
        recipe.shard_nodes    = shard_nodes;
        recipe.shard_clusters = shard_clusters;
    }

    fractureRefreshScene_(ctx);
    std::string summary = "Fractured '" + node + "' into " +
                          std::to_string(shards.size()) + " shards / " +
                          std::to_string(cluster_total) + " structural clusters.";
    // ★ Say which path the cuts took. A cell that fell back to the hull looks
    // EXACTLY like a cell that was never asked to be exact — a convex blob — so
    // without this line "I enabled Exact Surface and it still blobs" is
    // unreadable: an open mesh and a bug produce the same picture.
    if (fp.exact_surface && fstats.cells_approximated > 0) {
        summary += " " + std::to_string(fstats.cells_approximated) +
                   " cut across holes in the mesh and were closed approximately"
                   " (their mass is an estimate).";
    }
    if (fp.exact_surface && fstats.cells_unsealed > 0) {
        summary += " " + std::to_string(fstats.cells_unsealed) +
                   " found no cross-section at all and fell back to convex hulls.";
    }
    addViewportMessage(summary);
    SCENE_LOG_INFO("Fracture: '" + node + "' -> " + std::to_string(shards.size()) +
                   " shards, " + std::to_string(cluster_total) + " clusters, exact " +
                   std::to_string(fstats.cells_exact) + " / approx " +
                   std::to_string(fstats.cells_approximated) + " / unsealed " +
                   std::to_string(fstats.cells_unsealed) + " / hull " +
                   std::to_string(fstats.cells_hull));
}

void SceneUI::unfractureMesh(UIContext& ctx, const std::string& node) {
    if (node.empty()) return;
    // ★ REFUSE rather than destroy. A fracture adopted from a reopened project
    // has no parked original — the source mesh was out of the scene when the
    // project was written, so nothing saved it. Falling through here would
    // delete every shard and put nothing back, turning "undo the fracture" into
    // "delete the object". Better to say why.
    if (fracture_parked_originals_.find(node) == fracture_parked_originals_.end() &&
        fracture_shard_nodes_.find(node) != fracture_shard_nodes_.end()) {
        addViewportMessage("Cannot un-fracture '" + node + "': its original mesh "
                           "was not saved with the project. Delete the shards "
                           "manually if you want it gone.");
        SCENE_LOG_WARN("[Fracture] un-fracture refused for '" + node +
                       "': no parked original (fracture was adopted from a load).");
        return;
    }
    // Drop the shards.
    if (auto sit = fracture_shard_nodes_.find(node); sit != fracture_shard_nodes_.end()) {
        std::unordered_set<std::string> shard_set(sit->second.begin(), sit->second.end());
        fractureEraseNodes_(ctx, shard_set);
        fracture_shard_nodes_.erase(sit);
    }
    fracture_shard_clusters_.erase(node);
    // Put the parked originals back into the scene.
    if (auto pit = fracture_parked_originals_.find(node); pit != fracture_parked_originals_.end()) {
        for (auto& o : pit->second) ctx.scene.world.objects.push_back(o);
        fracture_parked_originals_.erase(pit);
    }
    ctx.scene.fracture_parked_nodes.erase(node);  // back in the scene: no longer parked
    // The recipe describes a fracture that no longer exists. Leaving it would
    // resurrect the shards on the next load of a project saved un-fractured.
    ctx.scene.fracture_recipes.erase(node);
    fractureRefreshScene_(ctx);
    addViewportMessage("Restored '" + node + "' (un-fractured).");
}

// ── Restoring a saved fracture ───────────────────────────────────────────────
// Runs once, on the main thread, after the load finishes: it can touch
// world.objects and kick every backend's geometry rebuild, neither of which is
// safe from the loader thread.
//
// ★★ ADOPT FIRST, CUT ONLY IF THERE IS NOTHING TO ADOPT. Measured, not assumed:
// with `save_geometry` on (the default) the shard meshes come back from the
// project's binary geometry sidecar, whole and correctly named. What does NOT
// come back is SceneUI's bookkeeping, and that is what makes the object know it
// is fractured. Re-cutting in that state would emit a second set of shards
// carrying the same names as the restored ones.
//
// Re-cutting is for projects saved with geometry off, where meshes are
// re-imported or procedurally regenerated instead.
void SceneUI::replayFractureRecipes(UIContext& ctx) {
    if (ctx.scene.fracture_recipes.empty()) return;

    // Every node name currently in the scene, gathered once.
    //
    // ★★ BOTH REPRESENTATIONS. A scene object is either a flat TriangleMesh
    // (SoA) or a Triangle facade, and which one you get depends on where the
    // geometry came from — the project's binary sidecar restores flat meshes,
    // while a fresh cut emits facades. Scanning only for facades made every
    // restored shard invisible to this pass, so a fracture that HAD come back
    // whole was reported as gone. Same enumeration as rtapi::listObjects.
    std::unordered_set<std::string> present;
    for (const auto& object : ctx.scene.world.objects) {
        if (auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object)) {
            if (!mesh->nodeName.empty()) present.insert(mesh->nodeName);
        } else if (auto tri = std::dynamic_pointer_cast<Triangle>(object)) {
            if (!tri->getNodeName().empty()) present.insert(tri->getNodeName());
        }
    }

    // Copied because fractureSelectedMesh writes the recipe back as it goes.
    const std::map<std::string, SceneData::FractureRecipe> recipes =
        ctx.scene.fracture_recipes;

    int adopted = 0, recut = 0, failed = 0;
    for (const auto& entry : recipes) {
        const std::string& node = entry.first;
        const SceneData::FractureRecipe& recipe = entry.second;

        std::vector<std::string> live_shards;
        std::vector<int> live_clusters;
        const bool parallel = recipe.shard_clusters.size() == recipe.shard_nodes.size();
        for (std::size_t i = 0; i < recipe.shard_nodes.size(); ++i) {
            if (present.count(recipe.shard_nodes[i]) == 0) continue;
            live_shards.push_back(recipe.shard_nodes[i]);
            live_clusters.push_back(parallel ? recipe.shard_clusters[i] : 0);
        }

        if (!live_shards.empty()) {
            // ── Adopt ────────────────────────────────────────────────────────
            fracture_shard_nodes_[node] = live_shards;
            fracture_shard_clusters_[node] = live_clusters;
            // The source mesh is out of the scene because it was parked when the
            // cut was made. Recording that keeps the saver from calling it
            // deleted on the next save.
            if (present.count(node) == 0)
                ctx.scene.fracture_parked_nodes.insert(node);
            // ★ fracture_parked_originals_ stays EMPTY, and that is a real
            // limitation rather than an oversight: the parked mesh is not in the
            // scene, so nothing wrote it to the geometry sidecar. unfractureMesh
            // refuses rather than deleting shards it cannot replace.
            ++adopted;
            if (live_shards.size() != recipe.shard_nodes.size()) {
                SCENE_LOG_WARN("[Fracture] '" + node + "': " +
                    std::to_string(recipe.shard_nodes.size() - live_shards.size()) +
                    " of its saved shards are missing from the scene; adopting the rest.");
            }
        } else if (present.count(node) != 0) {
            // ── Re-cut ───────────────────────────────────────────────────────
            // Geometry was not saved, but the source came back (re-imported or
            // regenerated), so the recorded sites reproduce the same cut.
            fractureSelectedMesh(ctx, node, recipe.site_count, recipe.seed,
                                 recipe.pattern, &recipe);
            if (isMeshFractured(node)) ++recut;
            else {
                ++failed;
                SCENE_LOG_WARN("[Fracture] Re-cutting '" + node +
                               "' from its saved recipe produced no shards.");
            }
        } else {
            ++failed;
            // ★ Say WHICH node and say it loudly. Neither the shards nor the
            // source came back, so the object is simply absent — and an absent
            // object is exactly what an artist does NOT notice until something
            // else fails much later.
            SCENE_LOG_WARN("[Fracture] '" + node + "' is gone: neither its saved "
                           "shards nor its source mesh came back with the project.");
            continue;
        }

        // Drop rigid bodies bound to shards that are not here. Their shards are
        // invisible and unhittable, but they still count as members of their
        // fracture group — which drags the group's centre toward the world
        // origin and corrupts every impulse computed from it.
        const auto& produced = fracture_shard_nodes_[node];
        const std::unordered_set<std::string> alive(produced.begin(), produced.end());
        const std::string prefix = node + "__shard_";
        auto& bodies = ctx.scene.rigid_bodies;
        const std::size_t before = bodies.size();
        bodies.erase(std::remove_if(bodies.begin(), bodies.end(),
            [&](const RayTrophiSim::RigidBodyObject& rb) {
                return rb.source_name.rfind(prefix, 0) == 0 &&
                       alive.count(rb.source_name) == 0;
            }), bodies.end());
        if (bodies.size() != before) {
            SCENE_LOG_WARN("[Fracture] '" + node + "' came back with a different "
                           "shard set than it was saved with; dropped " +
                           std::to_string(before - bodies.size()) +
                           " rigid bodies with no shard left to bind to.");
            if (ctx.scene.rigid_body_system)
                ctx.scene.rigid_body_system->setBodies(&ctx.scene.rigid_bodies);
        }
    }

    if (adopted + recut > 0 || failed > 0) {
        std::string message = "Restored " + std::to_string(adopted + recut) +
                              " fracture(s)";
        if (recut > 0) message += " (" + std::to_string(recut) + " re-cut)";
        message += ".";
        if (failed > 0)
            message += " " + std::to_string(failed) +
                       " could not be restored - see the log.";
        addViewportMessage(message);
        SCENE_LOG_INFO("[Fracture] Replay: " + std::to_string(adopted) +
                       " adopted, " + std::to_string(recut) + " re-cut, " +
                       std::to_string(failed) + " failed.");
    }
}

// Add a procedural cube to the scene
void SceneUI::addProceduralCube(UIContext& ctx) {
    std::shared_ptr<Transform> t = std::make_shared<Transform>();
    t->setBase(Matrix4x4::translation(Vec3(0,0,0)));
    
    auto def_mat = std::make_shared<PrincipledBSDF>();
    auto gpu = std::make_shared<GpuMaterial>();
    gpu->albedo = make_float3(0.8f, 0.8f, 0.8f);
    gpu->roughness = 0.5f;
    gpu->metallic = 0.0f;
    gpu->transmission = 0.0f;
    gpu->emission = make_float3(0.0f, 0.0f, 0.0f);
    gpu->opacity = 1.0f;
    gpu->ior = 1.5f;
    gpu->anisotropic = 0.0f;
    gpu->clearcoat = 0.0f;
    def_mat->gpuMaterial = gpu;

    // Keep the material's own display name (read by the Properties panel) in sync with the
    // registry key (read by the Paint panel's reverse-lookup getMaterialName) — leaving
    // materialName empty made the same material show as "Mat #N" in Properties but the
    // registry key in Paint, looking like two different materials.
    def_mat->materialName = "Obj_" + std::to_string(g_ProjectManager.getProjectData().next_object_id) + "_Material";
    uint16_t mat_id = MaterialManager::getInstance().getOrCreateMaterialID(def_mat->materialName, def_mat);

    Vec3 p[8] = {
        Vec3(-1,-1, 1), Vec3( 1,-1, 1), Vec3( 1, 1, 1), Vec3(-1, 1, 1),
        Vec3(-1,-1,-1), Vec3( 1,-1,-1), Vec3( 1, 1,-1), Vec3(-1, 1,-1)
    };
    int indices[36] = {
        0,1,2, 2,3,0,
        1,5,6, 6,2,1,
        7,6,5, 5,4,7,
        4,0,3, 3,7,4,
        4,5,1, 1,0,4,
        3,2,6, 6,7,3
    };

    const float w = 1.0f / 4.0f;
    const float h = 1.0f / 3.0f;
    Vec2 uv_data[36] = {
        // Face 0 (Front) 1,1: BL, BR, TR,  TR, TL, BL
        Vec2(1*w, 1*h), Vec2(2*w, 1*h), Vec2(2*w, 2*h),
        Vec2(2*w, 2*h), Vec2(1*w, 2*h), Vec2(1*w, 1*h),
        // Face 1 (Right) 2,1: BL, BR, TR,  TR, TL, BL
        Vec2(2*w, 1*h), Vec2(3*w, 1*h), Vec2(3*w, 2*h),
        Vec2(3*w, 2*h), Vec2(2*w, 2*h), Vec2(2*w, 1*h),
        // Face 2 (Back) 3,1: TR, TL, BL,  BL, BR, TR
        Vec2(4*w, 2*h), Vec2(3*w, 2*h), Vec2(3*w, 1*h),
        Vec2(3*w, 1*h), Vec2(4*w, 1*h), Vec2(4*w, 2*h),
        // Face 3 (Left) 0,1: BL, BR, TR,  TR, TL, BL
        Vec2(0*w, 1*h), Vec2(1*w, 1*h), Vec2(1*w, 2*h),
        Vec2(1*w, 2*h), Vec2(0*w, 2*h), Vec2(0*w, 1*h),
        // Face 4 (Bottom) 1,0: BL, BR, TR,  TR, TL, BL
        Vec2(1*w, 0*h), Vec2(2*w, 0*h), Vec2(2*w, 1*h),
        Vec2(2*w, 1*h), Vec2(1*w, 1*h), Vec2(1*w, 0*h),
        // Face 5 (Top) 1,2: BL, BR, TR,  TR, TL, BL
        Vec2(1*w, 2*h), Vec2(2*w, 2*h), Vec2(2*w, 3*h),
        Vec2(2*w, 3*h), Vec2(1*w, 3*h), Vec2(1*w, 2*h),
    };

    std::string name = "Cube_" + std::to_string(g_ProjectManager.getProjectData().next_object_id);

    for(int i=0; i<36; i+=3) {
        Vec3 v0 = p[indices[i]];
        Vec3 v1 = p[indices[i+1]];
        Vec3 v2 = p[indices[i+2]];
        Vec3 n = (v1-v0).cross(v2-v0).normalize();
        
        Vec2 uv0 = uv_data[i];
        Vec2 uv1 = uv_data[i+1];
        Vec2 uv2 = uv_data[i+2];

        auto tri = std::make_shared<Triangle>(v0, v1, v2, n, n, n, uv0, uv1, uv2, mat_id);
        tri->setTransformHandle(t);
        tri->setNodeName(name);
        tri->update_bounding_box();
        ctx.scene.world.objects.push_back(tri);
    }

    // Track in project
    ProceduralObjectData proc;
    proc.id = g_ProjectManager.getProjectData().generateObjectId();
    proc.mesh_type = ProceduralMeshType::Cube;
    proc.display_name = name;
    proc.transform = t->base;
    proc.material_id = mat_id;
    g_ProjectManager.getProjectData().procedural_objects.push_back(proc);

    collapseProceduralAddToFlat(ctx.scene.world.objects, name);
    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (g_backend) ctx.renderer.rebuildBackendGeometry(ctx.scene);
    g_viewport_raster_rebuild_pending = true;
    if (sceneUiMenuRenderBackendIsVulkan()) g_vulkan_rebuild_pending = true;
    // Ensure Main loop sees geometry change and triggers necessary syncs
    extern bool g_geometry_dirty;
    extern std::atomic<uint64_t> g_scene_geometry_generation;
    extern std::atomic<bool> g_needs_optix_sync;
    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);
    
    SCENE_LOG_INFO("Added Cube: " + name);
    addViewportMessage("Added Cube: " + name);
}


// Add a procedural UV sphere to the scene
void SceneUI::addProceduralSphere(UIContext& ctx) {
    std::shared_ptr<Transform> t = std::make_shared<Transform>();
    t->setBase(Matrix4x4::translation(Vec3(0,0,0)));
    
    auto def_mat = std::make_shared<PrincipledBSDF>();
    auto gpu = std::make_shared<GpuMaterial>();
    gpu->albedo = make_float3(0.8f, 0.8f, 0.8f);
    gpu->roughness = 0.5f;
    gpu->metallic = 0.0f;
    def_mat->gpuMaterial = gpu;

    // Keep the material's own display name (read by the Properties panel) in sync with the
    // registry key (read by the Paint panel's reverse-lookup getMaterialName) — leaving
    // materialName empty made the same material show as "Mat #N" in Properties but the
    // registry key in Paint, looking like two different materials.
    def_mat->materialName = "Obj_" + std::to_string(g_ProjectManager.getProjectData().next_object_id) + "_Material";
    uint16_t mat_id = MaterialManager::getInstance().getOrCreateMaterialID(def_mat->materialName, def_mat);

    std::string name = "Sphere_" + std::to_string(g_ProjectManager.getProjectData().next_object_id);

    int latitudeBands = 30;
    int longitudeBands = 30;
    float radius = 1.0f;

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;

    for (int latNumber = 0; latNumber <= latitudeBands; ++latNumber) {
        float theta = latNumber * 3.1415926535f / latitudeBands;
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (int longNumber = 0; longNumber <= longitudeBands; ++longNumber) {
            float phi = longNumber * 2.0f * 3.1415926535f / longitudeBands;
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            float x = cosPhi * sinTheta;
            float y = cosTheta;
            float z = sinPhi * sinTheta;
            float u = 1.0f - ((float)longNumber / (float)longitudeBands);
            float v = 1.0f - ((float)latNumber / (float)latitudeBands);

            positions.push_back(Vec3(radius * x, radius * y, radius * z));
            normals.push_back(Vec3(x, y, z));
            uvs.push_back(Vec2(u, v));
        }
    }

    for (int latNumber = 0; latNumber < latitudeBands; ++latNumber) {
        for (int longNumber = 0; longNumber < longitudeBands; ++longNumber) {
            int first = (latNumber * (longitudeBands + 1)) + longNumber;
            int second = first + longitudeBands + 1;

            auto tri1 = std::make_shared<Triangle>(
                positions[first], positions[first + 1], positions[second],
                normals[first], normals[first + 1], normals[second],
                uvs[first], uvs[first + 1], uvs[second], mat_id);
            tri1->setTransformHandle(t);
            tri1->setNodeName(name);
            tri1->update_bounding_box();
            ctx.scene.world.objects.push_back(tri1);

            auto tri2 = std::make_shared<Triangle>(
                positions[first + 1], positions[second + 1], positions[second],
                normals[first + 1], normals[second + 1], normals[second],
                uvs[first + 1], uvs[second + 1], uvs[second], mat_id);
            tri2->setTransformHandle(t);
            tri2->setNodeName(name);
            tri2->update_bounding_box();
            ctx.scene.world.objects.push_back(tri2);
        }
    }

    ProceduralObjectData proc;
    proc.id = g_ProjectManager.getProjectData().generateObjectId();
    proc.mesh_type = ProceduralMeshType::Sphere;
    proc.display_name = name;
    proc.transform = t->base;
    proc.material_id = mat_id;
    g_ProjectManager.getProjectData().procedural_objects.push_back(proc);

    collapseProceduralAddToFlat(ctx.scene.world.objects, name);
    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (g_backend) ctx.renderer.rebuildBackendGeometry(ctx.scene);
    g_viewport_raster_rebuild_pending = true;
    if (sceneUiMenuRenderBackendIsVulkan()) g_vulkan_rebuild_pending = true;
        // Ensure Main loop sees geometry change and triggers necessary syncs
        extern bool g_geometry_dirty;
        extern std::atomic<uint64_t> g_scene_geometry_generation;
        extern std::atomic<bool> g_needs_optix_sync;
        g_geometry_dirty = true;
        g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
        g_needs_optix_sync.store(true, std::memory_order_release);
    
    SCENE_LOG_INFO("Added Sphere: " + name);
    addViewportMessage("Added Sphere: " + name);
}

// Add a procedural cylinder to the scene
void SceneUI::addProceduralCylinder(UIContext& ctx) {
    std::shared_ptr<Transform> t = std::make_shared<Transform>();
    t->setBase(Matrix4x4::translation(Vec3(0,0,0)));
    
    auto def_mat = std::make_shared<PrincipledBSDF>();
    auto gpu = std::make_shared<GpuMaterial>();
    gpu->albedo = make_float3(0.8f, 0.8f, 0.8f);
    gpu->roughness = 0.5f;
    gpu->metallic = 0.0f;
    def_mat->gpuMaterial = gpu;

    // Keep the material's own display name (read by the Properties panel) in sync with the
    // registry key (read by the Paint panel's reverse-lookup getMaterialName) — leaving
    // materialName empty made the same material show as "Mat #N" in Properties but the
    // registry key in Paint, looking like two different materials.
    def_mat->materialName = "Obj_" + std::to_string(g_ProjectManager.getProjectData().next_object_id) + "_Material";
    uint16_t mat_id = MaterialManager::getInstance().getOrCreateMaterialID(def_mat->materialName, def_mat);

    std::string name = "Cylinder_" + std::to_string(g_ProjectManager.getProjectData().next_object_id);

    int segments = 32;
    float radius = 1.0f;
    float halfHeight = 1.0f;

    std::vector<Vec3> positions;
    std::vector<Vec3> normals;
    std::vector<Vec2> uvs;

    // Body
    for (int i = 0; i <= segments; ++i) {
        float theta = (float)i / segments * 2.0f * 3.1415926535f;
        float x = radius * cos(theta);
        float z = radius * sin(theta);
        float u = (float)i / segments;
        
        positions.push_back(Vec3(x, halfHeight, z));
        normals.push_back(Vec3(cos(theta), 0.0f, sin(theta)));
        uvs.push_back(Vec2(u, 0.5f)); // Top of body at V=0.5

        positions.push_back(Vec3(x, -halfHeight, z));
        normals.push_back(Vec3(cos(theta), 0.0f, sin(theta)));
        uvs.push_back(Vec2(u, 0.0f)); // Bottom of body at V=0.0
    }

    // Caps
    Vec3 topCenter(0, halfHeight, 0);
    Vec3 bottomCenter(0, -halfHeight, 0);
    
    // Create triangles for body
    for (int i = 0; i < segments; ++i) {
        int idx = i * 2;
        // Fix winding: 0, 2, 1
        auto tri1 = std::make_shared<Triangle>(
            positions[idx], positions[idx + 2], positions[idx + 1],
            normals[idx], normals[idx + 2], normals[idx + 1],
            uvs[idx], uvs[idx + 2], uvs[idx + 1], mat_id);
        tri1->setTransformHandle(t);
        tri1->setNodeName(name);
        tri1->update_bounding_box();
        ctx.scene.world.objects.push_back(tri1);

        // Fix winding: 1, 2, 3
        auto tri2 = std::make_shared<Triangle>(
            positions[idx + 1], positions[idx + 2], positions[idx + 3],
            normals[idx + 1], normals[idx + 2], normals[idx + 3],
            uvs[idx + 1], uvs[idx + 2], uvs[idx + 3], mat_id);
        tri2->setTransformHandle(t);
        tri2->setNodeName(name);
        tri2->update_bounding_box();
        ctx.scene.world.objects.push_back(tri2);
    }
    
    // Top cap
    for (int i = 0; i < segments; ++i) {
        Vec3 v0 = topCenter;
        Vec3 v1 = positions[i * 2];
        Vec3 v2 = positions[(i * 2 + 2) % (segments * 2 + 2)];
        Vec3 n(0, 1, 0);
        Vec2 uv0(0.25f, 0.75f);
        Vec2 uv1(0.25f + 0.25f * cos(i * 2.0f * 3.1415926535f / segments), 0.75f + 0.25f * sin(i * 2.0f * 3.1415926535f / segments));
        Vec2 uv2(0.25f + 0.25f * cos((i + 1) * 2.0f * 3.1415926535f / segments), 0.75f + 0.25f * sin((i + 1) * 2.0f * 3.1415926535f / segments));
        
        auto tri = std::make_shared<Triangle>(v0, v1, v2, n, n, n, uv0, uv1, uv2, mat_id);
        tri->setTransformHandle(t);
        tri->setNodeName(name);
        tri->update_bounding_box();
        ctx.scene.world.objects.push_back(tri);
    }

    // Bottom cap
    for (int i = 0; i < segments; ++i) {
        Vec3 v0 = bottomCenter;
        Vec3 v1 = positions[(i * 2 + 3) % (segments * 2 + 2)];
        Vec3 v2 = positions[i * 2 + 1];
        Vec3 n(0, -1, 0);
        Vec2 uv0(0.75f, 0.75f);
        float angle1 = (i + 1) * 2.0f * 3.1415926535f / segments;
        float angle2 = i * 2.0f * 3.1415926535f / segments;
        Vec2 uv1(0.75f + 0.25f * cos(angle1), 0.75f + 0.25f * sin(angle1));
        Vec2 uv2(0.75f + 0.25f * cos(angle2), 0.75f + 0.25f * sin(angle2));
        
        auto tri = std::make_shared<Triangle>(v0, v1, v2, n, n, n, uv0, uv1, uv2, mat_id);
        tri->setTransformHandle(t);
        tri->setNodeName(name);
        tri->update_bounding_box();
        ctx.scene.world.objects.push_back(tri);
    }

    ProceduralObjectData proc;
    proc.id = g_ProjectManager.getProjectData().generateObjectId();
    proc.mesh_type = ProceduralMeshType::Cylinder;
    proc.display_name = name;
    proc.transform = t->base;
    proc.material_id = mat_id;
    g_ProjectManager.getProjectData().procedural_objects.push_back(proc);

    collapseProceduralAddToFlat(ctx.scene.world.objects, name);
    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (g_backend) ctx.renderer.rebuildBackendGeometry(ctx.scene);
    g_viewport_raster_rebuild_pending = true;
    if (sceneUiMenuRenderBackendIsVulkan()) g_vulkan_rebuild_pending = true;
    // Ensure Main loop sees geometry change and triggers necessary syncs
    extern bool g_geometry_dirty;
    extern std::atomic<uint64_t> g_scene_geometry_generation;
    extern std::atomic<bool> g_needs_optix_sync;
    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);

    SCENE_LOG_INFO("Added Cylinder: " + name);
    addViewportMessage("Added Cylinder: " + name);
}

inline void addQuadToScene(UIContext& ctx, std::shared_ptr<Transform> t, const std::string& name,
                           const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& v3,
                           const Vec3& n0, const Vec3& n1, const Vec3& n2, const Vec3& n3,
                           const Vec2& uv0, const Vec2& uv1, const Vec2& uv2, const Vec2& uv3,
                           uint16_t mat_id) {
    auto tri1 = std::make_shared<Triangle>(v0, v1, v2, n0, n1, n2, uv0, uv1, uv2, mat_id);
    tri1->setTransformHandle(t);
    tri1->setNodeName(name);
    tri1->update_bounding_box();
    ctx.scene.world.objects.push_back(tri1);

    auto tri2 = std::make_shared<Triangle>(v0, v2, v3, n0, n2, n3, uv0, uv2, uv3, mat_id);
    tri2->setTransformHandle(t);
    tri2->setNodeName(name);
    tri2->update_bounding_box();
    ctx.scene.world.objects.push_back(tri2);
}

void SceneUI::drawProceduralGeneratorWindow(UIContext& ctx) {
    ImGui::SetNextWindowSize(ImVec2(400, 500), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Procedural Mesh Generator", &show_procedural_generator)) {
        ImGui::End();
        return;
    }
    
    const char* types[] = { "Rock / Stone", "Brick Wall", "Torus", "Staircase" };
    ImGui::Combo("Type", &procedural_generator_type, types, IM_ARRAYSIZE(types));
    ImGui::Separator();
    
    auto& mats = MaterialManager::getInstance().getAllMaterials();
    
    if (procedural_generator_type == 0) {
        ImGui::InputText("Name", rock_name, IM_ARRAYSIZE(rock_name));
        ImGui::SliderInt("Resolution", &rock_resolution, 4, 64);
        ImGui::SliderFloat("Radius", &rock_radius, 0.1f, 10.0f);
        ImGui::SliderFloat("Noise Scale", &rock_noise_scale, 0.1f, 5.0f);
        ImGui::SliderFloat("Noise Strength", &rock_noise_strength, 0.0f, 2.0f);
        ImGui::SliderInt("Noise Octaves", &rock_noise_octaves, 1, 8);
        ImGui::SliderFloat("Scale X", &rock_scale_x, 0.1f, 5.0f);
        ImGui::SliderFloat("Scale Y", &rock_scale_y, 0.1f, 5.0f);
        ImGui::SliderFloat("Scale Z", &rock_scale_z, 0.1f, 5.0f);
        ImGui::SliderFloat("Flatness", &rock_flatness, 0.0f, 1.0f);
        ImGui::InputInt("Seed", &rock_seed);
        
        std::string cur_name = "Default";
        if (rock_material_selection >= 0 && rock_material_selection < (int)mats.size()) {
            cur_name = mats[rock_material_selection]->materialName;
        }
        if (ImGui::BeginCombo("Material", cur_name.c_str())) {
            for (int i = 0; i < (int)mats.size(); ++i) {
                bool is_selected = (rock_material_selection == i);
                if (ImGui::Selectable(mats[i]->materialName.c_str(), is_selected)) {
                    rock_material_selection = i;
                }
            }
            ImGui::EndCombo();
        }
    }
    else if (procedural_generator_type == 1) {
        ImGui::InputText("Name", brick_name, IM_ARRAYSIZE(brick_name));
        ImGui::SliderInt("Rows", &brick_rows, 1, 50);
        ImGui::SliderInt("Columns", &brick_cols, 1, 50);
        ImGui::SliderFloat("Brick Width", &brick_width, 0.1f, 5.0f);
        ImGui::SliderFloat("Brick Height", &brick_height, 0.1f, 2.0f);
        ImGui::SliderFloat("Brick Depth", &brick_depth, 0.1f, 5.0f);
        ImGui::SliderFloat("Mortar Gap", &brick_mortar_gap, 0.0f, 0.5f);
        ImGui::SliderFloat("Tilt Variation", &brick_tilt_variation, 0.0f, 0.2f);
        ImGui::SliderFloat("Pos Variation", &brick_pos_variation, 0.0f, 0.2f);
        ImGui::InputInt("Seed", &brick_seed);
        
        std::string cur_name = "Default";
        if (brick_material_selection >= 0 && brick_material_selection < (int)mats.size()) {
            cur_name = mats[brick_material_selection]->materialName;
        }
        if (ImGui::BeginCombo("Material", cur_name.c_str())) {
            for (int i = 0; i < (int)mats.size(); ++i) {
                bool is_selected = (brick_material_selection == i);
                if (ImGui::Selectable(mats[i]->materialName.c_str(), is_selected)) {
                    brick_material_selection = i;
                }
            }
            ImGui::EndCombo();
        }
    }
    else if (procedural_generator_type == 2) {
        ImGui::InputText("Name", torus_name, IM_ARRAYSIZE(torus_name));
        ImGui::SliderFloat("Major Radius", &torus_major_radius, 0.1f, 10.0f);
        ImGui::SliderFloat("Minor Radius", &torus_minor_radius, 0.05f, 5.0f);
        ImGui::SliderInt("Radial Segments", &torus_radial_segments, 4, 128);
        ImGui::SliderInt("Tubular Segments", &torus_tubular_segments, 4, 128);
        
        std::string cur_name = "Default";
        if (torus_material_selection >= 0 && torus_material_selection < (int)mats.size()) {
            cur_name = mats[torus_material_selection]->materialName;
        }
        if (ImGui::BeginCombo("Material", cur_name.c_str())) {
            for (int i = 0; i < (int)mats.size(); ++i) {
                bool is_selected = (torus_material_selection == i);
                if (ImGui::Selectable(mats[i]->materialName.c_str(), is_selected)) {
                    torus_material_selection = i;
                }
            }
            ImGui::EndCombo();
        }
    }
    else if (procedural_generator_type == 3) {
        ImGui::InputText("Name", stairs_name, IM_ARRAYSIZE(stairs_name));
        ImGui::SliderInt("Steps", &stairs_steps, 1, 100);
        ImGui::SliderFloat("Width", &stairs_step_width, 0.1f, 10.0f);
        ImGui::SliderFloat("Depth", &stairs_step_depth, 0.1f, 5.0f);
        ImGui::SliderFloat("Height", &stairs_step_height, 0.1f, 5.0f);
        ImGui::Checkbox("Solid Support", &stairs_solid);
        
        std::string cur_name = "Default";
        if (stairs_material_selection >= 0 && stairs_material_selection < (int)mats.size()) {
            cur_name = mats[stairs_material_selection]->materialName;
        }
        if (ImGui::BeginCombo("Material", cur_name.c_str())) {
            for (int i = 0; i < (int)mats.size(); ++i) {
                bool is_selected = (stairs_material_selection == i);
                if (ImGui::Selectable(mats[i]->materialName.c_str(), is_selected)) {
                    stairs_material_selection = i;
                }
            }
            ImGui::EndCombo();
        }
    }
    
    if (ImGui::Button("+ Create Material")) {
        auto new_mat = std::make_shared<PrincipledBSDF>();
        auto gpu = std::make_shared<GpuMaterial>();
        gpu->albedo = make_float3(0.7f, 0.7f, 0.7f);
        gpu->roughness = 0.5f;
        gpu->metallic = 0.0f;
        new_mat->gpuMaterial = gpu;
        std::string m_name = "Procedural_Mat_" + std::to_string(mats.size() + 1);
        new_mat->materialName = m_name;
        
        MaterialManager::getInstance().getOrCreateMaterialID(m_name, new_mat);
        rock_material_selection = (int)mats.size() - 1;
        brick_material_selection = (int)mats.size() - 1;
        torus_material_selection = (int)mats.size() - 1;
        stairs_material_selection = (int)mats.size() - 1;
    }
    
    ImGui::SameLine();
    
    if (ImGui::Button("Generate")) {
        if (procedural_generator_type == 0) {
            addProceduralRock(ctx);
        } else if (procedural_generator_type == 1) {
            addProceduralBrickWall(ctx);
        } else if (procedural_generator_type == 2) {
            addProceduralTorus(ctx);
        } else if (procedural_generator_type == 3) {
            addProceduralStaircase(ctx);
        }
        g_ProjectManager.markModified();
        show_procedural_generator = false;
    }
    
    ImGui::End();
}

void SceneUI::addProceduralRock(UIContext& ctx) {
    std::shared_ptr<Transform> t = std::make_shared<Transform>();
    t->setBase(Matrix4x4::translation(Vec3(0,0,0)));
    
    uint16_t mat_id = 0;
    auto& mats = MaterialManager::getInstance().getAllMaterials();
    if (rock_material_selection >= 0 && rock_material_selection < (int)mats.size()) {
        mat_id = MaterialManager::getInstance().getMaterialID(mats[rock_material_selection]->materialName);
    } else {
        auto def_mat = std::make_shared<PrincipledBSDF>();
        auto gpu = std::make_shared<GpuMaterial>();
        gpu->albedo = make_float3(0.8f, 0.8f, 0.8f);
        gpu->roughness = 0.5f;
        gpu->metallic = 0.0f;
        def_mat->gpuMaterial = gpu;
        def_mat->materialName = "Obj_" + std::to_string(g_ProjectManager.getProjectData().next_object_id) + "_Material";
        mat_id = MaterialManager::getInstance().getOrCreateMaterialID(def_mat->materialName, def_mat);
    }
    
    std::string name = std::string(rock_name) + "_" + std::to_string(g_ProjectManager.getProjectData().next_object_id);
    
    int N = (std::max)(4, (std::min)(64, rock_resolution));
    Perlin perlin(rock_seed);
    
    struct FaceData {
        Vec3 normal;
        Vec3 tangent;
        Vec3 bitangent;
    };
    FaceData faces[6] = {
        { Vec3( 1, 0, 0), Vec3( 0, 0,-1), Vec3( 0, 1, 0) }, // +X
        { Vec3(-1, 0, 0), Vec3( 0, 0, 1), Vec3( 0, 1, 0) }, // -X
        { Vec3( 0, 1, 0), Vec3( 1, 0, 0), Vec3( 0, 0,-1) }, // +Y
        { Vec3( 0,-1, 0), Vec3( 1, 0, 0), Vec3( 0, 0, 1) }, // -Y
        { Vec3( 0, 0, 1), Vec3( 1, 0, 0), Vec3( 0, 1, 0) }, // +Z
        { Vec3( 0, 0,-1), Vec3(-1, 0, 0), Vec3( 0, 1, 0) }  // -Z
    };
    
    for (int f = 0; f < 6; ++f) {
        FaceData fd = faces[f];
        
        std::vector<std::vector<Vec3>> face_verts(N + 1, std::vector<Vec3>(N + 1));
        std::vector<std::vector<Vec2>> face_uvs(N + 1, std::vector<Vec2>(N + 1));
        
        for (int i = 0; i <= N; ++i) {
            float u_loc = (float)i / N;
            float u = -1.0f + 2.0f * u_loc;
            for (int j = 0; j <= N; ++j) {
                float v_loc = (float)j / N;
                float v = -1.0f + 2.0f * v_loc;
                
                Vec3 p_cube = fd.normal + fd.tangent * u + fd.bitangent * v;
                Vec3 p_sphere = p_cube.normalize();
                
                float noise_val = 0.0f;
                float amp = 1.0f;
                Vec3 noise_p = p_sphere * rock_noise_scale;
                for (int o = 0; o < rock_noise_octaves; ++o) {
                    noise_val += amp * perlin.noise(noise_p);
                    amp *= 0.5f;
                    noise_p *= 2.0f;
                }
                
                float disp = noise_val * rock_noise_strength;
                Vec3 pos = p_sphere * (rock_radius + disp);
                
                pos.x *= rock_scale_x;
                pos.y *= rock_scale_y;
                pos.z *= rock_scale_z;
                
                if (rock_flatness > 0.0f) {
                    float limit = -rock_radius * rock_scale_y * (1.0f - rock_flatness);
                    if (pos.y < limit) {
                        pos.y = limit;
                    }
                }
                
                face_verts[i][j] = pos;
                
                int col = f % 3;
                int row = f / 3;
                face_uvs[i][j] = Vec2((col + u_loc) / 3.0f, (row + v_loc) / 2.0f);
            }
        }
        
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                Vec3 p00 = face_verts[i][j];
                Vec3 p10 = face_verts[i+1][j];
                Vec3 p11 = face_verts[i+1][j+1];
                Vec3 p01 = face_verts[i][j+1];
                
                Vec2 uv00 = face_uvs[i][j];
                Vec2 uv10 = face_uvs[i+1][j];
                Vec2 uv11 = face_uvs[i+1][j+1];
                Vec2 uv01 = face_uvs[i][j+1];
                
                auto get_smooth_normal = [&](int x, int y) {
                    Vec3 T;
                    if (x > 0 && x < N) T = face_verts[x+1][y] - face_verts[x-1][y];
                    else if (x == 0) T = face_verts[1][y] - face_verts[0][y];
                    else T = face_verts[N][y] - face_verts[N-1][y];
                    
                    Vec3 B;
                    if (y > 0 && y < N) B = face_verts[x][y+1] - face_verts[x][y-1];
                    else if (y == 0) B = face_verts[x][1] - face_verts[x][0];
                    else B = face_verts[x][N] - face_verts[x][N-1];
                    
                    Vec3 n = T.cross(B);
                    if (n.length_squared() > 1e-8f) return n.normalize();
                    return fd.normal;
                };
                
                Vec3 n00 = get_smooth_normal(i, j);
                Vec3 n10 = get_smooth_normal(i+1, j);
                Vec3 n11 = get_smooth_normal(i+1, j+1);
                Vec3 n01 = get_smooth_normal(i, j+1);
                
                addQuadToScene(ctx, t, name, p00, p10, p11, p01, n00, n10, n11, n01, uv00, uv10, uv11, uv01, mat_id);
            }
        }
    }
    
    ProceduralObjectData proc;
    proc.id = g_ProjectManager.getProjectData().generateObjectId();
    proc.mesh_type = ProceduralMeshType::Rock;
    proc.display_name = name;
    proc.transform = t->base;
    proc.material_id = mat_id;
    g_ProjectManager.getProjectData().procedural_objects.push_back(proc);

    collapseProceduralAddToFlat(ctx.scene.world.objects, name);
    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (g_backend) ctx.renderer.rebuildBackendGeometry(ctx.scene);
    g_viewport_raster_rebuild_pending = true;
    if (sceneUiMenuRenderBackendIsVulkan()) g_vulkan_rebuild_pending = true;
    
    extern bool g_geometry_dirty;
    extern std::atomic<uint64_t> g_scene_geometry_generation;
    extern std::atomic<bool> g_needs_optix_sync;
    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);
    
    SCENE_LOG_INFO("Added Rock: " + name);
    addViewportMessage("Added Rock: " + name);
}

void SceneUI::addProceduralBrickWall(UIContext& ctx) {
    std::shared_ptr<Transform> t = std::make_shared<Transform>();
    t->setBase(Matrix4x4::translation(Vec3(0,0,0)));
    
    uint16_t mat_id = 0;
    auto& mats = MaterialManager::getInstance().getAllMaterials();
    if (brick_material_selection >= 0 && brick_material_selection < (int)mats.size()) {
        mat_id = MaterialManager::getInstance().getMaterialID(mats[brick_material_selection]->materialName);
    } else {
        auto def_mat = std::make_shared<PrincipledBSDF>();
        auto gpu = std::make_shared<GpuMaterial>();
        gpu->albedo = make_float3(0.8f, 0.8f, 0.8f);
        gpu->roughness = 0.5f;
        gpu->metallic = 0.0f;
        def_mat->gpuMaterial = gpu;
        def_mat->materialName = "Obj_" + std::to_string(g_ProjectManager.getProjectData().next_object_id) + "_Material";
        mat_id = MaterialManager::getInstance().getOrCreateMaterialID(def_mat->materialName, def_mat);
    }
    
    std::string name = std::string(brick_name) + "_" + std::to_string(g_ProjectManager.getProjectData().next_object_id);
    
    int rows = (std::max)(1, brick_rows);
    int cols = (std::max)(1, brick_cols);
    float W = brick_width;
    float H = brick_height;
    float D = brick_depth;
    float G = brick_mortar_gap;
    
    auto rotate_point = [](const Vec3& pt, float rx, float ry, float rz) {
        float cosX = cos(rx), sinX = sin(rx);
        float y1 = pt.y * cosX - pt.z * sinX;
        float z1 = pt.y * sinX + pt.z * cosX;
        
        float cosY = cos(ry), sinY = sin(ry);
        float x2 = pt.x * cosY + z1 * sinY;
        float z2 = -pt.x * sinY + z1 * cosY;
        
        float cosZ = cos(rz), sinZ = sin(rz);
        float x3 = x2 * cosZ - y1 * sinZ;
        float y3 = x2 * sinZ + y1 * cosZ;
        
        return Vec3(x3, y3, z2);
    };
    
    auto get_rand = [](unsigned int& state) {
        state = state * 1664525u + 1013904223u;
        return (float)state / 4294967295.0f;
    };
    
    unsigned int l_seed = brick_seed;
    
    for (int r = 0; r < rows; ++r) {
        float x_offset = (r % 2 == 1) ? (W + G) * 0.5f : 0.0f;
        for (int c = 0; c < cols; ++c) {
            float bx = -((cols - 1) * (W + G)) * 0.5f + c * (W + G) + x_offset;
            float by = r * (H + G) + H * 0.5f;
            float bz = 0.0f;
            
            float rx = (get_rand(l_seed) * 2.0f - 1.0f) * brick_pos_variation;
            float ry = (get_rand(l_seed) * 2.0f - 1.0f) * brick_pos_variation;
            float rz = (get_rand(l_seed) * 2.0f - 1.0f) * brick_pos_variation;
            
            float rot_x = (get_rand(l_seed) * 2.0f - 1.0f) * brick_tilt_variation;
            float rot_y = (get_rand(l_seed) * 2.0f - 1.0f) * brick_tilt_variation;
            float rot_z = (get_rand(l_seed) * 2.0f - 1.0f) * brick_tilt_variation;
            
            Vec3 corners[8] = {
                Vec3(-W*0.5f, -H*0.5f,  D*0.5f), // 0
                Vec3( W*0.5f, -H*0.5f,  D*0.5f), // 1
                Vec3( W*0.5f,  H*0.5f,  D*0.5f), // 2
                Vec3(-W*0.5f,  H*0.5f,  D*0.5f), // 3
                Vec3(-W*0.5f, -H*0.5f, -D*0.5f), // 4
                Vec3( W*0.5f, -H*0.5f, -D*0.5f), // 5
                Vec3( W*0.5f,  H*0.5f, -D*0.5f), // 6
                Vec3(-W*0.5f,  H*0.5f, -D*0.5f)  // 7
            };
            
            Vec3 transformed[8];
            for (int i = 0; i < 8; ++i) {
                Vec3 rotated = rotate_point(corners[i], rot_x, rot_y, rot_z);
                transformed[i] = rotated + Vec3(bx + rx, by + ry, bz + rz);
            }
            
            Vec3 n_front  = rotate_point(Vec3(0,0,1), rot_x, rot_y, rot_z).normalize();
            Vec3 n_back   = rotate_point(Vec3(0,0,-1), rot_x, rot_y, rot_z).normalize();
            Vec3 n_left   = rotate_point(Vec3(-1,0,0), rot_x, rot_y, rot_z).normalize();
            Vec3 n_right  = rotate_point(Vec3(1,0,0), rot_x, rot_y, rot_z).normalize();
            Vec3 n_top    = rotate_point(Vec3(0,1,0), rot_x, rot_y, rot_z).normalize();
            Vec3 n_bottom = rotate_point(Vec3(0,-1,0), rot_x, rot_y, rot_z).normalize();
            
            Vec2 uv0(0,0), uv1(1,0), uv2(1,1), uv3(0,1);
            
            addQuadToScene(ctx, t, name, transformed[0], transformed[1], transformed[2], transformed[3], n_front, n_front, n_front, n_front, uv0, uv1, uv2, uv3, mat_id);
            addQuadToScene(ctx, t, name, transformed[5], transformed[4], transformed[7], transformed[6], n_back, n_back, n_back, n_back, uv0, uv1, uv2, uv3, mat_id);
            addQuadToScene(ctx, t, name, transformed[4], transformed[0], transformed[3], transformed[7], n_left, n_left, n_left, n_left, uv0, uv1, uv2, uv3, mat_id);
            addQuadToScene(ctx, t, name, transformed[1], transformed[5], transformed[6], transformed[2], n_right, n_right, n_right, n_right, uv0, uv1, uv2, uv3, mat_id);
            addQuadToScene(ctx, t, name, transformed[3], transformed[2], transformed[6], transformed[7], n_top, n_top, n_top, n_top, uv0, uv1, uv2, uv3, mat_id);
            addQuadToScene(ctx, t, name, transformed[4], transformed[5], transformed[1], transformed[0], n_bottom, n_bottom, n_bottom, n_bottom, uv0, uv1, uv2, uv3, mat_id);
        }
    }
    
    ProceduralObjectData proc;
    proc.id = g_ProjectManager.getProjectData().generateObjectId();
    proc.mesh_type = ProceduralMeshType::BrickWall;
    proc.display_name = name;
    proc.transform = t->base;
    proc.material_id = mat_id;
    g_ProjectManager.getProjectData().procedural_objects.push_back(proc);

    collapseProceduralAddToFlat(ctx.scene.world.objects, name);
    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (g_backend) ctx.renderer.rebuildBackendGeometry(ctx.scene);
    g_viewport_raster_rebuild_pending = true;
    if (sceneUiMenuRenderBackendIsVulkan()) g_vulkan_rebuild_pending = true;
    
    extern bool g_geometry_dirty;
    extern std::atomic<uint64_t> g_scene_geometry_generation;
    extern std::atomic<bool> g_needs_optix_sync;
    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);
    
    SCENE_LOG_INFO("Added Brick Wall: " + name);
    addViewportMessage("Added Brick Wall: " + name);
}

void SceneUI::addProceduralTorus(UIContext& ctx) {
    std::shared_ptr<Transform> t = std::make_shared<Transform>();
    t->setBase(Matrix4x4::translation(Vec3(0,0,0)));
    
    uint16_t mat_id = 0;
    auto& mats = MaterialManager::getInstance().getAllMaterials();
    if (torus_material_selection >= 0 && torus_material_selection < (int)mats.size()) {
        mat_id = MaterialManager::getInstance().getMaterialID(mats[torus_material_selection]->materialName);
    } else {
        auto def_mat = std::make_shared<PrincipledBSDF>();
        auto gpu = std::make_shared<GpuMaterial>();
        gpu->albedo = make_float3(0.8f, 0.8f, 0.8f);
        gpu->roughness = 0.5f;
        gpu->metallic = 0.0f;
        def_mat->gpuMaterial = gpu;
        def_mat->materialName = "Obj_" + std::to_string(g_ProjectManager.getProjectData().next_object_id) + "_Material";
        mat_id = MaterialManager::getInstance().getOrCreateMaterialID(def_mat->materialName, def_mat);
    }
    
    std::string name = std::string(torus_name) + "_" + std::to_string(g_ProjectManager.getProjectData().next_object_id);
    
    int R_segs = (std::max)(4, torus_radial_segments);
    int T_segs = (std::max)(4, torus_tubular_segments);
    float R = torus_major_radius;
    float r = torus_minor_radius;
    
    std::vector<std::vector<Vec3>> positions(R_segs + 1, std::vector<Vec3>(T_segs + 1));
    std::vector<std::vector<Vec3>> normals(R_segs + 1, std::vector<Vec3>(T_segs + 1));
    std::vector<std::vector<Vec2>> uvs(R_segs + 1, std::vector<Vec2>(T_segs + 1));
    
    for (int i = 0; i <= R_segs; ++i) {
        float u = (float)i / R_segs * 2.0f * 3.1415926535f;
        float cosU = cos(u);
        float sinU = sin(u);
        
        for (int j = 0; j <= T_segs; ++j) {
            float v = (float)j / T_segs * 2.0f * 3.1415926535f;
            float cosV = cos(v);
            float sinV = sin(v);
            
            float x = (R + r * cosV) * cosU;
            float y = r * sinV;
            float z = (R + r * cosV) * sinU;
            positions[i][j] = Vec3(x, y, z);
            
            Vec3 norm(cosV * cosU, sinV, cosV * sinU);
            normals[i][j] = norm.normalize();
            
            uvs[i][j] = Vec2((float)i / R_segs, (float)j / T_segs);
        }
    }
    
    for (int i = 0; i < R_segs; ++i) {
        for (int j = 0; j < T_segs; ++j) {
            Vec3 p00 = positions[i][j];
            Vec3 p10 = positions[i+1][j];
            Vec3 p11 = positions[i+1][j+1];
            Vec3 p01 = positions[i][j+1];
            
            Vec3 n00 = normals[i][j];
            Vec3 n10 = normals[i+1][j];
            Vec3 n11 = normals[i+1][j+1];
            Vec3 n01 = normals[i][j+1];
            
            Vec2 uv00 = uvs[i][j];
            Vec2 uv10 = uvs[i+1][j];
            Vec2 uv11 = uvs[i+1][j+1];
            Vec2 uv01 = uvs[i][j+1];
            
            addQuadToScene(ctx, t, name, p00, p10, p11, p01, n00, n10, n11, n01, uv00, uv10, uv11, uv01, mat_id);
        }
    }
    
    ProceduralObjectData proc;
    proc.id = g_ProjectManager.getProjectData().generateObjectId();
    proc.mesh_type = ProceduralMeshType::Torus;
    proc.display_name = name;
    proc.transform = t->base;
    proc.material_id = mat_id;
    g_ProjectManager.getProjectData().procedural_objects.push_back(proc);

    collapseProceduralAddToFlat(ctx.scene.world.objects, name);
    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (g_backend) ctx.renderer.rebuildBackendGeometry(ctx.scene);
    g_viewport_raster_rebuild_pending = true;
    if (sceneUiMenuRenderBackendIsVulkan()) g_vulkan_rebuild_pending = true;
    
    extern bool g_geometry_dirty;
    extern std::atomic<uint64_t> g_scene_geometry_generation;
    extern std::atomic<bool> g_needs_optix_sync;
    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);
    
    SCENE_LOG_INFO("Added Torus: " + name);
    addViewportMessage("Added Torus: " + name);
}

void SceneUI::addProceduralStaircase(UIContext& ctx) {
    std::shared_ptr<Transform> t = std::make_shared<Transform>();
    t->setBase(Matrix4x4::translation(Vec3(0,0,0)));
    
    uint16_t mat_id = 0;
    auto& mats = MaterialManager::getInstance().getAllMaterials();
    if (stairs_material_selection >= 0 && stairs_material_selection < (int)mats.size()) {
        mat_id = MaterialManager::getInstance().getMaterialID(mats[stairs_material_selection]->materialName);
    } else {
        auto def_mat = std::make_shared<PrincipledBSDF>();
        auto gpu = std::make_shared<GpuMaterial>();
        gpu->albedo = make_float3(0.8f, 0.8f, 0.8f);
        gpu->roughness = 0.5f;
        gpu->metallic = 0.0f;
        def_mat->gpuMaterial = gpu;
        def_mat->materialName = "Obj_" + std::to_string(g_ProjectManager.getProjectData().next_object_id) + "_Material";
        mat_id = MaterialManager::getInstance().getOrCreateMaterialID(def_mat->materialName, def_mat);
    }
    
    std::string name = std::string(stairs_name) + "_" + std::to_string(g_ProjectManager.getProjectData().next_object_id);
    
    int steps = (std::max)(1, stairs_steps);
    float W = stairs_step_width;
    float D = stairs_step_depth;
    float H = stairs_step_height;
    
    for (int s = 0; s < steps; ++s) {
        Vec3 tr0(-W*0.5f, (s + 1) * H, s * D);
        Vec3 tr1( W*0.5f, (s + 1) * H, s * D);
        Vec3 tr2( W*0.5f, (s + 1) * H, (s + 1) * D);
        Vec3 tr3(-W*0.5f, (s + 1) * H, (s + 1) * D);
        
        Vec3 n_up(0, 1, 0);
        Vec2 uv_tr0(0, s * D), uv_tr1(W, s * D), uv_tr2(W, (s+1) * D), uv_tr3(0, (s+1) * D);
        addQuadToScene(ctx, t, name, tr0, tr1, tr2, tr3, n_up, n_up, n_up, n_up, uv_tr0, uv_tr1, uv_tr2, uv_tr3, mat_id);
        
        Vec3 ri0(-W*0.5f, s * H, s * D);
        Vec3 ri1( W*0.5f, s * H, s * D);
        Vec3 ri2( W*0.5f, (s + 1) * H, s * D);
        Vec3 ri3(-W*0.5f, (s + 1) * H, s * D);
        
        Vec3 n_front(0, 0, -1);
        Vec2 uv_ri0(0, s * H), uv_ri1(W, s * H), uv_ri2(W, (s+1) * H), uv_ri3(0, (s+1) * H);
        addQuadToScene(ctx, t, name, ri0, ri1, ri2, ri3, n_front, n_front, n_front, n_front, uv_ri0, uv_ri1, uv_ri2, uv_ri3, mat_id);
        
        if (stairs_solid) {
            Vec3 le0(-W*0.5f, 0, (s + 1) * D);
            Vec3 le1(-W*0.5f, 0, s * D);
            Vec3 le2(-W*0.5f, (s + 1) * H, s * D);
            Vec3 le3(-W*0.5f, (s + 1) * H, (s + 1) * D);
            
            Vec3 n_left(-1, 0, 0);
            Vec2 uv_le0((s+1)*D, 0), uv_le1(s*D, 0), uv_le2(s*D, (s+1)*H), uv_le3((s+1)*D, (s+1)*H);
            addQuadToScene(ctx, t, name, le0, le1, le2, le3, n_left, n_left, n_left, n_left, uv_le0, uv_le1, uv_le2, uv_le3, mat_id);
            
            Vec3 ri_s0(W*0.5f, 0, s * D);
            Vec3 ri_s1(W*0.5f, 0, (s + 1) * D);
            Vec3 ri_s2(W*0.5f, (s + 1) * H, (s + 1) * D);
            Vec3 ri_s3(W*0.5f, (s + 1) * H, s * D);
            
            Vec3 n_right(1, 0, 0);
            Vec2 uv_rs0(s*D, 0), uv_rs1((s+1)*D, 0), uv_rs2((s+1)*D, (s+1)*H), uv_rs3(s*D, (s+1)*H);
            addQuadToScene(ctx, t, name, ri_s0, ri_s1, ri_s2, ri_s3, n_right, n_right, n_right, n_right, uv_rs0, uv_rs1, uv_rs2, uv_rs3, mat_id);
        }
    }
    
    if (stairs_solid) {
        Vec3 bo0(-W*0.5f, 0, steps * D);
        Vec3 bo1( W*0.5f, 0, steps * D);
        Vec3 bo2( W*0.5f, 0, 0);
        Vec3 bo3(-W*0.5f, 0, 0);
        
        Vec3 n_down(0, -1, 0);
        Vec2 uv_bo0(0, steps * D), uv_bo1(W, steps * D), uv_bo2(W, 0), uv_bo3(0, 0);
        addQuadToScene(ctx, t, name, bo0, bo1, bo2, bo3, n_down, n_down, n_down, n_down, uv_bo0, uv_bo1, uv_bo2, uv_bo3, mat_id);
        
        Vec3 ba0( W*0.5f, 0, steps * D);
        Vec3 ba1(-W*0.5f, 0, steps * D);
        Vec3 ba2(-W*0.5f, steps * H, steps * D);
        Vec3 ba3( W*0.5f, steps * H, steps * D);
        
        Vec3 n_back(0, 0, 1);
        Vec2 uv_ba0(W, 0), uv_ba1(0, 0), uv_ba2(0, steps * H), uv_ba3(W, steps * H);
        addQuadToScene(ctx, t, name, ba0, ba1, ba2, ba3, n_back, n_back, n_back, n_back, uv_ba0, uv_ba1, uv_ba2, uv_ba3, mat_id);
    }
    
    ProceduralObjectData proc;
    proc.id = g_ProjectManager.getProjectData().generateObjectId();
    proc.mesh_type = ProceduralMeshType::Staircase;
    proc.display_name = name;
    proc.transform = t->base;
    proc.material_id = mat_id;
    g_ProjectManager.getProjectData().procedural_objects.push_back(proc);

    collapseProceduralAddToFlat(ctx.scene.world.objects, name);
    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (g_backend) ctx.renderer.rebuildBackendGeometry(ctx.scene);
    g_viewport_raster_rebuild_pending = true;
    if (sceneUiMenuRenderBackendIsVulkan()) g_vulkan_rebuild_pending = true;
    
    extern bool g_geometry_dirty;
    extern std::atomic<uint64_t> g_scene_geometry_generation;
    extern std::atomic<bool> g_needs_optix_sync;
    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);
    
    SCENE_LOG_INFO("Added Staircase: " + name);
    addViewportMessage("Added Staircase: " + name);
}

#endif
