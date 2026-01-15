#ifndef SCENE_UI_MENU_HPP
#define SCENE_UI_MENU_HPP

// Implementation of SceneUI::drawMainMenuBar
// This file is included by scene_ui.cpp to reduce file size

#include "ProjectManager.h"
#include "scene_data.h"
#include "renderer.h"
#include "SceneSelection.h"
#include "OptixWrapper.h"
#include "TerrainManager.h"
#include "scene_ui_animgraph.hpp"  // For AnimGraphUIState
#include "SceneExporter.h" // For GLTF Export
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif
extern bool show_controls_window; // Assume defined elsewhere



void SceneUI::drawMainMenuBar(UIContext& ctx)
{
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            // ================================================================
            // NEW PROJECT
            // ================================================================
            if (ImGui::MenuItem("New Project", "Ctrl+N")) {
                 tryNew(ctx);
            }

            ImGui::Separator();

            // ================================================================
            // OPEN PROJECT (.rtp / .rts)
            // ================================================================
            if (ImGui::MenuItem("Open Project...", "Ctrl+O")) {
                tryOpen(ctx);
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
                                 
                                 // 1. Terrain Node Graph
                                 rootJson["terrain_graph"] = terrainNodeGraph.toJson();
                                 
                                 // 2. Viewport Settings
                                 rootJson["viewport_settings"] = {
                                     {"shading_mode", viewport_settings.shading_mode},
                                     {"show_gizmos", viewport_settings.show_gizmos},
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
                                     
                                     rootJson["terrain_graph"] = terrainNodeGraph.toJson();
                                     rootJson["viewport_settings"] = {
                                         {"shading_mode", viewport_settings.shading_mode},
                                         {"show_gizmos", viewport_settings.show_gizmos},
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
                    scene_loading_progress = 0;
                    scene_loading_stage = "Importing model...";

                    std::thread loader_thread([this, file, &ctx]() {
                         int wait_count = 0;
                         while (rendering_in_progress && wait_count < 20) {
                             std::this_thread::sleep_for(std::chrono::milliseconds(100));
                             wait_count++;
                         }
                         
                         // Import WITHOUT clearing scene
                         bool success = ProjectManager::getInstance().importModel(file, ctx.scene, ctx.renderer, ctx.optix_gpu_ptr,
                             [this](int p, const std::string& s) {
                                 scene_loading_progress = p; 
                                 scene_loading_stage = s;
                             }, false); // false = DO NOT REBUILD in thread (Crash fix)
                         
                         // Set flags to trigger rebuild on MAIN thread
                         if (success) {
                             g_needs_geometry_rebuild = true;
                             g_needs_optix_sync = (ctx.optix_gpu_ptr != nullptr);
                         }

                         if(ctx.scene.camera) ctx.scene.camera->update_camera_vectors();
                         
                         // Wait for GPU to finish all pending operations
                         if (ctx.optix_gpu_ptr) {
                             cudaDeviceSynchronize();
                         }
                         
                         scene_loading = false;
                         scene_loading_done = true;
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
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Edit")) {
             if (ImGui::MenuItem("Undo", "Ctrl+Z", false, history.canUndo())) {
                 history.undo(ctx);
                 rebuildMeshCache(ctx.scene.world.objects);
                 mesh_cache_valid = false;
                 ctx.selection.updatePositionFromSelection();
                 ctx.selection.selected.has_cached_aabb = false;
                 g_ProjectManager.markModified();
            }
            if (ImGui::MenuItem("Redo", "Ctrl+Y", false, history.canRedo())) {
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
             ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View"))
        {
            ImGui::MenuItem("Properties Panel", nullptr, &showSidePanel);
            ImGui::MenuItem("Bottom Panel", nullptr, &show_animation_panel);
            // Auto open log if check (optional)
            if (ImGui::MenuItem("Log Window", nullptr, &show_scene_log)) {
                 if (show_scene_log) show_animation_panel = false;
            }
            ImGui::Separator();
            

            if (ImGui::MenuItem("Water Tab", nullptr, &show_water_tab)) { 
                if (show_water_tab) tab_to_focus = "Water"; 
            }
            if (ImGui::MenuItem("Terrain Tab", nullptr, &show_terrain_tab)) { 
                if (show_terrain_tab) tab_to_focus = "Terrain"; 
            }
            if (ImGui::MenuItem("VDB Tab (Volumes)", nullptr, &show_vdb_tab)) { 
                if (show_vdb_tab) tab_to_focus = "VDB"; 
            }
            ImGui::Separator();
            if (ImGui::MenuItem("System Tab", nullptr, &show_system_tab)) { 
                if (show_system_tab) tab_to_focus = "System"; 
            }
            
            ImGui::Separator();
            
            // Animation Graph Editor (floating window)
            if (ImGui::MenuItem("Animation Graph Editor", nullptr, &g_animGraphUI.showNodeEditor)) {
                // Toggle handled by MenuItem
            }
            
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
            ImGui::MenuItem("Legacy Controls", nullptr, &show_controls_window);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}

// ============================================================================
// Helper Functions for Menu Actions
// ============================================================================

// Update project data from current scene state (for saving)
void SceneUI::updateProjectFromScene(UIContext& ctx) {
    // Delegate to ProjectManager for robust sync (Deletions, Transforms, Procedurals)
    g_ProjectManager.syncProjectToScene(ctx.scene);
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
    
    uint16_t mat_id = MaterialManager::getInstance().getOrCreateMaterialID("Default", def_mat);

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
    
    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (ctx.optix_gpu_ptr) ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
    
    SCENE_LOG_INFO("Added Plane: " + name);
    addViewportMessage("Added Plane: " + name);
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

    uint16_t mat_id = MaterialManager::getInstance().getOrCreateMaterialID("Default", def_mat);

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

    std::string name = "Cube_" + std::to_string(g_ProjectManager.getProjectData().next_object_id);

    for(int i=0; i<36; i+=3) {
        Vec3 v0 = p[indices[i]];
        Vec3 v1 = p[indices[i+1]];
        Vec3 v2 = p[indices[i+2]];
        Vec3 n = (v1-v0).cross(v2-v0).normalize();
        
        auto tri = std::make_shared<Triangle>(v0, v1, v2, n, n, n, Vec2(0,0), Vec2(1,0), Vec2(0,1), mat_id);
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

    rebuildMeshCache(ctx.scene.world.objects);
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    if (ctx.optix_gpu_ptr) ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
    
    SCENE_LOG_INFO("Added Cube: " + name);
    addViewportMessage("Added Cube: " + name);
}

#endif
