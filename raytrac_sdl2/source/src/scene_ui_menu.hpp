#ifndef SCENE_UI_MENU_HPP
#define SCENE_UI_MENU_HPP

// Implementation of SceneUI::drawMainMenuBar
// This file is included by scene_ui.cpp to reduce file size

#include "ProjectManager.h"

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
                 // Check for unsaved changes
                 if (g_ProjectManager.hasUnsavedChanges()) {
                     // TODO: Show "Save changes?" dialog
                 }
                 
                 rendering_stopped_cpu = true;
                 rendering_stopped_gpu = true;
                 
                 // Clear project and scene
                 g_ProjectManager.newProject();
                 ctx.scene.clear();
                 ctx.renderer.resetCPUAccumulation();
                 if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                 
                 // Create default scene
                 createDefaultScene(ctx.scene, ctx.renderer, ctx.optix_gpu_ptr);
                 invalidateCache(); 
                 
                 // Rebuild BVH and OptiX
                 ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                 if (ctx.optix_gpu_ptr) {
                     ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                 }
                 if(ctx.scene.camera) ctx.scene.camera->update_camera_vectors();
                 
                 active_model_path = "Untitled";
                 ctx.start_render = true;
                 
                 SCENE_LOG_INFO("New project created.");
            }

            ImGui::Separator();

            // ================================================================
            // OPEN PROJECT (.rtp / .rts)
            // ================================================================
            if (ImGui::MenuItem("Open Project...", "Ctrl+O")) {
                std::string filepath = openFileDialogW(L"RayTrophi Project (.rtp;.rts)\0*.rtp;*.rts\0All Files\0*.*\0");
                if (!filepath.empty()) {
                    rendering_stopped_cpu = true;
                    rendering_stopped_gpu = true;
                    
                    scene_loading = true;
                    scene_loading_done = false;
                    scene_loading_progress = 0;
                    scene_loading_stage = "Opening project...";
                    
                    std::thread loader_thread([this, filepath, &ctx]() {
                        // Wait for render threads to pause
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        
                        // Use new ProjectManager for .rtp files, fallback to old serializer for .rts
                        std::string ext = filepath.substr(filepath.find_last_of('.'));
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        
                        if (ext == ".rtp") {
                            g_ProjectManager.openProject(filepath, ctx.scene, ctx.render_settings, ctx.renderer, ctx.optix_gpu_ptr,
                                [this](int p, const std::string& s) {
                                    scene_loading_progress = p;
                                    scene_loading_stage = s;
                                });
                        } else {
                            // Legacy .rts format
                            SceneSerializer::Deserialize(ctx.scene, ctx.render_settings, ctx.renderer, ctx.optix_gpu_ptr, filepath);
                        }
                        
                        invalidateCache();
                        active_model_path = g_ProjectManager.getProjectName();
                        
                        if (ctx.optix_gpu_ptr) cudaDeviceSynchronize();
                        
                        scene_loading = false;
                        scene_loading_done = true;
                    });
                    loader_thread.detach();
                    
                    SCENE_LOG_INFO("Opening project: " + filepath);
                }
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
                     scene_loading = true;
                     scene_loading_done = false;
                     scene_loading_progress = 0;
                     scene_loading_stage = "Saving project...";
                     
                     SCENE_LOG_INFO("Starting background save...");
                     
                     std::thread save_thread([this, current_path]() {
                         std::this_thread::sleep_for(std::chrono::milliseconds(50));
                         g_ProjectManager.saveProject(current_path,
                            [this](int p, const std::string& s) {
                                scene_loading_progress = p;
                                scene_loading_stage = s;
                            });
                         scene_loading = false;
                         scene_loading_done = true;
                     });
                     save_thread.detach();
                 } else {
                     // No path yet, prompt Save As
                     std::string filepath = saveFileDialogW(L"RayTrophi Project (.rtp)\0*.rtp\0", L"rtp");
                     if (!filepath.empty()) {
                         scene_loading = true;
                         scene_loading_done = false;
                         scene_loading_progress = 0;
                         scene_loading_stage = "Saving project...";
                         
                         SCENE_LOG_INFO("Starting background save...");
                         
                         std::thread save_thread([this, filepath]() {
                             std::this_thread::sleep_for(std::chrono::milliseconds(50));
                             g_ProjectManager.saveProject(filepath,
                                [this](int p, const std::string& s) {
                                    scene_loading_progress = p;
                                    scene_loading_stage = s;
                                });
                             scene_loading = false;
                             scene_loading_done = true;
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

                     scene_loading = true;
                     scene_loading_done = false;
                     scene_loading_progress = 0;
                     scene_loading_stage = "Saving project...";
                     
                     SCENE_LOG_INFO("Starting background save...");
                     
                     std::thread save_thread([this, filepath]() {
                         std::this_thread::sleep_for(std::chrono::milliseconds(50));
                         g_ProjectManager.saveProject(filepath,
                            [this](int p, const std::string& s) {
                                scene_loading_progress = p;
                                scene_loading_stage = s;
                            });
                         scene_loading = false;
                         scene_loading_done = true;
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
                         bool success = g_ProjectManager.importModel(file, ctx.scene, ctx.renderer, ctx.optix_gpu_ptr,
                             [this](int p, const std::string& s) {
                                 scene_loading_progress = p; 
                                 scene_loading_stage = s;
                             }, false); // false = DO NOT REBUILD in thread (Crash fix)
                         
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
            
            if (ImGui::MenuItem("Export Scene (.glb)...", nullptr)) {
                 // TODO: Export Logic using tinygltf or assimp export
                 SCENE_LOG_WARN("Export not yet implemented.");
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Save Render Image...", nullptr)) {
                ctx.render_settings.save_image_requested = true;
            }
            
            ImGui::Separator();
            
            // Show project info
            ImGui::TextDisabled("Project: %s", g_ProjectManager.getProjectName().c_str());
            if (g_ProjectManager.hasUnsavedChanges()) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1,0.5f,0,1), "*");
            }
            
            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Alt+F4")) {
                 extern bool quit; 
                 quit = true;
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
             }
             ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View"))
        {
            ImGui::MenuItem("Properties Panel", nullptr, &showSidePanel);
            ImGui::MenuItem("Bottom Panel", nullptr, &show_animation_panel);
            ImGui::MenuItem("Log Window", nullptr, &show_scene_log);
            ImGui::EndMenu();
        }
        

        if (ImGui::BeginMenu("Help"))
        {
            ImGui::MenuItem("Controls / Help", "F1", &show_controls_window);
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
}

#endif
