#ifndef SCENE_UI_TERRAIN_HPP
#define SCENE_UI_TERRAIN_HPP

#include "scene_ui.h"
#include "MaterialManager.h" // Added for material selection
#include <TerrainManager.h>
#include "PrincipledBSDF.h" // For layer texture editing

// ===============================================================================
// TERRAIN PANEL UI
// ===============================================================================

void SceneUI::drawTerrainPanel(UIContext& ctx) {
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 0.5f, 1.0f), "TERRAIN SYSTEM");
    ImGui::Separator();

    // -----------------------------------------------------------------------------
    // 1. TERRAIN MANAGEMENT (Create / Import)
    // -----------------------------------------------------------------------------
    if (ImGui::CollapsingHeader("Manage Terrains", ImGuiTreeNodeFlags_DefaultOpen)) {
        static int new_res = 128;
        static float new_size = 100.0f;
        static float import_height = 20.0f;

        // Creation Params
        ImGui::InputInt("Resolution", &new_res);
        ImGui::InputFloat("Size (m)", &new_size);

        if (ImGui::Button("Create Grid Terrain")) {
            auto t = TerrainManager::getInstance().createTerrain(ctx.scene, new_res, new_size);
            if (t) {
                terrain_brush.active_terrain_id = t->id;
                terrain_brush.enabled = true;
                SCENE_LOG_INFO("Terrain created: " + t->name);
                ctx.renderer.resetCPUAccumulation();
                g_bvh_rebuild_pending = true;
                g_optix_rebuild_pending = true;
            }
        }

        ImGui::Separator();

        // Import Params
        ImGui::InputFloat("Max Height (m)", &import_height);

        static int import_res_idx = 0; // Default 512
        const char* res_items[] = { "512 (Fast)", "1024 (Balanced)", "2048 (High)", "4096 (Extreme)" };
        ImGui::Combo("Import Resolution", &import_res_idx, res_items, IM_ARRAYSIZE(res_items));
        
        static int target_res = 512;
        if (import_res_idx == 0) target_res = 512;
        else if (import_res_idx == 1) target_res = 1024;
        else if (import_res_idx == 2) target_res = 2048;
        else if (import_res_idx == 3) target_res = 4096;

        if (ImGui::Button("Import Heightmap...")) {
            std::string path = openFileDialogW(L"Image Files\0*.png;*.jpg;*.jpeg;*.bmp\0");
            if (!path.empty()) {
                auto t = TerrainManager::getInstance().createTerrainFromHeightmap(ctx.scene, path, new_size, import_height, target_res);
                if (t) {
                    terrain_brush.active_terrain_id = t->id;
                    terrain_brush.enabled = true;
                    SCENE_LOG_INFO("Terrain imported from: " + path);
                    ctx.renderer.resetCPUAccumulation();
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    if (ctx.optix_gpu_ptr) {
                        cudaDeviceSynchronize();
                        ctx.optix_gpu_ptr->resetAccumulation();
                    }
                }
            }
        }

        ImGui::Spacing();
        ImGui::Separator();

        if (ImGui::Button("Clear All Terrains")) {
            TerrainManager::getInstance().removeAllTerrains(ctx.scene);
            terrain_brush.active_terrain_id = -1;
            SCENE_LOG_INFO("All terrains cleared.");
            ctx.renderer.resetCPUAccumulation();
            g_bvh_rebuild_pending = true;
            g_optix_rebuild_pending = true;
        }
    }

    ImGui::Spacing();

    // -----------------------------------------------------------------------------
    // 2. MESH QUALITY SETTINGS
    // -----------------------------------------------------------------------------
    if (terrain_brush.active_terrain_id != -1) {
        auto* t = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
        if (t) {
            if (ImGui::CollapsingHeader("Mesh Quality")) {
                // Normal Quality Dropdown
                const char* normalQualityItems[] = { "Fast (4-neighbor)", "Sobel (8-neighbor)", "High Quality" };
                int nq = (int)t->normal_quality;
                if (ImGui::Combo("Normal Quality", &nq, normalQualityItems, IM_ARRAYSIZE(normalQualityItems))) {
                    t->normal_quality = (NormalQuality)nq;
                    t->dirty_mesh = true;
                    TerrainManager::getInstance().updateTerrainMesh(t);
                    ctx.renderer.resetCPUAccumulation();
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                }
                UIWidgets::HelpMarker("Fast: Simple 4-neighbor average\nSobel: Smooth 8-neighbor filter (recommended)\nHigh Quality: Enhanced edge detection");
                
                // Normal Strength Slider
                if (ImGui::SliderFloat("Normal Strength", &t->normal_strength, 0.1f, 3.0f, "%.2f")) {
                    t->dirty_mesh = true;
                    TerrainManager::getInstance().updateTerrainMesh(t);
                    ctx.renderer.resetCPUAccumulation();
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                }
                UIWidgets::HelpMarker("Adjust normal intensity for shading.\n1.0 = Default\n<1.0 = Flatter appearance\n>1.0 = More pronounced details");
                
                // Dirty Sectors Info
                int dirtySectors = t->dirty_region.countDirtySectors();
                if (dirtySectors > 0) {
                    ImGui::TextColored(ImVec4(1.0f, 0.7f, 0.3f, 1.0f), "Dirty Sectors: %d/256", dirtySectors);
                }
            }
            
            ImGui::Spacing();
            
            // -----------------------------------------------------------------------------
            // 3. LAYER MANAGEMENT
            // -----------------------------------------------------------------------------
            ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Layer Management");
            ImGui::Separator();

            if (t->layers.empty()) {
                if (ImGui::Button("Initialize Layers")) {
                    TerrainManager::getInstance().initLayers(t);
                    SCENE_LOG_INFO("Terrain layers initialized for: " + t->name);
                    g_optix_rebuild_pending = true; // Rebuild SBT
                }
            }
            else {
                // Layer Editors
                // Define layer names for auto-creation
                static const char* autoLayerNames[4] = {"Grass", "Rock", "Snow", "Dirt"};
                static const Vec3 autoLayerColors[4] = {
                    Vec3(0.3f, 0.5f, 0.2f),  // Grass: green-ish
                    Vec3(0.4f, 0.4f, 0.4f),  // Rock: gray
                    Vec3(0.9f, 0.9f, 0.95f), // Snow: white
                    Vec3(0.5f, 0.35f, 0.2f)  // Dirt: brown
                };
                
                for (int i = 0; i < 4; i++) {
                    ImGui::PushID(i);
                    std::string layerName = "";
                    ImVec4 layerColor;
                    if (i == 0) { layerName = "Layer 0 (Grass)"; layerColor = ImVec4(0.5, 0.8, 0.5, 1); }
                    else if (i == 1) { layerName = "Layer 1 (Rock)"; layerColor = ImVec4(0.6, 0.6, 0.6, 1); }
                    else if (i == 2) { layerName = "Layer 2 (Snow)"; layerColor = ImVec4(0.9, 0.9, 1.0, 1); }
                    else { layerName = "Layer 3 (Dirt)"; layerColor = ImVec4(0.7, 0.6, 0.4, 1); }

                    ImGui::TextColored(layerColor, "%s", layerName.c_str());

                    // Material Selector
                    std::string currentMatName = "None";
                    if (t->layers[i]) currentMatName = t->layers[i]->materialName;
                    else currentMatName = "[None]";

                    if (ImGui::BeginCombo("Material", currentMatName.c_str())) {
                        auto& materials = MaterialManager::getInstance().getAllMaterials();
                        for (auto& mat : materials) {
                            bool is_selected = (t->layers[i] == mat);
                            if (ImGui::Selectable(mat->materialName.c_str(), is_selected)) {
                                t->layers[i] = mat;
                                g_optix_rebuild_pending = true;
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
                            g_optix_rebuild_pending = true;
                            SCENE_LOG_INFO("Created terrain layer material: " + matName);
                        }
                    }

                    // UV Scale
                    if (i < t->layer_uv_scales.size()) {
                        if (ImGui::DragFloat("UV Scale", &t->layer_uv_scales[i], 0.1f, 0.1f, 1000.0f)) {
                            g_optix_rebuild_pending = true;
                        }
                    }
                    ImGui::Separator();
                    ImGui::PopID();
                }
            }

            ImGui::Spacing();

            // Auto Mask
            UIWidgets::ColoredHeader("Procedural Auto-Mask", ImVec4(0.8f, 0.4f, 1.0f, 1.0f));
            static float am_height_min = 20.0f;
            static float am_height_max = 80.0f;
            static float am_slope = 5.0f;

            ImGui::DragFloat("Height Start (Snow)", &am_height_min, 0.5f, 0.0f, 200.0f);
            ImGui::DragFloat("Height End", &am_height_max, 0.5f, 0.0f, 200.0f);
            ImGui::DragFloat("Slope Steepness", &am_slope, 0.1f, 1.0f, 20.0f);

            if (ImGui::Button("Generate Mask")) {
                TerrainManager::getInstance().autoMask(t, 0.0f, 0.0f, am_height_min, am_height_max, am_slope);
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                g_optix_rebuild_pending = true;
                SCENE_LOG_INFO("Auto-mask generated for: " + t->name);
            }
            
            ImGui::SameLine();
            if (ImGui::Button("Export Splat Map")) {
                if (t->splatMap && !t->splatMap->pixels.empty()) {
                    std::string path = SceneUI::saveFileDialogW(L"PNG Files\0*.png\0", L"png");
                    if (!path.empty()) {
                        TerrainManager::getInstance().exportSplatMap(t, path);
                        SCENE_LOG_INFO("Splat map exported to: " + path);
                    }
                }
            }
            
            // ===============================================================
            // ROCK HARDNESS (for realistic erosion)
            // ===============================================================
            UIWidgets::ColoredHeader("Rock Hardness", ImVec4(0.7f, 0.5f, 0.3f, 1.0f));
            
            static float defaultHardness = 0.3f;
            ImGui::SliderFloat("Default Hardness", &defaultHardness, 0.0f, 1.0f, "%.2f");
            UIWidgets::HelpMarker("0 = Soft (sand/soil), 1 = Hard (bedrock)");
            
            bool hasHardness = !t->hardnessMap.empty();
            if (hasHardness) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Hardness Map: Active (%dx%d)", 
                    t->heightmap.width, t->heightmap.height);
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Hardness Map: Not initialized");
            }
            
            if (ImGui::Button("Init Hardness Map")) {
                TerrainManager::getInstance().initHardnessMap(t, defaultHardness);
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
            }
            ImGui::SameLine();
            if (ImGui::Button("Auto-Generate")) {
                TerrainManager::getInstance().autoGenerateHardness(t, 0.7f, 0.15f);
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
            }
            UIWidgets::HelpMarker("Auto-generate based on slope:\n- Steep = Hard (rock)\n- Flat = Soft (soil)");
            
            // ===============================================================
            // EROSION TOOLS
            // ===============================================================
            UIWidgets::ColoredHeader("Erosion Tools", ImVec4(0.6f, 0.4f, 0.8f, 1.0f));

            static HydraulicErosionParams hydro_params;
            static ThermalErosionParams thermal_params;
            static float wind_strength = 1.0f;
            static float wind_direction = 45.0f;
            static int wind_iters = 50;
            static float fluvial_strength = 1.0f;
            static int erosion_mode = 0;
            static bool use_gpu = true;

            // GPU Toggle (applies to Hydraulic, Fluvial, Thermal)
            ImGui::Checkbox("Use GPU Acceleration", &use_gpu);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Run erosion on GPU (up to 100x faster).\nAll erosion types support GPU acceleration.");
            ImGui::Separator();

            ImGui::RadioButton("Hydraulic", &erosion_mode, 0); ImGui::SameLine();
            ImGui::RadioButton("Fluvial", &erosion_mode, 1); ImGui::SameLine();
            ImGui::RadioButton("Thermal", &erosion_mode, 2); ImGui::SameLine();
            ImGui::RadioButton("Wind", &erosion_mode, 3);

            if (erosion_mode == 0) {
                // HYDRAULIC EROSION
                UIWidgets::ColoredHeader("Hydraulic Erosion", ImVec4(0.4f, 0.6f, 1.0f, 1.0f));
                
                static int current_preset = -1;
                auto detectPreset = [&]() -> int {
                    if (hydro_params.iterations == 10000 && hydro_params.erodeSpeed == 0.2f) return 0;
                    if (hydro_params.iterations == 50000 && hydro_params.erodeSpeed == 0.3f) return 1;
                    if (hydro_params.iterations == 200000 && hydro_params.erodeSpeed == 0.5f) return 2;
                    if (hydro_params.iterations == 500000 && hydro_params.erodeSpeed == 0.8f) return 3;
                    return -1;
                };
                current_preset = detectPreset();
                
                const char* presetNames[] = { "Light", "Medium", "Heavy", "Extreme" };
                ImVec4 activeColor(0.3f, 0.8f, 0.3f, 1.0f);
                
                ImGui::Text("Presets:"); ImGui::SameLine();
                if (current_preset == 0) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                if (ImGui::SmallButton("Light")) { hydro_params.iterations = 10000; hydro_params.erodeSpeed = 0.2f; hydro_params.depositSpeed = 0.4f; hydro_params.sedimentCapacity = 3.0f; }
                if (current_preset == 0) ImGui::PopStyleColor();
                ImGui::SameLine();
                if (current_preset == 1) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                if (ImGui::SmallButton("Medium")) { hydro_params.iterations = 50000; hydro_params.erodeSpeed = 0.3f; hydro_params.depositSpeed = 0.3f; hydro_params.sedimentCapacity = 4.0f; }
                if (current_preset == 1) ImGui::PopStyleColor();
                ImGui::SameLine();
                if (current_preset == 2) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                if (ImGui::SmallButton("Heavy")) { hydro_params.iterations = 200000; hydro_params.erodeSpeed = 0.5f; hydro_params.depositSpeed = 0.2f; hydro_params.sedimentCapacity = 6.0f; }
                if (current_preset == 2) ImGui::PopStyleColor();
                ImGui::SameLine();
                if (current_preset == 3) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                if (ImGui::SmallButton("Extreme")) { hydro_params.iterations = 500000; hydro_params.erodeSpeed = 0.8f; hydro_params.depositSpeed = 0.1f; hydro_params.sedimentCapacity = 10.0f; }
                if (current_preset == 3) ImGui::PopStyleColor();
                ImGui::SameLine();
                if (current_preset >= 0) ImGui::TextColored(activeColor, "(%s)", presetNames[current_preset]);
                else ImGui::TextDisabled("(Custom)");
                
                ImGui::DragInt("Iterations##H", &hydro_params.iterations, 1000, 1000, 500000);
                ImGui::DragFloat("Erosion Speed", &hydro_params.erodeSpeed, 0.01f, 0.0f, 1.0f);
                ImGui::DragFloat("Deposit Speed", &hydro_params.depositSpeed, 0.01f, 0.0f, 1.0f);
                ImGui::DragFloat("Inertia", &hydro_params.inertia, 0.01f, 0.0f, 1.0f);
                ImGui::DragFloat("Sediment Capacity", &hydro_params.sedimentCapacity, 0.1f, 0.1f, 10.0f);
                ImGui::DragInt("Brush Radius", &hydro_params.erosionRadius, 1, 1, 8);

                if (ImGui::Button("Apply Hydraulic Erosion")) {
                    if (use_gpu) TerrainManager::getInstance().hydraulicErosionGPU(t, hydro_params);
                    else TerrainManager::getInstance().hydraulicErosion(t, hydro_params);
                    ctx.renderer.resetCPUAccumulation();
                    g_bvh_rebuild_pending = true; g_optix_rebuild_pending = true;
                    if (ctx.optix_gpu_ptr) { cudaDeviceSynchronize(); ctx.optix_gpu_ptr->resetAccumulation(); }
                }
            }
            else if (erosion_mode == 1) {
                // FLUVIAL EROSION
                UIWidgets::ColoredHeader("Fluvial Erosion (River Networks)", ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
                ImGui::DragFloat("Erosion Strength##F", &fluvial_strength, 0.1f, 0.1f, 5.0f);
                UIWidgets::HelpMarker("Controls river channel depth and width.");
                ImGui::TextWrapped("Creates realistic river networks using flow accumulation.");
                
                if (ImGui::Button("Apply Fluvial Erosion")) {
                    HydraulicErosionParams fluvParams;
                    fluvParams.erodeSpeed = fluvial_strength;
                    fluvParams.depositSpeed = 0.1f;
                    fluvParams.iterations = 2000;
                    if (use_gpu) TerrainManager::getInstance().fluvialErosionGPU(t, fluvParams);
                    else TerrainManager::getInstance().fluvialErosion(t, fluvParams);
                    ctx.renderer.resetCPUAccumulation();
                    g_bvh_rebuild_pending = true; g_optix_rebuild_pending = true;
                    if (ctx.optix_gpu_ptr) { cudaDeviceSynchronize(); ctx.optix_gpu_ptr->resetAccumulation(); }
                }
            }
            else if (erosion_mode == 2) {
                // THERMAL EROSION
                UIWidgets::ColoredHeader("Thermal Erosion (Slope Collapse)", ImVec4(0.8f, 0.5f, 0.4f, 1.0f));
                ImGui::DragInt("Iterations##T", &thermal_params.iterations, 1, 1, 200);
                ImGui::DragFloat("Talus Angle", &thermal_params.talusAngle, 0.01f, 0.1f, 1.0f);
                ImGui::DragFloat("Amount", &thermal_params.erosionAmount, 0.1f, 0.1f, 2.0f);

                if (ImGui::Button("Apply Thermal Erosion")) {
                    if (use_gpu) TerrainManager::getInstance().thermalErosionGPU(t, thermal_params);
                    else TerrainManager::getInstance().thermalErosion(t, thermal_params);
                    ctx.renderer.resetCPUAccumulation();
                    g_bvh_rebuild_pending = true; g_optix_rebuild_pending = true;
                    if (ctx.optix_gpu_ptr) { cudaDeviceSynchronize(); ctx.optix_gpu_ptr->resetAccumulation(); }
                }
            }
            else if (erosion_mode == 3) {
                // WIND EROSION
                UIWidgets::ColoredHeader("Wind Erosion", ImVec4(0.7f, 0.7f, 0.5f, 1.0f));
                ImGui::DragFloat("Wind Strength", &wind_strength, 0.1f, 0.1f, 10.0f);
                ImGui::DragFloat("Direction (deg)", &wind_direction, 1.0f, 0.0f, 360.0f);
                ImGui::DragInt("Iterations##W", &wind_iters, 1, 1, 200);

                if (ImGui::Button("Apply Wind Erosion")) {
                    if (use_gpu) TerrainManager::getInstance().windErosionGPU(t, wind_strength, wind_direction, wind_iters);
                    else TerrainManager::getInstance().windErosion(t, wind_strength, wind_direction, wind_iters);
                    ctx.renderer.resetCPUAccumulation();
                    g_bvh_rebuild_pending = true; g_optix_rebuild_pending = true;
                    if (ctx.optix_gpu_ptr) { cudaDeviceSynchronize(); ctx.optix_gpu_ptr->resetAccumulation(); }
                }
            }

            // Wizard Mode
            ImGui::Separator();
            UIWidgets::ColoredHeader("Erosion Wizard (Combined)", ImVec4(0.4f, 0.8f, 0.5f, 1.0f));
            static int wiz_iters = 50;
            static float wiz_strength = 1.0f;
            ImGui::DragInt("Wizard Steps", &wiz_iters, 1, 10, 200);
            ImGui::DragFloat("Wizard Strength", &wiz_strength, 0.1f, 0.1f, 5.0f);
            
            if (ImGui::Button("Run Combined Process (Slow!)", ImVec2(-1, 0))) {
                TerrainManager::getInstance().applyCombinedErosion(t, wiz_iters, wiz_strength, use_gpu);
                ctx.renderer.resetCPUAccumulation();
                g_bvh_rebuild_pending = true;
                g_optix_rebuild_pending = true;
                if (ctx.optix_gpu_ptr) {
                    cudaDeviceSynchronize();
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
            }

            // Timeline Integration
            ImGui::Separator();
            UIWidgets::ColoredHeader("Timeline & Keyframes", ImVec4(0.7f, 0.4f, 0.8f, 1.0f));
            
            int current_frame = ctx.scene.timeline.current_frame;
            ImGui::Text("Current Frame: %d", current_frame);
            
            std::string trackName = t->name.empty() ? "Terrain" : t->name;
            bool hasTrack = ctx.scene.timeline.tracks.find(trackName) != ctx.scene.timeline.tracks.end();
            
            if (ImGui::Button("Capture State Keyframe")) {
                // Get or create track
                ObjectAnimationTrack& track = ctx.scene.timeline.tracks[trackName]; // Creates if not exists
                track.object_name = trackName;
                track.object_index = -1; // Not a standard object list item maybe? Or find index
                
                TerrainManager::getInstance().captureKeyframeToTrack(t, track, current_frame);
                //ctx.status_message = "Terrain Keyframe Captured at Frame " + std::to_string(current_frame);
            }
            
            if (hasTrack) {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Track Exists");
                
                // Show keyframe count for this track
                int kf_count = (int)ctx.scene.timeline.tracks[trackName].keyframes.size();
                ImGui::SameLine();
                ImGui::TextDisabled("(%d keys)", kf_count);
                
                // NOTE: Animation updates are now handled globally in TimelineWidget::draw()
                // This ensures terrain animation works regardless of which panel is open
            } else {
                ImGui::SameLine();
                ImGui::TextDisabled("(No Track)");
            }

            // Heightmap Export
            ImGui::Separator();
            if (ImGui::Button("Export Heightmap (16-bit RAW)")) {
                std::string path = saveFileDialogW(L"RAW Files\0*.raw\0");
                if (!path.empty()) {
                    TerrainManager::getInstance().exportHeightmap(t, path);
                }
            }
        }

        ImGui::Spacing();

        // -----------------------------------------------------------------------------
        // 3. SCULPTING & PAINTING
        // -----------------------------------------------------------------------------
        UIWidgets::ColoredHeader("Tools (Sculpt/Paint)", ImVec4(1.0f, 0.7f, 0.4f, 1.0f));

        ImGui::Checkbox("Enable Tool", &terrain_brush.enabled);

        if (terrain_brush.enabled) {
            // Active Terrain Selector (Same as before)
            auto& terrains = TerrainManager::getInstance().getTerrains();
            if (terrains.empty()) {
                ImGui::TextColored(ImVec4(1, 0, 0, 1), "No Terrains available.");
            }
            else {
                std::string current_name = "None";
                if (terrain_brush.active_terrain_id != -1) {
                    auto* t = TerrainManager::getInstance().getTerrain(terrain_brush.active_terrain_id);
                    if (t) current_name = t->name + " (ID: " + std::to_string(t->id) + ")";
                }

                if (ImGui::BeginCombo("Active Terrain", current_name.c_str())) {
                    for (auto& t : terrains) {
                        bool is_selected = (t.id == terrain_brush.active_terrain_id);
                        std::string label = t.name + " (ID: " + std::to_string(t.id) + ")";
                        if (ImGui::Selectable(label.c_str(), is_selected)) {
                            terrain_brush.active_terrain_id = t.id;
                        }
                        if (is_selected) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                ImGui::Separator();

                // Brush Settings
                ImGui::RadioButton("Raise", &terrain_brush.mode, 0); ImGui::SameLine();
                ImGui::RadioButton("Lower", &terrain_brush.mode, 1); ImGui::SameLine();
                ImGui::RadioButton("Flatten", &terrain_brush.mode, 2); ImGui::SameLine();
                ImGui::RadioButton("Smooth", &terrain_brush.mode, 3); ImGui::SameLine();
                ImGui::RadioButton("Stamp", &terrain_brush.mode, 4);
                ImGui::RadioButton("Paint Splat", &terrain_brush.mode, 5); ImGui::SameLine();
                ImGui::RadioButton("Paint Hard", &terrain_brush.mode, 6); ImGui::SameLine();
                ImGui::RadioButton("Paint Soft", &terrain_brush.mode, 7);

                if (terrain_brush.mode == 2) { // Flatten Settings
                    ImGui::Indent();
                    ImGui::Checkbox("Use Fixed Height", &terrain_brush.use_fixed_height);
                    if (terrain_brush.use_fixed_height) {
                        ImGui::DragFloat("Target Height", &terrain_brush.flatten_target, 0.1f, 0.0f, 100.0f);
                    }
                    ImGui::Unindent();
                }
                else if (terrain_brush.mode == 4) { // Stamp Settings
                    ImGui::Indent();
                    ImGui::SliderFloat("Rotation", &terrain_brush.stamp_rotation, 0.0f, 360.0f);
                    if (ImGui::Button("Load Stamp Texture...")) {
                        std::string path = SceneUI::openFileDialogW(L"Image Files\0*.png;*.jpg;*.jpeg;*.bmp\0");
                        if (!path.empty()) {
                            terrain_brush.stamp_texture = std::make_shared<Texture>(path, TextureType::Albedo);
                        }
                    }
                    if (terrain_brush.stamp_texture) {
                        ImGui::Text("Stamp: %s", terrain_brush.stamp_texture->name.c_str());
                    }
                    ImGui::Unindent();
                }
                else if (terrain_brush.mode == 5) { // Paint Settings
                    ImGui::Indent();
                    const char* channels[] = { "Layer 0 (Red)", "Layer 1 (Green)", "Layer 2 (Blue)", "Layer 3 (Alpha)" };
                    ImGui::Combo("Channel", &terrain_brush.paint_channel, channels, IM_ARRAYSIZE(channels));
                    ImGui::Unindent();
                }

                ImGui::SliderFloat("Radius", &terrain_brush.radius, 1.0f, 100.0f);
                ImGui::SliderFloat("Strength", &terrain_brush.strength, 0.1f, 5.0f);

                ImGui::Checkbox("Show Preview", &terrain_brush.show_preview);

                UIWidgets::HelpMarker("Left Click to Sculpt/Paint.\nPaint Splat modifies texture layers.\nPaint Hard/Soft modifies erosion resistance.");
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
    if (ctx.scene.world.hit(r, 0.001f, 1e9f, rec)) {
        bool is_terrain = (rec.materialID == terrain->material_id);
        
        if (is_terrain) {
            Vec3 hitPoint = rec.point;
            
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
                         ctx.optix_gpu_ptr->updateTLASGeometry(ctx.scene.world.objects);
                     }
                     g_bvh_rebuild_pending = true;
                 }
            }
        }
    }
}

#endif
