#ifndef SCENE_UI_TERRAIN_HPP
#define SCENE_UI_TERRAIN_HPP

#include "scene_ui.h"
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
                // Don't auto-enable edit tool - let user manually enable if needed
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
                    // Don't auto-enable edit tool - let user manually enable if needed
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
            // GPU sync BEFORE removal - ensure OptiX is not using terrain data
            if (ctx.optix_gpu_ptr) {
                cudaDeviceSynchronize();
            }
            TerrainManager::getInstance().removeAllTerrains(ctx.scene);
            terrain_brush.active_terrain_id = -1;
            SCENE_LOG_INFO("All terrains cleared.");
            ctx.renderer.resetCPUAccumulation();
            g_bvh_rebuild_pending = true;
            g_optix_rebuild_pending = true;
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->resetAccumulation();
            }
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
                        if (ImGui::DragFloat("UV Scale", &t->layer_uv_scales[i], 0.1f, 0.1f, 1000.0f)) {
                            g_optix_rebuild_pending = true;
                        }
                    
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
                            g_optix_rebuild_pending = true;
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
                                        TextureType type = isNormal ? TextureType::Normal : TextureType::Albedo;
                                        if (std::string(label) == "Roughness") type = TextureType::Roughness;
                                        
                                        texSlot = std::make_shared<Texture>(path, type);
                                        g_optix_rebuild_pending = true;
                                        ctx.renderer.resetCPUAccumulation();
                                        if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                                    }
                                }
                                
                                if (texSlot) {
                                    ImGui::SameLine();
                                    if (ImGui::SmallButton((std::string("X##") + label).c_str())) {
                                        texSlot = nullptr;
                                        g_optix_rebuild_pending = true;
                                        ctx.renderer.resetCPUAccumulation();
                                        if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                                    }
                                }
                            };

                            DrawTextureSlot("Albedo", pMat->albedoProperty.texture);
                            DrawTextureSlot("Normal", pMat->normalProperty.texture, true);
                            DrawTextureSlot("Roughness", pMat->roughnessProperty.texture); 
                            
                            ImGui::Unindent();
                        }
                    }
                    
                    // FOLIAGE UI INTEGRATION
                    ImGui::Spacing();
                    if (ImGui::TreeNode(("Foliage Settings##" + std::to_string(i)).c_str())) {
                        // Ensure foliage layers exist
                        if (t->foliageLayers.size() <= i) {
                            t->foliageLayers.resize(4);
                            // Set defaults for new layers
                            t->foliageLayers[i].targetMaskLayerId = i;
                            t->foliageLayers[i].name = std::string(autoLayerNames[i]) + " Foliage";
                        }
                        
                        auto& fLayer = t->foliageLayers[i];
                        
                        ImGui::Checkbox("Enable Foliage", &fLayer.enabled);
                        if (fLayer.enabled) {
                            // Mesh Selection
                            std::string meshName = fLayer.meshPath.empty() ? "[No Mesh]" : std::filesystem::path(fLayer.meshPath).filename().string();
                            ImGui::Text("Mesh: %s", meshName.c_str());
                            if (ImGui::SameLine(); ImGui::SmallButton("Load Mesh...")) {
                                std::string path = SceneUI::openFileDialogW(L"Mesh Files\0*.obj;*.fbx;*.glb;*.gltf\0");
                                if (!path.empty()) {
                                    fLayer.meshPath = path;
                                    // Trigger load? For V1 we rely on existing Scene loading or need a dedicated loader.
                                    // Ideally we load it now into OptixAccelManager blindly? 
                                    // For now, let's look up if it's already in scene or add it.
                                    // Using a hack: we need a BLAS ID. 
                                    // We can try to assume it's loaded as a "FoliageAsset"?
                                    // For this iteration, we will just store the path. 
                                    // Backend updateFoliage will handle loading if we improve it, 
                                    // OR we force user to pick from loaded objects?
                                    // "Pick from Loaded Objects" is safer for V1.
                                }
                            }

                            // Mesh ID (Debug / Manual Override)
                            // ImGui::InputInt("Mesh BLAS ID", &fLayer.meshId);

                            // TEMPORARY: Pick from existing scene objects to get BLAS ID
                            // Source Object Selection
                            // NEW: Select by node name from scene objects (like scatter brush)
                            // This captures actual triangles for cloning
                            std::string selectedName = "Select Object...";
                            if (!fLayer.meshPath.empty() && !fLayer.sourceTriangles.empty()) {
                                selectedName = fLayer.meshPath + " (" + std::to_string(fLayer.sourceTriangles.size()) + " tris)";
                            } else if (!fLayer.meshPath.empty()) {
                                selectedName = fLayer.meshPath + " (no tris)";
                            }
                            
                            if (ImGui::BeginCombo("Source Object", selectedName.c_str())) {
                                // Auto-recapture: If meshPath is set but sourceTriangles is empty, recapture now
                                if (!fLayer.meshPath.empty() && fLayer.sourceTriangles.empty()) {
                                    for (const auto& obj : ctx.scene.world.objects) {
                                        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                        if (tri && tri->getNodeName() == fLayer.meshPath) {
                                            fLayer.sourceTriangles.push_back(tri);
                                        }
                                    }
                                    if (!fLayer.sourceTriangles.empty()) {
                                        fLayer.calculateMeshCenter();
                                        SCENE_LOG_INFO("[Foliage] Auto-recaptured '" + fLayer.meshPath + 
                                                      "' with " + std::to_string(fLayer.sourceTriangles.size()) + " triangles");
                                    }
                                }
                                
                                // Build unique node names from scene
                                std::set<std::string> nodeNames;
                                for (const auto& obj : ctx.scene.world.objects) {
                                    auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                    if (tri) {
                                        std::string name = tri->getNodeName();
                                        // Skip terrain and scatter instances
                                        if (!name.empty() && 
                                            name.find("Terrain") == std::string::npos &&
                                            name.find("_inst_") == std::string::npos &&
                                            name.find("Foliage_") == std::string::npos) {
                                            nodeNames.insert(name);
                                        }
                                    }
                                }
                                
                                for (const auto& nodeName : nodeNames) {
                                    bool is_selected = (fLayer.meshPath == nodeName);
                                    if (ImGui::Selectable(nodeName.c_str(), is_selected)) {
                                        // Capture all triangles with this node name
                                        fLayer.meshPath = nodeName;
                                        fLayer.sourceTriangles.clear();
                                        
                                        for (const auto& obj : ctx.scene.world.objects) {
                                            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                            if (tri && tri->getNodeName() == nodeName) {
                                                fLayer.sourceTriangles.push_back(tri);
                                            }
                                        }
                                        
                                        // Calculate mesh center for proper placement
                                        fLayer.calculateMeshCenter();
                                        
                                        SCENE_LOG_INFO("[Foliage] Set source '" + nodeName + 
                                                      "' with " + std::to_string(fLayer.sourceTriangles.size()) + " triangles");
                                    }
                                    if (is_selected) ImGui::SetItemDefaultFocus();
                                }
                                
                                if (nodeNames.empty()) {
                                    ImGui::TextDisabled("No objects in scene (load a model first)");
                                }
                                
                                ImGui::EndCombo();
                            }
                            
                            // Let's implement basic properties first
                            ImGui::DragInt("Density", &fLayer.density, 10, 0, 10000);
                            ImGui::SliderFloat("Threshold", &fLayer.maskThreshold, 0.0f, 1.0f);
                            
                            ImGui::DragFloat2("Scale Range", &fLayer.scaleRange.x, 0.01f, 0.1f, 5.0f);
                            ImGui::DragFloat2("Rot Range (Y)", &fLayer.rotationRange.x, 1.0f, 0.0f, 360.0f);
                            
                            ImGui::Spacing();
                            ImGui::Separator();
                            
                            // === BRUSH MODE ===
                            bool brushActive = foliage_brush.enabled && foliage_brush.active_layer_name == fLayer.name;
                            
                            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Paint Mode");
                            if (ImGui::Checkbox("Enable Brush", &brushActive)) {
                                if (brushActive) {
                                    foliage_brush.enabled = true;
                                    foliage_brush.active_layer_name = fLayer.name;
                                    
                                    // Ensure source is captured
                                    if (!fLayer.meshPath.empty() && fLayer.sourceTriangles.empty()) {
                                        for (const auto& obj : ctx.scene.world.objects) {
                                            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                            if (tri && tri->getNodeName() == fLayer.meshPath) {
                                                fLayer.sourceTriangles.push_back(tri);
                                            }
                                        }
                                        if (!fLayer.sourceTriangles.empty()) {
                                            fLayer.calculateMeshCenter();
                                        }
                                    }
                                } else {
                                    foliage_brush.enabled = false;
                                }
                            }
                            
                            if (brushActive) {
                                ImGui::SameLine();
                                ImGui::TextDisabled("(Click terrain to paint)");
                                
                                ImGui::SliderFloat("Brush Radius", &foliage_brush.radius, 1.0f, 50.0f);
                                ImGui::SliderInt("Instances/Stroke", &foliage_brush.density, 1, 20);
                                
                                const char* modes[] = { "Add", "Remove" };
                                ImGui::Combo("Mode", &foliage_brush.mode, modes, 2);
                            }
                            
                            ImGui::Spacing();
                            
                            // === SCATTER NOW (Bulk scatter) ===
                            if (ImGui::Button("Scatter Now")) {
                                // Auto-recapture source if needed
                                if (!fLayer.meshPath.empty() && fLayer.sourceTriangles.empty()) {
                                    SCENE_LOG_INFO("[Foliage] Trying to recapture source: '" + fLayer.meshPath + "'");
                                    for (const auto& obj : ctx.scene.world.objects) {
                                        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                        if (tri && tri->getNodeName() == fLayer.meshPath) {
                                            fLayer.sourceTriangles.push_back(tri);
                                        }
                                    }
                                    if (!fLayer.sourceTriangles.empty()) {
                                        fLayer.calculateMeshCenter();
                                        SCENE_LOG_INFO("[Foliage] Auto-recaptured sources.");
                                    }
                                }
                                
                                if (!fLayer.hasValidSource()) {
                                    SCENE_LOG_WARN("[Foliage] No valid source - select an object first");
                                } else {
                                    // 1. Legacy Cleanup (Remove old non-instanced objects)
                                    // This cleans up objects from the previous system to prevent duplicates
                                    std::string legacyPrefix = "Foliage_" + t->name + "_" + fLayer.name + "_";
                                    auto& objects = ctx.scene.world.objects;
                                    objects.erase(
                                        std::remove_if(objects.begin(), objects.end(),
                                            [&legacyPrefix](const std::shared_ptr<Hittable>& obj) {
                                                auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                                if (tri) return tri->getNodeName().find(legacyPrefix) == 0;
                                                return false;
                                            }),
                                        objects.end()
                                    );
                                    
                                    // 2. Instance Group Setup (New System)
                                    std::string groupName = "Foliage_" + t->name + "_" + fLayer.name;
                                    InstanceManager& im = InstanceManager::getInstance();
                                    InstanceGroup* group = im.findGroupByName(groupName);
                                    
                                    if (!group) {
                                        int newId = im.createGroup(groupName, fLayer.meshPath, fLayer.sourceTriangles);
                                        group = im.getGroup(newId);
                                    }
                                    
                                    if (group) {
                                        // Clear existing instances in the group to restart scatter
                                        group->clearInstances();
                                        
                                        // Sync settings
                                        group->brush_settings.scale_min = fLayer.scaleRange.x;
                                        group->brush_settings.scale_max = fLayer.scaleRange.y;
                                        group->brush_settings.rotation_random_y = (fLayer.rotationRange.y - fLayer.rotationRange.x);

                                        // Scatter Parameters
                                        int targetCount = fLayer.density;
                                        std::mt19937 rng(12345 + t->id);
                                        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                                        
                                        int spawnedCount = 0;
                                        int maxAttempts = targetCount * 5;
                                        
                                        SCENE_LOG_INFO("[Foliage] Starting scatter: " + std::to_string(targetCount) + " instances");
                                        
                                        for (int attempt = 0; attempt < maxAttempts && spawnedCount < targetCount; ++attempt) {
                                            float u = dist(rng);
                                            float v = dist(rng);
                                            
                                            // Check heightmap mask
                                            int gx = (int)(u * (t->heightmap.width - 1));
                                            int gy = (int)(v * (t->heightmap.height - 1));
                                            
                                            float maskValue = 1.0f;
                                            if (t->splatMap && t->splatMap->is_loaded()) {
                                                Vec3 col = t->splatMap->get_color(u, v);
                                                int ch = fLayer.targetMaskLayerId;
                                                if (ch == 0) maskValue = col.x;
                                                else if (ch == 1) maskValue = col.y;
                                                else if (ch == 2) maskValue = col.z;
                                                else maskValue = t->splatMap->get_alpha(u, v);
                                            }
                                            
                                            if (maskValue < fLayer.maskThreshold) continue;
                                            
                                            // Calculate Position
                                            float terrainX = (u - 0.5f) * t->heightmap.scale_xz;
                                            float terrainZ = (v - 0.5f) * t->heightmap.scale_xz;
                                            float terrainY = t->heightmap.getHeight(gx, gy);
                                            
                                            float x = terrainX;
                                            float y = terrainY;
                                            float z = terrainZ;
                                            
                                            // Random Scale & Rotation
                                            float scale = fLayer.scaleRange.x + (fLayer.scaleRange.y - fLayer.scaleRange.x) * dist(rng);
                                            float rotY_deg = fLayer.rotationRange.x + (fLayer.rotationRange.y - fLayer.rotationRange.x) * dist(rng);
                                            
                                            // Add Instance
                                            InstanceTransform inst;
                                            inst.position = Vec3(x, y, z);
                                            inst.scale = Vec3(scale, scale, scale);
                                            inst.rotation = Vec3(0, rotY_deg, 0);
                                            group->addInstance(inst);
                                            
                                            spawnedCount++;
                                        }
                                        
                                        // Sync to Scene
                                        SceneUI::syncInstancesToScene(ctx, *group, false);
                                        
                                        // Rebuild
                                        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                                        ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                                        ctx.renderer.resetCPUAccumulation();
                                        if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                                        
                                        SCENE_LOG_INFO("[Foliage] Scattered " + std::to_string(spawnedCount) + " instances.");
                                    }
                                }
                            }
                            ImGui::SameLine();
                            if (ImGui::Button("Clear")) {
                                // 1. Legacy Cleanup
                                std::string legacyPrefix = "Foliage_" + t->name + "_" + fLayer.name + "_";
                                auto& objects = ctx.scene.world.objects;
                                objects.erase(
                                    std::remove_if(objects.begin(), objects.end(),
                                        [&legacyPrefix](const std::shared_ptr<Hittable>& obj) {
                                            auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                            if (tri) return tri->getNodeName().find(legacyPrefix) == 0;
                                            return false;
                                        }),
                                    objects.end()
                                );
                                
                                // 2. InstanceGroup Cleanup
                                std::string groupName = "Foliage_" + t->name + "_" + fLayer.name;
                                InstanceGroup* group = InstanceManager::getInstance().findGroupByName(groupName);
                                if (group) {
                                    group->clearInstances();
                                    SceneUI::syncInstancesToScene(ctx, *group, true); // Force remove from scene
                                }
                                
                                // Rebuild
                                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                                ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                                ctx.renderer.resetCPUAccumulation();
                                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                            }
                        }
                        ImGui::TreePop();
                    }
                    
                    ImGui::Separator();
                    ImGui::PopID();
                }
            }

            ImGui::Spacing();

            // Auto Mask
            UIWidgets::ColoredHeader("Procedural Auto-Mask", ImVec4(0.8f, 0.4f, 1.0f, 1.0f));
            static float am_height_min = 5.0f;
            static float am_height_max = 20.0f;
            static float am_slope = 5.0f;

            ImGui::DragFloat("Height Start (Snow)", &am_height_min, 0.1f, 0.0f, 50.0f);
            ImGui::DragFloat("Height End", &am_height_max, 0.1f, 0.0f, 50.0f);
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

// ═══════════════════════════════════════════════════════════════════════════════
// TERRAIN FOLIAGE BRUSH - Paint to add/remove foliage instances
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::handleTerrainFoliageBrush(UIContext& ctx) {
    if (!foliage_brush.enabled) return;
    if (ImGui::GetIO().WantCaptureMouse) return;
    
    // Get active terrain
    auto& terrains = TerrainManager::getInstance().getTerrains();
    if (terrains.empty()) return;
    
    TerrainObject* t = nullptr;
    TerrainFoliageLayer* activeLayer = nullptr;
    
    // Find active terrain and layer by name
    for (auto& terrain : terrains) {
        for (auto& fLayer : terrain.foliageLayers) {
            if (fLayer.name == foliage_brush.active_layer_name) {
                t = &terrain;
                activeLayer = &fLayer;
                break;
            }
        }
        if (t) break;
    }
    
    if (!t || !activeLayer) return;
    auto& fLayer = *activeLayer;
    
    // Auto-capture source if missing but meshPath is set
    // This fixes the issue where user enables brush but hasn't clicked "Scatter Now" yet
    if (fLayer.sourceTriangles.empty() && !fLayer.meshPath.empty()) {
        for (const auto& obj : ctx.scene.world.objects) {
             auto tri = std::dynamic_pointer_cast<Triangle>(obj);
             if (tri && tri->getNodeName() == fLayer.meshPath) {
                 fLayer.sourceTriangles.push_back(tri);
             }
        }
        if (!fLayer.sourceTriangles.empty()) {
             fLayer.calculateMeshCenter();
        }
    }
    
    if (!fLayer.hasValidSource()) return;
    
    // Check for mouse click on terrain
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) return;
    
    // Throttle painting (paint every 200ms)
    static float last_paint_time = 0.0f;
    float current_time = static_cast<float>(ImGui::GetTime());
    if (current_time - last_paint_time < 0.2f) return;
    last_paint_time = current_time;
    
    // Get mouse position and raycast to terrain
    ImVec2 mp = ImGui::GetMousePos();
    
    // Use globals for width/height
    extern int image_width, image_height;
    int width = image_width;
    int height = image_height;
    
    float u = mp.x / width;
    float v = mp.y / height;
    
    Ray ray;
    if (ctx.scene.camera) {
        ray = ctx.scene.camera->get_ray(u, 1.0f - v);
    } else {
        return;
    }
    
    HitRecord hit;
    if (!ctx.scene.bvh || !ctx.scene.bvh->hit(ray, 0.001f, 10000.0f, hit)) return;
    
    // Check if we hit terrain
    if (!hit.triangle) return;
    std::string hitName = hit.triangle->getNodeName();
    // Simplified check: checking for "Terrain" string might fail if user renamed object.
    // Since we are raycasting against the whole scene, we assume if user clicks on something while 
    // aiming at terrain, they intend to paint. 
    // Ideally we would check if hit object is the active terrain, but we don't have easy pointer comparison here 
    // without iterating objects.
    
    Vec3 hitPoint = hit.point;
    
    // Declarations for flags
    extern bool g_bvh_rebuild_pending;
    extern bool g_optix_rebuild_pending;
    
    // Verify InstanceGroup exists for this layer
    std::string groupName = "Foliage_" + t->name + "_" + fLayer.name;
    InstanceManager& im = InstanceManager::getInstance();
    InstanceGroup* group = im.findGroupByName(groupName);

    if (!group) {
        // Create new group if missing
        int newId = im.createGroup(groupName, fLayer.meshPath, fLayer.sourceTriangles);
        group = im.getGroup(newId);
    }

    if (group) {
        // Sync settings
        group->brush_settings.scale_min = fLayer.scaleRange.x;
        group->brush_settings.scale_max = fLayer.scaleRange.y;
        group->brush_settings.rotation_random_y = (fLayer.rotationRange.y - fLayer.rotationRange.x);

        if (foliage_brush.mode == 0) {
            // ADD MODE - Add instances to group
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            
            int added = 0;
            for (int i = 0; i < foliage_brush.density; ++i) {
                // Random offset within brush radius
                float angle = dist(rng) * 6.28318f;
                float r = dist(rng) * foliage_brush.radius;
                float offsetX = cosf(angle) * r;
                float offsetZ = sinf(angle) * r;
                
                // Position (XZ)
                float x = hitPoint.x + offsetX;
                float z = hitPoint.z + offsetZ;
                
                // Get Y from heightmap
                float scale = t->heightmap.scale_xz;
                float u_pos = (x / scale) + 0.5f;
                float v_pos = (z / scale) + 0.5f;
                u_pos = std::clamp(u_pos, 0.0f, 1.0f);
                v_pos = std::clamp(v_pos, 0.0f, 1.0f);
                
                int gx = (int)(u_pos * (t->heightmap.width - 1));
                int gy = (int)(v_pos * (t->heightmap.height - 1));
                float y = t->heightmap.getHeight(gx, gy);
                
                // Properties
                float instanceScale = fLayer.scaleRange.x + (fLayer.scaleRange.y - fLayer.scaleRange.x) * dist(rng);
                float rotY_deg = fLayer.rotationRange.x + (fLayer.rotationRange.y - fLayer.rotationRange.x) * dist(rng);
                
                InstanceTransform inst;
                inst.position = Vec3(x, y, z);
                inst.scale = Vec3(instanceScale, instanceScale, instanceScale);
                inst.rotation = Vec3(0, rotY_deg, 0); 
                
                group->addInstance(inst);
                added++;
            }
            
            if (added > 0) {
                hud_captured_mouse = true;
                
                // Sync group to scene (updates renderable objects)
                SceneUI::syncInstancesToScene(ctx, *group, false);
                
                // Rebuild acceleration structures
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                
                g_bvh_rebuild_pending = true;
                g_optix_rebuild_pending = true;
                
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                
                SCENE_LOG_INFO("[Foliage Brush] Added " + std::to_string(added) + " instances to " + groupName);
            }
        }
        else if (foliage_brush.mode == 1) {
            // REMOVE MODE
            size_t before = group->getInstanceCount();
            group->removeInstancesInRadius(hitPoint, foliage_brush.radius);
            size_t after = group->getInstanceCount();
            
            if (before != after) {
                hud_captured_mouse = true;
                
                // Sync group to scene
                SceneUI::syncInstancesToScene(ctx, *group, false);
                
                // Rebuild
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                
                g_bvh_rebuild_pending = true;
                g_optix_rebuild_pending = true;
                
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                
                SCENE_LOG_INFO("[Foliage Brush] Removed " + std::to_string(before - after) + " instances");
            }
        }
    }
}

#endif
