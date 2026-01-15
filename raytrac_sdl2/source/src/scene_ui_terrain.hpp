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
                if (SceneUI::DrawSmartFloat("nstr", "Normal Str", &t->normal_strength, 0.1f, 3.0f, "%.2f", false, nullptr, 16)) {
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
                        if (SceneUI::DrawSmartFloat("uvs", "UV Scale", &t->layer_uv_scales[i], 0.1f, 1000.0f, "%.1f", false, nullptr, 12)) {
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
                                        // CRITICAL: Stop rendering before modifying GPU resources to prevent "illegal memory access"
                                        ctx.renderer.stopRendering();
                                        // Wait a tiny bit to ensure kernel finished
                                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                                        cudaDeviceSynchronize(); // Force GPU idle
                                        
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
                                        g_optix_rebuild_pending = true;

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

                                        cudaDeviceSynchronize(); // Force GPU idle

                                        texSlot = nullptr;

                                        // Update Material properties (scalars)
                                        if (ctx.optix_gpu_ptr) {
                                            ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                                        }

                                        // CRITICAL: Request Geometry/SBT Rebuild to update texture handles in SBT
                                        // The main loop will handle this, ensuring synchronization.
                                        g_optix_rebuild_pending = true;

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
                    if (false) { /* LEGACY UI REMOVED
                    {
                        InstanceManager& im = InstanceManager::getInstance();
                        std::string foliageGroupName = "Foliage_" + t->name + "_Layer_" + std::to_string(i);
                        
                        // Find or Create
                        int groupID = im.getGroupIdByName(foliageGroupName);
                        InstanceGroup* foliageGroup = nullptr;
                        
                        if (groupID == -1) {
                             if (ImGui::Button("Initialize Foliage Layer")) {
                                 groupID = im.createGroup(foliageGroupName, "", {}); 
                                 foliageGroup = im.getGroup(groupID);
                                 foliageGroup->brush_settings.splat_map_channel = i; // Lock channel
                                 foliageGroup->brush_settings.use_global_settings = false; // Default to per-source
                                 SCENE_LOG_INFO("[Foliage] Initialized layer: " + foliageGroupName);
                             }
                        } else {
                            foliageGroup = im.getGroup(groupID);
                        }

                        if (foliageGroup) {
                            // Enforce Rules
                            foliageGroup->brush_settings.splat_map_channel = i; 

                            // 1. TOP BAR (Stats & Global Controls)
                            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.6f, 1.0f), "%zu instances", foliageGroup->instances.size());
                            ImGui::SameLine();
                            if (ImGui::SmallButton("Clear All")) {
                                foliageGroup->clearInstances();
                                SceneUI::syncInstancesToScene(ctx, *foliageGroup, true);
                                g_optix_rebuild_pending = true;
                                ctx.renderer.resetCPUAccumulation();
                            }
                            
                            // 2. PLACEMENT RULES (Group Level) - MOVED TO SCATTER SETTINGS
                            // if (ImGui::TreeNodeEx("Placement Rules", ...)) { ... }
                            
                            // 3. SOURCE MESHES LIST
                            ImGui::Separator();
                            ImGui::Text("Source Meshes");
                            ImGui::SameLine();
                            
                            // Add Source Button Logic
                            bool has_selection = ctx.selection.hasSelection();
                            if (ImGui::Button("+ Add Selected")) {
                                if (has_selection) {
                                    std::string node_name = ctx.selection.selected.name;
                                    // Collect triangles
                                    std::vector<std::shared_ptr<Triangle>> selected_tris;
                                    for (auto& obj : ctx.scene.world.objects) {
                                        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                        if (tri && tri->getNodeName() == node_name) selected_tris.push_back(tri);
                                    }
                                    
                                    if (!selected_tris.empty()) {
                                        foliageGroup->sources.emplace_back(node_name, selected_tris);
                                        // Update BVH for source
                                        // foliageGroup->updateSourceBVH(); // Handled by instance generation usually? 
                                        // Actually calculateMeshCenter is needed. ScatterSource constructor does it.
                                        SCENE_LOG_INFO("Added source: " + node_name);
                                    } else {
                                        SCENE_LOG_WARN("Selected object has no triangles or name mismatch.");
                                    }
                                } else {
                                    SCENE_LOG_WARN("Select an object in the viewport first.");
                                }
                            }
                            
                            // Sources Table
                            if (ImGui::BeginTable("FolSrcTable", 4, ImGuiTableFlags_BordersInner | ImGuiTableFlags_RowBg | ImGuiTableFlags_Resizable)) {
                                ImGui::TableSetupColumn("Object", ImGuiTableColumnFlags_WidthStretch);
                                ImGui::TableSetupColumn("Wgt", ImGuiTableColumnFlags_WidthFixed, 40.0f);
                                ImGui::TableSetupColumn("Scale", ImGuiTableColumnFlags_WidthFixed, 60.0f);
                                ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, 20.0f);
                                ImGui::TableHeadersRow();
                                
                                int remove_idx = -1;
                                for (size_t src_i = 0; src_i < foliageGroup->sources.size(); src_i++) {
                                    auto& src = foliageGroup->sources[src_i];
                                    ImGui::PushID((int)src_i);
                                    ImGui::TableNextRow();
                                    
                                    ImGui::TableSetColumnIndex(0);
                                    bool open = ImGui::TreeNodeEx(src.name.c_str(), ImGuiTreeNodeFlags_SpanFullWidth);
                                    
                                    ImGui::TableSetColumnIndex(1);
                                    ImGui::SetNextItemWidth(-1);
                                    ImGui::DragFloat("##w", &src.weight, 0.1f, 0.0f, 100.0f, "%.1f");
                                    
                                    ImGui::TableSetColumnIndex(2);
                                    ImGui::Text("%.1f-%.1f", src.settings.scale_min, src.settings.scale_max);
                                    
                                    ImGui::TableSetColumnIndex(3);
                                    if (ImGui::SmallButton("X")) remove_idx = (int)src_i;
                                    
                                    if (open) {
                                        ImGui::TableNextRow();
                                        ImGui::TableSetColumnIndex(0);
                                        ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(ImGuiCol_FrameBg)); // Darker bg
                                        
                                        // Per-Source Settings
                                        ImGui::Indent();
                                        ImGui::TextDisabled("Transformation Rules:");
                                        ImGui::DragFloatRange2("Scale Rng", &src.settings.scale_min, &src.settings.scale_max, 0.01f, 0.01f, 10.0f);
                                        ImGui::DragFloat("Rot Rand Y", &src.settings.rotation_random_y, 1.0f, 0.0f, 360.0f);
                                        ImGui::DragFloat("Rot Rand XZ", &src.settings.rotation_random_xz, 0.1f, 0.0f, 45.0f);
                                        ImGui::DragFloatRange2("Y Offset", &src.settings.y_offset_min, &src.settings.y_offset_max, 0.01f, -2.0f, 2.0f);
                                        ImGui::Checkbox("Align Normal", &src.settings.align_to_normal);
                                        if (src.settings.align_to_normal) ImGui::SliderFloat("Infl.", &src.settings.normal_influence, 0.0f, 1.0f);
                                        ImGui::Unindent();
                                        
                                        ImGui::TreePop();
                                    }
                                    ImGui::PopID();
                                }
                                ImGui::EndTable();
                                
                                if (remove_idx >= 0) {
                                    foliageGroup->sources.erase(foliageGroup->sources.begin() + remove_idx);
                                }
                            }
                            
                            ImGui::Spacing();
                            
                            // 4. GENERATION ACTION
                            if (ImGui::Button("Generate (Scatter)")) {
                                // Procedural Scatter Implementation
                                if (foliageGroup->sources.empty()) {
                                    SCENE_LOG_ERROR("No source meshes! Add a source first.");
                                } else {
                                    int count = 5000; // Default count or expose param
                                    // Expose count param
                                }
                            }
                            // Quick param for density
                            static int genCount = 1000;
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(100);
                            ImGui::InputInt("Count", &genCount);
                            
                            // Scatter Function
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Number of instances to attempt spawning");
                            
                            if (ImGui::Button("Run Scatter")) {
                                std::mt19937 rng(1234 + i * 99);
                                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                                int spawned = 0;
                                int max_attempts = genCount * 5;
                                
                                for (int attempt = 0; attempt < max_attempts && spawned < genCount; attempt++) {
                                    float u = dist(rng);
                                    float v = dist(rng);
                                    
                                    // Mask Check
                                    bool maskPass = false;
                                    if (t->splatMap && t->splatMap->is_loaded()) {
                                        Vec3 col = t->splatMap->get_color(u, v);
                                        float val = 0.0f;
                                        if (i == 0) val = col.x;
                                        else if (i == 1) val = col.y;
                                        else if (i == 2) val = col.z;
                                        else val = t->splatMap->get_alpha(u, v);
                                        
                                        if (val > 0.2f) maskPass = true; // Hardcoded threshold
                                    } else {
                                        // If no splatmap, maybe allow everywhere? or just error
                                        if (i == 0) maskPass = true; // Base layer
                                    }
                                    
                                    if (!maskPass) continue;
                                    
                                    // Height sample
                                    // float h = t->sampleHeight(u, v); // Need world pos conversion
                                    float tx = (u - 0.5f) * t->heightmap.scale_xz;
                                    float tz = (v - 0.5f) * t->heightmap.scale_xz;
                                    // Grid coords
                                    int gx = u * (t->heightmap.width-1);
                                    int gy = v * (t->heightmap.height-1);
                                    float h = t->heightmap.getHeight(gx, gy); // Fast sample
                                    
                                    // Filter Slope/Height rules?
                                    // Assuming calculateNormal is available
                                    // Vec3 normal = t->calculateNormal(gx, gy);
                                    // float slope = acos(normal.y) * 180.0f/3.14159f;
                                    // if (slope > foliageGroup->brush_settings.slope_max) continue;
                                    // if (h < foliageGroup->brush_settings.height_min || h > foliageGroup->brush_settings.height_max) continue;
                                    
                                    // Position for transform generation
                                    Vec3 surfacePos(tx, h, tz);
                                    
                                    // Generate
                                    InstanceTransform inst = foliageGroup->generateRandomTransform(surfacePos);
                                    
                                    // Add
                                    foliageGroup->addInstance(inst);
                                    spawned++;
                                }
                                
                                // Sync
                                SceneUI::syncInstancesToScene(ctx, *foliageGroup, false);
                                g_optix_rebuild_pending = true;
                                ctx.renderer.resetCPUAccumulation();
                                SCENE_LOG_INFO("Scatter complete: " + std::to_string(spawned));
                            }
                        }
                    }
                    
                    // DISABLE LEGACY UI
                    if (false) {
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
                                
                                // Build unique node names from scene (Cached for performance)
                                static std::vector<std::string> cachedNodeNames;
                                static size_t lastObjectCount = 0;
                                static float cacheTimer = 0.0f;
                                
                                // Auto-invalidate cache if object count changes or periodically
                                bool cacheInvalid = (ctx.scene.world.objects.size() != lastObjectCount);
                                
                                // Force refresh every 2 seconds to catch name changes or other non-size updates
                                cacheTimer += ImGui::GetIO().DeltaTime;
                                if (cacheTimer > 2.0f) {
                                    cacheInvalid = true;
                                    cacheTimer = 0.0f;
                                }

                                if (cacheInvalid || cachedNodeNames.empty()) {
                                    std::set<std::string> uniqueNames;
                                    for (const auto& obj : ctx.scene.world.objects) {
                                        auto tri = std::dynamic_pointer_cast<Triangle>(obj);
                                        if (tri) {
                                            std::string name = tri->getNodeName();
                                            // Skip terrain, scatter instances, and baked geometry
                                            if (!name.empty() && 
                                                name.find("Terrain") == std::string::npos &&
                                                name.find("_inst_") == std::string::npos &&
                                                name.find("_BAKED") == std::string::npos && 
                                                name.find("Foliage_") == std::string::npos) {
                                                uniqueNames.insert(name);
                                            }
                                        }
                                    }
                                    cachedNodeNames.assign(uniqueNames.begin(), uniqueNames.end());
                                    lastObjectCount = ctx.scene.world.objects.size();
                                    // SCENE_LOG_INFO("[UI] Refreshed object list cache");
                                }
                                
                                for (const auto& nodeName : cachedNodeNames) {
                                    bool is_selected = (fLayer.meshPath == nodeName);
                                    if (ImGui::Selectable(nodeName.c_str(), is_selected)) {
                                        // Capture all triangles with this node name
                                        fLayer.meshPath = nodeName;
                                        fLayer.sourceTriangles.clear(); // Clear old sources (important!)
                                        
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
                                    ImGui::TextDisabled("No objects in scene (import a model first)");
                                }
                                
                                ImGui::EndCombo();
                            }
                            
                            // Let's implement basic properties first
                            ImGui::DragInt("Density", &fLayer.density, 10, 0, 10000);
                            ImGui::SliderFloat("Threshold", &fLayer.maskThreshold, 0.0f, 1.0f);
                            
                            ImGui::DragFloat2("Scale Range", &fLayer.scaleRange.x, 0.01f, 0.1f, 5.0f);
                            ImGui::DragFloat2("Rot Range (Y)", &fLayer.rotationRange.x, 1.0f, 0.0f, 360.0f);
                            ImGui::DragFloat2("Y-Offset Range", &fLayer.yOffsetRange.x, 0.05f, -10.0f, 10.0f);
                            
                            ImGui::Spacing();
                            ImGui::Separator();
                            
                            // === BRUSH MODE ===
                            // Note: active_layer_name was removed. This legacy section is largely superseded by "Foliage System" below.
                            // We map it loosely or disable it.
                            bool brushActive = foliage_brush.enabled && foliage_brush.active_group_id != -1; // Looser check for legacy
                            
                            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Paint Mode");
                            if (ImGui::Checkbox("Enable Brush", &brushActive)) {
                                if (brushActive) {
                                    foliage_brush.enabled = true;
                                    foliage_brush.active_group_id = -1; // Cannot map Name -> ID easily here. Brush logic relies on ID now.
                                    // This legacy button will just enable the brush but not bind to a valid group!
                                    // To fix properly, user should use the new UI below.
                                    
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
                                
                                if (SceneUI::DrawSmartFloat("brad", "Radius", &foliage_brush.radius, 1.0f, 50.0f, "%.1f", false, nullptr, 12)) {}
                                ImGui::SliderInt("Instances/Stroke", &foliage_brush.density, 1, 20);
                                
                                const char* modes[] = { "Add", "Remove" };
                                ImGui::Combo("Mode", &foliage_brush.mode, modes, 2);
                                
                                ImGui::Checkbox("Lazy Update (Wait for Release)", &foliage_brush.lazy_update);
                                if (ImGui::IsItemHovered()) ImGui::SetTooltip("If enabled, instances appear only when mouse is released.\nUseful for weak GPUs to prevent lag.");
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
                                    } else {
                                        // Update existing group source if it changed (Fix for Cube issue)
                                        // This ensures we don't get stuck with old sources
                                        if (group->source_node_name != fLayer.meshPath) {
                                            group->source_node_name = fLayer.meshPath;
                                            group->source_triangles = fLayer.sourceTriangles;
                                            
                                            // Also update the detailed sources list
                                            group->sources.clear();
                                            group->sources.push_back(ScatterSource(fLayer.meshPath, fLayer.sourceTriangles));
                                            
                                            SCENE_LOG_INFO("[Foliage] Updated group source to '" + fLayer.meshPath + "'");
                                        }
                                    }
                                    
                                    if (group) {
                                        // Clear existing instances in the group to restart scatter
                                        group->clearInstances();
                                        
                                        // Sync settings
                                        group->brush_settings.scale_min = fLayer.scaleRange.x;
                                        group->brush_settings.scale_max = fLayer.scaleRange.y;
                                        group->brush_settings.rotation_random_y = (fLayer.rotationRange.y - fLayer.rotationRange.x);
                                        group->brush_settings.y_offset_min = fLayer.yOffsetRange.x;
                                        group->brush_settings.y_offset_max = fLayer.yOffsetRange.y;

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
                                            
                                            // Apply Y-Offset
                                            float yOffset = fLayer.yOffsetRange.x + (fLayer.yOffsetRange.y - fLayer.yOffsetRange.x) * dist(rng);
                                            terrainY += yOffset;
                                            
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
                    */ }
                    
                    ImGui::Separator();
                    ImGui::PopID();
                }
            }

            ImGui::Spacing();

            // ===============================================================
            // 4. CENTRALIZED FOLIAGE SYSTEM
            // ===============================================================
            UIWidgets::ColoredHeader("Foliage System", ImVec4(0.3f, 0.9f, 0.4f, 1.0f));
            
            InstanceManager& im = InstanceManager::getInstance();
            static char newFolGroupName[64] = "New Foliage Layer";
            ImGui::InputText("##NewFolName", newFolGroupName, 64);
            ImGui::SameLine();
            if (ImGui::Button("Create Foliage Layer")) {
                std::string gName = std::string(newFolGroupName);
                if (gName.find("Foliage") == std::string::npos) gName = "Foliage_" + gName;
                
                // Fix: Explicit empty vector type
                int newId = im.createGroup(gName, "", std::vector<std::shared_ptr<Triangle>>{});
                InstanceGroup* g = im.getGroup(newId);
                if (g) g->brush_settings.splat_map_channel = -1; // Default to None so it works immediately
                SCENE_LOG_INFO("Created Foliage Layer: " + gName);
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
                // Display Name + Count. Use ### to separate ID from changing label.
                std::string headerLabel = group.name + " (" + std::to_string(group.instances.size()) + ")###Header";
                if (ImGui::TreeNode(headerLabel.c_str())) {                   
                    
                    // SOURCES
                     ImGui::Separator();
                     ImGui::Text("Sources:");
                     if (ImGui::Button("+ Add Selection")) {
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
                     if (ImGui::Button("Pick from List")) {
                         ImGui::OpenPopup("ObjPicker");
                     }
                     
                     if (ImGui::BeginPopup("ObjPicker")) {
                         static char filter[64] = "";
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
                             ImGui::DragFloat("Weight", &src.weight, 0.1f);
                             ImGui::DragFloatRange2("Scale", &src.settings.scale_min, &src.settings.scale_max, 0.01f, 0.001f, 1000.0f);
                             ImGui::DragFloatRange2("Y-Off", &src.settings.y_offset_min, &src.settings.y_offset_max, 0.01f);
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
                     // ACTIONS
                     ImGui::Text("Scatter Settings:");
                     ImGui::DragInt("Total Count", &group.brush_settings.target_count, 100, 1, 10000000);
                     ImGui::InputInt("Seed", &group.brush_settings.seed);
                     ImGui::DragFloat("Min Distance", &group.brush_settings.min_distance, 0.1f, 0.0f, 50.0f);
                     
                     // Helper helper for brush state
                     bool is_active_group = (foliage_brush.active_group_id == group.id);
                     bool is_painting = foliage_brush.enabled && is_active_group && foliage_brush.mode == 0;
                     bool is_erasing = foliage_brush.enabled && is_active_group && foliage_brush.mode == 1;

                     // Paint Button
                     if (is_painting) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
                     if (ImGui::Button("Paint##Fol")) {
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
                     if (ImGui::Button("Erase##Fol")) {
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
                        ImGui::DragFloat("radius##br", &foliage_brush.radius, 0.1f, 0.1f, 100.0f, "%.1f m");
                        if (is_painting) {
                            ImGui::DragInt("density##br", &foliage_brush.density, 1, 1, 20);
                        }
                        ImGui::Checkbox("Lazy Update", &foliage_brush.lazy_update);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Update scene only on mouse release (Better performance for large terrains)");
                     }
                     
                     ImGui::Separator();

                        ImGui::Text("Placement Rules");
                        // Height Range
                        ImGui::DragFloatRange2("Height Range", &group.brush_settings.height_min, &group.brush_settings.height_max, 1.0f, -500.0f, 2000.0f, "Min: %.1f m", "Max: %.1f m");
                        
                        // Slope Limit
                        ImGui::DragFloat("Max Slope", &group.brush_settings.slope_max, 1.0f, 0.0f, 90.0f, "%.1f deg");
                        
                        // Curvature (Flow/Ridge detection)
                        ImGui::TextDisabled("Curvature Filter:");
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(80);
                        ImGui::DragInt("Scale", &group.brush_settings.curvature_step, 0.1f, 1, 20, "%d px");
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Feature Scale (Step Size)\nIncrease to detect larger features like river beds or ignore noise.\n1 = Micro/Pixel details\n5+ = Macro/Terrain features");
                        
                        // Line 1: Ridges
                        ImGui::Checkbox("Ridges", &group.brush_settings.allow_ridges);
                        if (group.brush_settings.allow_ridges) {
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(80);
                            SceneUI::DrawSmartFloat("##RThresh", "", &group.brush_settings.curvature_min, -50.0f, -0.1f, "< %.1f", false, nullptr, 0); // 0 label width as label is empty
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Threshold: Values below this are considered Ridges");
                        }

                        // Line 2: Gullies
                        ImGui::SameLine();
                        ImGui::Checkbox("Gullies", &group.brush_settings.allow_gullies);
                         if (group.brush_settings.allow_gullies) {
                            ImGui::SameLine();
                            ImGui::SetNextItemWidth(80);
                            SceneUI::DrawSmartFloat("##GThresh", "", &group.brush_settings.curvature_max, 0.1f, 50.0f, "> %.1f", false, nullptr, 0);
                            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Threshold: Values above this are considered Gullies/Channels");
                        }

                        // Line 3: Flats (Middle)
                        ImGui::Checkbox("Flats", &group.brush_settings.allow_flats);
                        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Allow areas between Ridge and Gully thresholds");
                        
                        // Splat Map Channel
                        const char* channels[] = { "None", "Red (Grass/Flat)", "Green (Slope)", "Blue (Height/Cliff)", "Alpha (Flow/Mask)" };
                        int current_idx = group.brush_settings.splat_map_channel + 1; // Map -1 to 0, 0 to 1, etc.
                        if (ImGui::Combo("Mask Channel", &current_idx, channels, 5)) {
                            group.brush_settings.splat_map_channel = current_idx - 1;
                        }

                        if (group.brush_settings.splat_map_channel != -1) {
                             if (!t->splatMap || !t->splatMap->is_loaded()) {
                                 ImGui::TextColored(ImVec4(1,0,0,1), "No Splat Map Loaded!");
                             }
                        }
                        
                        // Exclusion Mask UI
                        int ex_idx = group.brush_settings.exclusion_channel + 1;
                        if (ImGui::Combo("Exclude Channel", &ex_idx, channels, 5)) {
                            group.brush_settings.exclusion_channel = ex_idx - 1;
                        }
                        if (group.brush_settings.exclusion_channel != -1) {
                             ImGui::SameLine();
                             ImGui::SetNextItemWidth(80);
                             ImGui::DragFloat("Thresh##Ex", &group.brush_settings.exclusion_threshold, 0.05f, 0.0f, 1.0f);
                             if (ImGui::IsItemHovered()) ImGui::SetTooltip("Values in this channel ABOVE this threshold will be excluded.");
                        }

                        if (ImGui::Button("Scatter Procedurally")) {
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
                                             
                            // 3. World Height Check (vs Min/Max)
                            float worldHeight = h_interp;
                            if (t->transform) worldHeight += t->transform->position.y;
                            
                            if (worldHeight < group.brush_settings.height_min || worldHeight > group.brush_settings.height_max) {
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
                            
                            // Sample neighbors with step
                            float hl = t->heightmap.getHeight(sx - step, sz);
                            float hr = t->heightmap.getHeight(sx + step, sz);
                            float hu = t->heightmap.getHeight(sx, sz - step);
                            float hd = t->heightmap.getHeight(sx, sz + step);
                            
                            float slopeRunX = 2.0f * hmCellSizeX;
                            float slopeRunZ = 2.0f * hmCellSizeZ;
                            
                            float dX_central = fabsf(hr - hl) / slopeRunX;
                            float dZ_central = fabsf(hd - hu) / slopeRunZ;
                            float slopeTan = sqrtf(dX_central*dX_central + dZ_central*dZ_central);
                            float slopeDeg = atan(slopeTan) * 57.2958f;
                            
                            if (slopeDeg > group.brush_settings.slope_max) {
                                continue;
                            }
                            
                            // 5. Curvature Check (Laplacian)
                            // (Left + Right + Up + Down - 4*Center)
                            // Positive = Concave (Bowl/Valley)
                            // Negative = Convex (Hill/Ridge)
                            // NOTE: Neighbors (hl, hr, hu, hd) and center (h_center) MUST be sampled from the same grid indices used for slope.
                            // We used neighbors of (sx, sz) for slope. We need h_center at (sx, sz).
                            float h_center = t->heightmap.getHeight(sx, sz);
                            
                            float laplacian = (hl + hr + hu + hd) - 4.0f * h_center;
                            
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
                     if (ImGui::Button("Clear Instances")) {
                         group.clearInstances();
                         SceneUI::syncInstancesToScene(ctx, group, true);
                         g_optix_rebuild_pending = true;
                         ctx.renderer.resetCPUAccumulation();
                     }
                     ImGui::SameLine();
                     if (ImGui::Button("Delete Layer")) {
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

                    g_optix_rebuild_pending = true;
                    ctx.renderer.resetCPUAccumulation();
                 }
            }

            // Auto Mask
            UIWidgets::ColoredHeader("Procedural Auto-Mask", ImVec4(0.8f, 0.4f, 1.0f, 1.0f));
            static float am_height_min = 5.0f;
            static float am_height_max = 20.0f;
            static float am_slope = 5.0f;

            if (SceneUI::DrawSmartFloat("mhmin", "Height Start", &am_height_min, 0.0f, 500.0f, "%.1f", false, nullptr, 12)) {}
            if (SceneUI::DrawSmartFloat("mhmax", "Height End", &am_height_max, 0.0f, 500.0f, "%.1f", false, nullptr, 12)) {}
            if (SceneUI::DrawSmartFloat("mslope", "Slope Steep", &am_slope, 1.0f, 20.0f, "%.1f", false, nullptr, 12)) {}

            if (ImGui::Button("Generate Mask")) {
                TerrainManager::getInstance().autoMask(t, 0.0f, 0.0f, am_height_min, am_height_max, am_slope);
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
                g_optix_rebuild_pending = true;
                SCENE_LOG_INFO("Auto-mask generated for: " + t->name);
            }
            
            ImGui::SameLine();
            ImGui::SameLine();
            static float bake_flow_threshold = 25.0f;
            if (ImGui::Button("Bake Flow to Alpha")) {
                if (t->splatMap && t->splatMap->is_loaded()) {
                    // Simple Flow Accumulation Logic (Single Pass Downhill)
                    // 1. Compute Flow
                    std::vector<float> flow(t->heightmap.width * t->heightmap.height, 1.0f); // Init with 1 (rain)
                    
                    // Iterate by height (highest first) - approximate by just grid iteration? 
                    // No, for flow we really need sorted height. But for UI tool, maybe a simpler local slope accumulation?
                    // Let's do a simple multi-pass accumulation.
                    
                    int w = t->heightmap.width;
                    int h = t->heightmap.height;
                    
                    // Create indices and sort by height (descending)
                    std::vector<std::pair<float, int>> sorted_idx;
                    sorted_idx.reserve(w * h);
                    for(int z=0; z<h; z++) {
                        for(int x=0; x<w; x++) {
                            sorted_idx.push_back({t->heightmap.getHeight(x, z), z*w + x});
                        }
                    }
                    std::sort(sorted_idx.rbegin(), sorted_idx.rend()); // High to Low
                    
                    // Propagate Flow
                    for(const auto& pair : sorted_idx) {
                        int idx = pair.second;
                        int x = idx % w;
                        int z = idx / w;
                        float currentH = pair.first;
                        float water = flow[idx];
                        
                        // Find lowest neighbor
                        float minH = currentH;
                        int targetIdx = -1;
                        
                        int neighbors[8][2] = {{-1,-1}, {0,-1}, {1,-1}, {-1,0}, {1,0}, {-1,1}, {0,1}, {1,1}};
                        for(auto& n : neighbors) {
                            int nx = x + n[0];
                            int nz = z + n[1];
                            if(nx >=0 && nx < w && nz >=0 && nz < h) {
                                float nh = t->heightmap.getHeight(nx, nz);
                                if(nh < minH) {
                                    minH = nh;
                                    targetIdx = nz * w + nx;
                                }
                            }
                        }
                        
                        if(targetIdx != -1) {
                            flow[targetIdx] += water;
                        }
                    }
                    
                    // Log-scale and Normalize
                    float maxFlow = 0.0f;
                    for(float v : flow) if(v > maxFlow) maxFlow = v;
                    
                    for(int i=0; i<w*h; i++) {
                         float f = flow[i];
                         
                         // Threshold filtering
                         if (f < bake_flow_threshold) {
                             f = 0.0f;
                         }
                         
                         // Log scale for better visibility of channels
                         float val = 0.0f;
                         if (f > 0.0f) {
                             val = log(f) / log(maxFlow > 1 ? maxFlow : 2.0f);
                             val = std::pow(val, 0.5f); // Gamma correct usually helps flow maps
                             if(val > 1.0f) val = 1.0f;
                         }
                         
                         // Write to Alpha (Channel 3)
                         // Texture uses std::vector<CompactVec4> pixels.
                         if (i < t->splatMap->pixels.size()) {
                             t->splatMap->pixels[i].a = static_cast<uint8_t>(std::clamp(val * 255.0f, 0.0f, 255.0f));
                         }
                    }
                    
                    t->splatMap->upload_to_gpu(); // Important!
                    SCENE_LOG_INFO("Flow baked to Splat Map Alpha channel.");
                } else {
                    SCENE_LOG_ERROR("No Splat Map loaded/generated to bake flow into.");
                }
            }
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100);
            ImGui::DragFloat("Flow Thresh", &bake_flow_threshold, 1.0f, 1.0f, 500.0f, "%.0f");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Minimum flow accumulation to be written to Alpha.\nIncrease this to ignore flat areas/rain and keep only rivers.");
            
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
            if (SceneUI::DrawSmartFloat("hard", "Def Hardness", &defaultHardness, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) {}
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
                if(SceneUI::DrawSmartFloat("espeed", "Erode Spd", &hydro_params.erodeSpeed, 0.0f, 1.0f, "%.2f", false, nullptr, 12)) {}
                if(SceneUI::DrawSmartFloat("dspeed", "Dep Speed", &hydro_params.depositSpeed, 0.0f, 1.0f, "%.2f", false, nullptr, 12)) {}
                if(SceneUI::DrawSmartFloat("inert", "Inertia", &hydro_params.inertia, 0.0f, 1.0f, "%.2f", false, nullptr, 12)) {}
                if(SceneUI::DrawSmartFloat("scap", "SedCap", &hydro_params.sedimentCapacity, 0.1f, 10.0f, "%.1f", false, nullptr, 12)) {}
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
                if(SceneUI::DrawSmartFloat("fstr", "Erode Str", &fluvial_strength, 0.1f, 5.0f, "%.1f", false, nullptr, 12)) {}
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
                if(SceneUI::DrawSmartFloat("ttal", "Talus Ang", &thermal_params.talusAngle, 0.1f, 1.0f, "%.2f", false, nullptr, 12)) {}
                if(SceneUI::DrawSmartFloat("tamt", "Amount", &thermal_params.erosionAmount, 0.1f, 2.0f, "%.1f", false, nullptr, 12)) {}

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
                if(SceneUI::DrawSmartFloat("wstr", "Wind Str", &wind_strength, 0.1f, 10.0f, "%.1f", false, nullptr, 12)) {}
                if(SceneUI::DrawSmartFloat("wdir", "Direction", &wind_direction, 0.0f, 360.0f, "%.0f", false, nullptr, 12)) {}
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
                         ctx.optix_gpu_ptr->updateMeshBLASFromTriangles(terrain->name, terrain->mesh_triangles);
                     }
                     g_bvh_rebuild_pending = true;
                 }
            }
        }
    }
}



#endif
