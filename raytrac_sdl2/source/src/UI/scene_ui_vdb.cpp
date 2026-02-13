/**
 * @file scene_ui_vdb.cpp
 * @brief VDB Volume UI Panel - EmberGen-style interface
 * 
 * This file contains the UI for:
 * - VDB Volume import (single file and sequences)
 * - Volume shader properties (density, scattering, emission)
 * - Animation controls for VDB sequences
 */

#include "scene_ui.h"
#include "VDBVolume.h"
#include "VolumeShader.h"
#include "VDBVolumeManager.h"
#include "SceneSelection.h"
#include "globals.h"
#include "ProjectManager.h"
#include "Renderer.h"
#include "OptixWrapper.h"
#include <imgui.h>
#include <filesystem>
#include <algorithm>
#include "scene_ui_gas.hpp"
#include <VolumetricRenderer.h>

// External GPU availability flag
extern bool g_hasOptix;

// ─────────────────────────────────────────────────────────────────────────────
// HELPER: Sync VDB volumes to GPU (OptiX) - with GPU availability check
// ─────────────────────────────────────────────────────────────────────────────
void SceneUI::syncVDBVolumesToGPU(UIContext& ctx) {
    // SAFETY: Only sync to GPU if OptiX is actually available
    if (!g_hasOptix) {
        return; // Silent skip - CPU-only mode
    }
    if (ctx.optix_gpu_ptr) {
        VolumetricRenderer::syncVolumetricData(ctx.scene, ctx.optix_gpu_ptr);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VDB VOLUME IMPORT DIALOG
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::importVDBVolume(UIContext& ctx) {
    std::string path = openFileDialogW(L"OpenVDB Files\0*.vdb;*.nvdb\0All Files\0*.*\0", "", "");
    if (path.empty()) return;
    
    // Create VDB Volume object
    auto vdb = std::make_shared<VDBVolume>();
    
    if (!vdb->loadVDB(path)) {
        addViewportMessage("Failed to load VDB: " + path, 3.0f, ImVec4(1.0f, 0.3f, 0.3f, 1.0f));
        return;
    }
    
    // Auto-scale huge VDBs (likely cm instead of m)
    Vec3 bmin = vdb->getLocalBoundsMin();
    Vec3 bmax = vdb->getLocalBoundsMax();
    float dim_x = bmax.x - bmin.x;
    float dim_y = bmax.y - bmin.y;
    float dim_z = bmax.z - bmin.z;
    float max_dim = (dim_x > dim_y) ? ((dim_x > dim_z) ? dim_x : dim_z) : ((dim_y > dim_z) ? dim_y : dim_z);
    
    if (max_dim > 50.0f) {
        float target_size = 5.0f;
        float scale_factor = target_size / max_dim;
        vdb->setScale(Vec3(scale_factor));
        SCENE_LOG_INFO("[Import] Auto-scaled VDB from " + std::to_string(max_dim) + " to " + std::to_string(target_size) + " units.");
    }
    
    // Auto-Align to Ground (Y=0) and Center (X/Z=0)
    // Shift position so that the bottom-center of the volume sits at world origin (0,0,0)
    Vec3 scale = vdb->getScale();
    Vec3 center = (bmin + bmax) * 0.5f;
    
    float offset_x = -center.x * scale.x;
    float offset_y = -bmin.y * scale.y; // Bottom aligns to 0
    float offset_z = -center.z * scale.z;
    
    vdb->setPosition(Vec3(offset_x, offset_y, offset_z));
    
    SCENE_LOG_INFO("[Import] Auto-aligned VDB. Center X/Z, Ground Y. Pos: " + 
        std::to_string(offset_x) + ", " + std::to_string(offset_y) + ", " + std::to_string(offset_z));

    // Upload to GPU
    if (!vdb->uploadToGPU()) {
        SCENE_LOG_WARN("VDB uploaded to CPU only (GPU upload failed)");
    }
    
    // Create default shader based on available grids
    if (vdb->hasGrid("temperature")) {
        vdb->setShader(VolumeShader::createFirePreset());
    } else {
        vdb->setShader(VolumeShader::createSmokePreset());
    }
    
    // Add to scene VDB list
    ctx.scene.addVDBVolume(vdb);
    
    // Also add to world.objects as Hittable for CPU ray intersection
    ctx.scene.world.objects.push_back(vdb);
    
    // Trigger BVH rebuild for the new hittable
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    
    // GPU sync only if OptiX is available
    if (g_hasOptix && ctx.optix_gpu_ptr) {
        // Upload VDB volumes to GPU for OptiX ray marching
        syncVDBVolumesToGPU(ctx);
        ctx.optix_gpu_ptr->resetAccumulation();
    }
    
    g_ProjectManager.markModified();
    
    // Auto-select the new VDB
    ctx.selection.selectVDBVolume(vdb, -1, vdb->name);
    
    // Auto-switch to VDB tab removed per user request
    
    addViewportMessage("Imported VDB: " + vdb->name, 3.0f, ImVec4(0.5f, 1.0f, 0.5f, 1.0f));
    
    SCENE_LOG_INFO("Imported VDB Volume: " + path);
}

// Import VDB Sequence
void SceneUI::importVDBSequence(UIContext& ctx) {
#ifdef _WIN32
    std::string path = openFileDialogW(L"VDB Files\0*.vdb\0All Files\0*.*\0");
    if (path.empty()) return;
    
    // Create new VDB volume
    auto vdb = std::make_shared<VDBVolume>();
    
    // Load sequence (detects pattern from single file)
    if (!vdb->loadVDBSequence(path)) {
        SCENE_LOG_ERROR("Failed to load VDB sequence: " + path);
        addViewportMessage("Failed to load VDB sequence!", 3.0f, ImVec4(1, 0.2f, 0.2f, 1));
        return;
    }
    
    // Auto-scale huge VDBs (detect unit system: mm, cm, m)
    Vec3 bmin = vdb->getLocalBoundsMin();
    Vec3 bmax = vdb->getLocalBoundsMax();
    
    // Validate bbox values - check for overflow/garbage (like 2^32)
    const float MAX_VALID_BBOX = 1e9f;  // 1 billion is reasonable max
    bool bbox_valid = true;
    
    if (std::abs(bmin.x) > MAX_VALID_BBOX || std::abs(bmin.y) > MAX_VALID_BBOX || std::abs(bmin.z) > MAX_VALID_BBOX ||
        std::abs(bmax.x) > MAX_VALID_BBOX || std::abs(bmax.y) > MAX_VALID_BBOX || std::abs(bmax.z) > MAX_VALID_BBOX ||
        std::isnan(bmin.x) || std::isnan(bmax.x) || std::isinf(bmin.x) || std::isinf(bmax.x)) {
        SCENE_LOG_WARN("[Import] Invalid bbox detected! min=(" + 
                       std::to_string(bmin.x) + "," + std::to_string(bmin.y) + "," + std::to_string(bmin.z) + 
                       ") max=(" + std::to_string(bmax.x) + "," + std::to_string(bmax.y) + "," + std::to_string(bmax.z) + ")");
        bbox_valid = false;
    }
    
    float scale_factor = 1.0f;
    
    if (!bbox_valid) {
        // Fallback for invalid/empty VDBs
        scale_factor = 5.0f; // Reasonable default size
        SCENE_LOG_WARN("[Import] Using fallback scale 5.0 due to invalid bbox. Resetting bounds.");
        
        // CRITICAL: Overwrite the invalid bounds with safe defaults to prevent math errors
        // This fixes the "devasa volume" issue caused by subsequent math with 2^32
        vdb->setLocalBounds(Vec3(-0.5), Vec3(0.5));
        bmin = Vec3(-0.5);
        bmax = Vec3(0.5);
    }
    else {
        // Valid bbox - detect unit system
        float dim_x = bmax.x - bmin.x;
        float dim_y = bmax.y - bmin.y;
        float dim_z = bmax.z - bmin.z;
        float max_dim = std::max({dim_x, dim_y, dim_z});
        
        // Detect unit system and apply conversion:
        // - If max_dim > 1000: likely millimeters, convert to meters (÷1000)
        // - If max_dim > 50:   likely centimeters, convert to meters (÷100) then scale to reasonable size
        // - Otherwise:         likely meters, keep as is
        
        if (max_dim > 1000.0f) {
            // Likely millimeters - convert to meters and scale to ~5m target
            float meter_size = max_dim / 1000.0f;
            scale_factor = 5.0f / meter_size;
            SCENE_LOG_INFO("[Import] Detected mm units (max=" + std::to_string(max_dim) + 
                        "), scale factor: " + std::to_string(scale_factor));
        }
        else if (max_dim > 50.0f) {
            // Likely centimeters - convert to meters and scale to ~5m target
            float meter_size = max_dim / 100.0f;
            scale_factor = 5.0f / meter_size;
            SCENE_LOG_INFO("[Import] Detected cm units (max=" + std::to_string(max_dim) + 
                        "), scale factor: " + std::to_string(scale_factor));
        }
        else if (max_dim > 0.01f) {
            // Likely meters already - keep as is or scale if too small
            scale_factor = 1.0f;
            SCENE_LOG_INFO("[Import] Detected m units (max=" + std::to_string(max_dim) + "), no scaling");
        }
        else {
            // Extremely small - might be normalized, scale up
            scale_factor = 5.0f / std::max(max_dim, 0.001f);
            SCENE_LOG_INFO("[Import] Detected tiny VDB (max=" + std::to_string(max_dim) + 
                        "), scale factor: " + std::to_string(scale_factor));
        }
    }
    
    // Clamp scale to reasonable range (never 0, never too extreme)
    scale_factor = std::max(0.0001f, std::min(scale_factor, 1000.0f));
    
    vdb->setScale(Vec3(scale_factor));
    
    // Center pivot to bottom center of bbox (proper pivot for effects)
    // This offsets local_bbox so origin is at (centerX, bottomY, centerZ)
    vdb->centerPivotToBottomCenter();
    
    // Position at world origin (bbox bottom center is now at 0,0,0)
    vdb->setPosition(Vec3(0, 0, 0));
    
    // Upload to GPU (Start Frame)
    if (!vdb->uploadToGPU()) {
        SCENE_LOG_WARN("VDB uploaded to CPU only (GPU upload failed)");
    }
    
    // Default Shader
    if (vdb->hasGrid("temperature")) {
        vdb->setShader(VolumeShader::createFirePreset());
    } else {
        vdb->setShader(VolumeShader::createSmokePreset());
    }
    
    // Add to scene
    ctx.scene.addVDBVolume(vdb);
    ctx.scene.world.objects.push_back(vdb);
    
    // Trigger BVH build
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    
    if (ctx.optix_gpu_ptr) {
        syncVDBVolumesToGPU(ctx);
        ctx.optix_gpu_ptr->resetAccumulation();
    }
    
    g_ProjectManager.markModified();
    
    ctx.selection.selectVDBVolume(vdb, -1, vdb->name);
    
    // Focus tab removed per user request
    show_volumetric_tab = true;
    
    addViewportMessage("Imported Sequence: " + vdb->name + " (" + std::to_string(vdb->getFrameCount()) + " frames)", 
                       4.0f, ImVec4(0.5f, 1.0f, 0.5f, 1.0f));
    
    SCENE_LOG_INFO("Imported VDB Sequence: " + vdb->name);
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// VOLUMETRIC PROPERTIES PANEL (UNIFIED VDB & GAS)
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::drawVolumetricPanel(UIContext& ctx) {
    auto& scene = ctx.scene;
    auto& selection = ctx.selection;
    
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
    
    // -------------------------------------------------------------
    // TOP SECTION: CREATION & IMPORT
    // -------------------------------------------------------------
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Add / Import Volumetrics");
    if (ImGui::Button("+ New Gas Simulation", ImVec2(160, 0))) {
        auto gas = std::make_shared<GasVolume>("Gas Sim " + std::to_string(scene.gas_volumes.size() + 1));
        gas->initialize();
        
        // Add default emitter
        FluidSim::Emitter emitter;
        emitter.position = Vec3(2.5f, 0.5f, 2.5f);
        emitter.radius = 0.4f;
        emitter.density_rate = 15.0f;
        gas->addEmitter(emitter);
        
        scene.addGasVolume(gas);
        selection.selectGasVolume(gas, -1, gas->name);
    }
    ImGui::SameLine();
    if (ImGui::Button("Import VDB File", ImVec2(120, 0))) {
        importVDBVolume(ctx);
    }
    ImGui::SameLine();
    if (ImGui::Button("Import VDB Seq", ImVec2(120, 0))) {
        importVDBSequence(ctx);
    }
    
    ImGui::Separator();
    
    // -------------------------------------------------------------
    // MIDDLE SECTION: COMBINED OBJECT LIST
    // -------------------------------------------------------------
    size_t total_vols = scene.vdb_volumes.size() + scene.gas_volumes.size();
    ImGui::Text("Scene Volumes (%zu)", total_vols);
    
    if (ImGui::BeginListBox("##volume_list", ImVec2(-FLT_MIN, 150))) {
        // List Gas Simulations
        for (size_t i = 0; i < scene.gas_volumes.size(); ++i) {
            auto& gas = scene.gas_volumes[i];
            bool is_selected = (selection.selected.type == SelectableType::GasVolume && selection.selected.gas_volume == gas);
            
            ImGui::PushID(("gas_" + std::to_string(i)).c_str());
            if (ImGui::Selectable(("[Sim] " + gas->name).c_str(), is_selected)) {
                selection.selectGasVolume(gas, static_cast<int>(i), gas->name);
            }
            
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Delete")) {
                    scene.removeGasVolume(gas);
                    selection.clearSelection();
                    ctx.renderer.updateOptiXGasVolumes(scene, ctx.optix_gpu_ptr);
                    SceneUI::syncVDBVolumesToGPU(ctx);
                }
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
        
        // List VDB Volumes
        for (size_t i = 0; i < scene.vdb_volumes.size(); ++i) {
            auto& vdb = scene.vdb_volumes[i];
            bool is_selected = (selection.selected.type == SelectableType::VDBVolume && selection.selected.vdb_volume == vdb);
            
            ImGui::PushID(("vdb_" + std::to_string(i)).c_str());
            std::string label = "[VDB] " + vdb->name;
            if (vdb->isAnimated()) label += " (Seq)";
            
            if (ImGui::Selectable(label.c_str(), is_selected)) {
                selection.selectVDBVolume(vdb, static_cast<int>(i), vdb->name);
            }
            
            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Delete")) {
                    // Logic from existing drawVDBVolumePanel
                    auto it = std::find(scene.vdb_volumes.begin(), scene.vdb_volumes.end(), vdb);
                    if (it != scene.vdb_volumes.end()) {
                        vdb->unload();
                        scene.vdb_volumes.erase(it);
                        auto it_obj = std::find(scene.world.objects.begin(), scene.world.objects.end(), vdb);
                        if (it_obj != scene.world.objects.end()) scene.world.objects.erase(it_obj);
                        selection.clearSelection();
                        syncVDBVolumesToGPU(ctx);
                    }
                }
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
        
        ImGui::EndListBox();
    }
    
    ImGui::Separator();
    
    // -------------------------------------------------------------
    // BOTTOM SECTION: CONTEXTUAL PROPERTIES
    // -------------------------------------------------------------
    if (selection.selected.type == SelectableType::GasVolume && selection.selected.gas_volume) {
        GasUI::drawGasSimulationProperties(ctx, selection.selected.gas_volume);
    }
    else if (selection.selected.type == SelectableType::VDBVolume && selection.selected.vdb_volume) {
        drawVDBVolumeProperties(ctx, selection.selected.vdb_volume.get());
    }
    else {
        ImGui::TextDisabled("Select a Volume object to edit properties.");
    }
    
    ImGui::PopStyleVar();
}

void SceneUI::drawVDBVolumeProperties(UIContext& ctx, VDBVolume* vdb) {
    if (!vdb) return;
    
    // UNIQUE ID SCOPE: Use VDB volume ID to prevent ImGui ID conflicts
    ImGui::PushID(vdb->getVDBVolumeID());
    
    bool changed = false;
    
    // Track initial state to detect changes for Undo/Redo (if implemented) or Project Modified flag
    bool was_modified = false;
    
    // ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.2f, 0.5f, 0.8f)); // Removed to use Main Theme
    
    // ═══════════════════════════════════════════════════════════════════════════
    // FILE INFO
    // ═══════════════════════════════════════════════════════════════════════════
    if (UIWidgets::BeginSection("File Information", ImVec4(0.3f, 0.5f, 0.7f, 1.0f))) {
        ImGui::TextColored(ImVec4(0.8f, 0.6f, 1.0f, 1.0f), "Volume: %s", vdb->name.c_str());
        
        auto* vol_data = VDBVolumeManager::getInstance().getVolume(vdb->getVDBVolumeID());
        if (vol_data) {
            ImGui::TextDisabled("File: %s", std::filesystem::path(vol_data->filepath).filename().string().c_str());
            ImGui::TextDisabled("Size: %.2f MB", vol_data->gpu_buffer_size / (1024.0f * 1024.0f));
            
            // Bounds info
            Vec3 bmin = vdb->getLocalBoundsMin();
            Vec3 bmax = vdb->getLocalBoundsMax();
            ImGui::TextDisabled("Bounds: (%.1f, %.1f, %.1f) - (%.1f, %.1f, %.1f)",
                bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z);
        }
        
        // Available grids
        auto grids = vdb->getAvailableGrids();
        if (!grids.empty()) {
            ImGui::TextDisabled("Grids: ");
            ImGui::SameLine();
            for (size_t i = 0; i < grids.size(); ++i) {
                if (i > 0) ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "%s", grids[i].c_str());
            }
        }
        UIWidgets::EndSection();
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ANIMATION
    // ═══════════════════════════════════════════════════════════════════════════
    // ═══════════════════════════════════════════════════════════════════════════
    // ANIMATION
    // ═══════════════════════════════════════════════════════════════════════════
    if (vdb->isAnimated() && UIWidgets::BeginSection("Animation", ImVec4(0.4f, 0.6f, 1.0f, 1.0f))) {
        ImGui::PushID("vdb_anim");  // Unique ID scope for animation widgets
        
        ImGui::Text("Sequence: %d Frames", vdb->getFrameCount());
        ImGui::Text("Current File: %s", std::filesystem::path(vdb->getFilePath()).filename().string().c_str());
        
        bool linked = vdb->isLinkedToTimeline();
        if (ImGui::Checkbox("Link to Timeline", &linked)) {
            vdb->setLinkedToTimeline(linked);
        }
        
        int offset = vdb->getFrameOffset();
        if (ImGui::DragInt("Frame Offset", &offset)) {
            vdb->setFrameOffset(offset);
            // Force update to preview
            vdb->updateFromTimeline(timeline.getCurrentFrame());
            changed = true;
        }
        
        ImGui::PopID();
        UIWidgets::EndSection();
    }
    
    ImGui::Separator();
    
    // ═══════════════════════════════════════════════════════════════════════════
    // TRANSFORM
    // ═══════════════════════════════════════════════════════════════════════════
    // ═══════════════════════════════════════════════════════════════════════════
    // TRANSFORM
    // ═══════════════════════════════════════════════════════════════════════════
    if (UIWidgets::BeginSection("Transform", ImVec4(0.5f, 0.9f, 0.5f, 1.0f))) {
        Vec3 pos = vdb->getPosition();
        Vec3 rot = vdb->getRotation();
        Vec3 scale = vdb->getScale();
        
        if (ImGui::DragFloat3("Position", &pos.x, 0.05f, 0, 0, "%.3f")) {
            vdb->setPosition(pos);
            changed = true;
        }
        
        if (ImGui::DragFloat3("Rotation", &rot.x, 1.0f, -360.0f, 360.0f, "%.1f")) {
            vdb->setRotation(rot);
            changed = true;
        }
        
        // Use higher precision for scale, as mm-to-m conversion uses tiny numbers (0.001)
        if (ImGui::DragFloat3("Scale", &scale.x, 0.0001f, 1e-7f, 1000.0f, "%.6f")) {
            vdb->setScale(scale);
            changed = true;
        }

        // Manual Rotation & Scale Overrides
        ImGui::SetNextItemWidth(200);
        if (ImGui::Button("Rotate -90 X (Blender Fix)")) {
            vdb->setRotation(Vec3(-90.0f, 0, 0));
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Scale 0.001 (mm Fix)")) {
            vdb->setScale(Vec3(0.001f));
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset All")) {
            vdb->setRotation(Vec3(0));
            vdb->setScale(Vec3(1.0f));
            changed = true;
        }

        UIWidgets::EndSection();
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // VOLUME SHADER PROPERTIES (Unified Path)
    // ═══════════════════════════════════════════════════════════════════════════
    auto shader = vdb->getOrCreateShader();
    if (SceneUI::drawVolumeShaderUI(ctx, shader, vdb, nullptr)) {
        changed = true;
    }
    
    // Animation section already exists above (lines 429-449)
    
    // ImGui::PopStyleColor();
    
    // Apply changes
    if (changed) {
        ctx.renderer.resetCPUAccumulation();
        if (ctx.optix_gpu_ptr) {
            syncVDBVolumesToGPU(ctx);  // Sync shader changes to GPU
            ctx.optix_gpu_ptr->resetAccumulation();
        }
        g_ProjectManager.markModified();
    }
    
    ImGui::PopID();  // Match PushID at function start
}

// ═══════════════════════════════════════════════════════════════════════════════
// MENU INTEGRATION
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::drawVDBImportMenu(UIContext& ctx) {
    if (ImGui::MenuItem("VDB Volume (.vdb)")) {
        importVDBVolume(ctx);
    }
    
    if (ImGui::MenuItem("VDB Sequence (folder)")) {
        // TODO: Implement folder selection and sequence import
        addViewportMessage("VDB Sequence import coming soon!", 3.0f, ImVec4(1.0f, 0.8f, 0.4f, 1.0f));
    }
}
