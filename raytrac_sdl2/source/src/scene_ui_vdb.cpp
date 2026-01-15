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

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER: Sync VDB volumes to GPU (OptiX)
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::syncVDBVolumesToGPU(UIContext& ctx) {
    if (!ctx.optix_gpu_ptr) return;
    
    std::vector<GpuVDBVolume> gpu_volumes;
    
    for (auto& vdb : ctx.scene.vdb_volumes) {
        if (!vdb || !vdb->isLoaded()) continue;
        
        GpuVDBVolume gv = {};
        
        // Grid pointers (NanoVDB on GPU)
        gv.density_grid = VDBVolumeManager::getInstance().getGPUGrid(vdb->getVDBVolumeID());
        gv.temperature_grid = VDBVolumeManager::getInstance().getGPUTemperatureGrid(vdb->getVDBVolumeID());
        
        if (!gv.density_grid) continue;
        
        // Transform (3x4 row-major)
        Matrix4x4 m = vdb->getTransform();
        Matrix4x4 inv = vdb->getInverseTransform();
        
        // Copy to float[12] (row-major 3x4)
        gv.transform[0] = m.m[0][0]; gv.transform[1] = m.m[0][1]; gv.transform[2] = m.m[0][2]; gv.transform[3] = m.m[0][3];
        gv.transform[4] = m.m[1][0]; gv.transform[5] = m.m[1][1]; gv.transform[6] = m.m[1][2]; gv.transform[7] = m.m[1][3];
        gv.transform[8] = m.m[2][0]; gv.transform[9] = m.m[2][1]; gv.transform[10] = m.m[2][2]; gv.transform[11] = m.m[2][3];
        
        gv.inv_transform[0] = inv.m[0][0]; gv.inv_transform[1] = inv.m[0][1]; gv.inv_transform[2] = inv.m[0][2]; gv.inv_transform[3] = inv.m[0][3];
        gv.inv_transform[4] = inv.m[1][0]; gv.inv_transform[5] = inv.m[1][1]; gv.inv_transform[6] = inv.m[1][2]; gv.inv_transform[7] = inv.m[1][3];
        gv.inv_transform[8] = inv.m[2][0]; gv.inv_transform[9] = inv.m[2][1]; gv.inv_transform[10] = inv.m[2][2]; gv.inv_transform[11] = inv.m[2][3];
        
        // Bounds (World Space)
        AABB world_bounds = vdb->getWorldBounds();
        gv.world_bbox_min = make_float3(world_bounds.min.x, world_bounds.min.y, world_bounds.min.z);
        gv.world_bbox_max = make_float3(world_bounds. max.x, world_bounds.max.y, world_bounds.max.z);
        
        // Local Bounds
        Vec3 lmin = vdb->getLocalBoundsMin();
        Vec3 lmax = vdb->getLocalBoundsMax();
        gv.local_bbox_min = make_float3(lmin.x, lmin.y, lmin.z);
        gv.local_bbox_max = make_float3(lmax.x, lmax.y, lmax.z);
        
        // Shader parameters
        auto shader = vdb->volume_shader;
        if (shader) {
            // Density
            gv.density_multiplier = shader->density.multiplier * vdb->density_scale;
            gv.density_remap_low = shader->density.remap_low;
            gv.density_remap_high = shader->density.remap_high;
            
            // Scattering
            gv.scatter_color = make_float3(shader->scattering.color.x, shader->scattering.color.y, shader->scattering.color.z);
            gv.scatter_coefficient = shader->scattering.coefficient;
            gv.scatter_anisotropy = shader->scattering.anisotropy;
            gv.scatter_anisotropy_back = shader->scattering.anisotropy_back;
            gv.scatter_lobe_mix = shader->scattering.lobe_mix;
            gv.scatter_multi = shader->scattering.multi_scatter;
            
            // Absorption
            gv.absorption_color = make_float3(shader->absorption.color.x, shader->absorption.color.y, shader->absorption.color.z);
            gv.absorption_coefficient = shader->absorption.coefficient;
            
            // Emission
            gv.emission_mode = static_cast<int>(shader->emission.mode);
            gv.emission_color = make_float3(shader->emission.color.x, shader->emission.color.y, shader->emission.color.z);
            gv.emission_intensity = shader->emission.intensity;
            gv.temperature_scale = shader->emission.temperature_scale;
            gv.blackbody_intensity = shader->emission.blackbody_intensity;

            // Temperature fallback: If no temperature grid, use density for blackbody
            // This allows blackbody emission to work even without explicit temperature data
            if (!gv.temperature_grid && gv.density_grid && 
                shader->emission.mode == VolumeEmissionMode::Blackbody) {
                gv.temperature_grid = gv.density_grid;
            }
            
            // Allow using Density as Temperature source (if selected in UI)
            if (shader->emission.temperature_channel == "density" || shader->emission.temperature_channel == "Density") {
                gv.temperature_grid = gv.density_grid;
            }

            // Color Ramp (Gradient)
            gv.color_ramp_enabled = shader->emission.color_ramp.enabled ? 1 : 0;
            if (gv.color_ramp_enabled) {
                const auto& stops = shader->emission.color_ramp.stops;
                gv.ramp_stop_count = std::min((int)stops.size(), 8);
                
                // Copy stops to fixed-size GPU arrays
                for (int i = 0; i < gv.ramp_stop_count; ++i) {
                    gv.ramp_positions[i] = stops[i].position;
                    gv.ramp_colors[i] = make_float3(stops[i].color.x, stops[i].color.y, stops[i].color.z);
                }
            } else {
                gv.ramp_stop_count = 0;
            }
            
            // Quality
            gv.step_size = shader->quality.step_size;
            gv.max_steps = shader->quality.max_steps;
            gv.shadow_steps = shader->quality.shadow_steps;
            gv.shadow_strength = shader->quality.shadow_strength;
        } else {
            // Default values
            gv.density_multiplier = vdb->density_scale;
            gv.density_remap_low = 0.0f;
            gv.density_remap_high = 1.0f;
            gv.scatter_color = make_float3(1.0f, 1.0f, 1.0f);
            gv.scatter_coefficient = 1.0f;
            gv.scatter_anisotropy = 0.0f;
            gv.absorption_coefficient = 0.1f;
            gv.step_size = 0.1f;
            gv.max_steps = 128;
        }
        
        gpu_volumes.push_back(gv);
    }
    
    ctx.optix_gpu_ptr->updateVDBVolumeBuffer(gpu_volumes);
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
    
    if (ctx.optix_gpu_ptr) {
        // Upload VDB volumes to GPU for OptiX ray marching
        syncVDBVolumesToGPU(ctx);
        ctx.optix_gpu_ptr->resetAccumulation();
    }
    
    g_ProjectManager.markModified();
    
    // Auto-select the new VDB
    ctx.selection.selectVDBVolume(vdb, -1, vdb->name);
    
    // Auto-switch to VDB tab
    tab_to_focus = "VDB";
    show_vdb_tab = true;
    
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
    
    // Focus tab
    tab_to_focus = "VDB";
    show_vdb_tab = true;
    
    addViewportMessage("Imported Sequence: " + vdb->name + " (" + std::to_string(vdb->getFrameCount()) + " frames)", 
                       4.0f, ImVec4(0.5f, 1.0f, 0.5f, 1.0f));
    
    SCENE_LOG_INFO("Imported VDB Sequence: " + vdb->name);
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// VDB VOLUME PROPERTIES PANEL
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::drawVDBVolumePanel(UIContext& ctx) {
    if (ctx.scene.vdb_volumes.empty()) {
        ImGui::TextDisabled("No VDB volumes in scene.");
        ImGui::TextDisabled("Use File > Import > VDB Volume to add one.");
        return;
    }
    
    // VDB Volume list
    ImGui::TextColored(ImVec4(0.7f, 0.5f, 1.0f, 1.0f), "VDB Volumes (%zu)", 
                       ctx.scene.vdb_volumes.size());
    ImGui::Separator();
    
    static int selected_vdb_index = 0;
    
    // List of VDB volumes
    if (ImGui::BeginListBox("##vdb_list", ImVec2(-FLT_MIN, 100))) {
        for (size_t i = 0; i < ctx.scene.vdb_volumes.size(); ++i) {
            auto& vol = ctx.scene.vdb_volumes[i];
            if (!vol) continue;
            
            bool is_selected = (selected_vdb_index == static_cast<int>(i));
            
            std::string label = vol->name;
            if (vol->isAnimated()) {
                label += " [Seq: " + std::to_string(vol->getFrameCount()) + " frames]";
            }
            
            // Add unique ImGui ID suffix to prevent duplicate name conflicts
            std::string selectable_id = label + "##vdb_" + std::to_string(i);
            if (ImGui::Selectable(selectable_id.c_str(), is_selected)) {
                selected_vdb_index = static_cast<int>(i);
            }
        }
        ImGui::EndListBox();
    }
    
    // Buttons
    if (ImGui::Button("Import VDB")) {
        importVDBVolume(ctx);
    }
    ImGui::SameLine();
    if (ImGui::Button("Import Sequence")) {
        importVDBSequence(ctx);
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove Selected") && selected_vdb_index < (int)ctx.scene.vdb_volumes.size()) {
        auto& vol = ctx.scene.vdb_volumes[selected_vdb_index];
        if (vol) {
            // Safety: Ensure GPU isn't using the volume before we free it
            if (ctx.optix_gpu_ptr) cudaDeviceSynchronize();
            vol->unload();
        }
        ctx.scene.vdb_volumes.erase(ctx.scene.vdb_volumes.begin() + selected_vdb_index);
        if (selected_vdb_index > 0) selected_vdb_index--;
        
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        ctx.renderer.resetCPUAccumulation();
        
        if (ctx.optix_gpu_ptr) {
            syncVDBVolumesToGPU(ctx); // Update GPU buffer with new list
            ctx.optix_gpu_ptr->resetAccumulation();
        }
        
        g_ProjectManager.markModified();
    }
    
    ImGui::Separator();
    
    // Selected VDB properties
    if (selected_vdb_index >= 0 && selected_vdb_index < (int)ctx.scene.vdb_volumes.size()) {
        auto& vdb = ctx.scene.vdb_volumes[selected_vdb_index];
        if (vdb) {
            drawVDBVolumeProperties(ctx, vdb.get());
        }
    }
}

void SceneUI::drawVDBVolumeProperties(UIContext& ctx, VDBVolume* vdb) {
    if (!vdb) return;
    
    // UNIQUE ID SCOPE: Use VDB volume ID to prevent ImGui ID conflicts
    ImGui::PushID(vdb->getVDBVolumeID());
    
    bool changed = false;
    
    // Track initial state to detect changes for Undo/Redo (if implemented) or Project Modified flag
    bool was_modified = false;
    
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.3f, 0.2f, 0.5f, 0.8f));
    
    // ═══════════════════════════════════════════════════════════════════════════
    // FILE INFO
    // ═══════════════════════════════════════════════════════════════════════════
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
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ANIMATION
    // ═══════════════════════════════════════════════════════════════════════════
    if (vdb->isAnimated() && ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
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
    }
    
    ImGui::Separator();
    
    // ═══════════════════════════════════════════════════════════════════════════
    // TRANSFORM
    // ═══════════════════════════════════════════════════════════════════════════
    if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
        Vec3 pos = vdb->getPosition();
        Vec3 rot = vdb->getRotation();
        Vec3 scale = vdb->getScale();
        
        if (ImGui::DragFloat3("Position", &pos.x, 0.1f)) {
            vdb->setPosition(pos);
            changed = true;
        }
        
        if (ImGui::DragFloat3("Rotation", &rot.x, 1.0f, -360.0f, 360.0f)) {
            vdb->setRotation(rot);
            changed = true;
        }
        
        if (ImGui::DragFloat3("Scale", &scale.x, 0.01f, 0.01f, 100.0f)) {
            vdb->setScale(scale);
            changed = true;
        }
        
        // Fit to VDB bounds button
        if (ImGui::Button("Reset Transform")) {
            vdb->setPosition(Vec3(0));
            vdb->setRotation(Vec3(0));
            vdb->setScale(Vec3(1));
            changed = true;
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // VOLUME SHADER PROPERTIES
    // ═══════════════════════════════════════════════════════════════════════════
    auto shader = vdb->getOrCreateShader();
    
    // ─────────────────────────────────────────────────────────────────────────
    // DENSITY
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Density", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("density");
        
        // Channel selection
        auto grids_list = vdb->getAvailableGrids();
        if (!grids_list.empty() && ImGui::BeginCombo("Channel", shader->density.channel.c_str())) {
            for (const auto& grid_name : grids_list) {
                if (ImGui::Selectable(grid_name.c_str(), shader->density.channel == grid_name)) {
                    shader->density.channel = grid_name;
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
        
        if (ImGui::SliderFloat("Multiplier", &shader->density.multiplier, 0.0f, 100.0f)) {
            changed = true;
        }
        
        if (ImGui::DragFloatRange2("Remap", &shader->density.remap_low, 
                                    &shader->density.remap_high, 0.01f, 0.0f, 1.0f)) {
            changed = true;
        }
        
        // Edge handling (fixes dark bounding box)
        ImGui::Separator();
        ImGui::TextDisabled("Edge Handling (fixes dark box edges):");
        
        // CUTOFF REMOVED: User requested - was zeroing low densities
        
        if (ImGui::SliderFloat("Edge Falloff", &shader->density.edge_falloff, 0.0f, 2.0f, "%.2f")) {
            changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Smooth density fade near bounding box edges.\n0 = disabled");
        }
        
        ImGui::PopID();
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // SCATTERING
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Scattering")) {
        ImGui::PushID("scatter");
        
        float color[3] = {
            static_cast<float>(shader->scattering.color.x),
            static_cast<float>(shader->scattering.color.y),
            static_cast<float>(shader->scattering.color.z)
        };
        if (ImGui::ColorEdit3("Color", color)) {
            shader->scattering.color = Vec3(color[0], color[1], color[2]);
            changed = true;
        }
        
        if (ImGui::SliderFloat("Strength", &shader->scattering.coefficient, 0.0f, 10.0f)) {
            changed = true;
        }
        
        if (ImGui::SliderFloat("Anisotropy (G)", &shader->scattering.anisotropy, -0.99f, 0.99f)) {
            changed = true;
        }
        
        if (ImGui::TreeNode("Advanced Scattering")) {
            if (ImGui::SliderFloat("Back Scatter G", &shader->scattering.anisotropy_back, -0.99f, 0.0f)) {
                changed = true;
            }
            if (ImGui::SliderFloat("Lobe Mix", &shader->scattering.lobe_mix, 0.0f, 1.0f)) {
                changed = true;
            }
            if (ImGui::SliderFloat("Multi-Scatter", &shader->scattering.multi_scatter, 0.0f, 1.0f)) {
                changed = true;
            }
            ImGui::TreePop();
        }
        
        ImGui::PopID();
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // ABSORPTION
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Absorption")) {
        ImGui::PushID("absorb");
        
        float color[3] = {
            static_cast<float>(shader->absorption.color.x),
            static_cast<float>(shader->absorption.color.y),
            static_cast<float>(shader->absorption.color.z)
        };
        if (ImGui::ColorEdit3("Color", color)) {
            shader->absorption.color = Vec3(color[0], color[1], color[2]);
            changed = true;
        }
        
        if (ImGui::SliderFloat("Strength", &shader->absorption.coefficient, 0.0f, 5.0f)) {
            changed = true;
        }
        
        ImGui::PopID();
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // EMISSION (Fire/Explosions)
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Emission / Fire", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushID("emission");
        
        const char* modes[] = { "None", "Constant Color", "Blackbody (Temperature)", "Channel Driven" };
        int mode = static_cast<int>(shader->emission.mode);
        if (ImGui::Combo("Mode", &mode, modes, 4)) {
            shader->emission.mode = static_cast<VolumeEmissionMode>(mode);
            changed = true;
        }
        
        if (shader->emission.mode == VolumeEmissionMode::Constant) {
            float color[3] = {
                static_cast<float>(shader->emission.color.x),
                static_cast<float>(shader->emission.color.y),
                static_cast<float>(shader->emission.color.z)
            };
            if (ImGui::ColorEdit3("Color", color)) {
                shader->emission.color = Vec3(color[0], color[1], color[2]);
                changed = true;
            }
            if (ImGui::SliderFloat("Intensity", &shader->emission.intensity, 0.0f, 100.0f)) {
                changed = true;
            }
        }
        else if (shader->emission.mode == VolumeEmissionMode::Blackbody) {
            // Temperature channel selection
            auto temp_grids = vdb->getAvailableGrids();
            if (!temp_grids.empty() && ImGui::BeginCombo("Temperature Channel", 
                                                          shader->emission.temperature_channel.c_str())) {
                for (const auto& grid_name : temp_grids) {
                    if (ImGui::Selectable(grid_name.c_str(), 
                                          shader->emission.temperature_channel == grid_name)) {
                        shader->emission.temperature_channel = grid_name;
                        changed = true;
                    }
                }
                ImGui::EndCombo();
            }
            
            if (ImGui::SliderFloat("Temp Scale", &shader->emission.temperature_scale, 0.1f, 10.0f)) {
                changed = true;
            }
            if (ImGui::SliderFloat("Blackbody Intensity", &shader->emission.blackbody_intensity, 0.0f, 100.0f)) {
                changed = true;
            }
            
            // ═══════════════════════════════════════════════════════════════
            // COLOR RAMP EDITOR
            // ═══════════════════════════════════════════════════════════════
            ImGui::Separator();
            if (ImGui::Checkbox("Use Color Ramp", &shader->emission.color_ramp.enabled)) {
                changed = true;
            }
            
            if (shader->emission.color_ramp.enabled) {
                ImGui::Indent();
                
                // Interactive Gradient Logic
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 p = ImGui::GetCursorScreenPos();
                float width = ImGui::GetContentRegionAvail().x;
                float height = 24.0f;
                float marker_size = 6.0f;
                
                auto& stops = shader->emission.color_ramp.stops;
                static int selected_stop_idx = -1;
                static int dragging_stop_idx = -1;

                // Invisible Button for Interaction
                ImGui::InvisibleButton("gradient_bar", ImVec2(width, height + marker_size * 2));
                
                bool is_hovered = ImGui::IsItemHovered();
                bool is_active = ImGui::IsItemActive(); // Held down
                bool is_clicked = ImGui::IsItemClicked(0);
                bool is_right_clicked = ImGui::IsMouseReleased(1) && is_hovered;
                ImVec2 mouse_pos = ImGui::GetIO().MousePos;
                
                float mouse_t = (mouse_pos.x - p.x) / width;
                mouse_t = std::max(0.0f, std::min(1.0f, mouse_t));

                // Handle Input
                if (is_clicked) {
                    // Check collision with markers
                    bool hit_marker = false;
                    for (int i = 0; i < (int)stops.size(); ++i) {
                        float stop_x = p.x + stops[i].position * width;
                        if (fabs(mouse_pos.x - stop_x) < marker_size + 2 && mouse_pos.y > p.y && mouse_pos.y < p.y + height + marker_size * 2) {
                            selected_stop_idx = i;
                            dragging_stop_idx = i;
                            hit_marker = true;
                            break;
                        }
                    }
                    
                    // Add new stop if clicked empty space
                    if (!hit_marker) {
                        ColorRampStop new_stop;
                        new_stop.position = mouse_t;
                        new_stop.color = shader->emission.color_ramp.sample(mouse_t); // Sample grid color
                        new_stop.alpha = 1.0f;
                        stops.push_back(new_stop);
                        
                        // Sort and find new index
                        std::sort(stops.begin(), stops.end(), [](const auto& a, const auto& b){ return a.position < b.position; });
                        for(int i=0; i<(int)stops.size(); ++i) if(stops[i].position == mouse_t) { selected_stop_idx = i; dragging_stop_idx = i; break; }
                        changed = true;
                    }
                }
                
                // Dragging Logic
                if (dragging_stop_idx != -1 && ImGui::IsMouseDown(0)) {
                    stops[dragging_stop_idx].position = mouse_t;
                    
                    // Keep sorted
                    std::sort(stops.begin(), stops.end(), [](const auto& a, const auto& b){ return a.position < b.position; });
                    // Re-find dragging index (since sort might have moved it)
                    // We rely on position matching (slight risk with floats, but mouse_t is exact assignee)
                    // Better: find by unique property? But stops lack ID. 
                    // Robustness: If we drag one past another, index *should* swap.
                    // Simple heuristic: Look for the stop with 'mouse_t' position.
                    for(int i=0; i<(int)stops.size(); ++i) {
                         if(fabs(stops[i].position - mouse_t) < 0.0001f) {
                             dragging_stop_idx = i;
                             selected_stop_idx = i;
                             break;
                         }
                    }
                    changed = true;
                } else {
                    dragging_stop_idx = -1;
                }

                // Delete Stop (Right Click or Delete Key)
                if ((is_right_clicked && selected_stop_idx != -1) || (ImGui::IsKeyPressed(ImGuiKey_Delete) && selected_stop_idx != -1)) {
                     // Check if mouse is near selected stop for right click context
                     // For simplicity, just delete selected if right click happened in bar? No, unsafe.
                     // Only delete if right clicked NEAR the marker.
                     if (stops.size() > 2) {
                         // Find which one was right clicked
                         for (int i = 0; i < (int)stops.size(); ++i) {
                            float stop_x = p.x + stops[i].position * width;
                            if (fabs(mouse_pos.x - stop_x) < marker_size + 2) {
                                stops.erase(stops.begin() + i);
                                selected_stop_idx = -1;
                                dragging_stop_idx = -1;
                                changed = true;
                                break;
                            }
                        }
                     }
                }

                // Draw Gradient
                for (int i = 0; i < (int)width; ++i) {
                    float t = (float)i / width;
                    Vec3 c = shader->emission.color_ramp.sample(t);
                    ImU32 col = IM_COL32((int)(c.x * 255), (int)(c.y * 255), (int)(c.z * 255), 255);
                    draw_list->AddRectFilled(ImVec2(p.x + i, p.y), ImVec2(p.x + i + 1, p.y + height), col);
                }
                
                // Draw Markers
                for (int i = 0; i < (int)stops.size(); ++i) {
                    float x = p.x + stops[i].position * width;
                    bool is_sel = (i == selected_stop_idx);
                    ImU32 marker_col = is_sel ? IM_COL32(255, 255, 0, 255) : IM_COL32(255, 255, 255, 255);
                    
                    // Triangle pointing up
                    draw_list->AddTriangleFilled(
                        ImVec2(x, p.y + height + marker_size * 2),
                        ImVec2(x - marker_size, p.y + height),
                        ImVec2(x + marker_size, p.y + height),
                        marker_col
                    );
                    draw_list->AddTriangle( // Outline
                        ImVec2(x, p.y + height + marker_size * 2),
                        ImVec2(x - marker_size, p.y + height),
                        ImVec2(x + marker_size, p.y + height),
                        IM_COL32(0, 0, 0, 180)
                    );
                }
                
                ImGui::Dummy(ImVec2(width, marker_size * 2 + 5)); // Spacing below

                // Selected Stop Editor
                if (selected_stop_idx >= 0 && selected_stop_idx < (int)stops.size()) {
                    ImGui::Separator();
                    ImGui::Text("Selected Stop (%d)", selected_stop_idx);
                    
                    ColorRampStop& s = stops[selected_stop_idx];
                    
                    if (ImGui::SliderFloat("Position##stop", &s.position, 0.0f, 1.0f)) {
                         std::sort(stops.begin(), stops.end(), [](const auto& a, const auto& b){ return a.position < b.position; });
                         // Reselect after sort
                         for(int i=0; i<(int)stops.size(); ++i) if(fabs(stops[i].position - s.position) < 0.001f) { selected_stop_idx = i; break; }
                         changed = true;
                    }
                    
                    float col[3] = {s.color.x, s.color.y, s.color.z};
                    if (ImGui::ColorEdit3("Color##stop", col)) {
                        s.color = Vec3(col[0], col[1], col[2]);
                        changed = true;
                    }
                    
                    if (ImGui::Button("Delete Stop") && stops.size() > 2) {
                        stops.erase(stops.begin() + selected_stop_idx);
                        selected_stop_idx = -1;
                        changed = true;
                    }
                }
                
                if (ImGui::Button("Reset Fire Preset")) {
                     vdb->setShader(VolumeShader::createFirePreset());
                     changed = true;
                }

                
                ImGui::Unindent();
            }
            else {
                // Standard blackbody preview when ramp is disabled
                ImDrawList* draw_list_bb = ImGui::GetWindowDrawList();
                ImVec2 p_bb = ImGui::GetCursorScreenPos();
                float width_bb = ImGui::GetContentRegionAvail().x;
                float height_bb = 20.0f;
                
                for (int i = 0; i < (int)width_bb; ++i) {
                    float t = (float)i / width_bb;
                    float r = std::min(1.0f, t * 2.0f);
                    float g = std::max(0.0f, std::min(1.0f, (t - 0.3f) * 1.5f));
                    float b = std::max(0.0f, (t - 0.6f) * 2.5f);
                    ImU32 col = IM_COL32((int)(r*255), (int)(g*255), (int)(b*255), 255);
                    draw_list_bb->AddRectFilled(
                        ImVec2(p_bb.x + i, p_bb.y),
                        ImVec2(p_bb.x + i + 1, p_bb.y + height_bb),
                        col
                    );
                }
                ImGui::Dummy(ImVec2(width_bb, height_bb + 4));
                ImGui::TextDisabled("Cold -------- Hot (Physical Blackbody)");
            }
        }
        else if (shader->emission.mode == VolumeEmissionMode::ChannelDriven) {
            auto em_grids = vdb->getAvailableGrids();
            if (!em_grids.empty() && ImGui::BeginCombo("Emission Channel", 
                                                        shader->emission.emission_channel.c_str())) {
                for (const auto& grid_name : em_grids) {
                    if (ImGui::Selectable(grid_name.c_str(), 
                                          shader->emission.emission_channel == grid_name)) {
                        shader->emission.emission_channel = grid_name;
                        changed = true;
                    }
                }
                ImGui::EndCombo();
            }
        }
        
        ImGui::PopID();
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // QUALITY
    // ─────────────────────────────────────────────────────────────────────────
    if (ImGui::CollapsingHeader("Ray Marching Quality")) {
        ImGui::PushID("quality");
        
        if (ImGui::SliderFloat("Step Size", &shader->quality.step_size, 0.01f, 1.0f)) {
            changed = true;
        }
        if (ImGui::SliderInt("Max Steps", &shader->quality.max_steps, 8, 1024)) {
            changed = true;
        }
        if (ImGui::SliderInt("Shadow Steps", &shader->quality.shadow_steps, 0, 32)) {
            changed = true;
        }
        if (ImGui::SliderFloat("Shadow Strength", &shader->quality.shadow_strength, 0.0f, 1.0f)) {
            changed = true;
        }
        
        ImGui::PopID();
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // SHADER PRESETS
    // ─────────────────────────────────────────────────────────────────────────
    ImGui::Separator();
    ImGui::Text("Presets:");
    ImGui::SameLine();
    if (ImGui::SmallButton("Smoke")) {
        vdb->setShader(VolumeShader::createSmokePreset());
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Fire")) {
        vdb->setShader(VolumeShader::createFirePreset());
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Explosion")) {
        vdb->setShader(VolumeShader::createExplosionPreset());
        changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Cloud")) {
        vdb->setShader(VolumeShader::createCloudPreset());
        changed = true;
    }
    
    // Animation section already exists above (lines 429-449)
    
    ImGui::PopStyleColor();
    
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
