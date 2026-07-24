/**
 * @file scene_ui_vdb.cpp
 * @brief VDB Volume UI Panel
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
#include "Backend/IBackend.h"
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanBackend.h"
#include <imgui.h>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <VolumetricRenderer.h>

// External GPU availability flag
extern bool g_hasOptix;
extern std::unique_ptr<Backend::IBackend> g_backend;
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

namespace {
Backend::IBackend* getVdbRenderBackend(UIContext& ctx) {
    if (g_backend) {
        return g_backend.get();
    }
    if (ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) == nullptr) {
        return ctx.backend_ptr;
    }
    return nullptr;
}

bool vdbRenderBackendIsVulkan(UIContext& ctx) {
    return dynamic_cast<Backend::VulkanBackendAdapter*>(getVdbRenderBackend(ctx)) != nullptr;
}
}

// ─────────────────────────────────────────────────────────────────────────────
// HELPER: Sync VDB volumes to GPU (OptiX) - with GPU availability check
// ─────────────────────────────────────────────────────────────────────────────
void SceneUI::syncVDBVolumesToGPU(UIContext& ctx) {
    // Sync to GPU for both OptiX and Vulkan backends.
    // Previously guarded by !g_hasOptix which caused animated VDB frames
    // to never be re-uploaded in Vulkan mode.
    if (Backend::IBackend* renderBackend = getVdbRenderBackend(ctx)) {
        WorldData wd = ctx.renderer.world.getGPUData();
        VolumetricRenderer::syncVolumetricData(ctx.scene, renderBackend, &wd);
    }
}

bool SceneUI::shouldApplySpecialVDBOrientation(const std::string& source_hint) {
    std::string lowered = source_hint;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered.find("embergen") != std::string::npos;
}

void SceneUI::applyVDBImportOrientation(VDBVolume& vdb, int orientation_preset, const std::string& source_hint) {
    bool apply_axis_conversion = false;
    if (orientation_preset == 2) {
        apply_axis_conversion = true;
    } else if (orientation_preset == 0) {
        apply_axis_conversion = shouldApplySpecialVDBOrientation(source_hint);
    }

    if (apply_axis_conversion) {
        vdb.setRotation(Vec3(-90.0f, 0.0f, 0.0f));
    } else if (orientation_preset == 1) {
        vdb.setRotation(Vec3(0.0f, 0.0f, 0.0f));
    }
}

SceneUI::VDBShaderDefaults SceneUI::estimateVDBShaderDefaults(const VDBVolume& vdb) {
    VDBShaderDefaults out;

    const VDBDensityStats density_stats = VDBVolumeManager::getInstance().analyzeDensityStats(vdb.getVDBVolumeID());
    const Vec3 bmin = vdb.getLocalBoundsMin();
    const Vec3 bmax = vdb.getLocalBoundsMax();
    const Vec3 extent = bmax - bmin;
    const float max_extent = (std::max)(extent.x, (std::max)(extent.y, extent.z));
    const bool is_fire = vdb.hasGrid("temperature");

    const float ref_density = density_stats.valid
        ? (density_stats.p99_value > 1e-5f ? density_stats.p99_value :
           (density_stats.p95_value > 1e-5f ? density_stats.p95_value : density_stats.max_value))
        : 0.0f;

    if (ref_density > 1e-6f) {
        const float target = is_fire ? 0.18f : 0.35f;
        const float min_mult = is_fire ? 0.75f : 2.0f;
        const float max_mult = is_fire ? 16.0f : 35.0f;
        out.density_multiplier = (std::max)(min_mult, (std::min)(max_mult, target / ref_density));
    }

    if (is_fire) {
        out.scattering_coefficient = (std::max)(0.35f, (std::min)(0.9f, 0.35f + out.density_multiplier * 0.08f));
        out.absorption_coefficient = (std::max)(0.12f, (std::min)(0.35f, 0.12f + out.density_multiplier * 0.03f));
        out.max_steps = 96;
        // shadow_steps: 6 → 4. OptiX uses 4 by default and visual difference past 4 is
        // imperceptible for diffuse phase functions; cuts shadow march cost ~33%.
        out.shadow_steps = 4;
    } else {
        out.scattering_coefficient = (std::max)(0.9f, (std::min)(1.8f, 0.9f + out.density_multiplier * 0.08f));
        out.absorption_coefficient = (std::max)(0.03f, (std::min)(0.12f, 0.03f + out.density_multiplier * 0.005f));
        out.max_steps = 72;
        // shadow_steps: 8 → 4. Match OptiX default; halves shadow march cost on smoke.
        out.shadow_steps = 4;
    }

    const float voxel_size = vdb.getVoxelSize();
    const float voxel_based_step = voxel_size > 1e-5f ? voxel_size * (is_fire ? 1.25f : 1.75f) : (is_fire ? 0.08f : 0.15f);
    const float extent_based_step = max_extent > 1e-4f ? max_extent / (is_fire ? 160.0f : 192.0f) : 0.05f;
    out.step_size = (std::max)(0.01f, (std::min)(0.35f, (std::max)(voxel_based_step, extent_based_step)));

    return out;
}

void SceneUI::applyEstimatedVDBShaderDefaults(VDBVolume& vdb) {
    auto shader = vdb.getShader();
    if (!shader) return;

    const VDBShaderDefaults defaults = estimateVDBShaderDefaults(vdb);
    shader->density.multiplier = defaults.density_multiplier;
    shader->scattering.coefficient = defaults.scattering_coefficient;
    shader->absorption.coefficient = defaults.absorption_coefficient;
    shader->quality.step_size = defaults.step_size;
    const float voxel_size = vdb.getVoxelSize();
    if (voxel_size > 1e-5f) {
        shader->quality.voxel_step_multiplier =
            (std::max)(0.1f, (std::min)(2.0f, defaults.step_size / voxel_size));
        shader->quality.adaptive_stepping = true;
        shader->quality.quality_preset = 4;
    }
    shader->quality.max_steps = defaults.max_steps;
    shader->quality.shadow_steps = defaults.shadow_steps;
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
    applyVDBImportOrientation(*vdb, vdb_import_orientation_preset, path);
    
    // Add to scene VDB list
    ctx.scene.addVDBVolume(vdb);
    
    // Also add to world.objects as Hittable for CPU ray intersection
    ctx.scene.world.objects.push_back(vdb);
    
    // Trigger BVH rebuild for the new hittable
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    
    if (vdbRenderBackendIsVulkan(ctx)) {
        // Vulkan: must rebuild TLAS to add the new AABB instance.
        extern bool g_viewport_raster_rebuild_pending;
        g_viewport_raster_rebuild_pending = true;
        g_vulkan_rebuild_pending = true;
    } else if (Backend::IBackend* renderBackend = getVdbRenderBackend(ctx)) {
        syncVDBVolumesToGPU(ctx);
        renderBackend->resetAccumulation();
        extern bool g_viewport_raster_rebuild_pending;
        g_viewport_raster_rebuild_pending = true;
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
    applyVDBImportOrientation(*vdb, vdb_import_orientation_preset, path);
    
    // Add to scene
    ctx.scene.addVDBVolume(vdb);
    ctx.scene.world.objects.push_back(vdb);
    
    // Trigger BVH build
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();
    
    if (ctx.backend_ptr) {
        if (vdbRenderBackendIsVulkan(ctx)) {
            extern bool g_viewport_raster_rebuild_pending;
            g_viewport_raster_rebuild_pending = true;
            g_vulkan_rebuild_pending = true;
        } else {
            syncVDBVolumesToGPU(ctx);
            extern bool g_viewport_raster_rebuild_pending;
            g_viewport_raster_rebuild_pending = true;
        }
        if (Backend::IBackend* renderBackend = getVdbRenderBackend(ctx)) {
            renderBackend->resetAccumulation();
        }
        if (g_viewport_backend && g_viewport_backend.get() != getVdbRenderBackend(ctx)) {
            g_viewport_backend->resetAccumulation();
        }
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
// VOLUMETRIC PROPERTIES PANEL (VDB IMPORT / RENDER)
// ═══════════════════════════════════════════════════════════════════════════════

void SceneUI::drawVolumetricPanel(UIContext& ctx) {
    auto& scene = ctx.scene;
    auto& selection = ctx.selection;
    
    UIWidgets::PushControlSurfaceStyle(ImVec4(0.62f, 0.78f, 1.0f, 1.0f));
    float vdb_child_round = 4.0f;
    if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
        vdb_child_round = ThemeManager::instance().current().style.windowRounding;
    }
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, vdb_child_round);
    
    // -------------------------------------------------------------
    // TOP SECTION: CREATION & IMPORT
    // -------------------------------------------------------------
    ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "Import VDB Volumes");
    ImGui::TextDisabled("Legacy GasVolume UI moved to Simulations > Domains; this panel keeps VDB import/render only.");
    // Legacy GasVolume creation/listing is intentionally retired from this panel.
    // Keep the GasVolume runtime code for old scene compatibility and Simulation UI ownership.
    if (UIWidgets::IconActionButton("VolumeImportVDB", UIWidgets::IconType::Assets, "Import VDB", false,
        ImVec4(0.82f, 0.70f, 1.0f, 1.0f), ImVec2(132, 30), "Import a single VDB volume file.")) {
        importVDBVolume(ctx);
    }
    ImGui::SameLine();
    if (UIWidgets::IconActionButton("VolumeImportVDBSeq", UIWidgets::IconType::Timeline, "Import Seq", false,
        ImVec4(0.98f, 0.76f, 0.42f, 1.0f), ImVec2(132, 30), "Import an animated VDB sequence.")) {
        importVDBSequence(ctx);
    }
    const char* vdb_orientation_labels[] = { "Auto", "Standard", "Axis Convert (-90 X)" };
    ImGui::SetNextItemWidth(180.0f);
    ImGui::Combo("VDB Orientation", &vdb_import_orientation_preset, vdb_orientation_labels, IM_ARRAYSIZE(vdb_orientation_labels));
    
    ImGui::Separator();
    
    // -------------------------------------------------------------
    // MIDDLE SECTION: VDB OBJECT LIST
    // -------------------------------------------------------------
    size_t total_vols = scene.vdb_volumes.size();
    ImGui::Text("VDB Volumes (%zu)", total_vols);
    
    if (ImGui::BeginListBox("##volume_list", ImVec2(-FLT_MIN, 150))) {
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
                        // Vulkan: rebuild TLAS to remove the stale AABB instance.
                        // OptiX: sync SSBO to remove volume.
                        if (vdbRenderBackendIsVulkan(ctx)) {
                            g_vulkan_rebuild_pending = true;
                            syncVDBVolumesToGPU(ctx);
                            if (ctx.backend_ptr) ctx.backend_ptr->resetAccumulation();
                        } else {
                            syncVDBVolumesToGPU(ctx);
                        }
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
        ImGui::TextDisabled("Legacy GasVolume properties are no longer edited here. Use Simulations > Domains.");
    }
    else if (selection.selected.type == SelectableType::VDBVolume && selection.selected.vdb_volume) {
        drawVDBVolumeProperties(ctx, selection.selected.vdb_volume.get());
    }
    else {
        ImGui::TextDisabled("Select a Volume object to edit properties.");
    }
    
    ImGui::PopStyleVar();
    UIWidgets::PopControlSurfaceStyle();
}

void SceneUI::drawVDBVolumeProperties(UIContext& ctx, VDBVolume* vdb) {
    if (!vdb) return;
    
    // UNIQUE ID SCOPE: Use VDB volume ID to prevent ImGui ID conflicts
    ImGui::PushID(vdb->getVDBVolumeID());
    UIWidgets::PushControlSurfaceStyle(ImVec4(0.76f, 0.64f, 1.0f, 1.0f));
    
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
        if (UIWidgets::SecondaryButton("Apply Axis Convert -90 X")) {
            vdb->setRotation(Vec3(-90.0f, 0, 0));
            changed = true;
        }
        ImGui::SameLine();
        if (UIWidgets::SecondaryButton("Scale 0.001 (mm Fix)")) {
            vdb->setScale(Vec3(0.001f));
            changed = true;
        }
        ImGui::SameLine();
        if (UIWidgets::DangerButton("Reset All")) {
            vdb->setRotation(Vec3(0));
            vdb->setScale(Vec3(1.0f));
            changed = true;
        }

        UIWidgets::EndSection();
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // VOLUME RENDER MODE
    // ═══════════════════════════════════════════════════════════════════════════
    if (UIWidgets::BeginSection("Volume Render Mode", ImVec4(0.3f, 0.7f, 0.9f, 1.0f))) {
        int current_mode = vdb->render_as_isosurface ? 1 : 0;
        const char* render_modes[] = { "Volumetric Fog / Gas", "Refractive Fluid Surface (SDF Isosurface)" };
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 10.0f);
        if (ImGui::Combo("##VDBRenderModeCombo", &current_mode, render_modes, IM_ARRAYSIZE(render_modes))) {
            vdb->render_as_isosurface = (current_mode == 1);
            
            // Auto-apply beautiful defaults on mode change to make life super easy for the user!
            if (vdb->render_as_isosurface) {
                // Initialize with premium fluid look
                vdb->render_isosurface_ior = 1.333f; // Water IOR
                vdb->render_isosurface_roughness = 0.02f; // Smooth liquid
                vdb->render_isosurface_foam = 0.0f;
                
                // Automatically set shader values matching simulation level-set look
                auto shader = vdb->getOrCreateShader();
                shader->name = "Liquid Surface (SDF)";
                shader->density.multiplier = 60.0f; // High density to create solid boundary
                shader->density.cutoff_threshold = 0.05f;
                shader->scattering.color = Vec3(0.92f, 0.96f, 1.0f); // Light blue tint
                shader->scattering.coefficient = 0.4f;
                shader->scattering.anisotropy = 0.0f;
                shader->absorption.color = Vec3(0.85f, 0.40f, 0.12f); // Orange absorption -> blue water depth look!
                shader->absorption.coefficient = 2.5f;
                shader->emission.mode = VolumeEmissionMode::None;
                
                // Raymarching steps for high quality refraction/depth
                shader->quality.max_steps = 256;
                shader->quality.step_size = 0.05f;
            } else {
                // Restore standard smoke preset
                auto shader = vdb->getOrCreateShader();
                shader->name = "Smoke Preset";
                shader->density.multiplier = 1.0f;
                shader->density.cutoff_threshold = 0.0f;
                shader->scattering.color = Vec3(0.8f, 0.8f, 0.8f);
                shader->scattering.coefficient = 1.0f;
                shader->scattering.anisotropy = 0.0f;
                shader->absorption.color = Vec3(0.1f, 0.1f, 0.1f);
                shader->absorption.coefficient = 0.1f;
            }
            changed = true;
        }
        
        if (vdb->render_as_isosurface) {
            ImGui::Spacing();
            ImGui::Indent();
            
            if (ImGui::DragFloat("Index of Refraction (IOR)", &vdb->render_isosurface_ior, 0.01f, 1.0f, 3.0f, "%.3f")) changed = true;
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Water: 1.333 | Glass: 1.500 | Ice: 1.309 | Diamond: 2.417");
            }
            
            if (ImGui::SliderFloat("Surface Roughness", &vdb->render_isosurface_roughness, 0.0f, 1.0f, "%.3f")) changed = true;
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Roughness of the liquid surface boundary (GGX microfacet).");
            }
            
            if (ImGui::SliderFloat("Curvature Foam Strength", &vdb->render_isosurface_foam, 0.0f, 1.0f, "%.3f")) changed = true;
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Strength of the curvature-driven foam/whitewater highlights.");
            }
            
            ImGui::Spacing();
            if (UIWidgets::SecondaryButton("Reset Fluid Material Preset")) {
                auto shader = vdb->getOrCreateShader();
                shader->density.multiplier = 60.0f;
                shader->density.cutoff_threshold = 0.05f;
                shader->scattering.color = Vec3(0.92f, 0.96f, 1.0f);
                shader->scattering.coefficient = 0.4f;
                shader->absorption.color = Vec3(0.85f, 0.40f, 0.12f);
                shader->absorption.coefficient = 2.5f;
                shader->emission.mode = VolumeEmissionMode::None;
                shader->quality.max_steps = 256;
                shader->quality.step_size = 0.05f;
                changed = true;
            }
            
            ImGui::Unindent();
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
        if (ctx.backend_ptr) {
            syncVDBVolumesToGPU(ctx);  // Sync shader changes to GPU
            ctx.backend_ptr->resetAccumulation();
        }
        // [VULKAN FIX] Explicitly kick off a fresh render pass so property changes
        // are visible immediately even when accumulation was already complete.
        // The auto-progressive loop also handles this via isAccumulationComplete(),
        // but setting start_render=true here guarantees the very next main-loop
        // frame dispatches a new GPU sample with the updated SSBO.
        ctx.start_render = true;
        g_ProjectManager.markModified();
    }
    
    UIWidgets::PopControlSurfaceStyle();
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
