/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_water.hpp
* Author:        Kemal DemirtaÅŸ
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
#ifndef SCENE_UI_WATER_HPP
#define SCENE_UI_WATER_HPP

#include "scene_ui.h"
#include "imgui.h"
#include "scene_data.h" // Required for ctx.scene members (camera, etc.)

// ═══════════════════════════════════════════════════════════════════════════════
// WATER PANEL UI
// ═══════════════════════════════════════════════════════════════════════════════

#include "WaterSystem.h"
#include "PrincipledBSDF.h"
#include "ProjectManager.h"
#include "Backend/IViewportBackend.h"
#include "Backend/VulkanBackend.h"

extern std::unique_ptr<Backend::IBackend> g_backend;
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;
extern bool g_viewport_raster_rebuild_pending;
extern bool g_optix_rebuild_pending;
extern bool g_vulkan_rebuild_pending;
extern bool g_materials_dirty;

extern bool g_geometry_dirty;
extern std::atomic<uint64_t> g_scene_geometry_generation;
extern std::atomic<bool> g_needs_optix_sync;
extern bool g_mesh_cache_dirty;

namespace {
bool BeginWaterSection(const char* title, const ImVec4& accent, bool defaultOpen = true) {
    ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(accent.x * 0.24f, accent.y * 0.24f, accent.z * 0.24f, 0.92f));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(accent.x * 0.32f, accent.y * 0.32f, accent.z * 0.32f, 0.98f));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(accent.x * 0.38f, accent.y * 0.38f, accent.z * 0.38f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.94f, 0.96f, 0.99f, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 7.0f));
    
    float fr = 10.0f;
    if (ThemeManager::instance().getIconSettings().overridePanelAccentsWithTheme) {
        fr = ThemeManager::instance().current().style.frameRounding;
    }
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, fr);
    
    const bool open = ImGui::CollapsingHeader(
        title,
        (defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen : 0) | ImGuiTreeNodeFlags_SpanAvailWidth);
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(4);
    return open;
}

void EndWaterSection() {}

Backend::IBackend* getWaterRenderBackend(UIContext& ctx) {
    if (g_backend) {
        return g_backend.get();
    }
    if (ctx.backend_ptr && dynamic_cast<Backend::IViewportBackend*>(ctx.backend_ptr) == nullptr) {
        return ctx.backend_ptr;
    }
    return nullptr;
}

bool waterRenderBackendIsVulkan(UIContext& ctx) {
    return dynamic_cast<Backend::VulkanBackendAdapter*>(getWaterRenderBackend(ctx)) != nullptr;
}

void syncWaterMaterialPreview(UIContext& ctx, WaterSurface& surf) {
    WaterManager::getInstance().syncSurfaceMaterial(&surf);
    g_materials_dirty = true;

    ctx.renderer.updateBackendMaterial(ctx.scene, surf.material_id);
    ctx.renderer.resetCPUAccumulation();

    if (Backend::IBackend* renderBackend = getWaterRenderBackend(ctx)) {
        renderBackend->resetAccumulation();
    }
    if (g_viewport_backend && g_viewport_backend.get() != getWaterRenderBackend(ctx)) {
        g_viewport_backend->resetAccumulation();
    }
}

void rebuildWaterSceneMutation(UIContext& ctx, bool updateMaterials) {
    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
    ctx.renderer.resetCPUAccumulation();

    if (Backend::IBackend* renderBackend = getWaterRenderBackend(ctx)) {
        ctx.renderer.rebuildBackendGeometry(ctx.scene);
        if (updateMaterials) {
            ctx.renderer.updateBackendMaterials(ctx.scene);
        }
        renderBackend->resetAccumulation();
    }

    if (g_viewport_backend) {
        g_viewport_raster_rebuild_pending = true;
        g_viewport_backend->resetAccumulation();
    }
    if (waterRenderBackendIsVulkan(ctx)) {
        g_vulkan_rebuild_pending = true;
    }
    
    // Force global scene geometry updates for Vulkan Raster backend when water is created/updated
    // Variables are declared at the global scope above
    
    g_geometry_dirty = true;
    g_scene_geometry_generation.fetch_add(1, std::memory_order_release);
    g_needs_optix_sync.store(true, std::memory_order_release);
    g_mesh_cache_dirty = true;
}
}

inline bool SceneUI::drawWaterSurfaceMaterialEditor(UIContext& ctx, WaterSurface& surf, bool allow_delete) {
    bool changed = false;

    const char* profileName = "Ocean";
    if (surf.type == WaterSurface::Type::River) profileName = "River";
    else if (surf.type == WaterSurface::Type::Lake) profileName = "Lake";
    ImGui::TextDisabled("Vulkan RT Profile");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.30f, 0.82f, 1.0f, 1.0f), "%s", profileName);
    ImGui::SetItemTooltip("The render profile is owned by the water geometry. Presets tune its appearance; they do not change a lake mesh into a river.");

    const char* preset_names[4] = { "Custom", nullptr, nullptr, nullptr };
    WaterWaveParams::WaterPreset preset_values[4] = { WaterWaveParams::WaterPreset::Custom };
    int preset_count = 1;
    if (surf.type == WaterSurface::Type::River) {
        preset_names[preset_count] = "River Flow";
        preset_values[preset_count++] = WaterWaveParams::WaterPreset::River;
    } else if (surf.type == WaterSurface::Type::Lake) {
        preset_names[preset_count] = "Still Lake";
        preset_values[preset_count++] = WaterWaveParams::WaterPreset::Lake;
        preset_names[preset_count] = "Clear Pool";
        preset_values[preset_count++] = WaterWaveParams::WaterPreset::Pool;
        preset_names[preset_count] = "Murky Pond";
        preset_values[preset_count++] = WaterWaveParams::WaterPreset::Pond;
    } else {
        preset_names[preset_count] = "Calm Ocean";
        preset_values[preset_count++] = WaterWaveParams::WaterPreset::CalmOcean;
        preset_names[preset_count] = "Stormy Ocean";
        preset_values[preset_count++] = WaterWaveParams::WaterPreset::StormyOcean;
        preset_names[preset_count] = "Tropical Ocean";
        preset_values[preset_count++] = WaterWaveParams::WaterPreset::TropicalOcean;
    }

    int preset_idx = 0;
    for (int i = 1; i < preset_count; ++i) {
        if (preset_values[i] == surf.params.current_preset) preset_idx = i;
    }
    ImGui::SetNextItemWidth(-1);
    if (ImGui::Combo("Preset", &preset_idx, preset_names, preset_count)) {
        WaterWaveParams::WaterPreset new_preset = preset_values[preset_idx];
        if (new_preset != WaterWaveParams::WaterPreset::Custom) {
            surf.params.applyPreset(new_preset);
            changed = true;
        } else {
            surf.params.current_preset = WaterWaveParams::WaterPreset::Custom;
        }
    }

    if (BeginWaterSection("Colors", ImVec4(0.0f, 0.8f, 0.8f, 1.0f))) {
        changed |= ImGui::ColorEdit3("Shallow Color", &surf.params.shallow_color.x);
        changed |= ImGui::ColorEdit3("Deep Color", &surf.params.deep_color.x);
        EndWaterSection();
    }

    if (BeginWaterSection("Depth & Optics", ImVec4(0.0f, 0.2f, 0.6f, 1.0f), false)) {
        changed |= SceneUI::DrawSmartFloat("shared_w_ior", "IOR", &surf.params.ior, 1.0f, 2.0f, "%.3f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_rgh", "Roughness", &surf.params.roughness, 0.0f, 0.2f, "%.3f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_dmax", "Max Depth", &surf.params.depth_max, 1.0f, 100.0f, "%.1f m", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_absd", "Absorption Density", &surf.params.absorption_density, 0.0f, 2.0f, "%.2f", false, nullptr, 16);
        EndWaterSection();
    }

    if (BeginWaterSection("Surface Motion", ImVec4(0.18f, 0.62f, 1.0f, 1.0f))) {
        const char* speedLabel = surf.type == WaterSurface::Type::River ? "Flow Speed" : "Wave Speed";
        const char* strengthLabel = surf.type == WaterSurface::Type::River ? "Ripple Strength" : "Wave Strength";
        changed |= SceneUI::DrawSmartFloat("shared_w_speed", speedLabel, &surf.params.wave_speed, 0.0f, 8.0f, "%.2f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_strength", strengthLabel, &surf.params.wave_strength, 0.0f, 2.0f, "%.3f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_frequency", "Wave Frequency", &surf.params.wave_frequency, 0.02f, 8.0f, "%.2f", false, nullptr, 16);

        if (surf.type != WaterSurface::Type::River) {
            float directionDegrees = surf.params.fft_wind_direction * 180.0f / 3.14159265f;
            if (SceneUI::DrawSmartFloat("shared_w_direction", "Travel Direction", &directionDegrees, 0.0f, 360.0f, "%.0f deg", false, nullptr, 16)) {
                surf.params.fft_wind_direction = directionDegrees * 3.14159265f / 180.0f;
                changed = true;
            }
        }
        EndWaterSection();
    }

    if (BeginWaterSection("Foam", ImVec4(0.9f, 0.9f, 0.9f, 1.0f), false)) {
        changed |= SceneUI::DrawSmartFloat("shared_w_wf", "Wave Foam", &surf.params.foam_level, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_fi", "Shore Intensity", &surf.params.shore_foam_intensity, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_fd", "Shore Distance", &surf.params.shore_foam_distance, 0.1f, 10.0f, "%.1f m", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_fns", "Foam Noise Scale", &surf.params.foam_noise_scale, 1.0f, 50.0f, "%.1f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_fth", "Foam Threshold", &surf.params.foam_threshold, 0.0f, 1.0f, "%.2f", false, nullptr, 16);
        EndWaterSection();
    }

    if (BeginWaterSection("Surface Detail", ImVec4(1.0f, 0.6f, 0.0f, 1.0f), false)) {
        changed |= SceneUI::DrawSmartFloat("shared_w_mds", "Micro Detail Strength", &surf.params.micro_detail_strength, 0.0f, 0.2f, "%.3f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_msc", "Micro Detail Scale", &surf.params.micro_detail_scale, 1.0f, 100.0f, "%.1f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_mas", "Micro Anim Speed", &surf.params.micro_anim_speed, 0.01f, 1.0f, "%.3f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_mms", "Micro Morph Speed", &surf.params.micro_morph_speed, 0.1f, 5.0f, "%.2f", false, nullptr, 16);
        EndWaterSection();
    }

    if (BeginWaterSection("Caustics", ImVec4(0.25f, 0.85f, 0.70f, 1.0f), false)) {
        changed |= SceneUI::DrawSmartFloat("shared_w_ci", "Intensity", &surf.params.caustic_intensity, 0.0f, 2.0f, "%.2f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_cs", "Scale", &surf.params.caustic_scale, 0.1f, 20.0f, "%.2f", false, nullptr, 16);
        changed |= SceneUI::DrawSmartFloat("shared_w_csp", "Animation Speed", &surf.params.caustic_speed, 0.0f, 5.0f, "%.2f", false, nullptr, 16);
        EndWaterSection();
    }

    if (changed) {
        surf.params.current_preset = WaterWaveParams::WaterPreset::Custom;
        syncWaterMaterialPreview(ctx, surf);
        ProjectManager::getInstance().markModified();
    }

    if (allow_delete) {
        ImGui::Spacing();
        ImGui::Separator();
        if (UIWidgets::DangerButton("Delete Water Surface", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
            WaterManager::getInstance().removeWaterSurface(ctx.scene, surf.id);
            rebuildWaterSceneMutation(ctx, true);
            ProjectManager::getInstance().markModified();
            return true;
        }
    }

    return false;
}

void SceneUI::drawWaterPanel(UIContext& ctx) {
    ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "WATER SYSTEM");
    ImGui::Separator();
    
    // Create new water
    static float new_water_size = 20.0f;
    static int new_water_density_idx = 2;
    const float water_density_values[] = { 1.5f, 3.0f, 5.0f, 8.0f, 12.0f };
    const char* water_density_labels[] = { "Low", "Medium", "High", "Very High", "Ultra" };

    SceneUI::DrawSmartFloat("new_water_size", "Plane Size", &new_water_size, 2.0f, 500.0f, "%.1f m", false, nullptr, 16);
    ImGui::SetItemTooltip("Initial world size of the generated water plane");
    ImGui::Combo("Mesh Quality", &new_water_density_idx, water_density_labels, IM_ARRAYSIZE(water_density_labels));
    ImGui::SetItemTooltip("Initial triangle density for the generated plane. Higher values reduce faceting on large geometric waves but cost more.");

    if (UIWidgets::PrimaryButton("Add Water Plane", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
        // Create centered on world origin so the generated plane is truly centered in scene space.
        Vec3 spawn_pos(0, 0, 0);

        int density_idx = std::clamp(new_water_density_idx, 0, (int)IM_ARRAYSIZE(water_density_values) - 1);
        WaterManager::getInstance().createWaterPlane(ctx.scene, spawn_pos, new_water_size, water_density_values[density_idx]);

        rebuildWaterSceneMutation(ctx, false);
        SCENE_LOG_INFO("[Water] Created new water plane");
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    
    // List existing water surfaces
    auto& waters = WaterManager::getInstance().getWaterSurfaces();
    
    if (waters.empty()) {
        ImGui::TextDisabled("No water surfaces in scene.");
    } else {
        ImGui::Text("Active Water Surfaces (%zu):", waters.size());
        
        if (ImGui::BeginListBox("##waterlist", ImVec2(-1, 80))) {
            for (int i = 0; i < waters.size(); i++) {
                bool is_selected = (selected_water_surface_id == waters[i].id);
                if (ImGui::Selectable(waters[i].name.c_str(), is_selected)) {
                    selected_water_surface_id = waters[i].id;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndListBox();
        }

        const char* preview_mode_names[] = {
            "Realtime Preview",
            "Timeline Preview",
            "Static Preview"
        };
        int preview_mode = static_cast<int>(WaterManager::getInstance().getPreviewTimeMode());
        ImGui::SetNextItemWidth(-1);
        if (ImGui::Combo("Water Preview Mode", &preview_mode, preview_mode_names, IM_ARRAYSIZE(preview_mode_names))) {
            WaterManager::getInstance().setPreviewTimeMode(static_cast<WaterPreviewTimeMode>(preview_mode));
            ctx.renderer.resetCPUAccumulation();
            if (Backend::IBackend* renderBackend = getWaterRenderBackend(ctx)) {
                renderBackend->resetAccumulation();
            }
            if (g_viewport_backend && g_viewport_backend.get() != getWaterRenderBackend(ctx)) {
                g_viewport_backend->resetAccumulation();
            }
        }
        ImGui::SetItemTooltip("Realtime animates continuously. Timeline follows the current playback frame. Static freezes water at the current preview moment.");
        
        // Edit selected water
        WaterSurface* selected_surface = WaterManager::getInstance().getWaterSurface(selected_water_surface_id);
        if (!selected_surface && !waters.empty()) {
            selected_water_surface_id = waters.front().id;
            selected_surface = WaterManager::getInstance().getWaterSurface(selected_water_surface_id);
        }

        if (selected_surface) {
            WaterSurface& surf = *selected_surface;
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Edit: %s", surf.name.c_str());
            if (drawWaterSurfaceMaterialEditor(ctx, surf, true)) {
                selected_water_surface_id = -1;
                return;
            }
#if 0
            // Legacy CUDA FFT and animated mesh authoring UI. Kept temporarily
            // as implementation reference while the native Vulkan compute
            // spectrum is built; it is deliberately not exposed to artists.
            bool changed = false;
            bool geom_changed = false;     // wave shape params → mesh rebuild required
            bool fft_geom_changed = false; // FFT mesh params → FFT mesh rebuild required

            // === FFT OCEAN (TESSENDORF) ===
            if (BeginWaterSection("FFT Ocean (Film Quality)", ImVec4(1.0f, 0.8f, 0.0f, 1.0f))) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.9f, 0.5f, 1.0f));
                ImGui::Text("Tessendorf Algorithm");
                ImGui::PopStyleColor();
                ImGui::SetItemTooltip("Physically accurate ocean simulation used in major film productions");
                
                if (!g_hasCUDA) {
                    ImGui::BeginDisabled();
                    surf.params.use_fft_ocean = false;
                }
                changed |= ImGui::Checkbox("Enable FFT Ocean", &surf.params.use_fft_ocean);
                if (!g_hasCUDA) {
                    ImGui::EndDisabled();
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "  [Requires CUDA GPU]");
                }
                
                if (surf.params.use_fft_ocean) {
                    ImGui::Indent();
                    
                    // ═══════════════════════════════════════════════════════════════════════
                    // KEYFRAME HELPER LAMBDAS (Diamond Button System)
                    // ═══════════════════════════════════════════════════════════════════════
                    std::string water_track_name = "Water_" + std::to_string(surf.id);
                    int current_frame = ctx.render_settings.animation_playback_frame;
                    
                    // Draw diamond keyframe button (returns true if clicked)
                    auto KeyframeButton = [&](const char* id, bool keyed) -> bool {
                        ImGui::PushID(id);
                        float s = ImGui::GetFrameHeight();
                        ImVec2 pos = ImGui::GetCursorScreenPos();
                        bool clicked = ImGui::InvisibleButton("kbtn", ImVec2(s, s));

                        ImU32 bg = keyed ? IM_COL32(255, 200, 0, 255) : IM_COL32(40, 40, 40, 255);
                        ImU32 border = IM_COL32(180, 180, 180, 255);

                        if (ImGui::IsItemHovered()) {
                            border = IM_COL32(255, 255, 255, 255);
                            bg = keyed ? IM_COL32(255, 220, 50, 255) : IM_COL32(70, 70, 70, 255);
                        }

                        ImDrawList* dl = ImGui::GetWindowDrawList();
                        float cx = pos.x + s * 0.5f;
                        float cy = pos.y + s * 0.5f;
                        float r = s * 0.22f;

                        ImVec2 p[4] = {
                            ImVec2(cx, cy - r),
                            ImVec2(cx + r, cy),
                            ImVec2(cx, cy + r),
                            ImVec2(cx - r, cy)
                        };

                        dl->AddQuadFilled(p[0], p[1], p[2], p[3], bg);
                        dl->AddQuad(p[0], p[1], p[2], p[3], border, 1.0f);

                        ImGui::PopID();
                        return clicked;
                    };
                    
                    // Check if a specific FFT property is keyed at current frame
                    auto isFFTPropertyKeyed = [&](bool has_wind_speed, bool has_wind_dir, bool has_amplitude, 
                                                  bool has_choppiness, bool has_time_scale, bool has_ocean_size) -> bool {
                        auto& tracks = ctx.scene.timeline.tracks;
                        if (tracks.find(water_track_name) == tracks.end()) return false;
                        for (auto& kf : tracks[water_track_name].keyframes) {
                            if (kf.frame == current_frame && kf.has_water) {
                                if (has_wind_speed && kf.water.has_fft_wind_speed) return true;
                                if (has_wind_dir && kf.water.has_fft_wind_direction) return true;
                                if (has_amplitude && kf.water.has_fft_amplitude) return true;
                                if (has_choppiness && kf.water.has_fft_choppiness) return true;
                                if (has_time_scale && kf.water.has_fft_time_scale) return true;
                                if (has_ocean_size && kf.water.has_fft_ocean_size) return true;
                            }
                        }
                        return false;
                    };
                    
                    // Toggle keyframe for specific FFT properties
                    auto toggleFFTKeyframe = [&](bool key_wind_speed, bool key_wind_dir, bool key_amplitude,
                                                 bool key_choppiness, bool key_time_scale, bool key_ocean_size) {
                        auto& track = ctx.scene.timeline.tracks[water_track_name];
                        track.object_name = water_track_name;
                        
                        // Check if we need to remove existing keyframe (toggle off)
                        for (auto it = track.keyframes.begin(); it != track.keyframes.end(); ++it) {
                            if (it->frame == current_frame && it->has_water) {
                                bool removed = false;
                                if (key_wind_speed && it->water.has_fft_wind_speed) { it->water.has_fft_wind_speed = false; removed = true; }
                                if (key_wind_dir && it->water.has_fft_wind_direction) { it->water.has_fft_wind_direction = false; removed = true; }
                                if (key_amplitude && it->water.has_fft_amplitude) { it->water.has_fft_amplitude = false; removed = true; }
                                if (key_choppiness && it->water.has_fft_choppiness) { it->water.has_fft_choppiness = false; removed = true; }
                                if (key_time_scale && it->water.has_fft_time_scale) { it->water.has_fft_time_scale = false; removed = true; }
                                if (key_ocean_size && it->water.has_fft_ocean_size) { it->water.has_fft_ocean_size = false; removed = true; }
                                
                                if (removed) {
                                    // Check if any property still keyed
                                    bool hasAny = it->water.has_fft_wind_speed || it->water.has_fft_wind_direction ||
                                                  it->water.has_fft_amplitude || it->water.has_fft_choppiness ||
                                                  it->water.has_fft_time_scale || it->water.has_fft_ocean_size ||
                                                  it->water.has_wave_height; // Include geo waves too
                                    if (!hasAny) {
                                        it->has_water = false;
                                    }
                                    return;
                                }
                            }
                        }
                        
                        // Add new keyframe
                        Keyframe kf(current_frame);
                        kf.has_water = true;
                        kf.water.water_surface_id = surf.id;
                        
                        if (key_wind_speed) { kf.water.has_fft_wind_speed = true; kf.water.fft_wind_speed = surf.params.fft_wind_speed; }
                        if (key_wind_dir) { kf.water.has_fft_wind_direction = true; kf.water.fft_wind_direction = surf.params.fft_wind_direction * 180.0f / 3.14159f; }
                        if (key_amplitude) { kf.water.has_fft_amplitude = true; kf.water.fft_amplitude = surf.params.fft_amplitude; }
                        if (key_choppiness) { kf.water.has_fft_choppiness = true; kf.water.fft_choppiness = surf.params.fft_choppiness; }
                        if (key_time_scale) { kf.water.has_fft_time_scale = true; kf.water.fft_time_scale = surf.params.fft_time_scale; }
                        if (key_ocean_size) { kf.water.has_fft_ocean_size = true; kf.water.fft_ocean_size = surf.params.fft_ocean_size; }
                        
                        track.addKeyframe(kf);
                    };
                    
                    // Resolution dropdown (not animatable)
                    const char* res_items[] = { "64", "128", "256", "512" };
                    int res_index = 0;
                    if (surf.params.fft_resolution == 128) res_index = 1;
                    else if (surf.params.fft_resolution == 256) res_index = 2;
                    else if (surf.params.fft_resolution == 512) res_index = 3;
                    
                    if (ImGui::Combo("Resolution", &res_index, res_items, 4)) {
                        int resolutions[] = { 64, 128, 256, 512 };
                        surf.params.fft_resolution = resolutions[res_index];
                        changed = true;
                    }
                    ImGui::SetItemTooltip("FFT grid resolution - higher = more detail, slower");
                    
                    // Ocean size (animatable)
                    if (KeyframeButton("kf_fft_sz", isFFTPropertyKeyed(false, false, false, false, false, true))) {
                        toggleFFTKeyframe(false, false, false, false, false, true);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("fft_sz", "Ocean Size", &surf.params.fft_ocean_size, 10.0f, 10000.0f, "%.0f m", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("World space covered by one tile (tiles infinitely)");
                    ImGui::BeginDisabled(true);
                    bool auto_domain_disabled = false;
                    ImGui::Checkbox("Auto Domain From Mesh", &auto_domain_disabled);
                    ImGui::EndDisabled();
                    surf.params.auto_domain_from_mesh = false;
                    ImGui::SetItemTooltip("Disabled: explicit Ocean Size now keeps OptiX/Vulkan micro detail stable across mesh scale.");
                    surf.params.domain_size_multiplier = 1.0f;
                    ImGui::BeginDisabled(true);
                    float fixed_domain_multiplier = 1.0f;
                    SceneUI::DrawSmartFloat("dom_mul", "Domain Multiplier", &fixed_domain_multiplier, 1.0f, 1.0f, "%.2fx", false, nullptr, 16);
                    ImGui::EndDisabled();
                    ImGui::SetItemTooltip("Disabled: Ocean Size is the only water domain scale.");
                    ImGui::Text("Resolved Domain: %.2f m", WaterManager::getInstance().resolveWaveDomainSize(&surf));
                    
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Wind Settings (Animatable):");
                    
                    // Wind speed (animatable)
                    if (KeyframeButton("kf_fft_ws", isFFTPropertyKeyed(true, false, false, false, false, false))) {
                        toggleFFTKeyframe(true, false, false, false, false, false);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("fft_ws", "Wind Speed", &surf.params.fft_wind_speed, 0.0f, 200.0f, "%.1f m/s", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Higher = larger, more energetic waves. >30 is storm force. [Keyframeable]");
                    
                    // Wind direction (animatable)
                    if (KeyframeButton("kf_fft_wd", isFFTPropertyKeyed(false, true, false, false, false, false))) {
                        toggleFFTKeyframe(false, true, false, false, false, false);
                    }
                    ImGui::SameLine();
                    float wind_deg = surf.params.fft_wind_direction * 180.0f / 3.14159f;
                    if (SceneUI::DrawSmartFloat("fft_wd", "Wind Direction", &wind_deg, 0.0f, 360.0f, "%.0f deg", false, nullptr, 16)) {
                        surf.params.fft_wind_direction = wind_deg * 3.14159f / 180.0f;
                        changed = true;
                    }
                    ImGui::SetItemTooltip("Direction the waves travel [Keyframeable]");
                    
                    ImGui::Separator();
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Wave Appearance (Animatable):");
                    
                    // Choppiness (animatable)
                    if (KeyframeButton("kf_fft_ch", isFFTPropertyKeyed(false, false, false, true, false, false))) {
                        toggleFFTKeyframe(false, false, false, true, false, false);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("fft_ch", "Choppiness", &surf.params.fft_choppiness, 0.0f, 5.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Horizontal displacement - makes peaks sharper. [Keyframeable]");
                    
                    // Amplitude (animatable)
                    if (KeyframeButton("kf_fft_amp", isFFTPropertyKeyed(false, false, true, false, false, false))) {
                        toggleFFTKeyframe(false, false, true, false, false, false);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("fft_amp", "Amplitude Scale", &surf.params.fft_amplitude, 0.000001f, 0.1f, "%.6f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Overall wave height multiplier (Phillips A parameter). [Keyframeable]");
                    
                    // Time scale (animatable)
                    if (KeyframeButton("kf_fft_ts", isFFTPropertyKeyed(false, false, false, false, true, false))) {
                        toggleFFTKeyframe(false, false, false, false, true, false);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("fft_ts", "Animation Speed", &surf.params.fft_time_scale, 0.0f, 20.0f, "%.2f", false, nullptr, 16)) {
                        surf.params.geo_wave_speed = surf.params.fft_time_scale;
                        changed = true;
                    }
                    ImGui::SetItemTooltip("Master water animation speed for FFT and geometric waves. [Keyframeable]");
                    
                    // ═══════════════════════════════════════════════════════════════════════
                    // FFT MESH DISPLACEMENT (Best Quality - Physical Mesh from FFT Data)
                    // ═══════════════════════════════════════════════════════════════════════
                    ImGui::Separator();
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.8f, 1.0f));
                    ImGui::Text("FFT Mesh Displacement:");
                    ImGui::PopStyleColor();
                    ImGui::SetItemTooltip("Physically displace mesh vertices using FFT ocean data.\nCreates real 3D waves with proper shadows and reflections.");
                    
                    bool fft_mesh_changed = ImGui::Checkbox("Enable FFT Mesh Displacement", &surf.params.use_fft_mesh_displacement);
                    if (fft_mesh_changed) {
                        changed = true;
                        fft_geom_changed = true;
                        if (surf.params.use_fft_mesh_displacement) {
                            // Auto-enable mesh animation
                            surf.animate_mesh = true;
                            // Disable geometric waves to avoid conflict
                            surf.params.use_geometric_waves = false;
                            // Initialize with reasonable defaults if zero
                            if (surf.params.fft_mesh_height_scale < 0.1f) 
                                surf.params.fft_mesh_height_scale = 1.0f;
                        }
                    }
                    ImGui::SetItemTooltip("Apply FFT ocean heights to physical mesh.\nBest quality: combines FFT precision with raytraced geometry.");
                    
                    if (surf.params.use_fft_mesh_displacement) {
                        ImGui::Indent();
                        
                        // Height Scale - Amplifies FFT output (raw FFT values are small)
                        if (SceneUI::DrawSmartFloat("fft_mhs", "Height Scale", &surf.params.fft_mesh_height_scale, 1.0f, 200.0f, "%.1f", false, nullptr, 16)) fft_geom_changed = true;
                        ImGui::SetItemTooltip("Amplifies FFT wave height.\nTypical: 20-50 calm, 50-100 stormy.\n(FFT produces normalized values that need scaling)");

                        // Choppiness - Horizontal displacement
                        if (SceneUI::DrawSmartFloat("fft_mch", "Choppiness", &surf.params.fft_mesh_choppiness, 0.0f, 5.0f, "%.2f", false, nullptr, 16)) fft_geom_changed = true;
                        ImGui::SetItemTooltip("Horizontal displacement for sharper wave peaks.\n0 = smooth swells, 1-2 = realistic ocean, 2+ = very choppy.");
                        
                        ImGui::Unindent();
                        
                        // Status indicator
                        if (surf.fft_state) {
                            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1.0f), "  FFT Mesh: Active");
                        } else {
                            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "  Waiting for FFT...");
                        }
                    }
                    
                    ImGui::Unindent();
                } else {
                    ImGui::TextDisabled("Enable to use film-quality FFT ocean");
                }
                EndWaterSection();
            }
            
            // === GEOMETRIC WAVES (Physical Mesh Displacement) ===
            // (Physics + Surface Detail panels intentionally removed here —
            //  drawWaterSurfaceMaterialEditor above already exposes ior/roughness/
            //  clarity (Depth & Optics), foam_noise_scale/foam_threshold (Foam),
            //  and micro_detail_* (Surface Detail). Duplicating them here let
            //  the same `surf.params.*` field be edited from two places, which
            //  was confusing and triggered redundant syncWaterMaterialPreview.)
            // NOTE: This is a legacy/alternative to FFT Mesh Displacement
            // When FFT Mesh Displacement is enabled, this section is disabled
            if (surf.params.use_fft_mesh_displacement) {
                if (BeginWaterSection("Geometric Waves (Disabled)", ImVec4(0.4f, 0.4f, 0.4f, 1.0f), false)) {
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), 
                        "FFT Mesh Displacement is active - provides better quality.");
                    ImGui::TextDisabled("Disable FFT Mesh Displacement above to use this.");
                    EndWaterSection();
                }
            } else if (BeginWaterSection("Geometric Waves (CPU Mesh)", ImVec4(0.6f, 0.9f, 0.4f, 1.0f))) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.6f, 1.0f, 0.6f, 1.0f));
                ImGui::Text("Physical Mesh Displacement");
                ImGui::PopStyleColor();
                ImGui::SetItemTooltip("Deforms the water mesh geometry for true 3D waves (CPU-based, affects shadows/reflections).\nFor best quality, use FFT Ocean + FFT Mesh Displacement instead.");
                
                // Show recommendation if FFT is enabled but FFT mesh displacement is not
                if (surf.params.use_fft_ocean) {
                    ImGui::TextColored(ImVec4(1.0f, 0.9f, 0.4f, 1.0f), 
                        "Tip: Enable 'FFT Mesh Displacement' in FFT Ocean for film-quality mesh waves.");
                }
                
                bool geo_changed = ImGui::Checkbox("Enable Geometric Waves", &surf.params.use_geometric_waves);
                if (geo_changed) {
                    changed = true;
                    geom_changed = true;
                    if (surf.params.use_geometric_waves) {
                        surf.animate_mesh = true;
                    }
                }
                
                if (surf.params.use_geometric_waves) {
                    ImGui::Indent();
                    
                    // ═══════════════════════════════════════════════════════════════════════
                    // KEYFRAME HELPER LAMBDAS FOR GEOMETRIC WAVES
                    // ═══════════════════════════════════════════════════════════════════════
                    std::string geo_track_name = "Water_" + std::to_string(surf.id);
                    int geo_current_frame = ctx.render_settings.animation_playback_frame;
                    
                    // Draw diamond keyframe button
                    auto GeoKeyframeButton = [&](const char* id, bool keyed) -> bool {
                        ImGui::PushID(id);
                        float s = ImGui::GetFrameHeight();
                        ImVec2 pos = ImGui::GetCursorScreenPos();
                        bool clicked = ImGui::InvisibleButton("kbtn", ImVec2(s, s));

                        ImU32 bg = keyed ? IM_COL32(100, 255, 100, 255) : IM_COL32(40, 40, 40, 255);
                        ImU32 border = IM_COL32(180, 180, 180, 255);

                        if (ImGui::IsItemHovered()) {
                            border = IM_COL32(255, 255, 255, 255);
                            bg = keyed ? IM_COL32(150, 255, 150, 255) : IM_COL32(70, 70, 70, 255);
                        }

                        ImDrawList* dl = ImGui::GetWindowDrawList();
                        float cx = pos.x + s * 0.5f;
                        float cy = pos.y + s * 0.5f;
                        float r = s * 0.22f;

                        ImVec2 p[4] = {
                            ImVec2(cx, cy - r),
                            ImVec2(cx + r, cy),
                            ImVec2(cx, cy + r),
                            ImVec2(cx - r, cy)
                        };

                        dl->AddQuadFilled(p[0], p[1], p[2], p[3], bg);
                        dl->AddQuad(p[0], p[1], p[2], p[3], border, 1.0f);

                        ImGui::PopID();
                        return clicked;
                    };
                    
                    // Check if geo property is keyed at current frame
                    auto isGeoPropertyKeyed = [&](bool has_wave_height, bool has_wave_scale, bool has_choppiness, bool has_geo_speed) -> bool {
                        auto& tracks = ctx.scene.timeline.tracks;
                        if (tracks.find(geo_track_name) == tracks.end()) return false;
                        for (auto& kf : tracks[geo_track_name].keyframes) {
                            if (kf.frame == geo_current_frame && kf.has_water) {
                                if (has_wave_height && kf.water.has_wave_height) return true;
                                if (has_wave_scale && kf.water.has_wave_scale) return true;
                                if (has_choppiness && kf.water.has_choppiness) return true;
                                if (has_geo_speed && kf.water.has_geo_speed) return true;
                            }
                        }
                        return false;
                    };
                    
                    // Toggle geo keyframe
                    auto toggleGeoKeyframe = [&](bool key_wave_height, bool key_wave_scale, bool key_choppiness, bool key_geo_speed) {
                        auto& track = ctx.scene.timeline.tracks[geo_track_name];
                        track.object_name = geo_track_name;
                        
                        // Check if we need to remove existing keyframe (toggle off)
                        for (auto it = track.keyframes.begin(); it != track.keyframes.end(); ++it) {
                            if (it->frame == geo_current_frame && it->has_water) {
                                bool removed = false;
                                if (key_wave_height && it->water.has_wave_height) { it->water.has_wave_height = false; removed = true; }
                                if (key_wave_scale && it->water.has_wave_scale) { it->water.has_wave_scale = false; removed = true; }
                                if (key_choppiness && it->water.has_choppiness) { it->water.has_choppiness = false; removed = true; }
                                if (key_geo_speed && it->water.has_geo_speed) { it->water.has_geo_speed = false; removed = true; }
                                
                                if (removed) return;
                            }
                        }
                        
                        // Add new keyframe
                        Keyframe kf(geo_current_frame);
                        kf.has_water = true;
                        kf.water.water_surface_id = surf.id;
                        
                        if (key_wave_height) { kf.water.has_wave_height = true; kf.water.wave_height = surf.params.geo_wave_height; }
                        if (key_wave_scale) { kf.water.has_wave_scale = true; kf.water.wave_scale = surf.params.geo_wave_scale; }
                        if (key_choppiness) { kf.water.has_choppiness = true; kf.water.choppiness = surf.params.geo_wave_choppiness; }
                        if (key_geo_speed) { kf.water.has_geo_speed = true; kf.water.geo_speed = surf.params.geo_wave_speed; }
                        
                        track.addKeyframe(kf);
                    };
                    
                    // Noise Type Dropdown
                    const char* noise_items[] = { "Perlin", "FBM", "Ridge", "Voronoi", "Billow", "Gerstner", "Tessendorf Simple" };
                    int noise_idx = static_cast<int>(surf.params.geo_noise_type);
                    if (ImGui::Combo("Noise Type", &noise_idx, noise_items, IM_ARRAYSIZE(noise_items))) {
                        surf.params.geo_noise_type = static_cast<WaterWaveParams::NoiseType>(noise_idx);
                        geom_changed = true;
                    }
                    ImGui::SetItemTooltip("Algorithm used for wave generation");
                    
                    ImGui::Separator();
                    ImGui::TextDisabled("Wave Shape (Animatable)");
                    
                    // Wave Height (animatable)
                    if (GeoKeyframeButton("kf_geo_wh", isGeoPropertyKeyed(true, false, false, false))) {
                        toggleGeoKeyframe(true, false, false, false);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("geo_wh", "Wave Height", &surf.params.geo_wave_height, 0.0f, 20.0f, "%.2f m", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("Maximum vertical displacement amplitude");

                    // Wave Scale (animatable)
                    if (GeoKeyframeButton("kf_geo_ws", isGeoPropertyKeyed(false, true, false, false))) {
                        toggleGeoKeyframe(false, true, false, false);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("geo_ws", "Wave Scale", &surf.params.geo_wave_scale, 0.5f, 500.0f, "%.2f", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("Wavelength in world units.\nShould be smaller than mesh size for visible waves.\nTypical: 2-10 for a 20-unit plane, 20-50 for large ocean tiles.");

                    // Choppiness (animatable)
                    if (GeoKeyframeButton("kf_geo_wc", isGeoPropertyKeyed(false, false, true, false))) {
                        toggleGeoKeyframe(false, false, true, false);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("geo_wc", "Choppiness", &surf.params.geo_wave_choppiness, 0.0f, 3.0f, "%.2f", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("Sharpness of wave peaks (Ridge offset)");

                    // Animation Speed — only affects playback rate, no mesh/accumulation reset
                    if (GeoKeyframeButton("kf_geo_sp", isGeoPropertyKeyed(false, false, false, true))) {
                        toggleGeoKeyframe(false, false, false, true);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("geo_sp", "Animation Speed", &surf.params.geo_wave_speed, 0.0f, 5.0f, "%.2f", false, nullptr, 16)) {
                        surf.params.fft_time_scale = surf.params.geo_wave_speed;
                        changed = true;
                    }
                    ImGui::SetItemTooltip("Legacy alias of the master water animation speed. Keeps geometric waves in sync with FFT/timeline playback.");

                    ImGui::Separator();
                    ImGui::TextDisabled("Noise Detail (Fractal)");

                    if (ImGui::SliderInt("Octaves", &surf.params.geo_octaves, 1, 8)) geom_changed = true;
                    ImGui::SetItemTooltip("Number of noise layers (more = finer detail)");

                    if (SceneUI::DrawSmartFloat("geo_ps", "Persistence", &surf.params.geo_persistence, 0.1f, 1.0f, "%.2f", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("How much each octave contributes (amplitude decay)");

                    if (SceneUI::DrawSmartFloat("geo_lc", "Lacunarity", &surf.params.geo_lacunarity, 1.0f, 4.0f, "%.2f", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("Frequency multiplier between octaves");

                    if (SceneUI::DrawSmartFloat("geo_ro", "Ridge Offset", &surf.params.geo_ridge_offset, 0.0f, 2.0f, "%.2f", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("Offset for ridge noise (affects wave peak sharpness)");

                    ImGui::Separator();
                    ImGui::TextDisabled("Ocean Simulation");

                    // geo_depth is currently unused in wave computation — no rebuild needed
                    SceneUI::DrawSmartFloat("geo_dp", "Ocean Depth", &surf.params.geo_depth, 1.0f, 1000.0f, "%.0f m", false, nullptr, 16);
                    ImGui::SetItemTooltip("Ocean depth - affects shallow water wave behavior");

                    if (SceneUI::DrawSmartFloat("geo_dm", "Damping", &surf.params.geo_damping, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("Damping for wind-perpendicular waves");

                    if (SceneUI::DrawSmartFloat("geo_al", "Alignment", &surf.params.geo_alignment, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("Wave alignment to wind direction (0=omni, 1=aligned)");

                    float swell_deg = surf.params.geo_swell_direction;
                    if (SceneUI::DrawSmartFloat("geo_sd", "Swell Direction", &swell_deg, 0.0f, 360.0f, "%.0f deg", false, nullptr, 16)) {
                        surf.params.geo_swell_direction = swell_deg;
                        geom_changed = true;
                    }
                    ImGui::SetItemTooltip("Direction offset for long-distance swell waves");

                    if (SceneUI::DrawSmartFloat("geo_sa", "Swell Amplitude", &surf.params.geo_swell_amplitude, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("Contribution of swell (long-distance) waves");

                    if (SceneUI::DrawSmartFloat("geo_sh", "Sharpening", &surf.params.geo_sharpening, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("Post-process peak sharpening (0=smooth, 1=peaked)");

                    ImGui::Separator();
                    ImGui::TextDisabled("Secondary Detail");

                    if (SceneUI::DrawSmartFloat("geo_ds", "Detail Scale", &surf.params.geo_detail_scale, 1.0f, 10.0f, "%.1f", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("Scale multiplier for secondary detail noise");

                    if (SceneUI::DrawSmartFloat("geo_dt", "Detail Strength", &surf.params.geo_detail_strength, 0.0f, 0.5f, "%.3f", false, nullptr, 16)) geom_changed = true;
                    ImGui::SetItemTooltip("Strength of secondary small-scale detail");

                    if (ImGui::Checkbox("Smooth Normals", &surf.params.geo_smooth_normals)) geom_changed = true;
                    ImGui::SetItemTooltip("Average vertex normals for smooth shading");

                    ImGui::Separator();
                    ImGui::TextDisabled("Animation");

                    // Animate Mesh toggle — no immediate mesh/accumulation change needed
                    if (ImGui::Checkbox("Animate Mesh", &surf.animate_mesh)) {
                        geom_changed = true;
                    }
                    ImGui::SetItemTooltip("Enable real-time mesh animation for geometric waves");
                    
                    if (surf.animate_mesh) {
                        // GPU/CPU toggle
                        if (!g_hasCUDA) {
                            ImGui::BeginDisabled();
                            surf.use_gpu_animation = false;
                        }
                        
                        if (ImGui::Checkbox("Use GPU Acceleration", &surf.use_gpu_animation)) {
                            geom_changed = true;
                        }
                        
                        if (!g_hasCUDA) {
                            ImGui::EndDisabled();
                            ImGui::SameLine();
                            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "(No CUDA)");
                        }
                        
                        if (surf.use_gpu_animation && g_hasCUDA) {
                            ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "GPU Mode: Fast");
                        } else {
                            ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.0f, 1.0f), "CPU Mode: Slower for large meshes");
                        }
                    }
                    
                    ImGui::Unindent();
                    
                    // Rebuild mesh ONLY when wave shape parameters actually changed
                    if (geom_changed) {
                        WaterManager::getInstance().invalidateGeometricAnimationState(&surf);
                        WaterManager::getInstance().updateWaterMesh(&surf);
                        WaterManager::getInstance().cacheOriginalPositions(&surf);

                        rebuildWaterSceneMutation(ctx, false);
                        syncWaterMaterialPreview(ctx, surf);
                    }
                } else {
                    ImGui::TextDisabled("Enable to use physical mesh displacement");
                }
                EndWaterSection();
            }
            
            // === APPLY CHANGES ===
            if (changed) {
                // Update material with new parameters
                auto mat = MaterialManager::getInstance().getMaterialShared(surf.material_id);
                if (mat) {
                    WaterManager::getInstance().syncSurfaceMaterial(&surf);
                }
                
                // FFT Mesh Displacement - Only rebuild BVH when FFT mesh geometry params changed
                if (fft_geom_changed && surf.params.use_fft_mesh_displacement && surf.params.use_fft_ocean) {
                    WaterManager::getInstance().updateFFTDrivenMesh(&surf, surf.animation_time);
                    rebuildWaterSceneMutation(ctx, false);
                }
                
                // Mark preset as Custom when any parameter is manually changed
                surf.params.current_preset = WaterWaveParams::WaterPreset::Custom;
                
                // Reset accumulation and sync GPU materials for real-time preview
                syncWaterMaterialPreview(ctx, surf);
            }
#endif
        }
    }
}

#endif
