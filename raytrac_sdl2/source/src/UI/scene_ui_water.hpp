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

void SceneUI::drawWaterPanel(UIContext& ctx) {
    ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "WATER SYSTEM");
    ImGui::Separator();
    
    // Create new water
    if (ImGui::Button("Add Water Plane", ImVec2(UIWidgets::GetInspectorActionWidth(), 30))) {
        // Default spawn at origin or focus point
        Vec3 spawn_pos = ctx.scene.camera ? 
            (ctx.scene.camera->lookfrom + (ctx.scene.camera->w * -10.0f)) : Vec3(0,0,0);
        
        spawn_pos.y = 0; // Keep flat
        
        WaterManager::getInstance().createWaterPlane(ctx.scene, spawn_pos, 20.0f, 4.0f); // Size 20, density 4
        
        // Rebuild BVH after adding geometry
        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
        ctx.renderer.resetCPUAccumulation();
        if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
            ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
        }
        SCENE_LOG_INFO("[Water] Created new water plane");
    }
    
    ImGui::Spacing();
    ImGui::Separator();
    
    // List existing water surfaces
    auto& waters = WaterManager::getInstance().getWaterSurfaces();
    
    if (waters.empty()) {
        ImGui::TextDisabled("No water surfaces in scene.");
    } else {
        static int selected_water_idx = -1;
        
        ImGui::Text("Active Water Surfaces (%zu):", waters.size());
        
        if (ImGui::BeginListBox("##waterlist", ImVec2(-1, 80))) {
            for (int i = 0; i < waters.size(); i++) {
                // Show all water types, including rivers (since River panel delegates material editing here)
                
                bool is_selected = (selected_water_idx == i);
                if (ImGui::Selectable(waters[i].name.c_str(), is_selected)) {
                    selected_water_idx = i;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndListBox();
        }
        
        // Edit selected water
        if (selected_water_idx >= 0 && selected_water_idx < waters.size()) {
            WaterSurface& surf = waters[selected_water_idx];
            bool changed = false;
            
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.3f, 0.8f, 1.0f, 1.0f), "Edit: %s", surf.name.c_str());
            
            // ═══════════════════════════════════════════════════════════════════════════
            // WATER PRESETS DROPDOWN
            // ═══════════════════════════════════════════════════════════════════════════
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.95f, 0.4f, 1.0f));
            ImGui::Text("Quick Presets");
            ImGui::PopStyleColor();
            
            const char* preset_names[] = {
                "Custom (Manual Settings)",
                "Calm Ocean",
                "Stormy Ocean",
                "Tropical Ocean (Crystal)",
                "Lake (Still)",
                "River (Flowing)",
                "Pool (Very Calm)",
                "Pond (Murky)"
            };
            
            int preset_idx = static_cast<int>(surf.params.current_preset);
            ImGui::SetNextItemWidth(-1);
            if (ImGui::Combo("##waterpreset", &preset_idx, preset_names, IM_ARRAYSIZE(preset_names))) {
                WaterWaveParams::WaterPreset new_preset = static_cast<WaterWaveParams::WaterPreset>(preset_idx);
                if (new_preset != WaterWaveParams::WaterPreset::Custom) {
                    surf.params.applyPreset(new_preset);
                    changed = true;
                    SCENE_LOG_INFO("[Water] Applied preset: %s", preset_names[preset_idx]);
                } else {
                    surf.params.current_preset = WaterWaveParams::WaterPreset::Custom;
                }
            }
            ImGui::SetItemTooltip("Quick setup for common water types.\nSelect a preset to auto-configure all parameters.");
            
            ImGui::Spacing();
            ImGui::Separator();
            
            // === SIMPLE WAVES (Legacy - Only when FFT is disabled) ===
            // This section is hidden when FFT Ocean is enabled because FFT provides
            // film-quality waves that replace these simple procedural waves.
            if (!surf.params.use_fft_ocean) {
                if (UIWidgets::BeginSection("Simple Waves (Fallback)", ImVec4(0.5f, 0.5f, 0.5f, 1.0f), false)) {
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), "Legacy Mode - Use FFT Ocean for better quality");
                    if (SceneUI::DrawSmartFloat("w_wspd", "Speed", &surf.params.wave_speed, 0.0f, 10.0f, "%.2f", false, nullptr, 16)) changed = true;
                    if (SceneUI::DrawSmartFloat("w_whgt", "Height", &surf.params.wave_strength, 0.0f, 3.0f, "%.2f", false, nullptr, 16)) changed = true;
                    if (SceneUI::DrawSmartFloat("w_wfrq", "Frequency", &surf.params.wave_frequency, 0.1f, 5.0f, "%.2f", false, nullptr, 16)) changed = true;
                    UIWidgets::EndSection();
                }
            }
            
            // === COLORS ===
            if (UIWidgets::BeginSection("Colors", ImVec4(0.0f, 0.8f, 0.8f, 1.0f))) {
                changed |= ImGui::ColorEdit3("Shallow Color", &surf.params.shallow_color.x);
                changed |= ImGui::ColorEdit3("Deep Color", &surf.params.deep_color.x);
                UIWidgets::EndSection();
            }
            
            // === DEPTH & ABSORPTION ===
            if (UIWidgets::BeginSection("Depth & Absorption", ImVec4(0.0f, 0.2f, 0.6f, 1.0f), false)) {
                if (SceneUI::DrawSmartFloat("w_dmax", "Max Depth", &surf.params.depth_max, 1.0f, 100.0f, "%.1f m", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("Distance at which water reaches full deep color");
                
                changed |= ImGui::ColorEdit3("Absorption Tint", &surf.params.absorption_color.x);
                ImGui::SetItemTooltip("What colors are absorbed by the water");
                
                if (SceneUI::DrawSmartFloat("w_absd", "Absorption Density", &surf.params.absorption_density, 0.0f, 2.0f, "%.2f", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("How quickly light is absorbed (murkiness)");
                UIWidgets::EndSection();
            }
            
            // === FOAM ===
            if (UIWidgets::BeginSection("Foam", ImVec4(0.9f, 0.9f, 0.9f, 1.0f), false)) {
                if (SceneUI::DrawSmartFloat("w_wf", "Wave Foam", &surf.params.foam_level, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("Foam on wave crests");
                
                ImGui::Separator();
                ImGui::TextDisabled("Shore Foam");
                if (SceneUI::DrawSmartFloat("w_fi", "Intensity", &surf.params.shore_foam_intensity, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) changed = true;
                if (SceneUI::DrawSmartFloat("w_fd", "Distance", &surf.params.shore_foam_distance, 0.1f, 10.0f, "%.1f m", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("How far from shore foam appears");
                UIWidgets::EndSection();
            }
            
            // === CAUSTICS ===
            if (UIWidgets::BeginSection("Caustics", ImVec4(0.4f, 1.0f, 1.0f, 1.0f), false)) {
                if (SceneUI::DrawSmartFloat("w_ci", "Intensity", &surf.params.caustic_intensity, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) changed = true;
                if (SceneUI::DrawSmartFloat("w_cs", "Scale", &surf.params.caustic_scale, 0.1f, 10.0f, "%.2f", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("Size of caustic patterns");
                if (SceneUI::DrawSmartFloat("w_csp", "Speed", &surf.params.caustic_speed, 0.1f, 5.0f, "%.2f", false, nullptr, 16)) changed = true;
                UIWidgets::EndSection();
            }
            
            // === SSS ===
            if (UIWidgets::BeginSection("Scattering (SSS)", ImVec4(1.0f, 0.8f, 0.6f, 1.0f), false)) {
                if (SceneUI::DrawSmartFloat("w_sssi", "SSS Intensity", &surf.params.sss_intensity, 0.0f, 0.5f, "%.3f", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("Sub-surface light scattering at edges");
                changed |= ImGui::ColorEdit3("SSS Color", &surf.params.sss_color.x);
                UIWidgets::EndSection();
            }
            
            // === FFT OCEAN (TESSENDORF) ===
            if (UIWidgets::BeginSection("FFT Ocean (Film Quality)", ImVec4(1.0f, 0.8f, 0.0f, 1.0f))) {
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
                    if (SceneUI::DrawSmartFloat("fft_ts", "Animation Speed", &surf.params.fft_time_scale, 0.0f, 20.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Speed of wave animation [Keyframeable]");
                    
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
                        bool height_changed = SceneUI::DrawSmartFloat("fft_mhs", "Height Scale", &surf.params.fft_mesh_height_scale, 1.0f, 200.0f, "%.1f", false, nullptr, 16);
                        if (height_changed) changed = true;
                        ImGui::SetItemTooltip("Amplifies FFT wave height.\nTypical: 20-50 calm, 50-100 stormy.\n(FFT produces normalized values that need scaling)");
                        
                        // Choppiness - Horizontal displacement 
                        bool chop_changed = SceneUI::DrawSmartFloat("fft_mch", "Choppiness", &surf.params.fft_mesh_choppiness, 0.0f, 5.0f, "%.2f", false, nullptr, 16);
                        if (chop_changed) changed = true;
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
                UIWidgets::EndSection();
            }
            
            // === PHYSICS ===
            if (UIWidgets::BeginSection("Physics", ImVec4(0.2f, 1.0f, 0.2f, 1.0f), false)) {
                if (SceneUI::DrawSmartFloat("w_ior", "IOR", &surf.params.ior, 1.0f, 2.0f, "%.3f", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("Index of Refraction (Water = 1.333)");
                if (SceneUI::DrawSmartFloat("w_rgh", "Roughness", &surf.params.roughness, 0.0f, 0.2f, "%.3f", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("Surface micro-roughness");
                if (SceneUI::DrawSmartFloat("w_clr", "Clarity", &surf.params.clarity, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) changed = true;
                UIWidgets::EndSection();
            }
            
            // === SURFACE DETAIL ===
            if (UIWidgets::BeginSection("Surface Detail (Realism)", ImVec4(1.0f, 0.6f, 0.0f, 1.0f))) {
                if (SceneUI::DrawSmartFloat("w_mds", "Micro Detail Strength", &surf.params.micro_detail_strength, 0.0f, 0.2f, "%.3f", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("Adds high-frequency noise/ripples to break up the smooth surface");
                if (SceneUI::DrawSmartFloat("w_msc", "Micro Detail Scale", &surf.params.micro_detail_scale, 1.0f, 100.0f, "%.1f", false, nullptr, 16)) changed = true;
                
                ImGui::Separator();
                ImGui::TextDisabled("Animation Speed");
                if (SceneUI::DrawSmartFloat("w_mas", "Animation Speed", &surf.params.micro_anim_speed, 0.01f, 1.0f, "%.3f", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("How fast the micro ripples move (lower = calmer water)");
                if (SceneUI::DrawSmartFloat("w_mms", "Morph Speed", &surf.params.micro_morph_speed, 0.1f, 5.0f, "%.2f", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("How fast the ripple shapes change (lower = more stable patterns)");
                
                ImGui::Separator();
                ImGui::TextDisabled("Foam Tuning");
                if (SceneUI::DrawSmartFloat("w_fns", "Foam Noise Scale", &surf.params.foam_noise_scale, 1.0f, 50.0f, "%.1f", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("Scale of noise used to break up foam");
                if (SceneUI::DrawSmartFloat("w_fth", "Foam Threshold", &surf.params.foam_threshold, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) changed = true;
                UIWidgets::EndSection();
            }
            
            // === GEOMETRIC WAVES (Physical Mesh Displacement) ===
            // NOTE: This is a legacy/alternative to FFT Mesh Displacement
            // When FFT Mesh Displacement is enabled, this section is disabled
            if (surf.params.use_fft_mesh_displacement) {
                if (UIWidgets::BeginSection("Geometric Waves (Disabled)", ImVec4(0.4f, 0.4f, 0.4f, 1.0f), false)) {
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.4f, 1.0f), 
                        "FFT Mesh Displacement is active - provides better quality.");
                    ImGui::TextDisabled("Disable FFT Mesh Displacement above to use this.");
                    UIWidgets::EndSection();
                }
            } else if (UIWidgets::BeginSection("Geometric Waves (CPU Mesh)", ImVec4(0.6f, 0.9f, 0.4f, 1.0f))) {
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
                if (geo_changed) changed = true;
                
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
                        changed = true;
                    }
                    ImGui::SetItemTooltip("Algorithm used for wave generation");
                    
                    ImGui::Separator();
                    ImGui::TextDisabled("Wave Shape (Animatable)");
                    
                    // Wave Height (animatable)
                    if (GeoKeyframeButton("kf_geo_wh", isGeoPropertyKeyed(true, false, false, false))) {
                        toggleGeoKeyframe(true, false, false, false);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("geo_wh", "Wave Height", &surf.params.geo_wave_height, 0.0f, 20.0f, "%.2f m", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Maximum vertical displacement amplitude");
                    
                    // Wave Scale (animatable)
                    if (GeoKeyframeButton("kf_geo_ws", isGeoPropertyKeyed(false, true, false, false))) {
                        toggleGeoKeyframe(false, true, false, false);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("geo_ws", "Wave Scale", &surf.params.geo_wave_scale, 1.0f, 500.0f, "%.1f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Global scale of wave patterns (larger = broader waves)");
                    
                    // Choppiness (animatable)
                    if (GeoKeyframeButton("kf_geo_wc", isGeoPropertyKeyed(false, false, true, false))) {
                        toggleGeoKeyframe(false, false, true, false);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("geo_wc", "Choppiness", &surf.params.geo_wave_choppiness, 0.0f, 3.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Sharpness of wave peaks (Ridge offset)");
                    
                    // Animation Speed (animatable)
                    if (GeoKeyframeButton("kf_geo_sp", isGeoPropertyKeyed(false, false, false, true))) {
                        toggleGeoKeyframe(false, false, false, true);
                    }
                    ImGui::SameLine();
                    if (SceneUI::DrawSmartFloat("geo_sp", "Animation Speed", &surf.params.geo_wave_speed, 0.0f, 5.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Speed of wave animation (phase shift rate)");
                    
                    ImGui::Separator();
                    ImGui::TextDisabled("Noise Detail (Fractal)");
                    
                    if (ImGui::SliderInt("Octaves", &surf.params.geo_octaves, 1, 8)) changed = true;
                    ImGui::SetItemTooltip("Number of noise layers (more = finer detail)");
                    
                    if (SceneUI::DrawSmartFloat("geo_ps", "Persistence", &surf.params.geo_persistence, 0.1f, 1.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("How much each octave contributes (amplitude decay)");
                    
                    if (SceneUI::DrawSmartFloat("geo_lc", "Lacunarity", &surf.params.geo_lacunarity, 1.0f, 4.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Frequency multiplier between octaves");
                    
                    if (SceneUI::DrawSmartFloat("geo_ro", "Ridge Offset", &surf.params.geo_ridge_offset, 0.0f, 2.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Offset for ridge noise (affects wave peak sharpness)");
                    
                    ImGui::Separator();
                    ImGui::TextDisabled("Ocean Simulation");
                    
                    if (SceneUI::DrawSmartFloat("geo_dp", "Ocean Depth", &surf.params.geo_depth, 1.0f, 1000.0f, "%.0f m", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Ocean depth - affects shallow water wave behavior");
                    
                    if (SceneUI::DrawSmartFloat("geo_dm", "Damping", &surf.params.geo_damping, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Damping for wind-perpendicular waves");
                    
                    if (SceneUI::DrawSmartFloat("geo_al", "Alignment", &surf.params.geo_alignment, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Wave alignment to wind direction (0=omni, 1=aligned)");
                    
                    float swell_deg = surf.params.geo_swell_direction;
                    if (SceneUI::DrawSmartFloat("geo_sd", "Swell Direction", &swell_deg, 0.0f, 360.0f, "%.0f deg", false, nullptr, 16)) {
                        surf.params.geo_swell_direction = swell_deg;
                        changed = true;
                    }
                    ImGui::SetItemTooltip("Direction offset for long-distance swell waves");
                    
                    if (SceneUI::DrawSmartFloat("geo_sa", "Swell Amplitude", &surf.params.geo_swell_amplitude, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Contribution of swell (long-distance) waves");
                    
                    if (SceneUI::DrawSmartFloat("geo_sh", "Sharpening", &surf.params.geo_sharpening, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Post-process peak sharpening (0=smooth, 1=peaked)");
                    
                    ImGui::Separator();
                    ImGui::TextDisabled("Secondary Detail");
                    
                    if (SceneUI::DrawSmartFloat("geo_ds", "Detail Scale", &surf.params.geo_detail_scale, 1.0f, 10.0f, "%.1f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Scale multiplier for secondary detail noise");
                    
                    if (SceneUI::DrawSmartFloat("geo_dt", "Detail Strength", &surf.params.geo_detail_strength, 0.0f, 0.5f, "%.3f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Strength of secondary small-scale detail");
                    
                    changed |= ImGui::Checkbox("Smooth Normals", &surf.params.geo_smooth_normals);
                    ImGui::SetItemTooltip("Average vertex normals for smooth shading");
                    
                    ImGui::Separator();
                    ImGui::TextDisabled("Animation");
                    
                    changed |= ImGui::Checkbox("Animate Mesh", &surf.animate_mesh);
                    ImGui::SetItemTooltip("Enable real-time mesh animation for geometric waves");
                    
                    if (surf.animate_mesh) {
                        // GPU/CPU toggle
                        if (!g_hasCUDA) {
                            ImGui::BeginDisabled();
                            surf.use_gpu_animation = false;
                        }
                        
                        if (ImGui::Checkbox("Use GPU Acceleration", &surf.use_gpu_animation)) {
                            changed = true;
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
                    
                    // If geometric waves enabled/changed, rebuild mesh
                    if (geo_changed || changed) {
                        WaterManager::getInstance().updateWaterMesh(&surf);
                        WaterManager::getInstance().cacheOriginalPositions(&surf);
                        
                        // Rebuild BVH after mesh change
                        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                        ctx.renderer.resetCPUAccumulation();
                        if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                            ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                        }
                    }
                } else {
                    ImGui::TextDisabled("Enable to use physical mesh displacement");
                }
                UIWidgets::EndSection();
            }
            
            // === APPLY CHANGES ===
            if (changed) {
                // Update material with new parameters
                auto mat = MaterialManager::getInstance().getMaterialShared(surf.material_id);
                if (mat) {
                    auto pbsdf = std::dynamic_pointer_cast<PrincipledBSDF>(mat);
                    if (pbsdf) {
                        // Sync CPU Fields - disable simple wave params when FFT is active
                        pbsdf->anisotropic = surf.params.use_fft_ocean ? 0.0f : surf.params.wave_speed;
                        pbsdf->sheen = fmaxf(0.001f, surf.params.wave_strength);  // IS_WATER flag always needed
                        pbsdf->sheen_tint = surf.params.use_fft_ocean ? 0.0f : surf.params.wave_frequency;
                        pbsdf->transmission = surf.params.ior > 1.01f ? 1.0f : 0.0f; // Enable transmission for water
                        pbsdf->translucent = surf.params.foam_level;
                        pbsdf->clearcoat = surf.params.shore_foam_intensity;
                        pbsdf->clearcoatRoughness = surf.params.caustic_intensity;
                        pbsdf->subsurface = surf.params.depth_max / 100.0f;
                        pbsdf->subsurfaceScale = surf.params.absorption_density;
                        pbsdf->subsurfaceColor = surf.params.absorption_color;
                        pbsdf->roughness = surf.params.roughness;
                        pbsdf->ior = surf.params.ior;
                        // Synchronize deep_color and shallow_color with PrincipledBSDF properties
                        pbsdf->albedoProperty.color = surf.params.deep_color;
                        pbsdf->emissionProperty.color = surf.params.shallow_color;
                        pbsdf->emissionProperty.intensity = 1.0f;
                    }

                    if (mat->gpuMaterial) {
                        auto& gpu = mat->gpuMaterial;
                        
                        // Wave params - disable simple waves when FFT is active
                        gpu->anisotropic = surf.params.use_fft_ocean ? 0.0f : surf.params.wave_speed;
                        gpu->sheen = fmaxf(0.001f, surf.params.wave_strength);  // IS_WATER flag always needed
                        gpu->sheen_tint = surf.params.use_fft_ocean ? 0.0f : surf.params.wave_frequency;
                        
                        // Colors
                        gpu->albedo = make_float3(surf.params.deep_color.x, surf.params.deep_color.y, surf.params.deep_color.z);
                        gpu->emission = make_float3(surf.params.shallow_color.x, surf.params.shallow_color.y, surf.params.shallow_color.z);
                        
                        // Depth & Absorption
                        gpu->subsurface = surf.params.depth_max / 100.0f;
                        gpu->subsurface_scale = surf.params.absorption_density;
                        gpu->subsurface_color = make_float3(surf.params.absorption_color.x, surf.params.absorption_color.y, surf.params.absorption_color.z);
                        
                        // Foam
                        gpu->translucent = surf.params.foam_level;
                        gpu->clearcoat = surf.params.shore_foam_intensity;
                        gpu->subsurface_radius.x = surf.params.shore_foam_distance;
                        
                        // Caustics
                        gpu->clearcoat_roughness = surf.params.caustic_intensity;
                        gpu->subsurface_radius.y = surf.params.caustic_scale;
                        gpu->subsurface_anisotropy = surf.params.caustic_speed;
                        
                        // SSS
                        gpu->subsurface_radius.z = surf.params.sss_intensity;
                        
                        // Physics
                        gpu->ior = surf.params.ior;
                        gpu->roughness = surf.params.roughness;
                        
                        // Details (New)
                        gpu->micro_detail_strength = surf.params.micro_detail_strength;
                        gpu->micro_detail_scale = surf.params.micro_detail_scale;
                        gpu->micro_anim_speed = surf.params.micro_anim_speed;
                        gpu->micro_morph_speed = surf.params.micro_morph_speed;
                        gpu->foam_noise_scale = surf.params.foam_noise_scale;
                        gpu->foam_threshold = surf.params.foam_threshold;
                        
                        // FFT (Sync all parameters)
                        gpu->fft_ocean_size = surf.params.fft_ocean_size;
                        gpu->fft_choppiness = surf.params.fft_choppiness;
                        gpu->fft_wind_speed = surf.params.fft_wind_speed;
                        gpu->fft_wind_direction = surf.params.fft_wind_direction;
                        gpu->fft_amplitude = surf.params.fft_amplitude;
                        gpu->fft_time_scale = surf.params.fft_time_scale;
                    }
                }
                
                // CRITICAL: Force FFT Ocean update when parameters change
                // This regenerates the spectrum with new wind/amplitude settings
                if (surf.params.use_fft_ocean) {
                    WaterManager::getInstance().update(0.0f);
                }
                
                // FFT Mesh Displacement - Update mesh and rebuild BVH when params change
                if (surf.params.use_fft_mesh_displacement && surf.params.use_fft_ocean) {
                    // Force mesh update with new height/choppiness values
                    WaterManager::getInstance().updateFFTDrivenMesh(&surf, surf.animation_time);
                    
                    // Rebuild BVH for correct raytracing
                    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                    if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                        ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                    }
                }
                
                // Mark preset as Custom when any parameter is manually changed
                surf.params.current_preset = WaterWaveParams::WaterPreset::Custom;
                
                // Reset accumulation and sync GPU materials for real-time preview
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) {
                    ctx.renderer.updateOptiXMaterialsOnly(ctx.scene, ctx.optix_gpu_ptr);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }
            }
            
            ImGui::Spacing();
            ImGui::Separator();
            
            // Delete button
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.2f, 0.2f, 1.0f));
            if (ImGui::Button("Delete Water Surface", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
                WaterManager::getInstance().removeWaterSurface(ctx.scene, surf.id);
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                    ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                }
                selected_water_idx = -1;
            }
            ImGui::PopStyleColor();
        }
    }
}

#endif

