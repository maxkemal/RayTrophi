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

// ═══════════════════════════════════════════════════════════════════════════════
// WATER PANEL UI
// ═══════════════════════════════════════════════════════════════════════════════

#include "WaterSystem.h"


void SceneUI::drawWaterPanel(UIContext& ctx) {
    ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "WATER SYSTEM");
    ImGui::Separator();
    
    // Create new water
    if (ImGui::Button("Add Water Plane", ImVec2(-1, 30))) {
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
                if (waters[i].type == WaterSurface::Type::River) continue; // Skip rivers (handled in River panel)
                
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
            
            // === WAVES ===
            if (UIWidgets::BeginSection("Waves", ImVec4(0.0f, 0.4f, 1.0f, 1.0f))) {
                if (SceneUI::DrawSmartFloat("w_wspd", "Speed", &surf.params.wave_speed, 0.0f, 10.0f, "%.2f", false, nullptr, 16)) changed = true;
                if (SceneUI::DrawSmartFloat("w_whgt", "Height", &surf.params.wave_strength, 0.0f, 3.0f, "%.2f", false, nullptr, 16)) changed = true;
                if (SceneUI::DrawSmartFloat("w_wfrq", "Frequency", &surf.params.wave_frequency, 0.1f, 5.0f, "%.2f", false, nullptr, 16)) changed = true;
                UIWidgets::EndSection();
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
                
                changed |= ImGui::Checkbox("Enable FFT Ocean", &surf.params.use_fft_ocean);
                
                if (surf.params.use_fft_ocean) {
                    ImGui::Indent();
                    
                    // Resolution dropdown
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
                    
                    // Ocean size
                    // Ocean size
                    if (SceneUI::DrawSmartFloat("fft_sz", "Ocean Size", &surf.params.fft_ocean_size, 10.0f, 10000.0f, "%.0f m", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("World space covered by one tile (tiles infinitely)");
                    
                    ImGui::Separator();
                    ImGui::Text("Wind Settings:");
                    
                    // Wind speed
                    if (SceneUI::DrawSmartFloat("fft_ws", "Wind Speed", &surf.params.fft_wind_speed, 0.0f, 200.0f, "%.1f m/s", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Higher = larger, more energetic waves. >30 is storm force.");
                    
                    // Wind direction
                    float wind_deg = surf.params.fft_wind_direction * 180.0f / 3.14159f;
                    if (SceneUI::DrawSmartFloat("fft_wd", "Wind Direction", &wind_deg, 0.0f, 360.0f, "%.0f deg", false, nullptr, 16)) {
                        surf.params.fft_wind_direction = wind_deg * 3.14159f / 180.0f;
                        changed = true;
                    }
                    ImGui::SetItemTooltip("Direction the waves travel");
                    
                    ImGui::Separator();
                    ImGui::Text("Wave Appearance:");
                    
                    // Choppiness
                    if (SceneUI::DrawSmartFloat("fft_ch", "Choppiness", &surf.params.fft_choppiness, 0.0f, 5.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Horizontal displacement - makes peaks sharper. Too high causes loops.");
                    
                    // Amplitude
                    if (SceneUI::DrawSmartFloat("fft_amp", "Amplitude Scale", &surf.params.fft_amplitude, 0.000001f, 0.1f, "%.6f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Overall wave height multiplier (Phillips A parameter). Very sensitive!");
                    
                    // Time scale
                    if (SceneUI::DrawSmartFloat("fft_ts", "Animation Speed", &surf.params.fft_time_scale, 0.0f, 20.0f, "%.2f", false, nullptr, 16)) changed = true;
                    ImGui::SetItemTooltip("Speed of wave animation");
                    
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
                ImGui::TextDisabled("Foam Tuning");
                if (SceneUI::DrawSmartFloat("w_fns", "Foam Noise Scale", &surf.params.foam_noise_scale, 1.0f, 50.0f, "%.1f", false, nullptr, 16)) changed = true;
                ImGui::SetItemTooltip("Scale of noise used to break up foam");
                if (SceneUI::DrawSmartFloat("w_fth", "Foam Threshold", &surf.params.foam_threshold, 0.0f, 1.0f, "%.2f", false, nullptr, 16)) changed = true;
                UIWidgets::EndSection();
            }
            
            // === APPLY CHANGES ===
            if (changed) {
                // Update GPU material with new parameters
                auto mat = MaterialManager::getInstance().getMaterial(surf.material_id);
                if (mat && mat->gpuMaterial) {
                    auto& gpu = mat->gpuMaterial;
                    
                    // Wave params
                    gpu->anisotropic = surf.params.wave_speed;
                    gpu->sheen = fmaxf(0.001f, surf.params.wave_strength);
                    gpu->sheen_tint = surf.params.wave_frequency;
                    
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
                    gpu->foam_noise_scale = surf.params.foam_noise_scale;
                    gpu->foam_threshold = surf.params.foam_threshold;
                    
                    // FFT (Sync)
                    gpu->fft_ocean_size = surf.params.fft_ocean_size;
                    gpu->fft_choppiness = surf.params.fft_choppiness;
                }
                
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
            if (ImGui::Button("Delete Water Surface", ImVec2(-1, 0))) {
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

