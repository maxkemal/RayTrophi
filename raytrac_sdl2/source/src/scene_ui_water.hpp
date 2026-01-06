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
            if (ImGui::CollapsingHeader("Waves", ImGuiTreeNodeFlags_DefaultOpen)) {
                changed |= ImGui::DragFloat("Speed##wave", &surf.params.wave_speed, 0.05f, 0.0f, 10.0f);
                changed |= ImGui::SliderFloat("Height##wave", &surf.params.wave_strength, 0.0f, 3.0f);
                changed |= ImGui::SliderFloat("Frequency##wave", &surf.params.wave_frequency, 0.1f, 5.0f);
            }
            
            // === COLORS ===
            if (ImGui::CollapsingHeader("Colors", ImGuiTreeNodeFlags_DefaultOpen)) {
                changed |= ImGui::ColorEdit3("Shallow Color", &surf.params.shallow_color.x);
                changed |= ImGui::ColorEdit3("Deep Color", &surf.params.deep_color.x);
            }
            
            // === DEPTH & ABSORPTION ===
            if (ImGui::CollapsingHeader("Depth & Absorption")) {
                changed |= ImGui::DragFloat("Max Depth", &surf.params.depth_max, 0.5f, 1.0f, 100.0f, "%.1f m");
                ImGui::SetItemTooltip("Distance at which water reaches full deep color");
                
                changed |= ImGui::ColorEdit3("Absorption Tint", &surf.params.absorption_color.x);
                ImGui::SetItemTooltip("What colors are absorbed by the water");
                
                changed |= ImGui::SliderFloat("Absorption Density", &surf.params.absorption_density, 0.0f, 2.0f);
                ImGui::SetItemTooltip("How quickly light is absorbed (murkiness)");
            }
            
            // === FOAM ===
            if (ImGui::CollapsingHeader("Foam", ImGuiTreeNodeFlags_DefaultOpen)) {
                changed |= ImGui::SliderFloat("Wave Foam", &surf.params.foam_level, 0.0f, 1.0f);
                ImGui::SetItemTooltip("Foam on wave crests");
                
                ImGui::Separator();
                ImGui::TextDisabled("Shore Foam");
                changed |= ImGui::SliderFloat("Intensity##shore", &surf.params.shore_foam_intensity, 0.0f, 1.0f);
                changed |= ImGui::DragFloat("Distance##shore", &surf.params.shore_foam_distance, 0.1f, 0.1f, 10.0f, "%.1f m");
                ImGui::SetItemTooltip("How far from shore foam appears");
            }
            
            // === CAUSTICS ===
            if (ImGui::CollapsingHeader("Caustics")) {
                changed |= ImGui::SliderFloat("Intensity##caustic", &surf.params.caustic_intensity, 0.0f, 1.0f);
                changed |= ImGui::DragFloat("Scale##caustic", &surf.params.caustic_scale, 0.1f, 0.1f, 10.0f);
                ImGui::SetItemTooltip("Size of caustic patterns");
                changed |= ImGui::DragFloat("Speed##caustic", &surf.params.caustic_speed, 0.1f, 0.1f, 5.0f);
            }
            
            // === SSS ===
            if (ImGui::CollapsingHeader("Scattering")) {
                changed |= ImGui::SliderFloat("SSS Intensity", &surf.params.sss_intensity, 0.0f, 0.5f);
                ImGui::SetItemTooltip("Sub-surface light scattering at edges");
                changed |= ImGui::ColorEdit3("SSS Color", &surf.params.sss_color.x);
            }
            
            // === FFT OCEAN (TESSENDORF) ===
            if (ImGui::CollapsingHeader("FFT Ocean (Film Quality)")) {
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
                    changed |= ImGui::DragFloat("Ocean Size", &surf.params.fft_ocean_size, 1.0f, 10.0f, 10000.0f, "%.0f m");
                    ImGui::SetItemTooltip("World space covered by one tile (tiles infinitely)");
                    
                    ImGui::Separator();
                    ImGui::Text("Wind Settings:");
                    
                    // Wind speed
                    changed |= ImGui::DragFloat("Wind Speed", &surf.params.fft_wind_speed, 0.1f, 0.0f, 200.0f, "%.1f m/s");
                    ImGui::SetItemTooltip("Higher = larger, more energetic waves. >30 is storm force.");
                    
                    // Wind direction
                    changed |= ImGui::SliderAngle("Wind Direction", &surf.params.fft_wind_direction, 0.0f, 360.0f);
                    ImGui::SetItemTooltip("Direction the waves travel");
                    
                    ImGui::Separator();
                    ImGui::Text("Wave Appearance:");
                    
                    // Choppiness
                    changed |= ImGui::DragFloat("Choppiness", &surf.params.fft_choppiness, 0.05f, 0.0f, 5.0f);
                    ImGui::SetItemTooltip("Horizontal displacement - makes peaks sharper. Too high causes loops.");
                    
                    // Amplitude
                    changed |= ImGui::DragFloat("Amplitude Scale", &surf.params.fft_amplitude, 0.00001f, 0.000001f, 0.1f, "%.6f");
                    ImGui::SetItemTooltip("Overall wave height multiplier (Phillips A parameter). Very sensitive!");
                    
                    // Time scale
                    changed |= ImGui::DragFloat("Animation Speed", &surf.params.fft_time_scale, 0.1f, 0.0f, 20.0f);
                    ImGui::SetItemTooltip("Speed of wave animation");
                    
                    ImGui::Unindent();
                } else {
                    ImGui::TextDisabled("Enable to use film-quality FFT ocean");
                }
            }
            
            // === PHYSICS ===
            if (ImGui::CollapsingHeader("Physics")) {
                changed |= ImGui::SliderFloat("IOR", &surf.params.ior, 1.0f, 2.0f, "%.3f");
                ImGui::SetItemTooltip("Index of Refraction (Water = 1.333)");
                changed |= ImGui::SliderFloat("Roughness", &surf.params.roughness, 0.0f, 0.2f, "%.3f");
                ImGui::SetItemTooltip("Surface micro-roughness");
                changed |= ImGui::SliderFloat("Clarity", &surf.params.clarity, 0.0f, 1.0f);
            }
            
            // === SURFACE DETAIL ===
            if (ImGui::CollapsingHeader("Surface Detail (Realism)", ImGuiTreeNodeFlags_DefaultOpen)) {
                changed |= ImGui::SliderFloat("Micro Detail Strength", &surf.params.micro_detail_strength, 0.0f, 0.2f);
                ImGui::SetItemTooltip("Adds high-frequency noise/ripples to break up the smooth surface");
                changed |= ImGui::DragFloat("Micro Detail Scale", &surf.params.micro_detail_scale, 0.5f, 1.0f, 100.0f);
                
                ImGui::Separator();
                ImGui::TextDisabled("Foam Tuning");
                changed |= ImGui::DragFloat("Foam Noise Scale", &surf.params.foam_noise_scale, 0.1f, 1.0f, 50.0f);
                ImGui::SetItemTooltip("Scale of noise used to break up foam");
                changed |= ImGui::SliderFloat("Foam Threshold", &surf.params.foam_threshold, 0.0f, 1.0f);
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
