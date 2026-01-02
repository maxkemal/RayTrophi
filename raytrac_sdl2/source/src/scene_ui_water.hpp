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
                    // Also select in main scene selection logic if needed
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
            ImGui::TextColored(ImVec4(1,1,0,1), "Edit: %s", surf.name.c_str());
            
            ImGui::Text("Waves");
            changed |= ImGui::DragFloat("Speed", &surf.params.wave_speed, 0.1f, 0.0f, 10.0f);
            changed |= ImGui::SliderFloat("Height", &surf.params.wave_strength, 0.0f, 5.0f);
            changed |= ImGui::SliderFloat("Frequency", &surf.params.wave_frequency, 0.1f, 5.0f);
             
            ImGui::Spacing();
            ImGui::Text("Appearance");
            changed |= ImGui::ColorEdit3("Deep Color", &surf.params.deep_color.x);
            changed |= ImGui::ColorEdit3("Shallow Color", &surf.params.shallow_color.x);
            changed |= ImGui::SliderFloat("Clarity", &surf.params.clarity, 0.0f, 5.0f);
            changed |= ImGui::SliderFloat("Foam", &surf.params.foam_level, 0.0f, 1.0f);
            
            if (changed) {
                // If we had GPU wave simulation, we'd update params here
                // For now just wake up renderer
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr) ctx.optix_gpu_ptr->resetAccumulation();
            }
            
            ImGui::Spacing();
            if (ImGui::Button("Delete Water Surface", ImVec2(-1, 0))) {
                WaterManager::getInstance().removeWaterSurface(ctx.scene, surf.id);
                // Rebuild
                ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                ctx.renderer.resetCPUAccumulation();
                if (ctx.optix_gpu_ptr && ctx.render_settings.use_optix) {
                    ctx.renderer.rebuildOptiXGeometry(ctx.scene, ctx.optix_gpu_ptr);
                }
                selected_water_idx = -1;
            }
        }
    }
}

#endif
