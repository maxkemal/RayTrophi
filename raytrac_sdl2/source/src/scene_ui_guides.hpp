#pragma once

// ═══════════════════════════════════════════════════════════════════════════════════
// VIEWPORT GUIDES - Safe Areas, Letterbox, Composition Grids
// For RayTrophi Renderer - Professional viewport overlay system
// ═══════════════════════════════════════════════════════════════════════════════════

#include <imgui.h>
#include "CameraPresets.h"

namespace ViewportGuides {

// ═══════════════════════════════════════════════════════════════════════════════════
// VIEWPORT GUIDE SETTINGS (Store these in SceneUI or UIContext)
// ═══════════════════════════════════════════════════════════════════════════════════
struct GuideSettings {
    // Safe Areas
    bool show_safe_areas = false;
    int safe_area_type = 0;  // 0=Both, 1=Title Only, 2=Action Only
    float title_safe_percent = 0.80f;   // 80% of frame
    float action_safe_percent = 0.90f;  // 90% of frame
    
    // Letterbox / Aspect Ratio Overlay
    bool show_letterbox = false;
    int aspect_ratio_index = 0;  // Index into CameraPresets::ASPECT_RATIOS
    float letterbox_opacity = 0.7f;
    
    // Composition Grid
    bool show_grid = false;
    int grid_type = 0;  // 0=Rule of Thirds, 1=Golden Ratio, 2=Center Cross, 3=Diagonal
    
    // Center Crosshair
    bool show_center = false;
    
    // Passepartout (darken outside frame)
    bool show_passepartout = false;
    float passepartout_opacity = 0.5f;
};

// ═══════════════════════════════════════════════════════════════════════════════════
// DRAWING FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════════

// Draw all enabled guides
inline void drawAllGuides(ImDrawList* draw_list, ImVec2 viewport_min, ImVec2 viewport_max, 
                          const GuideSettings& settings) {
    
    float vw = viewport_max.x - viewport_min.x;
    float vh = viewport_max.y - viewport_min.y;
    ImVec2 center(viewport_min.x + vw * 0.5f, viewport_min.y + vh * 0.5f);
    
    // ─────────────────────────────────────────────────────────────────────────────
    // LETTERBOX / ASPECT RATIO OVERLAY
    // ─────────────────────────────────────────────────────────────────────────────
    if (settings.show_letterbox && settings.aspect_ratio_index > 0) {
        float target_ratio = CameraPresets::ASPECT_RATIOS[settings.aspect_ratio_index].ratio;
        float current_ratio = vw / vh;
        
        ImU32 matte_color = IM_COL32(0, 0, 0, (int)(settings.letterbox_opacity * 255));
        
        if (target_ratio > current_ratio) {
            // Horizontal letterbox (bars on top/bottom)
            float target_height = vw / target_ratio;
            float bar_height = (vh - target_height) * 0.5f;
            
            // Top bar
            draw_list->AddRectFilled(
                ImVec2(viewport_min.x, viewport_min.y),
                ImVec2(viewport_max.x, viewport_min.y + bar_height),
                matte_color
            );
            // Bottom bar
            draw_list->AddRectFilled(
                ImVec2(viewport_min.x, viewport_max.y - bar_height),
                ImVec2(viewport_max.x, viewport_max.y),
                matte_color
            );
        } else if (target_ratio < current_ratio) {
            // Vertical pillarbox (bars on sides)
            float target_width = vh * target_ratio;
            float bar_width = (vw - target_width) * 0.5f;
            
            // Left bar
            draw_list->AddRectFilled(
                ImVec2(viewport_min.x, viewport_min.y),
                ImVec2(viewport_min.x + bar_width, viewport_max.y),
                matte_color
            );
            // Right bar
            draw_list->AddRectFilled(
                ImVec2(viewport_max.x - bar_width, viewport_min.y),
                ImVec2(viewport_max.x, viewport_max.y),
                matte_color
            );
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────────
    // SAFE AREAS
    // ─────────────────────────────────────────────────────────────────────────────
    if (settings.show_safe_areas) {
        ImU32 title_color = IM_COL32(255, 100, 100, 180);   // Red for title safe
        ImU32 action_color = IM_COL32(100, 255, 100, 150);  // Green for action safe
        
        // Action Safe (90%) - outer
        if (settings.safe_area_type == 0 || settings.safe_area_type == 2) {
            float margin_x = vw * (1.0f - settings.action_safe_percent) * 0.5f;
            float margin_y = vh * (1.0f - settings.action_safe_percent) * 0.5f;
            draw_list->AddRect(
                ImVec2(viewport_min.x + margin_x, viewport_min.y + margin_y),
                ImVec2(viewport_max.x - margin_x, viewport_max.y - margin_y),
                action_color, 0.0f, 0, 1.5f
            );
        }
        
        // Title Safe (80%) - inner
        if (settings.safe_area_type == 0 || settings.safe_area_type == 1) {
            float margin_x = vw * (1.0f - settings.title_safe_percent) * 0.5f;
            float margin_y = vh * (1.0f - settings.title_safe_percent) * 0.5f;
            draw_list->AddRect(
                ImVec2(viewport_min.x + margin_x, viewport_min.y + margin_y),
                ImVec2(viewport_max.x - margin_x, viewport_max.y - margin_y),
                title_color, 0.0f, 0, 1.5f
            );
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────────
    // COMPOSITION GRID
    // ─────────────────────────────────────────────────────────────────────────────
    if (settings.show_grid) {
        ImU32 grid_color = IM_COL32(255, 255, 255, 80);
        
        switch (settings.grid_type) {
            case 0: // Rule of Thirds
            {
                float third_x = vw / 3.0f;
                float third_y = vh / 3.0f;
                
                // Vertical lines
                draw_list->AddLine(
                    ImVec2(viewport_min.x + third_x, viewport_min.y),
                    ImVec2(viewport_min.x + third_x, viewport_max.y),
                    grid_color, 1.0f
                );
                draw_list->AddLine(
                    ImVec2(viewport_min.x + 2 * third_x, viewport_min.y),
                    ImVec2(viewport_min.x + 2 * third_x, viewport_max.y),
                    grid_color, 1.0f
                );
                
                // Horizontal lines
                draw_list->AddLine(
                    ImVec2(viewport_min.x, viewport_min.y + third_y),
                    ImVec2(viewport_max.x, viewport_min.y + third_y),
                    grid_color, 1.0f
                );
                draw_list->AddLine(
                    ImVec2(viewport_min.x, viewport_min.y + 2 * third_y),
                    ImVec2(viewport_max.x, viewport_min.y + 2 * third_y),
                    grid_color, 1.0f
                );
                break;
            }
            
            case 1: // Golden Ratio (phi = 1.618)
            {
                float phi = 1.618033988749895f;
                float golden_x1 = vw / (1.0f + phi);
                float golden_x2 = vw * phi / (1.0f + phi);
                float golden_y1 = vh / (1.0f + phi);
                float golden_y2 = vh * phi / (1.0f + phi);
                
                // Vertical lines
                draw_list->AddLine(
                    ImVec2(viewport_min.x + golden_x1, viewport_min.y),
                    ImVec2(viewport_min.x + golden_x1, viewport_max.y),
                    grid_color, 1.0f
                );
                draw_list->AddLine(
                    ImVec2(viewport_min.x + golden_x2, viewport_min.y),
                    ImVec2(viewport_min.x + golden_x2, viewport_max.y),
                    grid_color, 1.0f
                );
                
                // Horizontal lines
                draw_list->AddLine(
                    ImVec2(viewport_min.x, viewport_min.y + golden_y1),
                    ImVec2(viewport_max.x, viewport_min.y + golden_y1),
                    grid_color, 1.0f
                );
                draw_list->AddLine(
                    ImVec2(viewport_min.x, viewport_min.y + golden_y2),
                    ImVec2(viewport_max.x, viewport_min.y + golden_y2),
                    grid_color, 1.0f
                );
                break;
            }
            
            case 2: // Center Cross
            {
                draw_list->AddLine(
                    ImVec2(center.x, viewport_min.y),
                    ImVec2(center.x, viewport_max.y),
                    grid_color, 1.0f
                );
                draw_list->AddLine(
                    ImVec2(viewport_min.x, center.y),
                    ImVec2(viewport_max.x, center.y),
                    grid_color, 1.0f
                );
                break;
            }
            
            case 3: // Diagonal
            {
                draw_list->AddLine(viewport_min, viewport_max, grid_color, 1.0f);
                draw_list->AddLine(
                    ImVec2(viewport_max.x, viewport_min.y),
                    ImVec2(viewport_min.x, viewport_max.y),
                    grid_color, 1.0f
                );
                break;
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────────
    // CENTER CROSSHAIR
    // ─────────────────────────────────────────────────────────────────────────────
    if (settings.show_center) {
        ImU32 crosshair_color = IM_COL32(255, 255, 0, 200);
        float size = 20.0f;
        float gap = 5.0f;
        
        // Horizontal
        draw_list->AddLine(
            ImVec2(center.x - size, center.y),
            ImVec2(center.x - gap, center.y),
            crosshair_color, 2.0f
        );
        draw_list->AddLine(
            ImVec2(center.x + gap, center.y),
            ImVec2(center.x + size, center.y),
            crosshair_color, 2.0f
        );
        
        // Vertical
        draw_list->AddLine(
            ImVec2(center.x, center.y - size),
            ImVec2(center.x, center.y - gap),
            crosshair_color, 2.0f
        );
        draw_list->AddLine(
            ImVec2(center.x, center.y + gap),
            ImVec2(center.x, center.y + size),
            crosshair_color, 2.0f
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════
// UI DRAWING FUNCTION (Call from Selection Properties when camera is selected)
// ═══════════════════════════════════════════════════════════════════════════════════
inline void drawGuideSettingsUI(GuideSettings& settings) {
    if (ImGui::CollapsingHeader("Viewport Guides")) {
        
        // Safe Areas
        ImGui::Checkbox("Safe Areas", &settings.show_safe_areas);
        if (settings.show_safe_areas) {
            ImGui::Indent();
            const char* safe_types[] = {"Both", "Title Safe", "Action Safe"};
            ImGui::Combo("Type##SafeArea", &settings.safe_area_type, safe_types, 3);
            ImGui::Unindent();
        }
        
        // Letterbox
        ImGui::Checkbox("Aspect Ratio Overlay", &settings.show_letterbox);
        if (settings.show_letterbox) {
            ImGui::Indent();
            if (ImGui::BeginCombo("Aspect##Letterbox", 
                CameraPresets::ASPECT_RATIOS[settings.aspect_ratio_index].name)) 
            {
                for (size_t i = 0; i < CameraPresets::ASPECT_RATIO_COUNT; ++i) {
                    bool is_selected = (settings.aspect_ratio_index == (int)i);
                    if (ImGui::Selectable(CameraPresets::ASPECT_RATIOS[i].name, is_selected)) {
                        settings.aspect_ratio_index = (int)i;
                    }
                    if (is_selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            ImGui::SliderFloat("Opacity##Letterbox", &settings.letterbox_opacity, 0.0f, 1.0f);
            ImGui::Unindent();
        }
        
        // Composition Grid
        ImGui::Checkbox("Composition Grid", &settings.show_grid);
        if (settings.show_grid) {
            ImGui::Indent();
            const char* grid_types[] = {"Rule of Thirds", "Golden Ratio", "Center Cross", "Diagonal"};
            ImGui::Combo("Grid##Comp", &settings.grid_type, grid_types, 4);
            ImGui::Unindent();
        }
        
        // Center Crosshair
        ImGui::Checkbox("Center Crosshair", &settings.show_center);
    }
}

} // namespace ViewportGuides
