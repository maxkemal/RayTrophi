// ===============================================================================
// SCENE UI - PRO CAMERA HUD
// ===============================================================================
// Professional camera overlay features:
//   - Histogram         : RGB/Luma brightness distribution graph (draggable)
//   - Focus Peaking     : Sharp edge highlighting for manual focus
//   - Zebra Stripes     : Overexposure warning pattern
//   - Multi-Point AF    : Autofocus point grid with light metering
// ===============================================================================

#include "scene_ui.h"
#include "renderer.h"
#include "scene_data.h"
#include "Ray.h"
#include "Hittable.h"
#include "ParallelBVHNode.h"
#include "Camera.h"
#include <cstdio>
#include "imgui.h"
#include <cmath>
#include <algorithm>
#include <array>

// =============================================================================
// HISTOGRAM OVERLAY (Draggable, real camera viewfinder style)
// =============================================================================
void SceneUI::drawHistogramOverlay(UIContext& ctx) {
    if (!viewport_settings.show_histogram) return;
    if (!viewport_settings.show_camera_hud) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    // Draggable histogram position (stored as static)
    static ImVec2 hist_pos = ImVec2(-1, -1);
    if (hist_pos.x < 0) {
        // Initial position: top-left of viewport area (after left panel)
        hist_pos = ImVec2(380.0f, 50.0f);
    }

    float hist_width = 180.0f;
    float hist_height = 70.0f;

    // Check for dragging
    ImVec2 mouse = io.MousePos;
    bool is_over = (mouse.x >= hist_pos.x && mouse.x <= hist_pos.x + hist_width &&
                    mouse.y >= hist_pos.y && mouse.y <= hist_pos.y + 15.0f);  // Title bar area

    static bool is_dragging = false;
    static ImVec2 drag_offset;

    if (is_over && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
        is_dragging = true;
        drag_offset = ImVec2(mouse.x - hist_pos.x, mouse.y - hist_pos.y);
        hud_captured_mouse = true;
    }

    if (is_dragging) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            hist_pos.x = mouse.x - drag_offset.x;
            hist_pos.y = mouse.y - drag_offset.y;
            // Clamp to screen
            hist_pos.x = std::max(0.0f, std::min(hist_pos.x, io.DisplaySize.x - hist_width));
            hist_pos.y = std::max(20.0f, std::min(hist_pos.y, io.DisplaySize.y - hist_height - 100.0f));
        } else {
            is_dragging = false;
        }
    }

    // Background
    float alpha = viewport_settings.histogram_opacity;
    ImU32 col_bg = IM_COL32(20, 20, 25, (int)(220 * alpha));
    ImU32 col_border = IM_COL32(80, 80, 90, (int)(255 * alpha));
    ImU32 col_title = IM_COL32(40, 40, 50, (int)(255 * alpha));

    // Draw background
    draw_list->AddRectFilled(
        hist_pos,
        ImVec2(hist_pos.x + hist_width, hist_pos.y + hist_height),
        col_bg, 3.0f
    );
    
    // Title bar (draggable area)
    draw_list->AddRectFilled(
        hist_pos,
        ImVec2(hist_pos.x + hist_width, hist_pos.y + 14.0f),
        is_over ? IM_COL32(60, 60, 70, 255) : col_title, 3.0f
    );
    
    draw_list->AddRect(
        hist_pos,
        ImVec2(hist_pos.x + hist_width, hist_pos.y + hist_height),
        col_border, 3.0f, 0, 1.0f
    );

    // Title text
    const char* title = viewport_settings.histogram_mode == 0 ? "HISTOGRAM RGB" : "HISTOGRAM LUMA";
    draw_list->AddText(ImVec2(hist_pos.x + 5, hist_pos.y + 1), IM_COL32(200, 200, 200, 255), title);

    // Get frame buffer
    const auto& frame_buffer = ctx.renderer.getFrameBuffer();
    int fb_width = ctx.renderer.getImageWidth();
    int fb_height = ctx.renderer.getImageHeight();
    
    // Check if we have valid data
    if (frame_buffer.empty() || fb_width <= 0 || fb_height <= 0) {
        // Show "No Data" message
        draw_list->AddText(
            ImVec2(hist_pos.x + hist_width/2 - 25, hist_pos.y + hist_height/2),
            IM_COL32(150, 150, 150, 200),
            "No Data"
        );
        return;
    }

    // Calculate histogram
    const int num_bins = 64;
    std::array<int, num_bins> bins_r = {0};
    std::array<int, num_bins> bins_g = {0};
    std::array<int, num_bins> bins_b = {0};
    std::array<int, num_bins> bins_luma = {0};

    int sample_step = std::max(1, (int)frame_buffer.size() / 8000);
    
    for (size_t i = 0; i < frame_buffer.size(); i += sample_step) {
        const Vec3& pixel = frame_buffer[i];
        
        float r = std::clamp(pixel.x, 0.0f, 1.0f);
        float g = std::clamp(pixel.y, 0.0f, 1.0f);
        float b = std::clamp(pixel.z, 0.0f, 1.0f);
        float luma = 0.299f * r + 0.587f * g + 0.114f * b;
        
        bins_r[std::min((int)(r * num_bins), num_bins - 1)]++;
        bins_g[std::min((int)(g * num_bins), num_bins - 1)]++;
        bins_b[std::min((int)(b * num_bins), num_bins - 1)]++;
        bins_luma[std::min((int)(luma * num_bins), num_bins - 1)]++;
    }

    // Find max for normalization
    int max_val = 1;
    for (int i = 0; i < num_bins; i++) {
        max_val = std::max(max_val, std::max({bins_r[i], bins_g[i], bins_b[i], bins_luma[i]}));
    }

    // Draw histogram bars
    float graph_x = hist_pos.x + 3;
    float graph_y = hist_pos.y + 18;
    float graph_w = hist_width - 6;
    float graph_h = hist_height - 22;
    float bar_width = graph_w / num_bins;

    for (int i = 0; i < num_bins; i++) {
        float x = graph_x + i * bar_width;
        
        if (viewport_settings.histogram_mode == 0) {  // RGB
            float h_r = (float)bins_r[i] / max_val * graph_h;
            float h_g = (float)bins_g[i] / max_val * graph_h;
            float h_b = (float)bins_b[i] / max_val * graph_h;
            
            // Draw with slight transparency for overlap effect
            draw_list->AddRectFilled(ImVec2(x, graph_y + graph_h - h_r), ImVec2(x + bar_width, graph_y + graph_h), IM_COL32(255, 60, 60, 150));
            draw_list->AddRectFilled(ImVec2(x, graph_y + graph_h - h_g), ImVec2(x + bar_width, graph_y + graph_h), IM_COL32(60, 255, 60, 150));
            draw_list->AddRectFilled(ImVec2(x, graph_y + graph_h - h_b), ImVec2(x + bar_width, graph_y + graph_h), IM_COL32(60, 60, 255, 150));
        } else {  // Luma
            float h = (float)bins_luma[i] / max_val * graph_h;
            draw_list->AddRectFilled(ImVec2(x, graph_y + graph_h - h), ImVec2(x + bar_width, graph_y + graph_h), IM_COL32(220, 220, 220, 200));
        }
    }

    // Clipping indicators
    if (bins_r[num_bins-1] > max_val/10 || bins_g[num_bins-1] > max_val/10 || bins_b[num_bins-1] > max_val/10) {
        draw_list->AddText(ImVec2(hist_pos.x + hist_width - 40, hist_pos.y + 1), IM_COL32(255, 100, 100, 255), "CLIP!");
    }
}

// =============================================================================
// FOCUS PEAKING OVERLAY - Actual edge detection on rendered image
// =============================================================================
void SceneUI::drawFocusPeakingOverlay(UIContext& ctx) {
    if (!viewport_settings.show_focus_peaking) return;
    if (!viewport_settings.show_camera_hud) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    const auto& frame_buffer = ctx.renderer.getFrameBuffer();
    int width = ctx.renderer.getImageWidth();
    int height = ctx.renderer.getImageHeight();
    
    if (width <= 0 || height <= 0 || frame_buffer.empty()) return;

    // Peaking colors
    ImU32 peak_colors[] = {
        IM_COL32(255, 50, 50, 220),   // Red
        IM_COL32(255, 255, 50, 220),  // Yellow
        IM_COL32(50, 255, 50, 220),   // Green
        IM_COL32(50, 150, 255, 220),  // Blue
        IM_COL32(255, 255, 255, 220)  // White
    };
    ImU32 peak_color = peak_colors[viewport_settings.focus_peaking_color % 5];

    // Scale factor from render resolution to display
    float scale_x = io.DisplaySize.x / (float)width;
    float scale_y = io.DisplaySize.y / (float)height;

    // Optimized edge detection with larger step for performance
    int step = 4;
    float threshold = viewport_settings.focus_peaking_threshold;

    // Lambda to get luma at position
    auto getLuma = [&](int x, int y) -> float {
        if (x < 0 || x >= width || y < 0 || y >= height) return 0.0f;
        int idx = y * width + x;
        if (idx < 0 || idx >= (int)frame_buffer.size()) return 0.0f;
        const Vec3& c = frame_buffer[idx];
        return 0.299f * c.x + 0.587f * c.y + 0.114f * c.z;
    };

    for (int y = step; y < height - step; y += step) {
        for (int x = step; x < width - step; x += step) {
            // Sobel operator for edge detection
            float gx = 0.0f, gy = 0.0f;
            
            // Sobel X kernel
            gx -= getLuma(x-1, y-1);
            gx -= 2.0f * getLuma(x-1, y);
            gx -= getLuma(x-1, y+1);
            gx += getLuma(x+1, y-1);
            gx += 2.0f * getLuma(x+1, y);
            gx += getLuma(x+1, y+1);
            
            // Sobel Y kernel
            gy -= getLuma(x-1, y-1);
            gy -= 2.0f * getLuma(x, y-1);
            gy -= getLuma(x+1, y-1);
            gy += getLuma(x-1, y+1);
            gy += 2.0f * getLuma(x, y+1);
            gy += getLuma(x+1, y+1);
            
            float edge_strength = sqrtf(gx*gx + gy*gy);

            if (edge_strength > threshold) {
                float screen_x = x * scale_x;
                float screen_y = y * scale_y;
                
                // Draw peaking indicator (small square for visibility)
                float size = 2.0f;
                draw_list->AddRectFilled(
                    ImVec2(screen_x - size, screen_y - size),
                    ImVec2(screen_x + size, screen_y + size),
                    peak_color
                );
            }
        }
    }
}

// =============================================================================
// ZEBRA STRIPES OVERLAY
// =============================================================================
// =============================================================================
// ZEBRA STRIPES OVERLAY - Fixed to detect HDR overexposure
// =============================================================================
void SceneUI::drawZebraOverlay(UIContext& ctx) {
    if (!viewport_settings.show_zebra) return;
    if (!viewport_settings.show_camera_hud) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    const auto& frame_buffer = ctx.renderer.getFrameBuffer();
    int width = ctx.renderer.getImageWidth();
    int height = ctx.renderer.getImageHeight();
    
    if (width <= 0 || height <= 0 || frame_buffer.empty()) return;

    // Threshold - for HDR, values can be > 1.0
    // 0.95 means 95% of max displayable brightness
    float threshold = viewport_settings.zebra_threshold;
    int step = 5;

    float scale_x = io.DisplaySize.x / (float)width;
    float scale_y = io.DisplaySize.y / (float)height;

    static float stripe_offset = 0.0f;
    stripe_offset += io.DeltaTime * 50.0f;
    if (stripe_offset > 8.0f) stripe_offset -= 8.0f;

    for (int y = 0; y < height; y += step) {
        for (int x = 0; x < width; x += step) {
            int idx = y * width + x;
            if (idx >= (int)frame_buffer.size()) continue;

            const Vec3& c = frame_buffer[idx];
            
            // Check for overexposure - any channel above threshold OR luminance above threshold
            // For HDR, also check if any channel > 1.0
            float luma = 0.299f * c.x + 0.587f * c.y + 0.114f * c.z;
            bool is_overexposed = (c.x > threshold || c.y > threshold || c.z > threshold || luma > threshold);

            if (is_overexposed) {
                float screen_x = x * scale_x;
                float screen_y = y * scale_y;
                
                int stripe_phase = (int)(screen_x + screen_y + stripe_offset) % 8;
                if (stripe_phase < 4) {
                    draw_list->AddRectFilled(
                        ImVec2(screen_x, screen_y),
                        ImVec2(screen_x + step * scale_x, screen_y + step * scale_y),
                        IM_COL32(255, 0, 255, 140)
                    );
                }
            }
        }
    }
}

// =============================================================================
// MULTI-POINT AF OVERLAY - Real Focus Detection!
// When focus ring is adjusted, points that are IN FOCUS light up GREEN
// =============================================================================
void SceneUI::drawAFPointsOverlay(UIContext& ctx) {
    if (!viewport_settings.show_af_points) return;
    if (!viewport_settings.show_camera_hud) return;
    if (!ctx.scene.camera) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    Camera& cam = *ctx.scene.camera;
    float focus_dist = cam.focus_dist;

    float cx = io.DisplaySize.x * 0.5f;
    float cy = io.DisplaySize.y * 0.5f;

    int grid_cols = 3;
    int grid_rows = 3;
    float spacing_x = io.DisplaySize.x * 0.12f;
    float spacing_y = io.DisplaySize.y * 0.12f;

    if (viewport_settings.af_mode == 2) {  // Zone21
        grid_cols = 5;
        grid_rows = 5;
        spacing_x = io.DisplaySize.x * 0.08f;
        spacing_y = io.DisplaySize.y * 0.08f;
    }

    // Colors
    ImU32 col_out_of_focus = IM_COL32(150, 150, 150, 150);  // Gray (not in focus)
    ImU32 col_in_focus = IM_COL32(0, 255, 0, 255);          // Bright GREEN (in focus!)
    ImU32 col_selected = IM_COL32(255, 100, 100, 255);      // Red (selected point)

    float point_size = 12.0f;
    int point_idx = 0;

    float start_x = cx - (grid_cols - 1) * spacing_x * 0.5f;
    float start_y = cy - (grid_rows - 1) * spacing_y * 0.5f;

    // Get BVH for raycasting
    const auto* bvh = ctx.scene.bvh.get();

    for (int row = 0; row < grid_rows; row++) {
        for (int col = 0; col < grid_cols; col++) {
            float px = start_x + col * spacing_x;
            float py = start_y + row * spacing_y;

            bool is_selected = (point_idx == viewport_settings.af_selected_point);
            
            // === REAL FOCUS CHECK ===
            // Cast a ray from camera through this screen point and check hit distance
            bool is_in_focus = false;
            float hit_distance = -1.0f;

            if (bvh) {
                // Convert screen position to normalized device coordinates
                float ndc_x = (px / io.DisplaySize.x) * 2.0f - 1.0f;
                float ndc_y = 1.0f - (py / io.DisplaySize.y) * 2.0f;  // Flip Y
                
                // Generate ray from camera through this point
                float half_height = tanf(cam.vfov * 0.5f * 3.14159f / 180.0f);
                float half_width = cam.aspect_ratio * half_height;
                
                Vec3 ray_dir = cam.w * (-1.0f) + cam.u * (ndc_x * half_width) + cam.v * (ndc_y * half_height);
                ray_dir = ray_dir.normalize();
                
                Ray ray(cam.lookfrom, ray_dir);
                HitRecord rec;
                
                if (bvh->hit(ray, 0.001f, 10000.0f, rec)) {
                    hit_distance = rec.t;
                    
                    // === AF-C (Continuous) Logic ===
                    if (is_selected && viewport_settings.focus_mode == 2) { 
                         if (std::abs(cam.focus_dist - hit_distance) > 0.02f) { 
                             cam.focus_dist = hit_distance;
                             ctx.renderer.resetCPUAccumulation();
                             if (ctx.backend_ptr) {
                                 ctx.backend_ptr->syncCamera(cam);
                                 ctx.backend_ptr->resetAccumulation();
                             }
                         }
                    }
                    
                    // Check if hit distance is close to focus distance
                    // Base tolerance: 3.0%
                    float tolerance = focus_dist * 0.03f;
                    
                    if (cam.aperture > 0.0f) {
                        tolerance /= (1.0f + cam.aperture * 10.0f);
                    }
                    tolerance = std::max(tolerance, 0.05f);
                    
                    if (std::abs(hit_distance - focus_dist) < tolerance) {
                        is_in_focus = true;
                    }
                }
            }

            // Determine bracket color based on focus state
            ImU32 bracket_col = col_out_of_focus;
            if (is_in_focus) {
                bracket_col = col_in_focus;  // GREEN when in focus!
            }
            if (is_selected) {
                bracket_col = col_selected;  // Override with red if selected
            }

            // Draw AF point bracket (camera viewfinder style)
            float half = point_size;
            float corner = 5.0f;
            float thickness = (is_in_focus || is_selected) ? 2.5f : 1.5f;

            // Top-left
            draw_list->AddLine(ImVec2(px - half, py - half), ImVec2(px - half + corner, py - half), bracket_col, thickness);
            draw_list->AddLine(ImVec2(px - half, py - half), ImVec2(px - half, py - half + corner), bracket_col, thickness);
            // Top-right
            draw_list->AddLine(ImVec2(px + half, py - half), ImVec2(px + half - corner, py - half), bracket_col, thickness);
            draw_list->AddLine(ImVec2(px + half, py - half), ImVec2(px + half, py - half + corner), bracket_col, thickness);
            // Bottom-left
            draw_list->AddLine(ImVec2(px - half, py + half), ImVec2(px - half + corner, py + half), bracket_col, thickness);
            draw_list->AddLine(ImVec2(px - half, py + half), ImVec2(px - half, py + half - corner), bracket_col, thickness);
            // Bottom-right
            draw_list->AddLine(ImVec2(px + half, py + half), ImVec2(px + half - corner, py + half), bracket_col, thickness);
            draw_list->AddLine(ImVec2(px + half, py + half), ImVec2(px + half, py + half - corner), bracket_col, thickness);

            // Center dot when in focus
            if (is_in_focus) {
                draw_list->AddCircleFilled(ImVec2(px, py), 3.0f, col_in_focus, 8);
            }

            // Handle click to select point
            ImVec2 mouse = io.MousePos;
            bool mouse_over = (mouse.x >= px - half && mouse.x <= px + half &&
                              mouse.y >= py - half && mouse.y <= py + half);

            if (mouse_over && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
                viewport_settings.af_selected_point = point_idx;
                viewport_settings.af_mode = 0;
                hud_captured_mouse = true;
                
                // Auto-focus: Set focus distance to hit distance (ONLY FOR AF-S)
                if (viewport_settings.focus_mode == 1 && hit_distance > 0.0f) {
                    cam.focus_dist = hit_distance;
                    cam.update_camera_vectors();
                    
                    // Update GPU
                    if (ctx.backend_ptr) {
                        ctx.backend_ptr->syncCamera(cam);
                        ctx.backend_ptr->resetAccumulation();
                    }
                    ctx.renderer.resetCPUAccumulation();
                }
            }

            point_idx++;
        }
    }

    // Draw mode label (TRANSPARENT - no black background!)
    // AF Area Mode (Top Text)
    const char* mode_labels[] = {"[ SINGLE ]", "[ ZONE 9 ]", "[ ZONE 21 ]", "[ WIDE ]", "[ CENTER ]"};
    const char* area_label = mode_labels[std::min(viewport_settings.af_mode, 4)];
    ImVec2 area_sz = ImGui::CalcTextSize(area_label);
    
    // Position Area Label above
    draw_list->AddText(ImVec2(cx - area_sz.x * 0.5f + 1, start_y - 25.0f + 1), IM_COL32(0, 0, 0, 150), area_label);
    draw_list->AddText(ImVec2(cx - area_sz.x * 0.5f, start_y - 25.0f), IM_COL32(150, 255, 150, 200), area_label);  
    
    // Position: Below Exposure Triangle (Right Side)
    // -------------------------------------------------------------
    // FOCUS MODE SLIDER (MF | AF-S | AF-C)
    // -------------------------------------------------------------
    

    // Re-added definitions as they seemed to be missing
    float exp_tri_cx = io.DisplaySize.x - 30.0f - (85.0f * 0.7f);
    float exp_tri_cy = io.DisplaySize.y * 0.30f;
    
    // Dimensions
    float slider_w = 110.0f; 
    float slider_h = 22.0f;
    float slider_x = exp_tri_cx - slider_w * 0.5f;
    float slider_y = exp_tri_cy + 100.0f; 
    
    // Interaction Area
    ImVec2 area_min(slider_x, slider_y);
    ImVec2 area_max(slider_x + slider_w, slider_y + slider_h);
    
    // Mouse Interaction
    bool is_hovering = (io.MousePos.x >= area_min.x && io.MousePos.x <= area_max.x &&
                        io.MousePos.y >= area_min.y && io.MousePos.y <= area_max.y);
    
    float segment_w = slider_w / 3.0f;

    if (is_hovering) {
        hud_captured_mouse = true; // Lock viewport clicking behind
        
        // Handle Click Directly
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            float rel_x = io.MousePos.x - slider_x;
            int clicked_mode = (int)(rel_x / segment_w);
            clicked_mode = std::max(0, std::min(clicked_mode, 2));

            viewport_settings.focus_mode = clicked_mode;
            if (clicked_mode > 0) ctx.renderer.resetCPUAccumulation();
        }
    }
    
    // Labels
    const char* labels[] = { "MF", "AF-S", "AF-C" };

    // Draw Labels (Active one highlighted)
    for (int i = 0; i < 3; i++) {
        float seg_x = slider_x + i * segment_w;
        ImVec2 txt_sz = ImGui::CalcTextSize(labels[i]);
        float txt_x = seg_x + (segment_w - txt_sz.x) * 0.5f;
        float txt_y = slider_y + (slider_h - txt_sz.y) * 0.5f;
        
        bool is_active = (viewport_settings.focus_mode == i);
        
        // Active: Bright Green, Inactive: Dim Gray
        ImU32 txt_col = is_active ? IM_COL32(100, 255, 100, 255) : IM_COL32(150, 150, 150, 150);
        
        // Add subtle shadow for legibility
        draw_list->AddText(ImVec2(txt_x + 1, txt_y + 1), IM_COL32(0, 0, 0, 150), labels[i]);
        draw_list->AddText(ImVec2(txt_x, txt_y), txt_col, labels[i]);
        
        // Minimal separators
        if (i < 2) {
            draw_list->AddLine(
                ImVec2(seg_x + segment_w, slider_y + 6), 
                ImVec2(seg_x + segment_w, slider_y + slider_h - 6), 
                IM_COL32(255, 255, 255, 30));
        }
    }
}

// =============================================================================
// PRO CAMERA SETTINGS PANEL (Stub - settings via PRO button popup)
// =============================================================================
void SceneUI::drawProCameraPanel(UIContext& ctx) {
    (void)ctx;  // Settings accessed via viewport controls PRO button popup
}

// Continuous Autofocus Update (Called independently of draw)
void SceneUI::updateAutofocus(UIContext& ctx) {
    if (!ctx.scene.camera || !ctx.scene.bvh) return;
    
    // Only active if Focus Mode is AF-C (Continuous)
    if (viewport_settings.focus_mode != 2) return;

    Camera& cam = *ctx.scene.camera;
    
    // Raycast from center (NDC 0,0)
    float ndc_x = 0.0f; 
    float ndc_y = 0.0f;
    
    float half_height = tanf(cam.vfov * 0.5f * 3.14159f / 180.0f);
    float half_width = cam.aspect_ratio * half_height;
    
    Vec3 ray_dir = cam.w * (-1.0f) + cam.u * (ndc_x * half_width) + cam.v * (ndc_y * half_height);
    ray_dir = ray_dir.normalize();
    
    Ray ray(cam.lookfrom, ray_dir);
    HitRecord rec;
    
    // Use large t_max
    if (ctx.scene.bvh->hit(ray, 0.001f, 10000.0f, rec)) {
        if (std::abs(cam.focus_dist - rec.t) > 0.05f) { 
            cam.focus_dist = rec.t;
            
            if (ctx.backend_ptr) {
                ctx.backend_ptr->syncCamera(cam);
                ctx.backend_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
        }
    }
}

