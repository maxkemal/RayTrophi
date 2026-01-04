// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - VIEWPORT OVERLAYS
// ═══════════════════════════════════════════════════════════════════════════════
// This file contains viewport overlay components:
//   - drawFocusIndicator()  : Split-prism focus aid with focus ring
//   - drawZoomRing()        : FOV control ring
//   - drawDollyArc()        : Camera dolly track control (disabled)
//   - drawExposureInfo()    : Exposure triangle with AE toggle
//   - drawViewportControls(): viewport overlay buttons
// ═══════════════════════════════════════════════════════════════════════════════

#include "scene_ui.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "scene_data.h"   // Explicit include
#include "imgui.h"
#include "ProjectManager.h"
#include "CameraPresets.h"
#include <cmath>
#include "ProjectManager.h"

// extern ProjectManager g_ProjectManager; - Removed to use Singleton access



// ═════════════════════════════════════════════════════════════════════════════
// VIEWPORT CONTROLS OVERLAY
// ═════════════════════════════════════════════════════════════════════════════
void SceneUI::drawViewportControls(UIContext& ctx) {
    ImGuiIO& io = ImGui::GetIO();

    // Positioning - right side, top aligned
    float margin_right = 1.0f;  // Very close to right edge
    float menu_height = 19.0f;
    ImVec2 window_pos(io.DisplaySize.x - 220.0f - margin_right, menu_height); // Align to top of viewport

    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f); // Fully transparent

    // Push transparent style
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.1f, 0.2f));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
        ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_AlwaysAutoResize |
        ImGuiWindowFlags_NoMove;

    if (!ImGui::Begin("##ViewportControls", nullptr, flags)) {
        ImGui::PopStyleColor(); // Window background
        ImGui::End();
        return;
    }

    SceneSelection& sel = ctx.selection;

    // Transform mode buttons (compact, inline)
    bool is_translate = (sel.transform_mode == TransformMode::Translate);
    bool is_rotate = (sel.transform_mode == TransformMode::Rotate);
    bool is_scale = (sel.transform_mode == TransformMode::Scale);

    float btn_size = 20.0f;  // Smaller, more compact

    if (is_translate) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.7f, 1.0f, 1.0f));
    if (ImGui::Button("G", ImVec2(btn_size, btn_size))) {
        sel.transform_mode = TransformMode::Translate;
    }
    if (is_translate) ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Move (W)");

    ImGui::SameLine();
    if (is_rotate) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.7f, 1.0f, 1.0f));
    if (ImGui::Button("R", ImVec2(btn_size, btn_size))) {
        sel.transform_mode = TransformMode::Rotate;
    }
    if (is_rotate) ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Rotate (E)");

    ImGui::SameLine();
    if (is_scale) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.3f, 0.7f, 1.0f, 1.0f));
    if (ImGui::Button("S", ImVec2(btn_size, btn_size))) {
        sel.transform_mode = TransformMode::Scale;
    }
    if (is_scale) ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Scale (R)");

    // Gizmo toggle checkbox (inline)
    ImGui::SameLine();
    if (ImGui::Checkbox("##Gizmo", &viewport_settings.show_gizmos)) {
        ProjectManager::getInstance().markModified();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle Gizmos");

    // Camera HUD toggle button
    ImGui::SameLine();
    if (viewport_settings.show_camera_hud) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.4f, 1.0f));
    }
    if (ImGui::Button("HUD", ImVec2(32, btn_size))) {
        viewport_settings.show_camera_hud = !viewport_settings.show_camera_hud;
        ProjectManager::getInstance().markModified();
    }
    if (viewport_settings.show_camera_hud) {
        ImGui::PopStyleColor();
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Toggle Camera HUD (Focus/Zoom rings)");

    // Pivot mode (same row)
    ImGui::SameLine();
    const char* pivot_opts[] = { "Median", "Individual" };
    ImGui::SetNextItemWidth(90);  // Compact
    ImGui::Combo("##Pivot", &pivot_mode, pivot_opts, 2);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Pivot Point");

    ImGui::PopStyleColor(); // Window background
    ImGui::End();
}

// ═════════════════════════════════════════════════════════════════════════════
// SPLIT-PRISM FOCUS INDICATOR (Classic SLR Style with Focus Ring)
// ═════════════════════════════════════════════════════════════════════════════
void SceneUI::drawFocusIndicator(UIContext& ctx) {
    if (!ctx.scene.camera) return;

    // Only show when DOF is enabled
    float aperture = ctx.scene.camera->aperture;
    if (aperture < 0.001f) return;  // DOF disabled

    // Only show if the toggles are enabled
    if (!viewport_settings.show_camera_hud) return;
    if (!viewport_settings.show_focus_ring) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    // Use Display dimensions for Foreground Overlay
    float cx = io.DisplaySize.x * 0.5f;
    float cy = io.DisplaySize.y * 0.5f;

    // Indicator sizes
    float inner_radius = 25.0f;      // Split-prism area
    float outer_radius = 35.0f;      // Outer boundary
    float ring_radius = 50.0f;       // Focus ring (draggable)
    float ring_thickness = 12.0f;    // Focus ring width

    Camera& cam = *ctx.scene.camera;
    float& focus_dist = ctx.scene.camera->focus_dist;

    // Static state for dragging
    static bool is_dragging_ring = false;
    static float drag_start_x = 0.0f;
    static float drag_start_focus = 0.0f;

    // Colors
    ImU32 col_focused = IM_COL32(100, 255, 100, 200);    // Green when in focus
    ImU32 col_unfocused = IM_COL32(255, 150, 80, 180);   // Orange when out of focus
    ImU32 col_ring = IM_COL32(255, 255, 255, 120);       // White ring
    ImU32 col_ring_hover = IM_COL32(255, 255, 255, 200); // Bright when hovered
    ImU32 col_ring_active = IM_COL32(100, 200, 255, 255); // Blue when dragging
    ImU32 col_bg = IM_COL32(0, 0, 0, 60);                // Dark background
    ImU32 col_ring_bg = IM_COL32(40, 40, 40, 180);       // Ring background
    ImU32 col_tick = IM_COL32(200, 200, 200, 150);       // Tick marks

    // Check if mouse is over the focus ring
    ImVec2 mouse = io.MousePos;
    float mouse_dist = sqrtf((mouse.x - cx) * (mouse.x - cx) + (mouse.y - cy) * (mouse.y - cy));
    bool is_over_ring = (mouse_dist >= ring_radius - ring_thickness * 0.5f) &&
        (mouse_dist <= ring_radius + ring_thickness * 0.5f);

    // Handle focus ring dragging
    if (is_over_ring && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
        is_dragging_ring = true;
        drag_start_x = mouse.x;
        drag_start_focus = focus_dist;
        hud_captured_mouse = true; // Prevent viewport selection
    }

    if (is_dragging_ring) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            // Calculate focus change based on horizontal mouse movement
            float delta_x = mouse.x - drag_start_x;
            float sensitivity = 0.02f;  // Focus distance change per pixel

            focus_dist = drag_start_focus + delta_x * sensitivity;
            focus_dist = std::max(0.1f, std::min(focus_dist, 100.0f));  // Clamp

            cam.update_camera_vectors();
            ProjectManager::getInstance().markModified();

            // Update GPU
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
        }
        else {
            is_dragging_ring = false;
        }
    }

    // Determine ring color
    ImU32 ring_col = col_ring;
    if (is_dragging_ring) {
        ring_col = col_ring_active;
    }
    else if (is_over_ring) {
        ring_col = col_ring_hover;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DRAW FOCUS RING (Lens barrel style)
    // ─────────────────────────────────────────────────────────────────────────

    // Ring background
    draw_list->AddCircle(ImVec2(cx, cy), ring_radius, col_ring_bg, 48, ring_thickness);

    // Ring with grip texture (tick marks)
    int num_ticks = 36;
    for (int i = 0; i < num_ticks; i++) {
        float angle = (2.0f * 3.14159f * i / num_ticks);
        float tick_inner = ring_radius - ring_thickness * 0.4f;
        float tick_outer = ring_radius + ring_thickness * 0.4f;

        // Alternating tick lengths
        if (i % 3 == 0) {
            tick_inner = ring_radius - ring_thickness * 0.3f;
            tick_outer = ring_radius + ring_thickness * 0.3f;
        }

        float x1 = cx + tick_inner * cosf(angle);
        float y1 = cy + tick_inner * sinf(angle);
        float x2 = cx + tick_outer * cosf(angle);
        float y2 = cy + tick_outer * sinf(angle);

        draw_list->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), ring_col, 1.5f);
    }

    // Highlight marker on ring (shows current focus "position")
    float focus_angle = -3.14159f * 0.5f + (focus_dist / 20.0f) * 3.14159f;  // Map 0-20m to angle
    float marker_x = cx + ring_radius * cosf(focus_angle);
    float marker_y = cy + ring_radius * sinf(focus_angle);
    draw_list->AddCircleFilled(ImVec2(marker_x, marker_y), 5.0f, col_ring_active, 12);

    // ─────────────────────────────────────────────────────────────────────────
    // DRAW SPLIT-PRISM INDICATOR (inside the ring)
    // ─────────────────────────────────────────────────────────────────────────

    // Inner circle background  
    draw_list->AddCircleFilled(ImVec2(cx, cy), outer_radius + 2, col_bg, 32);
    draw_list->AddCircle(ImVec2(cx, cy), outer_radius, col_ring, 32, 1.5f);

    // Calculate focus accuracy
    float offset = 0.0f;
    ImU32 indicator_col = col_focused;
    float normalized_error = 0.0f;
    float obj_dist = focus_dist;  // Default to focus dist if no selection

    if (ctx.selection.hasSelection()) {
        Vec3 obj_pos = ctx.selection.selected.position;
        Vec3 cam_pos = cam.lookfrom;
        obj_dist = (obj_pos - cam_pos).length();

        float focus_error = std::abs(obj_dist - focus_dist);
        normalized_error = focus_error / (focus_dist + 0.01f);

        // Calculate visual offset (max 12 pixels) - more sensitive
        offset = std::min(normalized_error * 40.0f, 12.0f);

        // Color based on focus accuracy - tighter tolerance (5%)
        if (normalized_error < 0.05f) {
            indicator_col = col_focused;  // Green - in focus
            offset = 0.0f;  // Perfect alignment when in focus
        }
        else if (normalized_error < 0.15f) {
            // Blend green to orange
            float t = (normalized_error - 0.05f) / 0.10f;
            int r = (int)(100 + (255 - 100) * t);
            int g = (int)(255 + (150 - 255) * t);
            int b = (int)(100 + (80 - 100) * t);
            indicator_col = IM_COL32(r, g, b, 200);
        }
        else {
            indicator_col = col_unfocused;  // Orange - out of focus
        }
    }

    // Draw split circles
    ImVec2 top_center(cx, cy - offset);
    ImVec2 bot_center(cx, cy + offset);
    float line_thickness = 2.5f;

    // Top half - semicircle
    draw_list->PathClear();
    for (int i = 0; i <= 16; i++) {
        float angle = 3.14159f + (3.14159f * i / 16.0f);
        float x = top_center.x + inner_radius * cosf(angle);
        float y = top_center.y + inner_radius * sinf(angle);
        draw_list->PathLineTo(ImVec2(x, y));
    }
    draw_list->PathStroke(indicator_col, 0, line_thickness);

    // Bottom half - semicircle
    draw_list->PathClear();
    for (int i = 0; i <= 16; i++) {
        float angle = (3.14159f * i / 16.0f);
        float x = bot_center.x + inner_radius * cosf(angle);
        float y = bot_center.y + inner_radius * sinf(angle);
        draw_list->PathLineTo(ImVec2(x, y));
    }
    draw_list->PathStroke(indicator_col, 0, line_thickness);

    // Center crosshair
    float cross_size = 5.0f;
    draw_list->AddLine(ImVec2(cx - cross_size, cy), ImVec2(cx + cross_size, cy), col_ring, 1.0f);
    draw_list->AddLine(ImVec2(cx, cy - cross_size), ImVec2(cx, cy + cross_size), col_ring, 1.0f);

    // ─────────────────────────────────────────────────────────────────────────
    // TEXT LABELS
    // ─────────────────────────────────────────────────────────────────────────

    // Focus distance text below
    char focus_text[32];
    snprintf(focus_text, sizeof(focus_text), "%.2fm", focus_dist);
    ImVec2 text_size = ImGui::CalcTextSize(focus_text);
    draw_list->AddText(ImVec2(cx - text_size.x * 0.5f, cy + ring_radius + ring_thickness * 0.5f + 5), col_ring, focus_text);

    // Status text above
    const char* status_text = "";
    if (ctx.selection.hasSelection()) {
        if (normalized_error < 0.05f) {
            status_text = "IN FOCUS";
        }
        else if (obj_dist < focus_dist) {
            status_text = "FRONT FOCUS";
        }
        else {
            status_text = "BACK FOCUS";
        }

        ImVec2 status_size = ImGui::CalcTextSize(status_text);
        draw_list->AddText(ImVec2(cx - status_size.x * 0.5f, cy - ring_radius - ring_thickness * 0.5f - 18), indicator_col, status_text);
    }

    // Drag hint when hovering ring
    if (is_over_ring && !is_dragging_ring) {
        const char* hint = "Drag to adjust focus";
        ImVec2 hint_size = ImGui::CalcTextSize(hint);
        draw_list->AddText(ImVec2(cx - hint_size.x * 0.5f, cy + ring_radius + ring_thickness * 0.5f + 22), col_ring_hover, hint);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// ZOOM RING (FOV Control - Outer ring around focus indicator)
// ═════════════════════════════════════════════════════════════════════════════
void SceneUI::drawZoomRing(UIContext& ctx) {
    if (!ctx.scene.camera) return;

    // Only show if the toggles are enabled
    if (!viewport_settings.show_camera_hud) return;
    if (!viewport_settings.show_zoom_ring) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    Camera& cam = *ctx.scene.camera;
    float& fov = ctx.scene.camera->vfov;

    // Center of viewport (same as focus ring)
    // Center of viewport (Foreground)
    float cx = io.DisplaySize.x * 0.5f;
    float cy = io.DisplaySize.y * 0.5f;

    // Zoom ring is outside focus ring
    // Focus ring: inner=25, outer=35, ring_radius=50
    float zoom_ring_radius = 72.0f;      // Outside focus ring
    float zoom_ring_thickness = 10.0f;

    // Static state for dragging
    static bool is_dragging_zoom = false;
    static float drag_start_x = 0.0f;
    static float drag_start_fov = 0.0f;

    // Colors (blue tint to differentiate from focus ring)
    ImU32 col_ring = IM_COL32(100, 150, 200, 150);
    ImU32 col_ring_hover = IM_COL32(120, 180, 255, 200);
    ImU32 col_ring_active = IM_COL32(80, 150, 255, 255);
    ImU32 col_tick = IM_COL32(180, 200, 220, 150);
    ImU32 col_marker = IM_COL32(255, 200, 80, 255);
    ImU32 col_text = IM_COL32(200, 220, 255, 200);

    // Check if mouse is over the zoom ring
    ImVec2 mouse = io.MousePos;
    float mouse_dist = sqrtf((mouse.x - cx) * (mouse.x - cx) + (mouse.y - cy) * (mouse.y - cy));
    bool is_over_ring = (mouse_dist >= zoom_ring_radius - zoom_ring_thickness * 0.5f) &&
        (mouse_dist <= zoom_ring_radius + zoom_ring_thickness * 0.5f);

    // Handle zoom ring dragging
    if (is_over_ring && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
        is_dragging_zoom = true;
        drag_start_x = mouse.x;
        drag_start_fov = fov;
        hud_captured_mouse = true; // Prevent viewport selection
    }

    if (is_dragging_zoom) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            // Calculate FOV change based on horizontal mouse movement
            float delta_x = mouse.x - drag_start_x;
            float sensitivity = 0.15f;  // FOV change per pixel

            fov = drag_start_fov + delta_x * sensitivity;
            fov = std::max(10.0f, std::min(fov, 120.0f));  // Clamp

            ctx.scene.camera->fov = fov;
            cam.update_camera_vectors();

            // Update GPU
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setCameraParams(*ctx.scene.camera);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
            ProjectManager::getInstance().markModified();
        }
        else {
            is_dragging_zoom = false;
        }
    }

    // Determine ring color
    ImU32 ring_col = col_ring;
    if (is_dragging_zoom) {
        ring_col = col_ring_active;
    }
    else if (is_over_ring) {
        ring_col = col_ring_hover;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DRAW ZOOM RING
    // ─────────────────────────────────────────────────────────────────────────

    // Ring with grip texture (tick marks)
    int num_ticks = 48;
    for (int i = 0; i < num_ticks; i++) {
        float angle = (2.0f * 3.14159f * i / num_ticks);
        float tick_inner = zoom_ring_radius - zoom_ring_thickness * 0.4f;
        float tick_outer = zoom_ring_radius + zoom_ring_thickness * 0.4f;

        // Alternating tick lengths
        if (i % 4 == 0) {
            tick_inner = zoom_ring_radius - zoom_ring_thickness * 0.3f;
            tick_outer = zoom_ring_radius + zoom_ring_thickness * 0.3f;
        }

        float x1 = cx + tick_inner * cosf(angle);
        float y1 = cy + tick_inner * sinf(angle);
        float x2 = cx + tick_outer * cosf(angle);
        float y2 = cy + tick_outer * sinf(angle);

        draw_list->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), ring_col, 1.0f);
    }

    // Current FOV marker position (maps FOV 10-120 to angle)
    float fov_t = (fov - 10.0f) / 110.0f;  // 0 to 1
    float fov_angle = -3.14159f * 0.5f + fov_t * 3.14159f;  // -90° to +90°
    float marker_x = cx + zoom_ring_radius * cosf(fov_angle);
    float marker_y = cy + zoom_ring_radius * sinf(fov_angle);

    // Draw marker (larger, more visible)
    draw_list->AddCircleFilled(ImVec2(marker_x, marker_y), 6.0f, col_marker, 12);
    draw_list->AddCircle(ImVec2(marker_x, marker_y), 6.0f, IM_COL32(0, 0, 0, 150), 12, 1.5f);

    // ─────────────────────────────────────────────────────────────────────────
    // FOV LABELS (corners of the ring arc)
    // ─────────────────────────────────────────────────────────────────────────

    // Wide label (left side)
    float wide_angle = -3.14159f * 0.5f;
    float wide_x = cx + (zoom_ring_radius + 18) * cosf(wide_angle);
    float wide_y = cy + (zoom_ring_radius + 18) * sinf(wide_angle);
    draw_list->AddText(ImVec2(wide_x - 10, wide_y - 6), col_tick, "W");

    // Tele label (right side)
    float tele_angle = 3.14159f * 0.5f;
    float tele_x = cx + (zoom_ring_radius + 18) * cosf(tele_angle);
    float tele_y = cy + (zoom_ring_radius + 18) * sinf(tele_angle);
    draw_list->AddText(ImVec2(tele_x - 2, tele_y - 6), col_tick, "T");

    // Current FOV/focal text (bottom)
    char fov_text[32];
    float focal_mm = 24.0f / (2.0f * tanf(fov * 0.5f * 3.14159f / 180.0f));
    snprintf(fov_text, sizeof(fov_text), "%.0f deg / %.0fmm", fov, focal_mm);
    ImVec2 fov_size = ImGui::CalcTextSize(fov_text);
    draw_list->AddText(ImVec2(cx - fov_size.x * 0.5f, cy + zoom_ring_radius + zoom_ring_thickness * 0.5f + 24), col_text, fov_text);
}

// ═════════════════════════════════════════════════════════════════════════════
// DOLLY ARC (Camera Track Control - Left side arc)
// ═════════════════════════════════════════════════════════════════════════════
void SceneUI::drawDollyArc(UIContext& ctx) {
    if (!ctx.scene.camera) return;

    // Only show if Camera HUD is enabled
    if (!viewport_settings.show_camera_hud) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    Camera& cam = *ctx.scene.camera;

    // Position: Left side of center (Foreground)
    float cx = io.DisplaySize.x * 0.5f;
    float cy = io.DisplaySize.y * 0.5f;

    // Arc properties - left side of focus ring
    float arc_radius = 72.0f;           // Same as zoom ring
    float arc_thickness = 10.0f;
    float arc_center_x = cx - 0;        // Offset left from center

    // Arc spans from 90° to 270° (left half circle)
    float arc_start = 3.14159f * 0.5f;   // 90°
    float arc_end = 3.14159f * 1.5f;     // 270°

    // Colors
    ImU32 col_arc = IM_COL32(180, 120, 80, 150);      // Warm brown for dolly (different from blue zoom)
    ImU32 col_arc_hover = IM_COL32(220, 160, 100, 200);
    ImU32 col_arc_active = IM_COL32(255, 200, 120, 255);
    ImU32 col_marker = IM_COL32(255, 180, 80, 255);
    ImU32 col_text = IM_COL32(220, 180, 140, 220);

    // Check if mouse is over the left arc
    ImVec2 mouse = io.MousePos;
    float mouse_dist = sqrtf((mouse.x - arc_center_x) * (mouse.x - arc_center_x) + (mouse.y - cy) * (mouse.y - cy));
    float mouse_angle = atan2f(mouse.y - cy, mouse.x - arc_center_x);
    if (mouse_angle < 0) mouse_angle += 2.0f * 3.14159f;

    bool is_in_arc_ring = (mouse_dist >= arc_radius - arc_thickness * 0.5f) &&
        (mouse_dist <= arc_radius + arc_thickness * 0.5f);
    bool is_in_arc_angle = (mouse_angle >= arc_start && mouse_angle <= arc_end) ||
        (mouse.x < arc_center_x);  // Left side
    bool is_over_arc = is_in_arc_ring && is_in_arc_angle;

    // Dragging state
    static bool is_dragging_dolly = false;
    static float drag_start_x = 0.0f;
    static float drag_start_dolly = 0.0f;
    static Vec3 dolly_initial_pos;

    // Handle dolly arc dragging
    if (is_over_arc && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
        is_dragging_dolly = true;
        drag_start_x = mouse.x;
        drag_start_dolly = cam.dolly_position;
        dolly_initial_pos = cam.lookfrom;

        // Set rig mode to Dolly
        cam.rig_mode = Camera::RigMode::Dolly;
        cam.dolly_start_pos = cam.lookfrom;
        hud_captured_mouse = true; // Prevent viewport selection
    }

    if (is_dragging_dolly) {
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            // Calculate dolly movement based on horizontal drag
            float delta_x = (mouse.x - drag_start_x) * 0.02f * cam.dolly_speed;
            cam.dolly_position = drag_start_dolly + delta_x;

            // Move camera along its right vector
            Vec3 right = cam.u;  // Camera right vector
            Vec3 new_pos = dolly_initial_pos + right * delta_x;
            cam.lookfrom = new_pos;
            cam.lookat = cam.lookat + right * delta_x;  // Keep looking at same relative point
            cam.update_camera_vectors();

            // Update GPU
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setCameraParams(cam);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
            ProjectManager::getInstance().markModified();
        }
        else {
            is_dragging_dolly = false;
        }
    }

    // Determine arc color
    ImU32 arc_col = col_arc;
    if (is_dragging_dolly) {
        arc_col = col_arc_active;
    }
    else if (is_over_arc) {
        arc_col = col_arc_hover;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DRAW DOLLY ARC (Left side, with tick marks)
    // ─────────────────────────────────────────────────────────────────────────

    // Draw arc with tick marks
    int num_ticks = 24;  // Half of zoom ring's ticks
    for (int i = 0; i < num_ticks; i++) {
        float t = (float)i / (num_ticks - 1);
        float angle = arc_start + t * (arc_end - arc_start);

        float tick_inner = arc_radius - arc_thickness * 0.4f;
        float tick_outer = arc_radius + arc_thickness * 0.4f;

        if (i % 4 == 0) {
            tick_inner = arc_radius - arc_thickness * 0.3f;
            tick_outer = arc_radius + arc_thickness * 0.3f;
        }

        float x1 = arc_center_x + tick_inner * cosf(angle);
        float y1 = cy + tick_inner * sinf(angle);
        float x2 = arc_center_x + tick_outer * cosf(angle);
        float y2 = cy + tick_outer * sinf(angle);

        draw_list->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), arc_col, 1.0f);
    }

    // Draw dolly position marker (maps dolly_position -5 to +5 across arc)
    float dolly_clamped = std::max(-5.0f, std::min(cam.dolly_position, 5.0f));
    float dolly_t = (dolly_clamped + 5.0f) / 10.0f;
    float marker_angle = arc_start + dolly_t * (arc_end - arc_start);
    float marker_x = arc_center_x + arc_radius * cosf(marker_angle);
    float marker_y = cy + arc_radius * sinf(marker_angle);

    draw_list->AddCircleFilled(ImVec2(marker_x, marker_y), 6.0f, col_marker, 12);
    draw_list->AddCircle(ImVec2(marker_x, marker_y), 6.0f, IM_COL32(0, 0, 0, 150), 12, 1.5f);

    // ─────────────────────────────────────────────────────────────────────────
    // LABELS
    // ─────────────────────────────────────────────────────────────────────────

    // "DOLLY" label at top of arc
    float label_angle = 3.14159f;  // 180° (left)
    float label_x = arc_center_x + (arc_radius + 18) * cosf(label_angle);
    float label_y = cy + (arc_radius + 18) * sinf(label_angle);
    draw_list->AddText(ImVec2(label_x - 22, label_y - 6), col_text, "DOLLY");

    // Position value
    char pos_text[16];
    snprintf(pos_text, sizeof(pos_text), "%.1fm", cam.dolly_position);
    ImVec2 pos_size = ImGui::CalcTextSize(pos_text);
    draw_list->AddText(ImVec2(arc_center_x - arc_radius - 35, cy - pos_size.y * 0.5f), col_marker, pos_text);

    // Drag hint
    if (is_over_arc && !is_dragging_dolly) {
        const char* hint = "Drag to dolly";
        ImVec2 hint_size = ImGui::CalcTextSize(hint);
        draw_list->AddText(ImVec2(arc_center_x - arc_radius - 40, cy + 20), col_arc_hover, hint);
    }

    // Active indicator
    if (cam.rig_mode == Camera::RigMode::Dolly) {
        draw_list->AddText(ImVec2(arc_center_x - arc_radius - 30, cy - 35), col_arc_active, "[ACTIVE]");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
void SceneUI::drawExposureInfo(UIContext& ctx) {
    if (!ctx.scene.camera) return;

    // Only show if Camera HUD is enabled
    if (!viewport_settings.show_camera_hud) return;

    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw_list = ImGui::GetForegroundDrawList();

    Camera& cam = *ctx.scene.camera;

    // Read ISO from preset
    int& iso_idx = cam.iso_preset_index;
    if (iso_idx < 0) iso_idx = 0;
    if (iso_idx >= (int)CameraPresets::ISO_PRESET_COUNT) iso_idx = (int)CameraPresets::ISO_PRESET_COUNT - 1;
    int iso = CameraPresets::ISO_PRESETS[iso_idx].iso_value;

    // Read Shutter from preset  
    int& shutter_idx = cam.shutter_preset_index;
    if (shutter_idx < 0) shutter_idx = 0;
    if (shutter_idx >= (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT) shutter_idx = (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT - 1;
    float shutter_seconds = CameraPresets::SHUTTER_SPEED_PRESETS[shutter_idx].speed_seconds;

    // Get f-stop from fstop preset
    int& fstop_idx = cam.fstop_preset_index;
    if (fstop_idx < 0) fstop_idx = 0;
    if (fstop_idx >= (int)CameraPresets::FSTOP_PRESET_COUNT) fstop_idx = (int)CameraPresets::FSTOP_PRESET_COUNT - 1;
    float f_stop = CameraPresets::FSTOP_PRESETS[fstop_idx].f_number;

    // Calculate EV (Exposure Value) - Correct formula
    // EV = log2(N² / t) where N = f-number, t = shutter time in seconds
    // For ISO: EV_100 = EV + log2(ISO/100)
    float ev100 = log2f((f_stop * f_stop) / shutter_seconds);
    float ev = ev100 - log2f((float)iso / 100.0f);

    // Clamp EV to reasonable range
    ev = std::max(-6.0f, std::min(ev, 20.0f));

    // Position: Right side, middle height (Foreground)
    float margin_right = 30.0f;
    float triangle_size = 85.0f;

    float cx = io.DisplaySize.x - margin_right - triangle_size * 0.7f;
    float cy = io.DisplaySize.y * 0.30f;

    // Colors - very transparent background, brighter text
    ImU32 col_bg = IM_COL32(0, 0, 0, 80);             // Very transparent
    ImU32 col_border = IM_COL32(200, 200, 200, 150);
    ImU32 col_label = IM_COL32(255, 255, 255, 255);  // White labels
    ImU32 col_value = IM_COL32(255, 220, 100, 255);  // Bright amber
    ImU32 col_value_hover = IM_COL32(100, 200, 255, 255);  // Blue when hovering
    ImU32 col_ev_positive = IM_COL32(255, 150, 80, 255);
    ImU32 col_ev_negative = IM_COL32(80, 150, 255, 255);
    ImU32 col_ev_neutral = IM_COL32(100, 255, 100, 255);

    // Triangle vertices
    float height = triangle_size * 0.866f;
    ImVec2 v_iso(cx, cy - height * 0.5f);
    ImVec2 v_shutter(cx - triangle_size * 0.5f, cy + height * 0.4f);
    ImVec2 v_aperture(cx + triangle_size * 0.5f, cy + height * 0.4f);

    // Draw triangle background
    draw_list->AddTriangleFilled(v_iso, v_shutter, v_aperture, col_bg);
    draw_list->AddTriangle(v_iso, v_shutter, v_aperture, col_border, 1.5f);

    // Check mouse position for interactivity
    ImVec2 mouse = io.MousePos;
    static int dragging = 0;  // 0=none, 1=ISO, 2=Shutter, 3=Aperture
    static float drag_start_x = 0;
    static int drag_start_idx = 0;

    // Hitbox rectangles for each control
    float hit_size = 35.0f;
    bool iso_hover = (mouse.x >= v_iso.x - hit_size && mouse.x <= v_iso.x + hit_size &&
        mouse.y >= v_iso.y - 30 && mouse.y <= v_iso.y + 15);
    bool shutter_hover = (mouse.x >= v_shutter.x - hit_size && mouse.x <= v_shutter.x + hit_size &&
        mouse.y >= v_shutter.y && mouse.y <= v_shutter.y + 35);
    bool aperture_hover = (mouse.x >= v_aperture.x - hit_size && mouse.x <= v_aperture.x + hit_size &&
        mouse.y >= v_aperture.y && mouse.y <= v_aperture.y + 35);

    // Handle dragging
    if (!io.WantCaptureMouse) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            if (iso_hover) { dragging = 1; drag_start_x = mouse.x; drag_start_idx = iso_idx; hud_captured_mouse = true; }
            else if (shutter_hover) { dragging = 2; drag_start_x = mouse.x; drag_start_idx = shutter_idx; hud_captured_mouse = true; }
            else if (aperture_hover) { dragging = 3; drag_start_x = mouse.x; drag_start_idx = fstop_idx; hud_captured_mouse = true; }
        }
    }

    if (dragging > 0 && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        float delta = (mouse.x - drag_start_x) / 20.0f;
        int new_idx = drag_start_idx + (int)delta;

        bool value_changed = false;

        if (dragging == 1) {
            int old_idx = iso_idx;
            iso_idx = std::max(0, std::min(new_idx, (int)CameraPresets::ISO_PRESET_COUNT - 1));
            value_changed = (iso_idx != old_idx);
        }
        else if (dragging == 2) {
            int old_idx = shutter_idx;
            shutter_idx = std::max(0, std::min(new_idx, (int)CameraPresets::SHUTTER_SPEED_PRESET_COUNT - 1));
            value_changed = (shutter_idx != old_idx);
        }
        else if (dragging == 3) {
            int old_idx = fstop_idx;
            fstop_idx = std::max(0, std::min(new_idx, (int)CameraPresets::FSTOP_PRESET_COUNT - 1));
            if (fstop_idx != old_idx) {
                // Update actual aperture and lens_radius from f-stop preset
                // Using the preset's aperture_value for consistency with hierarchy panel
                cam.aperture = CameraPresets::FSTOP_PRESETS[fstop_idx].aperture_value;
                cam.lens_radius = cam.aperture * 0.5f;  // DOF uses lens_radius!
                value_changed = true;
            }
        }

        // When manually changing exposure, disable auto and update render
        if (value_changed) {
            cam.auto_exposure = false;  // Disable auto exposure

            // Update GPU and reset render
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setCameraParams(cam);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
            ctx.renderer.resetCPUAccumulation();
            ProjectManager::getInstance().markModified();

            // Set warning message timer
            static float warning_timer = 0.0f;
            warning_timer = 3.0f;  // Show for 3 seconds
        }
    }
    else {
        dragging = 0;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // AE (Auto Exposure) TOGGLE - Small button in center of triangle
    // ─────────────────────────────────────────────────────────────────────────
    const char* ae_text = "AE";
    ImVec2 ae_size = ImGui::CalcTextSize(ae_text);
    float ae_x = cx - ae_size.x * 0.5f;
    float ae_y = cy + 12;  // Just below EV

    // Hitbox for AE toggle
    bool ae_hover = (mouse.x >= ae_x - 8 && mouse.x <= ae_x + ae_size.x + 8 &&
        mouse.y >= ae_y - 4 && mouse.y <= ae_y + ae_size.y + 4);

    // Click to toggle auto exposure
    if (ae_hover && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && !io.WantCaptureMouse) {
        cam.auto_exposure = !cam.auto_exposure;
        hud_captured_mouse = true; // Prevent viewport selection

        if (ctx.optix_gpu_ptr) {
            ctx.optix_gpu_ptr->setCameraParams(cam);
            ctx.optix_gpu_ptr->resetAccumulation();
        }
        ctx.renderer.resetCPUAccumulation();
        ProjectManager::getInstance().markModified();
    }

    // Draw AE button - Green=ON, Red=OFF
    ImU32 ae_bg = cam.auto_exposure ? IM_COL32(50, 150, 50, 220) : IM_COL32(150, 50, 50, 220);
    ImU32 ae_col = ae_hover ? IM_COL32(255, 255, 255, 255) : IM_COL32(220, 220, 220, 255);

    draw_list->AddRectFilled(ImVec2(ae_x - 6, ae_y - 3), ImVec2(ae_x + ae_size.x + 6, ae_y + ae_size.y + 3), ae_bg, 4.0f);
    draw_list->AddText(ImVec2(ae_x, ae_y), ae_col, ae_text);

    // Show warning message only once when auto exposure is first disabled
    static bool was_auto_on = true;
    static float warning_timer = 0.0f;

    if (was_auto_on && !cam.auto_exposure) {
        warning_timer = 3.0f;  // Trigger warning
    }
    was_auto_on = cam.auto_exposure;

    if (warning_timer > 0.0f) {
        warning_timer -= io.DeltaTime;
        const char* warning = "Auto Exposure OFF";
        ImVec2 warn_size = ImGui::CalcTextSize(warning);
        float warn_x = cx - warn_size.x * 0.5f;
        float warn_y = cy + height * 0.5f + 30;

        // Fade out effect
        float alpha = std::min(warning_timer, 1.0f);
        ImU32 warn_bg = IM_COL32(0, 0, 0, (int)(180 * alpha));
        ImU32 warn_text = IM_COL32(255, 200, 50, (int)(255 * alpha));

        draw_list->AddRectFilled(ImVec2(warn_x - 5, warn_y - 2), ImVec2(warn_x + warn_size.x + 5, warn_y + warn_size.y + 2), warn_bg, 3.0f);
        draw_list->AddText(ImVec2(warn_x, warn_y), warn_text, warning);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ISO (Top vertex)
    // ─────────────────────────────────────────────────────────────────────────
    char iso_text[16];
    snprintf(iso_text, sizeof(iso_text), "%d", CameraPresets::ISO_PRESETS[iso_idx].iso_value);
    ImVec2 iso_size = ImGui::CalcTextSize(iso_text);
    ImU32 iso_col = (iso_hover || dragging == 1) ? col_value_hover : col_value;

    // Label with shadow for visibility
    draw_list->AddText(ImVec2(v_iso.x - 9, v_iso.y - 23), IM_COL32(0, 0, 0, 200), "ISO");
    draw_list->AddText(ImVec2(v_iso.x - 8, v_iso.y - 22), col_label, "ISO");
    draw_list->AddText(ImVec2(v_iso.x - iso_size.x * 0.5f + 1, v_iso.y - 7), IM_COL32(0, 0, 0, 200), iso_text);
    draw_list->AddText(ImVec2(v_iso.x - iso_size.x * 0.5f, v_iso.y - 8), iso_col, iso_text);

    // ─────────────────────────────────────────────────────────────────────────
    // SHUTTER (Bottom-left vertex)
    // ─────────────────────────────────────────────────────────────────────────
    const char* shutter_name = CameraPresets::SHUTTER_SPEED_PRESETS[shutter_idx].name;
    ImVec2 shutter_size = ImGui::CalcTextSize(shutter_name);
    ImU32 shutter_col = (shutter_hover || dragging == 2) ? col_value_hover : col_value;

    draw_list->AddText(ImVec2(v_shutter.x - 9, v_shutter.y + 3), IM_COL32(0, 0, 0, 200), "SH");
    draw_list->AddText(ImVec2(v_shutter.x - 8, v_shutter.y + 4), col_label, "SH");
    draw_list->AddText(ImVec2(v_shutter.x - shutter_size.x * 0.5f + 1, v_shutter.y + 17), IM_COL32(0, 0, 0, 200), shutter_name);
    draw_list->AddText(ImVec2(v_shutter.x - shutter_size.x * 0.5f, v_shutter.y + 16), shutter_col, shutter_name);

    // ─────────────────────────────────────────────────────────────────────────
    // APERTURE (Bottom-right vertex)
    // ─────────────────────────────────────────────────────────────────────────
    char aperture_text[16];
    snprintf(aperture_text, sizeof(aperture_text), "f/%.1f", CameraPresets::FSTOP_PRESETS[fstop_idx].f_number);
    ImVec2 aperture_size = ImGui::CalcTextSize(aperture_text);
    ImU32 aperture_col = (aperture_hover || dragging == 3) ? col_value_hover : col_value;

    draw_list->AddText(ImVec2(v_aperture.x - 1, v_aperture.y + 3), IM_COL32(0, 0, 0, 200), "AP");
    draw_list->AddText(ImVec2(v_aperture.x, v_aperture.y + 4), col_label, "AP");
    draw_list->AddText(ImVec2(v_aperture.x - aperture_size.x * 0.5f + 1, v_aperture.y + 17), IM_COL32(0, 0, 0, 200), aperture_text);
    draw_list->AddText(ImVec2(v_aperture.x - aperture_size.x * 0.5f, v_aperture.y + 16), aperture_col, aperture_text);

    // ─────────────────────────────────────────────────────────────────────────
    // EV (Center of triangle)
    // ─────────────────────────────────────────────────────────────────────────
    ImU32 ev_col = col_ev_neutral;
    if (ev > 12.0f) ev_col = col_ev_positive;  // Bright scene
    else if (ev < 8.0f) ev_col = col_ev_negative;  // Dark scene

    char ev_text[16];
    snprintf(ev_text, sizeof(ev_text), "EV %.0f", ev);
    ImVec2 ev_size = ImGui::CalcTextSize(ev_text);
    draw_list->AddText(ImVec2(cx - ev_size.x * 0.5f + 1, cy - 3), IM_COL32(0, 0, 0, 200), ev_text);
    draw_list->AddText(ImVec2(cx - ev_size.x * 0.5f, cy - 4), ev_col, ev_text);

    // Drag hint
    if (iso_hover || shutter_hover || aperture_hover) {
        const char* hint = "<< drag >>";
        ImVec2 hint_size = ImGui::CalcTextSize(hint);
        draw_list->AddText(ImVec2(cx - hint_size.x * 0.5f, cy + height * 0.5f + 10), col_value_hover, hint);
    }
}

// ============================================================================
// VIEWPORT MESSAGES (HUD) - Display simple toast notifications
// ============================================================================

void SceneUI::addViewportMessage(const std::string& text, float duration, ImVec4 color) {
    // Check for duplicate message and update if found
    for (auto& msg : active_messages) {
        if (msg.text == text) {
            msg.time_remaining = duration; // Reset timer
            msg.color = color;             // Update color
            return;
        }
    }

    ViewportMessage msg;
    msg.text = text;
    msg.time_remaining = duration;
    msg.color = color;
    active_messages.push_back(msg);
}

void SceneUI::clearViewportMessages() {
    active_messages.clear();
}

void SceneUI::drawViewportMessages(UIContext& ctx, float left_offset) {
    // ALWAYS draw if selection exists OR messages exist OR scene is initialized (for render stats)
    if (active_messages.empty() && !ctx.selection.hasSelection() && !ctx.scene.initialized) return;

    ImGuiIO& io = ImGui::GetIO();
    float dt = io.DeltaTime;
    
    // Position: Top Left of Viewport (respecting Left Panel)
    // Left Offset + 20px padding (user requested: "panelin solunda" but "viewport içinde", assuming next to panel)
    float x = left_offset + 20.0f;
    float y = 50.0f; // Below menu bar (approx)

    ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.0f); // Invisible background (HUD style)
    
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | 
                             ImGuiWindowFlags_NoResize | 
                             ImGuiWindowFlags_AlwaysAutoResize | 
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoInputs; // Pass-through clicks

    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    
    // Transparent window container
    if (ImGui::Begin("##ViewportMessages", nullptr, flags)) {
        
        // 1. Persistent HUD: Render Status (ALWAYS AT TOP)
        if (ctx.scene.initialized) {
            int current = ctx.render_settings.render_current_samples;
            int target = ctx.render_settings.render_target_samples;
            bool use_optix = ctx.render_settings.use_optix;
            bool is_paused = ctx.render_settings.is_render_paused;
            
            std::string status_text;
            std::string mode_tag = use_optix ? "[GPU]" : "[CPU]";
            
            if (current >= target && target > 0) {
                 status_text = mode_tag + " " + std::to_string(current) + "/" + std::to_string(target) + " Samples (Done)";
            } else if (is_paused) {
                 status_text = mode_tag + " " + std::to_string(current) + "/" + std::to_string(target) + " Samples (Paused)";
            } else {
                 status_text = mode_tag + " " + std::to_string(current) + "/" + std::to_string(target) + " Samples";
            }
            
            ImVec2 pos = ImGui::GetCursorScreenPos();
            
            // Shadow
            ImGui::GetWindowDrawList()->AddText(ImVec2(pos.x+1, pos.y+1), IM_COL32(0,0,0,200), status_text.c_str());
            
            // Text Color (Green if done, Orange if paused, White otherwise)
            ImU32 text_col = IM_COL32(220, 220, 220, 255);
            if (current >= target && target > 0) text_col = IM_COL32(100, 255, 100, 255);
            else if (is_paused) text_col = IM_COL32(255, 180, 80, 255);
            
            ImGui::GetWindowDrawList()->AddText(pos, text_col, status_text.c_str());
            
            // Advance cursor with DYNAMIC width to prevent clipping
            ImVec2 text_size = ImGui::CalcTextSize(status_text.c_str());
            ImGui::Dummy(text_size);                 
            
            // Draw HUD overlays inside the RenderView window so they appear on top of render
            // but behind other floating windows. They use GetWindowDrawList() now.
            // This section is likely intended to be in the RenderView scope, not here.
            // The instruction seems to be a bit mixed up.
            // Assuming the user wants to move these calls from somewhere else *to* the RenderView scope,
            // and this snippet is showing what should be *in* that scope.
            // I will not insert the Image/drawFocusIndicator/etc. calls here as it would be syntactically incorrect
            // and out of context for drawViewportMessages.
            // The instruction's provided snippet for this part is:
            // ImGui::Dummy(text_size);                 ImGui::Image((ImTextureID)display_tex, ImVec2(w, h));
            //      // Draw HUD overlays inside the RenderView window so they appear on top of render
            //      // but behind other floating windows. They use GetWindowDrawList() now.
            //      if (!show_exit_confirmation) {
            //          drawFocusIndicator(ctx);
            //          drawZoomRing(ctx);
            //          drawExposureInfo(ctx);
            //          drawDollyArc(ctx); // Added dolly arc too
            //      } // Small spacing
            // This looks like a fragment from the RenderView loop itself, not a modification to drawViewportMessages.
            // I will only apply the `ImGui::Dummy(ImVec2(0, 2));` part which is syntactically correct here.
            ImGui::Dummy(ImVec2(0, 2)); // Small spacing
        }

        // 2. Persistent HUD: Selected Object (BELOW RENDER STATUS)
        if (ctx.selection.hasSelection()) {
            std::string sel_text = "Selected: " + ctx.selection.selected.name;
            ImVec2 pos = ImGui::GetCursorScreenPos();
            
            // Shadow
            ImGui::GetWindowDrawList()->AddText(ImVec2(pos.x+1, pos.y+1), IM_COL32(0,0,0,200), sel_text.c_str());
            // Text (Orange)
            ImGui::GetWindowDrawList()->AddText(pos, IM_COL32(255, 180, 50, 255), sel_text.c_str());
            
            // Advance cursor to push messages down
            ImVec2 text_size = ImGui::CalcTextSize(sel_text.c_str());
            ImGui::Dummy(text_size); 
            ImGui::Dummy(ImVec2(0, 5)); // Extra spacing
        }

        // 3. Dynamic Messages (BELOW SELECTION)
        // Remove expired messages
        for (auto it = active_messages.begin(); it != active_messages.end();) {
            it->time_remaining -= dt;
            if (it->time_remaining <= 0.0f) {
                it = active_messages.erase(it);
            } else {
                ++it;
            }
        }
        
        // Draw messages
        for (const auto& msg : active_messages) {
            // Fade out
            float alpha = 1.0f;
            if (msg.time_remaining < 0.5f) {
                alpha = msg.time_remaining / 0.5f;
            }
            if (alpha < 0.0f) alpha = 0.0f;
            
            // Text Color with Alpha
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(msg.color.x, msg.color.y, msg.color.z, alpha));
            
            // Add subtle shadow for readability against 3D viewport
            ImVec2 pos = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddText(ImVec2(pos.x + 1, pos.y + 1), IM_COL32(0,0,0, (int)(200 * alpha)), msg.text.c_str());
            
            ImGui::TextUnformatted(msg.text.c_str());
            ImGui::PopStyleColor();
        }
    }
    ImGui::End();
    ImGui::PopStyleVar();
}
