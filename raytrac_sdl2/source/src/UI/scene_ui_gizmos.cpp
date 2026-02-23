// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - GIZMOS & TRANSFORM
// ═══════════════════════════════════════════════════════════════════════════════
// This file handles 3D Gizmos (Move/Rotate/Scale), Bounding Boxes, and overlays.
// ═══════════════════════════════════════════════════════════════════════════════

#include "scene_ui.h"
#include "renderer.h"
#include "OptixWrapper.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "imgui.h"
#include "ImGuizmo.h"
#include "scene_data.h"
#include "ProjectManager.h"
#include "VDBVolumeManager.h"
#include "GasVolume.h"  // For gas simulation gizmos
#include "scene_ui_gas.hpp"  // For GasUI::selected_gas_volume
#include "scene_ui_forcefield.hpp"
// ═════════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════════
// SELECTION BOUNDING BOX DRAWING (Multi-selection support)
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawSelectionBoundingBox(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;
    Camera& cam = *ctx.scene.camera;
    ImGuiIO& io = ImGui::GetIO();
    float screen_w = io.DisplaySize.x;
    float screen_h = io.DisplaySize.y;

    // Camera basis vectors
    Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();

    // FOV calculations
    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);

    // Helper lambda to draw a bounding box with granular occlusion
    auto DrawBoundingBox = [&](Vec3 bb_min, Vec3 bb_max, ImU32 color, float thickness) {
        Vec3 corners[8] = {
            Vec3(bb_min.x, bb_min.y, bb_min.z),
            Vec3(bb_max.x, bb_min.y, bb_min.z),
            Vec3(bb_max.x, bb_max.y, bb_min.z),
            Vec3(bb_min.x, bb_max.y, bb_min.z),
            Vec3(bb_min.x, bb_min.y, bb_max.z),
            Vec3(bb_max.x, bb_min.y, bb_max.z),
            Vec3(bb_max.x, bb_max.y, bb_max.z),
            Vec3(bb_min.x, bb_max.y, bb_max.z),
        };

        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
        
        // Helper to draw a line with subdivision and occlusion check
        auto DrawSegmentedLine = [&](const Vec3& p_start, const Vec3& p_end) {
            const int segments = 8; // Subdivision level for accurate occlusion
            Vec3 prev_p = p_start;
            ImVec2 prev_scr;
            
            // Project start point
            bool prev_vis = false;
            {
                Vec3 to_pt = prev_p - cam.lookfrom;
                float depth = to_pt.dot(cam_forward);
                if (depth > 0.01f) {
                    float local_x = to_pt.dot(cam_right);
                    float local_y = to_pt.dot(cam_up);
                    float half_h = depth * tan_half_fov;
                    float half_w = half_h * aspect_ratio;
                    prev_scr.x = ((local_x / half_w) * 0.5f + 0.5f) * screen_w;
                    prev_scr.y = (0.5f - (local_y / half_h) * 0.5f) * screen_h;
                    prev_vis = true;
                }
            }

            for (int i = 1; i <= segments; ++i) {
                float t = (float)i / (float)segments;
                Vec3 curr_p = p_start * (1.0f - t) + p_end * t;
                
                ImVec2 curr_scr;
                bool curr_vis = false;
                
                // Project current point
                Vec3 to_pt = curr_p - cam.lookfrom;
                float depth = to_pt.dot(cam_forward);
                if (depth > 0.01f) {
                    float local_x = to_pt.dot(cam_right);
                    float local_y = to_pt.dot(cam_up);
                    float half_h = depth * tan_half_fov;
                    float half_w = half_h * aspect_ratio;
                    curr_scr.x = ((local_x / half_w) * 0.5f + 0.5f) * screen_w;
                    curr_scr.y = (0.5f - (local_y / half_h) * 0.5f) * screen_h;
                    curr_vis = true;
                }

                if (prev_vis && curr_vis) {
                    // Check visibility of the segment midpoint
                    Vec3 mid_p = (prev_p + curr_p) * 0.5f;
                    ImU32 segment_color = color;
                    
                    extern bool g_bvh_rebuild_pending;
                    if (ctx.scene.bvh && !g_bvh_rebuild_pending) {
                        Vec3 to_mid = mid_p - cam.lookfrom;
                        float dist = to_mid.length();
                        if (dist > 0.1f) {
                            Ray r(cam.lookfrom, to_mid / dist);
                            HitRecord rec;
                            // Check occlusion: if hit anything closer than the segment
                            if (ctx.scene.bvh->hit(r, 0.001f, dist - 0.05f, rec, true)) {
                                // Occluded: Fade out to 20%
                                int alpha = (color >> 24) & 0xFF;
                                alpha = alpha / 5;
                                if (alpha < 30) alpha = 30; // Minimum visibility
                                segment_color = (color & 0x00FFFFFF) | (alpha << 24);
                            }
                        }
                    }
                    
                    draw_list->AddLine(prev_scr, curr_scr, segment_color, thickness);
                }

                prev_p = curr_p;
                prev_scr = curr_scr;
                prev_vis = curr_vis;
            }
        };

        // Draw 12 edges of the box
        DrawSegmentedLine(corners[0], corners[1]);
        DrawSegmentedLine(corners[1], corners[2]);
        DrawSegmentedLine(corners[2], corners[3]);
        DrawSegmentedLine(corners[3], corners[0]);

        DrawSegmentedLine(corners[4], corners[5]);
        DrawSegmentedLine(corners[5], corners[6]);
        DrawSegmentedLine(corners[6], corners[7]);
        DrawSegmentedLine(corners[7], corners[4]);

        DrawSegmentedLine(corners[0], corners[4]);
        DrawSegmentedLine(corners[1], corners[5]);
        DrawSegmentedLine(corners[2], corners[6]);
        DrawSegmentedLine(corners[3], corners[7]);
    };

    // ─────────────────────────────────────────────────────────────────────────
    // DRAW VDB GHOST BOUNDS (Always visible)
    // ─────────────────────────────────────────────────────────────────────────
    for (const auto& vdb : ctx.scene.vdb_volumes) {
        // Skip if this VDB is currently selected (will be drawn with highlight)
        if (sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume == vdb) continue;
        
        AABB bounds = vdb->getWorldBounds();
        // Ghost outline - visible but not obtrusive (Alpha 100)
        DrawBoundingBox(bounds.min, bounds.max, IM_COL32(180, 180, 180, 100), 1.0f);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DRAW GAS VOLUME BOUNDS (Smoke/Fire Simulation) - With Picking Support
    // ─────────────────────────────────────────────────────────────────────────
    for (const auto& gas : ctx.scene.gas_volumes) {
        if (!gas || !gas->visible) continue;
        
        Vec3 bounds_min, bounds_max;
        gas->getWorldBounds(bounds_min, bounds_max);
        
        // Ray-AABB Intersection for Picking
        // ----------------------------------------------------
        // 1. Generate Ray from Camera through Mouse position
        float mouse_x = io.MousePos.x;
        float mouse_y = io.MousePos.y;
        
        // Convert screen space to NDC (-1 to 1)
        float ndc_x = (mouse_x / screen_w) * 2.0f - 1.0f;
        float ndc_y = 1.0f - (mouse_y / screen_h) * 2.0f; // Flip Y
        
        // Ray direction in camera space
        float view_h = tan_half_fov;
        float view_w = view_h * aspect_ratio;
        
        Vec3 ray_dir_cam(ndc_x * view_w, ndc_y * view_h, -1.0f);
        ray_dir_cam = ray_dir_cam.normalize();
        
        // Transform ray direction to world space
        // cam_right, cam_up, -cam_forward are the basis vectors
        Vec3 ray_dir_world = (cam_right * ray_dir_cam.x + cam_up * ray_dir_cam.y + cam_forward * ray_dir_cam.z).normalize();
        
        // 2. Perform Ray-AABB Intersection
        Vec3 inv_dir(1.0f / ray_dir_world.x, 1.0f / ray_dir_world.y, 1.0f / ray_dir_world.z);
        
        float t1 = (bounds_min.x - cam.lookfrom.x) * inv_dir.x;
        float t2 = (bounds_max.x - cam.lookfrom.x) * inv_dir.x;
        float t3 = (bounds_min.y - cam.lookfrom.y) * inv_dir.y;
        float t4 = (bounds_max.y - cam.lookfrom.y) * inv_dir.y;
        float t5 = (bounds_min.z - cam.lookfrom.z) * inv_dir.z;
        float t6 = (bounds_max.z - cam.lookfrom.z) * inv_dir.z;
        
        float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
        float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));
        
        // If tmax < 0, ray (line) is intersecting AABB, but whole AABB is behind us
        // If tmin > tmax, ray doesn't intersect AABB
        if (tmax >= 0 && tmin <= tmax) {
             // Intersection found!
             if (ImGui::IsMouseClicked(0) && !ImGuizmo::IsOver()) {
                // Use SceneSelection!
                ctx.selection.selectGasVolume(gas, -1, gas->name);
                GasUI::selected_gas_volume = gas; // Keep sync for now if needed, but rely on ctx
            }
        }
        
        // Check if selected using Scene Selection
        bool is_selected = (
            ctx.selection.selected.type == SelectableType::GasVolume && 
            ctx.selection.selected.gas_volume == gas
        );
        
        if (is_selected) {
            // Bright cyan for selected gas volume
            DrawBoundingBox(bounds_min, bounds_max, IM_COL32(0, 200, 255, 255), 2.0f);
            
            // Draw emitter wireframes for selected gas volume
            const auto& emitters = gas->getEmitters();
            for (const auto& e : emitters) {
                if (!e.enabled) continue;
                
                Vec3 epos = bounds_min + e.position;  // Emitter pos relative to grid origin
                
                if (e.shape == FluidSim::EmitterShape::Sphere) {
                    // Draw sphere as circle gizmo (8 segments)
                    float r = e.radius;
                    ImU32 emit_col = IM_COL32(255, 100, 50, 200);
                    
                    // Draw in 3 planes
                    for (int plane = 0; plane < 3; ++plane) {
                        for (int i = 0; i < 8; ++i) {
                            float a0 = i * (6.28318f / 8.0f);
                            float a1 = (i + 1) * (6.28318f / 8.0f);
                            Vec3 p0, p1;
                            
                            if (plane == 0) { // XY
                                p0 = epos + Vec3(cosf(a0) * r, sinf(a0) * r, 0);
                                p1 = epos + Vec3(cosf(a1) * r, sinf(a1) * r, 0);
                            } else if (plane == 1) { // XZ
                                p0 = epos + Vec3(cosf(a0) * r, 0, sinf(a0) * r);
                                p1 = epos + Vec3(cosf(a1) * r, 0, sinf(a1) * r);
                            } else { // YZ
                                p0 = epos + Vec3(0, cosf(a0) * r, sinf(a0) * r);
                                p1 = epos + Vec3(0, cosf(a1) * r, sinf(a1) * r);
                            }
                            
                            DrawBoundingBox(p0 - Vec3(0.02f), p0 + Vec3(0.02f), emit_col, 1.0f);
                        }
                    }
                } else if (e.shape == FluidSim::EmitterShape::Box) {
                    // Draw box emitter
                    Vec3 half = e.size * 0.5f;
                    Vec3 emin = epos - half;
                    Vec3 emax = epos + half;
                    DrawBoundingBox(emin, emax, IM_COL32(255, 150, 50, 200), 1.5f);
                } else {
                    // Point emitter - small box
                    DrawBoundingBox(epos - Vec3(0.1f), epos + Vec3(0.1f), IM_COL32(255, 50, 50, 255), 1.5f);
                }
            }
        } else {
            // Ghost outline for unselected gas volumes - purple/magenta tint
            DrawBoundingBox(bounds_min, bounds_max, IM_COL32(180, 100, 200, 100), 1.0f);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DRAW SELECTION HIGHLIGHTS
    // ─────────────────────────────────────────────────────────────────────────
    if (sel.hasSelection()) {
        // Draw bounding box for each selected item (multi-selection support)
        for (size_t idx = 0; idx < sel.multi_selection.size(); ++idx) {
            auto& item = sel.multi_selection[idx];

            // Primary selection (last one) gets a brighter color
            bool is_primary = (idx == sel.multi_selection.size() - 1);
            ImU32 color = is_primary ? IM_COL32(0, 255, 128, 255) : IM_COL32(0, 200, 100, 180);
            float thickness = is_primary ? 2.0f : 1.5f;

            Vec3 bb_min, bb_max;
            bool has_bounds = false;

            if (item.type == SelectableType::Object && item.object) {
                std::string selectedName = item.object->nodeName;
                if (selectedName.empty()) selectedName = "Unnamed";

                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

                // USE CACHED BOUNDING BOX (O(1) lookup instead of O(N) triangle scan!)
                auto bbox_it = bbox_cache.find(selectedName);
                if (bbox_it != bbox_cache.end()) {
                    Vec3 cached_min = bbox_it->second.first;
                    Vec3 cached_max = bbox_it->second.second;

                    // TRANSFORM THE BOUNDING BOX by object's current transform matrix
                    // This ensures bbox follows the object in TLAS mode where CPU vertices aren't updated
                    auto transform = item.object->getTransformHandle();
                    if (transform) {
                        Matrix4x4& m = transform->base;

                        // Transform all 8 corners and find new AABB
                        Vec3 corners[8] = {
                            Vec3(cached_min.x, cached_min.y, cached_min.z),
                            Vec3(cached_max.x, cached_min.y, cached_min.z),
                            Vec3(cached_min.x, cached_max.y, cached_min.z),
                            Vec3(cached_max.x, cached_max.y, cached_min.z),
                            Vec3(cached_min.x, cached_min.y, cached_max.z),
                            Vec3(cached_max.x, cached_min.y, cached_max.z),
                            Vec3(cached_min.x, cached_max.y, cached_max.z),
                            Vec3(cached_max.x, cached_max.y, cached_max.z)
                        };

                        bb_min = Vec3(1e10f, 1e10f, 1e10f);
                        bb_max = Vec3(-1e10f, -1e10f, -1e10f);

                        for (int c = 0; c < 8; c++) {
                            Vec3 p = corners[c];
                            // Apply transform: p' = M * p
                            Vec3 tp(
                                m.m[0][0] * p.x + m.m[0][1] * p.y + m.m[0][2] * p.z + m.m[0][3],
                                m.m[1][0] * p.x + m.m[1][1] * p.y + m.m[1][2] * p.z + m.m[1][3],
                                m.m[2][0] * p.x + m.m[2][1] * p.y + m.m[2][2] * p.z + m.m[2][3]
                            );
                            bb_min.x = fminf(bb_min.x, tp.x);
                            bb_min.y = fminf(bb_min.y, tp.y);
                            bb_min.z = fminf(bb_min.z, tp.z);
                            bb_max.x = fmaxf(bb_max.x, tp.x);
                            bb_max.y = fmaxf(bb_max.y, tp.y);
                            bb_max.z = fmaxf(bb_max.z, tp.z);
                        }
                    }
                    else {
                        // No transform - use cached values directly
                        bb_min = cached_min;
                        bb_max = cached_max;
                    }
                    has_bounds = true;
                }
            }
            else if (item.type == SelectableType::Light && item.light) {
                Vec3 lightPos = item.light->position;
                float boxSize = 0.15f; // Küçültüldü: 0.5 -> 0.15
                bb_min = Vec3(lightPos.x - boxSize, lightPos.y - boxSize, lightPos.z - boxSize);
                bb_max = Vec3(lightPos.x + boxSize, lightPos.y + boxSize, lightPos.z + boxSize);
                has_bounds = true;
                color = is_primary ? IM_COL32(255, 255, 100, 255) : IM_COL32(200, 200, 80, 180);
            }
            else if (item.type == SelectableType::Camera && item.camera) {
                Vec3 camPos = item.camera->lookfrom;
                float boxSize = 0.5f;
                bb_min = Vec3(camPos.x - boxSize, camPos.y - boxSize, camPos.z - boxSize);
                bb_max = Vec3(camPos.x + boxSize, camPos.y + boxSize, camPos.z + boxSize);
                has_bounds = true;
                color = is_primary ? IM_COL32(100, 200, 255, 255) : IM_COL32(80, 160, 200, 180);
            }
            else if (item.type == SelectableType::VDBVolume && item.vdb_volume) {
                AABB bounds = item.vdb_volume->getWorldBounds();
                bb_min = bounds.min;
                bb_max = bounds.max;
                has_bounds = true;
                // Orange for VDB
                color = is_primary ? IM_COL32(255, 128, 0, 255) : IM_COL32(200, 100, 0, 180);
            }
            else if (item.type == SelectableType::ForceField && item.force_field) {
                // Determine bounds based on shape
                float r = item.force_field->falloff_radius;
                Vec3 p = item.force_field->position;
                
                if (item.force_field->shape == Physics::ForceFieldShape::Infinite) {
                    // Just a small box at origin if infinite
                    bb_min = p - Vec3(0.5f);
                    bb_max = p + Vec3(0.5f);
                } else {
                    bb_min = p - Vec3(r);
                    bb_max = p + Vec3(r);
                }
                has_bounds = true;
                // Purple/Pink for Force Fields
                color = is_primary ? IM_COL32(255, 0, 255, 255) : IM_COL32(200, 0, 200, 180);
            }

            if (has_bounds) {
                DrawBoundingBox(bb_min, bb_max, color, thickness);
            }
        }
    }
    // Force Field Picking Logic (Similar to Gas)
    for (const auto& ff : ctx.scene.force_field_manager.force_fields) {
        if (!ff || !ff->enabled) continue;
        
        float r = (ff->shape == Physics::ForceFieldShape::Infinite) ? 0.5f : ff->falloff_radius;
        Vec3 bounds_min = ff->position - Vec3(r);
        Vec3 bounds_max = ff->position + Vec3(r);
        
        // Ray-AABB Intersection for Picking
        float mouse_x = io.MousePos.x;
        float mouse_y = io.MousePos.y;
        float ndc_x = (mouse_x / screen_w) * 2.0f - 1.0f;
        float ndc_y = 1.0f - (mouse_y / screen_h) * 2.0f;
        
        float view_h = tan_half_fov;
        float view_w = view_h * aspect_ratio;
        Vec3 ray_dir_cam(ndc_x * view_w, ndc_y * view_h, -1.0f);
        ray_dir_cam = ray_dir_cam.normalize();
        Vec3 ray_dir_world = (cam_right * ray_dir_cam.x + cam_up * ray_dir_cam.y + cam_forward * ray_dir_cam.z).normalize();
        Vec3 inv_dir(1.0f / ray_dir_world.x, 1.0f / ray_dir_world.y, 1.0f / ray_dir_world.z);
        
        float t1 = (bounds_min.x - cam.lookfrom.x) * inv_dir.x;
        float t2 = (bounds_max.x - cam.lookfrom.x) * inv_dir.x;
        float t3 = (bounds_min.y - cam.lookfrom.y) * inv_dir.y;
        float t4 = (bounds_max.y - cam.lookfrom.y) * inv_dir.y;
        float t5 = (bounds_min.z - cam.lookfrom.z) * inv_dir.z;
        float t6 = (bounds_max.z - cam.lookfrom.z) * inv_dir.z;
        
        float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
        float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));
        
        if (tmax >= 0 && tmin <= tmax) {
             if (ImGui::IsMouseClicked(0) && !ImGuizmo::IsOver()) {
                ctx.selection.selectForceField(ff, -1, ff->name);
                ForceFieldUI::selected_force_field = ff;
            }
        }
    }
    // VDB debug bounds removed (User Request)
}

void SceneUI::drawLightGizmos(UIContext& ctx, bool& gizmo_hit)
{
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImGuiIO& io = ImGui::GetIO();
    Camera& cam = *ctx.scene.camera;

    Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();

    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);
    float aspect = io.DisplaySize.x / io.DisplaySize.y;

    auto Project = [&](const Vec3& p) -> ImVec2 {
        Vec3 to_point = p - cam.lookfrom;
        float depth = to_point.dot(cam_forward);
        if (depth <= 0.1f) return ImVec2(-10000, -10000);

        float local_x = to_point.dot(cam_right);
        float local_y = to_point.dot(cam_up);

        float half_h = depth * tan_half_fov;
        float half_w = half_h * aspect;

        return ImVec2(
            ((local_x / half_w) * 0.5f + 0.5f) * io.DisplaySize.x,
            (0.5f - (local_y / half_h) * 0.5f) * io.DisplaySize.y
        );
        };

    auto IsOnScreen = [](const ImVec2& v) { return v.x > -5000; };

    for (auto& light : ctx.scene.lights) {
        if (!light->visible) continue;

        bool selected =
            (ctx.selection.selected.type == SelectableType::Light &&
                ctx.selection.selected.light == light);

        ImU32 col = selected
            ? IM_COL32(255, 100, 50, 255)
            : IM_COL32(255, 255, 100, 180);

        Vec3 pos = light->position;

        // [FIX] Depth/Occlusion Check for Light Gizmos
        extern bool g_bvh_rebuild_pending;
        if (ctx.scene.bvh && !g_bvh_rebuild_pending) {
            Vec3 to_pos = pos - cam.lookfrom;
            float dist = to_pos.length();
             if (dist > 0.1f) {
                 Ray r(cam.lookfrom, to_pos / dist);
                 HitRecord rec;
                 if (ctx.scene.bvh->hit(r, 0.001f, dist - 0.1f, rec, true)) {
                     // Occluded: Fade out
                     int alpha = (col >> 24) & 0xFF;
                     alpha = alpha / 5;
                     if (alpha < 20) alpha = 20;
                     col = (col & 0x00FFFFFF) | (alpha << 24);
                 }
            }
        }

        ImVec2 center = Project(pos);
        bool visible = IsOnScreen(center);

        if (!visible) continue;

        // -------- PICKING --------
        float dx = io.MousePos.x - center.x;
        float dy = io.MousePos.y - center.y;
        float d = sqrtf(dx * dx + dy * dy);

        if (d < 20.0f && ImGui::IsMouseClicked(0) && !ImGuizmo::IsOver()) {
            ctx.selection.selectLight(light);
            gizmo_hit = true;
        }

        if (selected) {
            std::string label = light->nodeName.empty() ? "Light" : light->nodeName;
            draw_list->AddText(ImVec2(center.x + 12, center.y - 12), col, label.c_str());
        }

        // ================== DRAW BY TYPE ==================

        // ---- POINT (Orta Daire + D\u0131\u015f Halka) ----
        if (light->type() == LightType::Point) {
            // Parlak merkez daire
            draw_list->AddCircleFilled(center, 5.0f, IM_COL32(255, 240, 180, 255));
            // D\u0131\u015f halka
            draw_list->AddCircle(center, 12.0f, col, 0, 2.0f);
            // \u0130nce i\u00e7 halka (derinlik etkisi i\u00e7in)
            draw_list->AddCircle(center, 8.0f, IM_COL32(255, 220, 100, 120), 0, 1.0f);
        }

        // ---- DIRECTIONAL (Sun + Arrow) ----
        else if (light->type() == LightType::Directional) {
            draw_list->AddCircle(center, 8.0f, col, 0, 2.0f);

            for (int i = 0; i < 8; ++i) {
                float a = i * (6.28f / 8.0f);
                ImVec2 dir(cosf(a), sinf(a));
                draw_list->AddLine(
                    ImVec2(center.x + dir.x * 12, center.y + dir.y * 12),
                    ImVec2(center.x + dir.x * 18, center.y + dir.y * 18),
                    col);
            }

            auto dl = std::dynamic_pointer_cast<DirectionalLight>(light);
            if (dl) {
                Vec3 end3d = pos + dl->direction.normalize() * 3.0f;
                ImVec2 end = Project(end3d);
                if (IsOnScreen(end)) {
                    draw_list->AddLine(center, end, col, 2.0f);
                    draw_list->AddCircleFilled(end, 3.0f, col);
                }
            }
        }

        // ---- AREA (Rectangle) ----
        else if (light->type() == LightType::Area) {
            auto al = std::dynamic_pointer_cast<AreaLight>(light);
            if (!al) continue;

            // Normalleştirilmiş u ve v vektörleri
            Vec3 u = al->getU();
            Vec3 v = al->getV();
            float halfW = al->getWidth() * 0.5f;
            float halfH = al->getHeight() * 0.5f;

            // pos merkez noktası, köşeleri merkezden hesapla
            Vec3 corner1 = pos - u * halfW - v * halfH;  // Sol-Alt
            Vec3 corner2 = pos + u * halfW - v * halfH;  // Sağ-Alt  
            Vec3 corner3 = pos + u * halfW + v * halfH;  // Sağ-Üst
            Vec3 corner4 = pos - u * halfW + v * halfH;  // Sol-Üst

            ImVec2 c1 = Project(corner1);
            ImVec2 c2 = Project(corner2);
            ImVec2 c3 = Project(corner3);
            ImVec2 c4 = Project(corner4);

            draw_list->AddLine(c1, c2, col);
            draw_list->AddLine(c2, c3, col);
            draw_list->AddLine(c3, c4, col);
            draw_list->AddLine(c4, c1, col);
            // X çizgisi: merkez artık gerçek merkez
            draw_list->AddLine(c1, c3, col, 1.0f);
            draw_list->AddLine(c2, c4, col, 1.0f);
        }

        // ---- SPOT (Cone) ----
        else if (light->type() == LightType::Spot) {
            auto sl = std::dynamic_pointer_cast<SpotLight>(light);
            if (!sl) continue;

            Vec3 dir = sl->direction.normalize();
            float len = 3.0f;
            float radius = len * tanf(sl->getAngleDegrees() * 3.14159f / 360.0f);

            Vec3 base = pos + dir * len;
            Vec3 right = (fabs(dir.y) > 0.9f) ? Vec3(1, 0, 0)
                : dir.cross(Vec3(0, 1, 0)).normalize();
            Vec3 up = right.cross(dir).normalize();

            const int segs = 12;
            ImVec2 last;
            for (int i = 0; i <= segs; ++i) {
                float a = i * (6.28f / segs);
                Vec3 p = base + right * (cosf(a) * radius)
                    + up * (sinf(a) * radius);

                ImVec2 sp = Project(p);
                if (i > 0 && IsOnScreen(sp) && IsOnScreen(last))
                    draw_list->AddLine(last, sp, col);
                if (i < segs && IsOnScreen(sp))
                    draw_list->AddLine(center, sp, col);

                last = sp;
            }
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════════════
// IMGUIZMO TRANSFORM GIZMO
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawTransformGizmo(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;
    if (!sel.hasSelection() || !sel.show_gizmo || !ctx.scene.camera) return;

    // Check visibility of selected item
    bool is_visible = true;
    if (sel.selected.type == SelectableType::Object && sel.selected.object) is_visible = sel.selected.object->visible;
    else if (sel.selected.type == SelectableType::Light && sel.selected.light) is_visible = sel.selected.light->visible;
    else if (sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume) is_visible = sel.selected.vdb_volume->visible;
    else if (sel.selected.type == SelectableType::GasVolume && sel.selected.gas_volume) is_visible = sel.selected.gas_volume->visible;
    
    if (!is_visible) return;

    Camera& cam = *ctx.scene.camera;
    ImGuiIO& io = ImGui::GetIO();

    // Setup ImGuizmo
    ImGuizmo::BeginFrame();
    ImGuizmo::SetOrthographic(false);
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

    // ─────────────────────────────────────────────────────────────────────────
    // Build View Matrix (LookAt)
    // ─────────────────────────────────────────────────────────────────────────
    Vec3 eye = cam.lookfrom;
    Vec3 target = cam.lookat;
    Vec3 up = cam.vup;

    Vec3 f = (target - eye).normalize();  // Forward
    Vec3 r = f.cross(up).normalize();     // Right
    Vec3 u = r.cross(f);                   // Up

    float viewMatrix[16] = {
        r.x,  u.x, -f.x, 0.0f,
        r.y,  u.y, -f.y, 0.0f,
        r.z,  u.z, -f.z, 0.0f,
        -r.dot(eye), -u.dot(eye), f.dot(eye), 1.0f
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Build Projection Matrix (Perspective)
    // ─────────────────────────────────────────────────────────────────────────
    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float near_plane = 0.1f;
    float far_plane = 10000.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);

    float projMatrix[16] = { 0 };
    projMatrix[0] = 1.0f / (aspect_ratio * tan_half_fov);
    projMatrix[5] = 1.0f / tan_half_fov;
    projMatrix[10] = -(far_plane + near_plane) / (far_plane - near_plane);
    projMatrix[11] = -1.0f;
    projMatrix[14] = -(2.0f * far_plane * near_plane) / (far_plane - near_plane);

    auto Project = [&](Vec3 p) -> ImVec2 {
        float x = p.x, y = p.y, z = p.z;
        float vx = viewMatrix[0] * x + viewMatrix[4] * y + viewMatrix[8] * z + viewMatrix[12];
        float vy = viewMatrix[1] * x + viewMatrix[5] * y + viewMatrix[9] * z + viewMatrix[13];
        float vz = viewMatrix[2] * x + viewMatrix[6] * y + viewMatrix[10] * z + viewMatrix[14];
        float vw = viewMatrix[3] * x + viewMatrix[7] * y + viewMatrix[11] * z + viewMatrix[15];
        float cx = projMatrix[0] * vx + projMatrix[4] * vy + projMatrix[8] * vz + projMatrix[12] * vw;
        float cy = projMatrix[1] * vx + projMatrix[5] * vy + projMatrix[9] * vz + projMatrix[13] * vw;
        float cw = projMatrix[3] * vx + projMatrix[7] * vy + projMatrix[11] * vz + projMatrix[15] * vw;
        if (cw < 0.1f) return ImVec2(-10000, -10000);
        return ImVec2(((cx / cw) * 0.5f + 0.5f) * io.DisplaySize.x, (1.0f - ((cy / cw) * 0.5f + 0.5f)) * io.DisplaySize.y);
        };

    // ─────────────────────────────────────────────────────────────────────────
    // Get Object Matrix
    // ─────────────────────────────────────────────────────────────────────────
    float objectMatrix[16];
    Vec3 pos = sel.selected.position;

    // AreaLight: position zaten merkez noktas\u0131, ek offset gerekli de\u011fil

    // Initialize as identity with position
    Matrix4x4 startMat = Matrix4x4::identity();
    startMat.m[0][3] = pos.x;
    startMat.m[1][3] = pos.y;
    startMat.m[2][3] = pos.z;

    // Handle Light Rotation (Directional/Spot)
    if (sel.selected.type == SelectableType::Light && sel.selected.light) {
        Vec3 dir(0, 0, 0);
        bool hasDir = false;

        if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(sel.selected.light)) {
            // DirectionalLight: getDirection returns vector TO Light (inverse of direction)
            // We want the light direction for visualization
            dir = dl->getDirection(Vec3(0)).normalize() * -1.0f;
            hasDir = true;
        }
        else if (auto sl = std::dynamic_pointer_cast<SpotLight>(sel.selected.light)) {
            // Manual Matrix for SpotLight to include Angle as Scale
            hasDir = false;
            float angle = sl->getAngleDegrees();
            if (angle < 1.0f) angle = 1.0f;

            // Direction Basis
            Vec3 dirVec = sl->direction.normalize();
            Vec3 Z = -dirVec;
            Vec3 Y_temp(0, 1, 0);
            if (abs(Vec3::dot(Z, Y_temp)) > 0.99f) Y_temp = Vec3(1, 0, 0);

            Vec3 X = Vec3::cross(Y_temp, Z).normalize();
            Vec3 Y = Vec3::cross(Z, X).normalize();

            // Scale X and Y by Angle for Gizmo Interaction
            Vec3 X_scaled = X * angle;
            Vec3 Y_scaled = Y * angle;

            // Scale Z by (1 + Falloff) for Falloff interaction
            float zScale = 1.0f + sl->getFalloff();
            Vec3 Z_scaled = Z * zScale;

            startMat.m[0][0] = X_scaled.x; startMat.m[0][1] = Y_scaled.x; startMat.m[0][2] = Z_scaled.x;
            startMat.m[1][0] = X_scaled.y; startMat.m[1][1] = Y_scaled.y; startMat.m[1][2] = Z_scaled.y;
            startMat.m[2][0] = X_scaled.z; startMat.m[2][1] = Y_scaled.z; startMat.m[2][2] = Z_scaled.z;

            // -----------------------------------------------------
            // Visual Helper: Cone Draw
            // -----------------------------------------------------
            ImDrawList* dl = ImGui::GetBackgroundDrawList();
            Vec3 pos = sl->position;
            float h = 5.0f; // Visual height
            float r = tanf(angle * 3.14159f / 180.0f * 0.5f) * h;

            ImVec2 pTip = Project(pos);
            Vec3 centerBase = pos + dirVec * h;

            ImU32 col = IM_COL32(255, 255, 0, 180);
            Vec3 prevP;
            bool first = true;

            for (int i = 0; i <= 24; ++i) {
                float t = (float)i / 24.0f * 6.28318f;
                Vec3 pBase = centerBase + (X * cosf(t) + Y * sinf(t)) * r;
                ImVec2 pScreen = Project(pBase);

                if (!first) dl->AddLine(Project(prevP), pScreen, col, 2.0f);

                if (i % 6 == 0) dl->AddLine(pTip, pScreen, col, 1.0f);

                prevP = pBase;
                first = false;
            }

            // Inner Cone (Falloff)
            float falloff = sl->getFalloff();
            if (falloff > 0.05f) {
                float innerAngle = angle * (1.0f - falloff);
                float rInner = tanf(innerAngle * 3.14159f / 180.0f * 0.5f) * h;
                ImU32 colInner = IM_COL32(255, 160, 20, 120);
                Vec3 prevIn;
                bool firstIn = true;
                for (int i = 0; i <= 24; i++) {
                    float t = (float)i / 24.0f * 6.28318f;
                    Vec3 pBase = centerBase + (X * cosf(t) + Y * sinf(t)) * rInner;
                    if (!firstIn && i % 2 == 0) dl->AddLine(Project(prevIn), Project(pBase), colInner, 1.0f); // Dashed-ish effect
                    prevIn = pBase;
                    firstIn = false;
                }
            }
        }
        else if (auto al = std::dynamic_pointer_cast<AreaLight>(sel.selected.light)) {
            hasDir = false;
            // Normalize vekt\u00f6rleri width/height ile \u00f6l\u00e7eklendirerek ger\u00e7ek boyutu yans\u0131t
            Vec3 X = al->u * al->getWidth();   // u normalize, width ile \u00f6l\u00e7ekle
            Vec3 Z = al->v * al->getHeight();  // v normalize, height ile \u00f6l\u00e7ekle
            // Normalized Y (Normal)
            Vec3 Y = Vec3::cross(al->u, al->v).normalize();

            startMat.m[0][0] = X.x; startMat.m[0][1] = Y.x; startMat.m[0][2] = Z.x;
            startMat.m[1][0] = X.y; startMat.m[1][1] = Y.y; startMat.m[1][2] = Z.y;
            startMat.m[2][0] = X.z; startMat.m[2][1] = Y.z; startMat.m[2][2] = Z.z;

            // Visualization: Direction Arrow (\u00d6l\u00e7e\u011fe g\u00f6re k\u0131salt)
            ImDrawList* dl = ImGui::GetBackgroundDrawList();
            Vec3 center = pos;
            Vec3 normal = Y.normalize();
            float len = std::min(al->getWidth(), al->getHeight()) * 0.5f; // Daha k\u0131sa, orant\u0131l\u0131 ok
            if (len < 0.3f) len = 0.3f;
            Vec3 pTip = center + normal * len;
            dl->AddLine(Project(center), Project(pTip), IM_COL32(255, 255, 0, 200), 2.0f);
            dl->AddCircleFilled(Project(pTip), 4.0f, IM_COL32(255, 255, 0, 255));
        }

        if (hasDir) {
            // Align Gizmo -Z with Light Direction
            Vec3 Z = -dir;
            Vec3 Y(0, 1, 0);
            if (abs(Vec3::dot(Z, Y)) > 0.99f) Y = Vec3(1, 0, 0); // Lock prevention
            Vec3 X = Vec3::cross(Y, Z).normalize();
            Y = Vec3::cross(Z, X).normalize();

            startMat.m[0][0] = X.x; startMat.m[0][1] = Y.x; startMat.m[0][2] = Z.x;
            startMat.m[1][0] = X.y; startMat.m[1][1] = Y.y; startMat.m[1][2] = Z.y;
            startMat.m[2][0] = X.z; startMat.m[2][1] = Y.z; startMat.m[2][2] = Z.z;
        }
    }

    objectMatrix[0] = startMat.m[0][0]; objectMatrix[1] = startMat.m[1][0]; objectMatrix[2] = startMat.m[2][0]; objectMatrix[3] = startMat.m[3][0];
    objectMatrix[4] = startMat.m[0][1]; objectMatrix[5] = startMat.m[1][1]; objectMatrix[6] = startMat.m[2][1]; objectMatrix[7] = startMat.m[3][1];
    objectMatrix[8] = startMat.m[0][2]; objectMatrix[9] = startMat.m[1][2]; objectMatrix[10] = startMat.m[2][2]; objectMatrix[11] = startMat.m[3][2];
    objectMatrix[12] = startMat.m[0][3]; objectMatrix[13] = startMat.m[1][3]; objectMatrix[14] = startMat.m[2][3]; objectMatrix[15] = startMat.m[3][3];

    // If object has transform, use it
    // If object has transform, use it - BUT ONLY if not a Mixed Group
    // Mixed groups use the Centroid pivot (startMat) to avoid fighting/resets
    bool is_mixed_group = false;
    if (sel.multi_selection.size() > 1) {
        is_mixed_group = true;
    }
    else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
        std::string name = sel.selected.object->nodeName;
        if (name.empty()) name = "Unnamed";
        auto it = mesh_cache.find(name);
        if (it != mesh_cache.end()) {
            // OPTIMIZED: Only check first 100 triangles, not all 2M!
            Transform* firstT = nullptr;
            const size_t MAX_CHECK = std::min((size_t)100, it->second.size());
            for (size_t i = 0; i < MAX_CHECK; ++i) {
                auto th = it->second[i].second->getTransformHandle().get();
                if (!firstT) firstT = th;
                else if (th != firstT) {
                    is_mixed_group = true;
                    break;
                }
            }
        }
    }

    static Matrix4x4 mixed_gizmo_matrix;
    static bool was_using_mixed = false;

    // Check global drag state
    bool is_using_gizmo_now = ImGuizmo::IsUsing();
    // auto transform = ... (Removed unsafe access)
    if (is_mixed_group) {
        // PERISISTENT GIZMO STATE for Mixed Groups
        // Prevents Gizmo from snapping back to Identity Rotation every frame which causes explosion

        if (!is_using_gizmo_now) {
            // Not dragging: Reset to Centroid Position + Identity Rotation
            // (Or we could average rotations, but Identity is safer for a group pivot)
            mixed_gizmo_matrix = Matrix4x4::identity();
            mixed_gizmo_matrix.m[0][3] = sel.selected.position.x;
            mixed_gizmo_matrix.m[1][3] = sel.selected.position.y;
            mixed_gizmo_matrix.m[2][3] = sel.selected.position.z;
        }

        // Use the persistent matrix
        objectMatrix[0] = mixed_gizmo_matrix.m[0][0]; objectMatrix[1] = mixed_gizmo_matrix.m[1][0]; objectMatrix[2] = mixed_gizmo_matrix.m[2][0]; objectMatrix[3] = mixed_gizmo_matrix.m[3][0];
        objectMatrix[4] = mixed_gizmo_matrix.m[0][1]; objectMatrix[5] = mixed_gizmo_matrix.m[1][1]; objectMatrix[6] = mixed_gizmo_matrix.m[2][1]; objectMatrix[7] = mixed_gizmo_matrix.m[3][1];
        objectMatrix[8] = mixed_gizmo_matrix.m[0][2]; objectMatrix[9] = mixed_gizmo_matrix.m[1][2]; objectMatrix[10] = mixed_gizmo_matrix.m[2][2]; objectMatrix[11] = mixed_gizmo_matrix.m[3][2];
        objectMatrix[12] = mixed_gizmo_matrix.m[0][3]; objectMatrix[13] = mixed_gizmo_matrix.m[1][3]; objectMatrix[14] = mixed_gizmo_matrix.m[2][3]; objectMatrix[15] = mixed_gizmo_matrix.m[3][3];
    }
    
    else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
        // Single Object (or Homogeneous Group) - Lock to Object Transform
        auto transform = sel.selected.object->getTransformHandle();
        if (transform) {
            Matrix4x4 mat = transform->base;
            objectMatrix[0] = mat.m[0][0]; objectMatrix[1] = mat.m[1][0]; objectMatrix[2] = mat.m[2][0]; objectMatrix[3] = mat.m[3][0];
            objectMatrix[4] = mat.m[0][1]; objectMatrix[5] = mat.m[1][1]; objectMatrix[6] = mat.m[2][1]; objectMatrix[7] = mat.m[3][1];
            objectMatrix[8] = mat.m[0][2]; objectMatrix[9] = mat.m[1][2]; objectMatrix[10] = mat.m[2][2]; objectMatrix[11] = mat.m[3][2];
            objectMatrix[12] = mat.m[0][3]; objectMatrix[13] = mat.m[1][3]; objectMatrix[14] = mat.m[2][3]; objectMatrix[15] = mat.m[3][3];
        }
    }
    else if (sel.selected.type == SelectableType::GasVolume && sel.selected.gas_volume) {
        Matrix4x4 mat = sel.selected.gas_volume->getTransform();
        objectMatrix[0] = mat.m[0][0]; objectMatrix[1] = mat.m[1][0]; objectMatrix[2] = mat.m[2][0]; objectMatrix[3] = mat.m[3][0];
        objectMatrix[4] = mat.m[0][1]; objectMatrix[5] = mat.m[1][1]; objectMatrix[6] = mat.m[2][1]; objectMatrix[7] = mat.m[3][1];
        objectMatrix[8] = mat.m[0][2]; objectMatrix[9] = mat.m[1][2]; objectMatrix[10] = mat.m[2][2]; objectMatrix[11] = mat.m[3][2];
        objectMatrix[12] = mat.m[0][3]; objectMatrix[13] = mat.m[1][3]; objectMatrix[14] = mat.m[2][3]; objectMatrix[15] = mat.m[3][3];
    }
    else if (sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume) {
        Matrix4x4 mat = sel.selected.vdb_volume->getTransform();
        objectMatrix[0] = mat.m[0][0]; objectMatrix[1] = mat.m[1][0]; objectMatrix[2] = mat.m[2][0]; objectMatrix[3] = mat.m[3][0];
        objectMatrix[4] = mat.m[0][1]; objectMatrix[5] = mat.m[1][1]; objectMatrix[6] = mat.m[2][1]; objectMatrix[7] = mat.m[3][1];
        objectMatrix[8] = mat.m[0][2]; objectMatrix[9] = mat.m[1][2]; objectMatrix[10] = mat.m[2][2]; objectMatrix[11] = mat.m[3][2];
        objectMatrix[12] = mat.m[0][3]; objectMatrix[13] = mat.m[1][3]; objectMatrix[14] = mat.m[2][3]; objectMatrix[15] = mat.m[3][3];
    }
    else if (sel.selected.type == SelectableType::ForceField && sel.selected.force_field) {
        Matrix4x4 mat = Matrix4x4::fromTRS(sel.selected.force_field->position, sel.selected.force_field->rotation, sel.selected.force_field->scale);

        objectMatrix[0] = mat.m[0][0]; objectMatrix[1] = mat.m[1][0]; objectMatrix[2] = mat.m[2][0]; objectMatrix[3] = mat.m[3][0];
        objectMatrix[4] = mat.m[0][1]; objectMatrix[5] = mat.m[1][1]; objectMatrix[6] = mat.m[2][1]; objectMatrix[7] = mat.m[3][1];
        objectMatrix[8] = mat.m[0][2]; objectMatrix[9] = mat.m[1][2]; objectMatrix[10] = mat.m[2][2]; objectMatrix[11] = mat.m[3][2];
        objectMatrix[12] = mat.m[0][3]; objectMatrix[13] = mat.m[1][3]; objectMatrix[14] = mat.m[2][3]; objectMatrix[15] = mat.m[3][3];
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Keyboard Shortcuts for Transform Mode
    // ─────────────────────────────────────────────────────────────────────────
    // Only process when viewport has focus (not UI panels)
    if (sel.hasSelection() && !ImGui::GetIO().WantCaptureKeyboard) {
        if (ImGui::IsKeyPressed(ImGuiKey_G)) {
            sel.transform_mode = TransformMode::Translate;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_R)) {
            sel.transform_mode = TransformMode::Rotate;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_S) && !ImGui::GetIO().KeyShift) {
            // S alone = Scale, Shift+S would trigger duplication so check
            sel.transform_mode = TransformMode::Scale;
        }
        else if (ImGui::IsKeyPressed(ImGuiKey_W)) {
            // Cycle through modes
            switch (sel.transform_mode) {
            case TransformMode::Translate: sel.transform_mode = TransformMode::Rotate; break;
            case TransformMode::Rotate: sel.transform_mode = TransformMode::Scale; break;
            case TransformMode::Scale: sel.transform_mode = TransformMode::Translate; break;
            }
        }

        // Shift + D = Duplicate Object
        if (ImGui::IsKeyPressed(ImGuiKey_D) && ImGui::GetIO().KeyShift) {
            if (sel.selected.type == SelectableType::Object && sel.selected.object) {
                std::string targetName = sel.selected.object->nodeName;
                if (targetName.empty()) targetName = "Unnamed";

                // Ensure mesh cache is valid
                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

                // Unique name generation - USE MESH_CACHE instead of scanning all triangles!
                std::string baseName = targetName;
                size_t lastUnderscore = baseName.rfind('_');
                if (lastUnderscore != std::string::npos) {
                    std::string suffix = baseName.substr(lastUnderscore + 1);
                    if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
                        baseName = baseName.substr(0, lastUnderscore);
                    }
                }

                int counter = 1;
                std::string newName;
                bool nameExists = true;
                while (nameExists) {
                    newName = baseName + "_" + std::to_string(counter);
                    // O(log N) lookup in mesh_cache instead of O(N) triangle scan!
                    nameExists = mesh_cache.find(newName) != mesh_cache.end();
                    counter++;
                }

                // Find source triangles
                auto it = mesh_cache.find(targetName);
                if (it != mesh_cache.end() && !it->second.empty()) {
                    // Pre-allocate for performance
                    size_t numTris = it->second.size();
                    std::vector<std::shared_ptr<Hittable>> newTriangles;
                    newTriangles.reserve(numTris);
                    
                    std::vector<std::pair<int, std::shared_ptr<Triangle>>> newCacheEntries;
                    newCacheEntries.reserve(numTris);

                    std::shared_ptr<Triangle> firstNewTri = nullptr;
                    int baseIndex = (int)ctx.scene.world.objects.size();

                    // Create duplicates
                    for (size_t i = 0; i < numTris; ++i) {
                        auto& oldTri = it->second[i].second;
                        auto newTri = std::make_shared<Triangle>(*oldTri);
                        newTri->setNodeName(newName);
                        newTriangles.push_back(newTri);
                        newCacheEntries.push_back({baseIndex + (int)i, newTri});
                        if (!firstNewTri) firstNewTri = newTri;
                    }

                    // Add to scene
                    ctx.scene.world.objects.insert(ctx.scene.world.objects.end(), newTriangles.begin(), newTriangles.end());
                    
                    // INCREMENTAL CACHE UPDATE (instead of full rebuildMeshCache!)
                    mesh_cache[newName] = std::move(newCacheEntries);
                    mesh_ui_cache.push_back({newName, mesh_cache[newName]});
                    
                    // Calculate bbox for new object (from original since it's a copy)
                    auto orig_bbox = bbox_cache.find(targetName);
                    if (orig_bbox != bbox_cache.end()) {
                        bbox_cache[newName] = orig_bbox->second;
                    }
                    
                    // Copy material slots cache
                    auto orig_mats = material_slots_cache.find(targetName);
                    if (orig_mats != material_slots_cache.end()) {
                        material_slots_cache[newName] = orig_mats->second;
                    }
                    
                    sel.selectObject(firstNewTri, -1, newName);

                    // Record undo command
                    std::vector<std::shared_ptr<Triangle>> new_tri_vec;
                    new_tri_vec.reserve(numTris);
                    for (auto& ht : newTriangles) {
                        auto tri = std::dynamic_pointer_cast<Triangle>(ht);
                        if (tri) new_tri_vec.push_back(tri);
                    }
                    auto command = std::make_unique<DuplicateObjectCommand>(targetName, newName, new_tri_vec);
                    history.record(std::move(command));

                    // ═══════════════════════════════════════════════════════════
                    // DEFERRED FULL REBUILD (Reliable - async in Main.cpp)
                    // ═══════════════════════════════════════════════════════════
                    extern bool g_optix_rebuild_pending;
                    g_optix_rebuild_pending = true;
                    
                    extern bool g_bvh_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    ctx.renderer.resetCPUAccumulation();
                    
                    is_bvh_dirty = false;
                    SCENE_LOG_INFO("Duplicated object: " + targetName + " -> " + newName + " (" + std::to_string(numTris) + " triangles)");
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Determine Gizmo Operation
    // ─────────────────────────────────────────────────────────────────────────
    ImGuizmo::OPERATION operation = ImGuizmo::TRANSLATE;
    switch (sel.transform_mode) {
    case TransformMode::Translate: operation = ImGuizmo::TRANSLATE; break;
    case TransformMode::Rotate: operation = ImGuizmo::ROTATE; break;
    case TransformMode::Scale: operation = ImGuizmo::SCALE; break;
    }

    // Restriction Removed: Fallback logic now handles Rot/Scale for mixed groups


    ImGuizmo::MODE mode = (sel.transform_space == TransformSpace::Local) ?
        ImGuizmo::LOCAL : ImGuizmo::WORLD;

    // ─────────────────────────────────────────────────────────────────────────
    // Shift + Drag Duplication Logic + IDLE PREVIEW
    // ─────────────────────────────────────────────────────────────────────────
    static bool was_using_gizmo = false;
    static LightState drag_start_light_state;
    static std::shared_ptr<Light> drag_light = nullptr;
    bool is_using = ImGuizmo::IsUsing();
    is_dragging = is_using; // Sync class member

    // IDLE PREVIEW: Track when mouse stops moving during drag
    // NOTE: In TLAS mode, transforms are already updated in real-time via instance matrices.
    // The heavy updateTLASGeometry and rebuildBVH calls here were causing major freezes!
    static ImVec2 last_mouse_pos = ImVec2(0, 0);
    static float idle_time = 0.0f;
    static bool preview_updated = false;
    const float IDLE_THRESHOLD = 0.3f;  // 0.3 seconds before preview update


    if (is_using && is_bvh_dirty) {
        ImVec2 current_mouse = io.MousePos;
        float mouse_delta = sqrtf(powf(current_mouse.x - last_mouse_pos.x, 2) +
            powf(current_mouse.y - last_mouse_pos.y, 2));

        if (mouse_delta < 1.0f) {  // Mouse essentially stationary
            idle_time += io.DeltaTime;

            // If idle for threshold and not yet updated, do preview update
            if (idle_time >= IDLE_THRESHOLD && !preview_updated) {
                // SCENE_LOG_INFO("[GIZMO] Idle preview - updating geometry");
                if (ctx.optix_gpu_ptr) {
                    if (ctx.optix_gpu_ptr->isUsingTLAS()) {
                        // TLAS MODE: Transforms are ALREADY updated via instance matrices!
                        // Just reset accumulation to show the updated render, NO heavy rebuild.
                        ctx.optix_gpu_ptr->resetAccumulation();
                        // Skip rebuildBVH too - picking uses linear search, not BVH.
                    } else {
                        // GAS MODE: Use fast vertex update (legacy)
                        ctx.optix_gpu_ptr->updateGeometry(ctx.scene.world.objects);
                        ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                        ctx.optix_gpu_ptr->resetAccumulation();
                        ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                    }
                } else {
                    // No OptiX: CPU mode still needs BVH
                    ctx.renderer.rebuildBVH(ctx.scene, ctx.render_settings.UI_use_embree);
                }
                ctx.renderer.resetCPUAccumulation();
                preview_updated = true;
                // Note: Don't set is_bvh_dirty = false, so final update still happens
            }
        }
        else {
            // Mouse moved - reset idle tracking
            idle_time = 0.0f;
            preview_updated = false;  // Allow another preview after next pause
        }
        last_mouse_pos = current_mouse;
    }
    else {
        // Not using gizmo - reset tracking
        idle_time = 0.0f;
        preview_updated = false;
        last_mouse_pos = io.MousePos;
    }

    if (is_using && !was_using_gizmo) {
        // ═══════════════════════════════════════════════════════════════════════════
        // MANIPULATION START
        // ═══════════════════════════════════════════════════════════════════════════
        if (io.KeyShift && sel.hasSelection()) {
            triggerDuplicate(ctx);
        }
        else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
            // START LIGHT TRANSFORM RECORDING
            drag_light = sel.selected.light;
            drag_start_light_state = LightState::capture(*drag_light);
        }
        else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            // START TRANSFORM RECORDING (Normal drag without Shift)
            auto transform = sel.selected.object->getTransformHandle();
            if (transform) {
                drag_start_state.matrix = transform->base;
                drag_object_name = sel.selected.object->nodeName;
            }
        }
    }

    // END DRAG (Release)
    if (!is_using && was_using_gizmo) {
        if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            // END TRANSFORM RECORDING
            auto t = sel.selected.object->getTransformHandle();
            if (t) {
                TransformState final_state;
                final_state.matrix = t->base;

                // Check delta
                bool changed = false;
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        if (std::abs(final_state.matrix.m[i][j] - drag_start_state.matrix.m[i][j]) > 0.0001f)
                            changed = true;

                if (changed) {
                    history.record(std::make_unique<TransformCommand>(drag_object_name, drag_start_state, final_state));
                    ProjectManager::getInstance().markModified();
                }
            }
        }
        else if (sel.selected.type == SelectableType::Light && drag_light) {
            LightState final_light_state = LightState::capture(*drag_light);

            // Check if position or other properties changed
            bool changed = (final_light_state.position - drag_start_light_state.position).length() > 0.0001f ||
                (final_light_state.direction - drag_start_light_state.direction).length() > 0.0001f ||
                std::abs(final_light_state.angle - drag_start_light_state.angle) > 0.0001f;

            if (changed) {
                history.record(std::make_unique<TransformLightCommand>(drag_light, drag_start_light_state, final_light_state));
                ProjectManager::getInstance().markModified();
            }
            drag_light = nullptr;
        }
    }

    // NOTE: was_using_gizmo update moved to END of function (after is_bvh_dirty is set)

    // ─────────────────────────────────────────────────────────────────────────
    // Render and Manipulate Gizmo
    // ─────────────────────────────────────────────────────────────────────────
    // Save old position BEFORE manipulation for delta calculation (multi-selection)
    // Save old position & MATRIX BEFORE manipulation for delta calculation
    Vec3 oldGizmoPos(objectMatrix[12], objectMatrix[13], objectMatrix[14]);

    Matrix4x4 oldMat;
    oldMat.m[0][0] = objectMatrix[0]; oldMat.m[1][0] = objectMatrix[1]; oldMat.m[2][0] = objectMatrix[2]; oldMat.m[3][0] = objectMatrix[3];
    oldMat.m[0][1] = objectMatrix[4]; oldMat.m[1][1] = objectMatrix[5]; oldMat.m[2][1] = objectMatrix[6]; oldMat.m[3][1] = objectMatrix[7];
    oldMat.m[0][2] = objectMatrix[8]; oldMat.m[1][2] = objectMatrix[9]; oldMat.m[2][2] = objectMatrix[10]; oldMat.m[3][2] = objectMatrix[11];
    oldMat.m[0][3] = objectMatrix[12]; oldMat.m[1][3] = objectMatrix[13]; oldMat.m[2][3] = objectMatrix[14]; oldMat.m[3][3] = objectMatrix[15];

    bool manipulated = ImGuizmo::Manipulate(viewMatrix, projMatrix, operation, mode, objectMatrix);

    if (manipulated && is_mixed_group) {
        // Update persistent matrix for next frame interaction
        mixed_gizmo_matrix.m[0][0] = objectMatrix[0]; mixed_gizmo_matrix.m[1][0] = objectMatrix[1]; mixed_gizmo_matrix.m[2][0] = objectMatrix[2]; mixed_gizmo_matrix.m[3][0] = objectMatrix[3];
        mixed_gizmo_matrix.m[0][1] = objectMatrix[4]; mixed_gizmo_matrix.m[1][1] = objectMatrix[5]; mixed_gizmo_matrix.m[2][1] = objectMatrix[6]; mixed_gizmo_matrix.m[3][1] = objectMatrix[7];
        mixed_gizmo_matrix.m[0][2] = objectMatrix[8]; mixed_gizmo_matrix.m[1][2] = objectMatrix[9]; mixed_gizmo_matrix.m[2][2] = objectMatrix[10]; mixed_gizmo_matrix.m[3][2] = objectMatrix[11];
        mixed_gizmo_matrix.m[0][3] = objectMatrix[12]; mixed_gizmo_matrix.m[1][3] = objectMatrix[13]; mixed_gizmo_matrix.m[2][3] = objectMatrix[14]; mixed_gizmo_matrix.m[3][3] = objectMatrix[15];
    }

    if (manipulated) {
        // Mark objects for lazy CPU sync (re-enable picking accuracy later)
        if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            objects_needing_cpu_sync.insert(sel.selected.object->nodeName);
        }
        for (const auto& item : sel.multi_selection) {
            if (item.type == SelectableType::Object && item.object) {
                objects_needing_cpu_sync.insert(item.object->nodeName);
            }
        }

        Vec3 newPos(objectMatrix[12], objectMatrix[13], objectMatrix[14]);

        // -------------------------------------------------------------------------
        // SINGULARITY FIX: Clamp extreme movements (Axis parallel to Camera View)
        // -------------------------------------------------------------------------
        if (operation == ImGuizmo::TRANSLATE) {
            // -------------------------------------------------------------------------
            // STABILITY FIX: Zero-Drift Check (If mouse is still, object stays still)
            // -------------------------------------------------------------------------
            float mouse_delta_sq = io.MouseDelta.x * io.MouseDelta.x + io.MouseDelta.y * io.MouseDelta.y;
            if (mouse_delta_sq < 0.01f) {
                // Determine if we should force-reset the position
                // If the user's hand is steady, we reject ANY jitter from the gizmo projection math.
                newPos = oldGizmoPos;
                objectMatrix[12] = oldGizmoPos.x;
                objectMatrix[13] = oldGizmoPos.y;
                objectMatrix[14] = oldGizmoPos.z;
            } else {   

            float dist_to_cam = (oldGizmoPos - cam.lookfrom).length();
            // Estimate world size of 1 pixel at object depth
            float pixel_world_size = dist_to_cam * (2.0f * tan_half_fov) / io.DisplaySize.y;

            float mouse_move_len = sqrtf(io.MouseDelta.x * io.MouseDelta.x + io.MouseDelta.y * io.MouseDelta.y);
            if (mouse_move_len < 1.0f) mouse_move_len = 1.0f;

            // -------------------------------------------------------------------------
            // CONSISTENT SPEED LIMIT (Fixes "Acceleration" feel)
            // -------------------------------------------------------------------------
            // Instead of variable damping, we enforce a strict "max world-units per mouse-pixel" limit.
            // This ensures the object moves linearly with the hand, destroying the singularity.

            Vec3 move_vector = newPos - oldGizmoPos;
            
            // -------------------------------------------------------------------------
            // DIRECTION CORRECTION (Fixes "Stuck" wrong-way movement)
            // -------------------------------------------------------------------------
            // In singularities, 3D projection can flip sign (moving mouse Right goes -Z instead of +Z).
            // We project the 3D move back to 2D and compare with Mouse Delta.
            ImVec2 s1 = Project(oldGizmoPos);
            ImVec2 s2 = Project(newPos);
            
            if (s1.x > -5000 && s2.x > -5000) { // Valid projection
                float dx = s2.x - s1.x;
                float dy = s2.y - s1.y;
                float dot = dx * io.MouseDelta.x + dy * io.MouseDelta.y;
                
                // If dot < 0, the visual movement opposes the mouse movement!
                if (dot < -0.01f) {
                    move_vector = move_vector * -1.0f; // FLIP DIRECTION
                    newPos = oldGizmoPos + move_vector; // Update target
                }
            }
            
            float world_move_dist = move_vector.length();

            // Calculate "World Units per Mouse Pixel" ratio
            // If this ratio is huge (e.g. 100.0), it means 1 pixel movement caused 100 units jump (Singularity).
            // Normal 1:1 interaction is roughly ratio ~ 1.0 (relative to pixel_world_size).
            
            float safe_ratio = 4.0f; // Base speed multiplier

            // DYNAMIC RATIO ADJUSTMENT:
            // If movement is parallel to camera view (Singularity Case), strictly reduce the ratio.
            // This prevents "runaway sensitivity" when dragging objects far away along the Z-axis.
            if (world_move_dist > 0.0001f) {
                Vec3 move_dir_norm = move_vector / world_move_dist;
                Vec3 cam_dir = (oldGizmoPos - cam.lookfrom).normalize();
                float dot = fabsf(move_dir_norm.dot(cam_dir)); // 0.0 = Perpendicular, 1.0 = Parallel

                if (dot > 0.7f) {
                    // Linearly reduce ratio from 4.0 to 1.0 as angle becomes parallel
                    // 0.7 -> 4.0
                    // 1.0 -> 1.0
                    float t = (dot - 0.7f) / 0.3f; // 0..1
                    safe_ratio = 4.0f * (1.0f - t) + 1.0f * t;
                }
            }
            
            float max_allowed_dist = mouse_move_len * pixel_world_size * safe_ratio;

            if (world_move_dist > max_allowed_dist) {
                // The projection wants to move too fast. Clamp it to the speed limit.
                // We preserve direction but limit magnitude.
                Vec3 dir = move_vector.normalize();
                newPos = oldGizmoPos + dir * max_allowed_dist;

                // Sync matrix
                objectMatrix[12] = newPos.x;
                objectMatrix[13] = newPos.y;
                objectMatrix[14] = newPos.z;
            }
            } // End of stationary check
        }
        // -------------------------------------------------------------------------
        Vec3 deltaPos = newPos - oldGizmoPos;  // Calculate delta from BEFORE manipulation
        sel.selected.position = newPos; // Update gizmo/bbox center

        // CRITICAL FIX: Update rotation and scale from object's transform matrix
        // This ensures keyframes capture correct rotation/scale values
        if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            auto transformHandle = sel.selected.object->getTransformHandle();
            if (transformHandle) {
                Matrix4x4 objTransform = transformHandle->getFinal();

                // Extract rotation (Euler angles in degrees)
                // Assuming rotation order: Z * Y * X
                float sy = sqrtf(objTransform.m[0][0] * objTransform.m[0][0] + objTransform.m[1][0] * objTransform.m[1][0]);
                bool singular = sy < 1e-6f;

                if (!singular) {
                    sel.selected.rotation.x = atan2f(objTransform.m[2][1], objTransform.m[2][2]) * (180.0f / 3.14159265f);
                    sel.selected.rotation.y = atan2f(-objTransform.m[2][0], sy) * (180.0f / 3.14159265f);
                    sel.selected.rotation.z = atan2f(objTransform.m[1][0], objTransform.m[0][0]) * (180.0f / 3.14159265f);
                }
                else {
                    sel.selected.rotation.x = atan2f(-objTransform.m[1][2], objTransform.m[1][1]) * (180.0f / 3.14159265f);
                    sel.selected.rotation.y = atan2f(-objTransform.m[2][0], sy) * (180.0f / 3.14159265f);
                    sel.selected.rotation.z = 0.0f;
                }

                // Extract scale
                sel.selected.scale.x = sqrtf(objTransform.m[0][0] * objTransform.m[0][0] +
                    objTransform.m[1][0] * objTransform.m[1][0] +
                    objTransform.m[2][0] * objTransform.m[2][0]);
                sel.selected.scale.y = sqrtf(objTransform.m[0][1] * objTransform.m[0][1] +
                    objTransform.m[1][1] * objTransform.m[1][1] +
                    objTransform.m[2][1] * objTransform.m[2][1]);
                sel.selected.scale.z = sqrtf(objTransform.m[0][2] * objTransform.m[0][2] +
                    objTransform.m[1][2] * objTransform.m[1][2] +
                    objTransform.m[2][2] * objTransform.m[2][2]);
            }
        }

        // Check if this is multi-selection (handle mixed types: lights + objects together)
        bool is_multi_select = sel.multi_selection.size() > 1;

        if (is_multi_select) {
            // MULTI-SELECTION: Apply delta to ALL selected items (mixed types)
            float deltaMagnitude = sqrtf(deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y + deltaPos.z * deltaPos.z);

            // For Rotation/Scale, deltaPos might be zero, so we check operation too
            if (deltaMagnitude >= 0.0001f || operation != ImGuizmo::TRANSLATE) {
                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

                // Calculate Delta Matrix for Rotation/Scale
                Matrix4x4 newMat;
                newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
                newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
                newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
                newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

                Matrix4x4 deltaMat = newMat * oldMat.inverse();

                // Decompose
                Vec3 deltaTranslation(deltaMat.m[0][3], deltaMat.m[1][3], deltaMat.m[2][3]);
                Matrix4x4 deltaRotScale = deltaMat;
                deltaRotScale.m[0][3] = 0; deltaRotScale.m[1][3] = 0; deltaRotScale.m[2][3] = 0;

                for (auto& item : sel.multi_selection) {
                    if (item.type == SelectableType::Object && item.object) {
                        std::string targetName = item.object->nodeName;
                        if (targetName.empty()) targetName = "Unnamed";

                        auto it = mesh_cache.find(targetName);
                        if (it != mesh_cache.end() && !it->second.empty()) {
                            auto& firstTri = it->second[0].second;
                            auto th = firstTri->getTransformHandle();
                            
                            if (th) {
                                // Apply transform to the shared handle (most objects share one)
                                if (pivot_mode == 1) {
                                    // Individual Origins
                                    Vec3 pos(th->base.m[0][3], th->base.m[1][3], th->base.m[2][3]);
                                    th->base.m[0][3] = 0; th->base.m[1][3] = 0; th->base.m[2][3] = 0;
                                    th->setBase(deltaRotScale * th->base);
                                    th->base.m[0][3] = pos.x + deltaTranslation.x;
                                    th->base.m[1][3] = pos.y + deltaTranslation.y;
                                    th->base.m[2][3] = pos.z + deltaTranslation.z;
                                }
                                else {
                                    // Median Point
                                    th->setBase(deltaMat * th->base);
                                }
                                
                                // TLAS MODE: Update GPU instance transform (fast path)
                                bool using_gpu_tlas = ctx.optix_gpu_ptr && ctx.render_settings.use_optix && ctx.optix_gpu_ptr->isUsingTLAS();
                                if (using_gpu_tlas) {
                                    std::vector<int> inst_ids = ctx.optix_gpu_ptr->getInstancesByNodeName(targetName);
                                    if (!inst_ids.empty()) {
                                        float t[12];
                                        Matrix4x4& m = th->base;
                                        t[0] = m.m[0][0]; t[1] = m.m[0][1]; t[2] = m.m[0][2]; t[3] = m.m[0][3];
                                        t[4] = m.m[1][0]; t[5] = m.m[1][1]; t[6] = m.m[1][2]; t[7] = m.m[1][3];
                                        t[8] = m.m[2][0]; t[9] = m.m[2][1]; t[10] = m.m[2][2]; t[11] = m.m[2][3];
                                        
                                        for (int inst_id : inst_ids) {
                                            ctx.optix_gpu_ptr->updateInstanceTransform(inst_id, t);
                                        }
                                    }
                                    // NOTE: TLAS mode - NO CPU vertex update during drag! (saves millions of calls)
                                }
                                // CPU mode handled on release
                            }
                        }
                        item.has_cached_aabb = false;
                    }
                    else if (item.type == SelectableType::Light && item.light) {
                        item.light->position = item.light->position + deltaPos;
                    }
                    else if (item.type == SelectableType::Camera && item.camera) {
                        // Skip active camera - moving it would affect viewport
                        if (item.camera != ctx.scene.camera) {
                            item.camera->lookfrom = item.camera->lookfrom + deltaPos;
                            item.camera->lookat = item.camera->lookat + deltaPos;
                            item.camera->update_camera_vectors();
                        }
                    }
                    else if (item.type == SelectableType::GasVolume && item.gas_volume) {
                        item.gas_volume->setPosition(item.gas_volume->getPosition() + deltaPos);
                    }
                    else if (item.type == SelectableType::ForceField && item.force_field) {
                        item.force_field->position = item.force_field->position + deltaPos;
                        
                        if (operation == ImGuizmo::ROTATE) {
                            // Apply rotation delta to direction vector (manual matrix multiply)
                            Vec3 d = item.force_field->direction;
                            Vec3 newDir(
                                deltaRotScale.m[0][0]*d.x + deltaRotScale.m[0][1]*d.y + deltaRotScale.m[0][2]*d.z,
                                deltaRotScale.m[1][0]*d.x + deltaRotScale.m[1][1]*d.y + deltaRotScale.m[1][2]*d.z,
                                deltaRotScale.m[2][0]*d.x + deltaRotScale.m[2][1]*d.y + deltaRotScale.m[2][2]*d.z
                            );
                            item.force_field->direction = newDir.normalize();
                            
                            // Also update vortex axis if applicable
                            if (item.force_field->type == Physics::ForceFieldType::Vortex) {
                                Vec3 a = item.force_field->axis;
                                Vec3 newAxis(
                                    deltaRotScale.m[0][0]*a.x + deltaRotScale.m[0][1]*a.y + deltaRotScale.m[0][2]*a.z,
                                    deltaRotScale.m[1][0]*a.x + deltaRotScale.m[1][1]*a.y + deltaRotScale.m[1][2]*a.z,
                                    deltaRotScale.m[2][0]*a.x + deltaRotScale.m[2][1]*a.y + deltaRotScale.m[2][2]*a.z
                                );
                                item.force_field->axis = newAxis.normalize();
                            }
                        }
                    }
                } // End of multi_selection loop

                // Trigger TLAS Update after processing all objects
                if (ctx.optix_gpu_ptr && ctx.optix_gpu_ptr->isUsingTLAS()) {
                    // Use fast matrix-only update instead of full rebuild
                    ctx.optix_gpu_ptr->updateTLASMatricesOnly(ctx.scene.world.objects);
                    ctx.optix_gpu_ptr->resetAccumulation();
                }

                sel.selected.has_cached_aabb = false;

                // DEFERRED UPDATE: Only mark dirty during drag (for CPU mode)
                bool using_gpu_tlas = ctx.optix_gpu_ptr && ctx.render_settings.use_optix && ctx.optix_gpu_ptr->isUsingTLAS();
                if (!using_gpu_tlas) {
                    is_bvh_dirty = true;
                }
            }
        }
        else if (sel.selected.type == SelectableType::Light && sel.selected.light) {
            sel.selected.light->position = newPos;
            Vec3 zAxis(objectMatrix[8], objectMatrix[9], objectMatrix[10]);
            Vec3 newDir = -zAxis.normalize(); // Gizmo -Z aligned

            if (auto dl = std::dynamic_pointer_cast<DirectionalLight>(sel.selected.light)) dl->setDirection(newDir);
            else if (auto sl = std::dynamic_pointer_cast<SpotLight>(sel.selected.light)) {
                sl->direction = newDir;

                // Update Angle from Gizmo Scale
                Vec3 right(objectMatrix[0], objectMatrix[1], objectMatrix[2]);
                float angle = right.length();

                if (angle < 0.1f) angle = 0.1f;
                if (angle > 179.0f) angle = 179.0f;

                sl->setAngleDegrees(angle);

                // Falloff Update (Z Scale represents 1.0 + Falloff)
                Vec3 forward(objectMatrix[8], objectMatrix[9], objectMatrix[10]);
                float sz = forward.length();
                float newF = sz - 1.0f;
                if (newF < 0.0f) newF = 0.0f;
                if (newF > 1.0f) newF = 1.0f;
                sl->setFalloff(newF);
            }
            else if (auto al = std::dynamic_pointer_cast<AreaLight>(sel.selected.light)) {
                Vec3 right(objectMatrix[0], objectMatrix[1], objectMatrix[2]);
                Vec3 forward(objectMatrix[8], objectMatrix[9], objectMatrix[10]);

                // Scale bilgisini vektörlerden çıkar
                float sx = right.length();
                float sz = forward.length();

                // Width ve Height güncelle
                if (sx > 0.001f) al->width = sx;
                if (sz > 0.001f) al->height = sz;

                // u ve v HER ZAMAN normalize tutulmalı!
                if (sx > 0.001f) al->u = right / sx;
                if (sz > 0.001f) al->v = forward / sz;

                // Position doğrudan gizmo merkezinden alınmalı (artık position = merkez)
                al->position = newPos;
            }
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
        }
        else if (sel.selected.type == SelectableType::ForceField && sel.selected.force_field) {
            // Apply matrix to Force Field
            Matrix4x4 newMat;
            newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
            newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
            newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
            newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

            Vec3 p, r, s;
            newMat.decompose(p, r, s);
            
            sel.selected.force_field->position = p;
            sel.selected.force_field->rotation = r;
            sel.selected.force_field->scale = s;
            
            // Update selection cache
            sel.selected.position = p;
            sel.selected.rotation = r;
            sel.selected.scale = s;
            
            // Removed direction override here, because ForceField direction is a LOCAL property
            // and should only be explicitly set in the UI, not implicitly by gizmo rotation
            // (The Gizmo rotation already rotates the final world force).

            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->resetAccumulation();
            }
        }
        else if (sel.selected.type == SelectableType::Camera && sel.selected.camera) {
            // Skip active camera - moving it would affect viewport directly
            if (sel.selected.camera != ctx.scene.camera) {
                Vec3 delta = newPos - sel.selected.camera->lookfrom;
                sel.selected.camera->lookfrom = newPos;
                sel.selected.camera->lookat = sel.selected.camera->lookat + delta;
                sel.selected.camera->update_camera_vectors();
                // Note: Don't call setCameraParams for non-active cameras
            }
        }
        else if (sel.selected.type == SelectableType::VDBVolume && sel.selected.vdb_volume) {
             Matrix4x4 newMat;
             newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
             newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
             newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
             newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

             sel.selected.vdb_volume->setTransform(newMat);
             
             // Update Selection struct to match new transform
             Vec3 p, r, s;
             newMat.decompose(p, r, s);
             sel.selected.position = p;
             sel.selected.rotation = r;
             sel.selected.scale = s;
             
             // VDB uses BVH, so we need a rebuild/refit
             // Since it's a box, refit is usually enough, but rebuild is safer for topology changes (though box just moves)
             extern bool g_bvh_rebuild_pending;
             g_bvh_rebuild_pending = true;
             
             ctx.renderer.resetCPUAccumulation();
             
             if (ctx.optix_gpu_ptr) {
                 SceneUI::syncVDBVolumesToGPU(ctx);
                 ctx.optix_gpu_ptr->resetAccumulation();
             }
        }
        else if (sel.selected.type == SelectableType::GasVolume && sel.selected.gas_volume) {
             Matrix4x4 newMat;
             newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
             newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
             newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
             newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

             Vec3 p, r, s;
             newMat.decompose(p, r, s);
             sel.selected.gas_volume->setPosition(p);
             sel.selected.gas_volume->setRotation(r);
             sel.selected.gas_volume->setScale(s);
             
             // Update selection struct to match new transform
             sel.selected.position = p;
             sel.selected.rotation = r;
             sel.selected.scale = s;
             
             ctx.renderer.resetCPUAccumulation();
             
             if (ctx.optix_gpu_ptr) {
                 if (sel.selected.gas_volume->render_path == GasVolume::VolumeRenderPath::VDBUnified) {
                     SceneUI::syncVDBVolumesToGPU(ctx);
                 } else {
                     ctx.renderer.updateOptiXGasVolumes(ctx.scene, ctx.optix_gpu_ptr);
                 }
                 ctx.optix_gpu_ptr->resetAccumulation();
             }
        }
        else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            // SINGLE SELECTION or Rotate/Scale operations
            // (Multi-select TRANSLATE is handled above)

            Matrix4x4 newMat;
            newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
            newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
            newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
            newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

            float deltaMagnitude = sqrtf(deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y + deltaPos.z * deltaPos.z);

            // Only apply transform if there's significant movement or it's rotate/scale
            if (deltaMagnitude >= 0.0001f || operation != ImGuizmo::TRANSLATE) {
                if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);

                std::string targetName = sel.selected.object->nodeName;
                if (targetName.empty()) targetName = "Unnamed";

                auto it = mesh_cache.find(targetName);
                if (it != mesh_cache.end() && !it->second.empty()) {
                    auto& firstTri = it->second[0].second;
                    auto t_handle = firstTri->getTransformHandle();

                    // Safety check for mixed transforms - OPTIMIZED: Only check first few triangles
                    // Most objects either share one transform or have completely different ones
                    bool all_same_transform = true;
                    const size_t MAX_CHECK = std::min((size_t)100, it->second.size()); // Check up to 100, not 2M
                    for (size_t i = 1; i < MAX_CHECK && all_same_transform; ++i) {
                        auto h = it->second[i].second->getTransformHandle();
                        if (h.get() != t_handle.get()) all_same_transform = false;
                    }

                    if (all_same_transform && t_handle) {
                        // Apply full matrix from gizmo (supports translate, rotate, scale)
                        t_handle->setBase(newMat);

                        // TLAS INSTANCING UPDATE (Fast GPU Path)
                        // Only use GPU path if both: OptiX enabled AND using TLAS mode
                        bool using_gpu_tlas = ctx.optix_gpu_ptr && ctx.render_settings.use_optix && ctx.optix_gpu_ptr->isUsingTLAS();
                        if (using_gpu_tlas) {
                             // Use unified update method
                             ctx.optix_gpu_ptr->updateObjectTransform(targetName, newMat);
                        }
                        else {
                            // CPU/GAS MODE: Update CPU vertices (required for BVH/picking)
                            for (auto& pair : it->second) {
                                pair.second->updateTransformedVertices();
                            }
                            
                            is_bvh_dirty = true;
                            
                            // Trigger Fast Refit during interaction (CPU Mode only)
                            extern bool g_cpu_bvh_refit_pending;
                            g_cpu_bvh_refit_pending = true;
                        }
                    }
                    else {
                        // Fallback: Mixed transforms, apply delta MATRIX to each unique transform (Supports all ops)
                        Matrix4x4 newMat;
                        newMat.m[0][0] = objectMatrix[0]; newMat.m[1][0] = objectMatrix[1]; newMat.m[2][0] = objectMatrix[2]; newMat.m[3][0] = objectMatrix[3];
                        newMat.m[0][1] = objectMatrix[4]; newMat.m[1][1] = objectMatrix[5]; newMat.m[2][1] = objectMatrix[6]; newMat.m[3][1] = objectMatrix[7];
                        newMat.m[0][2] = objectMatrix[8]; newMat.m[1][2] = objectMatrix[9]; newMat.m[2][2] = objectMatrix[10]; newMat.m[3][2] = objectMatrix[11];
                        newMat.m[0][3] = objectMatrix[12]; newMat.m[1][3] = objectMatrix[13]; newMat.m[2][3] = objectMatrix[14]; newMat.m[3][3] = objectMatrix[15];

                        Matrix4x4 deltaMat = newMat * oldMat.inverse();

                        // Decompose for Individual Origins logic
                        Vec3 deltaTranslation(deltaMat.m[0][3], deltaMat.m[1][3], deltaMat.m[2][3]);
                        Matrix4x4 deltaRotScale = deltaMat;
                        deltaRotScale.m[0][3] = 0; deltaRotScale.m[1][3] = 0; deltaRotScale.m[2][3] = 0;

                        std::unordered_set<Transform*> processed_transforms;
                        for (auto& pair : it->second) {
                            auto tri = pair.second;
                            auto th = tri->getTransformHandle();
                            if (th && processed_transforms.find(th.get()) == processed_transforms.end()) {
                                if (pivot_mode == 1) {
                                    // Individual Origins
                                    Vec3 pos(th->base.m[0][3], th->base.m[1][3], th->base.m[2][3]);
                                    th->base.m[0][3] = 0; th->base.m[1][3] = 0; th->base.m[2][3] = 0;
                                    th->setBase(deltaRotScale * th->base);
                                    th->base.m[0][3] = pos.x + deltaTranslation.x;
                                    th->base.m[1][3] = pos.y + deltaTranslation.y;
                                    th->base.m[2][3] = pos.z + deltaTranslation.z;
                                }
                                else {
                                    // Median Point
                                    th->setBase(deltaMat * th->base);
                                }
                                processed_transforms.insert(th.get());

                                // TLAS INSTANCING UPDATE (Fast Path for Multi-Select)
                                if (ctx.optix_gpu_ptr && ctx.optix_gpu_ptr->isUsingTLAS()) {
                                    // Use unified update method
                                    ctx.optix_gpu_ptr->updateObjectTransform(targetName, th->base);
                                } else {
                                    // CPU Mode: MUST update vertices for BVH refit/rebuild to see changes
                                    tri->updateTransformedVertices();
                                }
                            }
                        }
                    }
                }

                sel.selected.has_cached_aabb = false;

                // DEFERRED UPDATE: Mark dirty when NOT using GPU rendering
                // Check use_optix setting (from render_settings) not just isUsingTLAS
                bool using_gpu_render = ctx.optix_gpu_ptr && ctx.render_settings.use_optix && ctx.optix_gpu_ptr->isUsingTLAS();
                if (!using_gpu_render) {
                    is_bvh_dirty = true;
                    extern bool g_cpu_bvh_refit_pending;
                    g_cpu_bvh_refit_pending = true;
                }
            }
        }
    }

    // DEFERRED BVH UPDATE: Rebuild when gizmo drag ends (not during)
    // This check is at the END so is_bvh_dirty has been set above
    if (!is_using && was_using_gizmo && is_bvh_dirty) {
        // SCENE_LOG_INFO("[GIZMO] Released - Triggering deferred geometry update");
        // Check actual render mode, not just pointer existence
        bool using_gpu = ctx.optix_gpu_ptr && ctx.render_settings.use_optix;
        
        if (using_gpu && ctx.optix_gpu_ptr->isUsingTLAS()) {
            // TLAS MODE: Commits all pending transform changes
            // During drag, updateInstanceTransform() queues changes but doesn't rebuild TLAS.
            // On release, we must rebuild TLAS to apply those transforms to GPU.
            ctx.optix_gpu_ptr->rebuildTLAS();
        } else if (using_gpu) {
            // GAS MODE: Defer update to Main loop to avoid UI freeze
            extern bool g_gpu_refit_pending;
            g_gpu_refit_pending = true;
            
            // Only GAS mode needs CPU BVH rebuild because vertex positions change
            extern bool g_bvh_rebuild_pending;
            g_bvh_rebuild_pending = true;
        } else {
            // No OptiX / CPU rendering: needs BVH rebuild
            extern bool g_bvh_rebuild_pending;
            g_bvh_rebuild_pending = true;
        }
        
        is_bvh_dirty = false;
    }
    
    // LAZY CPU SYNC: Mark objects for later sync instead of updating now
    // This makes gizmo release INSTANT - sync happens when user tries to pick something
    if (!is_using && was_using_gizmo) {
        bool using_gpu_tlas = ctx.optix_gpu_ptr && ctx.render_settings.use_optix && ctx.optix_gpu_ptr->isUsingTLAS();
        
        if (sel.multi_selection.size() > 0) {
            for (auto& item : sel.multi_selection) {
                if (item.type == SelectableType::Object && item.object) {
                    std::string name = item.object->nodeName;
                    if (name.empty()) name = "Unnamed";
                    
                    if (using_gpu_tlas) {
                        // TLAS mode: Just mark for lazy sync (instant release!)
                        objects_needing_cpu_sync.insert(name);
                    } else {
                        // CPU mode: Need immediate update for rendering
                        auto cache_it = mesh_cache.find(name);
                        if (cache_it != mesh_cache.end()) {
                            for (auto& pair : cache_it->second) {
                                pair.second->updateTransformedVertices();
                            }
                        }
                    }
                }
            }
            // BVH rebuild needed for both GPU and CPU - for accurate picking!
            extern bool g_bvh_rebuild_pending;
            g_bvh_rebuild_pending = true;
        } else if (sel.selected.type == SelectableType::Object && sel.selected.object) {
            std::string name = sel.selected.object->nodeName;
            if (name.empty()) name = "Unnamed";
            
            if (using_gpu_tlas) {
                // TLAS mode: Just mark for lazy sync (instant release!)
                objects_needing_cpu_sync.insert(name);
                SCENE_LOG_INFO("Marked for lazy sync: " + name);
            } else {
                // CPU mode: Need immediate update
                auto cache_it = mesh_cache.find(name);
                if (cache_it != mesh_cache.end()) {
                    for (auto& pair : cache_it->second) {
                        pair.second->updateTransformedVertices();
                    }
                }
            }
            // BVH rebuild needed for both GPU and CPU - for accurate picking!
            extern bool g_bvh_rebuild_pending;
            g_bvh_rebuild_pending = true;
        }
    }

    // Update gizmo state tracking at the END of the function
    was_using_gizmo = is_using;
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAMERA GIZMOS - Draw camera icons in viewport
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawCameraGizmos(UIContext& ctx) {
    if (!ctx.scene.camera || ctx.scene.cameras.size() <= 1) return;

    Camera& activeCam = *ctx.scene.camera;
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();  // Changed from ForegroundDrawList to render behind UI panels
    ImGuiIO& io = ImGui::GetIO();
    float screen_w = io.DisplaySize.x;
    float screen_h = io.DisplaySize.y;

    // Camera basis vectors for projection
    Vec3 cam_forward = (activeCam.lookat - activeCam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(activeCam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    float tan_half_fov = tan(activeCam.vfov * 0.5f * M_PI / 180.0f);
    float aspect = screen_w / screen_h;

    // Lambda to project 3D point to screen
    auto Project = [&](const Vec3& world_pos, ImVec2& screen_pos) -> bool {
        Vec3 to_point = world_pos - activeCam.lookfrom;
        float depth = to_point.dot(cam_forward);
        if (depth < 0.1f) return false;  // Behind camera

        float local_x = to_point.dot(cam_right);
        float local_y = to_point.dot(cam_up);

        float half_height = depth * tan_half_fov;
        float half_width = half_height * aspect;

        float ndc_x = local_x / half_width;
        float ndc_y = local_y / half_height;

        if (fabs(ndc_x) > 1.2f || fabs(ndc_y) > 1.2f) return false;  // Outside frustum

        screen_pos.x = (ndc_x * 0.5f + 0.5f) * screen_w;
        screen_pos.y = (0.5f - ndc_y * 0.5f) * screen_h;
        return true;
        };

    // Draw each non-active camera with 3D frustum
    for (size_t i = 0; i < ctx.scene.cameras.size(); ++i) {
        if (i == ctx.scene.active_camera_index) continue;  // Skip active camera

        auto& cam = ctx.scene.cameras[i];
        if (!cam) continue;

        // Check if this camera is selected
        bool is_selected = (ctx.selection.hasSelection() &&
            ctx.selection.selected.type == SelectableType::Camera &&
            ctx.selection.selected.camera == cam);

        // Frustum colors
        ImU32 frustum_color = is_selected ? IM_COL32(255, 200, 50, 220) : IM_COL32(100, 200, 255, 180);
        ImU32 body_color = is_selected ? IM_COL32(255, 200, 50, 255) : IM_COL32(80, 150, 255, 255);
        float line_thickness = is_selected ? 2.5f : 1.5f;

        // Calculate frustum vertices in world space
        Vec3 cam_pos = cam->lookfrom;
        Vec3 look_dir = (cam->lookat - cam->lookfrom).normalize();
        Vec3 cam_right = look_dir.cross(cam->vup).normalize();
        Vec3 cam_up = cam_right.cross(look_dir).normalize();

        // Frustum dimensions at near and far planes
        float frustum_length = 1.5f;  // Length of frustum visualization
        float near_dist = 0.2f;
        float far_dist = frustum_length;
        float cam_fov_rad = cam->vfov * M_PI / 180.0f;
        float cam_aspect = screen_w / screen_h;

        float near_height = near_dist * tan(cam_fov_rad * 0.5f);
        float near_width = near_height * cam_aspect;
        float far_height = far_dist * tan(cam_fov_rad * 0.5f);
        float far_width = far_height * cam_aspect;

        // Near plane corners
        Vec3 near_center = cam_pos + look_dir * near_dist;
        Vec3 near_tl = near_center + cam_up * near_height - cam_right * near_width;
        Vec3 near_tr = near_center + cam_up * near_height + cam_right * near_width;
        Vec3 near_bl = near_center - cam_up * near_height - cam_right * near_width;
        Vec3 near_br = near_center - cam_up * near_height + cam_right * near_width;

        // Far plane corners
        Vec3 far_center = cam_pos + look_dir * far_dist;
        Vec3 far_tl = far_center + cam_up * far_height - cam_right * far_width;
        Vec3 far_tr = far_center + cam_up * far_height + cam_right * far_width;
        Vec3 far_bl = far_center - cam_up * far_height - cam_right * far_width;
        Vec3 far_br = far_center - cam_up * far_height + cam_right * far_width;

        // Project all points
        ImVec2 p_cam, p_near_tl, p_near_tr, p_near_bl, p_near_br;
        ImVec2 p_far_tl, p_far_tr, p_far_bl, p_far_br;

        bool visible = Project(cam_pos, p_cam);
        visible &= Project(near_tl, p_near_tl) && Project(near_tr, p_near_tr);
        visible &= Project(near_bl, p_near_bl) && Project(near_br, p_near_br);
        visible &= Project(far_tl, p_far_tl) && Project(far_tr, p_far_tr);
        visible &= Project(far_bl, p_far_bl) && Project(far_br, p_far_br);

        if (!visible) continue;  // Skip if frustum is behind camera or off-screen

        // Draw frustum lines from camera to far plane
        draw_list->AddLine(p_cam, p_far_tl, frustum_color, line_thickness);
        draw_list->AddLine(p_cam, p_far_tr, frustum_color, line_thickness);
        draw_list->AddLine(p_cam, p_far_bl, frustum_color, line_thickness);
        draw_list->AddLine(p_cam, p_far_br, frustum_color, line_thickness);

        // Draw near plane rectangle
        draw_list->AddLine(p_near_tl, p_near_tr, frustum_color, line_thickness);
        draw_list->AddLine(p_near_tr, p_near_br, frustum_color, line_thickness);
        draw_list->AddLine(p_near_br, p_near_bl, frustum_color, line_thickness);
        draw_list->AddLine(p_near_bl, p_near_tl, frustum_color, line_thickness);

        // Draw far plane rectangle (thicker to show viewing direction endpoint)
        draw_list->AddLine(p_far_tl, p_far_tr, frustum_color, line_thickness * 1.2f);
        draw_list->AddLine(p_far_tr, p_far_br, frustum_color, line_thickness * 1.2f);
        draw_list->AddLine(p_far_br, p_far_bl, frustum_color, line_thickness * 1.2f);
        draw_list->AddLine(p_far_bl, p_far_tl, frustum_color, line_thickness * 1.2f);

        // Draw camera body at position (small filled circle)
        float body_size = 6.0f;
        draw_list->AddCircleFilled(p_cam, body_size, body_color);
        draw_list->AddCircle(p_cam, body_size, IM_COL32(255, 255, 255, 255), 0, 1.5f);

        // Camera label
        std::string label = cam->nodeName.empty() ? "Cam " + std::to_string(i) : cam->nodeName;
        ImVec2 text_pos(p_cam.x + body_size + 5, p_cam.y - 8);
        draw_list->AddText(text_pos, IM_COL32(255, 255, 255, 220), label.c_str());
    }
}

void SceneUI::drawForceFieldGizmos(UIContext& ctx, bool& gizmo_hit) {
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImGuiIO& io = ImGui::GetIO();
    Camera& cam = *ctx.scene.camera;
    SceneSelection& sel = ctx.selection;

    Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);
    float aspect = io.DisplaySize.x / io.DisplaySize.y;

    auto Project = [&](const Vec3& p) -> ImVec2 {
        Vec3 to_point = p - cam.lookfrom;
        float depth = to_point.dot(cam_forward);
        if (depth <= 0.1f) return ImVec2(-10000, -10000);
        float local_x = to_point.dot(cam_right);
        float local_y = to_point.dot(cam_up);
        float half_h = depth * tan_half_fov;
        float half_w = half_h * aspect;
        return ImVec2(((local_x / half_w) * 0.5f + 0.5f) * io.DisplaySize.x, (0.5f - (local_y / half_h) * 0.5f) * io.DisplaySize.y);
    };

    for (const auto& ff : ctx.scene.force_field_manager.force_fields) {
        if (!ff->enabled || !ff->visible) continue;

        ImVec2 screen_pos = Project(ff->position);
        if (screen_pos.x < -5000) continue;

        bool is_selected = (sel.selected.type == SelectableType::ForceField && sel.selected.force_field == ff);
        ImU32 color = is_selected ? IM_COL32(255, 100, 255, 255) : IM_COL32(200, 100, 200, 180);

        // Picking check for Icon
        float mouse_dist = sqrtf(powf(io.MousePos.x - screen_pos.x, 2) + powf(io.MousePos.y - screen_pos.y, 2));
        if (mouse_dist < 15.0f && ImGui::IsMouseClicked(0) && !ImGuizmo::IsOver()) {
            sel.selectForceField(ff, -1, ff->name);
            ForceFieldUI::selected_force_field = ff;
            gizmo_hit = true;
        }

        // Draw Icon
        float size = 12.0f;
        draw_list->AddCircleFilled(screen_pos, size, IM_COL32(30, 30, 30, 200));
        draw_list->AddCircle(screen_pos, size, color, 0, 2.0f);
        
        // Symbol based on type
        const char* symbol = "?";
        bool has_direction = false;
        switch (ff->type) {
            case Physics::ForceFieldType::Wind:      symbol = "W"; has_direction = true; break;
            case Physics::ForceFieldType::Vortex:    symbol = "@"; break;
            case Physics::ForceFieldType::Gravity:   symbol = "G"; has_direction = true; break;
            case Physics::ForceFieldType::Magnetic:  symbol = "M"; has_direction = true; break;
            case Physics::ForceFieldType::Turbulence: symbol = "~"; break;
            case Physics::ForceFieldType::Attractor:  symbol = "+"; break;
            case Physics::ForceFieldType::Repeller:   symbol = "-"; break;
            default: break;
        }

        ImVec2 text_size = ImGui::GetFont()->CalcTextSizeA(ImGui::GetFontSize(), 100.0f, 0.0f, symbol);
        draw_list->AddText(ImVec2(screen_pos.x - text_size.x * 0.5f, screen_pos.y - text_size.y * 0.5f), color, symbol);

        // Draw Direction Arrow
        if (has_direction) {
            Matrix4x4 rot = Matrix4x4::rotationX(ff->rotation.x * 0.0174533f) * 
                            Matrix4x4::rotationY(ff->rotation.y * 0.0174533f) * 
                            Matrix4x4::rotationZ(ff->rotation.z * 0.0174533f);
            
            Vec3 local_dir = ff->direction;
            if (local_dir.length() < 0.001f) {
                // If it's a directional force but direction is 0,0,0, assume -Y (down) default
                local_dir = Vec3(0, -1, 0);
            }
            
            Vec3 world_dir = rot.transform_vector(local_dir).normalize();
            
            // Scale arrow length based on strength to visually indicate force magnitude
            float arrow_len = 1.5f + std::abs(ff->strength) * 0.1f;
            if (arrow_len > 15.0f) arrow_len = 15.0f; // Cap max length
            
            Vec3 arrow_end = ff->position + world_dir * arrow_len;
            ImVec2 screen_end = Project(arrow_end);
            if (screen_end.x > -5000) {
                draw_list->AddLine(screen_pos, screen_end, color, 2.0f);
                // Simple arrow head
                Vec3 right = world_dir.cross(cam_up).normalize() * 0.2f;
                ImVec2 h1 = Project(arrow_end - world_dir * 0.3f + right);
                ImVec2 h2 = Project(arrow_end - world_dir * 0.3f - right);
                if (h1.x > -5000 && h2.x > -5000) {
                    draw_list->AddLine(screen_end, h1, color, 2.0f);
                    draw_list->AddLine(screen_end, h2, color, 2.0f);
                }
            }
        }

        // Show name and strength for ALL force fields in the viewport
        char label[256];
        snprintf(label, sizeof(label), "%s (Str: %.1f)", ff->name.c_str(), ff->strength);
        draw_list->AddText(ImVec2(screen_pos.x + size + 5, screen_pos.y - 8), color, label);

        if (is_selected) {
            // Draw Radius for non-infinite
            if (ff->shape != Physics::ForceFieldShape::Infinite) {
                float r = ff->falloff_radius;
                for (int plane = 0; plane < 3; ++plane) {
                    const int segs = 32;
                    ImVec2 last_p;
                    for (int i = 0; i <= segs; ++i) {
                        float a = i * (6.28318f / segs);
                        Vec3 p3d;
                        if (plane == 0) p3d = ff->position + Vec3(cosf(a)*r, sinf(a)*r, 0);
                        else if (plane == 1) p3d = ff->position + Vec3(cosf(a)*r, 0, sinf(a)*r);
                        else p3d = ff->position + Vec3(0, cosf(a)*r, sinf(a)*r);
                        
                        ImVec2 p_screen = Project(p3d);
                        if (i > 0 && p_screen.x > -5000 && last_p.x > -5000) {
                            draw_list->AddLine(last_p, p_screen, IM_COL32(255, 0, 255, 80), 1.0f);
                        }
                        last_p = p_screen;
                    }
                }
            }
        }
    }
}
void SceneUI::drawSelectionGizmos(UIContext& ctx)
{
    if (ctx.selection.hasSelection() && ctx.selection.show_gizmo && ctx.scene.camera && viewport_settings.show_gizmos) {
        drawSelectionBoundingBox(ctx);
        drawTransformGizmo(ctx);
    }
}
