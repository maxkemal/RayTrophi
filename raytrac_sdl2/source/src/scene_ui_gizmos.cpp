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


// ═════════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════════
// SELECTION BOUNDING BOX DRAWING (Multi-selection support)
// ═══════════════════════════════════════════════════════════════════════════════
void SceneUI::drawSelectionBoundingBox(UIContext& ctx) {
    SceneSelection& sel = ctx.selection;
    if (!sel.hasSelection() || !ctx.scene.camera) return;

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

    // Helper lambda to draw a bounding box
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

        ImVec2 screen_pts[8];
        bool all_visible = true;

        for (int i = 0; i < 8; i++) {
            Vec3 to_corner = corners[i] - cam.lookfrom;
            float depth = to_corner.dot(cam_forward);

            if (depth <= 0.01f) {
                all_visible = false;
                break;
            }

            float local_x = to_corner.dot(cam_right);
            float local_y = to_corner.dot(cam_up);

            float half_height = depth * tan_half_fov;
            float half_width = half_height * aspect_ratio;

            float ndc_x = local_x / half_width;
            float ndc_y = local_y / half_height;

            screen_pts[i].x = (ndc_x * 0.5f + 0.5f) * screen_w;
            screen_pts[i].y = (0.5f - ndc_y * 0.5f) * screen_h;
        }

        if (!all_visible) return;

        ImDrawList* draw_list = ImGui::GetBackgroundDrawList();  // Draw behind UI panels

        draw_list->AddLine(screen_pts[0], screen_pts[1], color, thickness);
        draw_list->AddLine(screen_pts[1], screen_pts[2], color, thickness);
        draw_list->AddLine(screen_pts[2], screen_pts[3], color, thickness);
        draw_list->AddLine(screen_pts[3], screen_pts[0], color, thickness);

        draw_list->AddLine(screen_pts[4], screen_pts[5], color, thickness);
        draw_list->AddLine(screen_pts[5], screen_pts[6], color, thickness);
        draw_list->AddLine(screen_pts[6], screen_pts[7], color, thickness);
        draw_list->AddLine(screen_pts[7], screen_pts[4], color, thickness);

        draw_list->AddLine(screen_pts[0], screen_pts[4], color, thickness);
        draw_list->AddLine(screen_pts[1], screen_pts[5], color, thickness);
        draw_list->AddLine(screen_pts[2], screen_pts[6], color, thickness);
        draw_list->AddLine(screen_pts[3], screen_pts[7], color, thickness);
        };

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
                            m.m[0][0]*p.x + m.m[0][1]*p.y + m.m[0][2]*p.z + m.m[0][3],
                            m.m[1][0]*p.x + m.m[1][1]*p.y + m.m[1][2]*p.z + m.m[1][3],
                            m.m[2][0]*p.x + m.m[2][1]*p.y + m.m[2][2]*p.z + m.m[2][3]
                        );
                        bb_min.x = fminf(bb_min.x, tp.x);
                        bb_min.y = fminf(bb_min.y, tp.y);
                        bb_min.z = fminf(bb_min.z, tp.z);
                        bb_max.x = fmaxf(bb_max.x, tp.x);
                        bb_max.y = fmaxf(bb_max.y, tp.y);
                        bb_max.z = fmaxf(bb_max.z, tp.z);
                    }
                } else {
                    // No transform - use cached values directly
                    bb_min = cached_min;
                    bb_max = cached_max;
                }
                has_bounds = true;
            }
        }
        else if (item.type == SelectableType::Light && item.light) {
            Vec3 lightPos = item.light->position;
            float boxSize = 0.5f;
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

        if (has_bounds) {
            DrawBoundingBox(bb_min, bb_max, color, thickness);
        }
    }
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

        bool selected =
            (ctx.selection.selected.type == SelectableType::Light &&
                ctx.selection.selected.light == light);

        ImU32 col = selected
            ? IM_COL32(255, 100, 50, 255)
            : IM_COL32(255, 255, 100, 180);

        Vec3 pos = light->position;
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

        // ---- POINT (Diamond) ----
        if (light->type() == LightType::Point) {
            float r = 0.2f;
            Vec3 pts[6] = {
                pos + Vec3(0, r, 0), pos + Vec3(0, -r, 0),
                pos + Vec3(r, 0, 0), pos + Vec3(-r, 0, 0),
                pos + Vec3(0, 0, r), pos + Vec3(0, 0, -r)
            };

            ImVec2 s[6];
            for (int i = 0; i < 6; ++i) s[i] = Project(pts[i]);

            draw_list->AddCircleFilled(center, 4.0f,
                IM_COL32(255, 255, 200, 200));

            draw_list->AddLine(s[2], s[4], col); draw_list->AddLine(s[4], s[3], col);
            draw_list->AddLine(s[3], s[5], col); draw_list->AddLine(s[5], s[2], col);
            draw_list->AddLine(s[0], s[2], col); draw_list->AddLine(s[0], s[3], col);
            draw_list->AddLine(s[0], s[4], col); draw_list->AddLine(s[0], s[5], col);
            draw_list->AddLine(s[1], s[2], col); draw_list->AddLine(s[1], s[3], col);
            draw_list->AddLine(s[1], s[4], col); draw_list->AddLine(s[1], s[5], col);
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

            Vec3 u = al->getU();
            Vec3 v = al->getV();

            ImVec2 c1 = Project(pos);
            ImVec2 c2 = Project(pos + u);
            ImVec2 c3 = Project(pos + u + v);
            ImVec2 c4 = Project(pos + v);

            draw_list->AddLine(c1, c2, col);
            draw_list->AddLine(c2, c3, col);
            draw_list->AddLine(c3, c4, col);
            draw_list->AddLine(c4, c1, col);
            draw_list->AddLine(c1, c3, col, 1.0f);
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

    // Fix AreaLight Pivot: Use Center instead of Corner
    if (sel.selected.type == SelectableType::Light) {
        if (auto al = std::dynamic_pointer_cast<AreaLight>(sel.selected.light)) {
            pos = al->position + al->u * 0.5f + al->v * 0.5f;
        }
    }

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
            // Use actual magnitude for Scale interaction
            Vec3 X = al->u;
            Vec3 Z = al->v;
            // Normalized Y (Normal)
            Vec3 Y = Vec3::cross(X, Z).normalize();

            startMat.m[0][0] = X.x; startMat.m[0][1] = Y.x; startMat.m[0][2] = Z.x;
            startMat.m[1][0] = X.y; startMat.m[1][1] = Y.y; startMat.m[1][2] = Z.y;
            startMat.m[2][0] = X.z; startMat.m[2][1] = Y.z; startMat.m[2][2] = Z.z;

            // Visualization: Direction Arrow
            ImDrawList* dl = ImGui::GetBackgroundDrawList();
            Vec3 center = pos;
            Vec3 normal = Y.normalize();
            float len = 3.0f;
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
        // Manipülasyon yeni başladı
        if (ImGui::GetIO().KeyShift && sel.hasSelection()) {
            
            // Build a list of objects to duplicate
            // If multi-selection exists, use it. Otherwise use the single active selection.
            std::vector<SelectableItem> itemsToDuplicate;
            if (sel.multi_selection.size() > 0) {
                itemsToDuplicate = sel.multi_selection;
            } else {
                itemsToDuplicate.push_back(sel.selected);
            }

            std::vector<std::shared_ptr<Hittable>> allNewTriangles;
            std::vector<SelectableItem> newSelectionList;
            
            // Temporary map for name uniqueness check
            if (!mesh_cache_valid) rebuildMeshCache(ctx.scene.world.objects);
            
            // Perform duplication for each item
            bool anyDuplicated = false;
            
            for (const auto& item : itemsToDuplicate) {
                if (item.type == SelectableType::Object && item.object) {
                    
                    std::string targetName = item.object->nodeName;
                    if (targetName.empty()) targetName = "Unnamed";

                    // Unique name generation
                    std::string baseName = targetName;
                    size_t lastUnderscore = baseName.rfind('_');
                    if (lastUnderscore != std::string::npos) {
                        std::string suffix = baseName.substr(lastUnderscore + 1);
                        if (!suffix.empty() && std::all_of(suffix.begin(), suffix.end(), ::isdigit)) {
                            baseName = baseName.substr(0, lastUnderscore);
                        }
                    }
                    int copyNum = 1;
                    std::string newName;
                    do { newName = baseName + "_" + std::to_string(copyNum++); } while (mesh_cache.find(newName) != mesh_cache.end());

                    // Create Unique Transform
                    std::shared_ptr<Transform> newTransform = std::make_shared<Transform>();
                    if (item.object->getTransformHandle()) {
                        *newTransform = *item.object->getTransformHandle();
                    }

                    // Duplicate Triangles
                    std::shared_ptr<Triangle> firstNewTri = nullptr;
                    auto it = mesh_cache.find(targetName);
                    if (it != mesh_cache.end()) {
                        for (auto& pair : it->second) {
                            auto& oldTri = pair.second;
                            auto newTri = std::make_shared<Triangle>(*oldTri);
                            newTri->setTransformHandle(newTransform);
                            newTri->setNodeName(newName);
                            
                            allNewTriangles.push_back(newTri);
                            if (!firstNewTri) firstNewTri = newTri;
                        }
                    }
                    
                    if (firstNewTri) {
                        SelectableItem newItem;
                        newItem.type = SelectableType::Object;
                        newItem.object = firstNewTri;
                        newItem.name = newName;
                        newSelectionList.push_back(newItem);
                        anyDuplicated = true;
                    }
                }
                // TODO: Add support for duplicating Lights? (Currently UI usually handles lights separately)
            }

            if (anyDuplicated) {
                // Add all new triangles to world
                ctx.scene.world.objects.insert(ctx.scene.world.objects.end(), allNewTriangles.begin(), allNewTriangles.end());
                
                // Rebuild mesh cache with new objects
                rebuildMeshCache(ctx.scene.world.objects);
                
                // Select the NEW objects
                sel.clearSelection();
                for (const auto& newItem : newSelectionList) {
                    sel.addToSelection(newItem); // This automatically handles primary/multi logic
                }
                
                // Record Undo (Grouped)
                // For simplicity, we might record just the last one or need a GroupDuplicateCommand
                // Currently keeping it simple (no comprehensive undo for multi-duplicate yet or relies on individual commands)
                // Ideally: history.record(std::make_unique<MultiDuplicateCommand>(...)); 
                
                // ═══════════════════════════════════════════════════════════════════
                // DEFERRED FULL REBUILD (Reliable - async in Main.cpp)
                // ═══════════════════════════════════════════════════════════════════
                // Incremental clone had issues with SBT/transform sync.
                // Full rebuild is slower but reliable. Async mechanism prevents UI freeze.
                extern bool g_optix_rebuild_pending;
                g_optix_rebuild_pending = true;
                
                // Defer CPU BVH rebuild (async in Main.cpp)
                extern bool g_bvh_rebuild_pending;
                g_bvh_rebuild_pending = true;
                ctx.renderer.resetCPUAccumulation();
                is_bvh_dirty = false;
                
                SCENE_LOG_INFO("Multi-Duplicate: " + std::to_string(newSelectionList.size()) + " objects copied.");
                ProjectManager::getInstance().markModified();
            }
        }
        else if (ImGui::GetIO().KeyShift && sel.selected.type == SelectableType::Light && sel.selected.light) {
            std::shared_ptr<Light> newLight = nullptr;
            auto l = sel.selected.light;
            if (std::dynamic_pointer_cast<PointLight>(l)) newLight = std::make_shared<PointLight>(*(PointLight*)l.get());
            else if (std::dynamic_pointer_cast<DirectionalLight>(l)) newLight = std::make_shared<DirectionalLight>(*(DirectionalLight*)l.get());
            else if (std::dynamic_pointer_cast<SpotLight>(l)) newLight = std::make_shared<SpotLight>(*(SpotLight*)l.get());
            else if (std::dynamic_pointer_cast<AreaLight>(l)) newLight = std::make_shared<AreaLight>(*(AreaLight*)l.get());

            if (newLight) {
                ctx.scene.lights.push_back(newLight);
                history.record(std::make_unique<AddLightCommand>(newLight));
                sel.selectLight(newLight);
                if (ctx.optix_gpu_ptr) { ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights); ctx.optix_gpu_ptr->resetAccumulation(); }
                SCENE_LOG_INFO("Duplicated Light (Shift+Drag)");
                ProjectManager::getInstance().markModified();
            }
        }
        else if (sel.selected.type == SelectableType::Light && sel.selected.light && !ImGui::GetIO().KeyShift) {
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
    if (!is_using && was_using_gizmo && sel.selected.type == SelectableType::Object && sel.selected.object) {
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

    // END DRAG for Light
    if (!is_using && was_using_gizmo && sel.selected.type == SelectableType::Light && drag_light) {
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
        Vec3 newPos(objectMatrix[12], objectMatrix[13], objectMatrix[14]);
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
                        item.camera->lookfrom = item.camera->lookfrom + deltaPos;
                        item.camera->lookat = item.camera->lookat + deltaPos;
                        item.camera->update_camera_vectors();
                    }
                } // End of multi_selection loop

                // Trigger TLAS Update after processing all objects
                if (ctx.optix_gpu_ptr && ctx.optix_gpu_ptr->isUsingTLAS()) {
                    ctx.optix_gpu_ptr->rebuildTLAS(); // Fast Update
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

                // Scale handling: Assign directly (Fixes infinite growth)
                float sx = right.length();
                float sz = forward.length();

                if (sx > 0.001f) al->width = sx;
                if (sz > 0.001f) al->height = sz;

                // Set vectors directly (they carry rotation and scale)
                al->u = right;
                al->v = forward;

                // Correct Position
                al->position = newPos - (al->u * 0.5f) - (al->v * 0.5f);
            }

            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setLightParams(ctx.scene.lights);
                ctx.optix_gpu_ptr->resetAccumulation();
            }
        }
        else if (sel.selected.type == SelectableType::Camera && sel.selected.camera) {
            Vec3 delta = newPos - sel.selected.camera->lookfrom;
            sel.selected.camera->lookfrom = newPos;
            sel.selected.camera->lookat = sel.selected.camera->lookat + delta;
            sel.selected.camera->update_camera_vectors();
            if (ctx.optix_gpu_ptr) {
                ctx.optix_gpu_ptr->setCameraParams(*sel.selected.camera);
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
                            // Support multi-material instances (one object name -> multiple instances)
                            std::vector<int> inst_ids = ctx.optix_gpu_ptr->getInstancesByNodeName(targetName);
                            
                            if (!inst_ids.empty()) {
                                float t[12];
                                // Convert 4x4 -> 3x4 (row-major)
                                t[0] = newMat.m[0][0]; t[1] = newMat.m[0][1]; t[2] = newMat.m[0][2]; t[3] = newMat.m[0][3];
                                t[4] = newMat.m[1][0]; t[5] = newMat.m[1][1]; t[6] = newMat.m[1][2]; t[7] = newMat.m[1][3];
                                t[8] = newMat.m[2][0]; t[9] = newMat.m[2][1]; t[10] = newMat.m[2][2]; t[11] = newMat.m[2][3];
                                
                                // Update ALL instances belonging to this object
                                for (int inst_id : inst_ids) {
                                    ctx.optix_gpu_ptr->updateInstanceTransform(inst_id, t);
                                }
                                
                                ctx.optix_gpu_ptr->rebuildTLAS(); // Very fast (~0.01ms)
                                ctx.optix_gpu_ptr->resetAccumulation();
                            }
                            // NOTE: TLAS mode - NO CPU vertex update needed! Transform is applied via instance matrix.
                            // This saves 2M function calls per frame for large objects!
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
                                    std::vector<int> inst_ids = ctx.optix_gpu_ptr->getInstancesByNodeName(targetName);
                                    if (!inst_ids.empty()) {
                                        float t[12];
                                        Matrix4x4 finalMat = th->base; // Get the newly calculated base
                                        
                                        t[0] = finalMat.m[0][0]; t[1] = finalMat.m[0][1]; t[2] = finalMat.m[0][2]; t[3] = finalMat.m[0][3];
                                        t[4] = finalMat.m[1][0]; t[5] = finalMat.m[1][1]; t[6] = finalMat.m[1][2]; t[7] = finalMat.m[1][3];
                                        t[8] = finalMat.m[2][0]; t[9] = finalMat.m[2][1]; t[10] = finalMat.m[2][2]; t[11] = finalMat.m[2][3];
                                        
                                        for (int inst_id : inst_ids) {
                                            ctx.optix_gpu_ptr->updateInstanceTransform(inst_id, t);
                                        }
                                        // Auto-rebuild TLAS periodically or on release
                                    }
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
            if (!using_gpu_tlas) {
                extern bool g_bvh_rebuild_pending;
                g_bvh_rebuild_pending = true;
            }
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
                extern bool g_bvh_rebuild_pending;
                g_bvh_rebuild_pending = true;
            }
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
void SceneUI::drawSelectionGizmos(UIContext& ctx)
{
    if (ctx.selection.hasSelection() && ctx.selection.show_gizmo && ctx.scene.camera && viewport_settings.show_gizmos) {
        drawSelectionBoundingBox(ctx);
        drawTransformGizmo(ctx);
    }
}
