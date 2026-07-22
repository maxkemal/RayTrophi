/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          scene_ui_river.hpp
* Author:        Kemal Demirtas
* Date:          June 2024
* License:       [License Information - e.g. Proprietary / MIT / etc.]
* =========================================================================
*/
// ═══════════════════════════════════════════════════════════════════════════════
// SCENE UI - RIVER PANEL
// ═══════════════════════════════════════════════════════════════════════════════
// UI for creating and editing river splines
// ═══════════════════════════════════════════════════════════════════════════════
#ifndef SCENE_UI_RIVER_HPP
#define SCENE_UI_RIVER_HPP

#include "scene_ui.h"
#include "imgui.h"
#include "RiverSpline.h"
#include "TerrainManager.h"
#include "ProjectManager.h"
#include "WaterSystem.h"

namespace {
bool BeginRiverSection(const char* title, const ImVec4& accent, bool defaultOpen = true) {
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

void EndRiverSection() {}
}

// ═══════════════════════════════════════════════════════════════════════════════
// RIVER PANEL IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════
inline void SceneUI::drawRiverPanel(UIContext& ctx) {
    auto& riverMgr = RiverManager::getInstance();
    riverMgr.lastActiveFrame = ImGui::GetFrameCount();

    UIWidgets::PushControlSurfaceStyle(ImVec4(0.62f, 0.84f, 1.0f, 1.0f));

    ImGui::TextColored(ImVec4(0.62f, 0.84f, 1.0f, 1.0f), "River Spline");
    ImGui::TextDisabled("Bezier spline river editing.");
    ImGui::Separator();

    if (BeginRiverSection("Manage Rivers", ImVec4(0.62f, 0.84f, 1.0f, 1.0f))) {
        if (ImGui::Checkbox("Always Show Gizmos", &riverMgr.showGizmosWhenInactive)) {
            ProjectManager::getInstance().markModified();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Show river splines in viewport even when River panel is not focused");
        }

        if (UIWidgets::PrimaryButton("+ New River", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
            RiverSpline* newRiver = riverMgr.createRiver("River");
            riverMgr.editingRiverId = newRiver->id;
            riverMgr.isEditing = true;
            riverMgr.selectedControlPoint = -1;
            ProjectManager::getInstance().markModified();
        }

        ImGui::Separator();

        auto& rivers = riverMgr.getRivers();
        if (rivers.empty()) {
            ImGui::TextDisabled("No rivers in the scene.");
        }

        for (auto& river : rivers) {
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_SpanAvailWidth;
            if (riverMgr.editingRiverId == river.id) {
                flags |= ImGuiTreeNodeFlags_Selected;
            }

            bool open = ImGui::TreeNodeEx(
                (void*)(intptr_t)river.id,
                flags,
                "%s (%d pts)", river.name.c_str(), (int)river.controlPointCount());

            if (ImGui::IsItemClicked()) {
                riverMgr.editingRiverId = river.id;
                riverMgr.isEditing = true;
                riverMgr.selectedControlPoint = -1;
                if (river.waterSurfaceId >= 0) selected_water_surface_id = river.waterSurfaceId;
                selectManagedMesh(ctx, river.flatMesh);
            }

            if (ImGui::BeginPopupContextItem()) {
                if (ImGui::MenuItem("Delete")) {
                    const int deleteId = river.id;
                    if (riverMgr.editingRiverId == deleteId) {
                        riverMgr.editingRiverId = -1;
                        riverMgr.isEditing = false;
                        riverMgr.selectedControlPoint = -1;
                    }

                    riverMgr.removeRiver(ctx.scene, deleteId);
                    extern bool g_bvh_rebuild_pending;
                    extern bool g_optix_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    ProjectManager::getInstance().markModified();
                    ImGui::EndPopup();
                    if (open) ImGui::TreePop();
                    break;
                }
                ImGui::EndPopup();
            }

            if (open) ImGui::TreePop();
        }
        EndRiverSection();
    }

    RiverSpline* selectedRiver = riverMgr.getRiver(riverMgr.editingRiverId);
    if (selectedRiver) {
        if (BeginRiverSection("River Properties", ImVec4(0.58f, 0.78f, 1.0f, 1.0f))) {
            char nameBuf[128];
            strncpy(nameBuf, selectedRiver->name.c_str(), sizeof(nameBuf) - 1);
            nameBuf[sizeof(nameBuf) - 1] = 0;
            if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf))) {
                selectedRiver->name = nameBuf;
                ProjectManager::getInstance().markModified();
            }

            ImGui::Separator();
            ImGui::TextDisabled("Mesh Generation");

            bool rebuildNeeded = false;
            rebuildNeeded |= ImGui::SliderInt("Length Subdivs", &selectedRiver->lengthSubdivisions, 4, 128);
            rebuildNeeded |= ImGui::SliderInt("Width Segments", &selectedRiver->widthSegments, 1, 16);
            rebuildNeeded |= SceneUI::DrawSmartFloat("rbnk", "Bank Height", &selectedRiver->bankHeight, -0.5f, 1.0f, "%.3f", false, nullptr, 16);
            rebuildNeeded |= ImGui::Checkbox("Follow Terrain", &selectedRiver->followTerrain);

            ImGui::Separator();
            ImGui::TextDisabled("Default Values");
            bool authoringChanged = false;
            authoringChanged |= SceneUI::DrawSmartFloat("rdw", "Default Width", &riverMgr.defaultWidth, 0.5f, 20.0f, "%.1f", false, nullptr, 16);
            authoringChanged |= SceneUI::DrawSmartFloat("rdd", "Default Depth", &riverMgr.defaultDepth, 0.1f, 5.0f, "%.1f", false, nullptr, 16);
            if (authoringChanged) ProjectManager::getInstance().markModified();

            if (rebuildNeeded) {
                selectedRiver->needsRebuild = true;
                ProjectManager::getInstance().markModified();
            }
            EndRiverSection();
        }

        if (BeginRiverSection("Flow Physics & Dynamics", ImVec4(0.40f, 0.90f, 1.0f, 1.0f), false)) {
            RiverSpline::PhysicsParams& pp = selectedRiver->physics;
            bool changed = false;

            changed |= ImGui::Checkbox("Enable Rapids/Turbulence", &pp.enableTurbulence);
            if (pp.enableTurbulence) {
                ImGui::Indent();
                changed |= SceneUI::DrawSmartFloat("rts", "Turbulence Strength", &pp.turbulenceStrength, 0.0f, 5.0f, "%.2f", false, nullptr, 16);
                changed |= SceneUI::DrawSmartFloat("rrt", "Rapids Threshold", &pp.turbulenceThreshold, 0.01f, 0.2f, "%.2f", false, nullptr, 16);
                changed |= SceneUI::DrawSmartFloat("rns", "Noise Scale", &pp.noiseScale, 0.1f, 5.0f, "%.2f", false, nullptr, 16);
                ImGui::Unindent();
            }

            changed |= ImGui::Checkbox("Enable Banking (Curves)", &pp.enableBanking);
            if (pp.enableBanking) {
                ImGui::Indent();
                changed |= SceneUI::DrawSmartFloat("rbs", "Banking Strength", &pp.bankingStrength, 0.0f, 3.0f, "%.2f", false, nullptr, 16);
                ImGui::Unindent();
            }

            changed |= ImGui::Checkbox("Enable Flow Bulge", &pp.enableFlowBulge);
            if (pp.enableFlowBulge) {
                ImGui::Indent();
                changed |= SceneUI::DrawSmartFloat("rfb", "Flow Bulge", &pp.flowBulgeStrength, 0.0f, 2.0f, "%.2f", false, nullptr, 16);
                ImGui::Unindent();
            }

            if (changed) {
                selectedRiver->needsRebuild = true;
                if (selectedRiver->flatMesh) {
                    riverMgr.generateMesh(selectedRiver, ctx.scene);
                    extern bool g_bvh_rebuild_pending;
                    extern bool g_optix_rebuild_pending;
                    g_bvh_rebuild_pending = true;
                    g_optix_rebuild_pending = true;
                    ctx.renderer.resetCPUAccumulation();
                }
                ProjectManager::getInstance().markModified();
            }
            EndRiverSection();
        }

        if (BeginRiverSection("Control Points", ImVec4(1.0f, 0.60f, 0.60f, 1.0f))) {
            ImVec4 editColor = riverMgr.isEditing ? ImVec4(0.2f, 0.8f, 0.2f, 1.0f) : ImVec4(0.8f, 0.2f, 0.2f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Button, editColor);
            if (ImGui::Button(riverMgr.isEditing ? "Stop Editing" : "Start Editing", ImVec2(-1, 0))) {
                riverMgr.isEditing = !riverMgr.isEditing;
            }
            ImGui::PopStyleColor();

            if (riverMgr.isEditing) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f), "Click terrain to add points");
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f), "Right-click a point to delete");
            }

            ImGui::Separator();

            for (int i = 0; i < (int)selectedRiver->controlPointCount(); ++i) {
                BezierControlPoint* pt = selectedRiver->getControlPoint(i);
                if (!pt) continue;

                ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf;
                if (riverMgr.selectedControlPoint == i) {
                    flags |= ImGuiTreeNodeFlags_Selected;
                }

                char label[64];
                snprintf(label, sizeof(label), "Point %d (W: %.1f)", i, pt->userData1);
                bool open = ImGui::TreeNodeEx((void*)(intptr_t)(i + 1000), flags, "%s", label);

                if (ImGui::IsItemClicked()) {
                    riverMgr.selectedControlPoint = i;
                }

                if (open) ImGui::TreePop();
            }
            EndRiverSection();
        }

        if (riverMgr.selectedControlPoint >= 0 &&
            riverMgr.selectedControlPoint < (int)selectedRiver->controlPointCount()) {
            if (BeginRiverSection("Selected Point", ImVec4(1.0f, 0.74f, 0.42f, 1.0f), false)) {
                BezierControlPoint* pt = selectedRiver->getControlPoint(riverMgr.selectedControlPoint);
                if (pt) {
                    bool changed = false;

                    float pos[3] = { pt->position.x, pt->position.y, pt->position.z };
                    if (ImGui::DragFloat3("Position", pos, 0.1f)) {
                        pt->position = Vec3(pos[0], pos[1], pos[2]);
                        changed = true;
                    }

                    changed |= SceneUI::DrawSmartFloat("cpw", "Width", &pt->userData1, 0.1f, 50.0f, "%.1f", false, nullptr, 16);
                    changed |= SceneUI::DrawSmartFloat("cpd", "Depth", &pt->userData2, 0.0f, 10.0f, "%.2f", false, nullptr, 16);

                    if (ImGui::Checkbox("Auto Tangent", &pt->autoTangent)) {
                        if (pt->autoTangent) {
                            selectedRiver->spline.calculateAutoTangents();
                        }
                        changed = true;
                    }

                    if (!pt->autoTangent) {
                        float tin[3] = { pt->tangentIn.x, pt->tangentIn.y, pt->tangentIn.z };
                        float tout[3] = { pt->tangentOut.x, pt->tangentOut.y, pt->tangentOut.z };
                        if (ImGui::DragFloat3("Tangent In", tin, 0.1f)) {
                            pt->tangentIn = Vec3(tin[0], tin[1], tin[2]);
                            changed = true;
                        }
                        if (ImGui::DragFloat3("Tangent Out", tout, 0.1f)) {
                            pt->tangentOut = Vec3(tout[0], tout[1], tout[2]);
                            changed = true;
                        }
                    }

                    if (ImGui::Button("Delete Point [DEL]")) {
                        selectedRiver->removeControlPoint(riverMgr.selectedControlPoint);
                        riverMgr.selectedControlPoint = -1;
                        changed = true;
                    }
                    ImGui::SameLine();
                    ImGui::TextDisabled("(Press DEL key)");

                    if (ImGui::IsKeyPressed(ImGuiKey_Delete) && !ImGui::GetIO().WantTextInput) {
                        selectedRiver->removeControlPoint(riverMgr.selectedControlPoint);
                        if (selectedRiver->controlPointCount() > 0) {
                            riverMgr.selectedControlPoint = (std::min)(riverMgr.selectedControlPoint, (int)selectedRiver->controlPointCount() - 1);
                        } else {
                            riverMgr.selectedControlPoint = -1;
                        }
                        changed = true;
                    }

                    if (changed) {
                        selectedRiver->needsRebuild = true;
                        if (selectedRiver->flatMesh) {
                            riverMgr.generateMesh(selectedRiver, ctx.scene);
                            extern bool g_bvh_rebuild_pending;
                            extern bool g_optix_rebuild_pending;
                            g_bvh_rebuild_pending = true;
                            g_optix_rebuild_pending = true;
                            ctx.renderer.resetCPUAccumulation();
                        }
                        ProjectManager::getInstance().markModified();
                    }
                }
                EndRiverSection();
            }
        }

        if (BeginRiverSection("Terrain Interaction", ImVec4(0.56f, 0.90f, 0.68f, 1.0f), false)) {
            if (ImGui::Checkbox("Auto-Carve on Move", &riverMgr.autoCarveOnMove)) {
                ProjectManager::getInstance().markModified();
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Automatically carves terrain when points are moved.\nCarve settings live in the Terrain panel.");
            }
            EndRiverSection();
        }

        if (BeginRiverSection("Water Material", ImVec4(0.50f, 0.78f, 1.0f, 1.0f), false)) {
            if (selectedRiver->waterSurfaceId >= 0) {
                selected_water_surface_id = selectedRiver->waterSurfaceId;
                ImGui::BulletText("Registered as WaterSurface (ID: %d)", selectedRiver->waterSurfaceId);
                WaterSurface* surf = WaterManager::getInstance().getWaterSurface(selectedRiver->waterSurfaceId);
                if (surf) {
                    ImGui::TextDisabled("Shared editor for the selected linked water surface.");
                    drawWaterSurfaceMaterialEditor(ctx, *surf, false);
                    // Keep the spline-side mirror complete for code paths that
                    // operate before the next mesh rebuild/project save.
                    selectedRiver->waterParams = surf->params;
                } else {
                    ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "Linked WaterSurface could not be found");
                }
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.3f, 1.0f), "Build mesh to create water surface");
            }
            EndRiverSection();
        }

        if (BeginRiverSection("Actions", ImVec4(0.92f, 0.66f, 0.40f, 1.0f), false)) {
            if (selectedRiver->needsRebuild) {
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Mesh needs rebuild");
            }

            if (UIWidgets::PrimaryButton("Rebuild Mesh", ImVec2(UIWidgets::GetInspectorActionWidth(), 0))) {
                riverMgr.generateMesh(selectedRiver, ctx.scene);
                extern bool g_bvh_rebuild_pending;
                extern bool g_optix_rebuild_pending;
                g_bvh_rebuild_pending = true;
                g_optix_rebuild_pending = true;
                ctx.renderer.resetCPUAccumulation();
                ProjectManager::getInstance().markModified();
            }
            EndRiverSection();
        }
    } else {
        if (BeginRiverSection("Selection", ImVec4(0.48f, 0.72f, 0.96f, 1.0f), false)) {
            ImGui::TextDisabled("No river selected.");
            ImGui::TextDisabled("Create a river or select one from the list.");
            EndRiverSection();
        }
    }

    UIWidgets::PopControlSurfaceStyle();
}

// ═══════════════════════════════════════════════════════════════════════════════
// RIVER SPLINE VISUALIZATION (Call from draw loop)
// ═══════════════════════════════════════════════════════════════════════════════
inline void SceneUI::drawRiverGizmos(UIContext& ctx, bool& gizmo_hit) {
    auto& riverMgr = RiverManager::getInstance();
    
    // Check if panel is active (drawn this frame or last frame)
    bool isPanelActive = (riverMgr.lastActiveFrame >= ImGui::GetFrameCount() - 1);
    
    // Temporary safety: only draw while River panel is actively visible.
    if (!isPanelActive) {
        return;
    }
    
    if (!ctx.scene.camera) return;
    
    ImDrawList* draw_list = ImGui::GetBackgroundDrawList();
    ImGuiIO& io = ImGui::GetIO();
    Camera& cam = *ctx.scene.camera;
    
    Vec3 cam_forward = (cam.lookat - cam.lookfrom).normalize();
    Vec3 cam_right = cam_forward.cross(cam.vup).normalize();
    Vec3 cam_up = cam_right.cross(cam_forward).normalize();
    
    float fov_rad = cam.vfov * 3.14159265359f / 180.0f;
    float tan_half_fov = tanf(fov_rad * 0.5f);
    float aspect = io.DisplaySize.x / io.DisplaySize.y;
    
        // Calculate Full Window Sizes (since rendering spans the entire screen)
        float v_width = (std::max)(1.0f, io.DisplaySize.x);
        float v_height = (std::max)(1.0f, io.DisplaySize.y);

        auto Project = [&](const Vec3& p) -> ImVec2 {
            Vec3 dir = (p - cam.lookfrom);
            float dist = dir.length();
            dir = dir.normalize();

            Vec3 forward = (cam.lookat - cam.lookfrom).normalize();
            float depth = Vec3::dot(p - cam.lookfrom, forward);

            if (depth <= 0.01f) return ImVec2(-1, -1);

            float local_y = Vec3::dot(p - cam.lookfrom, cam.v);
            float local_x = Vec3::dot(p - cam.lookfrom, cam.u);

            float half_h = depth * tan_half_fov;
            float half_w = half_h * aspect;

            return ImVec2(
                ((local_x / half_w) * 0.5f + 0.5f) * v_width,
                (0.5f - (local_y / half_h) * 0.5f) * v_height
            );
        };
    
    auto IsOnScreen = [](const ImVec2& v) { return v.x > -5000; };
    
    // Draw all rivers
    for (auto& river : riverMgr.getRivers()) {
        bool isSelected = (riverMgr.editingRiverId == river.id);
        
        if (!river.showSpline) continue;
        if (river.spline.pointCount() < 2) {
            // Just draw single point if only one exists
            if (river.spline.pointCount() == 1) {
                ImVec2 pt = Project(river.spline.points[0].position);
                if (IsOnScreen(pt)) {
                    draw_list->AddCircleFilled(pt, 8.0f, IM_COL32(100, 150, 255, 200));
                    draw_list->AddCircle(pt, 8.0f, IM_COL32(255, 255, 255, 255), 0, 2.0f);
                }
            }
            continue;
        }
        
        // Draw spline curve
        ImU32 splineColor = isSelected ? IM_COL32(100, 200, 255, 200) : IM_COL32(50, 100, 200, 150);
        
        Vec3 prevPos = river.spline.samplePosition(0);
        for (int i = 1; i <= 50; ++i) {
            float t = (float)i / 50.0f;
            Vec3 pos = river.spline.samplePosition(t);
            
            ImVec2 p1 = Project(prevPos);
            ImVec2 p2 = Project(pos);
            
            if (IsOnScreen(p1) && IsOnScreen(p2)) {
                draw_list->AddLine(p1, p2, splineColor, 2.0f);
            }
            
            prevPos = pos;
        }
        
        // Draw width visualization (dashed lines on sides)
        if (isSelected) {
            for (int i = 0; i <= 20; ++i) {
                float t = (float)i / 20.0f;
                Vec3 center = river.spline.samplePosition(t);
                Vec3 right = river.spline.sampleRight(t);
                float width = river.spline.sampleUserData1(t);
                
                Vec3 left3d = center - right * (width * 0.5f);
                Vec3 right3d = center + right * (width * 0.5f);
                
                ImVec2 leftPt = Project(left3d);
                ImVec2 rightPt = Project(right3d);
                
                if (i % 2 == 0) { // Dashed effect
                    if (IsOnScreen(leftPt)) {
                        draw_list->AddCircleFilled(leftPt, 2.0f, IM_COL32(100, 200, 255, 100));
                    }
                    if (IsOnScreen(rightPt)) {
                        draw_list->AddCircleFilled(rightPt, 2.0f, IM_COL32(100, 200, 255, 100));
                    }
                }
            }
        }
        
        // Draw control points
        if (river.showControlPoints || isSelected) {
            for (int i = 0; i < (int)river.spline.pointCount(); ++i) {
                auto& pt = river.spline.points[i];
                ImVec2 screenPt = Project(pt.position);
                
                if (!IsOnScreen(screenPt)) continue;
                
                bool isPointSelected = (isSelected && riverMgr.selectedControlPoint == i);
                
                // Point appearance
                float radius = isPointSelected ? 10.0f : 7.0f;
                ImU32 fillColor = isPointSelected ? IM_COL32(255, 200, 50, 255) : IM_COL32(100, 150, 255, 200);
                ImU32 outlineColor = IM_COL32(255, 255, 255, 255);
                
                draw_list->AddCircleFilled(screenPt, radius, fillColor);
                draw_list->AddCircle(screenPt, radius, outlineColor, 0, 2.0f);
                
                // Point index label
                char label[16];
                snprintf(label, sizeof(label), "%d", i);
                draw_list->AddText(ImVec2(screenPt.x + 12, screenPt.y - 6), IM_COL32(255, 255, 255, 200), label);
                
                // Draw tangent handles for selected point
                if (isPointSelected && !pt.autoTangent) {
                    Vec3 tangentInWorld = pt.position + pt.tangentIn;
                    Vec3 tangentOutWorld = pt.position + pt.tangentOut;
                    
                    ImVec2 tin = Project(tangentInWorld);
                    ImVec2 tout = Project(tangentOutWorld);
                    
                    if (IsOnScreen(tin)) {
                        draw_list->AddLine(screenPt, tin, IM_COL32(255, 100, 100, 200), 1.5f);
                        draw_list->AddCircleFilled(tin, 5.0f, IM_COL32(255, 100, 100, 255));
                    }
                    if (IsOnScreen(tout)) {
                        draw_list->AddLine(screenPt, tout, IM_COL32(100, 255, 100, 200), 1.5f);
                        draw_list->AddCircleFilled(tout, 5.0f, IM_COL32(100, 255, 100, 255));
                    }
                }
                
                // Click detection for point selection
                if (isSelected && !ImGuizmo::IsOver()) {
                    float dx = io.MousePos.x - screenPt.x;
                    float dy = io.MousePos.y - screenPt.y;
                    float dist = sqrtf(dx * dx + dy * dy);
                    
                    if (dist < 15.0f) {
                        // Left click - select
                        if (ImGui::IsMouseClicked(0)) {
                            riverMgr.selectedControlPoint = i;
                            gizmo_hit = true;
                        }
                        // Right click - delete
                        else if (ImGui::IsMouseClicked(1)) {
                            river.removeControlPoint(i);
                            river.needsRebuild = true;
                            riverMgr.selectedControlPoint = -1;
                            ProjectManager::getInstance().markModified();
                            gizmo_hit = true;
                            break;
                        }
                    }
                }
                
                // ─────────────────────────────────────────────────────────────
                // DRAG TO MOVE selected control point
                // ─────────────────────────────────────────────────────────────
                if (isPointSelected && ImGui::IsMouseDragging(0) && !ImGui::GetIO().WantCaptureMouse) {
                    // Project mouse delta to world movement
                    Vec3 forward = cam_forward;
                    Vec3 right = cam_right;
                    Vec3 up = cam_up;
                    
                    // Calculate depth of point from camera
                    Vec3 toPoint = pt.position - cam.lookfrom;
                    float depth = toPoint.dot(forward);
                    
                    if (depth > 0.1f) {
                        // Convert pixel delta to world delta
                        float half_h = depth * tan_half_fov;
                        float pixelsPerUnit = io.DisplaySize.y / (2.0f * half_h);
                        
                        ImVec2 delta = io.MouseDelta;
                        float worldDeltaX = delta.x / pixelsPerUnit;
                        float worldDeltaY = -delta.y / pixelsPerUnit;
                        
                        // Move in camera plane (right + up)
                        Vec3 movement = right * worldDeltaX + up * worldDeltaY;
                        pt.position = pt.position + movement;
                        
                        // Optionally snap Y to terrain
                        if (river.followTerrain && TerrainManager::getInstance().hasActiveTerrain()) {
                            pt.position.y = TerrainManager::getInstance().sampleHeight(pt.position.x, pt.position.z);
                        }
                        
                        // Recalculate tangents if auto
                        if (pt.autoTangent) {
                            river.spline.calculateAutoTangents();
                        }
                        
                        river.needsRebuild = true;
                        riverMgr.isDraggingPoint = true;  // Track drag state
                        gizmo_hit = true;
                    }
                }
                
                // Rebuild mesh when drag ends
                if (isPointSelected && riverMgr.isDraggingPoint && ImGui::IsMouseReleased(0)) {
                    riverMgr.isDraggingPoint = false;
                    
                    // AUTO-CARVE ON MOVE
                    if (riverMgr.autoCarveOnMove && TerrainManager::getInstance().hasActiveTerrain()) {
                        // Backup before auto-carve if not exists
                        auto& tm = TerrainManager::getInstance();
                        if (!riverMgr.hasTerrainBackup && !tm.getTerrains().empty()) {
                            auto& t = tm.getTerrains()[0];
                            riverMgr.terrainBackupData = t.heightmap.data;
                            riverMgr.terrainBackupWidth = t.heightmap.width;
                            riverMgr.terrainBackupHeight = t.heightmap.height;
                            riverMgr.hasTerrainBackup = true;
                        }

                        // Sample many points along spline
                        std::vector<Vec3> samplePoints;
                        std::vector<float> sampleWidths;
                        std::vector<float> sampleDepths;
                        
                        int numSamples = river.lengthSubdivisions * 3;
                        for (int k = 0; k <= numSamples; ++k) {
                            float t = (float)k / (float)numSamples;
                            samplePoints.push_back(river.spline.samplePosition(t));
                            sampleWidths.push_back(river.spline.sampleUserData1(t));
                            sampleDepths.push_back(river.spline.sampleUserData2(t) * riverMgr.carveDepthMult);
                        }
                        
                        // Carve
                        TerrainManager::getInstance().carveRiverBed(
                            -1,
                            samplePoints,
                            sampleWidths,
                            sampleDepths,
                            riverMgr.carveSmoothness,
                            ctx.scene
                        );
                        
                        // Post-Erosion
                        if (riverMgr.carveAutoPostErosion) {
                            auto& terrains = TerrainManager::getInstance().getTerrains();
                            if (!terrains.empty()) {
                                ThermalErosionParams ep;
                                ep.iterations = riverMgr.carveErosionIterations;
                                ep.talusAngle = 0.3f;
                                ep.erosionAmount = 0.4f;
                                TerrainManager::getInstance().thermalErosion(&terrains[0], ep);
                            }
                        }
                    }
                    
                    // Rebuild the mesh after dragging

                    if (river.needsRebuild && river.flatMesh) {
                        riverMgr.generateMesh(&river, ctx.scene);
                        
                        extern bool g_bvh_rebuild_pending;
                        extern bool g_optix_rebuild_pending;
                        g_bvh_rebuild_pending = true;
                        g_optix_rebuild_pending = true;
                        ctx.renderer.resetCPUAccumulation();
                    }
                    
                    ProjectManager::getInstance().markModified();
                }
            }
        }
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // ADD NEW POINT ON TERRAIN CLICK (when editing)
    // ─────────────────────────────────────────────────────────────────────────
    RiverSpline* editingRiver = riverMgr.getRiver(riverMgr.editingRiverId);
    
    if (editingRiver && riverMgr.isEditing && !gizmo_hit && !ImGuizmo::IsOver()) {
        if (ImGui::IsMouseClicked(0) && !ImGui::GetIO().WantCaptureMouse) {
            // Raycast to terrain
            float mx = io.MousePos.x;
            float my = io.MousePos.y;

            // Full window size
            float v_width = (std::max)(1.0f, io.DisplaySize.x);
            float v_height = (std::max)(1.0f, io.DisplaySize.y);

            // Normalize mouse position relative to full window viewport
            float u = mx / v_width;
            float v = 1.0f - (my / v_height);

            // Use Camera's own ray generation for perfect consistency
            Ray cameraRay = cam.get_ray(u, v);
            Vec3 rayDir = cameraRay.direction;
            Vec3 rayOrigin = cameraRay.origin;
            
            // Perform accurate terrain raycast
            if (TerrainManager::getInstance().hasActiveTerrain()) {
                float closest_t = 1e20f;
                Vec3 hitPoint;
                bool found_terrain = false;
                
                auto& terrains = TerrainManager::getInstance().getTerrains();
                for (auto& terrain : terrains) {
                    float t_out;
                    Vec3 n_out;
                    if (TerrainManager::getInstance().intersectRay(&terrain, cameraRay, t_out, n_out)) {
                        if (t_out < closest_t) {
                            closest_t = t_out;
                            hitPoint = cameraRay.origin + cameraRay.direction * t_out;
                            found_terrain = true;
                        }
                    }
                }
                
                if (found_terrain) {
                    // Add control point
                    editingRiver->addControlPoint(hitPoint, riverMgr.defaultWidth, riverMgr.defaultDepth);
                    riverMgr.selectedControlPoint = (int)editingRiver->controlPointCount() - 1;
                    ProjectManager::getInstance().markModified();
                    gizmo_hit = true;
                }
            } else {
                // No terrain - intersect with Y=0 plane
                if (fabsf(rayDir.y) > 0.01f) {
                    float t = -rayOrigin.y / rayDir.y;
                    if (t > 0) {
                        Vec3 hitPoint = rayOrigin + rayDir * t;
                        editingRiver->addControlPoint(hitPoint, riverMgr.defaultWidth, riverMgr.defaultDepth);
                        riverMgr.selectedControlPoint = (int)editingRiver->controlPointCount() - 1;
                        ProjectManager::getInstance().markModified();
                        gizmo_hit = true;
                    }
                }
            }
        }
    }
}

#endif // SCENE_UI_RIVER_HPP

