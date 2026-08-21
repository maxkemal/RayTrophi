#include "MeshEdit/ProfileSplineOverlay.h"

#include "MeshEdit/SplineObject.h"
#include "MeshEdit/SplineEditService.h"
#include "SceneSelection.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "globals.h"
#include "Camera.h"
#include "imgui.h"
#include "ImGuizmo.h"

#include <string>
#include "ProjectManager.h"

#include <algorithm>
#include <cmath>

namespace MeshEdit {
namespace {
bool project(const Camera& camera, const Vec3& point, float width, float height, ImVec2& out) {
    const Vec3 forward = (camera.lookat - camera.lookfrom).normalize();
    const Vec3 right = forward.cross(camera.vup).normalize();
    const Vec3 up = right.cross(forward).normalize();
    const Vec3 delta = point - camera.lookfrom;
    const float depth = delta.dot(forward);
    const bool ortho = camera.orthographic;
    if (!ortho && depth <= 0.01f) return false;
    const float halfHeight = ortho
        ? std::max(0.001f, camera.ortho_height * 0.5f)
        : depth * std::tan(camera.vfov * 3.14159265359f / 360.0f);
    const float halfWidth = halfHeight * std::max(0.01f, width / std::max(1.0f, height));
    if (halfWidth <= 1e-5f) return false;
    out.x = ((delta.dot(right) / halfWidth) * 0.5f + 0.5f) * width;
    out.y = (0.5f - (delta.dot(up) / halfHeight) * 0.5f) * height;
    return true;
}
}

void drawProfileSplineOverlay(UIContext& ctx) {
    if (!ctx.scene.camera) return;
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* draw = ImGui::GetBackgroundDrawList();
    const bool selectedIsSpline = static_cast<bool>(ctx.selection.selected.spline_object);

    for (size_t objectIndex = 0; objectIndex < ctx.scene.world.objects.size(); ++objectIndex) {
        const auto& object = ctx.scene.world.objects[objectIndex];
        auto splineObject = std::dynamic_pointer_cast<SplineObject>(object);
        if (!splineObject || !splineObject->visible || splineObject->spline.points.empty()) continue;
        // The overlay also owns the viewport control-point edit path, so this
        // reference must remain mutable when a point is dragged.
        auto& spline = splineObject->spline;
        const bool selected = selectedIsSpline && ctx.selection.selected.spline_object == splineObject;
        const Matrix4x4 transform = splineObject->transform
            ? splineObject->transform->getFinal() : Matrix4x4::identity();

        auto projectLocal = [&](const Vec3& local, ImVec2& screen) {
            return project(*ctx.scene.camera, transform.transform_point(local),
                           io.DisplaySize.x, io.DisplaySize.y, screen);
        };

        // Rendered shading already spends the frame budget on the path tracer.
        // The source overlay is authoring feedback, not render geometry: use a
        // lighter screen approximation there while keeping the solid/edit view
        // crisp for hit testing.
        const int kSamples = g_solid_viewport_active ? 64 : 24;
        std::vector<ImVec2> curve;
        curve.reserve(kSamples + 1);
        for (int i = 0; i <= kSamples; ++i) {
            ImVec2 screen;
            if (projectLocal(spline.samplePosition(static_cast<float>(i) / kSamples), screen)) {
                curve.push_back(screen);
            }
        }
        const ImU32 curveColor = selected ? IM_COL32(80, 245, 215, 255)
                                          : IM_COL32(60, 145, 190, 190);
        for (size_t i = 1; i < curve.size(); ++i) {
            draw->AddLine(curve[i - 1], curve[i], curveColor, selected ? 2.5f : 1.8f);
        }
        if (spline.isClosed && curve.size() > 2) {
            draw->AddLine(curve.back(), curve.front(), curveColor, selected ? 2.5f : 1.8f);
        }

        std::vector<ImVec2> controls;
        controls.reserve(spline.points.size());
        for (const auto& control : spline.points) {
            ImVec2 screen;
            if (projectLocal(control.position, screen)) controls.push_back(screen);
        }
        const bool pointEditing = selected && splineObject->edit_mode;
        splineObject->has_insert_preview = false;
        if (selected || splineObject->edit_controls) {
            for (size_t i = 0; i < controls.size(); ++i) {
                const bool pointSelected = selected &&
                    std::find(splineObject->selected_points.begin(), splineObject->selected_points.end(),
                              static_cast<int>(i)) != splineObject->selected_points.end();
                const float radius = pointSelected ? 9.0f : 6.0f;
                draw->AddCircleFilled(controls[i], radius,
                    pointSelected ? IM_COL32(255, 205, 70, 255) : IM_COL32(70, 170, 220, 220));
                draw->AddCircle(controls[i], radius, IM_COL32(245, 245, 245, 230), 12, 1.5f);
                if (pointEditing) {
                    const std::string label = std::to_string(i);
                    draw->AddText(ImVec2(controls[i].x + 9.0f, controls[i].y - 7.0f),
                                  IM_COL32(235, 235, 235, 220), label.c_str());
                }
            }
        }

        // Blender-style right-drag box selection. The previous picker only
        // handled a point under the cursor, so dragging over an empty part of
        // the viewport never selected the enclosed controls. This path is
        // screen-space by design: spline objects are authoring sources and do
        // not participate in the renderable mesh ray-hit path.
        if (selected && pointEditing && !ImGui::GetIO().WantTextInput && !ImGuizmo::IsOver()) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                splineObject->selection_box_active = true;
                splineObject->selection_box_start = Vec2(io.MousePos.x, io.MousePos.y);
                splineObject->selection_box_current = splineObject->selection_box_start;
            }
            if (splineObject->selection_box_active && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
                splineObject->selection_box_current = Vec2(io.MousePos.x, io.MousePos.y);
                const ImVec2 a(splineObject->selection_box_start.x,
                               splineObject->selection_box_start.y);
                const ImVec2 b(splineObject->selection_box_current.x,
                               splineObject->selection_box_current.y);
                draw->AddRect(ImVec2(std::min(a.x, b.x), std::min(a.y, b.y)),
                              ImVec2(std::max(a.x, b.x), std::max(a.y, b.y)),
                              IM_COL32(255, 190, 70, 230), 0.0f, 0, 1.5f);
                draw->AddRectFilled(ImVec2(std::min(a.x, b.x), std::min(a.y, b.y)),
                                    ImVec2(std::max(a.x, b.x), std::max(a.y, b.y)),
                                    IM_COL32(255, 190, 70, 28));
            }
            if (splineObject->selection_box_active &&
                ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                splineObject->selection_box_current = Vec2(io.MousePos.x, io.MousePos.y);
                const float minX = std::min(splineObject->selection_box_start.x,
                                             splineObject->selection_box_current.x);
                const float maxX = std::max(splineObject->selection_box_start.x,
                                             splineObject->selection_box_current.x);
                const float minY = std::min(splineObject->selection_box_start.y,
                                             splineObject->selection_box_current.y);
                const float maxY = std::max(splineObject->selection_box_start.y,
                                             splineObject->selection_box_current.y);
                if (!io.KeyCtrl) splineObject->selected_points.clear();
                if ((maxX - minX) > 4.0f || (maxY - minY) > 4.0f) {
                    for (size_t i = 0; i < controls.size(); ++i) {
                        const ImVec2 point = controls[i];
                        if (point.x < minX || point.x > maxX || point.y < minY || point.y > maxY)
                            continue;
                        if (std::find(splineObject->selected_points.begin(),
                                      splineObject->selected_points.end(), static_cast<int>(i)) ==
                            splineObject->selected_points.end()) {
                            splineObject->selected_points.push_back(static_cast<int>(i));
                        }
                    }
                    splineObject->selected_point = splineObject->selected_points.empty()
                        ? -1 : splineObject->selected_points.back();
                }
                splineObject->selection_box_active = false;
            }
        }

        if (selected && pointEditing && splineObject->edit_tool == SplineEditTool::InsertPoint &&
            !curve.empty() && spline.segmentCount() > 0 &&
            spline.curveType != SplineCurveType::BSpline &&
            !ImGui::GetIO().WantTextInput && !ImGuizmo::IsOver()) {
            float previewDistance = 10.0f;
            int previewSample = -1;
            for (int i = 0; i < static_cast<int>(curve.size()); ++i) {
                const float dx = io.MousePos.x - curve[static_cast<size_t>(i)].x;
                const float dy = io.MousePos.y - curve[static_cast<size_t>(i)].y;
                const float distance = std::sqrt(dx * dx + dy * dy);
                if (distance < previewDistance) {
                    previewDistance = distance;
                    previewSample = i;
                }
            }
            if (previewSample >= 0) {
                const ImVec2 marker = curve[static_cast<size_t>(previewSample)];
                draw->AddCircle(marker, 8.0f, IM_COL32(255, 180, 60, 255), 16, 2.0f);
                const int sampleCount = static_cast<int>(curve.size()) - 1;
                const float globalT = std::clamp(
                    static_cast<float>(previewSample) / std::max(1, sampleCount), 0.001f, 0.999f);
                const float scaledT = globalT * static_cast<float>(spline.segmentCount());
                const int previewSegment = std::min(
                    static_cast<int>(std::floor(scaledT)),
                    static_cast<int>(spline.segmentCount()) - 1);
                const float previewT = scaledT - static_cast<float>(previewSegment);
                splineObject->insert_preview_position = spline.samplePosition(globalT);
                splineObject->has_insert_preview = true;
                if (ImGui::IsMouseClicked(0)) {
                    const int segment = previewSegment;
                    const float t = previewT;
                    int inserted = -1;
                    if (SplineEditService::insertBezierPoint(
                            spline, segment, t, &inserted)) {
                        splineObject->selected_point = inserted;
                        splineObject->selected_points = {inserted};
                        splineObject->edit_tool = SplineEditTool::Select;
                        ProjectManager::getInstance().markModified();
                        return;
                    }
                }
            }
        }

        // Viewport picking: unlike renderable mesh objects, authoring splines
        // intentionally do not ray-hit. Their screen-space source overlay is
        // the canonical picker and keeps the authoring object mesh-free.
        if (!ImGui::GetIO().WantTextInput && !ImGuizmo::IsOver()) {
            int hitPoint = -1;
            float bestPointDistance = 14.0f;
            for (size_t i = 0; i < controls.size(); ++i) {
                const float dx = io.MousePos.x - controls[i].x;
                const float dy = io.MousePos.y - controls[i].y;
                const float distance = std::sqrt(dx * dx + dy * dy);
                if (distance < bestPointDistance) {
                    bestPointDistance = distance;
                    hitPoint = static_cast<int>(i);
                }
            }
            int hitCurve = -1;
            float bestCurveDistance = 10.0f;
            for (size_t i = 0; i < curve.size(); ++i) {
                const float dx = io.MousePos.x - curve[i].x;
                const float dy = io.MousePos.y - curve[i].y;
                const float distance = std::sqrt(dx * dx + dy * dy);
                if (distance < bestCurveDistance) {
                    bestCurveDistance = distance;
                    hitCurve = static_cast<int>(i);
                }
            }
            if (ImGui::IsMouseClicked(0) && pointEditing &&
                splineObject->edit_tool == SplineEditTool::Subdivide && hitCurve >= 0 &&
                spline.segmentCount() > 0 && spline.curveType != SplineCurveType::BSpline) {
                const float globalT = static_cast<float>(hitCurve) /
                    static_cast<float>(std::max<size_t>(1, curve.size() - 1));
                const float scaledT = globalT * static_cast<float>(spline.segmentCount());
                const int segment = std::min(static_cast<int>(std::floor(scaledT)),
                                             static_cast<int>(spline.segmentCount()) - 1);
                int inserted = -1;
                if (SplineEditService::subdivideBezierSegment(
                        spline, segment, splineObject->subdivide_cuts, &inserted)) {
                    splineObject->selected_point = inserted;
                    splineObject->selected_points = {inserted};
                    ProjectManager::getInstance().markModified();
                }
                return;
            }
            if (ImGui::IsMouseClicked(0) && pointEditing &&
                splineObject->edit_tool == SplineEditTool::Extrude && hitPoint >= 0 &&
                !spline.isClosed && spline.points.size() >= 2 &&
                (hitPoint == 0 || hitPoint == static_cast<int>(spline.points.size()) - 1)) {
                const size_t pointIndex = static_cast<size_t>(hitPoint);
                const size_t neighbor = hitPoint == 0 ? 1u : spline.points.size() - 2u;
                const Vec3 direction = spline.points[pointIndex].position - spline.points[neighbor].position;
                int inserted = -1;
                if (SplineEditService::extrudeEndpoint(
                        spline, hitPoint, spline.points[pointIndex].position + direction, &inserted)) {
                    splineObject->selected_point = inserted;
                    splineObject->selected_points = {inserted};
                    ProjectManager::getInstance().markModified();
                }
                return;
            }
            if (ImGui::IsMouseClicked(0) && (hitPoint >= 0 || hitCurve >= 0)) {
                ctx.selection.selectObject(splineObject,
                    static_cast<int>(objectIndex), splineObject->nodeName);
                if (pointEditing && hitPoint >= 0) {
                    if (io.KeyCtrl) {
                        auto it = std::find(splineObject->selected_points.begin(),
                                            splineObject->selected_points.end(), hitPoint);
                        if (it == splineObject->selected_points.end()) {
                            splineObject->selected_points.push_back(hitPoint);
                        } else {
                            splineObject->selected_points.erase(it);
                        }
                    } else {
                        splineObject->selected_points = {hitPoint};
                    }
                    splineObject->selected_point = hitPoint;
                } else {
                    splineObject->selected_point = -1;
                    splineObject->selected_points.clear();
                }
                return;
            }
        }
    }
}

bool pickProfileSpline(UIContext& ctx) {
    if (!ctx.scene.camera || !ImGui::IsMouseClicked(0)) return false;
    ImGuiIO& io = ImGui::GetIO();
    for (size_t objectIndex = 0; objectIndex < ctx.scene.world.objects.size(); ++objectIndex) {
        auto splineObject = std::dynamic_pointer_cast<SplineObject>(ctx.scene.world.objects[objectIndex]);
        if (!splineObject || !splineObject->visible || splineObject->spline.points.empty()) continue;
        const Matrix4x4 transform = splineObject->transform
            ? splineObject->transform->getFinal() : Matrix4x4::identity();
        auto projectLocal = [&](const Vec3& local, ImVec2& screen) {
            return project(*ctx.scene.camera, transform.transform_point(local),
                           io.DisplaySize.x, io.DisplaySize.y, screen);
        };
        int pointHit = -1;
        float best = 14.0f;
        for (size_t i = 0; i < splineObject->spline.points.size(); ++i) {
            ImVec2 screen;
            if (!projectLocal(splineObject->spline.points[i].position, screen)) continue;
            const float dx = io.MousePos.x - screen.x;
            const float dy = io.MousePos.y - screen.y;
            const float distance = std::sqrt(dx * dx + dy * dy);
            if (distance < best) { best = distance; pointHit = static_cast<int>(i); }
        }
        if (pointHit < 0) {
            constexpr int kSamples = 64;
            for (int i = 0; i <= kSamples; ++i) {
                ImVec2 screen;
                if (!projectLocal(splineObject->spline.samplePosition(static_cast<float>(i) / kSamples), screen)) continue;
                const float dx = io.MousePos.x - screen.x;
                const float dy = io.MousePos.y - screen.y;
                best = (std::min)(best, std::sqrt(dx * dx + dy * dy));
            }
        }
        const bool pointEditing = ctx.selection.selected.spline_object == splineObject &&
                                  splineObject->edit_mode;
        // Insert/Subdivide/Extrude are consumed by the authoring overlay on
        // the next draw pass. The generic object picker must not swallow the
        // click first.
        if (pointEditing && splineObject->edit_tool != SplineEditTool::Select)
            return false;
        if (pointHit >= 0 || best < 10.0f) {
            ctx.selection.selectObject(splineObject, static_cast<int>(objectIndex), splineObject->nodeName);
            if (pointEditing && pointHit >= 0) {
                if (io.KeyCtrl) {
                    auto it = std::find(splineObject->selected_points.begin(),
                                        splineObject->selected_points.end(), pointHit);
                    if (it == splineObject->selected_points.end()) {
                        splineObject->selected_points.push_back(pointHit);
                    } else {
                        splineObject->selected_points.erase(it);
                    }
                } else {
                    splineObject->selected_points = {pointHit};
                }
                splineObject->selected_point = pointHit;
            } else {
                splineObject->selected_point = -1;
                splineObject->selected_points.clear();
            }
            return true;
        }
    }
    return false;
}

} // namespace MeshEdit
