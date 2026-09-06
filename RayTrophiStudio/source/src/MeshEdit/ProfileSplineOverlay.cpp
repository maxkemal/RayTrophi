#include "MeshEdit/ProfileSplineOverlay.h"

#include "MeshEdit/SplineObject.h"
#include "MeshEdit/SplineSurfaceAuthoring.h"
#include "MeshEdit/SplineEditService.h"
#include "MeshEdit/SplineEvaluationService.h"
#include "MeshEdit/SplineAnimation.h"
#include "DNA/GeometryDetail.h"
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
#include <cstdio>

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

// Surface Snap now lives in MeshEdit/SplineSurfaceAuthoring so the River tool
// and the scene.raycast IPC method share one placement policy. Profile spline
// authoring wants the full policy: a point may be laid on other geometry, on
// the terrain, or on the ground plane past the edge of the world.
bool surfaceSnapPosition(UIContext& ctx, const ImVec2& mousePos, Vec3& outPosition) {
    SurfaceSnapResult snap;
    if (!snapToSurface(ctx, mousePos, SurfaceFilter::MeshAndTerrain, snap)) return false;
    outPosition = snap.position;
    return true;
}

void addSmoothTriangle(ImDrawList* draw, const ImVec2& a, const ImVec2& b,
                       const ImVec2& c, ImU32 colorA, ImU32 colorB, ImU32 colorC) {
    const ImVec2 uv = draw->_Data->TexUvWhitePixel;
    draw->PrimReserve(3, 3);
    draw->PrimWriteIdx(static_cast<ImDrawIdx>(draw->_VtxCurrentIdx));
    draw->PrimWriteIdx(static_cast<ImDrawIdx>(draw->_VtxCurrentIdx + 1));
    draw->PrimWriteIdx(static_cast<ImDrawIdx>(draw->_VtxCurrentIdx + 2));
    draw->PrimWriteVtx(a, uv, colorA);
    draw->PrimWriteVtx(b, uv, colorB);
    draw->PrimWriteVtx(c, uv, colorC);
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
        if (!splineObject || !splineObject->visible) continue;
        // An empty curve normally has nothing to draw, but a Draw-armed one has
        // to stay in this loop or it can never receive the click that lays its
        // first point - which is exactly why drawing a curve from nothing was
        // impossible before.
        if (splineObject->spline.points.empty() &&
            splineObject->edit_tool != SplineEditTool::Draw) continue;
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

        if (selected && splineObject->transform) {
            const Vec3 pivotWorld =
                splineObject->transform->getPivotMatrix().getTranslation();
            ImVec2 pivotScreen;
            if (project(*ctx.scene.camera, pivotWorld, io.DisplaySize.x,
                        io.DisplaySize.y, pivotScreen)) {
                constexpr float arm = 8.0f;
                const ImU32 pivotColor = IM_COL32(255, 145, 45, 255);
                draw->AddLine(ImVec2(pivotScreen.x - arm, pivotScreen.y),
                              ImVec2(pivotScreen.x + arm, pivotScreen.y),
                              pivotColor, 2.0f);
                draw->AddLine(ImVec2(pivotScreen.x, pivotScreen.y - arm),
                              ImVec2(pivotScreen.x, pivotScreen.y + arm),
                              pivotColor, 2.0f);
                draw->AddCircle(pivotScreen, 3.0f, IM_COL32(255, 220, 150, 255),
                                12, 1.0f);
                draw->AddText(ImVec2(pivotScreen.x + 11.0f, pivotScreen.y - 9.0f),
                              pivotColor, "Pivot");
            }
        }

        // Rendered shading already spends the frame budget on the path tracer.
        // The source overlay is authoring feedback, not render geometry: use a
        // lighter screen approximation there while keeping the solid/edit view
        // crisp for hit testing.
        const int kSamples = g_solid_viewport_active ? 64 : 24;
        std::vector<ImVec2> curve;
        curve.reserve(kSamples + 1);
        for (int i = 0; i <= kSamples; ++i) {
            ImVec2 screen;
            const SplineEvaluation sample = SplineEvaluationService::evaluate(
                spline, static_cast<float>(i) / kSamples);
            if (sample.valid && projectLocal(sample.position, screen)) {
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

        if (selected && splineObject->profile_preview_geometry) {
            const auto& preview = *splineObject->profile_preview_geometry;
            const Vec3* positions = preview.get_attribute_data<Vec3>("P");
            const Vec3* normals = preview.get_attribute_data<Vec3>("N");
            const size_t triangleCount = preview.indices.size() / 3;
            const size_t stride = std::max<size_t>(1, triangleCount / 12000);
            if (positions) {
                struct SurfaceTriangle {
                    ImVec2 a, b, c;
                    float depth = 0.0f;
                    ImU32 colorA = 0, colorB = 0, colorC = 0;
                };
                std::vector<SurfaceTriangle> surface;
                surface.reserve((triangleCount + stride - 1) / stride);
                const Vec3 cameraForward =
                    (ctx.scene.camera->lookat - ctx.scene.camera->lookfrom).normalize();
                const Vec3 lightDirection = Vec3(-0.35f, 0.8f, -0.45f).normalize();
                for (size_t triangle = 0; triangle < triangleCount; triangle += stride) {
                    const size_t base = triangle * 3;
                    const uint32_t ia = preview.indices[base];
                    const uint32_t ib = preview.indices[base + 1];
                    const uint32_t ic = preview.indices[base + 2];
                    if (ia >= preview.get_vertex_count() || ib >= preview.get_vertex_count() ||
                        ic >= preview.get_vertex_count()) continue;
                    const Vec3 wa = splineObject->profile_preview_transform.transform_point(positions[ia]);
                    const Vec3 wb = splineObject->profile_preview_transform.transform_point(positions[ib]);
                    const Vec3 wc = splineObject->profile_preview_transform.transform_point(positions[ic]);
                    ImVec2 a, b, c;
                    if (!project(*ctx.scene.camera, wa, io.DisplaySize.x, io.DisplaySize.y, a) ||
                        !project(*ctx.scene.camera, wb, io.DisplaySize.x, io.DisplaySize.y, b) ||
                        !project(*ctx.scene.camera, wc, io.DisplaySize.x, io.DisplaySize.y, c)) continue;
                    const Vec3 face = (wb - wa).cross(wc - wa);
                    if (face.length_squared() <= 1.0e-12f) continue;
                    const Vec3 faceNormal = face.normalize();
                    auto previewColor = [&](uint32_t vertex) {
                        Vec3 normal = faceNormal;
                        if (normals && normals[vertex].length_squared() > 1.0e-12f) {
                            normal = splineObject->profile_preview_transform
                                .transform_vector(normals[vertex]).normalize();
                        }
                        const float diffuse = std::clamp(
                            0.24f + 0.76f * std::abs(normal.dot(lightDirection)), 0.0f, 1.0f);
                        // Neutral studio clay: preview no longer reads as a yellow
                        // selection mask, while per-vertex colour interpolation avoids
                        // the faceted look of one colour per projected triangle.
                        const int red = static_cast<int>(58.0f + 157.0f * diffuse);
                        const int green = static_cast<int>(64.0f + 158.0f * diffuse);
                        const int blue = static_cast<int>(72.0f + 160.0f * diffuse);
                        return IM_COL32(red, green, blue, 232);
                    };
                    const Vec3 center = (wa + wb + wc) / 3.0f;
                    surface.push_back({a, b, c,
                        (center - ctx.scene.camera->lookfrom).dot(cameraForward),
                        previewColor(ia), previewColor(ib), previewColor(ic)});
                }
                // Painter ordering gives the transient surface coherent occlusion without
                // publishing it into world.objects or polluting hierarchy/serialization.
                std::sort(surface.begin(), surface.end(),
                    [](const SurfaceTriangle& lhs, const SurfaceTriangle& rhs) {
                        return lhs.depth > rhs.depth;
                });
                for (const auto& triangle : surface) {
                    addSmoothTriangle(draw, triangle.a, triangle.b, triangle.c,
                                      triangle.colorA, triangle.colorB, triangle.colorC);
                }
            }
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
                splineObject->insert_preview_position =
                    SplineEvaluationService::evaluate(spline, globalT).position;
                splineObject->has_insert_preview = true;
                if (ImGui::IsMouseClicked(0)) {
                    const int segment = previewSegment;
                    const float t = previewT;
                    int inserted = -1;
                    if (SplineEditService::insertPoint(
                            spline, segment, t, &inserted)) {
                        propagateSplineInsertToKeys(
                            ctx.scene.timeline, splineObject->nodeName, segment, t);
                        splineObject->selected_point = inserted;
                        splineObject->selected_points = {inserted};
                        ProjectManager::getInstance().markModified();
                        return;
                    }
                }
            }
        }

        // Drag feedback. The gizmo manipulates; this reports. An overlay drawn in
        // screen space with no depth test cannot convey where a point sits in 3D
        // on its own, so the numbers carry what the picture cannot: how far the
        // point has moved, and where it is relative to the object it belongs to.
        if (selected && pointEditing && splineObject->point_drag_dirty &&
            splineObject->drag_origin_valid &&
            splineObject->selected_point >= 0 &&
            splineObject->selected_point < static_cast<int>(spline.points.size())) {
            const size_t activeIndex = static_cast<size_t>(splineObject->selected_point);
            const Vec3 localNow = spline.points[activeIndex].position;
            const Vec3 delta = localNow - splineObject->drag_origin_local;
            const Vec3 worldNow = transform.transform_point(localNow);

            ImVec2 anchorScreen;
            if (projectLocal(localNow, anchorScreen)) {
                // A ghost of where the drag started, plus the travel line. This
                // is the part that answers "how much did it move" at a glance,
                // before anyone reads a number.
                ImVec2 originScreen;
                if (projectLocal(splineObject->drag_origin_local, originScreen)) {
                    draw->AddCircle(originScreen, 5.0f, IM_COL32(255, 255, 255, 90), 12, 1.5f);
                    draw->AddLine(originScreen, anchorScreen, IM_COL32(255, 210, 90, 200), 1.5f);
                }

                // Neighbour segments with their lengths: a curve point's real
                // context is the points it connects to, not the world origin.
                const auto drawNeighbour = [&](int neighbourIndex) {
                    if (neighbourIndex < 0 ||
                        neighbourIndex >= static_cast<int>(spline.points.size())) return;
                    const Vec3 neighbourLocal =
                        spline.points[static_cast<size_t>(neighbourIndex)].position;
                    ImVec2 neighbourScreen;
                    if (!projectLocal(neighbourLocal, neighbourScreen)) return;
                    draw->AddLine(anchorScreen, neighbourScreen, IM_COL32(120, 200, 255, 130), 1.0f);
                    const Vec3 span = transform.transform_point(neighbourLocal) - worldNow;
                    char label[32];
                    std::snprintf(label, sizeof(label), "%.2f", span.length());
                    const ImVec2 mid((anchorScreen.x + neighbourScreen.x) * 0.5f,
                                     (anchorScreen.y + neighbourScreen.y) * 0.5f);
                    draw->AddText(mid, IM_COL32(160, 215, 255, 200), label);
                };
                drawNeighbour(splineObject->selected_point - 1);
                drawNeighbour(splineObject->selected_point + 1);

                // Distance to the object's own pivot, so the point can be read
                // against the object rather than against the world.
                ImVec2 pivotScreen;
                const Vec3 pivotLocal(0.0f, 0.0f, 0.0f);
                if (projectLocal(pivotLocal, pivotScreen)) {
                    draw->AddLine(anchorScreen, pivotScreen, IM_COL32(200, 200, 200, 70), 1.0f);
                }

                char readout[192];
                std::snprintf(readout, sizeof(readout),
                              "d %+.3f %+.3f %+.3f  (%.3f)\n"
                              "local %.3f %.3f %.3f\n"
                              "world %.3f %.3f %.3f",
                              delta.x, delta.y, delta.z, delta.length(),
                              localNow.x, localNow.y, localNow.z,
                              worldNow.x, worldNow.y, worldNow.z);
                const ImVec2 textSize = ImGui::CalcTextSize(readout);
                const ImVec2 boxMin(anchorScreen.x + 14.0f, anchorScreen.y + 14.0f);
                const ImVec2 boxMax(boxMin.x + textSize.x + 10.0f, boxMin.y + textSize.y + 8.0f);
                draw->AddRectFilled(boxMin, boxMax, IM_COL32(18, 18, 20, 205), 4.0f);
                draw->AddRect(boxMin, boxMax, IM_COL32(255, 210, 90, 140), 4.0f);
                draw->AddText(ImVec2(boxMin.x + 5.0f, boxMin.y + 4.0f),
                              IM_COL32(240, 240, 240, 255), readout);
                if (io.KeyCtrl) {
                    draw->AddText(ImVec2(boxMin.x + 5.0f, boxMax.y + 2.0f),
                                  IM_COL32(255, 210, 90, 220),
                                  io.KeyShift ? "snap 0.1" : "snap 1.0");
                }
            }
        }

        // Live cursor follow for Draw and Extrude. Without this the tool still
        // worked but felt like it was extending to some automatic point: the
        // snap only happened at the instant of the click, so nothing on screen
        // told the artist where the next point was going to land. This is the
        // half that makes it feel like the River tool.
        const bool appendToolActive = selected && pointEditing && !spline.isClosed &&
            (splineObject->edit_tool == SplineEditTool::Draw ||
             splineObject->edit_tool == SplineEditTool::Extrude);
        // The picker now claims every click while an append tool is active, so
        // there has to be a way OUT that is not "find the combo again". Escape
        // returns to Select, which is also what makes the claim safe: without an
        // exit the tool would trap the viewport.
        if (appendToolActive && !io.WantTextInput &&
            ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            splineObject->edit_tool = SplineEditTool::Select;
        }
        if (appendToolActive && !io.WantTextInput && !ImGuizmo::IsOver() && !io.WantCaptureMouse) {
            SurfaceSnapResult snap;
            if (snapToSurface(ctx, io.MousePos, SurfaceFilter::MeshAndTerrain, snap)) {
                ImVec2 cursorScreen;
                if (project(*ctx.scene.camera, snap.position,
                            io.DisplaySize.x, io.DisplaySize.y, cursorScreen)) {
                    // The marker colour reports WHAT was hit, so a point that is
                    // about to land on a rock instead of the terrain is visible
                    // before the click rather than after it.
                    ImU32 markerColor = IM_COL32(120, 200, 255, 255);   // terrain
                    if (snap.kind == SurfaceHitKind::Mesh)
                        markerColor = IM_COL32(255, 200, 90, 255);
                    else if (snap.kind == SurfaceHitKind::GroundPlane)
                        markerColor = IM_COL32(150, 150, 150, 255);
                    draw->AddCircle(cursorScreen, 7.0f, markerColor, 16, 2.0f);
                    draw->AddCircleFilled(cursorScreen, 2.5f, markerColor);

                    const int tailIndex = splineObject->edit_tool == SplineEditTool::Extrude
                        ? splineObject->selected_point
                        : (spline.points.empty()
                               ? -1 : static_cast<int>(spline.points.size()) - 1);
                    if (tailIndex >= 0 && tailIndex < static_cast<int>(spline.points.size())) {
                        ImVec2 tailScreen;
                        if (projectLocal(spline.points[static_cast<size_t>(tailIndex)].position,
                                         tailScreen)) {
                            draw->AddLine(tailScreen, cursorScreen,
                                          IM_COL32(255, 255, 255, 110), 1.5f);
                        }
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
                spline.segmentCount() > 0) {
                const float globalT = static_cast<float>(hitCurve) /
                    static_cast<float>(std::max<size_t>(1, curve.size() - 1));
                const float scaledT = globalT * static_cast<float>(spline.segmentCount());
                const int segment = std::min(static_cast<int>(std::floor(scaledT)),
                                             static_cast<int>(spline.segmentCount()) - 1);
                int inserted = -1;
                if (SplineEditService::subdivideSegment(
                        spline, segment, splineObject->subdivide_cuts, &inserted)) {
                    propagateSplineSubdivideToKeys(
                        ctx.scene.timeline, splineObject->nodeName, segment,
                        splineObject->subdivide_cuts);
                    splineObject->selected_point = inserted;
                    splineObject->selected_points = {inserted};
                    ProjectManager::getInstance().markModified();
                }
                return;
            }
            // Draw: the same "click lays the next point" model as Extrude, but
            // valid from ZERO points, so this is how a curve BEGINS. Placement
            // uses the full surface policy (mesh, terrain, then ground plane).
            if (pointEditing && splineObject->edit_tool == SplineEditTool::Draw &&
                !spline.isClosed) {
                if (ImGui::IsMouseClicked(0) && !io.WantCaptureMouse) {
                    SurfaceSnapResult snap;
                    if (snapToSurface(ctx, io.MousePos, SurfaceFilter::MeshAndTerrain, snap)) {
                        // The curve carries a transform, so the click has to be
                        // taken back into local space or every point after a
                        // moved/rotated curve lands somewhere else.
                        const Vec3 localPosition = transform.inverse().transform_point(snap.position);
                        const BezierSpline beforeSpline = spline;
                        const int previousEndpoint =
                            spline.points.empty()
                                ? -1 : static_cast<int>(spline.points.size()) - 1;
                        int inserted = -1;
                        if (SplineEditService::appendPointAtEnd(spline, localPosition, &inserted)) {
                            if (previousEndpoint >= 0 && spline.points.size() > 2) {
                                propagateSplineExtrudeToKeys(
                                    ctx.scene.timeline, splineObject->nodeName, beforeSpline,
                                    previousEndpoint, localPosition);
                            }
                            splineObject->selected_point = inserted;
                            splineObject->selected_points = {inserted};
                            ProjectManager::getInstance().markModified();
                        }
                    }
                    return;
                }
            }
            // Extrude follows River's "Add" model: once the tool is active on a
            // valid open-spline endpoint, every click lays down the next point
            // wherever the cursor is aiming (surface/terrain/ground snap) and
            // that new point becomes the endpoint for the next click - it does
            // not require re-aiming at the tiny endpoint marker each time.
            const int extrudeEndpoint = splineObject->selected_point;
            const bool extrudeToolActive = pointEditing &&
                splineObject->edit_tool == SplineEditTool::Extrude &&
                !spline.isClosed && spline.points.size() >= 2 &&
                extrudeEndpoint >= 0 &&
                (extrudeEndpoint == 0 ||
                 extrudeEndpoint == static_cast<int>(spline.points.size()) - 1);
            if (ImGui::IsMouseClicked(0) && extrudeToolActive) {
                Vec3 targetPosition;
                if (surfaceSnapPosition(ctx, io.MousePos, targetPosition)) {
                    // Control points are stored in the curve's local space. The
                    // snap result is a world position, so it has to come back
                    // through the inverse transform - without this, extruding a
                    // moved or rotated curve laid the point at the offset twice.
                    targetPosition = transform.inverse().transform_point(targetPosition);
                } else {
                    targetPosition = spline.points[static_cast<size_t>(extrudeEndpoint)].position;
                }
                const BezierSpline beforeSpline = spline;
                int inserted = -1;
                if (SplineEditService::extrudeEndpoint(
                        spline, extrudeEndpoint, targetPosition, &inserted)) {
                    propagateSplineExtrudeToKeys(
                        ctx.scene.timeline, splineObject->nodeName, beforeSpline,
                        extrudeEndpoint, targetPosition);
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
        if (!splineObject || !splineObject->visible) continue;
        if (splineObject->spline.points.empty() &&
            splineObject->edit_tool != SplineEditTool::Draw) continue;
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
                const SplineEvaluation sample = SplineEvaluationService::evaluate(
                    splineObject->spline, static_cast<float>(i) / kSamples);
                if (!sample.valid || !projectLocal(sample.position, screen)) continue;
                const float dx = io.MousePos.x - screen.x;
                const float dy = io.MousePos.y - screen.y;
                best = (std::min)(best, std::sqrt(dx * dx + dy * dy));
            }
        }
        const bool pointEditing = ctx.selection.selected.spline_object == splineObject &&
                                  splineObject->edit_mode;
        // An active authoring tool CLAIMS the click: the overlay consumes it on
        // this frame's draw pass, so the generic object picker must not run.
        //
        // This returned false, which is the opposite - false means "not a spline
        // click" and lets scene_ui_selection.cpp run the whole picker. Over
        // ordinary geometry that was survivable, because clicking empty space
        // selects nothing and the spline stays selected. Over a TERRAIN there is
        // no empty space: every click hit the terrain, selected it, dropped the
        // spline selection, and the next click had nothing to draw on. The
        // symptom was "I have to reselect the curve after every click" - the
        // tool looked broken only where a terrain existed, which is exactly the
        // surface the tool was built for.
        if (pointEditing && splineObject->edit_tool != SplineEditTool::Select)
            return true;
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
