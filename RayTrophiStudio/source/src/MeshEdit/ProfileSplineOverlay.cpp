#include "MeshEdit/ProfileSplineOverlay.h"

#include "MeshEdit/SplineObject.h"
#include "MeshEdit/SplineEditService.h"
#include "MeshEdit/SplineEvaluationService.h"
#include "MeshEdit/SplineAnimation.h"
#include "DNA/GeometryDetail.h"
#include "SceneSelection.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "globals.h"
#include "Camera.h"
#include "TerrainManager.h"
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

// Surface Snap: reuses the same scene BVH (+ linear fallback) the viewport
// selection tool queries, the terrain raycast River already uses, and
// River's own Y=0 ground fallback when neither is under the cursor - so a
// new spline point always lands where the cursor is actually aiming, the
// same River-style "click to place the next point" model. Only returns
// false in the genuinely degenerate case (ray nearly parallel to Y=0,
// looking at the horizon with nothing behind it).
bool surfaceSnapPosition(UIContext& ctx, const ImVec2& mousePos, Vec3& outPosition) {
    if (!ctx.scene.camera) return false;
    ImGuiIO& io = ImGui::GetIO();
    const float viewportWidth = std::max(1.0f, io.DisplaySize.x);
    const float viewportHeight = std::max(1.0f, io.DisplaySize.y);
    const float u = mousePos.x / viewportWidth;
    const float v = 1.0f - (mousePos.y / viewportHeight);
    const Ray ray = ctx.scene.camera->get_ray(u, v);

    float closestT = 1e9f;
    bool hitMesh = false;
    HitRecord hitRecord;
    if (ctx.scene.bvh) {
        HitRecord temp;
        if (ctx.scene.bvh->hit(ray, 0.001f, closestT, temp)) {
            hitMesh = true;
            closestT = temp.t;
            hitRecord = temp;
        }
    }
    if (!hitMesh) {
        for (const auto& object : ctx.scene.world.objects) {
            if (!object) continue;
            HitRecord temp;
            if (object->hit(ray, 0.001f, closestT, temp)) {
                hitMesh = true;
                closestT = temp.t;
                hitRecord = temp;
            }
        }
    }
    if (hitMesh) {
        outPosition = ray.origin + ray.direction * hitRecord.t;
        return true;
    }

    if (TerrainManager::getInstance().hasActiveTerrain()) {
        float closestTerrainT = 1e20f;
        bool hitTerrain = false;
        for (auto& terrain : TerrainManager::getInstance().getTerrains()) {
            float terrainT = 0.0f;
            Vec3 terrainNormal;
            if (TerrainManager::getInstance().intersectRay(&terrain, ray, terrainT, terrainNormal) &&
                terrainT < closestTerrainT) {
                closestTerrainT = terrainT;
                hitTerrain = true;
            }
        }
        if (hitTerrain) {
            outPosition = ray.origin + ray.direction * closestTerrainT;
            return true;
        }
    }

    // No mesh/terrain under the cursor: fall back to the same Y=0 ground
    // plane the River tool uses when no terrain is active, so clicking past
    // the edge of the world still places a point where the cursor is aiming.
    if (std::fabs(ray.direction.y) > 0.01f) {
        const float t = -ray.origin.y / ray.direction.y;
        if (t > 0.0f) {
            outPosition = ray.origin + ray.direction * t;
            return true;
        }
    }
    return false;
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
                if (!surfaceSnapPosition(ctx, io.MousePos, targetPosition)) {
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
