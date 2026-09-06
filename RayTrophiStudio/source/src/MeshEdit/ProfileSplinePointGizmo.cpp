#include "MeshEdit/ProfileSplinePointGizmo.h"

#include "MeshEdit/SplineObject.h"
#include "SceneSelection.h"
#include "ProjectManager.h"
#include "scene_ui.h"
#include "Camera.h"
#include "ImGuizmo.h"

#include <algorithm>
#include <cmath>

namespace MeshEdit {
namespace {
void flattenToPlane(Vec3& value, SplinePlane plane) {
    if (plane == SplinePlane::XY) value.z = 0.0f;
    else if (plane == SplinePlane::XZ) value.y = 0.0f;
    else if (plane == SplinePlane::YZ) value.x = 0.0f;
    // Free: unconstrained 3D authoring, no axis is zeroed.
}
}

bool drawProfileSplinePointGizmo(UIContext& ctx) {
    // Pivot edit owns the object gizmo even while spline point-edit mode is
    // enabled. Otherwise the point gizmo would consume the frame and make P
    // appear to do nothing for spline objects.
    if (ctx.scene_ui_ptr && ctx.scene_ui_ptr->pivot_edit_mode) return false;
    auto splineObject = ctx.selection.selected.spline_object;
    if (!splineObject || !splineObject->edit_mode || splineObject->selected_point < 0 ||
        splineObject->selected_point >= static_cast<int>(splineObject->spline.points.size()) ||
        !ctx.scene.camera) {
        return false;
    }

    ImGuiIO& io = ImGui::GetIO();
    const Camera& camera = *ctx.scene.camera;
    // ★ Must match drawTransformGizmo's rule, not just camera.orthographic. The
    // Rendered viewport path-traces in PERSPECTIVE even under an orthographic
    // camera, so claiming ortho there builds a projection that disagrees with
    // what is on screen: the handle you grab no longer points where it looks,
    // and a small mouse move maps to a large, arbitrary world delta. That is
    // the "small drag sometimes jumps somewhere meaningless" report, and it
    // only happens in Rendered + ortho - which is why it looked intermittent.
    const bool viewportIsRendered = ctx.scene_ui_ptr &&
        ctx.scene_ui_ptr->viewport_settings.shading_mode == 2;
    const bool ortho = camera.orthographic && !viewportIsRendered;
    const float aspect = io.DisplaySize.x / std::max(1.0f, io.DisplaySize.y);
    const float nearPlane = 0.1f;
    const float farPlane = 10000.0f;
    const Vec3 forward = (camera.lookat - camera.lookfrom).normalize();
    const Vec3 right = forward.cross(camera.vup).normalize();
    const Vec3 up = right.cross(forward).normalize();

    float view[16] = {
        right.x, up.x, -forward.x, 0.0f,
        right.y, up.y, -forward.y, 0.0f,
        right.z, up.z, -forward.z, 0.0f,
        -right.dot(camera.lookfrom), -up.dot(camera.lookfrom), forward.dot(camera.lookfrom), 1.0f
    };
    float projection[16] = {};
    if (ortho) {
        const float height = std::max(0.001f, camera.ortho_height);
        const float width = height * aspect;
        projection[0] = 2.0f / width;
        projection[5] = 2.0f / height;
        projection[10] = -2.0f / (farPlane - nearPlane);
        projection[14] = -(farPlane + nearPlane) / (farPlane - nearPlane);
        projection[15] = 1.0f;
    } else {
        const float tanHalfFov = std::tan(camera.vfov * 3.14159265359f / 360.0f);
        projection[0] = 1.0f / (aspect * tanHalfFov);
        projection[5] = 1.0f / tanHalfFov;
        projection[10] = -(farPlane + nearPlane) / (farPlane - nearPlane);
        projection[11] = -1.0f;
        projection[14] = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
    }

    // ImGuizmo's per-frame state must be reset even when this function bails
    // out below. It returns TRUE in those cases, so the caller does not run
    // drawTransformGizmo either - and then nothing calls BeginFrame that frame,
    // leaving IsOver() latched from the previous one and blocking viewport
    // clicks. Manipulate() is what pushes the clip rect, so calling these three
    // early keeps the original concern intact.
    ImGuizmo::SetOrthographic(ortho);
    ImGuizmo::BeginFrame();
    ImGuizmo::SetRect(0.0f, 0.0f, io.DisplaySize.x, io.DisplaySize.y);

    const Matrix4x4 objectTransform = splineObject->transform
        ? splineObject->transform->getFinal() : Matrix4x4::identity();
    const Vec3 localPoint = splineObject->spline.points[splineObject->selected_point].position;
    const Vec3 worldPoint = objectTransform.transform_point(localPoint);
    const float pointDepth = (worldPoint - camera.lookfrom).dot(forward);
    // ImGuizmo pushes a clip rect before its behind-camera early return. Guard
    // that case here so a hidden point cannot leak clip-rect entries over time.
    if (!ortho && pointDepth <= 0.01f) return true;
    if (!std::isfinite(worldPoint.x) || !std::isfinite(worldPoint.y) ||
        !std::isfinite(worldPoint.z)) return true;
    Matrix4x4 pointMatrix = Matrix4x4::translation(worldPoint);
    float gizmo[16] = {
        pointMatrix.m[0][0], pointMatrix.m[1][0], pointMatrix.m[2][0], pointMatrix.m[3][0],
        pointMatrix.m[0][1], pointMatrix.m[1][1], pointMatrix.m[2][1], pointMatrix.m[3][1],
        pointMatrix.m[0][2], pointMatrix.m[1][2], pointMatrix.m[2][2], pointMatrix.m[3][2],
        pointMatrix.m[0][3], pointMatrix.m[1][3], pointMatrix.m[2][3], pointMatrix.m[3][3]
    };

    ImGuizmo::SetGizmoSizeClipSpace(0.09f);
    // Ctrl snaps to a round metre grid. A point gizmo is small on screen, so a
    // one-pixel wobble is a real world offset at distance; snapping is what
    // makes "nudge it a bit" a decision rather than an estimate. Shift+Ctrl
    // takes the finer step for close work.
    const float snapStep = io.KeyShift ? 0.1f : 1.0f;
    const float snapValues[3] = {snapStep, snapStep, snapStep};
    ImGuizmo::Manipulate(view, projection, ImGuizmo::TRANSLATE,
                         ImGuizmo::WORLD, gizmo, nullptr,
                         io.KeyCtrl ? snapValues : nullptr);

    if (ImGuizmo::IsUsing() && splineObject->transform) {
        // Captured once per drag: point_drag_dirty is still false on the frame
        // the drag starts, so this records the position BEFORE any delta lands.
        if (!splineObject->point_drag_dirty) {
            splineObject->drag_origin_local = localPoint;
            splineObject->drag_origin_valid = true;
        }
        const Vec3 movedWorld(gizmo[12], gizmo[13], gizmo[14]);
        const Vec3 movedLocal = objectTransform.inverse().transform_point(movedWorld);
        Vec3 delta = movedLocal - localPoint;
        // A 2D authoring spline must never accumulate depth from the screen
        // plane or a world-axis gizmo handle. Constrain in object-local space,
        // so the lock remains correct when the spline object is transformed.
        if (splineObject->plane == SplinePlane::XY) delta.z = 0.0f;
        else if (splineObject->plane == SplinePlane::XZ) delta.y = 0.0f;
        else if (splineObject->plane == SplinePlane::YZ) delta.x = 0.0f;
        // Free: unconstrained 3D authoring, no axis is zeroed.
        bool activeIncluded = false;
        for (const int index : splineObject->selected_points) {
            if (index < 0 || index >= static_cast<int>(splineObject->spline.points.size())) continue;
            splineObject->spline.points[static_cast<size_t>(index)].position += delta;
            flattenToPlane(splineObject->spline.points[static_cast<size_t>(index)].position,
                           splineObject->plane);
            activeIncluded |= index == splineObject->selected_point;
        }
        // Older selections and scripted state can contain only selected_point.
        // Keep the active anchor authoritative without applying its delta twice.
        if (!activeIncluded) {
            splineObject->spline.points[static_cast<size_t>(splineObject->selected_point)].position += delta;
            flattenToPlane(
                splineObject->spline.points[static_cast<size_t>(splineObject->selected_point)].position,
                splineObject->plane);
        }
        splineObject->point_drag_dirty = true;
        ProjectManager::getInstance().markModified();
    } else if (splineObject->point_drag_dirty) {
        splineObject->spline.calculateAutoTangents();
        splineObject->point_drag_dirty = false;
        splineObject->drag_origin_valid = false;
    }
    return true;
}

} // namespace MeshEdit
