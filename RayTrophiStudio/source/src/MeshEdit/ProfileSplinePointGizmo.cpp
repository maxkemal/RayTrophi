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

bool drawProfileSplinePointGizmo(UIContext& ctx) {
    auto splineObject = ctx.selection.selected.spline_object;
    if (!splineObject || !splineObject->edit_mode || splineObject->selected_point < 0 ||
        splineObject->selected_point >= static_cast<int>(splineObject->spline.points.size()) ||
        !ctx.scene.camera) {
        return false;
    }

    ImGuiIO& io = ImGui::GetIO();
    const Camera& camera = *ctx.scene.camera;
    // The standalone authoring gizmo has no dependency on SceneUI's private
    // viewport state; camera projection is the stable fallback here.
    const bool ortho = camera.orthographic;
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

    ImGuizmo::SetOrthographic(ortho);
    ImGuizmo::BeginFrame();
    ImGuizmo::SetRect(0.0f, 0.0f, io.DisplaySize.x, io.DisplaySize.y);
    ImGuizmo::SetGizmoSizeClipSpace(0.09f);
    ImGuizmo::Manipulate(view, projection, ImGuizmo::TRANSLATE,
                         ImGuizmo::WORLD, gizmo);

    if (ImGuizmo::IsUsing() && splineObject->transform) {
        const Vec3 movedWorld(gizmo[12], gizmo[13], gizmo[14]);
        splineObject->spline.points[splineObject->selected_point].position =
            objectTransform.inverse().transform_point(movedWorld);
        splineObject->point_drag_dirty = true;
        ProjectManager::getInstance().markModified();
    } else if (splineObject->point_drag_dirty) {
        splineObject->spline.calculateAutoTangents();
        splineObject->point_drag_dirty = false;
    }
    return true;
}

} // namespace MeshEdit
