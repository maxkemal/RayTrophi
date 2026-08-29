#include "MeshEdit/SplineObjectService.h"

#include "SceneCommand.h"
#include "Triangle.h"
#include "TriangleMesh.h"
#include "scene_ui.h"

namespace MeshEdit {
namespace {
Vec3 mapPlane(const Vec3& value, SplinePlane plane) {
    switch (plane) {
    case SplinePlane::XZ: return Vec3(value.x, 0.0f, value.y);
    case SplinePlane::YZ: return Vec3(0.0f, value.y, value.x);
    case SplinePlane::XY:
    default: return value; // X = lateral/radius, Y = up, Z = plane normal
    }
}

void mapSplineToPlane(BezierSpline& spline, SplinePlane plane) {
    if (plane == SplinePlane::XY) return;
    for (auto& point : spline.points) {
        point.position = mapPlane(point.position, plane);
        point.tangentIn = mapPlane(point.tangentIn, plane);
        point.tangentOut = mapPlane(point.tangentOut, plane);
    }
}

std::string uniqueName(const UIContext& ctx, const std::string& requested) {
    const std::string base = requested.empty() ? "Spline" : requested;
    std::string candidate = base;
    int suffix = 1;
    auto exists = [&](const std::string& name) {
        for (const auto& object : ctx.scene.world.objects) {
            if (!object) continue;
            if (const auto spline = std::dynamic_pointer_cast<SplineObject>(object)) {
                if (spline->nodeName == name) return true;
            } else if (const auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object)) {
                if (mesh->nodeName == name) return true;
            } else if (const auto triangle = std::dynamic_pointer_cast<Triangle>(object);
                       triangle && triangle->getNodeName() == name) {
                return true;
            }
        }
        return false;
    };
    while (exists(candidate)) candidate = base + "." + std::to_string(suffix++);
    return candidate;
}
}

std::shared_ptr<SplineObject> addSplinePrimitiveObject(
    UIContext& ctx, SceneHistory& history, SplinePrimitiveType type,
    const std::string& requested_name, SplinePlane plane) {
    auto object = std::make_shared<SplineObject>();
    object->nodeName = uniqueName(ctx, requested_name);
    object->plane = plane;
    object->spline = makeSplinePrimitive(type);
    mapSplineToPlane(object->spline, plane);
    auto command = std::make_unique<AddObjectCommand>(object);
    command->execute(ctx);
    history.record(std::move(command));
    const int index = static_cast<int>(ctx.scene.world.objects.size()) - 1;
    ctx.selection.selectObject(object, index, object->nodeName);
    return object;
}

} // namespace MeshEdit
