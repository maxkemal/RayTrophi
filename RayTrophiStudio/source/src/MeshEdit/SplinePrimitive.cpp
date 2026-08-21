#include "MeshEdit/SplinePrimitive.h"

#include <algorithm>
#include <cmath>

namespace MeshEdit {
namespace {

constexpr float kCircleHandle = 0.5522847498f;

void setEditableHandle(BezierControlPoint& point, const Vec3& tangent, float length) {
    point.autoTangent = false;
    point.handleMode = BezierControlPoint::HandleMode::Mirrored;
    point.tangentOut = tangent * length;
    point.tangentIn = point.tangentOut * -1.0f;
}

void addPointWithHandle(BezierSpline& spline, const Vec3& position,
                        const Vec3& tangent, float length) {
    spline.points.emplace_back(position);
    setEditableHandle(spline.points.back(), tangent, length);
}

} // namespace

BezierSpline makeSplinePrimitive(SplinePrimitiveType type,
                                 const SplinePrimitiveSettings& settings) {
    BezierSpline spline;
    const float safeRadius = std::max(0.0001f, std::abs(settings.radius));

    switch (type) {
    case SplinePrimitiveType::Circle: {
        spline.isClosed = true;
        const float r = safeRadius;
        addPointWithHandle(spline, settings.center + Vec3(r, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), r * kCircleHandle);
        addPointWithHandle(spline, settings.center + Vec3(0.0f, r, 0.0f), Vec3(-1.0f, 0.0f, 0.0f), r * kCircleHandle);
        addPointWithHandle(spline, settings.center + Vec3(-r, 0.0f, 0.0f), Vec3(0.0f, -1.0f, 0.0f), r * kCircleHandle);
        addPointWithHandle(spline, settings.center + Vec3(0.0f, -r, 0.0f), Vec3(1.0f, 0.0f, 0.0f), r * kCircleHandle);
        break;
    }
    case SplinePrimitiveType::Rectangle: {
        spline.isClosed = true;
        const float halfWidth = std::max(0.0001f, std::abs(settings.width) * 0.5f);
        const float halfHeight = std::max(0.0001f, std::abs(settings.height) * 0.5f);
        addPointWithHandle(spline, settings.center + Vec3(-halfWidth, -halfHeight, 0.0f), Vec3(0.0f), 0.0f);
        addPointWithHandle(spline, settings.center + Vec3( halfWidth, -halfHeight, 0.0f), Vec3(0.0f), 0.0f);
        addPointWithHandle(spline, settings.center + Vec3( halfWidth,  halfHeight, 0.0f), Vec3(0.0f), 0.0f);
        addPointWithHandle(spline, settings.center + Vec3(-halfWidth,  halfHeight, 0.0f), Vec3(0.0f), 0.0f);
        break;
    }
    case SplinePrimitiveType::OpenLine: {
        spline.isClosed = false;
        const float halfWidth = std::max(0.0001f, std::abs(settings.width) * 0.5f);
        addPointWithHandle(spline, settings.center + Vec3(-halfWidth, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), halfWidth / 3.0f);
        addPointWithHandle(spline, settings.center + Vec3( halfWidth, 0.0f, 0.0f), Vec3(1.0f, 0.0f, 0.0f), halfWidth / 3.0f);
        break;
    }
    case SplinePrimitiveType::OpenArc: {
        spline.isClosed = false;
        const int count = std::max(2, settings.arc_points);
        const float radius = safeRadius;
        const float range = settings.end_angle - settings.start_angle;
        for (int i = 0; i < count; ++i) {
            const float u = static_cast<float>(i) / static_cast<float>(count - 1);
            const float angle = settings.start_angle + range * u;
            const Vec3 radial(std::cos(angle), std::sin(angle), 0.0f);
            const Vec3 tangent(-std::sin(angle), std::cos(angle), 0.0f);
            addPointWithHandle(spline, settings.center + radial * radius, tangent,
                               std::abs(range) * radius / static_cast<float>(count - 1) / 3.0f);
        }
        break;
    }
    }
    return spline;
}

} // namespace MeshEdit
