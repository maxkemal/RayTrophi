#pragma once

#include "BezierSpline.h"
#include <cstdint>

namespace MeshEdit {

enum class SplinePrimitiveType : uint8_t {
    Circle,
    Rectangle,
    OpenLine,
    OpenArc,
    // No points at all: the starting state for surface drawing. The other
    // primitives are planar because a PROFILE is planar; a route is not, which
    // is why an empty curve is created with SplinePlane::Free.
    Empty,
};

struct SplinePrimitiveSettings {
    Vec3 center = Vec3(0.0f);
    float radius = 1.0f;
    float width = 2.0f; // Rectangle width; also Open Line total length.
    float height = 2.0f;
    float start_angle = 0.0f;
    float end_angle = M_PI;
    int arc_points = 5;
};

// Creates editable control points and handles. The returned spline remains the
// authoring source; sweep only samples it and never replaces it with a mesh.
BezierSpline makeSplinePrimitive(SplinePrimitiveType type,
                                 const SplinePrimitiveSettings& settings = {});

} // namespace MeshEdit
