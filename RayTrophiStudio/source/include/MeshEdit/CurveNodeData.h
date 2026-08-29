#pragma once

#include "MeshEdit/SplineObject.h"

#include <cstddef>
#include <string>

namespace MeshEdit {

// Immutable-by-convention value carried by Geometry Nodes Curve sockets.
// The editable SplineObject remains the scene authority; source nodes take a
// fresh snapshot on every graph evaluation so edits and future keyframes are
// visible without baking the curve to triangles early.
struct CurveNodeData {
    BezierSpline spline;
    SplinePlane plane = SplinePlane::XY;
    Matrix4x4 local_to_world = Matrix4x4::identity();
    Vec3 object_scale = Vec3(1.0f);
    Vec3 pivot_offset = Vec3(0.0f);
    std::string source_name;
    std::size_t source_signature = 0;
};

} // namespace MeshEdit
