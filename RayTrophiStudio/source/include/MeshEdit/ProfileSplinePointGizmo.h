#pragma once

struct UIContext;

namespace MeshEdit {
// Draws and applies the transform gizmo for the selected spline control point.
// Returns true while spline point editing owns the gizmo for this frame.
bool drawProfileSplinePointGizmo(UIContext& ctx);
}
