#pragma once

struct UIContext;

namespace MeshEdit {
// Draws the gizmo at the active point and applies its translation delta to all
// selected spline control points.
// Returns true while spline point editing owns the gizmo for this frame.
bool drawProfileSplinePointGizmo(UIContext& ctx);
}
