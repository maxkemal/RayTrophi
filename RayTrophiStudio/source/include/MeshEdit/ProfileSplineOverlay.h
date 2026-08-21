#pragma once

struct UIContext;

namespace MeshEdit {

// Viewport-first authoring feedback for the selected mesh-free spline source.
// This intentionally draws only the source curve/control cage; sweep/revolve
// evaluation remains a modifier/Geometry Node responsibility.
void drawProfileSplineOverlay(UIContext& ctx);

// Called before the regular CPU/GPU mesh picker. Returns true when the click
// belongs to a mesh-free spline source, preventing the mesh picker from
// clearing that selection after the overlay has handled it.
bool pickProfileSpline(UIContext& ctx);

} // namespace MeshEdit
