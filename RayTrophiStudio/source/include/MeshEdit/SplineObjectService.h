#pragma once

#include "MeshEdit/SplineObject.h"
#include "MeshEdit/SplinePrimitive.h"
#include <memory>
#include <string>

struct UIContext;
class SceneHistory;

namespace MeshEdit {

std::shared_ptr<SplineObject> addSplinePrimitiveObject(
    UIContext& ctx, SceneHistory& history, SplinePrimitiveType type,
    const std::string& requested_name, SplinePlane plane = SplinePlane::XY);

// Creates an EMPTY curve armed for surface drawing: no points, SplinePlane::Free
// and the Draw tool already active, so the next click in the viewport lays the
// first point. The plane is Free because a route follows the ground in three
// axes; the planar primitives exist for profiles, and Faz 3.6 already makes
// Sweep/Revolve refuse a Free profile with a non_planar_profile diagnostic
// rather than silently reinterpreting it.
std::shared_ptr<SplineObject> addSurfaceDrawSplineObject(
    UIContext& ctx, SceneHistory& history, const std::string& requested_name);

} // namespace MeshEdit
