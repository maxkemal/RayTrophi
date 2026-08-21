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
    const std::string& requested_name, SplinePlane plane = SplinePlane::XZ);

} // namespace MeshEdit
