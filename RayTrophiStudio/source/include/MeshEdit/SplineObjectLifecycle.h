#pragma once

#include <memory>

struct UIContext;
class SceneHistory;

namespace MeshEdit {
class SplineObject;

// Removes a mesh-free spline source through an undoable scene command.
// UI, scripting and IPC object deletion share this operation.
bool deleteSplineObject(UIContext& ctx, SceneHistory& history,
                        const std::shared_ptr<SplineObject>& spline);

} // namespace MeshEdit
