#pragma once

#include "MeshEdit/MeshTool.h"

#include <memory>
#include <string>

struct UIContext;
class SceneHistory;
class TriangleMesh;

namespace MeshEdit {

class HalfEdgeMesh;
class FlatMeshEditService {
public:
    // Publishes an already validated topology through the canonical flat path,
    // records one SceneHistory command and schedules renderer/BVH refresh.
    static MeshOperationReport publishTopology(UIContext& ctx, SceneHistory& history,
                                               const std::string& object_name,
                                               const HalfEdgeMesh& topology,
                                               const std::string& operation_id,
                                               const std::string& undo_group);
};

} // namespace MeshEdit
