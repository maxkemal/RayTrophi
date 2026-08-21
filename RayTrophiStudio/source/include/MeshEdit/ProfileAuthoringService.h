#pragma once

#include "MeshEdit/MeshTool.h"
#include <memory>
#include <string>

struct UIContext;
class SceneHistory;
namespace DNA { class GeometryDetail; }

namespace MeshEdit {

struct ProfilePublishResult {
    MeshOperationReport report;
    std::string object_name;
};

// Publishes generated flat geometry as one TriangleMesh through AddObjectCommand.
// This is the shared scene mutation used by IPC, Python and the future viewport tool.
ProfilePublishResult publishGeneratedProfile(UIContext& ctx, SceneHistory& history,
                                             std::shared_ptr<DNA::GeometryDetail> geometry,
                                             const std::string& requested_name,
                                             const std::string& operation_id);

} // namespace MeshEdit
