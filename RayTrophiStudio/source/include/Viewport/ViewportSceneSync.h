#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Triangle;

namespace Backend {
class IBackend;
}

namespace Viewport {

enum class InteractiveMeshSyncResult : uint8_t {
    NotSupported = 0,
    UpdatedIncrementally,
    RequiresDeferredRebuild
};

struct MeshEditSyncRequest {
    std::string objectName;
    bool interactiveViewportActive = false;
    bool topologyChanged = false;
};

std::vector<std::shared_ptr<Triangle>> collectMeshTrianglesForObject(
    const std::map<std::string, std::vector<std::pair<int, std::shared_ptr<Triangle>>>>& meshCache,
    const std::string& objectName);

InteractiveMeshSyncResult syncInteractiveMeshEdit(
    Backend::IBackend* backend,
    const MeshEditSyncRequest& request,
    const std::vector<std::shared_ptr<Triangle>>& triangles);

} // namespace Viewport
