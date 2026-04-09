#include "Viewport/ViewportSceneSync.h"

#include "Backend/IBackend.h"

#include <unordered_set>

std::vector<std::shared_ptr<Triangle>> Viewport::collectMeshTrianglesForObject(
    const std::map<std::string, std::vector<std::pair<int, std::shared_ptr<Triangle>>>>& meshCache,
    const std::string& objectName) {
    std::vector<std::shared_ptr<Triangle>> triangles;
    const auto meshIt = meshCache.find(objectName);
    if (meshIt == meshCache.end()) {
        return triangles;
    }

    triangles.reserve(meshIt->second.size());
    std::unordered_set<const Triangle*> seenTriangles;
    seenTriangles.reserve(meshIt->second.size());
    for (const auto& entry : meshIt->second) {
        if (entry.second && seenTriangles.insert(entry.second.get()).second) {
            triangles.push_back(entry.second);
        }
    }
    return triangles;
}

Viewport::InteractiveMeshSyncResult Viewport::syncInteractiveMeshEdit(
    Backend::IBackend* backend,
    const MeshEditSyncRequest& request,
    const std::vector<std::shared_ptr<Triangle>>& triangles) {
    if (!backend || request.objectName.empty()) {
        return InteractiveMeshSyncResult::NotSupported;
    }

    if (request.topologyChanged || !request.interactiveViewportActive) {
        return InteractiveMeshSyncResult::RequiresDeferredRebuild;
    }

    if (triangles.empty()) {
        return InteractiveMeshSyncResult::RequiresDeferredRebuild;
    }

    if (backend->updateInteractiveMesh(request.objectName, triangles)) {
        return InteractiveMeshSyncResult::UpdatedIncrementally;
    }

    return InteractiveMeshSyncResult::RequiresDeferredRebuild;
}
