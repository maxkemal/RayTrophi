#pragma once

#include "Backend/IBackend.h"
#include <cstddef>
#include <utility>

namespace Backend {

class IViewportBackend : public IBackend {
public:
    virtual ~IViewportBackend() = default;

    virtual bool updateRasterMeshFromTriangles(
        const std::string& nodeName,
        const std::vector<std::shared_ptr<class Triangle>>& triangles) = 0;

    virtual bool patchRasterMeshTriangles(
        const std::string& nodeName,
        const std::vector<size_t>& dirtyIndices,
        const std::vector<std::pair<int, std::shared_ptr<class Triangle>>>& meshEntries) = 0;

    // Refit a flat (direct SoA) mesh's raster vertices straight from its DNA SoA (no per-face
    // Triangle facades). Returns false if unsupported / mesh not found so the caller can fall back.
    virtual bool updateRasterMeshFromMeshSoA(const std::string& /*nodeName*/,
                                             const class TriangleMesh* /*mesh*/) { return false; }
    virtual bool cloneRasterObjectByNodeName(
        const std::string& sourceNodeName,
        const std::string& newNodeName,
        const Matrix4x4& transform) = 0;

    virtual void buildRasterGeometry(const std::vector<std::shared_ptr<Hittable>>& objects) = 0;
    virtual void syncRasterInstanceTransforms(const std::vector<std::shared_ptr<Hittable>>& objects) = 0;
    virtual void syncRasterSkinnedVertices(const std::vector<std::shared_ptr<Hittable>>& objects,
                                           const std::vector<Matrix4x4>& boneMatrices) = 0;

    // Returns true if raster geometry is already built and matches the given
    // scene generation counter.  Callers use this to skip redundant rebuilds
    // during viewport-mode transitions when geometry hasn't changed.
    virtual bool hasValidRasterCache(uint64_t sceneGeometryGeneration) const = 0;
};

} // namespace Backend
