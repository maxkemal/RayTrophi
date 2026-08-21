#include "Api/RtApi.h"
#include "Api/RtApiInternal.h"

#include "MeshEdit/FlatMeshGeometryCommand.h"
#include "MeshEdit/FlatMeshValidator.h"
#include "GeometryNodesV2.h"
#include "TriangleMesh.h"
#include "SceneHistory.h"

#include <cstring>
#include <memory>

namespace rtapi {

Result setMeshPositionsUndoable(const std::string& name, const float* data,
                                size_t vertex_count) {
    if (!data) return Result::fail("positions data is null");
    if (!g_ctx || !g_history) return Result::fail("rtapi is not bound to SceneHistory");
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    TriangleMesh* mesh = nullptr;
    for (const auto& object : g_ctx->scene.world.objects) {
        auto candidate = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (candidate && candidate->nodeName == name) { mesh = candidate.get(); break; }
    }
    if (!mesh || !mesh->geometry) return Result::fail("object not found: " + name);
    if (mesh->geometry->get_vertex_count() != vertex_count) {
        return Result::fail("vertex count mismatch for '" + name + "'");
    }

    auto before = std::make_shared<DNA::GeometryDetail>(*mesh->geometry);
    auto after = std::make_shared<DNA::GeometryDetail>(*mesh->geometry);
    Vec3* positions = after->get_attribute_data_mut<Vec3>("P_orig");
    if (!positions) return Result::fail("object has no P_orig buffer: " + name);
    std::memcpy(positions, data, vertex_count * sizeof(Vec3));

    TriangleMesh candidate;
    candidate.nodeName = mesh->nodeName;
    candidate.transform = mesh->transform;
    candidate.geometry = std::move(after);
    GeometryNodesV2::rebakeFromOrig(candidate);
    const auto validation = MeshEdit::validateFlatMesh(candidate);
    if (!validation.valid) return Result::fail("position commit failed flat mesh validation");

    *mesh->geometry = *candidate.geometry;
    scheduleSceneMutationRebuilds(*g_ctx, true);
    g_history->record(std::make_unique<FlatMeshGeometryCommand>(
        name, std::move(before),
        std::make_shared<DNA::GeometryDetail>(*mesh->geometry),
        "Edit Mesh Positions " + name));
    return Result::success();
}

} // namespace rtapi
