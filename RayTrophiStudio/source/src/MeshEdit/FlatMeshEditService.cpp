#include "MeshEdit/FlatMeshEditService.h"

#include "MeshEdit/FlatMeshGeometryCommand.h"
#include "MeshEdit/FlatMeshPublisher.h"
#include "MeshEdit/FlatMeshValidator.h"
#include "SceneHistory.h"
#include "SceneCommand.h"
#include "TriangleMesh.h"
#include "scene_ui.h"

#include <memory>

namespace MeshEdit {

MeshOperationReport FlatMeshEditService::publishTopology(
    UIContext& ctx, SceneHistory& history, const std::string& object_name,
    const HalfEdgeMesh& topology, const std::string& operation_id,
    const std::string& undo_group) {
    MeshOperationReport report;
    report.operation_id = operation_id;
    report.undo_group = undo_group;

    TriangleMesh* source = nullptr;
    for (const auto& object : ctx.scene.world.objects) {
        auto mesh = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (mesh && mesh->nodeName == object_name) { source = mesh.get(); break; }
    }
    if (!source || !source->geometry) {
        report.addError("object_not_found", "flat mesh object not found: " + object_name);
        return report;
    }

    auto before = std::make_shared<DNA::GeometryDetail>(*source->geometry);
    TriangleMesh candidate;
    candidate.nodeName = source->nodeName;
    candidate.transform = source->transform;
    candidate.geometry = std::make_shared<DNA::GeometryDetail>(*source->geometry);
    const MeshOperationReport publish = publishHalfEdgeMeshToFlat(candidate, topology, source->geometry.get());
    if (!publish.ok || !candidate.geometry) return publish;
    const FlatMeshValidation validation = validateFlatMesh(candidate);
    if (!validation.valid) {
        report.addError("flat_validation_failed", "published flat mesh failed validation");
        return report;
    }

    *source->geometry = *candidate.geometry;
    scheduleSceneMutationRebuilds(ctx, true);
    history.record(std::make_unique<FlatMeshGeometryCommand>(
        object_name, std::move(before),
        std::make_shared<DNA::GeometryDetail>(*source->geometry),
        operation_id));

    report.ok = true;
    report.changed = publish.changed;
    report.diagnostics = publish.diagnostics;
    return report;
}

} // namespace MeshEdit
