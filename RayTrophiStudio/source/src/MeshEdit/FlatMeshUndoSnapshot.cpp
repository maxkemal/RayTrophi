#include "MeshEdit/FlatMeshUndoSnapshot.h"

#include "MeshEdit/FlatMeshPublisher.h"
#include "MeshEdit/FlatMeshValidator.h"
#include "TriangleMesh.h"

#include <memory>
#include <utility>

namespace MeshEdit {

bool FlatMeshUndoSnapshot::begin(TriangleMesh& mesh, std::string operation_id,
                                 std::string undo_group) {
    if (mesh_ || !mesh.geometry) return false;
    mesh_ = &mesh;
    before_ = std::make_shared<DNA::GeometryDetail>(*mesh.geometry);
    report_ = {};
    report_.operation_id = std::move(operation_id);
    report_.undo_group = std::move(undo_group);
    updateEstimate();
    return true;
}

bool FlatMeshUndoSnapshot::captureAfter() {
    if (!mesh_ || !mesh_->geometry) {
        report_.addError("snapshot_inactive", "cannot capture after-state without an active mesh");
        return false;
    }
    after_ = std::make_shared<DNA::GeometryDetail>(*mesh_->geometry);
    updateEstimate();
    report_.ok = true;
    return true;
}

void FlatMeshUndoSnapshot::updateEstimate() {
    const auto bytes = [](const std::shared_ptr<DNA::GeometryDetail>& geometry) -> size_t {
        if (!geometry) return 0;
        return geometry->get_vertex_count() * (sizeof(Vec3) * 4 + sizeof(Vec2) + sizeof(uint16_t)) +
               geometry->indices.size() * sizeof(uint32_t);
    };
    estimated_bytes_ = bytes(before_) + bytes(after_);
}

bool FlatMeshUndoSnapshot::restore(const std::shared_ptr<DNA::GeometryDetail>& snapshot,
                                   const char* phase) {
    if (!mesh_ || !snapshot || !mesh_->geometry) {
        report_.addError("snapshot_inactive", std::string(phase) + " has no valid snapshot");
        return false;
    }
    *mesh_->geometry = *snapshot;
    return true;
}

bool FlatMeshUndoSnapshot::undo() {
    return restore(before_, "undo");
}

bool FlatMeshUndoSnapshot::redo() {
    return restore(after_, "redo");
}

void FlatMeshUndoSnapshot::discard() {
    mesh_ = nullptr;
    before_.reset();
    after_.reset();
    estimated_bytes_ = 0;
    report_ = {};
}

bool runFlatMeshUndoSnapshotSelfTest(std::string& report) {
    HalfEdgeMesh topology;
    if (!topology.buildFromPolygons({
            Vec3(-1.0f, -1.0f, 0.0f), Vec3(1.0f, -1.0f, 0.0f),
            Vec3(1.0f, 1.0f, 0.0f), Vec3(-1.0f, 1.0f, 0.0f)}, {{0, 1, 2, 3}})) {
        report = "topology build failed";
        return false;
    }
    TriangleMesh mesh;
    if (!publishHalfEdgeMeshToFlat(mesh, topology).ok) {
        report = "initial publish failed";
        return false;
    }
    const size_t original_triangles = mesh.num_triangles();
    FlatMeshUndoSnapshot snapshot;
    if (!snapshot.begin(mesh, "self_test.undo", "self_test")) {
        report = "snapshot begin failed";
        return false;
    }
    topology.vertices[0].position.z = 1.0f;
    if (!publishHalfEdgeMeshToFlat(mesh, topology).ok || !snapshot.captureAfter()) {
        report = "edited publish/capture failed";
        return false;
    }
    if (!snapshot.undo() || mesh.num_triangles() != original_triangles ||
        !validateFlatMesh(mesh).valid || !snapshot.redo() || !validateFlatMesh(mesh).valid) {
        report = "undo/redo restore failed";
        return false;
    }
    report = "ok: snapshot_bytes=" + std::to_string(snapshot.estimatedBytes());
    return true;
}

} // namespace MeshEdit
