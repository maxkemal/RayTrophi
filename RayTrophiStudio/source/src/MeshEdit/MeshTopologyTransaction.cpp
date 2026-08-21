#include "MeshEdit/MeshTopologyTransaction.h"

#include <utility>

namespace MeshEdit {

bool MeshTopologyTransaction::begin(const HalfEdgeMesh& source,
                                    std::string operation_id,
                                    std::string undo_group) {
    if (active_) return false;
    before_ = source;
    working_ = source;
    report_ = {};
    report_.operation_id = std::move(operation_id);
    report_.undo_group = std::move(undo_group);
    active_ = true;
    return true;
}

bool MeshTopologyTransaction::checkActive(const char* operation) {
    if (active_) return true;
    report_.addError("transaction_inactive", std::string(operation) + " requires an active transaction");
    return false;
}

bool MeshTopologyTransaction::checkMutation(bool changed, const char* operation) {
    if (!checkActive(operation)) return false;
    if (!changed) {
        report_.addError("operation_rejected", std::string(operation) + " was rejected by topology preconditions");
        active_ = false;
        return false;
    }
    std::string error;
    if (!working_.validate(&error)) {
        report_.addError("topology_invalid", std::string(operation) + " produced invalid topology: " + error);
        active_ = false;
        return false;
    }
    return true;
}

HEIndex MeshTopologyTransaction::splitEdge(HEIndex edge, float t) {
    if (!checkActive("split_edge")) return kHEInvalid;
    const HEIndex result = working_.splitEdge(edge, t);
    return checkMutation(result != kHEInvalid, "split_edge") ? result : kHEInvalid;
}

HEIndex MeshTopologyTransaction::splitFace(HEIndex face, HEIndex vertex_a, HEIndex vertex_b) {
    if (!checkActive("split_face")) return kHEInvalid;
    const HEIndex result = working_.splitFace(face, vertex_a, vertex_b);
    return checkMutation(result != kHEInvalid, "split_face") ? result : kHEInvalid;
}

bool MeshTopologyTransaction::flipEdge(HEIndex edge) {
    if (!checkActive("flip_edge")) return false;
    return checkMutation(working_.flipEdge(edge), "flip_edge");
}

HEIndex MeshTopologyTransaction::collapseEdge(HEIndex edge, float t) {
    if (!checkActive("collapse_edge")) return kHEInvalid;
    const HEIndex result = working_.collapseEdge(edge, t);
    return checkMutation(result != kHEInvalid, "collapse_edge") ? result : kHEInvalid;
}

HEIndex MeshTopologyTransaction::dissolveEdge(HEIndex edge) {
    if (!checkActive("dissolve_edge")) return kHEInvalid;
    const HEIndex result = working_.dissolveEdge(edge);
    return checkMutation(result != kHEInvalid, "dissolve_edge") ? result : kHEInvalid;
}

HEIndex MeshTopologyTransaction::extrudeFace(HEIndex face, const Vec3& offset,
                                             std::vector<HEIndex>* side_faces) {
    if (!checkActive("extrude_face")) return kHEInvalid;
    const HEIndex result = working_.extrudeFace(face, offset, side_faces);
    return checkMutation(result != kHEInvalid, "extrude_face") ? result : kHEInvalid;
}

HEIndex MeshTopologyTransaction::insetFace(HEIndex face, float t,
                                           std::vector<HEIndex>* side_faces) {
    if (!checkActive("inset_face")) return kHEInvalid;
    const HEIndex result = working_.insetFace(face, t, side_faces);
    return checkMutation(result != kHEInvalid, "inset_face") ? result : kHEInvalid;
}

bool MeshTopologyTransaction::loopCut(HEIndex edge, float t,
                                      HalfEdgeMesh::LoopCutResult* result) {
    if (!checkActive("loop_cut")) return false;
    return checkMutation(working_.loopCut(edge, t, result), "loop_cut");
}

void MeshTopologyTransaction::updateCounts() {
    const auto delta = [](size_t a, size_t b) -> uint64_t {
        return static_cast<uint64_t>(a > b ? a - b : b - a);
    };
    report_.changed.vertices_changed = delta(working_.liveVertexCount(), before_.liveVertexCount());
    report_.changed.edges_changed = delta(working_.liveEdgeCount(), before_.liveEdgeCount());
    report_.changed.faces_changed = delta(working_.liveFaceCount(), before_.liveFaceCount());
    std::vector<std::array<HEIndex, 3>> before_triangles;
    std::vector<std::array<HEIndex, 3>> after_triangles;
    before_.triangulate(before_triangles);
    working_.triangulate(after_triangles);
    report_.changed.triangles_changed = delta(after_triangles.size(), before_triangles.size());
}

MeshOperationReport MeshTopologyTransaction::commit(HalfEdgeMesh& target) {
    if (!checkActive("commit")) return report_;
    std::string error;
    if (!working_.validate(&error)) {
        report_.addError("topology_invalid", "commit rejected: " + error);
        active_ = false;
        return report_;
    }
    updateCounts();
    target = std::move(working_);
    report_.ok = true;
    active_ = false;
    return report_;
}

MeshOperationReport MeshTopologyTransaction::cancel() {
    if (!active_) {
        report_.addError("transaction_inactive", "cancel requires an active transaction");
        return report_;
    }
    report_.addWarning("cancelled", "topology transaction was cancelled before publish");
    active_ = false;
    return report_;
}

bool runMeshTopologyTransactionSelfTest(std::string& report) {
    HalfEdgeMesh source;
    const std::vector<Vec3> positions = {
        Vec3(-1.0f, -1.0f, 0.0f), Vec3(1.0f, -1.0f, 0.0f),
        Vec3(1.0f, 1.0f, 0.0f), Vec3(-1.0f, 1.0f, 0.0f)
    };
    const std::vector<std::vector<int>> polygons = {{0, 1, 2, 3}};
    HalfEdgeBuildResult build;
    if (!source.buildFromPolygons(positions, polygons, &build)) {
        report = "source build failed: " + build.message;
        return false;
    }

    MeshTopologyTransaction tx;
    if (!tx.begin(source, "self_test.extrude", "self_test")) {
        report = "begin failed";
        return false;
    }
    std::vector<HEIndex> side_faces;
    if (tx.extrudeFace(0, Vec3(0.0f, 0.0f, 1.0f), &side_faces) == kHEInvalid) {
        report = "extrude rejected";
        return false;
    }
    HalfEdgeMesh committed;
    const MeshOperationReport result = tx.commit(committed);
    if (!result.ok || !committed.validate(&report)) {
        if (report.empty()) report = "commit validation failed";
        return false;
    }
    report = "ok: vertices=" + std::to_string(committed.liveVertexCount()) +
             ", faces=" + std::to_string(committed.liveFaceCount());
    return true;
}

} // namespace MeshEdit
