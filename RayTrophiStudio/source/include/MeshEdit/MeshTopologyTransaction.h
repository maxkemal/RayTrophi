/*
 * MeshEdit/MeshTopologyTransaction.h
 *
 * Transaction boundary for the index-based half-edge topology core.  A
 * working copy is edited and validated before it replaces the authoritative
 * edit mesh, so rejected topology never partially leaks into the scene.
 */
#pragma once

#include "MeshEdit/HalfEdgeMesh.h"
#include "MeshEdit/MeshTool.h"

#include <string>
#include <vector>

namespace MeshEdit {

class MeshTopologyTransaction {
public:
    MeshTopologyTransaction() = default;

    bool begin(const HalfEdgeMesh& source, std::string operation_id,
               std::string undo_group = {});
    bool active() const noexcept { return active_; }

    HEIndex splitEdge(HEIndex edge, float t);
    HEIndex splitFace(HEIndex face, HEIndex vertex_a, HEIndex vertex_b);
    bool flipEdge(HEIndex edge);
    HEIndex collapseEdge(HEIndex edge, float t = 0.5f);
    HEIndex dissolveEdge(HEIndex edge);
    HEIndex extrudeFace(HEIndex face, const Vec3& offset,
                        std::vector<HEIndex>* side_faces = nullptr);
    HEIndex insetFace(HEIndex face, float t,
                      std::vector<HEIndex>* side_faces = nullptr);
    bool loopCut(HEIndex edge, float t, HalfEdgeMesh::LoopCutResult* result = nullptr);

    // Validates and publishes the complete working topology atomically.
    MeshOperationReport commit(HalfEdgeMesh& target);
    MeshOperationReport cancel();
    const HalfEdgeMesh& previewMesh() const noexcept { return working_; }

private:
    bool active_ = false;
    HalfEdgeMesh before_;
    HalfEdgeMesh working_;
    MeshOperationReport report_;

    bool checkActive(const char* operation);
    bool checkMutation(bool changed, const char* operation);
    void updateCounts();
};

// Deterministic core smoke test; does not touch scene state or GPU resources.
bool runMeshTopologyTransactionSelfTest(std::string& report);

} // namespace MeshEdit
