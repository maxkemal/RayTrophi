/*
 * MeshEdit/FlatMeshUndoSnapshot.h
 *
 * GeometryDetail snapshot transaction for canonical flat mesh edits.  It is
 * intentionally independent from UI and SceneHistory so scripting, IPC and
 * addons can share the same reversible data primitive.
 */
#pragma once

#include "MeshEdit/MeshTool.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

class TriangleMesh;

namespace DNA { class GeometryDetail; }

namespace MeshEdit {

class FlatMeshUndoSnapshot {
public:
    bool begin(TriangleMesh& mesh, std::string operation_id,
               std::string undo_group = {});
    bool captureAfter();
    bool undo();
    bool redo();
    void discard();

    bool active() const noexcept { return mesh_ != nullptr; }
    bool hasAfter() const noexcept { return after_ != nullptr; }
    size_t estimatedBytes() const noexcept { return estimated_bytes_; }
    const MeshOperationReport& report() const noexcept { return report_; }

private:
    TriangleMesh* mesh_ = nullptr;
    std::shared_ptr<DNA::GeometryDetail> before_;
    std::shared_ptr<DNA::GeometryDetail> after_;
    size_t estimated_bytes_ = 0;
    MeshOperationReport report_;

    bool restore(const std::shared_ptr<DNA::GeometryDetail>& snapshot,
                 const char* phase);
    void updateEstimate();
};

bool runFlatMeshUndoSnapshotSelfTest(std::string& report);

} // namespace MeshEdit
