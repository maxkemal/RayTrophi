/*
 * MeshEdit/MeshOperation.h
 *
 * UI-independent preflight contract for mesh operations.  The planner is
 * deliberately side-effect free: topology/attribute mutation will be added
 * behind this contract, while UI, scripts, IPC and agents already receive the
 * same validation and backend-selection semantics.
 */
#pragma once

#include "MeshEdit/MeshTool.h"

#include <cstdint>
#include <string>
#include <vector>

namespace MeshEdit {

enum class MeshOperationBackend : uint8_t { Auto, CPU, GPU };

struct MeshOperationRequest {
    std::string operation_id;
    std::string object_name;
    MeshSelectionDomain selection_domain = MeshSelectionDomain::Object;
    std::vector<uint32_t> selection_ids;
    MeshOperationBackend backend = MeshOperationBackend::Auto;
    bool preview = false;
    bool commit = false;
    uint64_t expected_revision = 0;
};

struct MeshOperationPlan {
    bool ok = false;
    std::string operation_id;
    std::string object_name;
    std::string backend;
    bool preview = false;
    bool commit = false;
    bool undoable = false;
    bool requires_cpu_fallback = false;
    uint64_t expected_revision = 0;
    std::vector<MeshOperationDiagnostic> diagnostics;

    void addError(std::string code, std::string message);
    void addWarning(std::string code, std::string message);
};

MeshOperationPlan planMeshOperation(const MeshOperationRequest& request,
                                    const MeshToolDescriptor* tool,
                                    uint64_t vertex_count,
                                    uint64_t edge_count,
                                    uint64_t face_count,
                                    uint64_t revision = 0);

const char* backendName(MeshOperationBackend backend);

} // namespace MeshEdit
