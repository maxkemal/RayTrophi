#include "MeshEdit/MeshOperation.h"

#include <algorithm>
#include <utility>

namespace MeshEdit {

void MeshOperationPlan::addError(std::string code, std::string message) {
    diagnostics.push_back({std::move(code), std::move(message), false});
}

void MeshOperationPlan::addWarning(std::string code, std::string message) {
    diagnostics.push_back({std::move(code), std::move(message), true});
}

const char* backendName(MeshOperationBackend backend) {
    switch (backend) {
    case MeshOperationBackend::CPU: return "cpu";
    case MeshOperationBackend::GPU: return "gpu";
    case MeshOperationBackend::Auto: return "auto";
    }
    return "auto";
}

MeshOperationPlan planMeshOperation(const MeshOperationRequest& request,
                                    const MeshToolDescriptor* tool,
                                    uint64_t vertex_count,
                                    uint64_t edge_count,
                                    uint64_t face_count,
                                    uint64_t revision) {
    MeshOperationPlan plan;
    plan.operation_id = request.operation_id;
    plan.object_name = request.object_name;
    plan.backend = backendName(request.backend);
    plan.preview = request.preview;
    plan.commit = request.commit;
    plan.expected_revision = request.expected_revision;

    if (!tool) {
        plan.addError("unknown_tool", "mesh operation is not registered");
        return plan;
    }
    if (request.object_name.empty()) {
        plan.addError("missing_object", "object_name is required");
    }
    if (!request.preview && !request.commit) {
        plan.addError("missing_phase", "set preview or commit before executing an operation");
    }
    if (request.preview && request.commit) {
        plan.addError("ambiguous_phase", "preview and commit cannot be requested together");
    }
    if (tool->availability != MeshToolAvailability::Implemented && request.commit) {
        plan.addError("tool_not_executable", "planned or preview-only mesh tools cannot be committed");
    }
    if (request.selection_domain != tool->selection_domain &&
        tool->selection_domain != MeshSelectionDomain::Object) {
        plan.addError("selection_domain_mismatch", "selection domain does not match the mesh tool");
    }

    const uint64_t limit = request.selection_domain == MeshSelectionDomain::Vertex
        ? vertex_count : (request.selection_domain == MeshSelectionDomain::Face ||
                          request.selection_domain == MeshSelectionDomain::Polygon)
            ? face_count : edge_count;
    for (const uint32_t id : request.selection_ids) {
        if (static_cast<uint64_t>(id) >= limit) {
            plan.addError("selection_out_of_range", "one or more selected element ids are outside the flat mesh");
            break;
        }
    }

    const auto& caps = tool->capabilities;
    if (request.backend == MeshOperationBackend::GPU && !caps.gpu_preview && !caps.gpu_commit) {
        plan.addError("gpu_unavailable", "this tool has no GPU execution path");
    }
    if (request.backend == MeshOperationBackend::CPU && !caps.cpu) {
        plan.addError("cpu_unavailable", "this tool has no CPU execution path");
    }
    if (request.backend == MeshOperationBackend::Auto && !caps.cpu && !caps.gpu_preview && !caps.gpu_commit) {
        plan.addError("no_backend", "this tool has no available execution backend");
    }
    if (request.backend == MeshOperationBackend::GPU && caps.gpu_preview && !caps.gpu_commit && request.commit) {
        plan.addWarning("cpu_fallback", "GPU is preview-only; commit will require the CPU fallback");
        plan.requires_cpu_fallback = true;
    }
    if (request.expected_revision != 0 && revision != 0 && request.expected_revision != revision) {
        plan.addError("stale_revision", "mesh revision changed since the operation was prepared");
    }

    plan.undoable = caps.undoable && request.commit;
    plan.ok = std::none_of(plan.diagnostics.begin(), plan.diagnostics.end(),
                           [](const MeshOperationDiagnostic& d) { return !d.warning; });
    return plan;
}

} // namespace MeshEdit
