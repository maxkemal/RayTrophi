#include "Api/RtApi.h"
#include "Api/RtApiInternal.h"
#include "MeshEdit/FlatMeshValidator.h"
#include "MeshEdit/MeshOperation.h"
#include "TriangleMesh.h"

namespace rtapi {

Result validateMesh(const std::string& name, MeshValidationInfo& out) {
    if (!g_ctx) return notBound();

    TriangleMesh* mesh = nullptr;
    for (const auto& object : g_ctx->scene.world.objects) {
        auto candidate = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (candidate && candidate->nodeName == name) {
            mesh = candidate.get();
            break;
        }
    }
    if (!mesh) return Result::fail("object not found: " + name);

    const MeshEdit::FlatMeshValidation report = MeshEdit::validateFlatMesh(*mesh);
    out.valid = report.valid;
    out.vertex_count = static_cast<size_t>(report.vertex_count);
    out.triangle_count = static_cast<size_t>(report.triangle_count);
    out.non_finite_vertices = static_cast<size_t>(report.non_finite_vertices);
    out.out_of_range_indices = static_cast<size_t>(report.out_of_range_indices);
    out.degenerate_triangles = static_cast<size_t>(report.degenerate_triangles);
    out.non_finite_normals = static_cast<size_t>(report.non_finite_normals);
    return Result::success();
}

Result planMeshOperation(const std::string& object_name,
                         const std::string& operation_id,
                         const std::string& backend,
                         bool preview,
                         bool commit,
                         MeshOperationPlanInfo& out) {
    if (!g_ctx) return notBound();
    TriangleMesh* mesh = nullptr;
    for (const auto& object : g_ctx->scene.world.objects) {
        auto candidate = std::dynamic_pointer_cast<TriangleMesh>(object);
        if (candidate && candidate->nodeName == object_name) { mesh = candidate.get(); break; }
    }
    if (!mesh || !mesh->geometry) return Result::fail("object not found or has no flat geometry: " + object_name);

    MeshEdit::MeshOperationBackend requested = MeshEdit::MeshOperationBackend::Auto;
    if (backend == "cpu") requested = MeshEdit::MeshOperationBackend::CPU;
    else if (backend == "gpu") requested = MeshEdit::MeshOperationBackend::GPU;
    else if (!backend.empty() && backend != "auto") return Result::fail("unknown mesh operation backend: " + backend);

    const auto* tool = MeshEdit::MeshToolRegistry::instance().find(operation_id);
    MeshEdit::MeshOperationRequest request;
    request.operation_id = operation_id;
    request.object_name = object_name;
    request.selection_domain = tool ? tool->selection_domain : MeshEdit::MeshSelectionDomain::Object;
    request.backend = requested;
    request.preview = preview;
    request.commit = commit;
    const auto plan = MeshEdit::planMeshOperation(request, tool,
        static_cast<uint64_t>(mesh->geometry->get_vertex_count()),
        0, static_cast<uint64_t>(mesh->geometry->indices.size() / 3));

    out = {};
    out.ok = plan.ok;
    out.operation_id = plan.operation_id;
    out.object_name = plan.object_name;
    out.backend = plan.backend;
    out.preview = plan.preview;
    out.commit = plan.commit;
    out.undoable = plan.undoable;
    out.requires_cpu_fallback = plan.requires_cpu_fallback;
    out.expected_revision = plan.expected_revision;
    for (const auto& diagnostic : plan.diagnostics) {
        out.diagnostic_codes.push_back(diagnostic.code);
        out.diagnostic_messages.push_back(diagnostic.message);
        out.diagnostic_warnings.push_back(diagnostic.warning);
    }
    return Result::success();
}

} // namespace rtapi
