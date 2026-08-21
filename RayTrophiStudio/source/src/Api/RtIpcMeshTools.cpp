#include "RtIpcMeshTools.h"

#include "MeshEdit/MeshTool.h"
#include "MeshEdit/MeshTopologyTransaction.h"
#include "MeshEdit/FlatMeshPublisher.h"
#include "MeshEdit/FlatMeshUndoSnapshot.h"
#include "MeshEdit/ProfileSweepService.h"
#include "MeshEdit/ProfileLoft.h"
#include "MeshEdit/SplineSerialization.h"
#include "MeshEdit/ProfileAuthoringService.h"
#include "Api/RtApiInternal.h"
#include "Api/RtApi.h"

#include <string>
#include <utility>
#include <vector>

namespace {

using nlohmann::json;

const char* workspaceName(MeshEdit::MeshToolWorkspace workspace) {
    switch (workspace) {
    case MeshEdit::MeshToolWorkspace::Edit: return "edit";
    case MeshEdit::MeshToolWorkspace::Profile: return "profile";
    case MeshEdit::MeshToolWorkspace::Curve: return "curve";
    case MeshEdit::MeshToolWorkspace::Surface: return "surface";
    case MeshEdit::MeshToolWorkspace::Boolean: return "boolean";
    case MeshEdit::MeshToolWorkspace::Cleanup: return "cleanup";
    }
    return "unknown";
}

const char* domainName(MeshEdit::MeshSelectionDomain domain) {
    switch (domain) {
    case MeshEdit::MeshSelectionDomain::Object: return "object";
    case MeshEdit::MeshSelectionDomain::Vertex: return "vertex";
    case MeshEdit::MeshSelectionDomain::Edge: return "edge";
    case MeshEdit::MeshSelectionDomain::Face: return "face";
    case MeshEdit::MeshSelectionDomain::Polygon: return "polygon";
    case MeshEdit::MeshSelectionDomain::CurvePoint: return "curve_point";
    case MeshEdit::MeshSelectionDomain::SurfaceControl: return "surface_control";
    }
    return "unknown";
}

const char* availabilityName(MeshEdit::MeshToolAvailability availability) {
    switch (availability) {
    case MeshEdit::MeshToolAvailability::Implemented: return "implemented";
    case MeshEdit::MeshToolAvailability::PreviewOnly: return "preview_only";
    case MeshEdit::MeshToolAvailability::Planned: return "planned";
    case MeshEdit::MeshToolAvailability::Disabled: return "disabled";
    }
    return "unknown";
}

json serializeTool(const MeshEdit::MeshToolDescriptor& tool) {
    const auto& caps = tool.capabilities;
    return {
        {"id", tool.id},
        {"display_name", tool.display_name},
        {"workspace", workspaceName(tool.workspace)},
        {"selection_domain", domainName(tool.selection_domain)},
        {"availability", availabilityName(tool.availability)},
        {"summary", tool.summary},
        {"previewable", tool.previewable},
        {"scriptable", tool.scriptable},
        {"ipc_exposed", tool.ipc_exposed},
        {"capabilities", {
            {"cpu", caps.cpu},
            {"gpu_preview", caps.gpu_preview},
            {"gpu_commit", caps.gpu_commit},
            {"deterministic", caps.deterministic},
            {"supports_cancel", caps.supports_cancel},
            {"undoable", caps.undoable}
        }}
    };
}

bool isKnownWorkspace(const std::string& name) {
    return name == "edit" || name == "profile" || name == "curve" ||
           name == "surface" || name == "boolean" || name == "cleanup";
}

MeshEdit::MeshToolWorkspace parseWorkspace(const std::string& name) {
    if (name == "profile") return MeshEdit::MeshToolWorkspace::Profile;
    if (name == "curve") return MeshEdit::MeshToolWorkspace::Curve;
    if (name == "surface") return MeshEdit::MeshToolWorkspace::Surface;
    if (name == "boolean") return MeshEdit::MeshToolWorkspace::Boolean;
    if (name == "cleanup") return MeshEdit::MeshToolWorkspace::Cleanup;
    return MeshEdit::MeshToolWorkspace::Edit;
}

bool parsePrimitiveType(const std::string& value, MeshEdit::SplinePrimitiveType& out) {
    if (value == "circle") { out = MeshEdit::SplinePrimitiveType::Circle; return true; }
    if (value == "rectangle") { out = MeshEdit::SplinePrimitiveType::Rectangle; return true; }
    if (value == "line" || value == "open_line") { out = MeshEdit::SplinePrimitiveType::OpenLine; return true; }
    if (value == "arc" || value == "open_arc") { out = MeshEdit::SplinePrimitiveType::OpenArc; return true; }
    return false;
}

json profileSweepResultJson(const MeshEdit::ProfileSweepResult& result) {
    json diagnostics = json::array();
    for (const auto& diagnostic : result.report.diagnostics) {
        diagnostics.push_back({
            {"code", diagnostic.code}, {"message", diagnostic.message},
            {"warning", diagnostic.warning}
        });
    }
    return {
        {"ok", result.report.ok},
        {"operation", result.report.operation_id},
        {"vertex_count", result.geometry ? result.geometry->get_vertex_count() : 0},
        {"triangle_count", result.geometry ? result.geometry->indices.size() / 3 : 0},
        {"path_ring_count", result.path_ring_count},
        {"profile_ring_count", result.profile_ring_count},
        {"diagnostics", std::move(diagnostics)}
    };
}

json profileRevolveResultJson(const MeshEdit::ProfileRevolveResult& result) {
    json diagnostics = json::array();
    for (const auto& diagnostic : result.report.diagnostics) {
        diagnostics.push_back({
            {"code", diagnostic.code}, {"message", diagnostic.message},
            {"warning", diagnostic.warning}
        });
    }
    return {
        {"ok", result.report.ok}, {"operation", result.report.operation_id},
        {"vertex_count", result.geometry ? result.geometry->get_vertex_count() : 0},
        {"triangle_count", result.geometry ? result.geometry->indices.size() / 3 : 0},
        {"angle_ring_count", result.angle_ring_count},
        {"profile_ring_count", result.profile_ring_count},
        {"diagnostics", std::move(diagnostics)}
    };
}

json profileLoftResultJson(const MeshEdit::ProfileLoftResult& result) {
    json diagnostics = json::array();
    for (const auto& diagnostic : result.report.diagnostics) {
        diagnostics.push_back({
            {"code", diagnostic.code}, {"message", diagnostic.message},
            {"warning", diagnostic.warning}
        });
    }
    return {
        {"ok", result.report.ok}, {"operation", result.report.operation_id},
        {"vertex_count", result.geometry ? result.geometry->get_vertex_count() : 0},
        {"triangle_count", result.geometry ? result.geometry->indices.size() / 3 : 0},
        {"section_count", result.section_count}, {"ring_size", result.ring_size},
        {"diagnostics", std::move(diagnostics)}
    };
}

bool loadLoftSections(const json& params, std::vector<BezierSpline>& storage,
                      std::vector<const BezierSpline*>& sections,
                      std::string& error) {
    if (!params.contains("sections") || !params["sections"].is_array() ||
        params["sections"].size() < 2) {
        error = "sections must contain at least two spline names";
        return false;
    }
    for (const auto& value : params["sections"]) {
        if (!value.is_string()) { error = "sections must contain spline names"; return false; }
        std::string payload;
        const rtapi::Result fetched = rtapi::getSpline(value.get<std::string>(), payload);
        if (!fetched.ok) { error = fetched.error; return false; }
        MeshEdit::SplineObject object;
        std::string decodeError;
        try {
            if (!MeshEdit::deserializeSpline(json::parse(payload), object, decodeError)) {
                error = decodeError; return false;
            }
        } catch (const std::exception& e) {
            error = std::string("invalid spline payload: ") + e.what(); return false;
        }
        storage.push_back(std::move(object.spline));
    }
    sections.reserve(storage.size());
    for (const auto& spline : storage) sections.push_back(&spline);
    return true;
}

json profilePublishResultJson(const MeshEdit::ProfilePublishResult& result) {
    json diagnostics = json::array();
    for (const auto& diagnostic : result.report.diagnostics) {
        diagnostics.push_back({
            {"code", diagnostic.code}, {"message", diagnostic.message},
            {"warning", diagnostic.warning}
        });
    }
    return {
        {"ok", result.report.ok}, {"operation", result.report.operation_id},
        {"object", result.object_name},
        {"vertex_count", result.report.changed.vertices_changed},
        {"triangle_count", result.report.changed.triangles_changed},
        {"undoable", result.report.ok}, {"diagnostics", std::move(diagnostics)}
    };
}

} // namespace

bool dispatchMeshToolMethod(const std::string& method,
                            const nlohmann::json& params,
                            nlohmann::json& out_result) {
    if (method == "mesh.tools.list") {
        const std::string workspace = params.value("workspace", "edit");
        const bool includeUnavailable = params.value("include_unavailable", false);
        if (!isKnownWorkspace(workspace)) {
            out_result = {{"__error", "unknown mesh workspace: " + workspace}};
            return true;
        }

        json tools = json::array();
        for (const auto& tool : MeshEdit::MeshToolRegistry::instance().list(
                 parseWorkspace(workspace), includeUnavailable)) {
            tools.push_back(serializeTool(tool));
        }
        out_result = {{"workspace", workspace}, {"tools", std::move(tools)}};
        return true;
    }

    if (method == "mesh.tools.describe") {
        const std::string id = params.value("tool", "");
        if (id.empty()) {
            out_result = {{"__error", "missing required parameter: tool"}};
            return true;
        }
        const auto* tool = MeshEdit::MeshToolRegistry::instance().find(id);
        if (!tool) {
            out_result = {{"__error", "unknown mesh tool: " + id}};
            return true;
        }
        out_result = serializeTool(*tool);
        return true;
    }

    if (method == "mesh.asset.validate") {
        const std::string object = params.value("object", "");
        if (object.empty()) {
            out_result = {{"__error", "missing required parameter: object"}};
            return true;
        }
        rtapi::MeshValidationInfo info;
        const rtapi::Result result = rtapi::validateMesh(object, info);
        if (!result.ok) {
            out_result = {{"__error", result.error}};
            return true;
        }
        out_result = {
            {"object", object},
            {"valid", info.valid},
            {"vertex_count", info.vertex_count},
            {"triangle_count", info.triangle_count},
            {"non_finite_vertices", info.non_finite_vertices},
            {"out_of_range_indices", info.out_of_range_indices},
            {"degenerate_triangles", info.degenerate_triangles},
            {"non_finite_normals", info.non_finite_normals}
        };
        return true;
    }

    if (method == "mesh.operation.plan") {
        const std::string object = params.value("object", "");
        const std::string tool = params.value("tool", "");
        if (object.empty() || tool.empty()) {
            out_result = {{"__error", "object and tool are required"}};
            return true;
        }
        rtapi::MeshOperationPlanInfo info;
        const rtapi::Result result = rtapi::planMeshOperation(
            object, tool, params.value("backend", "auto"),
            params.value("preview", false), params.value("commit", false), info);
        if (!result.ok) { out_result = {{"__error", result.error}}; return true; }
        json diagnostics = json::array();
        for (size_t i = 0; i < info.diagnostic_codes.size(); ++i) {
            diagnostics.push_back({
                {"code", info.diagnostic_codes[i]},
                {"message", info.diagnostic_messages[i]},
                {"warning", info.diagnostic_warnings[i]}
            });
        }
        out_result = {
            {"ok", info.ok}, {"object", info.object_name},
            {"tool", info.operation_id}, {"backend", info.backend},
            {"preview", info.preview}, {"commit", info.commit},
            {"undoable", info.undoable},
            {"requires_cpu_fallback", info.requires_cpu_fallback},
            {"diagnostics", std::move(diagnostics)}
        };
        return true;
    }

    if (method == "mesh.operation.self_test") {
        std::string transactionReport;
        std::string publisherReport;
        const bool transactionOk = MeshEdit::runMeshTopologyTransactionSelfTest(transactionReport);
        const bool publisherOk = MeshEdit::runFlatMeshPublisherSelfTest(publisherReport);
        std::string undoReport;
        const bool undoOk = MeshEdit::runFlatMeshUndoSnapshotSelfTest(undoReport);
        out_result = {
            {"ok", transactionOk && publisherOk && undoOk},
            {"transaction_ok", transactionOk},
            {"transaction_report", transactionReport},
            {"publisher_ok", publisherOk},
            {"publisher_report", publisherReport},
            {"undo_ok", undoOk},
            {"undo_report", undoReport}
        };
        return true;
    }

    if (method == "mesh.profile.sweep.preview" || method == "mesh.profile.sweep.commit" ||
        method == "mesh.profile.sweep.self_test") {
        if (method == "mesh.profile.sweep.self_test") {
            std::string details;
            const bool ok = MeshEdit::runProfileSweepSelfTest(&details);
            out_result = {{"ok", ok}, {"details", details}};
            return true;
        }
        MeshEdit::ProfileSweepPreviewRequest request;
        if (!parsePrimitiveType(params.value("profile", "circle"), request.profile) ||
            !parsePrimitiveType(params.value("path", "line"), request.path)) {
            out_result = {{"__error", "profile/path must be circle, rectangle, line or arc"}};
            return true;
        }
        request.primitive.radius = params.value("radius", request.primitive.radius);
        request.primitive.width = params.value("width", request.primitive.width);
        request.primitive.height = params.value("height", request.primitive.height);
        request.primitive.start_angle = params.value("start_angle", request.primitive.start_angle);
        request.primitive.end_angle = params.value("end_angle", request.primitive.end_angle);
        request.primitive.arc_points = params.value("arc_points", request.primitive.arc_points);
        request.sweep.path_samples = params.value("path_samples", request.sweep.path_samples);
        request.sweep.profile_samples = params.value("profile_samples", request.sweep.profile_samples);
        request.sweep.cap_start = params.value("cap_start", request.sweep.cap_start);
        request.sweep.cap_end = params.value("cap_end", request.sweep.cap_end);
        request.sweep.profile_scale = params.value("profile_scale", request.sweep.profile_scale);
        const MeshEdit::ProfileSweepResult preview = MeshEdit::previewProfileSweep(request);
        if (method == "mesh.profile.sweep.commit") {
            if (!preview.report.ok) { out_result = profileSweepResultJson(preview); return true; }
            if (!rtapi::g_ctx || !rtapi::g_history) {
                out_result = {{"__error", "rtapi scene binding is unavailable"}};
                return true;
            }
            out_result = profilePublishResultJson(MeshEdit::publishGeneratedProfile(
                *rtapi::g_ctx, *rtapi::g_history, preview.geometry,
                params.value("object", ""), "profile.sweep"));
        } else {
            out_result = profileSweepResultJson(preview);
        }
        return true;
    }

    if (method == "mesh.profile.revolve.preview" || method == "mesh.profile.revolve.commit" ||
        method == "mesh.profile.revolve.self_test") {
        if (method == "mesh.profile.revolve.self_test") {
            std::string details;
            const bool ok = MeshEdit::runProfileRevolveSelfTest(&details);
            out_result = {{"ok", ok}, {"details", details}};
            return true;
        }
        MeshEdit::ProfileRevolveSettings settings;
        settings.angle_segments = params.value("angle_segments", settings.angle_segments);
        settings.profile_samples = params.value("profile_samples", settings.profile_samples);
        settings.start_angle = params.value("start_angle", settings.start_angle);
        settings.end_angle = params.value("end_angle", settings.end_angle);
        const MeshEdit::ProfileRevolveResult preview = MeshEdit::previewProfileRevolve(
            params.value("preset", "bottle"), settings);
        if (method == "mesh.profile.revolve.commit") {
            if (!preview.report.ok) { out_result = profileRevolveResultJson(preview); return true; }
            if (!rtapi::g_ctx || !rtapi::g_history) {
                out_result = {{"__error", "rtapi scene binding is unavailable"}};
                return true;
            }
            out_result = profilePublishResultJson(MeshEdit::publishGeneratedProfile(
                *rtapi::g_ctx, *rtapi::g_history, preview.geometry,
                params.value("object", ""), "profile.revolve"));
        } else {
            out_result = profileRevolveResultJson(preview);
        }
        return true;
    }

    if (method == "mesh.profile.loft.preview" || method == "mesh.profile.loft.commit" ||
        method == "mesh.profile.loft.self_test") {
        if (method == "mesh.profile.loft.self_test") {
            std::string details;
            const bool ok = MeshEdit::runProfileLoftSelfTest(&details);
            out_result = {{"ok", ok}, {"details", details}};
            return true;
        }
        std::vector<BezierSpline> storage;
        std::vector<const BezierSpline*> sections;
        std::string error;
        if (!loadLoftSections(params, storage, sections, error)) {
            out_result = {{"__error", error}};
            return true;
        }
        MeshEdit::ProfileLoftSettings settings;
        settings.samples_per_section = params.value("samples_per_section", settings.samples_per_section);
        settings.cap_start = params.value("cap_start", settings.cap_start);
        settings.cap_end = params.value("cap_end", settings.cap_end);
        const MeshEdit::ProfileLoftResult preview = MeshEdit::buildProfileLoft(sections, settings);
        if (method == "mesh.profile.loft.commit") {
            if (!preview.report.ok) { out_result = profileLoftResultJson(preview); return true; }
            if (!rtapi::g_ctx || !rtapi::g_history) {
                out_result = {{"__error", "rtapi scene binding is unavailable"}};
                return true;
            }
            out_result = profilePublishResultJson(MeshEdit::publishGeneratedProfile(
                *rtapi::g_ctx, *rtapi::g_history, preview.geometry,
                params.value("object", ""), "profile.loft"));
        } else {
            out_result = profileLoftResultJson(preview);
        }
        return true;
    }

    if (method == "mesh.operation.commit_positions") {
        const std::string object = params.value("object", "");
        if (object.empty() || !params.contains("positions") || !params["positions"].is_array()) {
            out_result = {{"__error", "object and positions array are required"}};
            return true;
        }
        const auto& rows = params["positions"];
        std::vector<float> values;
        values.reserve(rows.size() * 3);
        for (const auto& row : rows) {
            if (!row.is_array() || row.size() != 3) {
                out_result = {{"__error", "positions must be an array of [x,y,z] rows"}};
                return true;
            }
            for (const auto& component : row) {
                if (!component.is_number()) {
                    out_result = {{"__error", "position components must be numeric"}};
                    return true;
                }
                values.push_back(component.get<float>());
            }
        }
        const rtapi::Result result = rtapi::setMeshPositionsUndoable(
            object, values.data(), rows.size());
        if (!result.ok) { out_result = {{"__error", result.error}}; return true; }
        out_result = {{"ok", true}, {"object", object}, {"vertex_count", rows.size()},
                      {"undoable", true}};
        return true;
    }

    return false;
}
