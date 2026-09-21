#include "RtPythonMeshEdit.h"

#include "Api/RtApi.h"
#include "MeshEdit/ProfileSweepService.h"
#include "MeshEdit/ProfileLoft.h"
#include "MeshEdit/SplineSerialization.h"
#include "MeshEdit/ProfileAuthoringService.h"
#include "Api/RtApiInternal.h"
#include "RtPyCommon.h"   // vec3FromPython, shared by every binding TU

#include <stdexcept>
#include <pybind11/numpy.h>
// ★ std::vector<> arguments, RETURNS and default arguments all need this
// caster. Without it a vector default argument fails at MODULE REGISTRATION
// time ("could not convert default argument ... type not registered yet"),
// which kills the whole embedded interpreter rather than the one binding --
// so the cost of the missing include is paid by every script, not just the
// function that needed it.
#include <pybind11/stl.h>
#include "json.hpp"

namespace py = pybind11;

namespace rtpy {

namespace {

MeshEdit::SplinePrimitiveType parseSplinePrimitive(const std::string& value) {
    if (value == "circle") return MeshEdit::SplinePrimitiveType::Circle;
    if (value == "rectangle") return MeshEdit::SplinePrimitiveType::Rectangle;
    if (value == "line" || value == "open_line") return MeshEdit::SplinePrimitiveType::OpenLine;
    if (value == "arc" || value == "open_arc") return MeshEdit::SplinePrimitiveType::OpenArc;
    throw std::invalid_argument("primitive must be circle, rectangle, line or arc");
}

MeshEdit::ProfileRevolveAxis parseProfileRevolveAxis(const std::string& value) {
    if (value == "x" || value == "X") return MeshEdit::ProfileRevolveAxis::X;
    if (value == "y" || value == "Y") return MeshEdit::ProfileRevolveAxis::Y;
    if (value == "z" || value == "Z") return MeshEdit::ProfileRevolveAxis::Z;
    throw std::invalid_argument("axis must be x, y or z");
}

py::dict profileSweepPreviewDict(const MeshEdit::ProfileSweepResult& result) {
    py::dict out;
    out["ok"] = result.report.ok;
    out["operation"] = result.report.operation_id;
    out["vertex_count"] = result.geometry ? result.geometry->get_vertex_count() : 0;
    out["triangle_count"] = result.geometry ? result.geometry->indices.size() / 3 : 0;
    out["path_ring_count"] = result.path_ring_count;
    out["profile_ring_count"] = result.profile_ring_count;
    py::list diagnostics;
    for (const auto& diagnostic : result.report.diagnostics) {
        py::dict item;
        item["code"] = diagnostic.code;
        item["message"] = diagnostic.message;
        item["warning"] = diagnostic.warning;
        diagnostics.append(item);
    }
    out["diagnostics"] = diagnostics;
    return out;
}

py::dict profileRevolvePreviewDict(const MeshEdit::ProfileRevolveResult& result) {
    py::dict out;
    out["ok"] = result.report.ok;
    out["operation"] = result.report.operation_id;
    out["vertex_count"] = result.geometry ? result.geometry->get_vertex_count() : 0;
    out["triangle_count"] = result.geometry ? result.geometry->indices.size() / 3 : 0;
    out["angle_ring_count"] = result.angle_ring_count;
    out["profile_ring_count"] = result.profile_ring_count;
    py::list diagnostics;
    for (const auto& diagnostic : result.report.diagnostics) {
        py::dict item;
        item["code"] = diagnostic.code;
        item["message"] = diagnostic.message;
        item["warning"] = diagnostic.warning;
        diagnostics.append(item);
    }
    out["diagnostics"] = diagnostics;
    return out;
}

py::dict profileLoftPreviewDict(const MeshEdit::ProfileLoftResult& result) {
    py::dict out;
    out["ok"] = result.report.ok;
    out["operation"] = result.report.operation_id;
    out["vertex_count"] = result.geometry ? result.geometry->get_vertex_count() : 0;
    out["triangle_count"] = result.geometry ? result.geometry->indices.size() / 3 : 0;
    out["section_count"] = result.section_count;
    out["ring_size"] = result.ring_size;
    py::list diagnostics;
    for (const auto& diagnostic : result.report.diagnostics) {
        py::dict item;
        item["code"] = diagnostic.code;
        item["message"] = diagnostic.message;
        item["warning"] = diagnostic.warning;
        diagnostics.append(item);
    }
    out["diagnostics"] = diagnostics;
    return out;
}

MeshEdit::ProfileLoftResult buildLoftFromNames(const std::vector<std::string>& names,
                                                int samples, bool cap_start, bool cap_end) {
    if (names.size() < 2) throw std::invalid_argument("sections must contain at least two spline names");
    std::vector<BezierSpline> storage;
    storage.reserve(names.size());
    for (const auto& name : names) {
        std::string payload;
        const rtapi::Result fetched = rtapi::getSpline(name, payload);
        if (!fetched.ok) throw std::runtime_error(fetched.error);
        MeshEdit::SplineObject object;
        std::string error;
        if (!MeshEdit::deserializeSpline(nlohmann::json::parse(payload), object, error))
            throw std::invalid_argument(error);
        storage.push_back(std::move(object.spline));
    }
    std::vector<const BezierSpline*> sections;
    sections.reserve(storage.size());
    for (const auto& spline : storage) sections.push_back(&spline);
    MeshEdit::ProfileLoftSettings settings;
    settings.samples_per_section = samples;
    settings.cap_start = cap_start;
    settings.cap_end = cap_end;
    return MeshEdit::buildProfileLoft(sections, settings);
}

py::dict profilePublishDict(const MeshEdit::ProfilePublishResult& result) {
    py::dict out;
    out["ok"] = result.report.ok;
    out["operation"] = result.report.operation_id;
    out["object"] = result.object_name;
    out["vertex_count"] = result.report.changed.vertices_changed;
    out["triangle_count"] = result.report.changed.triangles_changed;
    out["undoable"] = result.report.ok;
    py::list diagnostics;
    for (const auto& diagnostic : result.report.diagnostics) {
        py::dict item;
        item["code"] = diagnostic.code;
        item["message"] = diagnostic.message;
        item["warning"] = diagnostic.warning;
        diagnostics.append(item);
    }
    out["diagnostics"] = diagnostics;
    return out;
}

} // namespace

void registerMeshEditBindings(py::module_& mesh) {
    mesh.def("validate", [](const std::string& object) -> py::dict {
        rtapi::MeshValidationInfo info;
        const rtapi::Result result = rtapi::validateMesh(object, info);
        if (!result.ok) throw std::runtime_error(result.error);
        py::dict out;
        out["valid"] = info.valid;
        out["vertex_count"] = info.vertex_count;
        out["triangle_count"] = info.triangle_count;
        out["non_finite_vertices"] = info.non_finite_vertices;
        out["out_of_range_indices"] = info.out_of_range_indices;
        out["degenerate_triangles"] = info.degenerate_triangles;
        out["non_finite_normals"] = info.non_finite_normals;
        return out;
    }, py::arg("object"));

    mesh.def("plan_operation", [](const std::string& object,
                                   const std::string& tool,
                                   const std::string& backend,
                                   bool preview,
                                   bool commit) -> py::dict {
        rtapi::MeshOperationPlanInfo info;
        const rtapi::Result result = rtapi::planMeshOperation(
            object, tool, backend, preview, commit, info);
        if (!result.ok) throw std::runtime_error(result.error);
        py::dict out;
        out["ok"] = info.ok;
        out["object"] = info.object_name;
        out["tool"] = info.operation_id;
        out["backend"] = info.backend;
        out["preview"] = info.preview;
        out["commit"] = info.commit;
        out["undoable"] = info.undoable;
        out["requires_cpu_fallback"] = info.requires_cpu_fallback;
        py::list diagnostics;
        for (size_t i = 0; i < info.diagnostic_codes.size(); ++i) {
            py::dict diagnostic;
            diagnostic["code"] = info.diagnostic_codes[i];
            diagnostic["message"] = info.diagnostic_messages[i];
            diagnostic["warning"] = static_cast<bool>(info.diagnostic_warnings[i]);
            diagnostics.append(diagnostic);
        }
        out["diagnostics"] = diagnostics;
        return out;
    }, py::arg("object"), py::arg("tool"), py::arg("backend") = "auto",
       py::arg("preview") = false, py::arg("commit") = false);

    mesh.def("set_positions_undoable", [](const std::string& object,
                                           py::array_t<float, py::array::c_style | py::array::forcecast> positions) {
        if (positions.ndim() != 2 || positions.shape(1) != 3) {
            throw std::invalid_argument("positions must have shape (vertex_count, 3)");
        }
        const rtapi::Result result = rtapi::setMeshPositionsUndoable(
            object, positions.data(), static_cast<size_t>(positions.shape(0)));
        if (!result.ok) throw std::runtime_error(result.error);
    }, py::arg("object"), py::arg("positions"));

    mesh.def("profile_sweep_preview", [](const std::string& profile,
                                           const std::string& path,
                                           int path_samples,
                                           int profile_samples,
                                           float radius,
                                           float width,
                                           float height,
                                           bool cap_start,
                                           bool cap_end) {
        MeshEdit::ProfileSweepPreviewRequest request;
        request.profile = parseSplinePrimitive(profile);
        request.path = parseSplinePrimitive(path);
        request.sweep.path_samples = path_samples;
        request.sweep.profile_samples = profile_samples;
        request.primitive.radius = radius;
        request.primitive.width = width;
        request.primitive.height = height;
        request.sweep.cap_start = cap_start;
        request.sweep.cap_end = cap_end;
        return profileSweepPreviewDict(MeshEdit::previewProfileSweep(request));
    }, py::arg("profile") = "circle", py::arg("path") = "line",
       py::arg("path_samples") = 32, py::arg("profile_samples") = 16,
       py::arg("radius") = 1.0f, py::arg("width") = 2.0f,
       py::arg("height") = 2.0f, py::arg("cap_start") = true,
       py::arg("cap_end") = true);

    mesh.def("profile_sweep_commit", [](const std::string& profile,
                                          const std::string& path,
                                          const std::string& object,
                                          int path_samples,
                                          int profile_samples,
                                          float radius,
                                          float width,
                                          float height) {
        if (!rtapi::g_ctx || !rtapi::g_history) throw std::runtime_error("rtapi scene binding is unavailable");
        MeshEdit::ProfileSweepPreviewRequest request;
        request.profile = parseSplinePrimitive(profile);
        request.path = parseSplinePrimitive(path);
        request.sweep.path_samples = path_samples;
        request.sweep.profile_samples = profile_samples;
        request.primitive.radius = radius;
        request.primitive.width = width;
        request.primitive.height = height;
        const auto preview = MeshEdit::previewProfileSweep(request);
        if (!preview.report.ok) return profileSweepPreviewDict(preview);
        return profilePublishDict(MeshEdit::publishGeneratedProfile(
            *rtapi::g_ctx, *rtapi::g_history, preview.geometry, object, "profile.sweep"));
    }, py::arg("profile") = "circle", py::arg("path") = "line",
       py::arg("object") = "", py::arg("path_samples") = 32,
       py::arg("profile_samples") = 16, py::arg("radius") = 1.0f,
       py::arg("width") = 2.0f, py::arg("height") = 2.0f);

    mesh.def("profile_revolve_preview", [](const std::string& preset,
                                             int angle_segments,
                                             int profile_samples,
                                             float start_angle,
                                             float end_angle,
                                             float radius_offset,
                                             const std::string& axis,
                                             float pivot_x, float pivot_y, float pivot_z) {
        MeshEdit::ProfileRevolveSettings settings;
        settings.angle_segments = angle_segments;
        settings.profile_samples = profile_samples;
        settings.start_angle = start_angle;
        settings.end_angle = end_angle;
        settings.axis = parseProfileRevolveAxis(axis);
        settings.radius_offset = radius_offset;
        settings.axis_pivot = Vec3(pivot_x, pivot_y, pivot_z);
        return profileRevolvePreviewDict(MeshEdit::previewProfileRevolve(preset, settings));
    }, py::arg("preset") = "bottle", py::arg("angle_segments") = 32,
       py::arg("profile_samples") = 24, py::arg("start_angle") = 0.0f,
       py::arg("end_angle") = 2.0f * M_PI, py::arg("radius_offset") = 0.0f,
       py::arg("axis") = "y", py::arg("pivot_x") = 0.0f,
       py::arg("pivot_y") = 0.0f, py::arg("pivot_z") = 0.0f);

    mesh.def("profile_revolve_commit", [](const std::string& preset,
                                            const std::string& object,
                                            int angle_segments,
                                            int profile_samples,
                                            float start_angle,
                                            float end_angle,
                                            float radius_offset,
                                            const std::string& axis,
                                            float pivot_x, float pivot_y, float pivot_z) {
        if (!rtapi::g_ctx || !rtapi::g_history) throw std::runtime_error("rtapi scene binding is unavailable");
        MeshEdit::ProfileRevolveSettings settings;
        settings.angle_segments = angle_segments;
        settings.profile_samples = profile_samples;
        settings.start_angle = start_angle;
        settings.end_angle = end_angle;
        settings.axis = parseProfileRevolveAxis(axis);
        settings.radius_offset = radius_offset;
        settings.axis_pivot = Vec3(pivot_x, pivot_y, pivot_z);
        const auto preview = MeshEdit::previewProfileRevolve(preset, settings);
        if (!preview.report.ok) return profileRevolvePreviewDict(preview);
        return profilePublishDict(MeshEdit::publishGeneratedProfile(
            *rtapi::g_ctx, *rtapi::g_history, preview.geometry, object, "profile.revolve"));
    }, py::arg("preset") = "bottle", py::arg("object") = "",
       py::arg("angle_segments") = 32, py::arg("profile_samples") = 24,
       py::arg("start_angle") = 0.0f, py::arg("end_angle") = 2.0f * M_PI,
       py::arg("radius_offset") = 0.0f, py::arg("axis") = "y",
       py::arg("pivot_x") = 0.0f, py::arg("pivot_y") = 0.0f,
       py::arg("pivot_z") = 0.0f);

    mesh.def("profile_loft_preview", [](const std::vector<std::string>& sections,
                                         int samples_per_section,
                                         bool cap_start, bool cap_end) {
        return profileLoftPreviewDict(buildLoftFromNames(
            sections, samples_per_section, cap_start, cap_end));
    }, py::arg("sections"), py::arg("samples_per_section") = 24,
       py::arg("cap_start") = true, py::arg("cap_end") = true);

    mesh.def("profile_loft_commit", [](const std::vector<std::string>& sections,
                                        const std::string& object,
                                        int samples_per_section,
                                        bool cap_start, bool cap_end) {
        if (!rtapi::g_ctx || !rtapi::g_history)
            throw std::runtime_error("rtapi scene binding is unavailable");
        const auto preview = buildLoftFromNames(
            sections, samples_per_section, cap_start, cap_end);
        if (!preview.report.ok) return profileLoftPreviewDict(preview);
        return profilePublishDict(MeshEdit::publishGeneratedProfile(
            *rtapi::g_ctx, *rtapi::g_history, preview.geometry, object, "profile.loft"));
    }, py::arg("sections"), py::arg("object") = "",
       py::arg("samples_per_section") = 24,
       py::arg("cap_start") = true, py::arg("cap_end") = true);

    // ── Polygon editing ───────────────────────────────────────────────
    // ★ These operators existed on SceneUI long before any script could
    // reach them; mesh.tools.list advertised them as scriptable while the
    // only entry point was a button. Parity with IPC is the point here --
    // whatever an agent can drive over the pipe, a script can drive too.
    auto stateDict = [](const rtapi::MeshEditState& state) {
        py::dict out;
        out["object"] = state.object;
        out["cache_valid"] = state.cache_valid;
        out["half_edge_valid"] = state.half_edge_valid;
        out["vertices"] = state.vertex_count;
        out["edges"] = state.edge_count;
        out["faces"] = state.face_count;
        out["triangles"] = state.triangle_count;
        out["selected_vertices"] = state.selected_vertices;
        out["selected_edges"] = state.selected_edges;
        out["selected_faces"] = state.selected_faces;
        out["half_edge_manifold"] = state.half_edge_manifold;
        out["non_manifold_edges"] = state.non_manifold_edges;
        out["skipped_polygons"] = state.skipped_polygons;
        out["half_edge_message"] = state.half_edge_message;
        return out;
    };
    auto parseDomain = [](const std::string& name) {
        if (name == "vertex") return rtapi::MeshElementDomain::Vertex;
        if (name == "edge") return rtapi::MeshElementDomain::Edge;
        if (name == "face") return rtapi::MeshElementDomain::Face;
        throw std::runtime_error("domain must be one of vertex|edge|face");
    };
    auto parseMode = [](const std::string& name) {
        if (name == "set") return rtapi::MeshSelectMode::Set;
        if (name == "add") return rtapi::MeshSelectMode::Add;
        if (name == "remove") return rtapi::MeshSelectMode::Remove;
        throw std::runtime_error("mode must be one of set|add|remove");
    };
    auto readState = [stateDict](const std::string& object) {
        rtapi::MeshEditState state;
        const rtapi::Result result = rtapi::meshEditGetState(object, state);
        if (!result.ok) throw std::runtime_error(result.error);
        return stateDict(state);
    };
    auto run = [readState](const std::string& object, const rtapi::Result& result) {
        if (!result.ok) throw std::runtime_error(result.error);
        return readState(object);
    };

    mesh.def("edit_begin", [readState](const std::string& object) {
        const rtapi::Result result = rtapi::meshEditBegin(object);
        if (!result.ok) throw std::runtime_error(result.error);
        return readState(object);
    }, py::arg("object"));

    mesh.def("edit_state", [readState](const std::string& object) {
        return readState(object);
    }, py::arg("object") = "");

    mesh.def("select", [readState, parseDomain, parseMode](
                 const std::string& object, const std::string& domain,
                 const std::vector<int>& ids, const std::string& mode, bool all) {
        const rtapi::Result result = all
            ? rtapi::meshEditSelectAll(object, parseDomain(domain))
            : rtapi::meshEditSelect(object, parseDomain(domain), parseMode(mode), ids);
        if (!result.ok) throw std::runtime_error(result.error);
        return readState(object);
    }, py::arg("object"), py::arg("domain"), py::arg("ids") = std::vector<int>{},
       py::arg("mode") = "set", py::arg("all") = false);

    mesh.def("clear_selection", [readState](const std::string& object) {
        const rtapi::Result result = rtapi::meshEditClearSelection(object);
        if (!result.ok) throw std::runtime_error(result.error);
        return readState(object);
    }, py::arg("object") = "");

    mesh.def("get_selection", [parseDomain](const std::string& object,
                                            const std::string& domain) {
        std::vector<int> ids;
        const rtapi::Result result = rtapi::meshEditGetSelection(object, parseDomain(domain), ids);
        if (!result.ok) throw std::runtime_error(result.error);
        return ids;
    }, py::arg("object"), py::arg("domain"));

    mesh.def("select_by_normal", [readState, parseMode](
                 const std::string& object, py::handle direction,
                 float max_angle, const std::string& mode) {
        const rtapi::Result result = rtapi::meshEditSelectByNormal(
            object, vec3FromPython(direction), max_angle, parseMode(mode), nullptr);
        if (!result.ok) throw std::runtime_error(result.error);
        return readState(object);
    }, py::arg("object"), py::arg("direction"), py::arg("max_angle") = 30.0f,
       py::arg("mode") = "set");

    mesh.def("select_by_box", [readState, parseDomain, parseMode](
                 const std::string& object, py::handle box_min, py::handle box_max,
                 const std::string& domain, const std::string& mode, bool world_space) {
        const rtapi::Result result = rtapi::meshEditSelectByBox(
            object, vec3FromPython(box_min), vec3FromPython(box_max),
            parseDomain(domain), parseMode(mode), world_space, nullptr);
        if (!result.ok) throw std::runtime_error(result.error);
        return readState(object);
    }, py::arg("object"), py::arg("min"), py::arg("max"), py::arg("domain") = "face",
       py::arg("mode") = "set", py::arg("world_space") = true);

    mesh.def("extrude", [run](const std::string& object, float distance) {
        return run(object, rtapi::meshExtrudeFaces(object, distance));
    }, py::arg("object"), py::arg("distance"));

    mesh.def("inset", [run](const std::string& object, float amount) {
        return run(object, rtapi::meshInsetFaces(object, amount));
    }, py::arg("object"), py::arg("amount"));

    mesh.def("bevel", [run](const std::string& object, float width, int segments, bool round) {
        return run(object, rtapi::meshBevelEdges(object, width, segments, round));
    }, py::arg("object"), py::arg("width"), py::arg("segments") = 1, py::arg("round") = false);

    mesh.def("loop_cut", [run](const std::string& object, float t) {
        return run(object, rtapi::meshLoopCut(object, t));
    }, py::arg("object"), py::arg("t") = 0.5f);

    mesh.def("dissolve_edges", [run](const std::string& object) {
        return run(object, rtapi::meshDissolveEdges(object));
    }, py::arg("object") = "");

    mesh.def("dissolve_vertices", [run](const std::string& object) {
        return run(object, rtapi::meshDissolveVertices(object));
    }, py::arg("object") = "");

    mesh.def("merge_vertices", [run](const std::string& object) {
        return run(object, rtapi::meshMergeVertices(object));
    }, py::arg("object") = "");

    mesh.def("weld_vertices", [run](const std::string& object, float distance) {
        return run(object, rtapi::meshWeldVertices(object, distance));
    }, py::arg("object"), py::arg("distance"));
}

} // namespace rtpy
