#include "RtPythonMeshEdit.h"

#include "Api/RtApi.h"
#include "MeshEdit/ProfileSweepService.h"
#include "MeshEdit/ProfileLoft.h"
#include "MeshEdit/SplineSerialization.h"
#include "MeshEdit/ProfileAuthoringService.h"
#include "Api/RtApiInternal.h"

#include <stdexcept>
#include <pybind11/numpy.h>
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
                                             float end_angle) {
        MeshEdit::ProfileRevolveSettings settings;
        settings.angle_segments = angle_segments;
        settings.profile_samples = profile_samples;
        settings.start_angle = start_angle;
        settings.end_angle = end_angle;
        return profileRevolvePreviewDict(MeshEdit::previewProfileRevolve(preset, settings));
    }, py::arg("preset") = "bottle", py::arg("angle_segments") = 32,
       py::arg("profile_samples") = 24, py::arg("start_angle") = 0.0f,
       py::arg("end_angle") = 2.0f * M_PI);

    mesh.def("profile_revolve_commit", [](const std::string& preset,
                                            const std::string& object,
                                            int angle_segments,
                                            int profile_samples) {
        if (!rtapi::g_ctx || !rtapi::g_history) throw std::runtime_error("rtapi scene binding is unavailable");
        MeshEdit::ProfileRevolveSettings settings;
        settings.angle_segments = angle_segments;
        settings.profile_samples = profile_samples;
        const auto preview = MeshEdit::previewProfileRevolve(preset, settings);
        if (!preview.report.ok) return profileRevolvePreviewDict(preview);
        return profilePublishDict(MeshEdit::publishGeneratedProfile(
            *rtapi::g_ctx, *rtapi::g_history, preview.geometry, object, "profile.revolve"));
    }, py::arg("preset") = "bottle", py::arg("object") = "",
       py::arg("angle_segments") = 32, py::arg("profile_samples") = 24);

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
}

} // namespace rtpy
