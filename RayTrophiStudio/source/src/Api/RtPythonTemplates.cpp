#include "RtPythonTemplates.h"

#include "Template/TemplateRegistry.h"
#include "Template/TemplateLoader.h"
#include "ProjectManager.h"
#include "Api/RtApi.h"

#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

py::dict metadataDict(const raytrophi::templates::TemplateMetadata& item) {
    py::dict ui;
    ui["properties_context"] = item.ui_state.properties_context;
    ui["bottom_editor"] = item.ui_state.bottom_editor;
    ui["contextual_dock"] = item.ui_state.contextual_dock;
    ui["contextual_dock_width"] = item.ui_state.contextual_dock_width;
    ui["viewport_shading"] = item.ui_state.viewport_shading;
    ui["frame_target"] = item.ui_state.frame_target;
    ui["show_timeline"] = item.ui_state.show_timeline;

    py::dict out;
    out["id"] = item.id;
    out["display_name"] = item.display_name;
    out["description"] = item.description;
    out["kind"] = item.kind;
    out["category"] = item.category;
    out["sort_order"] = item.sort_order;
    out["schema_version"] = item.schema_version;
    out["minimum_raytrophi_version"] = item.minimum_raytrophi_version;
    out["maximum_raytrophi_version"] = item.maximum_raytrophi_version;
    out["package_root"] = item.package_root.generic_string();
    out["manifest_path"] = item.manifest_path.generic_string();
    out["preview_path"] = item.preview_path.generic_string();
    out["preview_alt"] = item.preview_alt;
    out["scene_type"] = item.scene_type;
    out["scene_path"] = item.scene_path.generic_string();
    out["ui_state"] = std::move(ui);
    out["renderer_preference"] = item.renderer_preference;
    out["renderer_allow_fallback"] = item.renderer_allow_fallback;
    out["performance_class"] = item.performance_class;
    out["estimated_vram_mb"] = item.estimated_vram_mb;
    out["valid"] = item.valid;
    out["errors"] = item.errors;
    return out;
}

py::dict loadPlanDict(const raytrophi::templates::TemplateLoadPlan& plan) {
    py::dict out;
    out["template_id"] = plan.template_id;
    out["state"] = plan.state;
    out["code"] = plan.code;
    out["scene_type"] = plan.scene_type;
    out["manifest_path"] = plan.manifest_path.generic_string();
    out["scene_path"] = plan.scene_path.generic_string();
    out["binary_path"] = plan.binary_path.generic_string();
    out["auxiliary_path"] = plan.auxiliary_path.generic_string();
    out["ready"] = plan.ready;
    out["has_unsaved_changes"] = plan.has_unsaved_changes;
    out["requires_discard"] = plan.requires_discard;
    out["errors"] = plan.errors;
    out["warnings"] = plan.warnings;
    return out;
}

py::dict openInfoDict(const rtapi::TemplateOpenInfo& info) {
    py::dict out;
    out["template_id"] = info.template_id;
    out["state"] = info.state;
    out["code"] = info.code;
    out["opened"] = info.opened;
    out["ui_state_applied"] = info.ui_state_applied;
    out["errors"] = info.errors;
    out["warnings"] = info.warnings;
    return out;
}

} // namespace

namespace rtpy {

void registerTemplateBindings(py::module_& root) {
    py::module_ module = root.def_submodule(
        "templates", "Discover, validate, and open RayTrophi project templates");

    module.def("refresh", []() {
        raytrophi::templates::TemplateRegistry::instance().refresh();
    }, "Rescan the built-in template search roots.");

    module.def("list", [](bool include_invalid) {
        py::list out;
        const auto& entries = raytrophi::templates::TemplateRegistry::instance().entries();
        for (const auto& item : entries) {
            if (include_invalid || item.valid) out.append(metadataDict(item));
        }
        return out;
    }, py::arg("include_invalid") = false,
       "Return deterministically sorted template metadata.");

    module.def("get", [](const std::string& id) {
        const auto* item = raytrophi::templates::TemplateRegistry::instance().find(id);
        if (!item) throw py::key_error("template not found: " + id);
        return metadataDict(*item);
    }, py::arg("id"));

    module.def("validate", [](const std::string& id) {
        const auto* item = raytrophi::templates::TemplateRegistry::instance().find(id);
        if (!item) throw py::key_error("template not found: " + id);
        py::dict out;
        out["id"] = item->id;
        out["valid"] = item->valid;
        out["errors"] = item->errors;
        return out;
    }, py::arg("id"));

    module.def("prepare", [](const std::string& id, const std::string& conflict_policy) {
        raytrophi::templates::TemplateConflictPolicy policy;
        if (!raytrophi::templates::TemplateLoader::parseConflictPolicy(conflict_policy, policy))
            throw py::value_error("conflict_policy must be 'reject' or 'discard'");
        const auto plan = raytrophi::templates::TemplateLoader::instance().prepare(
            id, ProjectManager::getInstance().hasUnsavedChanges(), policy);
        return loadPlanDict(plan);
    }, py::arg("id"), py::arg("conflict_policy") = "reject",
       "Preflight a template without mutating the active project or UI.");

    module.def("open", [](const std::string& id, const std::string& conflict_policy) {
        rtapi::TemplateOpenInfo info;
        (void)rtapi::openTemplate(id, conflict_policy, info);
        return openInfoDict(info);
    }, py::arg("id"), py::arg("conflict_policy") = "reject",
       "Open a preflighted template through the canonical transactional recipe boundary.");
}

} // namespace rtpy
