#include "RtIpcTemplates.h"

#include "Template/TemplateRegistry.h"
#include "Template/TemplateLoader.h"
#include "ProjectManager.h"
#include "Api/RtApi.h"

namespace {

using json = nlohmann::json;

json metadataJson(const raytrophi::templates::TemplateMetadata& item) {
    return json{
        {"id", item.id},
        {"display_name", item.display_name},
        {"description", item.description},
        {"kind", item.kind},
        {"category", item.category},
        {"sort_order", item.sort_order},
        {"schema_version", item.schema_version},
        {"minimum_raytrophi_version", item.minimum_raytrophi_version},
        {"maximum_raytrophi_version", item.maximum_raytrophi_version},
        {"package_root", item.package_root.generic_string()},
        {"manifest_path", item.manifest_path.generic_string()},
        {"preview_path", item.preview_path.generic_string()},
        {"preview_alt", item.preview_alt},
        {"scene_type", item.scene_type},
        {"scene_path", item.scene_path.generic_string()},
        {"ui_state", {
            {"properties_context", item.ui_state.properties_context},
            {"bottom_editor", item.ui_state.bottom_editor},
            {"contextual_dock", item.ui_state.contextual_dock},
            {"contextual_dock_width", item.ui_state.contextual_dock_width},
            {"viewport_shading", item.ui_state.viewport_shading},
            {"frame_target", item.ui_state.frame_target},
            {"show_timeline", item.ui_state.show_timeline}}},
        {"renderer_preference", item.renderer_preference},
        {"renderer_allow_fallback", item.renderer_allow_fallback},
        {"performance_class", item.performance_class},
        {"estimated_vram_mb", item.estimated_vram_mb},
        {"valid", item.valid},
        {"errors", item.errors}};
}

json notFound(const std::string& id) {
    return json{{"__error", "template not found: " + id}, {"code", "template_not_found"}};
}

json loadPlanJson(const raytrophi::templates::TemplateLoadPlan& plan) {
    return json{{"template_id", plan.template_id}, {"state", plan.state},
                {"code", plan.code}, {"scene_type", plan.scene_type},
                {"manifest_path", plan.manifest_path.generic_string()},
                {"scene_path", plan.scene_path.generic_string()},
                {"binary_path", plan.binary_path.generic_string()},
                {"auxiliary_path", plan.auxiliary_path.generic_string()},
                {"ready", plan.ready}, {"has_unsaved_changes", plan.has_unsaved_changes},
                {"requires_discard", plan.requires_discard},
                {"errors", plan.errors}, {"warnings", plan.warnings}};
}

json openInfoJson(const rtapi::TemplateOpenInfo& info) {
    return json{{"template_id", info.template_id}, {"state", info.state},
                {"code", info.code}, {"opened", info.opened},
                {"ui_state_applied", info.ui_state_applied},
                {"errors", info.errors}, {"warnings", info.warnings}};
}

} // namespace

bool dispatchTemplateIpc(const std::string& method, const nlohmann::json& params,
                         const RtIpcTemplateEnqueue& enqueue, nlohmann::json& out_result) {
    if (method == "templates.refresh") {
        out_result = enqueue([](UIContext&) {
            auto& registry = raytrophi::templates::TemplateRegistry::instance();
            registry.refresh();
            return json{{"ok", true}, {"count", registry.entries().size()}};
        });
        return true;
    }
    if (method == "templates.list") {
        const bool include_invalid = params.value("include_invalid", false);
        out_result = enqueue([include_invalid](UIContext&) {
            json result = json::array();
            for (const auto& item : raytrophi::templates::TemplateRegistry::instance().entries()) {
                if (include_invalid || item.valid) result.push_back(metadataJson(item));
            }
            return result;
        });
        return true;
    }
    if (method == "templates.get" || method == "templates.validate") {
        if (!params.contains("id") || !params["id"].is_string()) {
            out_result = json{{"__error", "missing or invalid string parameter: id"},
                              {"code", "invalid_parameter"}};
            return true;
        }
        const std::string id = params["id"].get<std::string>();
        const bool validate_only = method == "templates.validate";
        out_result = enqueue([id, validate_only](UIContext&) {
            const auto* item = raytrophi::templates::TemplateRegistry::instance().find(id);
            if (!item) return notFound(id);
            if (validate_only)
                return json{{"id", item->id}, {"valid", item->valid}, {"errors", item->errors}};
            return metadataJson(*item);
        });
        return true;
    }
    if (method == "templates.prepare") {
        if (!params.contains("id") || !params["id"].is_string()) {
            out_result = json{{"__error", "missing or invalid string parameter: id"},
                              {"code", "invalid_parameter"}};
            return true;
        }
        const std::string id = params["id"].get<std::string>();
        const std::string policy_name = params.value("conflict_policy", "reject");
        raytrophi::templates::TemplateConflictPolicy policy;
        if (!raytrophi::templates::TemplateLoader::parseConflictPolicy(policy_name, policy)) {
            out_result = json{{"__error", "conflict_policy must be 'reject' or 'discard'"},
                              {"code", "invalid_parameter"}};
            return true;
        }
        out_result = enqueue([id, policy](UIContext&) {
            return loadPlanJson(raytrophi::templates::TemplateLoader::instance().prepare(
                id, ProjectManager::getInstance().hasUnsavedChanges(), policy));
        });
        return true;
    }
    if (method == "templates.open") {
        if (!params.contains("id") || !params["id"].is_string()) {
            out_result = json{{"__error", "missing or invalid string parameter: id"},
                              {"code", "invalid_parameter"}};
            return true;
        }
        const std::string id = params["id"].get<std::string>();
        const std::string policy_name = params.value("conflict_policy", "reject");
        out_result = enqueue([id, policy_name](UIContext&) {
            rtapi::TemplateOpenInfo info;
            (void)rtapi::openTemplate(id, policy_name, info);
            return openInfoJson(info);
        });
        return true;
    }
    return false;
}
