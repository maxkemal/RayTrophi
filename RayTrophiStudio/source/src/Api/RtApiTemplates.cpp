#include "Api/RtApi.h"
#include "RtApiInternal.h"

#include "Template/TemplateSession.h"
#include "Template/UserTemplateManager.h"

namespace rtapi {

Result openTemplate(const std::string& id, const std::string& conflict_policy,
                    TemplateOpenInfo& out) {
    out = {};
    out.template_id = id;
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");

    const auto opened = raytrophi::templates::TemplateSession::instance().open(
        id, conflict_policy, *g_ctx, ui, g_history);
    out.template_id = opened.template_id;
    out.state = opened.state;
    out.code = opened.code;
    out.opened = opened.opened;
    out.ui_state_applied = opened.ui_state_applied;
    out.errors = opened.errors;
    out.warnings = opened.warnings;
    if (!opened.opened) {
        return Result::fail(opened.errors.empty() ? opened.code : opened.errors.front());
    }
    notifySceneLoaded();
    return Result::success();
}

Result saveUserTemplate(const std::string& display_name,
                        const std::string& description,
                        const std::string& category) {
    if (!g_ctx) return notBound();
    if (renderJobActive()) return Result::fail("scene is locked by the final render job");
    if (display_name.empty()) return Result::fail("display_name cannot be empty");

    bool saved = raytrophi::templates::UserTemplateManager::instance().saveCurrentSceneAsTemplate(
        display_name, description, category, *g_ctx, ui, g_history);
    if (!saved) {
        return Result::fail("failed to save user template");
    }
    return Result::success();
}

Result deleteUserTemplate(const std::string& template_id) {
    if (template_id.empty()) return Result::fail("template_id cannot be empty");

    bool deleted = raytrophi::templates::UserTemplateManager::instance().deleteUserTemplate(template_id);
    if (!deleted) {
        return Result::fail("failed to delete user template or template not found");
    }
    return Result::success();
}

} // namespace rtapi
