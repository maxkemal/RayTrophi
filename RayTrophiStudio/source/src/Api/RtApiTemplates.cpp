#include "Api/RtApi.h"
#include "RtApiInternal.h"

#include "Template/TemplateSession.h"

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

} // namespace rtapi
