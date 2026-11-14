#include "RtViewportCutoutBindings.h"
#include "Api/RtApiViewportCutout.h"
#include "Api/RtApiInternal.h"
#include "ProjectManager.h"
#include "globals.h"
#include "Backend/IBackend.h"
#include <stdexcept>

namespace rtapi {
void forEachViewportBackend(const std::function<void(Backend::IBackend&)>& fn);
bool viewportAutomaticCutout() { return ::render_settings.viewport_automatic_cutout; }
Result setViewportAutomaticCutout(bool enabled) {
    if (!g_ctx) return Result::fail("Engine context not bound");
    if (::render_settings.viewport_automatic_cutout == enabled) return Result::success();
    ::render_settings.viewport_automatic_cutout = enabled;
    g_materials_dirty = true;
    g_ctx->start_render = true;
    forEachViewportBackend([](Backend::IBackend& b) { b.resetAccumulation(); });
    ProjectManager::getInstance().markModified();
    return Result::success();
}
}

using json = nlohmann::json;
bool dispatchViewportCutoutIpc(const std::string& method, const json& params,
                              const RtIpcTemplateEnqueue& enqueue, json& out) {
    if (method == "viewport.automatic_cutout") {
        out = enqueue([](UIContext&) { return json{{"enabled", rtapi::viewportAutomaticCutout()}}; });
        return true;
    }
    if (method == "viewport.set_automatic_cutout") {
        if (!params.contains("enabled") || !params["enabled"].is_boolean()) {
            out = {{"__error", "enabled must be a boolean"}, {"code", "invalid_parameter"}};
            return true;
        }
        const bool enabled = params["enabled"].get<bool>();
        out = enqueue([enabled](UIContext&) {
            const auto r = rtapi::setViewportAutomaticCutout(enabled);
            return r.ok ? json{{"ok", true}} : json{{"__error", r.error}};
        });
        return true;
    }
    return false;
}
