#include "Api/RtApiScreenGi.h"
#include "Backend/IBackend.h"
#include <functional>
namespace rtapi {
void forEachViewportBackend(const std::function<void(Backend::IBackend&)>& fn);
RayFusion::ScreenGiStatus screenGiStatus() {
    RayFusion::ScreenGiStatus out;
    forEachViewportBackend([&](Backend::IBackend& b) {
        const auto s = b.screenGiStatus();
        if (!out.supported && s.supported) out = s;
    });
    if (!out.supported) out.reason = "no hardware ray-query screen GI backend";
    return out;
}
bool setScreenGi(const RayFusion::ScreenGiSettings& settings, std::string& error) {
    if (!RayFusion::validateScreenGi(settings, error)) return false;
    bool applied = false;
    forEachViewportBackend([&](Backend::IBackend& b) {
        if (applied || !b.screenGiStatus().supported) return;
        applied = b.setScreenGi(settings, error);
    });
    if (!applied && error.empty()) error = "no hardware ray-query screen GI backend";
    // Viewport'u `VulkanBackendAdapter::setScreenGi` zaten backend icinde
    // dirty ediyor -- dogru kol orasi. Burada ikinci bir "yeniden ciz"
    // bayragi cekmek gereksizdi ve yol izleyiciyi suruyordu; bkz.
    // RtApiRayFusion.cpp.
    return applied;
}
}
