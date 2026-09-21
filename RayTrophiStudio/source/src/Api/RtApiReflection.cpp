#include "Api/RtApiReflection.h"
#include "Backend/IBackend.h"
#include <functional>
namespace rtapi {
void forEachViewportBackend(const std::function<void(Backend::IBackend&)>& fn);
RayFusion::ReflectionStatus reflectionStatus() {
    RayFusion::ReflectionStatus out;
    forEachViewportBackend([&](Backend::IBackend& b) {
        const auto s = b.reflectionStatus();
        if (!out.supported && s.supported) out = s;
    });
    if (!out.supported) out.reason = "no hardware ray-query reflection backend";
    return out;
}
bool setReflection(const RayFusion::ReflectionSettings& settings, std::string& error) {
    if (!RayFusion::validateReflection(settings, error)) return false;
    bool applied = false;
    forEachViewportBackend([&](Backend::IBackend& b) {
        if (applied || !b.reflectionStatus().supported) return;
        applied = b.setReflection(settings, error);
    });
    if (!applied && error.empty()) error = "no hardware ray-query reflection backend";
    return applied;
}
}
