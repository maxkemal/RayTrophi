#include "Api/DemBondApi.h"
#include "RtApiInternal.h"
#include <stdexcept>

namespace rtapi {
nlohmann::json demBodies(const std::string& domain, const nlohmann::json& request) {
    if (!g_ctx) {
        throw std::runtime_error("rtapi is not bound to a UIContext");
    }
    const std::string action = request.value("action", "inspect");
    const bool mutation = action == "bind" || action == "remove";
    if (mutation && (renderJobActive() || ui.timeline.isPlaying())) {
        throw std::runtime_error("pause simulation and finish the render before editing DEM bodies");
    }
    auto result = scriptSimulationRuntime().demBondOperation(domain, request);
    if (mutation) {
        // Preserve current particles and the explicit body edit; drop future
        // cache frames, rather than resetting the runtime back to loose grains.
        try {
            g_ctx->scene.invalidateSimCacheAfterCurrentFrame("DEM body authoring");
            const int frame = g_ctx->scene.simulationTimelineFrame();
            result["checkpoint"] = frame >= 0 && g_ctx->scene.captureSimFrame(frame, true);
        } catch (const std::exception& error) {
            // Body editing succeeded. A cache error must not be misreported
            // as a transactional authoring rejection after state was committed.
            result["checkpoint"] = false;
            result["checkpoint_error"] = error.what();
        }
    }
    return result;
}
}
