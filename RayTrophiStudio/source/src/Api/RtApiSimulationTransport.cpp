#include "RtSimulationTransport.h"
#include "RtApiInternal.h"
#include "UI/SimulationTransport.h"
#include "globals.h"

#include <stdexcept>

namespace rtapi {
namespace {
void requireTransportContext(bool mutation) {
    if (!g_ctx) {
        throw std::runtime_error("rtapi is not bound to a UIContext");
    }
    if (mutation && (renderJobActive() || rendering_in_progress.load())) {
        throw std::runtime_error("timeline transport is locked by the final render job");
    }
}
}

nlohmann::json timelineTransport() {
    requireTransportContext(false);
    SimCacheStatus cache;
    const auto result = simCacheStatus(cache);
    if (!result.ok) {
        throw std::runtime_error(result.error);
    }
    return {{"playing", ui.timeline.isPlaying()}, {"frame", ui.timeline.getCurrentFrame()},
        {"mode", g_sim_timeline_mode ? "timeline" : "live"},
        {"script_driving", g_ctx->scene.scriptDrivingPhysics()},
        {"cache", {{"valid", cache.valid}, {"baking", cache.baking},
            {"ram_frames", cache.ram_frames}, {"has_range", cache.has_range},
            {"first_frame", cache.first_frame}, {"last_frame", cache.last_frame},
            {"budget_reached", cache.budget_reached}}}};
}

nlohmann::json setTimelinePlaying(bool playing) {
    requireTransportContext(true);
    ui.timeline.setPlayback(*g_ctx, playing);
    return timelineTransport();
}

nlohmann::json setSimulationMode(const std::string& mode) {
    requireTransportContext(true);
    if (mode != "timeline" && mode != "live") {
        throw std::invalid_argument("mode must be timeline or live");
    }
    SimulationTransport::setTimelineMode(*g_ctx, mode == "timeline");
    return timelineTransport();
}
}
