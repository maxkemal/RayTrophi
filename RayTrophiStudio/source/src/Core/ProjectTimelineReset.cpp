#include "ProjectTimelineReset.h"
#include "TimelineWidget.h"
#include "scene_data.h"
#include "globals.h"
#include <atomic>

namespace {
std::atomic<bool> pending_project_timeline_reset{false};
}

void requestProjectTimelineReset() {
    pending_project_timeline_reset.store(true, std::memory_order_release);
}

bool applyPendingProjectTimelineReset(TimelineWidget& widget, SceneData& scene,
                                     RenderSettings& settings) {
    if (!pending_project_timeline_reset.exchange(false, std::memory_order_acq_rel)) {
        return false;
    }
    widget.resetForProjectLoad();
    scene.timeline.current_frame = 0;
    settings.animation_current_frame = 0;
    settings.animation_is_playing = false;
    render_settings.animation_current_frame = 0;
    render_settings.animation_is_playing = false;
    return true;
}
