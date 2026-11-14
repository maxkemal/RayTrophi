#pragma once

#include "RtApiInternal.h"
#include "scene_ui.h"

namespace rtapi {

// Manual fluid and rigid steps advance the same timeline claim. Otherwise the
// next paused viewport tick can restore frame zero over the completed step.
inline void publishManualSimulationStep(UIContext& context, float dt) {
    // Keep this rate aligned with updateSimulationTimeline's UI caller.
    const int frame = context.scene.claimPhysicsStep(
        dt, 24.0f, ui.timeline.getCurrentFrame());
    ui.timeline.setCurrentFrame(frame);
    context.scene.timeline.current_frame = frame;
    context.render_settings.animation_current_frame = frame;
    context.render_settings.animation_playback_frame = frame;
}

} // namespace rtapi
