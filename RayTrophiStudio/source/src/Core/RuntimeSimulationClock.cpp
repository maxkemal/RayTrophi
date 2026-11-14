#include "Core/RuntimeSimulationClock.h"

#include "Backend/VulkanBackend.h"
#include "Fluid/GasInventoryGpu.h"
#include "scene_data.h"

#include <algorithm>

namespace AppRuntime {

SimulationClock& simulationClock() {
    static SimulationClock clock;
    return clock;
}

SimulationClockResult SimulationClock::tick(SceneData& scene, const SimulationClockInput& input) {
    SimulationClockResult result;

    if (scene.isSimulationBaking()) {
        // A cooperative disk bake owns the sim timeline while it runs. Advance it a
        // time-budgeted slice per tick and keep the loop awake so the next slice
        // fires. Skip the normal live/timeline drive entirely: the bake is the sole
        // stepper, so the two cannot race on sim state.
        ++stats_.ticks;
        scene.tickSimulationDiskBake(20.0 /* ms budget per frame */);
        result.disk_bake_slice = true;
        result.render_updated = true;
        ++stats_.render_updates;
        return result;
    }
    if (input.render_owns_sim || !scene.anySimulationRuntimeEnabled()) {
        return result;
    }
    ++stats_.ticks;

    const float dt = std::clamp(input.dt_seconds, 0.0f, 1.0f / 30.0f);
    // Vulkan RT samples live Gas fields directly from the simulation compute
    // buffers (device addresses, not a copied NanoVDB snapshot). The interactive
    // renderer may have two asynchronous frame slots still reading those buffers.
    // Do not let the next simulation step overwrite them until both RT reads have
    // completed. Queue ordering inside the simulation compute context alone cannot
    // protect a different render queue; the old race eventually presented as a
    // pause-independent TDR during particle-driven explosions.
    const bool play_to_pause_edge = previous_playing_ && !input.playing;
    const int resident_sim_frame = scene.simulationTimelineFrame();
    const bool paused_frame_replace = !input.live_mode && !input.playing &&
        resident_sim_frame >= 0 && input.timeline_frame != resident_sim_frame;
    auto* vulkan_render_backend = dynamic_cast<Backend::VulkanBackendAdapter*>(input.render_backend);
    if ((input.live_mode || input.playing || paused_frame_replace || play_to_pause_edge) &&
        vulkan_render_backend != nullptr) {
        vulkan_render_backend->waitForCompletion();
    }
    previous_playing_ = input.playing;

    // Timeline Start/End only changes the playback window. It must never
    // invalidate a settled melt/fluid state or jump a heavy RT scene to frame zero
    // merely because the generic UI settle gate just closed.
    if (!range_known_) {
        previous_start_ = input.timeline_start;
        previous_end_ = input.timeline_end;
        range_known_ = true;
    } else if (input.timeline_start != previous_start_ || input.timeline_end != previous_end_) {
        scene.preserveSimulationForTimelineRangeEdit();
        previous_start_ = input.timeline_start;
        previous_end_ = input.timeline_end;
    }

    // Timeline mode (Studio default): play bakes into the cache, scrub restores, a
    // stopped timeline stays frozen. Live mode: continuous free-run. ui_editing
    // defers cache invalidation until a dragged widget settles.
    scene.updateSimulationTimeline(input.timeline_frame, input.playing, dt, 24.0f,
                                   input.live_mode, input.ui_editing);
    result.drove = true;
    ++stats_.driven_ticks;
    stats_.driven_seconds += dt;
    stats_.last_dt_seconds = dt;
    stats_.last_live_mode = input.live_mode;
    stats_.last_playing = input.playing;

    result.step_error = RayTrophiSim::Fluid::gasInventoryStepError(scene);
    if (!result.step_error.empty()) {
        result.pause_requested = true;
    }
    stats_.last_step_error = result.step_error;

    // A fluid-affecting edit rewinds the sim to frame 0 instead of auto-resimming up
    // to a high parked frame. The rewind request is generated after the frame-change
    // fence above, so fence again before frame-zero restores mutate RT-visible
    // geometry. Moving the playhead is the caller's (it owns the timeline).
    if (scene.consumeSimRewindRequest()) {
        if (vulkan_render_backend != nullptr) {
            vulkan_render_backend->waitForCompletion();
        }
        scene.deferMeltDisplacementOnce();
        result.rewind_to_start = true;
        ++stats_.rewinds;
    }

    // Keep rendering only while the simulation actually changes (live mode, or a
    // baked/scrubbed frame). A frozen timeline lets the render converge and idle.
    result.render_updated = input.live_mode || scene.simulation_render_updated;
    if (result.render_updated) {
        ++stats_.render_updates;
    }
    return result;
}

} // namespace AppRuntime
