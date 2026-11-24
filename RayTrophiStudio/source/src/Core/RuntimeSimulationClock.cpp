#include "Core/RuntimeSimulationClock.h"
#include <sstream>

#include "Backend/VulkanBackend.h"
#include "Fluid/GasInventoryGpu.h"
#include "Paint/WetPaintGpu.h"
#include "Paint/BalloonPaintDemo.h"
#include "Paint/GameDesign.h"
#include "Game/GameRuntime.h"
#include "PerfProfile.h"
#include "scene_data.h"

#include <algorithm>
#include <chrono>

namespace AppRuntime {

SimulationClock& simulationClock() {
    static SimulationClock clock;
    return clock;
}

namespace {

// GPU wet paint (Paint/WetPaintGpu.h) is not a timeline simulation: it runs on
// wall/game time whenever a surface exists, with or without a sim runtime, and
// idles (no dispatch) when nothing is wet and nothing arrives. Returns true when
// the paint changed this tick.
bool tickWetPaint(SceneData& scene, float dt) {
    Paint::WetPaintGpu& paint = Paint::wetPaintGpu();
    if (paint.empty()) return false;
    RTPERF_FRAME_SCOPE("simulation.wet_paint");
    if (!paint.tick(scene, dt)) return false;
    const Paint::WetPaintStats& st = paint.stats();
    return st.stamped_texels > 0 || st.ticked_tiles > 0;
}

} // namespace

SimulationClockResult SimulationClock::tick(SceneData& scene, const SimulationClockInput& input) {
    SimulationClockResult result;
    if (!input.render_owns_sim && !scene.isSimulationBaking()) {
        result.render_updated = Paint::reconcileGameDesign(scene);
    }
    if (Paint::gameDesignPending() && !input.render_owns_sim) {
        // A just-opened project's game design (paint surfaces, demo). Applied
        // here, not in the loader, so it lands on a ready scene in both apps.
        Paint::tickGameDesign(scene);
        result.render_updated = true;
    }

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
    if (input.render_owns_sim) {
        return result;
    }
    const auto demo_frame_start = std::chrono::steady_clock::now();
    if (input.live_mode && !input.ui_editing) {
        Game::tickRuntime(scene, input.dt_seconds);
    }
    if (input.live_mode && !input.ui_editing) {
        RTPERF_FRAME_SCOPE("simulation.balloon_demo");
        Paint::tickBalloonPaintDemo(scene, input.dt_seconds);
    }
    if (!scene.anySimulationRuntimeEnabled()) {
        if (tickWetPaint(scene, std::clamp(input.dt_seconds, 0.0f, 1.0f / 30.0f))) {
            result.render_updated = true;
            ++stats_.render_updates;
        }
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
    // Deferred frame submit: the simulation step leaves its recording open,
    // wet paint below records into the same command buffer, and the tick ends
    // with ONE async submission. The context is re-read after the step because
    // the world may switch its backend during it (CPU <-> Vulkan).
    const bool defer_submit = input.defer_compute_submit;
    if (defer_submit) scene.getSimulationWorld().compute().setDeferFrameSubmit(true);
    {
        RTPERF_FRAME_SCOPE("simulation.timeline");
        scene.updateSimulationTimeline(input.timeline_frame, input.playing, dt, 24.0f,
                                       input.live_mode, input.ui_editing);
    }
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
    // After the particle step, so this frame's contact events are stamped.
    if (tickWetPaint(scene, dt)) {
        result.render_updated = true;
    }
    if (defer_submit) {
        RTPERF_FRAME_SCOPE("simulation.compute_submit");
        auto& compute = scene.getSimulationWorld().compute();
        compute.submitAsync();             // no-op when wet paint already submitted
        compute.setDeferFrameSubmit(false); // never leaks into a non-owner caller
    }
    if (result.render_updated) {
        ++stats_.render_updates;
    }
    Paint::observeBalloonPaintFrame(std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - demo_frame_start).count(), input.dt_seconds);
    return result;
}

} // namespace AppRuntime
