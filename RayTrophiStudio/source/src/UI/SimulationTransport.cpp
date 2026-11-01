#include "UI/SimulationTransport.h"
#include "scene_ui.h"
#include "scene_data.h"
#include "renderer.h"
#include "globals.h"
#include "Backend/IBackend.h"
#include "Backend/IViewportBackend.h"
#include "Audio/AudioService.h"

#include <algorithm>
#include <chrono>
#include <thread>

extern std::unique_ptr<Backend::IBackend> g_backend;
extern std::unique_ptr<Backend::IViewportBackend> g_viewport_backend;

namespace SimulationTransport {

void drain(UIContext& context) {
    int wait_count = 0;
    while (rendering_in_progress.load() && wait_count < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait_count;
    }
    context.scene.simulation_world.compute().synchronize();
    auto* render = g_backend ? g_backend.get()
        : (dynamic_cast<Backend::IViewportBackend*>(context.backend_ptr) == nullptr
            ? context.backend_ptr : nullptr);
    auto* viewport = g_viewport_backend ? g_viewport_backend.get()
        : dynamic_cast<Backend::IViewportBackend*>(context.backend_ptr);
    if (render) {
        render->waitForCompletion();
    }
    if (viewport && static_cast<Backend::IBackend*>(viewport) != render) {
        viewport->waitForCompletion();
    }
}

void setTimelineMode(UIContext& context, bool timeline_mode) {
    if (g_sim_timeline_mode == timeline_mode) {
        return;
    }
    drain(context);
    context.scene.releaseScriptPhysicsClaim(SceneData::SimDriver::User);
    g_sim_timeline_mode = timeline_mode;
    context.renderer.resetCPUAccumulation();
    if (context.backend_ptr) {
        context.backend_ptr->resetAccumulation();
    }
    context.start_render = true;
}

} // namespace SimulationTransport

void TimelineWidget::setPlayback(UIContext& context, bool playing) {
    if (is_playing == playing) {
        return;
    }
    if (!playing) {
        SimulationTransport::drain(context);
    }
    context.scene.releaseScriptPhysicsClaim(playing
        ? SceneData::SimDriver::Playback : SceneData::SimDriver::User);
    is_playing = playing;
    if (is_playing && !loop_enabled && current_frame >= end_frame) {
        current_frame = start_frame;
    }
    context.render_settings.animation_is_playing = is_playing;
    context.scene.audio->syncTransport(
        static_cast<double>(std::max(0, current_frame)) /
            std::max(1, context.render_settings.animation_fps), is_playing);
}
