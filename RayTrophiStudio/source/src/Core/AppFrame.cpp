#include "Core/AppFrame.h"

#include "Api/RtApi.h"
#include "Api/RtIpc.h"

#ifdef RT_PLAYER
#include "Core/PlayerVideoCard.h"
#include "Core/RuntimeProjectHost.h"
#endif

#include <SDL.h>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"

#include <thread>

extern SDL_Window* window;
extern SDL_Renderer* renderer;
#ifdef RT_PLAYER
extern SDL_Surface* original_surface;
#endif

namespace AppRuntime {

#ifdef RT_PLAYER
namespace {
PlayerVideoCard video_card;
}
#endif

void FrameCycle::beginIteration() {
    started_ = std::chrono::steady_clock::now();
}

bool FrameCycle::pollEvent(SDL_Event& event, EventInfo& info) {
    if (!AppRuntime::pollEvent(event, info)) {
        return false;
    }
    // IPC wake events do not change the static bootstrap HUD.
    if (event.type < SDL_USEREVENT) {
        requestRedraw();
    }
    return true;
}

void FrameCycle::pumpMainThreadWork(bool scene_ready) {
    // A loader owns scene state until completion. This gate matches Studio's
    // existing dispatch boundary; do not drain queued mutations while loading.
    if (scene_ready) {
        rtapi::drainMainThreadQueue();
    }
    const bool ipc_running = rtipc::isRunning();
    if (ipc_running != observed_ipc_running_) {
        requestRedraw();
        observed_ipc_running_ = ipc_running;
    }
#ifdef RT_PLAYER
    static bool was_enabled = false;
    static std::uint64_t last_packet = 0;
    static Input::BodyPoseRuntime::Clock::time_point final_redraw{};
    const bool enabled = playerBodyOverlayEnabled();
    const auto now = Input::BodyPoseRuntime::Clock::now();
    const auto packet = Input::bodyPoses().packetCount();
    if (enabled != was_enabled || (enabled && packet != last_packet)) {
        requestRedraw();
        final_redraw = enabled ? now + std::chrono::milliseconds(550)
                               : Input::BodyPoseRuntime::Clock::time_point{};
    } else if (enabled && final_redraw != Input::BodyPoseRuntime::Clock::time_point{} &&
               now >= final_redraw) {
        requestRedraw();
        final_redraw = {};
    }
    was_enabled = enabled;
    last_packet = packet;
    if (video_card.poll(playerVideoCardEnabled(), renderer)) {
        requestRedraw();
    }
#endif
}

bool FrameCycle::isForeground() const {
    if (!window) {
        return false;
    }
    const Uint32 flags = SDL_GetWindowFlags(window);
    return (flags & SDL_WINDOW_INPUT_FOCUS) != 0 &&
           (flags & (SDL_WINDOW_MINIMIZED | SDL_WINDOW_HIDDEN)) == 0;
}

bool FrameCycle::shouldPresent(bool continuous) const {
    return isForeground() && (continuous || redraw_requested_);
}

void FrameCycle::requestRedraw() {
    redraw_requested_ = true;
}

void FrameCycle::beginUiFrame(bool draw_runtime_overlays) {
    ImGui_ImplSDLRenderer2_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
#ifdef RT_PLAYER
    if (draw_runtime_overlays && original_surface &&
        (playerBodyOverlayEnabled() || playerVideoCardEnabled())) {
        const PlayerOverlayProjection projection(original_surface->w, original_surface->h);
        video_card.draw(projection);
        if (playerBodyOverlayEnabled()) {
            drawPlayerBodyOverlay(projection);
        }
    }
#else
    (void)draw_runtime_overlays;
#endif
}

void FrameCycle::endUiFrame() {
    ImGui::Render();
}

void FrameCycle::drawUi() {
    ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
}

void FrameCycle::present() {
    SDL_RenderPresent(renderer);
    redraw_requested_ = false;
}

void FrameCycle::waitForNextIteration() const {
    const auto interval = isForeground() ? std::chrono::microseconds(16667)
                                         : std::chrono::microseconds(50000);
    std::this_thread::sleep_until(started_ + interval);
}

} // namespace AppRuntime
