#include "Core/AppFrame.h"

#include "Api/RtApi.h"
#include "Api/RtIpc.h"

#include <SDL.h>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"

#include <thread>

extern SDL_Window* window;
extern SDL_Renderer* renderer;

namespace AppRuntime {

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

void FrameCycle::beginUiFrame() {
    ImGui_ImplSDLRenderer2_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
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
