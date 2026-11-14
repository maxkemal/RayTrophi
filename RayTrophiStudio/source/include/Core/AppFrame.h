#pragma once

#include "Core/AppWindow.h"

#include <chrono>

namespace AppRuntime {

// Shared frame operations. Shells retain their existing order and loading gates
// while scene simulation and backend synchronization move out of the editor.
// Studio decides when to draw; Player uses shouldPresent(false) for its static HUD.
class FrameCycle {
public:
    void beginIteration();
    bool pollEvent(SDL_Event& event, EventInfo& info);
    void pumpMainThreadWork(bool scene_ready = true);

    bool isForeground() const;
    bool shouldPresent(bool continuous) const;
    void requestRedraw();

    void beginUiFrame();
    void endUiFrame();
    void drawUi();
    void present();

    // Player bootstrap pacing only. Studio keeps its existing idle/render policy.
    // Sleep covers the remainder of the frame, including time spent in present.
    void waitForNextIteration() const;

private:
    std::chrono::steady_clock::time_point started_ = std::chrono::steady_clock::now();
    bool redraw_requested_ = true;
    bool observed_ipc_running_ = false;
};

} // namespace AppRuntime
