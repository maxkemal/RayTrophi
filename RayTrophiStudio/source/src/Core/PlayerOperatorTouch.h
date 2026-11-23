#pragma once

#ifdef __ANDROID__
#include <SDL.h>
#include "imgui.h"
#include "UI/PlayerKidLauncher.h"
#include "UI/PlayerOperatorUI.h"

#include <cfloat>
#include <cmath>
#include <optional>

namespace AppRuntime {

// Touch routing. SDL's synthetic mouse events are off (game touches go to the balloon input).
// While the operator console is open (IPC only on Android) every finger is fed to ImGui as a
// touch-screen mouse, with vertical drags turned into wheel scrolling. Otherwise the kid
// launcher gets the touch while one of its screens is up, and during play every touch reaches
// the game untouched.
class PlayerOperatorTouch {
public:
    // Returns true when the event was consumed.
    bool handle(const SDL_Event& event, SDL_Window* window) {
        if (event.type != SDL_FINGERDOWN && event.type != SDL_FINGERMOTION &&
            event.type != SDL_FINGERUP) {
            return false;
        }
        int width = 0;
        int height = 0;
        SDL_GetWindowSize(window, &width, &height);
        const float x = event.tfinger.x * width;
        const float y = event.tfinger.y * height;
        if (!playerConsoleOpen()) {
            return kidLauncherTouch(event, window);
        }
        // Menu open: consume everything.
        auto& io = ImGui::GetIO();
        if (event.type == SDL_FINGERDOWN && !finger_) {
            finger_ = event.tfinger.fingerId;
            scrolling_ = false;
            start_x_ = last_x_ = x;
            start_y_ = last_y_ = y;
            io.AddMouseSourceEvent(ImGuiMouseSource_TouchScreen);
            io.AddMousePosEvent(x, y);
            io.AddMouseButtonEvent(0, true);
            return true;
        }
        if (!finger_ || event.tfinger.fingerId != *finger_) {
            return true;
        }
        io.AddMouseSourceEvent(ImGuiMouseSource_TouchScreen);
        if (event.type == SDL_FINGERMOTION) {
            const float slop = 12.0f * playerOperatorScale() + 8.0f;
            if (!scrolling_ && std::fabs(y - start_y_) > slop &&
                std::fabs(y - start_y_) > 1.5f * std::fabs(x - start_x_)) {
                // A vertical drag is a scroll, not a press: end the press away from any item so
                // it does not click, then scroll the window under the finger.
                scrolling_ = true;
                io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
                io.AddMouseButtonEvent(0, false);
            }
            io.AddMousePosEvent(x, y);
            if (scrolling_) {
                // ImGui scrolls five font heights per wheel step.
                const float step = 5.0f * 16.0f * std::max(1.0f, playerOperatorScale());
                io.AddMouseWheelEvent(0.0f, (y - last_y_) / step);
            }
            last_x_ = x;
            last_y_ = y;
            return true;
        }
        if (event.type == SDL_FINGERUP) {
            io.AddMousePosEvent(x, y);
            if (!scrolling_) {
                io.AddMouseButtonEvent(0, false);
            }
            finger_.reset();
            scrolling_ = false;
            // Move the pointer away so no item stays hovered after the finger lifts.
            io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
        }
        return true;
    }

private:
    std::optional<SDL_FingerID> finger_;
    bool scrolling_ = false;
    float start_x_ = 0, start_y_ = 0, last_x_ = 0, last_y_ = 0;
};

} // namespace AppRuntime
#endif
