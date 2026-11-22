#pragma once

#ifdef __ANDROID__
#include <SDL.h>
#include "imgui.h"

#include <optional>

namespace AppRuntime {

// Game touches bypass SDL's synthetic mouse events. Route only touches on the
// startup notice to ImGui so the log can be expanded on a phone.
class PlayerStartupFailureTouch {
public:
    bool handle(const SDL_Event& event, SDL_Window* window, bool noticeVisible) {
        if (event.type == SDL_FINGERDOWN && noticeVisible && !finger_) {
            finger_ = event.tfinger.fingerId;
        }
        if (event.type != SDL_FINGERDOWN && event.type != SDL_FINGERMOTION &&
            event.type != SDL_FINGERUP) {
            return false;
        }
        if (!finger_ || event.tfinger.fingerId != *finger_) {
            return noticeVisible;
        }
        int width = 0;
        int height = 0;
        SDL_GetWindowSize(window, &width, &height);
        auto& io = ImGui::GetIO();
        io.AddMouseSourceEvent(ImGuiMouseSource_TouchScreen);
        io.AddMousePosEvent(event.tfinger.x * width, event.tfinger.y * height);
        if (event.type == SDL_FINGERDOWN) {
            io.AddMouseButtonEvent(0, true);
        } else if (event.type == SDL_FINGERUP) {
            io.AddMouseButtonEvent(0, false);
            finger_.reset();
        }
        return true;
    }

private:
    std::optional<SDL_FingerID> finger_;
};

} // namespace AppRuntime
#endif
