#include "Core/PlayerBalloonInput.h"
#include "Api/RtApi.h"
#include "Api/RtApiPaintWet.h"
#include "Paint/BalloonInteraction.h"
#include "json.hpp"
#include <SDL.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace AppRuntime {
namespace {
using json = nlohmann::json;
std::vector<json> pending;
bool captured = false;

bool dispatch(const json& input) {
    std::string output;
    const auto result = rtapi::wetPaintDemoCommand("pointer", input.dump(), output);
    if (!result.ok) {
        if (captured) {
            std::string ignored;
            (void)rtapi::wetPaintDemoCommand("pointer", "{\"action\":\"cancel\"}", ignored);
            SDL_CaptureMouse(SDL_FALSE);
            captured = false;
        }
        return false;
    }
    const auto state = json::parse(output, nullptr, false);
    const bool held = state.is_object() && state.value("held", -1) >= 0 &&
                      state.value("owner", std::string()) == "mouse";
    if (held != captured) {
        SDL_CaptureMouse(held ? SDL_TRUE : SDL_FALSE);
        captured = held;
    }
    return true;
}

void enqueue(json value) {
    if (value["action"] == "move" && !pending.empty() && pending.back()["action"] == "move") {
        pending.back() = std::move(value);
    } else if (pending.size() < 64) {
        pending.push_back(std::move(value));
    } else {
        pending.clear();
        pending.push_back({{"action", "cancel"}});
    }
}
} // namespace

void queuePlayerBalloonInput(const SDL_Event& event, SDL_Window* window, bool enabled) {
    if ((event.type == SDL_WINDOWEVENT &&
         (event.window.event == SDL_WINDOWEVENT_FOCUS_LOST ||
          event.window.event == SDL_WINDOWEVENT_LEAVE)) ||
        event.type == SDL_QUIT ||
        (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
        // Leaving a captured window is allowed; loss of focus is not.
        if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_LEAVE &&
            captured) {
            return;
        }
        pending.clear();
        enqueue({{"action", "cancel"}});
        return;
    }
    if (!enabled || !window) {
        return;
    }
    int width = 0, height = 0;
    SDL_GetWindowSize(window, &width, &height);
    if (width <= 0 || height <= 0) {
        return;
    }
    const auto point = [&](int x, int y) {
        return json::array({std::clamp(static_cast<float>(x) / width, 0.0f, 1.0f),
                            std::clamp(static_cast<float>(y) / height, 0.0f, 1.0f)});
    };
    if ((event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEBUTTONUP) &&
        event.button.button == SDL_BUTTON_LEFT) {
        enqueue({{"action", event.type == SDL_MOUSEBUTTONDOWN ? "down" : "up"},
                 {"screen", point(event.button.x, event.button.y)}});
    } else if (event.type == SDL_MOUSEMOTION && (event.motion.state & SDL_BUTTON_LMASK)) {
        enqueue({{"action", "move"}, {"screen", point(event.motion.x, event.motion.y)}});
    } else if (event.type == SDL_MOUSEWHEEL) {
        const int direction = event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -1 : 1;
        enqueue({{"action", "wheel"}, {"wheel", std::clamp(event.wheel.y * direction, -100, 100)}});
    }
}

bool flushPlayerBalloonInput(bool enabled) {
    if (!enabled) {
        const bool changed = captured;
        cancelPlayerBalloonInput();
        return changed;
    }
    if (captured && (Paint::heldBalloon() < 0 ||
                     Paint::balloonInteractionStatus().value("owner", std::string()) != "mouse")) {
        SDL_CaptureMouse(SDL_FALSE);
        captured = false;
    }
    bool changed = false;
    auto events = std::move(pending);
    pending.clear();
    for (const auto& event : events) {
        if (event["action"] == "down" || event["action"] == "cancel" || captured) {
            changed = dispatch(event) || changed;
        }
    }
    return changed;
}

void cancelPlayerBalloonInput() {
    pending.clear();
    if (captured) {
        (void)dispatch({{"action", "cancel"}});
        SDL_CaptureMouse(SDL_FALSE);
        captured = false;
    }
}
} // namespace AppRuntime
