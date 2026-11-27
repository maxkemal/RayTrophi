#include "Core/PlayerBalloonInput.h"
#include "Api/RtApi.h"
#include "Api/RtApiPaintWet.h"
#include "Paint/BalloonInteraction.h"
#include "json.hpp"
#include <SDL.h>
#include <algorithm>
#ifdef __ANDROID__
#include <cmath>
#include <optional>
#endif
#include <string>
#include <utility>
#include <vector>

namespace AppRuntime {
namespace {
using json = nlohmann::json;
std::vector<json> pending;
std::string heldOwner;

void setHeldOwner(std::string owner) {
    const bool hadMouse = heldOwner == "mouse";
    const bool hasMouse = owner == "mouse";
    if (hadMouse != hasMouse) {
        SDL_CaptureMouse(hasMouse ? SDL_TRUE : SDL_FALSE);
    }
    heldOwner = std::move(owner);
}

bool dispatch(const json& input) {
    const auto owner = input.value("owner", std::string("mouse"));
    std::string output;
    const auto result = rtapi::wetPaintDemoCommand("pointer", input.dump(), output);
    if (!result.ok) {
        if (!heldOwner.empty()) {
            std::string ignored;
            const json cancel = {{"action", "cancel"}, {"owner", heldOwner}};
            (void)rtapi::wetPaintDemoCommand("pointer", cancel.dump(), ignored);
            setHeldOwner({});
        }
        return false;
    }
    const auto state = json::parse(output, nullptr, false);
    const bool held = state.is_object() && state.value("held", -1) >= 0 &&
                      state.value("owner", std::string()) == owner;
    setHeldOwner(held ? owner : std::string());
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

#ifdef __ANDROID__
std::optional<SDL_TouchFingerEvent> primaryFinger;
std::optional<SDL_TouchFingerEvent> pinchFinger;

bool matchesFinger(const SDL_TouchFingerEvent& event,
                   const std::optional<SDL_TouchFingerEvent>& finger) {
    return finger && event.touchId == finger->touchId && event.fingerId == finger->fingerId;
}

std::string fingerOwner(const SDL_TouchFingerEvent& finger) {
    return "finger:" + std::to_string(finger.fingerId);
}

float pinchDistance(float aspect) {
    return std::hypot((primaryFinger->x - pinchFinger->x) * aspect,
                      primaryFinger->y - pinchFinger->y);
}

void queueFingerInput(const SDL_TouchFingerEvent& event, float aspect) {
    const auto screen = json::array({std::clamp(event.x, 0.0f, 1.0f),
                                     std::clamp(event.y, 0.0f, 1.0f)});
    if (event.type == SDL_FINGERDOWN) {
        if (!primaryFinger && !pinchFinger) {
            primaryFinger = event;
            enqueue({{"action", "down"}, {"screen", screen}, {"owner", fingerOwner(event)}});
        } else if (primaryFinger && !pinchFinger && event.touchId == primaryFinger->touchId &&
                   !matchesFinger(event, primaryFinger)) {
            pinchFinger = event;
        }
        return;
    }
    const bool isPrimary = matchesFinger(event, primaryFinger);
    const bool isPinch = matchesFinger(event, pinchFinger);
    if (event.type == SDL_FINGERUP) {
        if (isPrimary) {
            enqueue({{"action", "up"}, {"screen", screen}, {"owner", fingerOwner(event)}});
            primaryFinger.reset();
            pinchFinger.reset();
        } else if (isPinch) {
            pinchFinger.reset();
        }
        return;
    }
    if (!isPrimary && !isPinch) {
        return;
    }
    const bool pinching = primaryFinger && pinchFinger;
    const float previousDistance = pinching ? pinchDistance(aspect) : 0.0f;
    if (isPrimary) {
        primaryFinger = event;
        enqueue({{"action", "move"}, {"screen", screen}, {"owner", fingerOwner(event)}});
    } else {
        pinchFinger = event;
    }
    if (pinching) {
        // Closing the gap moves the held balloon deeper, like scrolling up.
        constexpr float wheelPerScreenHeight = 10.0f;
        const float wheel = (previousDistance - pinchDistance(aspect)) * wheelPerScreenHeight;
        if (wheel != 0.0f) {
            enqueue({{"action", "wheel"}, {"owner", fingerOwner(*primaryFinger)},
                     {"wheel", std::clamp(wheel, -100.0f, 100.0f)}});
        }
    }
}
#endif
} // namespace

void queuePlayerBalloonInput(const SDL_Event& event, SDL_Window* window, bool enabled) {
    if ((event.type == SDL_WINDOWEVENT &&
         (event.window.event == SDL_WINDOWEVENT_FOCUS_LOST ||
          event.window.event == SDL_WINDOWEVENT_LEAVE)) ||
        event.type == SDL_QUIT ||
#ifdef __ANDROID__
        event.type == SDL_APP_WILLENTERBACKGROUND ||
#endif
        (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)) {
        // Leaving a captured window is allowed; loss of focus is not.
        if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_LEAVE &&
            !heldOwner.empty()) {
            return;
        }
        pending.clear();
#ifdef __ANDROID__
        primaryFinger.reset();
        pinchFinger.reset();
#endif
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
#ifdef __ANDROID__
    if (event.type == SDL_FINGERDOWN || event.type == SDL_FINGERMOTION ||
        event.type == SDL_FINGERUP) {
        if (event.tfinger.touchId != SDL_MOUSE_TOUCHID) {
            queueFingerInput(event.tfinger, static_cast<float>(width) / height);
        }
        return;
    }
#endif
    const auto point = [&](int x, int y) {
        return json::array({std::clamp(static_cast<float>(x) / width, 0.0f, 1.0f),
                            std::clamp(static_cast<float>(y) / height, 0.0f, 1.0f)});
    };
    if ((event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEBUTTONUP) &&
        event.button.button == SDL_BUTTON_LEFT) {
        if (event.type == SDL_MOUSEBUTTONUP && (SDL_GetModState() & KMOD_SHIFT)) {
            enqueue({{"action", "cancel"}});
        } else {
            enqueue({{"action", event.type == SDL_MOUSEBUTTONDOWN ? "down" : "up"},
                     {"screen", point(event.button.x, event.button.y)}});
        }
    } else if (event.type == SDL_MOUSEMOTION && (event.motion.state & SDL_BUTTON_LMASK)) {
        enqueue({{"action", "move"}, {"screen", point(event.motion.x, event.motion.y)}});
    } else if (event.type == SDL_MOUSEWHEEL) {
        const int direction = event.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -1 : 1;
        enqueue({{"action", "wheel"}, {"wheel", std::clamp(event.wheel.y * direction, -100, 100)}});
    }
}

bool flushPlayerBalloonInput(bool enabled) {
    if (!enabled) {
        const bool changed = !heldOwner.empty();
        cancelPlayerBalloonInput();
        return changed;
    }
    if (!heldOwner.empty() &&
        (Paint::heldBalloon() < 0 ||
         Paint::balloonInteractionStatus().value("owner", std::string()) != heldOwner)) {
        setHeldOwner({});
    }
    bool changed = false;
    auto events = std::move(pending);
    pending.clear();
    for (const auto& event : events) {
        const auto owner = event.value("owner", std::string("mouse"));
        if (event["action"] == "cancel" && !heldOwner.empty()) {
            changed = dispatch({{"action", "cancel"}, {"owner", heldOwner}}) || changed;
        } else if ((heldOwner.empty() && (event["action"] == "down" || event["action"] == "cancel")) ||
                   (!heldOwner.empty() && owner == heldOwner)) {
            changed = dispatch(event) || changed;
        }
    }
    return changed;
}

void cancelPlayerBalloonInput() {
    pending.clear();
#ifdef __ANDROID__
    primaryFinger.reset();
    pinchFinger.reset();
#endif
    if (!heldOwner.empty()) {
        (void)dispatch({{"action", "cancel"}, {"owner", heldOwner}});
        setHeldOwner({});
    }
}
} // namespace AppRuntime
