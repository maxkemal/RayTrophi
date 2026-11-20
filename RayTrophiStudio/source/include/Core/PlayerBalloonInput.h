#pragma once

union SDL_Event;
struct SDL_Window;

namespace AppRuntime {
void queuePlayerBalloonInput(const SDL_Event& event, SDL_Window* window, bool enabled);
// Returns true when input changed gameplay. Motion is coalesced before dispatch.
bool flushPlayerBalloonInput(bool enabled);
void cancelPlayerBalloonInput();
} // namespace AppRuntime
