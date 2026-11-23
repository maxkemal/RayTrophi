#pragma once
// Android kid launcher: the full-screen game library, loading view and pause screen of the
// consumer Player. It is a VIEW over the operator service (rtapi::playerOperator): the game
// list, load progress and audio state all come from it, and the launcher.* operations expose
// the same screen state as values so scripts and tests drive what a finger drives.
#include "json.hpp"
#include <SDL.h>
#include <string>

namespace AppRuntime {
#ifdef __ANDROID__
// True while any launcher screen (or its fade into the game) owns the display and input.
bool kidLauncherActive();
// True while the game clock should stand still (pause screen, library).
bool kidLauncherPaused();
void kidLauncherDraw(SDL_Renderer *renderer);
// Finger input; returns true when the event was consumed.
bool kidLauncherTouch(const SDL_Event &event, SDL_Window *window);
// App lifecycle: pauses a running game when the app loses the foreground.
void kidLauncherEvent(const SDL_Event &event);
// Android Back gesture/button: play <-> pause; ignored on the library and loading screens.
void kidLauncherBack();
// Select the packaged scene once the asynchronous library scan exposes its key.
void kidLauncherAutoOpen(const std::string &scene_key);
// Values for IPC: screen = library|loading|pause|playing.
bool kidLauncherShow(const std::string &screen, std::string &error);
bool kidLauncherSelect(int index, std::string &error);
// Pushes SDL_QUIT: the Player loop ends and SDL finishes the Android activity.
void kidLauncherExit();
nlohmann::json kidLauncherStatus();
#endif
} // namespace AppRuntime
