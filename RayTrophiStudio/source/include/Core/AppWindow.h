#pragma once

#include <string>

union SDL_Event;  // SDL2: typedef union SDL_Event { ... } SDL_Event;

// Shared runtime window lifecycle (Studio + Player). Owns the SDL platform,
// main window, SDL renderer, CPU presentation surfaces, presentation texture
// and the ImGui context/backends. Studio-only startup (splash, theme, project
// load, OptiX detection) and Studio-only teardown (Python, suspended OptiX
// backend, editor preview textures) stay in the Studio shell and call around
// these steps without reordering them.
//
// The window handles remain the existing globals (window, renderer, surface,
// original_surface, raytrace_texture); they are now defined in AppWindow.cpp.

namespace AppRuntime {

using StartupLogFn = void (*)(const std::string& message);

struct WindowConfig {
    const char* title = "RayTrophi";
    int width = 0;
    int height = 0;
    StartupLogFn startup_log = nullptr;
};

struct ImGuiConfig {
    // Empty path keeps ImGui's built-in default font.
    const char* font_path = nullptr;
    float font_size = 16.0f;
    bool enable_keyboard_nav = true;
    bool enable_docking = true;
    StartupLogFn startup_log = nullptr;
};

// SDL_Init(VIDEO | TIMER) + screen saver policy. Returns false (error logged)
// when SDL cannot start; nothing else has been created in that case.
bool initPlatform(StartupLogFn startup_log);

// Creates the main window hidden, the accelerated SDL renderer, the two RGBA32
// CPU surfaces (cleared to opaque black) and the streaming presentation texture,
// then enables SDL text input. Must follow initPlatform().
void createMainWindow(const WindowConfig& config);

// ImGui context, IO flags, font atlas and SDL2/SDLRenderer2 backends bound to
// the main window. Must follow createMainWindow().
void initImGui(const ImGuiConfig& config);

// Per-event classification produced by pollEvent(). Reset on every call.
struct EventInfo {
    bool quit_requested = false;  // SDL_QUIT: the shell decides (Studio asks to save)
    bool focus_gained = false;    // SDL_WINDOWEVENT_FOCUS_GAINED
};

// One step of the frame's event pump: SDL_PollEvent, then forwards the event to
// the ImGui SDL2 backend and classifies it. Returns false when the queue is
// empty. Every event (including IPC wake-up user events) is returned to the
// caller for shell-specific handling (Studio shortcuts, Player game input).
bool pollEvent(SDL_Event& event, EventInfo& info);

// Recreates the CPU surfaces and the presentation texture at w x h, clears them
// to opaque black and presents that black frame. Returns false (error logged)
// if any resource could not be created. Renderer/backend resizing is the
// caller's job.
bool recreatePresentationTargets(int w, int h);

// Releases, in this order: presentation texture, ImGui backends + context,
// CPU surfaces, SDL renderer, window. Callers release their own renderer-owned
// textures before this call.
void destroyMainWindowAndImGui();

// SDL_Quit. Last SDL call of the process.
void shutdownPlatform();

} // namespace AppRuntime
