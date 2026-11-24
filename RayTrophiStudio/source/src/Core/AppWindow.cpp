#include "Core/AppWindow.h"

#include "globals.h"

#include <SDL.h>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"
#if defined(RT_PLAYER) && defined(__ANDROID__)
#include "Viewport/PlayerSwapchainPresenter.h"
#endif

// Main window state shared by the runtime loop and (in Studio) editor panels.
SDL_Window* window = nullptr;
SDL_Renderer* renderer = nullptr;
SDL_Surface* surface = nullptr;
SDL_Surface* original_surface = nullptr;
SDL_Texture* raytrace_texture = nullptr;

namespace {
void startupLog(AppRuntime::StartupLogFn fn, const std::string& message) {
    if (fn) {
        fn(message);
    }
}
} // namespace

namespace AppRuntime {

bool initPlatform(StartupLogFn startup_log) {
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        SCENE_LOG_ERROR(std::string("SDL_Init Error: ") + SDL_GetError());
        return false;
    }
    startupLog(startup_log, "[Startup] SDL_Init done");
    SDL_EnableScreenSaver();
    startupLog(startup_log, "[Startup] SDL_EnableScreenSaver done");
    return true;
}

void createMainWindow(const WindowConfig& config) {
    // Enable IME UI (must be set before creating window) so SDL can show native IME when needed
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
#ifdef __ANDROID__
    // A resizable SDL window otherwise asks Android for "full user" orientation and the
    // manifest's sensorLandscape is overridden (observed: portrait on the emulator).
    SDL_SetHint(SDL_HINT_ORIENTATIONS, "LandscapeLeft LandscapeRight");
    SDL_SetHintWithPriority(SDL_HINT_TOUCH_MOUSE_EVENTS, "0", SDL_HINT_OVERRIDE);
    // Back is a key event for the game menu instead of closing the activity.
    SDL_SetHint(SDL_HINT_ANDROID_TRAP_BACK_BUTTON, "1");
#endif

#if defined(__ANDROID__) && defined(RT_PLAYER)
    // SDLActivity only enters immersive mode (status and navigation bars hidden, swiped back in
    // transiently) for a fullscreen window: the game owns the whole screen. The Player presents
    // through its own Vulkan swapchain, so the window needs the Vulkan flag as well.
    const Uint32 platform_flags = SDL_WINDOW_FULLSCREEN | SDL_WINDOW_VULKAN;
#elif defined(__ANDROID__)
    const Uint32 platform_flags = SDL_WINDOW_FULLSCREEN;
#elif defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
    const Uint32 platform_flags = SDL_WINDOW_VULKAN;
#else
    const Uint32 platform_flags = 0;
#endif
    startupLog(config.startup_log, "[Startup] before SDL_CreateWindow");
    window = SDL_CreateWindow(config.title,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        config.width,
        config.height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN | platform_flags);
#if defined(RT_PLAYER) && defined(__ANDROID__)
    if (!window) {
        // Leave SDL's window error intact for the startup log and native dialog.
        return;
    }
#endif
    startupLog(config.startup_log,
               std::string("[Startup] SDL_CreateWindow done, ptr=") + (window ? "valid" : "null"));

#ifdef __ANDROID__
    // SDL2 starts text input with the video subsystem; on a touch device that pops the
    // on-screen keyboard over the game (observed on a Lenovo tablet). ImGui's SDL backend
    // starts text input again only while a text field is active.
    SDL_StopTextInput();
#endif
#if !defined(RT_PLAYER) || !(defined(_WIN32) || defined(__ANDROID__))
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    startupLog(config.startup_log,
               std::string("[Startup] SDL_CreateRenderer done, ptr=") + (renderer ? "valid" : "null"));
#endif

    surface = SDL_CreateRGBSurfaceWithFormat(0, config.width, config.height, 32, SDL_PIXELFORMAT_RGBA32);
    original_surface = SDL_CreateRGBSurfaceWithFormat(0, config.width, config.height, 32, SDL_PIXELFORMAT_RGBA32);
    // Prevent uninitialized pixel data causing random black/white on startup
    if (surface) SDL_FillRect(surface, nullptr, SDL_MapRGBA(surface->format, 0, 0, 0, 255));
    if (original_surface) SDL_FillRect(original_surface, nullptr, SDL_MapRGBA(original_surface->format, 0, 0, 0, 255));
    startupLog(config.startup_log, "[Startup] SDL_CreateRGBSurfaceWithFormat done");

#if !defined(RT_PLAYER) || !(defined(_WIN32) || defined(__ANDROID__))
    raytrace_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
                                         SDL_TEXTUREACCESS_STREAMING, config.width, config.height);
    startupLog(config.startup_log,
               std::string("[Startup] SDL_CreateTexture done, ptr=") + (raytrace_texture ? "valid" : "null"));

    SDL_SetTextureBlendMode(raytrace_texture, SDL_BLENDMODE_NONE);
    startupLog(config.startup_log, "[Startup] SDL_SetTextureBlendMode done");
#endif
    // Ensure SDL_TEXTINPUT events are enabled so ImGui_ImplSDL2 can receive UTF-8 characters.
    // Not on Android: there it opens the on-screen keyboard over the game for the whole session.
    // The touch game has no text fields; one that needs typing must start input itself.
#ifndef __ANDROID__
    SDL_StartTextInput();
#endif
}

void initImGui(const ImGuiConfig& config) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    startupLog(config.startup_log, "[Startup] ImGui::CreateContext done");
    ImGuiIO& io = ImGui::GetIO();
    if (config.enable_keyboard_nav) {
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    }

    if (config.font_path && config.font_path[0] != '\0') {
        // Extended Latin (Turkish characters) on top of Basic Latin + Supplement.
        static const ImWchar ext_ranges[] = {
            0x0020, 0x00FF, // Basic Latin + Latin Supplement
            0x0100, 0x017F, // Latin Extended-A
            0
        };
        ImFontConfig font_cfg;
        font_cfg.OversampleH = 2;
        font_cfg.OversampleV = 2;
        io.Fonts->AddFontFromFileTTF(config.font_path, config.font_size, &font_cfg, ext_ranges);
    }

    // Internal docking only - the SDLRenderer2 backend does not support
    // multi-viewport (OS-detached) windows, so ViewportsEnable is never set.
    if (config.enable_docking) {
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    }
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
    ImGui_ImplSDL2_InitForVulkan(window);
#else
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);
#endif
    startupLog(config.startup_log, "[Startup] ImGui SDL init done");
}

bool pollEvent(SDL_Event& event, EventInfo& info) {
    info = EventInfo{};
    if (!SDL_PollEvent(&event)) {
        return false;
    }
    ImGui_ImplSDL2_ProcessEvent(&event);
    if (event.type == SDL_QUIT) {
        info.quit_requested = true;
    }
#if defined(RT_PLAYER) && defined(__ANDROID__)
    // The activity's native window dies while paused; a resumed activity has a
    // new one. The presenter drops its swapchain and recreates the VkSurface.
    if (event.type == SDL_APP_WILLENTERBACKGROUND || event.type == SDL_APP_DIDENTERBACKGROUND) {
        PlayerSwapchainPresenter::instance().onBackground();
    } else if (event.type == SDL_APP_DIDENTERFOREGROUND) {
        PlayerSwapchainPresenter::instance().onForeground();
    }
#endif
    if (event.type == SDL_WINDOWEVENT &&
        event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED) {
        info.focus_gained = true;
    }
    return true;
}

bool recreatePresentationTargets(int w, int h) {
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
    auto* next = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_RGBA32);
    if (!next) {
        return false;
    }
    SDL_FillRect(next, nullptr, SDL_MapRGBA(next->format, 0, 0, 0, 255));
    SDL_FreeSurface(original_surface);
    original_surface = next;
    return true;
#else
    if (raytrace_texture) SDL_DestroyTexture(raytrace_texture);
    if (surface) SDL_FreeSurface(surface);
    if (original_surface) SDL_FreeSurface(original_surface);

    surface = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_RGBA32);
    original_surface = SDL_CreateRGBSurfaceWithFormat(0, w, h, 32, SDL_PIXELFORMAT_RGBA32);
    raytrace_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
                                         SDL_TEXTUREACCESS_STREAMING, w, h);

    if (!surface || !original_surface || !raytrace_texture) {
        SCENE_LOG_ERROR("Failed to create SDL surfaces or texture!");
        return false;
    }

    // Start from a black frame.
    SDL_FillRect(surface, nullptr, SDL_MapRGBA(surface->format, 0, 0, 0, 255));
    SDL_FillRect(original_surface, nullptr, SDL_MapRGBA(original_surface->format, 0, 0, 0, 255));
    SDL_UpdateTexture(raytrace_texture, nullptr, surface->pixels, surface->pitch);
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
    return true;
#endif
}

void destroyMainWindowAndImGui() {
    // The presenter path has no SDL_Renderer and no streaming texture.
    if (raytrace_texture) {
        SDL_DestroyTexture(raytrace_texture);
        raytrace_texture = nullptr;
    }
#if !defined(RT_PLAYER) || !(defined(_WIN32) || defined(__ANDROID__))
    ImGui_ImplSDLRenderer2_Shutdown();
#endif
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_FreeSurface(surface);
    surface = nullptr;
    SDL_FreeSurface(original_surface);
    original_surface = nullptr;
    if (renderer) {
        SDL_DestroyRenderer(renderer);
        renderer = nullptr;
    }
    SDL_DestroyWindow(window);
    window = nullptr;
}

void shutdownPlatform() {
    SDL_Quit();
}

} // namespace AppRuntime
