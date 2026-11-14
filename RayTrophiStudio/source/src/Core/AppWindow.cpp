#include "Core/AppWindow.h"

#include "globals.h"

#include <SDL.h>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"

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

    startupLog(config.startup_log, "[Startup] before SDL_CreateWindow");
    window = SDL_CreateWindow(config.title,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        config.width,
        config.height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIDDEN);
    startupLog(config.startup_log,
               std::string("[Startup] SDL_CreateWindow done, ptr=") + (window ? "valid" : "null"));

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    startupLog(config.startup_log,
               std::string("[Startup] SDL_CreateRenderer done, ptr=") + (renderer ? "valid" : "null"));

    surface = SDL_CreateRGBSurfaceWithFormat(0, config.width, config.height, 32, SDL_PIXELFORMAT_RGBA32);
    original_surface = SDL_CreateRGBSurfaceWithFormat(0, config.width, config.height, 32, SDL_PIXELFORMAT_RGBA32);
    // Prevent uninitialized pixel data causing random black/white on startup
    if (surface) SDL_FillRect(surface, nullptr, SDL_MapRGBA(surface->format, 0, 0, 0, 255));
    if (original_surface) SDL_FillRect(original_surface, nullptr, SDL_MapRGBA(original_surface->format, 0, 0, 0, 255));
    startupLog(config.startup_log, "[Startup] SDL_CreateRGBSurfaceWithFormat done");

    raytrace_texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
                                         SDL_TEXTUREACCESS_STREAMING, config.width, config.height);
    startupLog(config.startup_log,
               std::string("[Startup] SDL_CreateTexture done, ptr=") + (raytrace_texture ? "valid" : "null"));

    SDL_SetTextureBlendMode(raytrace_texture, SDL_BLENDMODE_NONE);
    startupLog(config.startup_log, "[Startup] SDL_SetTextureBlendMode done");
    // Ensure SDL_TEXTINPUT events are enabled so ImGui_ImplSDL2 can receive UTF-8 characters
    SDL_StartTextInput();
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
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);
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
    if (event.type == SDL_WINDOWEVENT &&
        event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED) {
        info.focus_gained = true;
    }
    return true;
}

bool recreatePresentationTargets(int w, int h) {
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
}

void destroyMainWindowAndImGui() {
    SDL_DestroyTexture(raytrace_texture);
    raytrace_texture = nullptr;
    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_FreeSurface(surface);
    surface = nullptr;
    SDL_FreeSurface(original_surface);
    original_surface = nullptr;
    SDL_DestroyRenderer(renderer);
    renderer = nullptr;
    SDL_DestroyWindow(window);
    window = nullptr;
}

void shutdownPlatform() {
    SDL_Quit();
}

} // namespace AppRuntime
