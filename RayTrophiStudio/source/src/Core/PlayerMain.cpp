// RayTrophi Player entry point: static .rtp scene viewer.
//
// PLAYER BUILD ONLY. Studio has its own entry point in Main.cpp and lists this
// file as None; both define main() and must never be linked together.
//
// Shared AppRuntime frame operations, a Player-owned scene, the existing rtapi
// project.open facade and a Vulkan static scene viewer. Both IPC and drag/drop
// use the same loader; VulkanSceneSync is shared with Studio. Animation/simulation
// playback and editor-derived systems are later slices; see the Player plan.
//
// The Player's scene and renderer use their own names (no `scene`,
// `ray_renderer` globals): Studio-only globals that shared code still reaches
// for must surface as link errors, not be silently shadowed here. The one
// exception is `ui`: shared API code keeps runtime state on the SceneUI object
// (timeline cursor, shading mode, undo history), so the Player owns a
// never-drawn SceneUI as a state carrier. See UI/SceneUIStateHost.cpp.

#include "Core/AppWindow.h"
#include "Core/AppFrame.h"
#include "Core/RuntimeSceneViewer.h"

#include "globals.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "Renderer.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "SceneHistory.h"
#include "UIContext.h"
#include "Api/RtApi.h"
#include "Api/RtIpc.h"

#include <SDL.h>
#include <SDL_main.h>
#include "imgui.h"

#include <filesystem>
#include <memory>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

extern SDL_Window* window;
extern SDL_Renderer* renderer;
extern SDL_Texture* raytrace_texture;
extern SDL_Surface* original_surface;

// Shared runtime state (Core/RuntimeState.cpp) and the Player's SceneUI state
// carrier (UI/SceneUIStateHost.cpp).
extern ColorProcessor color_processor;
extern std::string active_model_path;
extern SceneUI ui;

namespace {

struct PlayerState {
    SceneData scene;
    Renderer renderer3d{image_width, image_height, 1, 1};
    SceneSelection selection;
    int sample_count = 1;
    bool start_render = false;
    bool apply_tonemap = false;
    bool reset_tonemap = false;
    bool mouse_control_enabled = true;

    UIContext ctx{
        scene,
        renderer3d,
        nullptr,  // optix_gpu_ptr: no OptiX in the Player
        nullptr,  // backend_ptr: assigned lazily by RuntimeSceneViewer
        ::color_processor,
        render_settings,
        selection,
        &ui,      // scene_ui_ptr: state carrier only, never drawn
        sample_count,
        start_render,
        ::active_model_path,
        apply_tonemap,
        reset_tonemap,
        mouse_control_enabled,
    };
};

void setWorkingDirectoryToExe(char* argv0) {
#ifdef _WIN32
    (void)argv0;
    char exe_path[MAX_PATH];
    GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
    std::filesystem::current_path(std::filesystem::path(exe_path).parent_path());
#else
    if (argv0) {
        std::filesystem::current_path(std::filesystem::path(argv0).parent_path());
    }
#endif
}

void drawHud(bool ipc_running, const AppRuntime::RuntimeSceneViewer& viewer) {
    ImGui::SetNextWindowPos(ImVec2(12.0f, 12.0f), ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha(0.6f);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoNav;
    if (ImGui::Begin("##player_hud", nullptr, flags)) {
        ImGui::TextUnformatted("RayTrophi Player");
        ImGui::Text("IPC: %s", ipc_running ? "running" : "stopped");
        const auto status = viewer.status();
        if (viewer.hasScene()) {
            const auto path = AppRuntime::projectPathFromUtf8(status.value("project_path", ""));
            const auto name = AppRuntime::projectPathToUtf8(path.filename());
            ImGui::TextUnformatted(name.c_str());
        } else {
            ImGui::TextUnformatted("Drop an .rtp project here");
        }
        const auto error = status.value("error", "");
        const auto open_error = status.value("last_open_error", "");
        if (!error.empty() || !open_error.empty()) {
            ImGui::TextWrapped("%s", (error.empty() ? open_error : error).c_str());
        }
    }
    ImGui::End();
}

} // namespace

int main(int argc, char* argv[]) {
    // Resolve a launch project before changing the working directory to the exe.
    std::string launch_project;
    if (argc == 2) {
        launch_project = AppRuntime::projectPathToUtf8(
            std::filesystem::absolute(AppRuntime::projectPathFromUtf8(argv[1])));
    } else if (argc == 3 && std::string(argv[1]) == "--project") {
        launch_project = AppRuntime::projectPathToUtf8(
            std::filesystem::absolute(AppRuntime::projectPathFromUtf8(argv[2])));
    }
    g_main_thread_id = std::this_thread::get_id();
    setWorkingDirectoryToExe(argc > 0 ? argv[0] : nullptr);
    g_sceneLog.initLogLocation();

    if (!AppRuntime::initPlatform(nullptr)) {
        return 1;
    }

    AppRuntime::WindowConfig window_config;
    window_config.title = "RayTrophi Player";
    window_config.width = image_width;
    window_config.height = image_height;
    AppRuntime::createMainWindow(window_config);

    AppRuntime::ImGuiConfig imgui_config;
    imgui_config.font_path = "C:\\Windows\\Fonts\\segoeui.ttf";
    imgui_config.font_size = 16.0f;
    imgui_config.enable_keyboard_nav = true;
    imgui_config.enable_docking = false;  // game HUD, no editor layout
    AppRuntime::initImGui(imgui_config);

    // The viewer renders flat meshes through Vulkan. Embree and CPU authoring
    // queries are not part of this Player slice.
    use_embree = false;
    render_settings.UI_use_embree = false;

    auto state = std::make_unique<PlayerState>();
    state->ctx.ray_texture = raytrace_texture;
    // Same wiring as Studio: undo history lives on the SceneUI state object.
    rtapi::bind(&state->ctx, &ui.history);
    ui.viewport_settings.shading_mode = 1;  // MaterialPreview, same Vulkan path as Studio.
    auto viewer = std::make_unique<AppRuntime::RuntimeSceneViewer>(state->ctx, ui.history);
    if (!launch_project.empty()) {
        const auto loaded = rtapi::openProject(launch_project);
        if (!loaded.ok) {
            SCENE_LOG_WARN("[Player] " + loaded.error);
        }
    }

    std::string ipc_error;
    if (rtipc::start(ipc_error)) {
        SCENE_LOG_INFO("[Player] IPC server started");
    } else {
        SCENE_LOG_WARN("[Player] IPC server failed to start: " + ipc_error);
    }

    SDL_ShowWindow(window);

    bool running = true;
    Uint64 previous_tick = SDL_GetPerformanceCounter();
    AppRuntime::FrameCycle frame_cycle;
    SDL_Event event;
    AppRuntime::EventInfo event_info;
    while (running) {
        frame_cycle.beginIteration();
        while (frame_cycle.pollEvent(event, event_info)) {
            if (event_info.quit_requested) {
                running = false;  // Player: no save prompt
            }
            if (event.type == SDL_DROPFILE) {
                const std::string path = event.drop.file ? event.drop.file : "";
                SDL_free(event.drop.file);
                const auto loaded = rtapi::openProject(path);
                if (!loaded.ok) {
                    SCENE_LOG_WARN("[Player] " + loaded.error);
                }
                frame_cycle.requestRedraw();
            }
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                int width = 0;
                int height = 0;
                if (SDL_GetRendererOutputSize(renderer, &width, &height) == 0 &&
                    width > 0 && height > 0 &&
                    AppRuntime::recreatePresentationTargets(width, height)) {
                    state->ctx.ray_texture = raytrace_texture;
                    viewer->requestFrame();
                    frame_cycle.requestRedraw();
                }
            }
        }

        if (!running) {
            break;
        }
        frame_cycle.pumpMainThreadWork();

        // Game clock: the shared simulation clock advances on wall time whether
        // or not the window is focused (a kiosk keeps playing); rendering below
        // stays foreground/capture gated.
        const Uint64 now_tick = SDL_GetPerformanceCounter();
        const float dt_seconds = static_cast<float>(
            static_cast<double>(now_tick - previous_tick) /
            static_cast<double>(SDL_GetPerformanceFrequency()));
        previous_tick = now_tick;
        viewer->advanceRuntime(dt_seconds);

        // Explicit capture can render while unfocused; ordinary background windows
        // stay idle. A settled static scene stops submitting Vulkan frames too.
        const bool render_requested = viewer->needsFrame();
        const bool capture_requested = rtapi::viewportCaptureEnabled();
        bool image_updated = false;
        if (render_requested && (frame_cycle.isForeground() || capture_requested)) {
            image_updated = viewer->render(original_surface, raytrace_texture);
            frame_cycle.requestRedraw();
        }
        if (frame_cycle.shouldPresent(image_updated)) {
            frame_cycle.beginUiFrame();
            drawHud(rtipc::isRunning(), *viewer);
            frame_cycle.endUiFrame();

            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
            if (viewer->hasFrame()) {
                SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
            }
            frame_cycle.drawUi();
            frame_cycle.present();
        }
        frame_cycle.waitForNextIteration();
    }

    // Same order as Studio: IPC stops before the facade it enqueues into.
    rtipc::stop();
    viewer.reset();
    rtapi::unbind();
    state.reset();

    AppRuntime::destroyMainWindowAndImGui();
    AppRuntime::shutdownPlatform();
    g_sceneLog.closeLogFile();
    return 0;
}
