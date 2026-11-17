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
#include "Core/RuntimeProjectHost.h"
#include "Core/PlayerRenderExtent.h"

#include "globals.h"
#include "scene_data.h"
#include "scene_ui.h"
#include "Renderer.h"
#include "ColorProcessingParams.h"
#include "SceneSelection.h"
#include "SceneHistory.h"
#include "UIContext.h"
#include <cmath>
#include "Api/RtApi.h"
#include "Api/RtApiPaintWet.h"
#include "json.hpp"
#include "Api/RtIpc.h"

#include <SDL.h>
#include <SDL_main.h>
#include "imgui.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

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

// Packaged scene list: player_scenes.json next to the exe.
//   { "scenes": [ { "name": "...", "path": "scenes/a/a.rtp" }, ... ] }
// The first entry opens at launch when no project is given on the command
// line; F6 shows the list to switch. The list is a file, not Player state, and
// switching is the same rtapi::openProject that IPC project.open and drag/drop
// use, so a test reads this file and calls project.open with its path.
struct PlayerSceneEntry {
    std::string name;
    std::string path_utf8;  // absolute
};

std::vector<PlayerSceneEntry> loadPackagedSceneList() {
    std::vector<PlayerSceneEntry> scenes;
    const std::filesystem::path list_path = std::filesystem::absolute("player_scenes.json");
    std::ifstream in(list_path, std::ios::binary);
    if (!in) {
        return scenes;  // not a packaged build: drag/drop and --project only
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // No exceptions: this build aborts on a json throw.
    const auto doc = nlohmann::json::parse(text, nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) {
        SCENE_LOG_WARN("[Player] player_scenes.json is not a JSON object; scene list ignored");
        return scenes;
    }
    const auto list = doc.find("scenes");
    if (list == doc.end() || !list->is_array()) {
        SCENE_LOG_WARN("[Player] player_scenes.json has no \"scenes\" array; scene list ignored");
        return scenes;
    }
    for (const auto& entry : *list) {
        const auto path = entry.is_object() ? entry.find("path") : entry.end();
        if (!entry.is_object() || path == entry.end() || !path->is_string()) {
            SCENE_LOG_WARN("[Player] player_scenes.json: entry without a string \"path\" skipped");
            continue;
        }
        const auto resolved = std::filesystem::absolute(
            AppRuntime::projectPathFromUtf8(path->get<std::string>()));
        std::string name;
        if (const auto n = entry.find("name"); n != entry.end() && n->is_string()) {
            name = n->get<std::string>();
        }
        if (name.empty()) {
            name = AppRuntime::projectPathToUtf8(resolved.stem());
        }
        if (!std::filesystem::exists(resolved)) {
            SCENE_LOG_WARN("[Player] player_scenes.json: missing scene " +
                           AppRuntime::projectPathToUtf8(resolved));
        }
        scenes.push_back({name, AppRuntime::projectPathToUtf8(resolved)});
    }
    SCENE_LOG_INFO("[Player] Scene list: " + std::to_string(scenes.size()) + " scene(s)");
    return scenes;
}

// Scene picker (F6). Returns the clicked entry, or -1.
int drawSceneMenu(const std::vector<PlayerSceneEntry>& scenes, const std::string& current_utf8) {
    int chosen = -1;
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(vp->WorkPos.x + vp->WorkSize.x * 0.5f,
                                   vp->WorkPos.y + vp->WorkSize.y * 0.5f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowBgAlpha(0.85f);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoSavedSettings;
    if (ImGui::Begin("##player_scene_menu", nullptr, flags)) {
        ImGui::TextUnformatted("Scenes");
        ImGui::Separator();
        const auto current = AppRuntime::projectPathFromUtf8(current_utf8);
        for (int i = 0; i < static_cast<int>(scenes.size()); ++i) {
            const bool active = !current_utf8.empty() &&
                                AppRuntime::projectPathFromUtf8(scenes[i].path_utf8) == current;
            const std::string label = std::to_string(i + 1) + "  " + scenes[i].name +
                                      (active ? "  (open)" : "");
            if (ImGui::Button(label.c_str(), ImVec2(360.0f, 0.0f))) {
                chosen = i;
            }
        }
        ImGui::Separator();
        ImGui::TextDisabled("1-9 or click to open, F6 to close");
    }
    ImGui::End();
    return chosen;
}

using Clock = std::chrono::steady_clock;

double millisecondsBetween(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

// Display surface and frame telemetry owned by the Player executable. Every
// setting is reachable from the command line, the operator keys and the
// player.set_display IPC call; every measurement from the HUD and IPC.
class PlayerDisplay final : public AppRuntime::DisplayHost {
public:
    bool fullscreen = false;
    float render_scale = 1.0f;     // fraction of the fixed initial render resolution
    int pacing_hz = 60;            // 0 = uncapped (measure the real frame cost)
    bool continuous = false;       // render every iteration, even a settled scene
    bool hud = true;
    bool targets_dirty = true;     // presentation targets must be rebuilt

    AppRuntime::PlayerRenderExtent render_extent{
        original_surface ? original_surface->w : image_width,
        original_surface ? original_surface->h : image_height};

    PlayerDisplay() { AppRuntime::bindDisplayHost(*this); }
    ~PlayerDisplay() override { AppRuntime::unbindDisplayHost(*this); }

    // --- telemetry, fed by the loop ---------------------------------------
    void recordIteration(double advance_ms) {
        ++iterations_;
        advance_ms_ += advance_ms;
    }
    // One DRAWN frame. Cost is measured only here: a settled scene that draws
    // nothing must not turn sleep or idle presents into a "frame time".
    void recordRender(double render_ms, double upload_ms,
                      const Backend::RasterFrameTelemetry& telemetry) {
        ++renders_;
        render_ms_ += render_ms;
        upload_ms_ += upload_ms;
        if (telemetry.available) {
            ++backend_samples_;
            backend_frame_ms_ += telemetry.frame_ms;
            cpu_record_ms_ += telemetry.cpu_record_ms;
            slot_wait_ms_ += telemetry.slot_wait_ms;
            submit_ms_ += telemetry.submit_ms;
            draw_calls_ = telemetry.draw_calls;
            visible_triangles_ = telemetry.visible_triangles;
        }
    }
    void recordPresent(Clock::time_point now, double present_ms) {
        ++presents_;
        present_ms_ += present_ms;
        if (last_present_.time_since_epoch().count() != 0) {
            const double interval = millisecondsBetween(last_present_, now);
            interval_sum_ += interval;
            interval_max_ = std::max(interval_max_, interval);
            ++intervals_;
        }
        last_present_ = now;
    }
    // Publishes a one-second window; returns true when a new snapshot exists.
    bool publish(Clock::time_point now) {
        if (window_start_.time_since_epoch().count() == 0) {
            window_start_ = now;
            return false;
        }
        const double elapsed = millisecondsBetween(window_start_, now);
        if (elapsed < 1000.0) {
            return false;
        }
        const auto per = [](double sum, std::uint64_t count) {
            return count ? sum / static_cast<double>(count) : 0.0;
        };
        const bool idle = renders_ == 0;
        // Rates over the window (these DO include pacing and idle time).
        snapshot_["window_ms"] = elapsed;
        snapshot_["idle"] = idle;
        snapshot_["rendered_fps"] = 1000.0 * static_cast<double>(renders_) / elapsed;
        snapshot_["presented_fps"] = 1000.0 * static_cast<double>(presents_) / elapsed;
        snapshot_["loop_hz"] = 1000.0 * static_cast<double>(iterations_) / elapsed;
        snapshot_["present_interval_ms_avg"] = per(interval_sum_, intervals_);
        snapshot_["present_interval_ms_max"] = interval_max_;
        snapshot_["advance_ms_avg"] = per(advance_ms_, iterations_);
        snapshot_["present_ms_avg"] = per(present_ms_, presents_);
        // Cost of a drawn frame. In an idle window the last measured cost is kept
        // (cost_age_s says how old it is) instead of reporting zero.
        if (!idle) {
            const double render = per(render_ms_, renders_);
            const double upload = per(upload_ms_, renders_);
            snapshot_["render_ms_avg"] = render;
            snapshot_["upload_ms_avg"] = upload;
            snapshot_["frame_cost_ms"] = render + upload;
            snapshot_["capacity_fps"] = render + upload > 0.0 ? 1000.0 / (render + upload) : 0.0;
            snapshot_["backend_frame_ms_avg"] = per(backend_frame_ms_, backend_samples_);
            snapshot_["cpu_record_ms_avg"] = per(cpu_record_ms_, backend_samples_);
            snapshot_["slot_wait_ms_avg"] = per(slot_wait_ms_, backend_samples_);
            snapshot_["submit_ms_avg"] = per(submit_ms_, backend_samples_);
            snapshot_["draw_calls"] = draw_calls_;
            snapshot_["visible_triangles"] = visible_triangles_;
            snapshot_["cost_frames"] = renders_;
            last_cost_time_ = now;
        }
        snapshot_["cost_age_s"] = last_cost_time_.time_since_epoch().count() == 0
            ? -1.0 : millisecondsBetween(last_cost_time_, now) / 1000.0;
        window_start_ = now;
        iterations_ = renders_ = presents_ = intervals_ = backend_samples_ = 0;
        advance_ms_ = render_ms_ = upload_ms_ = present_ms_ = interval_sum_ = interval_max_ = 0.0;
        backend_frame_ms_ = cpu_record_ms_ = slot_wait_ms_ = submit_ms_ = 0.0;
        return true;
    }
    const nlohmann::json& frameStats() const { return snapshot_; }

    // --- AppRuntime::DisplayHost -------------------------------------------
    nlohmann::json displayStatus() const override {
        int window_w = 0, window_h = 0, output_w = 0, output_h = 0;
        if (window) SDL_GetWindowSize(window, &window_w, &window_h);
        if (renderer) SDL_GetRendererOutputSize(renderer, &output_w, &output_h);
        return {
            {"fullscreen", fullscreen}, {"render_scale", render_scale},
            {"pacing_hz", pacing_hz}, {"continuous", continuous}, {"hud", hud},
            {"window", {window_w, window_h}}, {"output", {output_w, output_h}},
            {"render_base", {render_extent.base().first, render_extent.base().second}},
            {"render", {original_surface ? original_surface->w : 0,
                        original_surface ? original_surface->h : 0}},
            {"frame", snapshot_},
        };
    }

    rtapi::Result setDisplay(const nlohmann::json& params) override {
        if (!params.is_object()) return rtapi::Result::fail("player.set_display expects an object");
        // Validate everything before applying anything: a rejected call
        // changes nothing (a wrong-typed JSON value would otherwise throw).
        const auto boolean = [&](const char* key, bool& out) -> std::string {
            const auto it = params.find(key);
            if (it == params.end() || it->is_null()) return {};
            if (!it->is_boolean()) return std::string(key) + " must be a boolean";
            out = it->get<bool>();
            return {};
        };
        const auto number = [&](const char* key, double& out) -> std::string {
            const auto it = params.find(key);
            if (it == params.end() || it->is_null()) return {};
            if (!it->is_number()) return std::string(key) + " must be a number";
            out = it->get<double>();
            return {};
        };
        bool next_fullscreen = fullscreen, next_continuous = continuous, next_hud = hud;
        double scale = render_scale, pacing = pacing_hz, width = 0.0, height = 0.0;
        for (const std::string& error : {boolean("fullscreen", next_fullscreen),
                                         boolean("continuous", next_continuous),
                                         boolean("hud", next_hud),
                                         number("render_scale", scale),
                                         number("pacing_hz", pacing),
                                         number("width", width), number("height", height)}) {
            if (!error.empty()) return rtapi::Result::fail(error);
        }
        if (!(scale >= 0.25 && scale <= 1.0)) {
            return rtapi::Result::fail("render_scale must be within 0.25..1");
        }
        if (!(pacing == 0.0 || (pacing >= 10.0 && pacing <= 1000.0))) {
            return rtapi::Result::fail("pacing_hz must be 0 (uncapped) or 10..1000");
        }
        if ((width != 0.0 || height != 0.0) &&
            (width < 320.0 || height < 240.0 || width > 16384.0 || height > 16384.0)) {
            return rtapi::Result::fail("width/height must be at least 320x240");
        }
        if ((width != 0.0 || height != 0.0) && next_fullscreen) {
            return rtapi::Result::fail("width/height apply to windowed mode; send fullscreen:false");
        }
        continuous = next_continuous;
        hud = next_hud;
        pacing_hz = static_cast<int>(pacing);
        if (static_cast<float>(scale) != render_scale) {
            render_scale = static_cast<float>(scale);
            targets_dirty = true;
        }
        if (next_fullscreen != fullscreen) {
            applyFullscreen(next_fullscreen);
        }
        if (width != 0.0 && height != 0.0) {
            SDL_SetWindowSize(window, static_cast<int>(width), static_cast<int>(height));
        }
        return rtapi::Result::success();
    }

    void applyFullscreen(bool enable) {
        if (!window) return;
        // Borderless desktop fullscreen: no display mode switch, no device
        // reset; the window simply covers its monitor.
        if (SDL_SetWindowFullscreen(window, enable ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0) == 0) {
            fullscreen = enable;
        } else {
            SCENE_LOG_WARN(std::string("[Player] Fullscreen switch failed: ") + SDL_GetError());
        }
    }

    // Only an explicit render-scale change rebuilds the render surface.
    // SDL_RenderCopy scales that fixed texture to the current window output.
    bool applyTargets(UIContext& context, AppRuntime::RuntimeSceneViewer& viewer) {
        if (!targets_dirty || !renderer) return false;
        const auto extent = render_extent.scaled(render_scale);
        const int w = extent.first;
        const int h = extent.second;
        if (original_surface && original_surface->w == w && original_surface->h == h) {
            targets_dirty = false;
            return false;
        }
        if (!AppRuntime::recreatePresentationTargets(w, h)) {
            return false;
        }
        targets_dirty = false;
        context.ray_texture = raytrace_texture;
        viewer.requestFrame();
        return true;
    }

    // Operator keys. Gameplay input arrives later through the action layer
    // (body pose + keyboard), not here.
    bool handleKey(const SDL_KeyboardEvent& key) {
        if (key.repeat) return false;
        switch (key.keysym.sym) {
            case SDLK_F11: applyFullscreen(!fullscreen); return true;
            case SDLK_ESCAPE:
                if (fullscreen) { applyFullscreen(false); return true; }
                return false;
            case SDLK_F1: hud = !hud; return true;
            case SDLK_F2:
                render_scale = render_scale > 0.9f ? 0.75f : (render_scale > 0.6f ? 0.5f : 1.0f);
                targets_dirty = true;
                return true;
            case SDLK_F3: pacing_hz = pacing_hz == 0 ? 60 : 0; return true;
            case SDLK_F4: continuous = !continuous; return true;
            case SDLK_F9: {
                // Attendant: clean room now (paint cleared, popped balloons back).
                // Same core as paint.wet_demo reset and the session's room.reset.
                // A kiosk key gate should limit this to the attendant role.
                std::string output;
                const auto result = rtapi::wetPaintDemoCommand("reset", "{}", output);
                if (!result.ok) {
                    SCENE_LOG_WARN("[Player] room reset: " + result.error);
                }
                return true;
            }
            default: return false;
        }
    }

private:
    std::uint64_t iterations_ = 0, renders_ = 0, presents_ = 0, intervals_ = 0;
    std::uint64_t backend_samples_ = 0, visible_triangles_ = 0;
    std::uint32_t draw_calls_ = 0;
    double backend_frame_ms_ = 0.0, cpu_record_ms_ = 0.0, slot_wait_ms_ = 0.0, submit_ms_ = 0.0;
    Clock::time_point last_cost_time_{};
    double advance_ms_ = 0.0, render_ms_ = 0.0, upload_ms_ = 0.0, present_ms_ = 0.0;
    double interval_sum_ = 0.0, interval_max_ = 0.0;
    Clock::time_point window_start_{}, last_present_{};
    nlohmann::json snapshot_ = nlohmann::json::object();
};

void drawHud(bool ipc_running, const AppRuntime::RuntimeSceneViewer& viewer,
             const PlayerDisplay& display) {
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

        ImGui::Separator();
        const auto& frame = display.frameStats();
        const auto value = [&](const char* key) { return frame.value(key, 0.0); };
        if (frame.empty()) {
            ImGui::TextUnformatted("FPS: measuring...");
        } else {
            // Cost comes from drawn frames only; rates include pacing and idle.
            if (frame.contains("frame_cost_ms")) {
                ImGui::Text("frame cost %.2f ms  -> capacity %.1f FPS%s", value("frame_cost_ms"),
                            value("capacity_fps"), frame.value("idle", false) ? "  (last drawn)" : "");
                ImGui::Text("  backend %.2f (CPU record %.2f, submit %.2f, slot wait %.2f) ms",
                            value("backend_frame_ms_avg"), value("cpu_record_ms_avg"),
                            value("submit_ms_avg"), value("slot_wait_ms_avg"));
                ImGui::Text("  upload %.2f ms   %u draws, %.2f M tris",
                            value("upload_ms_avg"), frame.value("draw_calls", 0u),
                            frame.value("visible_triangles", 0.0) / 1.0e6);
                if (value("slot_wait_ms_avg") > 1.0) {
                    ImGui::TextUnformatted("  GPU-bound: the CPU waits for frame slots");
                }
            } else {
                ImGui::TextUnformatted("frame cost: no frame drawn yet");
            }
            if (frame.value("idle", false)) {
                ImGui::TextUnformatted("drawn 0 FPS: scene settled (F4 = draw continuously)");
            } else {
                ImGui::Text("drawn %.1f FPS, presented %.1f FPS, sim %.2f ms",
                            value("rendered_fps"), value("presented_fps"), value("advance_ms_avg"));
            }
        }
        const int render_w = original_surface ? original_surface->w : 0;
        const int render_h = original_surface ? original_surface->h : 0;
        // Report render and output separately: resizing only scales presentation.
        int output_w = 0, output_h = 0;
        if (renderer) SDL_GetRendererOutputSize(renderer, &output_w, &output_h);
        ImGui::Text("render %dx%d -> output %dx%d  scale %.2f  %s", render_w, render_h, output_w,
                    output_h, display.render_scale,
                    display.fullscreen ? "fullscreen" : "windowed");
        ImGui::Text("pacing %s%s", display.pacing_hz ? (std::to_string(display.pacing_hz) + " Hz").c_str()
                                                     : "uncapped",
                    display.continuous ? ", continuous" : "");
        ImGui::TextDisabled("F11 fullscreen  F1 HUD  F2 scale  F3 pacing  F4 continuous  F6 scenes");
    }
    ImGui::End();
}

} // namespace

int main(int argc, char* argv[]) {
    // Resolve a launch project before changing the working directory to the exe.
    //   RayTrophiPlayer.exe [project.rtp | --project <path>] [--fullscreen]
    //       [--width <px> --height <px>] [--render-scale <0.25..1>]
    //       [--pacing <hz, 0 = uncapped>] [--continuous] [--no-hud]
    std::string launch_project;
    nlohmann::json launch_display = nlohmann::json::object();
    int launch_width = 0;
    int launch_height = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--project" && has_value) {
            launch_project = argv[++i];
        } else if (arg == "--fullscreen") {
            launch_display["fullscreen"] = true;
        } else if (arg == "--continuous") {
            launch_display["continuous"] = true;
        } else if (arg == "--no-hud") {
            launch_display["hud"] = false;
        } else if (arg == "--width" && has_value) {
            launch_width = std::atoi(argv[++i]);
        } else if (arg == "--height" && has_value) {
            launch_height = std::atoi(argv[++i]);
        } else if (arg == "--render-scale" && has_value) {
            launch_display["render_scale"] = std::atof(argv[++i]);
        } else if (arg == "--pacing" && has_value) {
            launch_display["pacing_hz"] = std::atof(argv[++i]);
        } else if (arg.rfind("--", 0) != 0 && launch_project.empty()) {
            launch_project = arg;
        }
    }
    if (!launch_project.empty()) {
        launch_project = AppRuntime::projectPathToUtf8(
            std::filesystem::absolute(AppRuntime::projectPathFromUtf8(launch_project)));
    }
    g_main_thread_id = std::this_thread::get_id();
    setWorkingDirectoryToExe(argc > 0 ? argv[0] : nullptr);
    g_sceneLog.initLogLocation();

    // Packaged build: the scene list is relative to the exe (the working
    // directory now). A command-line project still wins over its first entry.
    const std::vector<PlayerSceneEntry> packaged_scenes = loadPackagedSceneList();
    if (launch_project.empty() && !packaged_scenes.empty()) {
        launch_project = packaged_scenes.front().path_utf8;
    }

    if (!AppRuntime::initPlatform(nullptr)) {
        return 1;
    }

    AppRuntime::WindowConfig window_config;
    window_config.title = "RayTrophi Player";
    window_config.width = launch_width >= 320 ? launch_width : image_width;
    window_config.height = launch_height >= 240 ? launch_height : image_height;
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

    PlayerDisplay display;
    if (!launch_display.empty()) {
        const auto applied = display.setDisplay(launch_display);
        if (!applied.ok) {
            SCENE_LOG_WARN("[Player] Ignoring launch display options: " + applied.error);
        }
    }

    bool running = true;
    bool scene_menu = false;
    int scene_request = -1;  // index into packaged_scenes, opened between events
    bool debug_camera = false;
    float debug_orbit_x = 0.0f, debug_orbit_y = 0.0f;
    float debug_pan_x = 0.0f, debug_pan_y = 0.0f;
    int debug_wheel = 0;
    Uint64 previous_tick = SDL_GetPerformanceCounter();
    AppRuntime::FrameCycle frame_cycle;
    SDL_Event event;
    AppRuntime::EventInfo event_info;
    while (running) {
        frame_cycle.beginIteration();
        const auto iteration_start = Clock::now();
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
                frame_cycle.requestRedraw();
            }
            if (event.type == SDL_KEYDOWN && !ImGui::GetIO().WantCaptureKeyboard &&
                display.handleKey(event.key)) {
                frame_cycle.requestRedraw();
            }
            // Scene picker (packaged builds). A kiosk key gate must keep F6 out
            // of the children's reach, like F5 and F9.
            if (event.type == SDL_KEYDOWN && !event.key.repeat && !packaged_scenes.empty()) {
                const SDL_Keycode key = event.key.keysym.sym;
                if (key == SDLK_F6) {
                    scene_menu = !scene_menu;
                    frame_cycle.requestRedraw();
                } else if (scene_menu && key >= SDLK_1 && key <= SDLK_9 &&
                           static_cast<size_t>(key - SDLK_1) < packaged_scenes.size()) {
                    scene_request = static_cast<int>(key - SDLK_1);
                } else if (scene_menu && key == SDLK_ESCAPE) {
                    scene_menu = false;
                    frame_cycle.requestRedraw();
                }
            }
            // Debug camera (F5, off at launch). A diagnostic aid, not gameplay
            // input: it drives the same rtapi camera calls camera.orbit /
            // camera.dolly expose over IPC. Left drag orbits, right or middle
            // drag pans, the wheel dollies. A kiosk key gate must keep F5 out.
            if (event.type == SDL_KEYDOWN && !event.key.repeat &&
                event.key.keysym.sym == SDLK_F5 && !ImGui::GetIO().WantCaptureKeyboard) {
                debug_camera = !debug_camera;
                SCENE_LOG_INFO(std::string("[Player] Debug camera ") +
                               (debug_camera ? "on" : "off"));
            }
            // Balloon paint demo: a left click pops the balloon under the cursor.
            // Game input, so it goes through the same core call as IPC
            // paint.wet_demo pop (screen point in 0..1 from the top-left). With
            // the debug camera on, the left button orbits instead. No demo
            // prepared, or nothing under the cursor: the call does nothing.
            if (event.type == SDL_MOUSEBUTTONDOWN && event.button.button == SDL_BUTTON_LEFT &&
                !debug_camera && !ImGui::GetIO().WantCaptureMouse) {
                int window_w = 0, window_h = 0;
                SDL_GetWindowSize(window, &window_w, &window_h);
                if (window_w > 0 && window_h > 0) {
                    const float px = std::clamp(static_cast<float>(event.button.x) /
                                                static_cast<float>(window_w), 0.0f, 1.0f);
                    const float py = std::clamp(static_cast<float>(event.button.y) /
                                                static_cast<float>(window_h), 0.0f, 1.0f);
                    std::string popped;
                    // json, not std::to_string: a Turkish locale writes "0,5".
                    (void)rtapi::wetPaintDemoCommand(
                        "pop", nlohmann::json{{"screen", {px, py}}}.dump(), popped);
                    frame_cycle.requestRedraw();
                }
            }
            // Only ACCUMULATED here. A mouse reports far more often than the
            // loop iterates; applying each event as its own camera update made
            // hand-driven motion stutter while one IPC orbit per iteration was
            // smooth. The sum is applied once, after the queue is empty.
            if (debug_camera && !ImGui::GetIO().WantCaptureMouse) {
                if (event.type == SDL_MOUSEMOTION) {
                    const float dx = static_cast<float>(event.motion.xrel);
                    const float dy = static_cast<float>(event.motion.yrel);
                    if (event.motion.state & SDL_BUTTON_LMASK) {
                        debug_orbit_x += dx; debug_orbit_y += dy;
                    } else if (event.motion.state & (SDL_BUTTON_RMASK | SDL_BUTTON_MMASK)) {
                        debug_pan_x += dx; debug_pan_y += dy;
                    }
                } else if (event.type == SDL_MOUSEWHEEL) {
                    debug_wheel += event.wheel.y;
                }
            }
        }
        if (debug_orbit_x != 0.0f || debug_orbit_y != 0.0f) {
            rtapi::orbitCamera(-debug_orbit_x * 0.25f, -debug_orbit_y * 0.25f);
            debug_orbit_x = debug_orbit_y = 0.0f;
            frame_cycle.requestRedraw();
        }
        if (debug_pan_x != 0.0f || debug_pan_y != 0.0f) {
            rtapi::panCamera(-debug_pan_x * 0.01f, debug_pan_y * 0.01f);
            debug_pan_x = debug_pan_y = 0.0f;
            frame_cycle.requestRedraw();
        }
        if (debug_wheel != 0) {
            rtapi::dollyCamera(std::pow(0.9f, static_cast<float>(debug_wheel)));
            debug_wheel = 0;
            frame_cycle.requestRedraw();
        }
        if (scene_request >= 0 && scene_request < static_cast<int>(packaged_scenes.size())) {
            const auto& entry = packaged_scenes[static_cast<size_t>(scene_request)];
            SCENE_LOG_INFO("[Player] Opening scene '" + entry.name + "'");
            const auto loaded = rtapi::openProject(entry.path_utf8);
            if (!loaded.ok) {
                SCENE_LOG_WARN("[Player] " + loaded.error);
            }
            scene_menu = false;
            frame_cycle.requestRedraw();
        }
        scene_request = -1;
        // Window/fullscreen changes affect presentation only. F2 and explicit
        // render_scale changes resize targets against the fixed render base.
        if (display.applyTargets(state->ctx, *viewer)) {
            frame_cycle.requestRedraw();
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
        const auto advance_start = Clock::now();
        viewer->advanceRuntime(dt_seconds);
        display.recordIteration(millisecondsBetween(advance_start, Clock::now()));
        if (display.continuous) {
            viewer->forceRedraw();
        }

        // Explicit capture can render while unfocused; ordinary background windows
        // stay idle. A settled static scene stops submitting Vulkan frames too.
        // Continuous (measurement) mode renders unfocused as well, WITHOUT capture:
        // capture switches the backend to synchronous present (a GPU wait inside
        // every frame), which measured the bedroom's 24.5 ms GPU frame as
        // "unaccounted host time" instead of the normal asynchronous cost.
        const bool render_requested = viewer->needsFrame();
        const bool capture_requested = rtapi::viewportCaptureEnabled();
        bool image_updated = false;
        if (render_requested &&
            (frame_cycle.isForeground() || capture_requested || display.continuous)) {
            image_updated = viewer->render(original_surface, raytrace_texture);
            display.recordRender(viewer->lastRenderMs(), viewer->lastUploadMs(),
                                 viewer->lastTelemetry());
            frame_cycle.requestRedraw();
        }
        if (frame_cycle.shouldPresent(image_updated)) {
            const auto present_start = Clock::now();
            frame_cycle.beginUiFrame();
            if (display.hud) {
                drawHud(rtipc::isRunning(), *viewer, display);
            }
            if (scene_menu) {
                const int clicked = drawSceneMenu(
                    packaged_scenes, viewer->status().value("project_path", ""));
                if (clicked >= 0) {
                    scene_request = clicked;  // opened next iteration, outside the UI frame
                    frame_cycle.requestRedraw();
                }
            }
            frame_cycle.endUiFrame();

            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
            if (viewer->hasFrame()) {
                SDL_RenderCopy(renderer, raytrace_texture, nullptr, nullptr);
            }
            frame_cycle.drawUi();
            frame_cycle.present();
            const auto present_end = Clock::now();
            display.recordPresent(present_end, millisecondsBetween(present_start, present_end));
        }
        if (display.publish(Clock::now())) {
            viewer->setLoopStatus(display.displayStatus());
            if (display.hud) {
                frame_cycle.requestRedraw();  // refresh the numbers once a second
            }
        }

        // Pacing. Foreground: the configured rate, or none at all when uncapped
        // (the honest frame cost). An uncapped loop with nothing to render still
        // yields 1 ms so a settled scene does not spin a core. Background: the
        // old 20 Hz IPC/event pump - except in continuous (measurement) mode,
        // where a background window would silently cap the result at 20.
        if (!frame_cycle.isForeground() && !display.continuous) {
            std::this_thread::sleep_until(iteration_start + std::chrono::milliseconds(50));
        } else if (display.pacing_hz > 0) {
            std::this_thread::sleep_until(iteration_start +
                std::chrono::microseconds(1000000 / display.pacing_hz));
        } else if (!image_updated) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
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
