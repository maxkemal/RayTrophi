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
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
#include "Viewport/PlayerSwapchainPresenter.h"
#endif
#include "Core/AppFrame.h"
#include "Core/RuntimeSceneViewer.h"
#include "Core/RuntimeProjectHost.h"
#include "Core/PlayerRenderExtent.h"
#include "Core/PlayerResolutionMenu.h"
#include "Core/PlayerOperator.h"
#include "Core/PlayerGameSettings.h"
#include "Core/Localization.h"
#include "Core/PlayerSceneLoad.h"
#include "Core/PlayerBalloonInput.h"
#include "Core/PlayerBoundedCamera.h"
#include "Paint/BalloonInteraction.h"
#include "Paint/BalloonMilestoneRewards.h"
#include "PlayerStartupFailureTouch.h"
#include "PlayerOperatorTouch.h"
#include "UI/PlayerOperatorUI.h"
#include "UI/PlayerKidLauncher.h"
#ifdef __ANDROID__
#include "UI/AndroidPerfHud.h"
#endif
#include "Api/RtApiPlayerOperator.h"

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
#include "Core/FatalSignalTrace.h"
#include <csignal>
#include <exception>

#include <SDL.h>
#include <SDL_main.h>
#ifdef __ANDROID__
#include <android/log.h>
#endif
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
#include <SDL_vulkan.h>
#endif
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

#include "Platform/Platform.h"

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

#ifdef __ANDROID__
constexpr const char* kAndroidStartupLogTag = "RayTrophiStartup";

void logAndroidStartup(const std::string& message) {
    __android_log_write(ANDROID_LOG_INFO, kAndroidStartupLogTag, message.c_str());
}

void logAndroidStartupError(const char* stage, const char* error) {
    __android_log_print(ANDROID_LOG_ERROR, kAndroidStartupLogTag, "%s: %s", stage,
                        error && *error ? error : "No error detail");
}
#endif

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

#ifndef __ANDROID__
// Android's debuggerd captures native faults. The desktop writer would
// turn a fatal signal into EXIT_SELF without a trace on Android.
void playerSignalHandler(int sig) {
    writeFatalSignalTrace(sig);
    std::_Exit(EXIT_FAILURE);
}

void playerTerminateHandler() {
    writeFatalSignalTrace(-1);
    std::_Exit(EXIT_FAILURE);
}

#ifdef _WIN32
LONG WINAPI playerExceptionFilter(EXCEPTION_POINTERS* info) {
    // Negative code = SEH exception; the low bits carry the NTSTATUS.
    writeFatalSignalTrace(info && info->ExceptionRecord
        ? -static_cast<int>(info->ExceptionRecord->ExceptionCode & 0x7FFFFFFF) : -2);
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

void installPlayerCrashHandlers() {
#ifdef _WIN32
    SetUnhandledExceptionFilter(playerExceptionFilter);
#endif
    std::set_terminate(playerTerminateHandler);
    std::signal(SIGABRT, playerSignalHandler);
    std::signal(SIGSEGV, playerSignalHandler);
    std::signal(SIGFPE, playerSignalHandler);
    std::signal(SIGILL, playerSignalHandler);
}
#endif

bool setWorkingDirectoryToExe(char* argv0) {
#if defined(__ANDROID__)
    (void)argv0;
    const auto directory = Platform::appDataRoot();
    std::error_code error;
    if (!directory.empty()) {
        std::filesystem::current_path(directory, error);
    }
    if (directory.empty() || error) {
        const std::string detail = directory.empty() ? "Internal data directory is unavailable"
                                                     : error.message();
        logAndroidStartupError("Player working directory", detail.c_str());
        return false;
    }
#elif defined(_WIN32)
    (void)argv0;
    std::filesystem::current_path(Platform::appDataRoot());
#else
    if (argv0) {
        std::filesystem::current_path(std::filesystem::path(argv0).parent_path());
    }
#endif
    return true;
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
    float render_scale = 1.0f;     // fraction of the explicitly selected render resolution
    int pacing_hz = 60;            // 0 = uncapped (measure the real frame cost)
    bool continuous = false;       // render every iteration, even a settled scene
#ifdef __ANDROID__
    bool hud = false;  // consumer app: no diagnostics overlay unless asked (display.hud / F1)
#else
    bool hud = true;
#endif
    bool targets_dirty = true;     // presentation targets must be rebuilt
    bool resolution_menu = false;

    AppRuntime::PlayerRenderExtent render_extent{
        original_surface ? original_surface->w : image_width,
        original_surface ? original_surface->h : image_height};

    PlayerDisplay() {
        const auto choices = AppRuntime::playerResolutionChoices(window);
        if (!choices.modes.empty()) {
            render_extent = AppRuntime::PlayerRenderExtent(
                choices.recommended.first, choices.recommended.second);
        }
        AppRuntime::bindDisplayHost(*this);
    }
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
    // Where ONE loop iteration went. Averages hide a single long frame (a
    // balloon pop: 18-29 ms with every simulation part under 7 ms, 2026-11-24),
    // so the window keeps the breakdown of its longest iteration.
    struct IterationBreakdown {
        double total_ms = 0.0;     // iteration start .. after pacing
        double input_ms = 0.0;     // event poll, balloon input flush, operator, targets
        double pump_ms = 0.0;      // main-thread work queue (IPC requests run here)
        double advance_ms = 0.0;   // game clock: simulation, wet paint, demo
        double render_ms = 0.0;    // viewer->render incl. its upload
        double present_ms = 0.0;   // UI frame + present
        double sleep_ms = 0.0;     // pacing / idle yield
        double backend_record_ms = 0.0, backend_slot_wait_ms = 0.0, backend_submit_ms = 0.0;
        bool rendered = false;
    };
    void recordIterationBreakdown(const IterationBreakdown& b) {
        if (b.total_ms > worst_.total_ms) worst_ = b;
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
        // The window's longest iteration, stage by stage. "unaccounted_ms" is
        // the loop's own glue (HUD status, device checks); a large value there
        // means a stage is missing from this breakdown.
        {
            const auto& w = worst_;
            const double accounted = w.input_ms + w.pump_ms + w.advance_ms + w.render_ms +
                                     w.present_ms + w.sleep_ms;
            snapshot_["worst_iteration"] = {
                {"total_ms", w.total_ms}, {"input_ms", w.input_ms}, {"pump_ms", w.pump_ms},
                {"advance_ms", w.advance_ms}, {"render_ms", w.render_ms},
                {"present_ms", w.present_ms}, {"sleep_ms", w.sleep_ms},
                {"unaccounted_ms", std::max(0.0, w.total_ms - accounted)},
                {"rendered", w.rendered},
                {"backend_record_ms", w.backend_record_ms},
                {"backend_slot_wait_ms", w.backend_slot_wait_ms},
                {"backend_submit_ms", w.backend_submit_ms}};
            worst_ = IterationBreakdown{};
        }
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
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
        if (window) SDL_Vulkan_GetDrawableSize(window, &output_w, &output_h);
#else
        if (auto* outputRenderer = renderer) {
            SDL_GetRendererOutputSize(outputRenderer, &output_w, &output_h);
        }
#endif
        auto result = AppRuntime::playerResolutionChoices(window).status();
        result.update(nlohmann::json{
            {"fullscreen", fullscreen}, {"render_scale", render_scale},
            {"pacing_hz", pacing_hz}, {"continuous", continuous}, {"hud", hud},
            {"window", {window_w, window_h}}, {"output", {output_w, output_h}},
            {"render_base", {render_extent.base().first, render_extent.base().second}},
            {"render", {original_surface ? original_surface->w : 0,
                        original_surface ? original_surface->h : 0}},
            {"frame", snapshot_},
        });
        return result;
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
        std::pair<int, int> resolution = render_extent.base();
        const bool selecting = params.contains("render_width") || params.contains("render_height");
        if (selecting) {
            const auto choices = AppRuntime::playerResolutionChoices(window);
            double render_width = 0.0, render_height = 0.0;
            for (const auto& error : {number("render_width", render_width),
                                      number("render_height", render_height)}) {
                if (!error.empty()) {
                    return rtapi::Result::fail(error);
                }
            }
            if (!std::isfinite(render_width) || !std::isfinite(render_height) ||
                render_width < 320 || render_width > 16384 || render_height < 240 ||
                render_height > 16384 || std::floor(render_width) != render_width ||
                std::floor(render_height) != render_height ||
                !choices.supports(static_cast<int>(render_width), static_cast<int>(render_height))) {
                return rtapi::Result::fail(
                    "render_width/render_height must be a supported resolution pair from player.display");
            }
            resolution = {static_cast<int>(render_width), static_cast<int>(render_height)};
            // An explicit resolution denotes actual pixels unless scale is also explicit.
            if (!params.contains("render_scale") || params["render_scale"].is_null()) {
                scale = 1.0;
            }
        }
        continuous = next_continuous;
        hud = next_hud;
        pacing_hz = static_cast<int>(pacing);
        if (resolution != render_extent.base()) {
            render_extent = AppRuntime::PlayerRenderExtent(resolution.first, resolution.second);
            targets_dirty = true;
        }
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

    // Only an explicit resolution/render-scale change rebuilds the render surface.
    // The presentation path scales the fixed render extent to the current window output.
    bool applyTargets(UIContext& context, AppRuntime::RuntimeSceneViewer& viewer) {
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
        if (!targets_dirty || !window) return false;
#else
        if (!targets_dirty || !renderer) return false;
#endif
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
                if (AppRuntime::playerOperatorOpen()) {
                    AppRuntime::closePlayerOperator();
                    return true;
                }
                if (resolution_menu) {
                    resolution_menu = false;
                    return true;
                }
                if (fullscreen) { applyFullscreen(false); return true; }
                return false;
#ifdef __ANDROID__
            case SDLK_AC_BACK:
                // Android Back: closes an IPC-opened console, otherwise play <-> pause (the
                // kid launcher ignores it on the library and loading screens).
                if (AppRuntime::playerConsoleOpen()) {
                    AppRuntime::closePlayerOperator();
                } else {
                    AppRuntime::kidLauncherBack();
                }
                return true;
#endif
            case SDLK_F1: hud = !hud; return true;
            case SDLK_F2:
                if (AppRuntime::playerOperatorOpen()) {
                    AppRuntime::closePlayerOperator();
                } else {
                    AppRuntime::showPlayerOperator("display");
                }
                return true;
            case SDLK_F6:
                AppRuntime::showPlayerOperator("games");
                return true;
            case SDLK_F3: pacing_hz = pacing_hz == 0 ? 60 : 0; return true;
            case SDLK_F4: continuous = !continuous; return true;
            case SDLK_F8: {
                nlohmann::json output;
                const auto result = rtapi::playerOperator(
                    "game.settings.clear_paint", nlohmann::json::object(), output);
                if (!result.ok) {
                    SCENE_LOG_WARN("[Player] clear paint: " + result.error);
                }
                return true;
            }
            case SDLK_F9: {
                // Attendant: clean room now (paint cleared, popped balloons back).
                // Same core as paint.wet_demo reset and the session's room.reset.
                // A kiosk key gate should limit this to the attendant role.
                std::string output;
                const auto result = rtapi::wetPaintDemoCommand("reset", "{}", output);
                if (!result.ok) {
                    SCENE_LOG_WARN("[Player] room reset: " + result.error);
                }
                AppRuntime::boundedCamera().returnHome();
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
    IterationBreakdown worst_;
    Clock::time_point window_start_{}, last_present_{};
    nlohmann::json snapshot_ = nlohmann::json::object();
};

// Keep startup errors and Android device loss readable with the debug HUD off.
enum class StartupNotice { None, SceneError, UnsupportedGpu, DeviceLost };

// Everything a startup/failure notice says, independent of how it is shown: the ImGui
// overlay draws it, and on Android a fatal one is also shown as a native dialog.
struct StartupNoticeInfo {
    StartupNotice kind = StartupNotice::None;
    std::string title;
    std::string headline;            // device lost: VK_ERROR_DEVICE_LOST; otherwise the error text
    std::string detail;              // device lost: reason + restart hint
    std::string log_title;
    std::vector<std::string> log;    // recent errors and warnings, newest last
};

StartupNoticeInfo classifyStartupNotice(const AppRuntime::RuntimeSceneViewer& viewer) {
    StartupNoticeInfo info;
#ifdef __ANDROID__
    const bool device_lost = viewer.deviceLost();
#else
    constexpr bool device_lost = false;
#endif
    if (viewer.hasScene() && !device_lost) return info;
    const auto status = viewer.status();
    // Backend initialization runs inside the project open, so its failure arrives as
    // last_open_error; a later runtime fault arrives as error.
    std::string error = status.value("error", "");
    if (error.empty()) error = status.value("last_open_error", "");
    if (error.empty() && !device_lost) return info;
    const bool unsupported_gpu = error.rfind("This device's GPU is not supported", 0) == 0;
    info.title = device_lost
        ? Localization::text("player.startup_device_lost", "Vulkan device lost")
        : unsupported_gpu
        ? Localization::text("player.startup_unsupported_gpu",
                             "RayTrophi Player cannot run on this device")
        : Localization::text("player.startup_scene_failed",
                             "RayTrophi Player could not open the scene");
    if (device_lost) {
        info.headline = "VK_ERROR_DEVICE_LOST";
        info.detail = status.value("device_lost_reason", "");
        if (!info.detail.empty()) info.detail += "\n";
        info.detail += Localization::text(
            "player.startup_device_lost_restart", "Close and reopen RayTrophi Player.");
    } else {
        info.headline = error;
    }
    info.log_title = Localization::text("player.startup_log", "Recent errors and warnings");
    const auto log = status.find("startup_log");
    if (log != status.end() && log->is_array()) {
        for (const auto& line : *log) {
            if (line.is_string()) info.log.push_back(line.get<std::string>());
        }
    }
    info.kind = device_lost ? StartupNotice::DeviceLost
        : unsupported_gpu ? StartupNotice::UnsupportedGpu : StartupNotice::SceneError;
    return info;
}

#ifdef __ANDROID__
// The Vulkan swapchain cannot show anything without a device (or after it is lost), so a
// fatal notice is a native dialog: the same headline, reason and recent log lines the
// overlay lists (newest last, capped), then the app quits.
bool isFatalStartupNotice(StartupNotice kind) {
    return kind == StartupNotice::UnsupportedGpu || kind == StartupNotice::DeviceLost;
}

void showFatalStartupDialog(const StartupNoticeInfo& info) {
    logAndroidStartupError(info.headline.c_str(), info.detail.c_str());
    // The native dialog blocks the main thread that services IPC requests.
    for (const std::string& line : info.log) {
        const int priority = line.rfind("[ERROR]", 0) == 0
            ? ANDROID_LOG_ERROR
            : ANDROID_LOG_WARN;
        __android_log_write(priority, kAndroidStartupLogTag, line.c_str());
    }
    std::string body = info.headline;
    if (!info.detail.empty()) body += "\n" + info.detail;
    if (!info.log.empty()) {
        constexpr size_t kMaxLines = 20;
        body += "\n\n" + info.log_title + ":";
        // Print newest lines first (and errors before warnings) so small phone
        // AlertDialogs immediately show [GpuBreadcrumb] / [Vulkan] errors at the
        // top instead of burying them under earlier [MatUpload] warnings.
        std::vector<std::string> errors;
        std::vector<std::string> warnings;
        for (auto it = info.log.rbegin(); it != info.log.rend(); ++it) {
            if (it->rfind("[ERROR]", 0) == 0) {
                errors.push_back(*it);
            } else {
                warnings.push_back(*it);
            }
        }
        size_t shown = 0;
        for (const std::string& line : errors) {
            if (shown++ >= kMaxLines) break;
            body += "\n" + line;
        }
        for (const std::string& line : warnings) {
            if (shown++ >= kMaxLines) break;
            body += "\n" + line;
        }
    }
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, info.title.c_str(), body.c_str(), window);
    SDL_Event quit{};
    quit.type = SDL_QUIT;
    SDL_PushEvent(&quit);
}
#endif

void drawStartupFailureNotice(const StartupNoticeInfo& info) {
    if (info.kind == StartupNotice::None) return;
#ifdef __ANDROID__
    // Fatal notices are native dialogs; the presenter may not even have a device.
    if (isFatalStartupNotice(info.kind)) return;
#endif
    const ImGuiViewport* view = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(view->WorkPos.x + view->WorkSize.x * 0.5f,
                                   view->WorkPos.y + view->WorkSize.y * 0.5f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
#ifdef __ANDROID__
    ImGui::SetNextWindowSize(ImVec2(view->WorkSize.x * 0.9f, 0.0f), ImGuiCond_Always);
    ImGui::SetNextWindowSizeConstraints(ImVec2(0, 0),
                                        ImVec2(view->WorkSize.x, view->WorkSize.y * 0.9f));
#else
    ImGui::SetNextWindowSize(ImVec2(view->WorkSize.x * 0.6f, 0.0f), ImGuiCond_Always);
#endif
    ImGui::SetNextWindowBgAlpha(0.85f);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoNav |
                                   ImGuiWindowFlags_NoFocusOnAppearing;
#ifdef __ANDROID__
    flags &= ~ImGuiWindowFlags_NoScrollbar;
#endif
    if (ImGui::Begin("##player_startup_failure", nullptr, flags)) {
        ImGui::SetWindowFontScale(1.6f);
        ImGui::TextUnformatted(info.title.c_str());
        ImGui::SetWindowFontScale(1.0f);
        ImGui::Separator();
        ImGui::TextWrapped("%s", info.headline.c_str());
        if (!info.detail.empty()) {
            ImGui::TextWrapped("%s", info.detail.c_str());
        }
#ifdef __ANDROID__
        if (!info.log.empty() &&
            ImGui::CollapsingHeader((info.log_title + "###player_startup_log").c_str())) {
            for (const auto& line : info.log) {
                ImGui::TextWrapped("%s", line.c_str());
            }
        }
#endif
    }
    ImGui::End();
}

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
        const auto error = status.value("error", "");
        const auto open_error = status.value("last_open_error", "");
        if (viewer.hasScene()) {
            const auto path = AppRuntime::projectPathFromUtf8(status.value("project_path", ""));
            const auto name = AppRuntime::projectPathToUtf8(path.filename());
            ImGui::TextUnformatted(name.c_str());
#ifndef __ANDROID__
        } else if (error.empty() && open_error.empty()) {
            // Only an invitation when dropping a project could actually work.
            ImGui::TextUnformatted("Drop an .rtp project here");
#endif
        }
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
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
        if (window) SDL_Vulkan_GetDrawableSize(window, &output_w, &output_h);
#else
        if (auto* outputRenderer = renderer) {
            SDL_GetRendererOutputSize(outputRenderer, &output_w, &output_h);
        }
#endif
        ImGui::Text("render %dx%d -> output %dx%d  scale %.2f  %s", render_w, render_h, output_w,
                    output_h, display.render_scale,
                    display.fullscreen ? "fullscreen" : "windowed");
        ImGui::Text("pacing %s%s", display.pacing_hz ? (std::to_string(display.pacing_hz) + " Hz").c_str()
                                                     : "uncapped",
                    display.continuous ? ", continuous" : "");
        ImGui::TextDisabled("F11 fullscreen  F1 HUD  F2 console  F3 pacing  F4 continuous  F6 library  F8 clear paint");
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
#ifndef __ANDROID__
    installPlayerCrashHandlers();
#endif
#ifdef __ANDROID__
    constexpr AppRuntime::StartupLogFn startup_log = &logAndroidStartup;
    logAndroidStartup("Player native entry started");
#else
    constexpr AppRuntime::StartupLogFn startup_log = nullptr;
#endif
    if (!setWorkingDirectoryToExe(argc > 0 ? argv[0] : nullptr)) {
        return 1;
    }
    g_sceneLog.initLogLocation();
    const auto startupDirectory = std::filesystem::current_path();
    SCENE_LOG_INFO("[Player Startup] Working directory: " +
                   AppRuntime::projectPathToUtf8(startupDirectory) +
                   " | Package shader directory: " +
                   AppRuntime::projectPathToUtf8(startupDirectory / "shaders"));

    // Android: shaders/scenes/assets live inside the APK; copy them under the working
    // directory first so everything below reads plain files exactly as on Windows.
    {
        std::string asset_report;
        const bool assets_ok = Platform::extractBundledAssets(asset_report);
        if (!asset_report.empty()) {
            if (assets_ok) SCENE_LOG_INFO("[Player Startup] " + asset_report);
            else SCENE_LOG_ERROR("[Player Startup] Bundled assets: " + asset_report);
        }
#ifdef __ANDROID__
        if (assets_ok) logAndroidStartup("Bundled assets ready");
        else logAndroidStartupError("Bundled assets", asset_report.c_str());
#endif
    }

    // An explicit project bypasses the library. Packaged games always start
    // on the game library / kid launcher card screen (even with a single game).
    const std::vector<PlayerSceneEntry> packaged_scenes = loadPackagedSceneList();
    const bool open_library = launch_project.empty() && !packaged_scenes.empty();

    if (!AppRuntime::initPlatform(startup_log)) {
#ifdef __ANDROID__
        logAndroidStartupError("SDL_Init", SDL_GetError());
#endif
        return 1;
    }

    AppRuntime::WindowConfig window_config;
    window_config.title = "RayTrophi Player";
    window_config.width = launch_width >= 320 ? launch_width : image_width;
    window_config.height = launch_height >= 240 ? launch_height : image_height;
    window_config.startup_log = startup_log;
    AppRuntime::createMainWindow(window_config);
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
    if (!window) {
#ifdef __ANDROID__
        logAndroidStartupError("SDL_CreateWindow (Vulkan)", SDL_GetError());
#endif
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "RayTrophi Player", SDL_GetError(), nullptr);
        AppRuntime::shutdownPlatform();
        return 1;
    }
#endif

    AppRuntime::ImGuiConfig imgui_config;
#ifdef __ANDROID__
    // Android has no operator language settings. Use the OS preference list.
    SDL_Locale* locales = SDL_GetPreferredLocales();
    for (const SDL_Locale* locale = locales; locale && locale->language; ++locale) {
        std::string error;
        if (Localization::setLocale(locale->language, error)) {
            break;
        }
    }
    SDL_free(locales);
    // The embedded ImGui font omits several Turkish characters.
    for (const char* path : {"/system/fonts/Roboto-Regular.ttf",
                             "/system/fonts/RobotoStatic-Regular.ttf",
                             "/system/fonts/RobotoFlex-Regular.ttf",
                             "/system/fonts/NotoSans-Regular.ttf"}) {
        std::error_code error;
        if (std::filesystem::is_regular_file(path, error)) {
            imgui_config.font_path = path;
            break;
        }
    }
#else
    imgui_config.font_path = "C:\\Windows\\Fonts\\segoeui.ttf";
#endif
    imgui_config.font_size = 16.0f;
    imgui_config.enable_keyboard_nav = true;
    imgui_config.enable_docking = false;  // game HUD, no editor layout
    imgui_config.startup_log = startup_log;
    AppRuntime::initImGui(imgui_config);
    AppRuntime::loadPlayerOperatorFonts();

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
    // Show the window BEFORE the project loads: a slow or failing load must not
    // look like "the app never started". Pump events so the OS does not mark
    // the window as not responding while the scene is read.
    SDL_ShowWindow(window);
    SDL_PumpEvents();
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
    // The launcher produces UI frames before any project is opened.
    const auto startup = viewer->initializeBackend();
    if (!startup.ok) {
#ifdef __ANDROID__
        logAndroidStartupError("Player Vulkan backend", startup.error.c_str());
#endif
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "RayTrophi Player",
                                 startup.error.c_str(), window);
        viewer.reset();
        rtapi::unbind();
        state.reset();
        AppRuntime::destroyMainWindowAndImGui();
        AppRuntime::shutdownPlatform();
        return 1;
    }
#ifdef __ANDROID__
    logAndroidStartup("Player Vulkan backend initialized");
#endif
#endif
    if (!launch_project.empty()) {
        SCENE_LOG_INFO("[Player] Window shown, opening project");
        const auto loaded = rtapi::openProject(launch_project);
        if (!loaded.ok) {
            SCENE_LOG_WARN("[Player] " + loaded.error);
        }
    }

    std::string ipc_error;
    if (rtipc::start(ipc_error)) {
        SCENE_LOG_INFO("[Player] IPC server started");
#ifdef __ANDROID__
        logAndroidStartup("Player IPC server started");
#endif
    } else {
        SCENE_LOG_WARN("[Player] IPC server failed to start: " + ipc_error);
#ifdef __ANDROID__
        logAndroidStartupError("Player IPC server", ipc_error.c_str());
#endif
    }

    PlayerDisplay display;
    if (!launch_display.empty()) {
        const auto applied = display.setDisplay(launch_display);
        if (!applied.ok) {
            SCENE_LOG_WARN("[Player] Ignoring launch display options: " + applied.error);
        }
    }

    AppRuntime::setPlayerOperatorLaunchOverrides(launch_display);
#ifdef __ANDROID__
    logAndroidStartup("Player display overrides applied");
    {
        // Binds the operator service to this context so saved game settings and the display
        // restore run even while the menu is closed (on Windows the open console does this).
        nlohmann::json ignored;
        rtapi::playerOperator("language.status", nlohmann::json::object(), ignored);
    }
    logAndroidStartup("Player operator preferences loaded");
#endif
#ifdef __ANDROID__
    if (open_library) {
        std::string launcher_error;
        AppRuntime::kidLauncherShow("library", launcher_error);
    }
    logAndroidStartup("Player launcher configured");
#else
    if (open_library) {
        AppRuntime::showPlayerOperator("games");
    }
#endif

    bool running = true;
    bool debug_camera = false;
    float debug_orbit_x = 0.0f, debug_orbit_y = 0.0f;
    float debug_pan_x = 0.0f, debug_pan_y = 0.0f;
    int debug_wheel = 0;
    Uint64 previous_tick = SDL_GetPerformanceCounter();
    AppRuntime::FrameCycle frame_cycle;
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
    bool reportedPresentationFailure = false;
#endif
    AppRuntime::playerSceneLoad().setPresenter([&frame_cycle, &viewer,
                                               last = Clock::time_point{}]() mutable {
#ifdef __ANDROID__
        if (viewer->deviceLost()) {
            return;  // no device to draw with; the main loop shows the native dialog
        }
#endif
        if (!AppRuntime::playerOperatorOpen()) {
            return;
        }
        const auto now = Clock::now();
        if (AppRuntime::playerSceneLoad().stage() != "check_project" &&
            now - last < std::chrono::milliseconds(50)) {
            return;
        }
        last = now;
        SDL_PumpEvents();
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
        if (!AppRuntime::PlayerSwapchainPresenter::instance().prepareUiFrame()) {
            return;
        }
#endif
        frame_cycle.beginUiFrame(false);
        AppRuntime::drawPlayerOperator(renderer);
        frame_cycle.endUiFrame();
        frame_cycle.present();
    });
    SDL_Event event;
    AppRuntime::EventInfo event_info;
#ifdef __ANDROID__
    AppRuntime::PlayerStartupFailureTouch startupFailureTouch;
    AppRuntime::PlayerOperatorTouch operatorTouch;
    StartupNotice startupNotice = StartupNotice::None;
#endif
    auto syncBoundedCameraHome = [&viewer, &state]() {
        auto& cam_ctrl = AppRuntime::boundedCamera();
        if (!viewer->hasScene() || !state->scene.camera || active_model_path.empty()) {
            if (cam_ctrl.home_captured) {
                cam_ctrl.clearHome();
            }
            return;
        }
        if (!cam_ctrl.home_captured || cam_ctrl.home_project != active_model_path) {
            cam_ctrl.resetHome(active_model_path, *state->scene.camera);
        }
    };
    while (running) {
        frame_cycle.beginIteration();
        syncBoundedCameraHome();
        const auto iteration_start = Clock::now();
        while (frame_cycle.pollEvent(event, event_info)) {
#ifdef __ANDROID__
            if (startupFailureTouch.handle(event, window, startupNotice != StartupNotice::None)) {
                frame_cycle.requestRedraw();
                continue;
            }
            const bool fatal_notice = viewer->deviceLost() ||
                                      startupNotice == StartupNotice::UnsupportedGpu;
            if (fatal_notice) {
                if (event_info.quit_requested) {
                    running = false;
                }
                if (event.type == SDL_DROPFILE) {
                    SDL_free(event.drop.file);
                }
                continue;
            }
            AppRuntime::kidLauncherEvent(event);
            if (operatorTouch.handle(event, window)) {
                frame_cycle.requestRedraw();
                continue;
            }
#endif
            if (event_info.quit_requested) {
                running = false;  // Player: no save prompt
            }
            if (event.type == SDL_DROPFILE) {
                const std::string path = event.drop.file ? event.drop.file : "";
                SDL_free(event.drop.file);
                nlohmann::json output;
                const auto loaded = rtapi::playerOperator("library.open", {{"path", path}}, output);
                if (!loaded.ok) {
                    SCENE_LOG_WARN("[Player] " + loaded.error);
                }
                frame_cycle.requestRedraw();
            }
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                frame_cycle.requestRedraw();
            }
            if (event.type == SDL_KEYDOWN &&
                (!ImGui::GetIO().WantCaptureKeyboard ||
                 ((display.resolution_menu || AppRuntime::playerOperatorOpen()) &&
                  (event.key.keysym.sym == SDLK_F2 || event.key.keysym.sym == SDLK_F6 ||
                   event.key.keysym.sym == SDLK_F8 || event.key.keysym.sym == SDLK_ESCAPE ||
                   event.key.keysym.sym == SDLK_AC_BACK))) &&
                display.handleKey(event.key)) {
                frame_cycle.requestRedraw();
            }
            // Debug camera (F5, off at launch). A diagnostic aid, not gameplay
            // input: it drives the same rtapi camera calls camera.orbit /
            // camera.dolly expose over IPC. Left drag orbits, right or middle
            // drag pans, the wheel dollies. A kiosk key gate must keep F5 out.
            if (event.type == SDL_KEYDOWN && !event.key.repeat &&
                event.key.keysym.sym == SDLK_F5 && !AppRuntime::playerOperatorOpen() &&
                !ImGui::GetIO().WantCaptureKeyboard) {
                debug_camera = !debug_camera;
                SCENE_LOG_INFO(std::string("[Player] Debug camera ") +
                               (debug_camera ? "on" : "off"));
            }
            const bool is_keyboard_event =
                (event.type == SDL_KEYDOWN || event.type == SDL_KEYUP);
            const bool ui_blocks_bounded_cam = is_keyboard_event
                ? ImGui::GetIO().WantCaptureKeyboard
                : ImGui::GetIO().WantCaptureMouse;
            if (debug_camera || display.resolution_menu ||
                AppRuntime::playerOperatorOpen() || AppRuntime::playerVideoCardEnabled()) {
                AppRuntime::boundedCamera().rmb_dragging = false;
            }
            bool bounded_cam_consumed = false;
            if (!debug_camera && !display.resolution_menu &&
                !AppRuntime::playerOperatorOpen() &&
                !AppRuntime::playerVideoCardEnabled() &&
                !AppRuntime::playerSceneLoad().busy() &&
                !ui_blocks_bounded_cam) {
                bounded_cam_consumed = AppRuntime::boundedCamera().handleEvent(
                    event, Paint::heldBalloon() >= 0);
                if (bounded_cam_consumed) {
                    frame_cycle.requestRedraw();
                }
            }
            // Direct touch/mouse hit test for the floating 3D gift box (required on Android where
            // SDL_FINGERDOWN bypasses ImGui::InvisibleButton).
            bool gift_box_consumed = false;
            if (!bounded_cam_consumed && !debug_camera && !display.resolution_menu &&
                !AppRuntime::playerOperatorOpen() && !AppRuntime::playerSceneLoad().busy() &&
                Paint::balloonMilestones().gift_box_active) {
                float hit_nx = -1.0f;
                float hit_ny = -1.0f;
                if (event.type == SDL_FINGERDOWN) {
                    hit_nx = event.tfinger.x;
                    hit_ny = event.tfinger.y;
                } else if (event.type == SDL_MOUSEBUTTONDOWN &&
                           event.button.button == SDL_BUTTON_LEFT && window) {
                    int win_w = 0;
                    int win_h = 0;
                    SDL_GetWindowSize(window, &win_w, &win_h);
                    if (win_w > 0 && win_h > 0) {
                        hit_nx = static_cast<float>(event.button.x) / static_cast<float>(win_w);
                        hit_ny = static_cast<float>(event.button.y) / static_cast<float>(win_h);
                    }
                }
                if (hit_nx >= 0.0f &&
                    Paint::balloonMilestones().hitTestGiftBoxNormalized(hit_nx, hit_ny)) {
                    std::string gift_out;
                    (void)rtapi::wetPaintDemoCommand("claim_gift", "{}", gift_out);
                    gift_box_consumed = true;
                    frame_cycle.requestRedraw();
                }
            }
            // Mouse and future tracked pointers share the balloon interaction service.
            if (!bounded_cam_consumed && !gift_box_consumed) {
                AppRuntime::queuePlayerBalloonInput(event, window,
                    !debug_camera && !display.resolution_menu &&
                    !AppRuntime::playerOperatorOpen() && !ImGui::GetIO().WantCaptureMouse);
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
        if (AppRuntime::flushPlayerBalloonInput(!debug_camera && !display.resolution_menu &&
                !AppRuntime::playerOperatorOpen() && !ImGui::GetIO().WantCaptureMouse &&
                (SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_FOCUS) != 0)) {
            frame_cycle.requestRedraw();
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
        AppRuntime::tickPlayerOperator();
        if (AppRuntime::playerOperatorOpen()) {
            frame_cycle.requestRedraw();
        }
        // Window/fullscreen changes affect presentation only. F2 and explicit
        // resolution/render_scale changes resize the render targets explicitly.
        if (display.applyTargets(state->ctx, *viewer)) {
            frame_cycle.requestRedraw();
        }

        if (!running) {
            break;
        }
        PlayerDisplay::IterationBreakdown breakdown;
        const auto pump_start = Clock::now();
        breakdown.input_ms = millisecondsBetween(iteration_start, pump_start);
        frame_cycle.pumpMainThreadWork();
        breakdown.pump_ms = millisecondsBetween(pump_start, Clock::now());

        // Game clock: the shared simulation clock advances on wall time whether
        // or not the window is focused (a kiosk keeps playing); rendering below
        // stays foreground/capture gated.
        const Uint64 now_tick = SDL_GetPerformanceCounter();
        const float dt_seconds = static_cast<float>(
            static_cast<double>(now_tick - previous_tick) /
            static_cast<double>(SDL_GetPerformanceFrequency()));
        previous_tick = now_tick;
        syncBoundedCameraHome();
        const bool gameplay_paused =
#ifdef __ANDROID__
            AppRuntime::kidLauncherPaused();
#else
            false;
#endif
        if (!gameplay_paused && !debug_camera && viewer->hasScene() && state->scene.camera) {
            if (AppRuntime::boundedCamera().tick(
                    dt_seconds, *state->scene.camera, AppRuntime::playerVideoCardEnabled())) {
                g_camera_dirty = true;
                viewer->requestFrame();
                frame_cycle.requestRedraw();
            }
        }
        const auto advance_start = Clock::now();
#ifdef __ANDROID__
        // The pause screen and the library freeze the game clock (kid launcher).
        if (!AppRuntime::kidLauncherPaused()) {
            viewer->advanceRuntime(dt_seconds);
        }
#else
        viewer->advanceRuntime(dt_seconds);
#endif
        if (AppRuntime::playerSceneLoad().busy() &&
            AppRuntime::playerSceneLoad().stage() == "first_frame") {
            // Apply saved gameplay settings before
            // Frame 0 renders so material_preview builds once without glass.
            AppRuntime::tickPlayerGameSettings(state->ctx);
        }
        breakdown.advance_ms = millisecondsBetween(advance_start, Clock::now());
        display.recordIteration(breakdown.advance_ms);
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
        const bool minimized = frame_cycle.isMinimized();
        const bool game_design_loading =
            AppRuntime::playerSceneLoad().busy() &&
            AppRuntime::playerSceneLoad().stage() == "game_design";
        bool image_updated = false;
        if (render_requested && !game_design_loading && (!minimized || capture_requested) &&
            (frame_cycle.isForeground() || capture_requested || display.continuous ||
             AppRuntime::playerSceneLoad().busy())) {
            const auto render_start = Clock::now();
            image_updated = viewer->render(original_surface, raytrace_texture);
            breakdown.render_ms = millisecondsBetween(render_start, Clock::now());
            breakdown.rendered = true;
            const auto& telemetry = viewer->lastTelemetry();
            if (telemetry.available) {
                breakdown.backend_record_ms = telemetry.cpu_record_ms;
                breakdown.backend_slot_wait_ms = telemetry.slot_wait_ms;
                breakdown.backend_submit_ms = telemetry.submit_ms;
            }
            display.recordRender(viewer->lastRenderMs(), viewer->lastUploadMs(), telemetry);
            frame_cycle.requestRedraw();
        } else if (game_design_loading) {
            // While game_design waits for collider SDF bakes, keep the loading
            // overlay animating without submitting occluded 3D raster frames.
            frame_cycle.requestRedraw();
        }
#ifdef __ANDROID__
        // A fatal notice (unsupported GPU, device lost) cannot be drawn through the swapchain:
        // show ONE native dialog, then quit. The status read is throttled; a lost device is
        // checked every iteration (cheap flag).
        if (!reportedPresentationFailure) {
            static auto last_check = Clock::time_point{};
            const auto check_now = Clock::now();
            if (viewer->deviceLost() ||
                millisecondsBetween(last_check, check_now) >= 250.0) {
                last_check = check_now;
                const StartupNoticeInfo fatal = classifyStartupNotice(*viewer);
                if (isFatalStartupNotice(fatal.kind)) {
                    reportedPresentationFailure = true;  // one dialog only
                    startupNotice = fatal.kind;
                    showFatalStartupDialog(fatal);
                }
            }
        }
        const bool present_blocked =
            isFatalStartupNotice(startupNotice) || viewer->deviceLost();
#else
        constexpr bool present_blocked = false;
#endif
        if (!present_blocked && frame_cycle.shouldPresent(image_updated)
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
            && AppRuntime::PlayerSwapchainPresenter::instance().prepareUiFrame()
#endif
        ) {
            const auto present_start = Clock::now();
            frame_cycle.beginUiFrame();
#ifdef __ANDROID__
            const StartupNoticeInfo notice = classifyStartupNotice(*viewer);
            startupNotice = notice.kind;
            if (display.hud && startupNotice == StartupNotice::None) {
                drawHud(rtipc::isRunning(), *viewer, display);
            } else if (!display.hud && startupNotice == StartupNotice::None &&
                       !AppRuntime::kidLauncherActive() && !AppRuntime::playerConsoleOpen() &&
                       viewer->hasScene()) {
                int render_w = original_surface ? original_surface->w : 0;
                int render_h = original_surface ? original_surface->h : 0;
                if (render_w <= 0 || render_h <= 0) {
                    const auto extent = display.render_extent.scaled(display.render_scale);
                    render_w = extent.first;
                    render_h = extent.second;
                }
                AppRuntime::drawAndroidPerfHud(display.frameStats(), render_w, render_h);
            }
            drawStartupFailureNotice(notice);
            if (!isFatalStartupNotice(startupNotice)) {
                AppRuntime::drawPlayerOperator(renderer);
            }
#else
            if (display.hud) {
                drawHud(rtipc::isRunning(), *viewer, display);
            }
            drawStartupFailureNotice(classifyStartupNotice(*viewer));
            AppRuntime::drawPlayerOperator(renderer);
#endif
            frame_cycle.endUiFrame();

            if (frame_cycle.present()) {
                const auto present_end = Clock::now();
                display.recordPresent(present_end, millisecondsBetween(present_start, present_end));
            }
            breakdown.present_ms = millisecondsBetween(present_start, Clock::now());
        }
#if defined(RT_PLAYER) && (defined(_WIN32) || defined(__ANDROID__))
#ifdef __ANDROID__
        // Device loss is the fatal dialog above; this covers a swapchain that failed alone.
        if (!reportedPresentationFailure &&
            AppRuntime::PlayerSwapchainPresenter::instance().failed()) {
            reportedPresentationFailure = true;
            logAndroidStartupError("Vulkan presentation", "Swapchain presenter failed");
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "RayTrophi Player",
                                     "Vulkan presentation failed. See the Player log.", window);
            SDL_Event quit{};
            quit.type = SDL_QUIT;
            SDL_PushEvent(&quit);
        }
#else
        if (!reportedPresentationFailure &&
            (viewer->deviceLost() || AppRuntime::PlayerSwapchainPresenter::instance().failed())) {
            reportedPresentationFailure = true;
            const auto message = viewer->deviceLost()
                ? g_vulkan_device_lost_msg : "Vulkan presentation failed. See the Player log.";
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "RayTrophi Player",
                                     message.c_str(), window);
        }
#endif
#endif
        if (display.publish(Clock::now())) {
            viewer->setLoopStatus(display.displayStatus());
#ifdef __ANDROID__
            if (display.hud || (!AppRuntime::kidLauncherActive() &&
                                !AppRuntime::playerConsoleOpen() && viewer->hasScene())) {
                frame_cycle.requestRedraw();  // refresh the numbers once a second
            }
#else
            if (display.hud) {
                frame_cycle.requestRedraw();  // refresh the numbers once a second
            }
#endif
        }

        // Pacing. Foreground: the configured rate, or none at all when uncapped
        // (the honest frame cost). An uncapped loop with nothing to render still
        // yields 1 ms so a settled scene does not spin a core. Background: the
        // old 20 Hz IPC/event pump - except in continuous (measurement) mode,
        // where a background window would silently cap the result at 20.
        const auto sleep_start = Clock::now();
        if ((minimized && !capture_requested) ||
            (!frame_cycle.isForeground() && !display.continuous)) {
            std::this_thread::sleep_until(iteration_start + std::chrono::milliseconds(50));
        } else if (display.pacing_hz > 0) {
            std::this_thread::sleep_until(iteration_start +
                std::chrono::microseconds(1000000 / display.pacing_hz));
        } else if (!image_updated) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto iteration_end = Clock::now();
        breakdown.sleep_ms = millisecondsBetween(sleep_start, iteration_end);
        breakdown.total_ms = millisecondsBetween(iteration_start, iteration_end);
        display.recordIterationBreakdown(breakdown);
    }

    // Hide the window immediately on quit so the user does not see a frozen
    // window or Windows wait cursor while Vulkan/scene teardown runs. The HWND
    // stays valid until destroyMainWindowAndImGui() destroys it after Vulkan.
    const auto shutdown_t0 = Clock::now();
    if (window) {
        SDL_HideWindow(window);
    }

    // Same order as Studio: IPC stops before the facade it enqueues into.
    AppRuntime::cancelPlayerBalloonInput();
    rtipc::stop();
    const auto shutdown_t1 = Clock::now();
    AppRuntime::playerSceneLoad().setPresenter({});
    AppRuntime::shutdownPlayerOperator();
    AppRuntime::releasePlayerOperatorUI();
    const auto shutdown_t2 = Clock::now();
    viewer.reset();
    rtapi::unbind();
    const auto shutdown_t3 = Clock::now();
    state.reset();
    const auto shutdown_t4 = Clock::now();

    AppRuntime::destroyMainWindowAndImGui();
    const auto shutdown_t5 = Clock::now();
    AppRuntime::shutdownPlatform();
    const auto shutdown_t6 = Clock::now();

    char shutdown_buf[256];
    std::snprintf(
        shutdown_buf, sizeof(shutdown_buf),
        "[PlayerShutdown] total=%.1f ms (ipc=%.1f, op=%.1f, viewer=%.1f, state=%.1f, win=%.1f, sdl=%.1f)",
        millisecondsBetween(shutdown_t0, shutdown_t6),
        millisecondsBetween(shutdown_t0, shutdown_t1),
        millisecondsBetween(shutdown_t1, shutdown_t2),
        millisecondsBetween(shutdown_t2, shutdown_t3),
        millisecondsBetween(shutdown_t3, shutdown_t4),
        millisecondsBetween(shutdown_t4, shutdown_t5),
        millisecondsBetween(shutdown_t5, shutdown_t6));
    SCENE_LOG_INFO(shutdown_buf);
    g_sceneLog.closeLogFile();
    return 0;
}
