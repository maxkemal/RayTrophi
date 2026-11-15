#pragma once

#include <string>

// One target policy for transport identity, dispatch and discovery registration.
// Expand this list only when the corresponding Player service is connected.
namespace rtipc_product {
#ifdef RT_PLAYER
inline constexpr bool isPlayer = true;
inline constexpr const char* appName = "RayTrophi Player";
inline constexpr const wchar_t* pipeName = LR"(\\.\pipe\RayTrophiPlayer)";
#else
inline constexpr bool isPlayer = false;
inline constexpr const char* appName = "RayTrophi Studio";
inline constexpr const wchar_t* pipeName = LR"(\\.\pipe\RayTrophiStudio)";
#endif

inline bool supportsMethod(const std::string& method) {
    return !isPlayer || method == "agent.discover" || method == "agent.describe" ||
           method == "agent.list_methods" || method == "agent.search_capabilities" ||
           method == "agent.get_state_summary" || method == "project.open" ||
           method == "project.path" || method == "camera.get" || method == "camera.orbit" ||
           method == "camera.dolly" || method == "viewport.capture" ||
           method == "viewport.get_screenshot" || method == "viewport.frame_telemetry" ||
           method == "viewport.status" ||
           // Read-only scene values: the simulation clock moves bodies in the
           // Player, and a test that cannot read the simulated pose cannot
           // verify it (rigid motion is baked into vertices, not transforms).
           method == "scene.list_objects" || method == "scene.get_world_transform" ||
           // Materials: the shared Renderer::updateBackendMaterial path uploads to
           // the Player's single viewport device; game rules and paint change them.
           method == "material.list" || method == "material.get" ||
           method == "material.of_object" || method == "material.set" ||
           // Display surface and frame timing (window, render scale, pacing, FPS).
           method == "player.display" || method == "player.set_display" ||
           // Per-stage host timings of the shared raster frame (read + reset).
           method == "viewport.frame_timings" || method == "viewport.reset_frame_timings" ||
           // Scene audio state (read-only): the Player ticks audio on its game clock.
           method == "audio.status" || method == "audio.list" ||
           method == "audio.source_runtime" || method == "audio.mixer_get" ||
           // Playback actions game rules use (a pop sound, a splash). Manual
           // emitters stay silent until played - same as in Studio.
           method == "audio.play" || method == "audio.play_all" ||
           method == "audio.stop" || method == "audio.stop_all" ||
           // GPU wet paint (step 3/4): surfaces, dabs, splat rules, reset, coverage.
           method.rfind("paint.wet_", 0) == 0;
}
} // namespace rtipc_product
