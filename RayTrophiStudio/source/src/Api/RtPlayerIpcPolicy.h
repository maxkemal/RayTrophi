#pragma once

#include <string>

// One target policy for transport identity, dispatch and discovery registration.
// Expand this list only when the corresponding Player service is connected.
namespace rtipc_product {
#ifdef RT_PLAYER
inline constexpr bool isPlayer = true;
inline constexpr const char* appName = "RayTrophi Player";
inline constexpr const wchar_t* pipeName = LR"(\\.\pipe\RayTrophiPlayer)";
// POSIX/Android: abstract unix socket name (no leading NUL here; the transport adds it).
// Reach it from the PC with: adb forward tcp:<port> localabstract:RayTrophiPlayer
inline constexpr const char* localSocketName = "RayTrophiPlayer";
#else
inline constexpr bool isPlayer = false;
inline constexpr const char* appName = "RayTrophi Studio";
inline constexpr const wchar_t* pipeName = LR"(\\.\pipe\RayTrophiStudio)";
inline constexpr const char* localSocketName = "RayTrophiStudio";
#endif

inline bool supportsMethod(const std::string& method) {
    return !isPlayer || method == "agent.discover" || method == "agent.describe" ||
           method == "agent.list_methods" || method == "agent.search_capabilities" ||
           method == "agent.get_state_summary" || method == "project.open" ||
           method == "project.path" || method == "camera.get" || method == "camera.orbit" ||
           method == "camera.dolly" || method == "viewport.capture" ||
           method == "viewport.get_screenshot" || method == "viewport.frame_telemetry" ||
           method == "viewport.status" ||
           // RayFusion probe field + sky bake counters (read-only): the Player
           // builds the same probe field, and a flicker there must be measurable.
           method == "rayfusion.probe_field" || method == "rayfusion.core_status" ||
           method == "rayfusion.scene_as" || method == "rayfusion.bake_counters" ||
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
           method == "player.operator" ||
           // The runtime viewer consumes the shared world dirty flags before rendering.
           method == "world.get" || method == "world.set_mode" ||
           method == "world.set_background_color" ||
           // Shared scene grading, consumed by the existing raster post pass.
           method == "post.get" || method == "post.set_exposure" ||
           method == "post.set_gamma" || method == "post.set_saturation" ||
           method == "post.set_color_temperature" || method == "post.set_tone_mapping" ||
           method == "post.set_vignette_enabled" || method == "post.set_vignette_strength" ||
           method == "post.get_exposure" || method == "post.configure_exposure" ||
           method == "post.reset_exposure" ||
           // Per-stage host timings of the shared raster frame (read + reset).
           method == "viewport.frame_timings" || method == "viewport.reset_frame_timings" ||
           method == "perf.list" || method == "perf.gpu_kernel_timings" ||
           method == "perf.set_gpu_kernel_timing" ||
           // RT shadow request/readiness: the shared raster pass falls back to
           // cascades without ray query; a test must reach that path on RT GPUs.
           method == "viewport.rt_shadow" || method == "viewport.set_rt_shadow" ||
           // Depth prepass request/effect: a GPU without ray query depends on it
           // (workshop measured 13.4 ms without, 6.3 ms with).
           method == "viewport.raster_depth_prepass" ||
           method == "viewport.set_raster_depth_prepass" ||
           // Shading cost levers for on-device A/B (PLAYER_ANDROID_PERF_PLAN):
           // without them every hypothesis on the tablet cost one APK round.
           // Quality sets the BRDF tier, PCF taps and the shadowed-light budget;
           // preview lighting the light model; TAA its resolve. Lights: the
           // Player services g_lights_dirty (RuntimeSceneViewer), so an
           // intensity of 0 really removes a light from the frame.
           // viewport.set_shading stays out: Solid/Matcap are not built here.
           method == "viewport.quality" || method == "viewport.set_quality" ||
           method == "viewport.preview_lighting" ||
           method == "viewport.set_preview_lighting" ||
           method == "viewport.taa" || method == "viewport.set_taa" ||
           method == "lights.list" || method == "lights.get" ||
           method == "lights.set_intensity" || method == "lights.set_cast_shadows" ||
           method == "lights.set_distance_falloff" ||
           // Scene audio state (read-only): the Player ticks audio on its game clock.
           method == "audio.status" || method == "audio.list" ||
           method == "audio.source_runtime" || method == "audio.mixer_get" ||
           // Playback actions game rules use (a pop sound, a splash). Manual
           // emitters stay silent until played - same as in Studio.
           method == "audio.play" || method == "audio.play_all" || method == "audio.play_at" ||
           method == "audio.stop" || method == "audio.stop_all" ||
           // GPU wet paint (step 3/4): surfaces, dabs, splat rules, reset, coverage.
           method.rfind("paint.wet_", 0) == 0 ||
           method == "input.set_body_poses" || method == "input.body_poses" ||
           method == "input.clear_body_poses" || method == "game.session" ||
           method == "game.rules" || method == "game.post_event" ||
           method == "anim.set_external_pose" || method == "anim.external_poses" ||
           method == "anim.clear_external_pose" || method == "anim.characters" ||
           method == "anim.character" || method == "rig.list_bones" ||
           // Existing Studio playback services; authoring/import remains in Studio.
           method == "anim.clips" || method == "anim.play" || method == "anim.stop" ||
           method == "anim.status" || method == "anim.set_paused" ||
           method == "anim.set_time" || method == "anim.set_speed" ||
           method == "anim.graph_status" ||
           method == "anim.set_graph_param" || method == "anim.trigger_graph_param" ||
           method == "anim.state_machines" || method == "anim.force_state" ||
           method.rfind("physics.collider.proxy_set.", 0) == 0 ||
           method == "physics.collider.proxy.set" || method == "physics.collider.proxy.delete";
}
} // namespace rtipc_product
