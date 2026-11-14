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
           method == "material.of_object" || method == "material.set";
}
} // namespace rtipc_product
