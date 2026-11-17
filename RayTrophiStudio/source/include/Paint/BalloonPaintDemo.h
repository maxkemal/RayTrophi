#pragma once

#include "json.hpp"
#include <string>

struct SceneData;

namespace Paint {

// One core for project startup, scripting and IPC. All calls run on the main thread.
// A project may opt in with <project.rtp>.balloons.json; absence disables the demo.
bool loadBalloonPaintProject(SceneData& scene, const std::string& project_path,
                             std::string& error);
// Release the old atlas while its compute backend is still alive, before reload.
void resetBalloonPaintDemo();
bool balloonPaintCommand(SceneData& scene, const std::string& operation,
                         const nlohmann::json& input, nlohmann::json& output,
                         std::string& error);
// The prepared demo's config in its saved form (no surface list); null when
// no demo is prepared. Paint/GameDesign saves it with the project.
nlohmann::json balloonPaintDesign();

// Droplet colliders: the existing simulation collider assignment, written into
// all four droplet systems as "<object> PaintCollider" bound to the object.
// mode none|aabb|obb|mesh_sdf; sdf_resolution_mode 0/1/2 = 32/64/128 (SDF cooks
// asynchronously). Saved with the particle systems, so not in the game design.
bool setBalloonPaintCollider(SceneData& scene, const std::string& object, const std::string& mode,
                             int sdf_resolution_mode, std::string& error);
// Every collider of the droplet systems with mode, object and cook state.
nlohmann::json balloonPaintColliders(SceneData& scene);
// False (and the collider name) while an enabled SDF collider is still cooking.
bool balloonPaintCollidersReady(SceneData& scene, std::string& waiting);
void tickBalloonPaintDemo(SceneData& scene, float dt);
void observeBalloonPaintFrame(double simulation_ms, float frame_interval_seconds);

} // namespace Paint
