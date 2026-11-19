#pragma once

#include "json.hpp"
#include <string>

struct SceneData;

namespace Paint {

// The game setup a project carries (project key "game_design"): which objects
// are wet paint surfaces, each surface's paint model settings, how particle
// contacts become paint, and the balloon demo's configuration. The paint itself
// is session state and is never saved.
//
// There is no second copy of this state. Saving reads the live wet paint atlas
// and the live demo; loading queues the document and the shared simulation
// clock applies it on its first tick, in Studio and in the Player alike. The
// Studio Game Design panel and paint.wet_* / paint.wet_demo over IPC edit that
// same live state.

// Resize every registered surface through the Paint Edit texture service.
// Rebuilds the atlas and clears transient wet paint; settings remain unchanged.
bool resizeGamePaintSurfaces(SceneData& scene, int resolution, std::string& error);

// Null when the scene has no wet paint surface and no prepared demo.
nlohmann::json saveGameDesign();
// Project load: the document to apply at the next simulation clock tick.
void queueGameDesign(const nlohmann::json& design);
// Project replacement: drop a queued document and the last apply report.
void resetGameDesign();
// True between queueGameDesign and the tick that applies it.
bool gameDesignPending();
// Remove references to deleted flat nodes. Shared by UI, API and runtime ticks.
bool reconcileGameDesign(SceneData& scene);
// Called by the shared simulation clock every tick; does nothing unless a
// document is queued. Applies what it can and reports the rest.
void tickGameDesign(SceneData& scene);
// Empty: nothing loaded. Otherwise "applied: ..." or the errors of the last apply.
std::string gameDesignStatus();

} // namespace Paint
