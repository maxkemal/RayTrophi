#pragma once
#include "json.hpp"
#include <string>

struct SceneData;
namespace Game {
bool runtimeCommand(SceneData &scene, const std::string &operation, const nlohmann::json &config,
                    nlohmann::json &output, std::string &error);
void tickRuntime(SceneData &scene, double dt);
void resetRuntime();
bool postExternalEvent(const nlohmann::json &value, std::string &error);

// Studio play test (2026-11-18): Studio runs the room the way the Player does -
// live simulation clock, particles drawn ONLY as billboards (no RT instance
// pools in the raster viewport), and the main loop keeps stepping while the
// window is in the background. Started and stopped by rtapi::gamePlaytest
// (Game panel, IPC game.playtest); the Player is always in this state.
struct PlaytestState {
    bool active = false;
    bool previous_timeline_mode = true;   // restored by stop
    bool started_demo = false;            // the play test started the balloon demo
    double started_seconds = 0.0;         // steady clock, for status
};
PlaytestState &playtest();
inline bool playtestActive() { return playtest().active; }
} // namespace Game
