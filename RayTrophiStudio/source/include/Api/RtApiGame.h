#pragma once
#include "json.hpp"
#include <string>
namespace rtapi {
struct Result;
Result gameSessionCommand(const std::string &operation, const nlohmann::json &config,
                          nlohmann::json &output);
Result postGameEvent(const nlohmann::json &event);
Result getGameRules(nlohmann::json &output);
// Studio play test (Game::PlaytestState): operation start | stop | status.
// start: live simulation clock, the balloon demo started when its design says
// autostart (as a Player project load does), particles as billboards only,
// background stepping. stop restores the previous clock mode and stops a demo
// the play test started; paint and balloons stay as they are.
Result gamePlaytest(const std::string &operation, nlohmann::json &output);
} // namespace rtapi
