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
} // namespace Game
