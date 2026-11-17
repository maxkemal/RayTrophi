#pragma once
#include "json.hpp"
#include <string>
namespace rtapi {
struct Result;
Result gameSessionCommand(const std::string &operation, const nlohmann::json &config,
                          nlohmann::json &output);
Result postGameEvent(const nlohmann::json &event);
Result getGameRules(nlohmann::json &output);
} // namespace rtapi
