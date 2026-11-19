#pragma once
#include "Api/RtApi.h"
#include "json.hpp"
struct UIContext;
namespace AppRuntime {
rtapi::Result playerGameSettingsCommand(UIContext &context, const std::string &operation,
                                        const nlohmann::json &config, nlohmann::json &output);
void tickPlayerGameSettings(UIContext &context);
void resetPlayerGameSettings();
} // namespace AppRuntime
