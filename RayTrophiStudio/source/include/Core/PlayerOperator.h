#pragma once
#include "Api/RtApi.h"
#include "json.hpp"
struct SDL_Window;
struct SDL_Renderer;
struct SDL_Texture;
struct UIContext;
namespace AppRuntime {
// Main-thread service, shared by the panel, Python and IPC.
rtapi::Result playerOperatorCommand(UIContext &context, const std::string &operation,
                                    const nlohmann::json &config, nlohmann::json &output);
SDL_Texture *playerOperatorPreview(SDL_Renderer *renderer);
void setPlayerOperatorLaunchOverrides(const nlohmann::json &display);
void tickPlayerOperator();
void shutdownPlayerOperator();
} // namespace AppRuntime
