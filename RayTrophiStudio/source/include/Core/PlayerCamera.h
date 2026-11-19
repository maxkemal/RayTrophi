#pragma once
#include "Api/RtApi.h"
#include "json.hpp"
#include <memory>
#include <SDL.h>

namespace AppRuntime {
// One process owner for device inventory, memory-only preview and the existing bridge.
class PlayerCamera {
  public:
    PlayerCamera();
    ~PlayerCamera();
    rtapi::Result command(const std::string &operation, const nlohmann::json &config);
    nlohmann::json status();
    void tick();
    SDL_Texture *preview(SDL_Renderer *renderer);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace AppRuntime
