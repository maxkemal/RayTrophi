#pragma once

#include "json.hpp"

#include <functional>
#include <string>

struct UIContext;

using RtIpcPaintWetQuery =
    std::function<nlohmann::json(UIContext&)>;
using RtIpcPaintWetEnqueue =
    std::function<nlohmann::json(RtIpcPaintWetQuery)>;

// paint.wet_* over IPC (GPU wet paint, PLAYER_GAME_PAINT_ROOM.md step 3/4).
// Returns false when `method` is not one of them, so the caller keeps dispatching.
bool dispatchPaintWetIpc(const std::string& method,
                         const nlohmann::json& params,
                         const RtIpcPaintWetEnqueue& enqueue,
                         nlohmann::json& out_result);
