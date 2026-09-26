#pragma once

#include "json.hpp"

#include <functional>
#include <string>

struct UIContext;

using RtIpcKinematicColliderQuery =
    std::function<nlohmann::json(UIContext&)>;
using RtIpcKinematicColliderEnqueue =
    std::function<nlohmann::json(RtIpcKinematicColliderQuery)>;

bool dispatchKinematicColliderIpc(
    const std::string& method,
    const nlohmann::json& params,
    const RtIpcKinematicColliderEnqueue& enqueue,
    nlohmann::json& out_result);
