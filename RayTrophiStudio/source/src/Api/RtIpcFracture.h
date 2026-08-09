#pragma once

#include "json.hpp"

#include <functional>
#include <string>

struct UIContext;

using RtIpcFractureQuery =
    std::function<nlohmann::json(UIContext&)>;
using RtIpcFractureEnqueue =
    std::function<nlohmann::json(RtIpcFractureQuery)>;

bool dispatchFractureIpc(const std::string& method,
                         const nlohmann::json& params,
                         const RtIpcFractureEnqueue& enqueue,
                         nlohmann::json& out_result);
