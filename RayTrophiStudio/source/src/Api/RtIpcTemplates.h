#pragma once

#include "json.hpp"

#include <functional>
#include <string>

struct UIContext;

using RtIpcTemplateQuery = std::function<nlohmann::json(UIContext&)>;
using RtIpcTemplateEnqueue = std::function<nlohmann::json(RtIpcTemplateQuery)>;

bool dispatchTemplateIpc(const std::string& method,
                         const nlohmann::json& params,
                         const RtIpcTemplateEnqueue& enqueue,
                         nlohmann::json& out_result);
