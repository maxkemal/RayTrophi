#pragma once
#include "RtIpcTemplates.h"
bool dispatchRayFusionIpc(const std::string& method, const nlohmann::json& params,
                         const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result);
