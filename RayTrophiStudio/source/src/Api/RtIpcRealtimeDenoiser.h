#pragma once
#include "RtIpcTemplates.h"
bool dispatchRealtimeDenoiserIpc(const std::string&, const nlohmann::json&,
    const RtIpcTemplateEnqueue&, nlohmann::json&);
