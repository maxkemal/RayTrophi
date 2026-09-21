#pragma once
#include "RtIpcTemplates.h"
bool dispatchReflectionIpc(const std::string&, const nlohmann::json&,
    const RtIpcTemplateEnqueue&, nlohmann::json&);
