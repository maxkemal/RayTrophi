#pragma once
#include "RtIpcTemplates.h"
bool dispatchUpscaleIpc(const std::string&, const nlohmann::json&,
                       const RtIpcTemplateEnqueue&, nlohmann::json&);
