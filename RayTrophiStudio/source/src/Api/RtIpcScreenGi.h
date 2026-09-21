#pragma once
#include "RtIpcTemplates.h"
bool dispatchScreenGiIpc(const std::string&,const nlohmann::json&,
    const RtIpcTemplateEnqueue&,nlohmann::json&);
