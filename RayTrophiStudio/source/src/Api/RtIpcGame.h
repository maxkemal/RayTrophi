#pragma once
#include "RtIpcPaintWet.h"
bool dispatchGameIpc(const std::string &method, const nlohmann::json &params,
                     const RtIpcPaintWetEnqueue &enqueue, nlohmann::json &output);
