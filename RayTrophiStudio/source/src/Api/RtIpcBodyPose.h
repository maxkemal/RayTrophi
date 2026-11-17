#pragma once
#include "RtIpcPaintWet.h"

bool dispatchBodyPoseIpc(const std::string &method, const nlohmann::json &params,
                         const RtIpcPaintWetEnqueue &enqueue, nlohmann::json &output);
