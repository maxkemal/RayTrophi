#pragma once

#include "RtIpcParticle.h"

bool dispatchDomainWallsIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcParticleEnqueue& enqueue, nlohmann::json& output);
