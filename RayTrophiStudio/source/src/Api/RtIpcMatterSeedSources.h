#pragma once
#include "RtIpcParticle.h"
bool dispatchMatterSeedSourcesIpc(const std::string &method, const nlohmann::json &params,
                                  const RtIpcParticleEnqueue &enqueue, nlohmann::json &output);
