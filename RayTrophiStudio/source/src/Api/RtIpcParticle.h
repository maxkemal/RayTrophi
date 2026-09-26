#pragma once

#include "json.hpp"

#include <functional>
#include <string>

struct UIContext;

using RtIpcParticleQuery =
    std::function<nlohmann::json(UIContext&)>;
using RtIpcParticleEnqueue =
    std::function<nlohmann::json(RtIpcParticleQuery)>;

// particle.* over IPC (particle roadmap Phase 1). Returns false when `method`
// is not a particle method, so the caller keeps dispatching.
bool dispatchParticleIpc(const std::string& method,
                         const nlohmann::json& params,
                         const RtIpcParticleEnqueue& enqueue,
                         nlohmann::json& out_result);
