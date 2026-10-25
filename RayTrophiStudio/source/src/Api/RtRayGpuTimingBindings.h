#pragma once

#include "RtIpcParticle.h"

namespace pybind11 { class module_; }

bool dispatchRayGpuTimingIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcParticleEnqueue& enqueue, nlohmann::json& output);
void bindRayGpuTimingPython(pybind11::module_& render);
