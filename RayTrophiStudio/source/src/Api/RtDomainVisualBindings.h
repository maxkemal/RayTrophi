#pragma once

#include "RtIpcParticle.h"
namespace pybind11 {
    class module_;
}

namespace rtpy {
void registerDomainVisualBindings(pybind11::module_& fluid);
}
bool dispatchDomainVisualIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcParticleEnqueue& enqueue, nlohmann::json& output);
