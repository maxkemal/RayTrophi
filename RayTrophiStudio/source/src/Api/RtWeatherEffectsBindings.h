#pragma once

#include "RtIpcTemplates.h"
#include <pybind11/pybind11.h>

namespace rtpy {
void registerWeatherEffectsBindings(pybind11::module_& world);
}

bool dispatchWeatherEffectsIpc(const std::string& method, const nlohmann::json& params,
                              const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result);
