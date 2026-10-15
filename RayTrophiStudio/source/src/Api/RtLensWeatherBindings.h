#pragma once
#include "RtIpcTemplates.h"
#include <pybind11/pybind11.h>
namespace rtpy {
void registerLensWeatherBindings(pybind11::module_& camera);
}
bool dispatchLensWeatherIpc(const std::string& method, const nlohmann::json& params,
                           const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result);
