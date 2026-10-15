#pragma once
#include "RtIpcTemplates.h"
#include <pybind11/pybind11.h>

namespace rtpy {
void registerObjectSubstanceBindings(pybind11::module_& substance);
}
bool dispatchObjectSubstanceIpc(const std::string& method, const nlohmann::json& params,
                                const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result);
