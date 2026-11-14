#pragma once
#include "RtIpcTemplates.h"
namespace pybind11 { class module_; }

namespace rtpy {
void registerObjectSubstanceBindings(pybind11::module_& substance);
}
bool dispatchObjectSubstanceIpc(const std::string& method, const nlohmann::json& params,
                                const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result);
