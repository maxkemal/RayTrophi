#pragma once
#include "RtIpcTemplates.h"
namespace pybind11 { class module_; }
namespace rtpy {
void registerLensWeatherBindings(pybind11::module_& camera);
}
bool dispatchLensWeatherIpc(const std::string& method, const nlohmann::json& params,
                           const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result);

namespace rt_lens_weather {
nlohmann::json invoke(UIContext&, const std::string&, const nlohmann::json&);
}
