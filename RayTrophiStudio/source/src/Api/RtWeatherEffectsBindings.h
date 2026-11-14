#pragma once

#include "RtIpcTemplates.h"
namespace pybind11 { class module_; }

namespace rtpy {
void registerWeatherEffectsBindings(pybind11::module_& world);
}

bool dispatchWeatherEffectsIpc(const std::string& method, const nlohmann::json& params,
                              const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result);

namespace rtapi { struct Result; }
namespace rt_weather_effects {
void requireWeatherResult(const rtapi::Result& result);
}
