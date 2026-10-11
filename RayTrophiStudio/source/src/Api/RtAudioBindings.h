#pragma once
#include "RtIpcTemplates.h"
namespace pybind11 { class module_; }

namespace rtpy {
void registerAudioBindings(pybind11::module_& module);
}
bool dispatchAudioIpc(const std::string& method, const nlohmann::json& params,
                      const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result);
