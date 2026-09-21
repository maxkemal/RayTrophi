#pragma once
#include "RtIpcTemplates.h"
namespace pybind11 { class module_; }
bool dispatchClipBindingIpc(const std::string&, const nlohmann::json&, const RtIpcTemplateEnqueue&, nlohmann::json&);
void registerClipBindingPython(pybind11::module_&);
