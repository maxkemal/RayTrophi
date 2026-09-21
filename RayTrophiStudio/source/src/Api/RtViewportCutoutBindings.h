#pragma once
#include "RtIpcTemplates.h"
namespace pybind11 { class module_; }
bool dispatchViewportCutoutIpc(const std::string&, const nlohmann::json&, const RtIpcTemplateEnqueue&, nlohmann::json&);
void registerViewportCutoutPython(pybind11::module_&);
