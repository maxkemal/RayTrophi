#pragma once
#include "RtIpcTemplates.h"
namespace pybind11 { class module_; }
bool dispatchRigIpc(const std::string&, const nlohmann::json&, const RtIpcTemplateEnqueue&, nlohmann::json&);
void registerRigPython(pybind11::module_&);
