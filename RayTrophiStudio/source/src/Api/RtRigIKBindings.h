#pragma once
#include "RtIpcTemplates.h"
namespace pybind11 {class module_;}
bool dispatchRigIKIpc(const std::string&,const nlohmann::json&,const RtIpcTemplateEnqueue&,nlohmann::json&);
void registerRigIKPython(pybind11::module_& rig);
