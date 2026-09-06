#pragma once
#include "RtIpcTemplates.h"
namespace pybind11 { class module_; }
bool dispatchPostExposureIpc(const std::string&,const nlohmann::json&,const RtIpcTemplateEnqueue&,nlohmann::json&);
void registerPostExposurePython(pybind11::module_&);
