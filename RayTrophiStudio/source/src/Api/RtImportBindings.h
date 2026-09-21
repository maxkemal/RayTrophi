#pragma once
#include "RtIpcTemplates.h"
namespace pybind11 { class module_; }
bool dispatchImportIpc(const std::string&, const nlohmann::json&,
                       const RtIpcTemplateEnqueue&, nlohmann::json&);
void registerImportPython(pybind11::module_&);
