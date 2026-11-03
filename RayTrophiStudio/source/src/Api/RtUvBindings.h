// SPDX-License-Identifier: MIT
#pragma once

#include "RtIpcTemplates.h"

namespace pybind11 { class module_; }

void registerUvPython(pybind11::module_& module);
nlohmann::json invokeUv(const std::string& operation, const nlohmann::json& params);
bool dispatchUvIpc(const std::string& method, const nlohmann::json& params,
                   const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result);
