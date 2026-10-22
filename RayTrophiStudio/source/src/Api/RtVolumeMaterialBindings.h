#pragma once

#include "RtIpcTemplates.h"

namespace pybind11 { class module_; }

bool dispatchVolumeMaterialIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcTemplateEnqueue& enqueue, nlohmann::json& out);
void registerVolumeMaterialPython(pybind11::module_& module);
