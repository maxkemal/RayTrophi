#pragma once

#include "Api/SimulationTransportApi.h"
#include "RtIpcTemplates.h"

namespace pybind11 { class module_; }
namespace rtpy { void registerSimulationTransportBindings(pybind11::module_& timeline,
                                                         pybind11::module_& sim); }
bool dispatchSimulationTransportIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcTemplateEnqueue& enqueue, nlohmann::json& result);
