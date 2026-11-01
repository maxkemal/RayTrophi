#pragma once
#include "RtIpcTemplates.h"
#include "Api/MatterDomainLanesApi.h"
namespace pybind11 { class module_; }
namespace rtpy { void registerMatterLaneBindings(pybind11::module_& fluid); }
bool dispatchMatterLanesIpc(const std::string& method, const nlohmann::json& params,
    const RtIpcTemplateEnqueue& enqueue, nlohmann::json& out);
