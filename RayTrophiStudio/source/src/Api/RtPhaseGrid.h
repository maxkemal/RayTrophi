#pragma once

#include "RtIpcTemplates.h"
namespace pybind11 { class module_; }

namespace rtapi {
nlohmann::json getPhaseGrids(const std::string& domain);
nlohmann::json setPhaseGrid(const std::string& domain, const std::string& phase,
                           bool inherit, const nlohmann::json& bounds_min,
                           const nlohmann::json& bounds_max, float voxel);
}

namespace rtpy {
void registerPhaseGridBindings(pybind11::module_& fluid);
}

bool dispatchPhaseGridIpc(const std::string& method, const nlohmann::json& params,
                          const RtIpcTemplateEnqueue& enqueue, nlohmann::json& out);
