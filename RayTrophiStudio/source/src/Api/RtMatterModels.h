#pragma once

#include "RtIpcTemplates.h"
namespace pybind11 { class module_; }

namespace rtapi {
nlohmann::json getMatterModels(const std::string& domain, bool include_transfer);
nlohmann::json setMatterPoreExchange(const std::string& domain, const nlohmann::json& patch);
nlohmann::json runGrainReferenceProbe(const nlohmann::json& params);
nlohmann::json matterGrainSettings(const std::string& domain, const nlohmann::json& patch,
                                 bool write);
}

namespace rtpy {
void registerMatterModelBindings(pybind11::module_& fluid);
}

bool dispatchMatterModelIpc(const std::string& method, const nlohmann::json& params,
                           const RtIpcTemplateEnqueue& enqueue, nlohmann::json& out);
