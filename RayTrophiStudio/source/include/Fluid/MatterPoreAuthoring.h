#pragma once

#include "MatterPoreExchange.h"
#include <json.hpp>

namespace RayTrophiSim::Fluid {
nlohmann::json matterPoreParamsToJson(const MatterPoreParams& params);
MatterPoreParams matterPoreParamsFromJson(const nlohmann::json& settings);
bool patchMatterPoreParams(const nlohmann::json& patch, MatterPoreParams& params,
                          std::string& error);
void drawMatterPoreControls(const std::string& domain_name, bool grains = false);
} // namespace RayTrophiSim::Fluid
