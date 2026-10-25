#pragma once

#include "Api/RtApi.h"
#include "Fluid/FluidViewResolver.h"
#include "json.hpp"

namespace rtapi {
Result inspectMatterFogFields(const std::string& domain, nlohmann::json& output);
Result getPhaseOutput(const std::string& domain,
    RayTrophiSim::Fluid::MatterPhaseOutput& output);
Result setPhaseOutput(const std::string& domain,
    const RayTrophiSim::Fluid::MatterPhaseOutput& output);
}
