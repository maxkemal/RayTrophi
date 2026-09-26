#pragma once

#include "KinematicColliderSource.h"
#include "json.hpp"

#include <string>

namespace RayTrophiSim {

nlohmann::json serializeKinematicColliders(
    const KinematicColliderRegistry& registry);

bool deserializeKinematicColliders(
    const nlohmann::json& data,
    KinematicColliderRegistry& registry,
    std::string& error);

} // namespace RayTrophiSim
