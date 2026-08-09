#pragma once

#include "RigidBodySystem.h"
#include "json.hpp"

namespace RayTrophiSim {

void serializeThermalFracture(const RigidBodyObject& body,
                              nlohmann::json& body_json);
void deserializeThermalFracture(const nlohmann::json& body_json,
                                RigidBodyObject& body);

} // namespace RayTrophiSim
