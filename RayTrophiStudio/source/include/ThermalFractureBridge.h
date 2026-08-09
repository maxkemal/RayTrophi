#pragma once

#include "MaterialStateField.h"
#include "RigidBodySystem.h"

namespace RayTrophiSim {

float effectiveFractureThreshold(const RigidBodyObject& body,
                                 const MaterialIntegritySummary& summary);

} // namespace RayTrophiSim
