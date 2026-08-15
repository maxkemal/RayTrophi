#pragma once

#include <cstdint>

namespace RayTrophiSim::Fluid {

// Splat rendering must reuse authored scene materials. These helpers only
// resolve existing MaterialManager entries; they never manufacture a material.
bool isExistingSplatMaterial(int material_id);
uint16_t resolveExistingSplatMaterial(int authored_material_id = -1);

} // namespace RayTrophiSim::Fluid
