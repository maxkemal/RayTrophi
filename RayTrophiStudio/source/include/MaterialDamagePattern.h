#pragma once

#include <cstdint>
#include <string>

namespace RayTrophiSim {

// Applies stable UV-space material structure to a derived visual damage channel.
// It never changes fuel, temperature or conserved mass in the simulation state.
float applyMaterialDamagePattern(const std::string& substance,
                                 uint32_t texel, int resolution, float value);

} // namespace RayTrophiSim
