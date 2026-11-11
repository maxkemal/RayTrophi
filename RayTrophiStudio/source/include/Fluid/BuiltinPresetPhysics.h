#pragma once

#include <vector>

namespace RayTrophiSim {
struct SubstanceProfile;
namespace Fluid {

// Final physical defaults for the built-in catalogue, before publication.
// Does not edit project overrides or an already running domain's tuning.
void configureBuiltinPresetPhysics(std::vector<SubstanceProfile>& profiles);

} // namespace Fluid
} // namespace RayTrophiSim
