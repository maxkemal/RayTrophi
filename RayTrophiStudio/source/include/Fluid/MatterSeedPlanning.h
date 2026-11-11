#pragma once
#include "Fluid/MatterSeedSources.h"

namespace FluidSim {
class FluidGrid;
}

namespace RayTrophiSim::Fluid {
// Fast conservative bounds, not an exact birth count or allocation promise.
nlohmann::json planMatterSeeds(const SimulationGridDomainDesc &domain,
                               const MatterSeedMeshResolver &meshes,
                               const FluidSim::FluidGrid &grid, float ambient_kelvin);
bool checkMatterSeedMemory(nlohmann::json &plan, std::string &error);
} // namespace RayTrophiSim::Fluid
