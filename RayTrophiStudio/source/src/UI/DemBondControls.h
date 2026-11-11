#pragma once
#include <json.hpp>
namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
namespace Fluid {
void drawDemBondControls(const SimulationGridDomainDesc& domain,
                         const SimulationGridDomainState* state);
bool editDemBondSeed(nlohmann::json& settings);
}
}
