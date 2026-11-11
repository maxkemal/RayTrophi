#pragma once

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
namespace Fluid {
void drawMatterMaterialResolution(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState* state);
}
}
