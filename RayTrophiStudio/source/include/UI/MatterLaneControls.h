#pragma once
namespace RayTrophiSim { struct SimulationGridDomainDesc; }
namespace RayTrophiSim { struct SimulationFlowSourceDesc; }
namespace RayTrophiSim::Fluid {
bool drawMatterLaneControls(const SimulationGridDomainDesc& domain);
bool drawMatterSourcePhaseControl(const SimulationGridDomainDesc& domain,
    SimulationFlowSourceDesc& source);
}
