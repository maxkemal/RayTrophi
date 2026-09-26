#pragma once

#include "Fluid/FluidThermalLiquid.h"
#include "ParticleSimulation.h"

#include <string>
#include <vector>

namespace ForceFieldUI {
namespace FluidThermalUI {

void drawBoundaryOverride(
    RayTrophiSim::SimulationGridDomainDesc& domain,
    float world_ambient_kelvin,
    float world_oxygen);

bool drawDomainControls(
    RayTrophiSim::Fluid::APICSolverParams& params,
    const RayTrophiSim::Fluid::ThermalLiquidStats* stats,
    float ambient_kelvin,
    const std::vector<RayTrophiSim::SimulationFlowSourceDesc>& sources,
    int domain_index);

void drawSourceControls(
    RayTrophiSim::SimulationFlowSourceDesc& source,
    float ambient_kelvin,
    const RayTrophiSim::Fluid::APICSolverParams& params);

std::string uniqueSourceName(
    const std::string& base_name,
    const std::vector<RayTrophiSim::SimulationFlowSourceDesc>& sources);

} // namespace FluidThermalUI
} // namespace ForceFieldUI
