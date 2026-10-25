#pragma once
#include "Fluid/MatterFogFields.h"

namespace RayTrophiSim {
class ParticleSimulationSystem;
class SimulationComputeContext;
namespace Fluid {
// On-demand physical snapshot shared by diagnostics and the live VDB publisher.
bool prepareMatterFogSnapshot(ParticleSimulationSystem& runtime, std::size_t index,
    const SimulationComputeContext& compute, MatterFogFields& output,
    std::string& snapshot, std::string& error);
}
}
