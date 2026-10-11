#pragma once

#include "ParticleSimulation.h"
#include <string>
#include <vector>

struct SceneData;
namespace RayTrophiSim::Fluid {
struct GasInventoryFrameCapture {
    bool ok = false;
    std::string error;
    std::vector<std::vector<SimulationGridDomainState>> states;
};

// Prepare all runtime snapshots before mutating the timeline cache entry.
GasInventoryFrameCapture captureGasInventoryFrame(::SceneData& scene);
}
