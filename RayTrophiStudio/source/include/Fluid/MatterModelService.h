#pragma once

struct SceneData;

#include "MatterTransfer.h"

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
}

namespace RayTrophiSim::Fluid {

bool inspectMatterModels(const SimulationGridDomainDesc& domain,
                         const SimulationGridDomainState* state,
                         bool include_transfer, MatterTransferFrame& frame,
                         std::string& error);

void drawMatterModelSummary(const SimulationGridDomainDesc& domain,
                           const SimulationGridDomainState* state);

// Uses the existing domain-substance service; no separate seed material authority.
bool drawMatterSeedSubstanceControls(const SimulationGridDomainDesc& domain);

void drawMatterOutputControls(::SceneData& scene, const SimulationGridDomainDesc& domain);

void drawMatterPoolControls(const SimulationGridDomainDesc& domain,
                           const SimulationGridDomainState* state);

} // namespace RayTrophiSim::Fluid
