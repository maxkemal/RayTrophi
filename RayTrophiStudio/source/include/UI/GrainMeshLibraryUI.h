#pragma once

struct SceneData;

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
namespace Fluid {

void drawGrainMeshNearControls(SceneData& scene, const SimulationGridDomainDesc& domain);

} // namespace Fluid
} // namespace RayTrophiSim
