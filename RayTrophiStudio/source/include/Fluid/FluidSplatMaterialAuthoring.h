#pragma once

#include <string>

namespace RayTrophiSim {

class ParticleSimulationSystem;
struct SimulationGridDomainDesc;

namespace Fluid {

enum class SplatMaterialAuthoringStatus {
    Applied,
    DomainNotFound,
    MaterialNotFound
};

struct SplatMaterialAuthoringResult {
    SplatMaterialAuthoringStatus status =
        SplatMaterialAuthoringStatus::DomainNotFound;
    bool changed = false;
    int material_id = -1;

    bool ok() const {
        return status == SplatMaterialAuthoringStatus::Applied;
    }
};

// Empty material_name clears the explicit override. Built-in icospheres then
// use the scene fallback; scene-mesh splats recover their per-face materials.
SplatMaterialAuthoringResult setFluidSplatMaterial(
    ParticleSimulationSystem& simulation,
    const std::string& domain_name,
    const std::string& material_name);

std::string fluidSplatMaterialName(const SimulationGridDomainDesc& domain);

} // namespace Fluid
} // namespace RayTrophiSim
