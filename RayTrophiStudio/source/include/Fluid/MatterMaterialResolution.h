#pragma once

#include "json.hpp"
#include <string>

namespace RayTrophiSim {
struct SubstanceProfile;
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;

namespace Fluid {

struct SubstanceViscosityResolution {
    float requested = 0.0f;
    float effective = 0.0f;
    float voxel_size = 0.0f;
    double reference_diffusion = 0.0;
    const char* reason = "active";
};

// The existing domain viscosity policy, shared by the solver and diagnostics.
SubstanceViscosityResolution resolveSubstanceViscosity(const SubstanceProfile& profile,
    float voxel_size, bool thermal_liquid_enabled, bool granular_enabled);

// Read-only configuration and synchronized-carrier snapshot, not a GPU execution
// measurement. On failure output is untouched and error explains the invalid input.
bool inspectMatterMaterialResolution(const SimulationGridDomainDesc& domain,
    const SimulationGridDomainState* state, nlohmann::json& output, std::string& error);

} // namespace Fluid
} // namespace RayTrophiSim
