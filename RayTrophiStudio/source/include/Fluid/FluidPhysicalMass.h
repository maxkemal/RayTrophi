#pragma once

#include "Fluid/APICFluidSolver.h"

#include <cstddef>

namespace RayTrophiSim {
struct SubstanceProfile;
namespace Fluid {

struct FluidPhysicalMassStats {
    std::size_t initialized_particles = 0;
    double rest_mass_kg = 0.0;
    double current_mass_kg = 0.0;
};

// Initializes only missing/invalid parcel masses. Exact masses supplied by a
// phase transfer are never overwritten. Untagged particles use the domain's
// chemistry preset; tagged particles use the canonical substance table.
FluidPhysicalMassStats ensureFluidParticleRestMasses(
    FluidParticles& particles,
    FluidChemistryPreset chemistry_preset,
    float voxel_size,
    int particles_per_cell);

const SubstanceProfile* resolveFluidSubstanceProfile(
    uint32_t substance_tag,
    FluidChemistryPreset chemistry_preset);

} // namespace Fluid
} // namespace RayTrophiSim
