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
// phase transfer are never overwritten. Untagged particles are the domain's
// default substance; tagged particles use their own entry in the library.
// Granular uses dry density; Fluid uses liquid density. Auto follows the
// domain legacy regime, matching CPU/GPU partition resolution.
FluidPhysicalMassStats ensureFluidParticleRestMasses(
    FluidParticles& particles,
    const SubstanceProfile* domain_substance,
    float voxel_size,
    int particles_per_cell, bool legacy_granular = false);

const SubstanceProfile* resolveFluidSubstanceProfile(
    uint32_t substance_tag,
    const SubstanceProfile* domain_substance);

float fluidParticleRestMassKg(uint32_t substance_tag,
                             const SubstanceProfile* domain_substance,
                             float voxel_size, int particles_per_cell,
                             MatterConstitutiveModel model = MatterConstitutiveModel::Auto,
                             bool legacy_granular = false);

} // namespace Fluid
} // namespace RayTrophiSim
