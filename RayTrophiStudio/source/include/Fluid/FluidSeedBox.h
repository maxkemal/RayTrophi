#pragma once

#include "ParticleSimulation.h"

namespace RayTrophiSim::Fluid {

// Only a new recipe needs defaults. Restored/API-authored bounds are intentional.
inline void initializeFluidSeedBox(SimulationGridDomainDesc& domain,
    const Vec3& bounds_min, const Vec3& bounds_max) {
    if (domain.fluid_seed_box_initialized) {
        return;
    }
    const Vec3 extent = bounds_max - bounds_min;
    const Vec3 seed_extent(extent.x * 0.45f, extent.y * 0.35f, extent.z * 0.45f);
    const Vec3 center = (bounds_min + bounds_max) * 0.5f;
    const Vec3 seed_center(center.x, bounds_min.y + extent.y * 0.75f, center.z);
    domain.fluid_seed_min = seed_center - seed_extent * 0.5f;
    domain.fluid_seed_max = seed_center + seed_extent * 0.5f;
    domain.fluid_seed_box_initialized = true;
}

} // namespace RayTrophiSim::Fluid
