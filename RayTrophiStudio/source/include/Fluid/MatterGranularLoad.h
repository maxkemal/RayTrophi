#pragma once

#include "GranularGpuDispatch.h"

namespace RayTrophiSim::Fluid {
// Same extent-based load proxy as the legacy granular path, restricted to the
// granular model. It is not a support/contact pressure measurement.
Granular::LoadMeasurement measureMatterGranularLoad(const FluidParticles& particles,
    bool legacy_granular, float gravity = 9.81f,
    float density = Granular::kGranularDensity);
} // namespace RayTrophiSim::Fluid
