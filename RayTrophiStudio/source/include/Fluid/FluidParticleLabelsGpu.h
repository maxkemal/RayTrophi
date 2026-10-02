#pragma once

#include "FluidParticleLabels.h"

namespace RayTrophiSim {
class SimulationComputeContext;
struct SimulationGridDomainComputeBuffers;
struct SimulationGridDomainState;

namespace Fluid {

// Same-frame Vulkan classifier. Returns false without changing host flags when
// unsupported, allocation/dispatch fails, or a bin overflows; callers then use
// updateParticleLabels as the exact CPU fallback.
bool updateParticleLabelsGpu(SimulationGridDomainState& state,
                             SimulationComputeContext* compute,
                             SimulationGridDomainComputeBuffers& buffers,
                             ParticleLabelStepStats& stats);

} // namespace Fluid
} // namespace RayTrophiSim
