#pragma once

#include "GranularGpuDispatch.h"

namespace RayTrophiSim::Fluid::Granular {

// The wave/strain CFL determines numerical resolution, never material stiffness.
// granular_max_solver_substeps remains serialized for old scenes/API round trips,
// but cannot truncate this request or change the authored Young modulus.
inline int solverSubsteps(const ElasticStepInfo& elastic) {
    return std::max(1, elastic.required_substeps);
}

} // namespace RayTrophiSim::Fluid::Granular
