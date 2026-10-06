#pragma once

#include "GranularGpuDispatch.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid::Granular {

// The wave/strain CFL determines numerical resolution, never material stiffness.
// granular_max_solver_substeps remains serialized for old scenes/API round trips,
// but cannot truncate this request or change the authored Young modulus.
inline int solverSubsteps(const ElasticStepInfo& elastic) {
    return std::max(1, elastic.required_substeps);
}

// Authored damping is a multiplier per outer solver step. Numerical subcycling
// must preserve its product; increasing stiffness must not add material drag.
inline float substepDamping(float outer_step_multiplier, int substeps) {
    const float multiplier = std::clamp(outer_step_multiplier, 0.0f, 1.0f);
    return std::pow(multiplier, 1.0f / static_cast<float>(std::max(1, substeps)));
}

inline constexpr float kDampingReferenceDt = 1.0f / 60.0f;

// Granular numerical damping uses the existing 60 Hz preset multiplier as its
// reference. Both outer-step and elastic-substep partitions preserve retention
// over equal physical time. Liquid keeps its existing per-outer-step policy.
inline float timeScaledSubstepDamping(float reference_multiplier, float outer_step_dt,
                                      int substeps) {
    if (outer_step_dt <= 0.0f) {
        return 1.0f;
    }
    const float exponent = outer_step_dt / kDampingReferenceDt /
        static_cast<float>(std::max(1, substeps));
    return std::pow(std::clamp(reference_multiplier, 0.0f, 1.0f), exponent);
}

} // namespace RayTrophiSim::Fluid::Granular
