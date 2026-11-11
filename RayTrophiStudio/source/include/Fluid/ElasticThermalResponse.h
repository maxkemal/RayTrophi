#pragma once

#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid::Granular {

// Power zero leaves a coefficient unchanged; power one follows the existing
// softening field. Shared semantics with elasticThermalScale in the shader.
inline float elasticThermalScale(float soft, float power) {
    if (!(power > 0.0f)) {
        return 1.0f;
    }
    const float bounded = std::clamp(soft, 0.0f, 1.0f);
    return power == 1.0f ? bounded : std::pow(bounded, power);
}

} // namespace RayTrophiSim::Fluid::Granular
