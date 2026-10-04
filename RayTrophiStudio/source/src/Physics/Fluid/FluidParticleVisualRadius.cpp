#include "Fluid/FluidParticleVisualRadius.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid {

FluidParticleVisualRadius resolveFluidParticleVisualRadius(
    bool granular_enabled,
    bool procedural_spheres,
    int particles_per_cell,
    float radius_factor,
    float size_multiplier) {
    FluidParticleVisualRadius result;
    result.authored_voxels = std::max(radius_factor * size_multiplier, 0.0f);
    result.effective_voxels = result.authored_voxels;

    (void)granular_enabled;
    if (!procedural_spheres) {
        return result;
    }

    constexpr float kFourThirdsPi = 4.1887902047863905f;
    const float count = static_cast<float>(std::max(particles_per_cell, 1));
    result.material_point_voxels = std::cbrt(1.0f / (kFourThirdsPi * count));
    if (result.material_point_voxels > result.effective_voxels) {
        result.effective_voxels = result.material_point_voxels;
        result.material_point_floor_applied = true;
    }
    return result;
}

} // namespace RayTrophiSim::Fluid
