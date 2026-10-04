#pragma once

#include "Vec3.h"

namespace RayTrophiSim::Fluid {

struct MatterPhaseSettings {
    bool override_enabled = false;
    // Relative to the padded logical domain origin, so object/gizmo movement
    // carries both phase boxes without changing authored phase settings.
    Vec3 offset_min = Vec3(0.0f);
    Vec3 offset_max = Vec3(1.0f);
    float voxel_size = 0.1f;
};

} // namespace RayTrophiSim::Fluid
