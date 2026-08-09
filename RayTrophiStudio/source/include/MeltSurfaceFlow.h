#pragma once

#include "Vec3.h"

#include <cstdint>
#include <vector>

namespace RayTrophiSim {

struct MeltSurfaceFlowSettings {
    float viscosity = 0.5f;
    float maximum_height_loss = 0.85f;
    float maximum_lateral_gain = 1.50f;
    uint32_t smoothing_passes = 2;
};

// Quasi-static solid-shell flow used before mass leaves the mesh for APIC.
// It is rest-pose based (timeline/cache safe), spreads only through connected
// triangle neighbours and approximately preserves the deformed envelope volume.
bool solveMeltSurfaceFlow(const std::vector<Vec3>& rest_positions,
                          const std::vector<uint32_t>& triangle_indices,
                          const std::vector<float>& melt,
                          const std::vector<float>& local_mass_fraction,
                          const MeltSurfaceFlowSettings& settings,
                          std::vector<Vec3>& out_positions);

} // namespace RayTrophiSim
