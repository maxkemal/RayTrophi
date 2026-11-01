#pragma once

#include "SimulationWorld.h"

#include <algorithm>
#include <vector>

namespace RayTrophiSim::Fluid {

// Mirror sim_fluid_particle_forces.comp's surface-only Wind predicate.
// Packed force type 0 is Wind; ordinary body forces never sample columns.
inline bool packedForcesNeedFluidColumns(const std::vector<PackedForceField>& fields) {
    return std::any_of(fields.begin(), fields.end(), [](const PackedForceField& field) {
        return field.enabled != 0 && field.type == 0 && field.fluid_surface_drag != 0 &&
            (field.affect_mask & static_cast<uint32_t>(SimulationSystemKind::Fluid)) != 0;
    });
}

inline bool fluidForcesNeedColumns(const SimulationForceFieldComputeBuffer* buffer,
                                  bool atmosphere_wind) {
    return atmosphere_wind || (buffer && buffer->valid() &&
                              buffer->fluid_surface_columns_required);
}

} // namespace RayTrophiSim::Fluid
