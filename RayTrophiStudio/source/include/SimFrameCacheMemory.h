#pragma once

#include "ParticleSimulation.h"
#include "MatterGridFrameCache.h"

#include <cstddef>
#include <vector>

namespace RayTrophiSim::SimFrameCacheMemory {

template <typename T>
std::size_t allocationBytes(const std::vector<T>& values) {
    return values.capacity() * sizeof(T);
}

// CachedGridDomain::meta owns deep copies of these arrays. Counting only the
// compressed scalar fields misses the fluid particle state on every frame.
inline std::size_t metaAllocationBytes(const SimulationGridDomainState& state) {
    const auto& grid = state.grid;
    std::size_t bytes = 0;
    bytes += allocationBytes(state.gas_phase_mass_kg);
    bytes += allocationBytes(state.gas_phase_energy_j);
    bytes += allocationBytes(state.gas_phase_condensate_kg);
    bytes += allocationBytes(state.gas_phase_rain_kg);
    bytes += allocationBytes(state.gas_phase_rain_energy_j);
    bytes += allocationBytes(state.gas_phase_foreign_kg);
    bytes += allocationBytes(state.gas_rain_landing);
    bytes += SimFrameCache::fluidGridAllocationBytes(grid);
    bytes += SimFrameCache::fluidGridAllocationBytes(state.matter_liquid_grid);

    const auto& particles = state.particles;
    bytes += allocationBytes(particles.particle_id);
    bytes += allocationBytes(particles.birth_source_uid);
    bytes += allocationBytes(particles.position);
    bytes += allocationBytes(particles.velocity);
    bytes += allocationBytes(particles.affine);
    bytes += allocationBytes(particles.flags);
    bytes += allocationBytes(particles.mass_fraction);
    bytes += allocationBytes(particles.rest_mass_kg);
    bytes += allocationBytes(particles.pore_water_mass_kg);
    bytes += allocationBytes(particles.pore_capacity_kg);
    bytes += allocationBytes(particles.pore_porosity);
    bytes += allocationBytes(particles.pore_water_energy_j);
    bytes += allocationBytes(particles.temperature);
    bytes += allocationBytes(particles.combustible_fraction);
    bytes += allocationBytes(particles.fusion_solid_fraction);
    bytes += allocationBytes(particles.substance_tag);
    bytes += allocationBytes(particles.constitutive_model);
    bytes += allocationBytes(particles.granular_deformation_col0);
    bytes += allocationBytes(particles.granular_deformation_col1);
    bytes += allocationBytes(particles.granular_deformation_col2);
    bytes += allocationBytes(particles.granular_plastic_volume);
    bytes += allocationBytes(particles.granular_softening);
    bytes += allocationBytes(particles.granular_bond_scale);
    bytes += allocationBytes(particles.granular_hardening);
    bytes += allocationBytes(particles.granular_material_flags);
    bytes += allocationBytes(particles.granular_stress_diag);
    bytes += allocationBytes(particles.granular_stress_shear);
    bytes += allocationBytes(particles.granular_yield_value);
    bytes += allocationBytes(particles.granular_plastic_increment);
    bytes += allocationBytes(particles.granular_damage);
    bytes += allocationBytes(particles.granular_fracture_history);
    bytes += allocationBytes(particles.uvw);
    bytes += allocationBytes(particles.uvw_b);

    const auto& foam = state.foam;
    bytes += allocationBytes(foam.position);
    bytes += allocationBytes(foam.velocity);
    bytes += allocationBytes(foam.lifetime);
    bytes += allocationBytes(foam.type);
    return bytes;
}

} // namespace RayTrophiSim::SimFrameCacheMemory
