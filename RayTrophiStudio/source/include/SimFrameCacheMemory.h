#pragma once

#include "ParticleSimulation.h"

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
    bytes += allocationBytes(grid.active_tiles);
    bytes += allocationBytes(grid.tile_active_mask);
    bytes += allocationBytes(grid.vel_x);
    bytes += allocationBytes(grid.vel_y);
    bytes += allocationBytes(grid.vel_z);
    bytes += allocationBytes(grid.density);
    bytes += allocationBytes(grid.temperature);
    bytes += allocationBytes(grid.fuel);
    bytes += allocationBytes(grid.interaction);
    bytes += allocationBytes(grid.pressure);
    bytes += allocationBytes(grid.divergence);
    bytes += allocationBytes(grid.solid);
    bytes += allocationBytes(grid.solid_vel);
    bytes += allocationBytes(grid.solid_cells);
    bytes += allocationBytes(grid.substance_solid_cells);
    bytes += allocationBytes(grid.substance_solid_prev_cells);
    bytes += allocationBytes(grid.substance_solid_prev_vel);
    bytes += allocationBytes(grid.solid_gas_density);
    bytes += allocationBytes(grid.solid_gas_temperature);
    bytes += allocationBytes(grid.solid_gas_fuel);
    bytes += allocationBytes(grid.solid_gas_flame);
    bytes += allocationBytes(grid.solid_gas_band);
    bytes += allocationBytes(grid.u_weight);
    bytes += allocationBytes(grid.v_weight);
    bytes += allocationBytes(grid.w_weight);
    bytes += allocationBytes(grid.fluid_phi);
    bytes += allocationBytes(grid.surface_dust_supply);

    const auto& particles = state.particles;
    bytes += allocationBytes(particles.position);
    bytes += allocationBytes(particles.velocity);
    bytes += allocationBytes(particles.affine);
    bytes += allocationBytes(particles.flags);
    bytes += allocationBytes(particles.mass_fraction);
    bytes += allocationBytes(particles.temperature);
    bytes += allocationBytes(particles.combustible_fraction);
    bytes += allocationBytes(particles.substance_tag);
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
