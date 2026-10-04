#pragma once

namespace RayTrophiSim::Fluid {

struct FluidParticleVisualRadius {
    float authored_voxels = 0.0f;
    float material_point_voxels = 0.0f;
    float effective_voxels = 0.0f;
    bool material_point_floor_applied = false;
};

// Each seeded material point represents 1 / particles_per_cell of a voxel's
// material volume. Procedural spheres smaller than that equivalent volume make
// both liquid and granular particle views change apparent density when the
// constitutive model changes. Preserve larger authored spheres, but apply the
// same represented-volume floor to every procedural fluid sphere.
FluidParticleVisualRadius resolveFluidParticleVisualRadius(
    bool granular_enabled,
    bool procedural_spheres,
    int particles_per_cell,
    float radius_factor,
    float size_multiplier);

} // namespace RayTrophiSim::Fluid
