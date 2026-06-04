/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          FluidObject.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * Scene-level container for an APIC liquid instance. Owns its own particle
 * set and MAC grid (used as transient scratch by the solver). Mirrors the
 * ParticleSystemObject pattern: the simulation runtime is the source of
 * truth, the render bridges (viewport overlay, NanoVDB SDF, RT shaders) are
 * added on top in later phases.
 */

#pragma once

#include "../FluidGrid.h"
#include "FluidParticles.h"
#include "APICFluidSolver.h"
#include "FluidLevelSet.h"
#include "FluidRenderMode.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <string>
#include <cstdint>

namespace RayTrophiSim {
namespace Fluid {

// FluidRenderMode now lives in FluidRenderMode.h so SimulationGridDomain can
// reuse it without including this entire struct.

struct FluidObject {
    uint32_t id = 0;
    std::string name = "Fluid";
    bool visible = true;
    bool enabled = true;

    // Render route selection (see FluidRenderMode). Mutually exclusive: the
    // volume bridge and the particle bridge cooperate so only the active mode
    // produces RT geometry — the other tears its scene contributions down.
    FluidRenderMode render_mode = FluidRenderMode::Volume;

    // ── Particle-mode render config (mirrored as sphere instances). ────────
    Vec3  particle_render_color = Vec3(0.40f, 0.65f, 0.95f);
    float particle_render_size_multiplier = 1.0f;
    // Sphere radius expressed as a fraction of the grid voxel size. APIC seeds
    // particles_per_cell points per voxel, so ~0.45*voxel reads as a tightly
    // packed bead column without overlapping enough to look like a fog.
    float particle_render_radius_factor = 0.45f;
    int   particle_render_subdivisions = 1;  // icosphere subdivs (0..3)
    bool  particle_render_emissive = false;
    float particle_render_emission = 0.0f;

    // Transient bridge state (NOT serialized — re-derived on load).
    int      render_instance_group_id = -1;
    // Pool capacity for the instance vector. The bridge only ever grows this so
    // a settled (shrinking) particle count does not invalidate the backends'
    // scatter-instance index bindings; surplus slots collapse to scale 0.
    size_t   render_pool_capacity = 0;

    // Domain definition. The MAC grid is built lazily from these on the first
    // step. Resizing the AABB or voxel size in the UI rebuilds it.
    Vec3  domain_min   = Vec3(-1.0f, 0.0f, -1.0f);
    Vec3  domain_max   = Vec3( 1.0f, 2.0f,  1.0f);
    float voxel_size   = 0.05f;
    size_t max_grid_cells = 8000000;
    bool  grid_dirty   = true;

    // Seeding (UI-driven; applied via emit*/seed* calls and then cleared).
    Vec3 seed_min = Vec3(-0.5f, 1.0f, -0.5f);
    Vec3 seed_max = Vec3( 0.5f, 1.5f,  0.5f);
    int  seed_particles_per_cell = 4;
    size_t max_particles = 1000000;
    bool replace_on_seed = true;
    bool pending_seed = false;

    // Live state — source of truth.
    FluidSim::FluidGrid grid;
    FluidParticles      particles;
    APICSolverParams    params;
    APICSolverStats     stats;

    // ── Surface SDF (Phase 2 narrow-band level set, see FluidLevelSet.h). ────
    // Live state, rebuilt every step from the particles. Layout follows
    // `grid` (cell_count = nx*ny*nz). Not serialized — derived from the
    // particle SoA.
    LevelSetParams   level_set_params;
    LevelSetStats    level_set_stats;
    std::vector<float> sdf;
    // Width of the SDF -> density-proxy ramp in voxels. Smaller = sharper
    // surface line; values below ~0.3 start to alias against the grid.
    // Used by SceneData::buildFluidSurfaceVolume.
    float surface_band_voxels = 0.5f;
    // Index of refraction for the isosurface dielectric boundary (1.33 water).
    float surface_ior = 1.33f;
    // Surface roughness 0..1 (GGX). 0 = mirror-smooth.
    float surface_roughness = 0.0f;
    // Whitewater/foam strength 0..1 (curvature-driven).
    float surface_foam = 0.0f;

    // Rebuild grid from domain_min/domain_max/voxel_size if dirty. Preserves
    // particles; the grid is scratch so reallocating is safe between steps.
    void ensureGrid() {
        if (!grid_dirty) return;
        Vec3 lo = Vec3::min(domain_min, domain_max);
        Vec3 hi = Vec3::max(domain_min, domain_max);
        Vec3 size = hi - lo;
        const float min_voxel = 0.001f;
        voxel_size = std::max(min_voxel, voxel_size);
        int nx = std::max(1, static_cast<int>(std::round(size.x / voxel_size)));
        int ny = std::max(1, static_cast<int>(std::round(size.y / voxel_size)));
        int nz = std::max(1, static_cast<int>(std::round(size.z / voxel_size)));
        size_t cells = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
        if (cells > max_grid_cells) {
            const double scale = std::cbrt(static_cast<double>(cells) / static_cast<double>(max_grid_cells));
            voxel_size = std::max(min_voxel, static_cast<float>(static_cast<double>(voxel_size) * scale));
            nx = std::max(1, static_cast<int>(std::round(size.x / voxel_size)));
            ny = std::max(1, static_cast<int>(std::round(size.y / voxel_size)));
            nz = std::max(1, static_cast<int>(std::round(size.z / voxel_size)));
        }
        grid.resize(nx, ny, nz, voxel_size, lo);
        domain_min = lo;
        domain_max = hi;
        grid_dirty = false;
    }

    // Convenience: clear particles + reset grid.
    void resetState() {
        particles.clear();
        grid.clear();
        grid_dirty = true;
    }

    // ── VDB Sequence Cache Bridge ──────────────────────────────────────────
    bool use_vdb_cache = false;
    std::string vdb_cache_pattern = "";
    int vdb_cache_start = 0;
    int vdb_cache_end = 100;
    int vdb_cache_digits = 4;

    // Export standard VDB file representing the fluid SDF/density volume
    bool exportToVDB(const std::string& filepath) const;
};

} // namespace Fluid
} // namespace RayTrophiSim
