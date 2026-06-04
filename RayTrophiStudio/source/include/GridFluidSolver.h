/*
* =========================================================================
* Project:       RayTrophi Studio
* Repository:    https://github.com/maxkemal/RayTrophi
* File:          GridFluidSolver.h
* Author:        Kemal Demirtas
* License:       MIT
* =========================================================================
*/
#pragma once

/**
 * @file GridFluidSolver.h
 * @brief Backend-independent grid fluid/gas solver for simulation grid domains.
 *
 * Mantaflow-style operator decomposition (advect -> wall bcs -> buoyancy ->
 * forces -> vorticity -> dissipation -> pressure project), but implemented from
 * the textbook Stam "Stable Fluids" math directly on our own MAC grid
 * (FluidSim::FluidGrid). No third-party (GPL) solver code is used.
 *
 * CPU reference path (Core Principle: CPU first, GPU compute second). Fire /
 * combustion is intentionally NOT handled here yet; it will be an opt-in stage.
 */

#include "Vec3.h"
#include "FluidGrid.h"

namespace RayTrophiSim {

class SimulationForceFieldSnapshot;

namespace GridFluid {

enum class Boundary {
    Open,      // outflow at domain walls (Dirichlet p = 0)
    Closed,    // no flow through walls (Neumann, wall-normal velocity zeroed)
    Periodic   // wrap-around domain
};

enum class Advection {
    SemiLagrange,
    MacCormack
};

struct SolverParams {
    Boundary boundary = Boundary::Open;
    Advection advection = Advection::SemiLagrange;

    Vec3 gravity = Vec3(0.0f, -9.81f, 0.0f);

    // Buoyancy along the up axis (-gravity): b = buoyancy_heat * (T - ambient)
    //                                          + buoyancy_density * density.
    float buoyancy_heat = 1.0f;
    float buoyancy_density = 0.0f;
    float ambient_temperature = 0.0f;

    float vorticity = 0.0f;          // vorticity confinement strength (0 = off)

    // ── Procedural curl-noise turbulence (divergence-free FBM) ───────────────
    // Adds natural swirling detail on top of the solved velocity field. The
    // injected field is the curl of an animated FBM potential, so it is
    // divergence-free by construction and does not fight the pressure solve.
    // Modulated by local activity (density / heat / density-edge) so empty air
    // stays still. 0 strength = off (the whole stage early-outs).
    float turbulence_strength = 0.0f;
    float turbulence_scale = 1.2f;       // base spatial frequency
    int   turbulence_octaves = 3;        // FBM octaves (1-8)
    float turbulence_lacunarity = 2.0f;  // frequency multiplier per octave
    float turbulence_persistence = 0.5f; // amplitude decay per octave
    float turbulence_speed = 0.5f;       // animation evolution speed
    int   turbulence_seed = 42;

    int pressure_iterations = 40;
    float sor_omega = 1.7f;          // 1.0 = Gauss-Seidel, ~1.7 optimal for 3D

    // Per-second loss rates (factor = exp(-rate * dt); 0 = no loss).
    float density_dissipation = 0.0f;
    float temperature_dissipation = 0.0f;
    float fuel_dissipation = 0.0f;
    float velocity_dissipation = 0.0f;

    float max_velocity = 1000.0f;

    // Which channels to evolve (mirrors the domain channel mask).
    bool channel_density = true;
    bool channel_temperature = true;
    bool channel_fuel = true;
    bool channel_velocity = true;
    bool skip_velocity_advection = false;   // lets a caller run velocity advection first
    bool skip_scalar_advection = false;     // lets an external GPU path advect scalars
    bool skip_velocity_dissipation_clamp = false; // lets an external GPU path run velocity loss/limit
    bool skip_pressure_projection = false; // lets an external GPU path run projection

    // ── Combustion (opt-in) ──────────────────────────────────────────────────
    // Fuel burns only where it is hot enough (temperature >= ignition). Heat is
    // released one-way and capped; smoke (density) is generated. There is NO
    // neighbor-flame spreading: fire propagates only because advection carries
    // hot temperature into cells that still hold fuel. This makes the old
    // "whole domain ignites" runaway impossible.
    bool fire_enabled = false;
    float ignition_temperature = 0.3f;   // in the domain's 0-based heat units
    float burn_rate = 1.5f;              // fraction of available fuel / second
    float heat_release = 2.0f;           // temperature added per unit fuel burned
    float smoke_generation = 0.6f;       // density added per unit fuel burned
    float flame_dissipation = 3.0f;      // flame (interaction) field decay / second
    float max_temperature = 10.0f;       // hard cap so heat release cannot run away

    // ── Thermal expansion (gas dilation) ─────────────────────────────────────
    // Hot gas expands. The pressure projection is given a positive divergence
    // TARGET proportional to (temperature - ambient) instead of solving for the
    // usual incompressible div = 0. This pushes gas outward from hot regions,
    // which is what makes fire roll/billow and — when a fuel pocket ignites and
    // temperature spikes in one step — produces a real explosion blast wave
    // (no separate impulse needed). 0 = pure incompressible smoke (old
    // behaviour). Acts only where temperature > ambient, so it never disturbs
    // cold smoke. Units: target divergence (1/s) per heat unit above ambient.
    float expansion = 0.0f;
};

/// @brief Advance one fluid step in place on @p grid.
/// @param forces optional shared force-field snapshot (evaluated as Gas kind).
/// @param time_seconds simulation time, for force-field noise animation.
void step(FluidSim::FluidGrid& grid,
          const SolverParams& params,
          float dt,
          const SimulationForceFieldSnapshot* forces = nullptr,
          float time_seconds = 0.0f);

/// @brief Advance one fluid step using Sparse OpenVDB in place on @p grid.
void stepSparseVDB(FluidSim::FluidGrid& grid,
                   const SolverParams& params,
                   float dt,
                   const SimulationForceFieldSnapshot* forces = nullptr,
                   float time_seconds = 0.0f);

/// @brief Run only the pressure projection stage in place on @p grid.
void projectPressure(FluidSim::FluidGrid& grid,
                     const SolverParams& params,
                     float dt);

/// @brief Run only the velocity advection stage in place on @p grid.
void advectVelocityField(FluidSim::FluidGrid& grid,
                         const SolverParams& params,
                         float dt);

} // namespace GridFluid
} // namespace RayTrophiSim
