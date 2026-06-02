/*
 * =========================================================================
 * Project:       RayTrophi Studio
 * File:          APICFluidSolver.h
 * Author:        Kemal Demirtas
 * License:       MIT
 * =========================================================================
 *
 * APIC liquid solver (CPU reference path).
 *
 * Pipeline per step:
 *   1. integrate external forces on particle velocities (gravity, fields)
 *   2. P2G: scatter particle velocity + affine to MAC grid (FluidGrid)
 *   3. boundary / solid enforcement on grid velocity
 *   4. pressure projection — delegated to GridFluid::projectPressure
 *      (incompressibility, same Poisson solver as gas)
 *   5. G2P: gather grid velocity + reconstruct affine C back to particles
 *   6. advect particle positions (forward Euler with substeps for CFL)
 *
 * The narrow-band level set (Phase 2) is rebuilt OUTSIDE this step from the
 * final particle positions — it is a render artifact, not part of the solve.
 *
 * GPU port plan: every loop here is grid-parallel except P2G scatter, which
 * will need atomic adds or a sort-bin pass on Vulkan compute. Faz 4.
 */

#pragma once

#include "../Vec3.h"
#include "../FluidGrid.h"
#include "FluidParticles.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace RayTrophiSim {

class SimulationForceFieldSnapshot;

namespace Fluid {

struct APICSolverParams {
    Vec3  gravity = Vec3(0.0f, -9.81f, 0.0f);

    // Particles per cell when seeding a fluid region. APIC is stable down to
    // 4-8; 8 is the standard target for visual fidelity.
    int   particles_per_cell = 8;

    // CFL safety factor for the position-advection substep count.
    float cfl = 0.5f;
    int   max_substeps = 12;

    // Pressure projection. With the PCG+MIC(0) solver this is now a *cap* on
    // iterations rather than a fixed count — early-exit kicks in once the
    // residual drops below tolerance. 24 is plenty for typical liquid frames;
    // bump for high-resolution / heavy-collision setups.
    int   pressure_iterations = 24;
    // Relative pressure residual target. 1e-5 preserves the current solver
    // behavior; relaxing toward 1e-4 often reduces GPU MGPCG sync cost on
    // dense preview scenes.
    float pressure_relative_residual = 1.0e-5f;
    // Experimental CUDA MGPCG Layer B. Uses a geometric V-cycle as the CG
    // preconditioner; keep optional because it trades extra dispatch work for
    // lower iteration count on large grids.
    bool  pressure_multigrid_preconditioner = false;
    // Legacy SOR relaxation factor. Retained only so older project files load
    // without losing the field; the PCG path ignores it. Hidden in UI.
    float sor_omega = 1.25f;

    // APIC affine blend. 1.0 = full APIC angular/detail preservation,
    // 0.0 = PIC-only smoothing. This does not blend old particle velocity.
    // Default 0.95: low values inject numerical viscosity (macunsu look).
    float apic_blend = 0.95f;

    // FLIP/PIC translational blend. 0 = pure PIC (smooth, dissipative — what
    // the old code did implicitly), 1 = pure FLIP (carries old particle
    // velocity + pressure impulse — splashy / energetic / noisy). The MAC-grid
    // velocity is snapshotted before pressure projection; G2P gathers both the
    // pre- and post-projection grid velocity and reconstructs the FLIP delta.
    // 0.97 is the Houdini / Bridson canonical value for water.
    float flip_blend = 0.97f;

    // Clamp particle velocity. 50 m/s lets ~12 m free-fall reach terminal
    // before the safety kicks in; older 8 m/s default capped water at ~3 m
    // drops, giving the "syrupy" look.
    float max_velocity = 50.0f;
    // Per-step multiplicative damping. 0.999 = ~0.1% energy loss per step
    // (matches numerical-only loss). The old 0.985 default ate ~1.5%/step,
    // ~30% per second at 60Hz — a major source of the "tired" liquid feel.
    float velocity_damping = 0.999f;
    float wall_damping = 0.15f;

    // Density-targeted pressure projection (Bridson 2007). Pure
    // divergence-from-velocity pressure does not see static over-packing:
    // 30 particles in one cell with zero velocity have div=0, so the
    // solver feels no need to expel them. This term injects a synthetic
    // divergence in over-populated cells proportional to the overshoot,
    // making the PCG raise pressure there and push particles outward.
    // Without it, FLIP piles collapse on top of each other ("2D pile").
    // 0 = off (raw NS divergence only), 1.0 = sensible default for water.
    float density_correction = 1.0f;

    // Per-particle viscous decay rate (1/s). Exponential energy loss:
    // v *= exp(-internal_friction * dt). Models inviscid → viscous → highly
    // damped behaviour without depending on velocity variance (so coherent
    // bulk flows ALSO settle, not only splashy/spread-out ones).
    // 0 = no decay, 0.5 ≈ water-like settle over seconds, 10+ ≈ near-instant.
    float internal_friction = 0.5f;

    // Quadratic air drag on particles whose containing cell has fewer than
    // `reseed_min_per_cell` particles — i.e. spray, droplets, splash debris.
    // Applied implicitly as v *= 1 / (1 + k|v|dt), unconditionally stable.
    // 0 disables. Bulk fluid particles are not affected; their dissipation
    // comes from internal_friction.
    float air_drag = 0.5f;

    // Particle redistribution (reseed). Each step, fluid cells whose particle
    // count drifts away from the seed target are corrected: starved cells get
    // new particles with grid-sampled velocity, over-populated cells lose
    // their surplus (highest-index removal — order-agnostic). Without this,
    // cell-per-particle drift causes density bands and apparent volume change.
    bool  reseed_enabled = true;
    // Target particles per fluid cell. 0 = use particles_per_cell (the seed
    // density). Reseed kicks in when count falls below min_per_cell or rises
    // above max_per_cell; otherwise the cell is untouched.
    int   reseed_target_per_cell = 0;
    int   reseed_min_per_cell = 3;
    int   reseed_max_per_cell = 16;
    std::size_t max_particles = 1000000;
    // How strongly pure domain translation drives the liquid mass. 0 = moving
    // bounds only, 1 = the container fully carries the nearby fluid velocity.
    float domain_motion_coupling = 1.0f;
    // Artistic kinematic viscosity for the CPU reference path. 0 = water-like,
    // higher values smooth the MAC velocity field before projection for syrupy
    // / honey-like motion.
    float viscosity = 0.0f;
    int   viscosity_iterations = 1;
    float affine_damping = 0.98f;
    float max_affine = 80.0f;

    // CPU reference path controls. Thread count 0 means automatic.
    int   cpu_threads = 0;
    int   parallel_particle_threshold = 32768;
    bool  external_forces_preintegrated = false;
    bool  p2g_precomputed = false;

    // CPU pressure runs normally; only G2P is skipped so the caller can run it
    // on GPU. The FLIP pre-snapshot (vel_x/y/z before pressure) is left in the
    // internal static buffers — access via Fluid::getLastFlipPreSnapshot*().
    // After GPU G2P downloads particle velocities, call Fluid::step again with
    // pressure_g2p_precomputed=true so only air_drag/damping/advect/reseed run.
    bool  skip_g2p = false;

    // GPU split-step second call: boundary+viscosity+pressure already done
    // externally; skip to air_drag → velocity_damping → advect → reseed only.
    bool  pressure_g2p_precomputed = false;

    // (Legacy GPU split-step — kept for compatibility, no longer used.)
    bool  stop_after_viscosity = false;

    // Use the free-surface pressure projection (treat empty cells as p=0)
    // instead of the gas-style fill-everywhere Poisson. Required for actual
    // liquid behavior — without it the fluid acts like a sealed container.
    bool  free_surface = true;

    // Domain-wall behaviour for the PARTICLES (mirrors the domain's boundary
    // mode). Without this the advection step always clamps + bounces particles
    // back inside, so a domain reads as a sealed box even when the UI says
    // "Open (Outflow)". Closed = clamp + bounce; Open = particles crossing a
    // wall flow out (deleted); Periodic = wrap to the opposite wall.
    enum class BoundaryMode : int { Closed = 0, Open = 1, Periodic = 2 };
    BoundaryMode boundary = BoundaryMode::Closed;

    // Variational solid coupling (Batty/Bridson 2007). When true, the pressure
    // projection uses the fractional MAC-face open weights (FluidGrid::u/v/w_weight,
    // filled by the collider voxelizer's analytic super-sampling) instead of a
    // binary solid-cell test. Gives sub-grid-accurate collisions (no blocky
    // leaking) AND lets a MOVING collider's face velocity enter the divergence
    // RHS, so the solid actually pushes/splashes the fluid through the pressure
    // solve rather than only via particle ejection. Falls back to the binary
    // path when false or when the weight arrays aren't present. CPU path.
    bool  variational_solids = true;

    // Ghost-fluid free surface (Gibou/Enright). The default free-surface
    // projection puts the p=0 Dirichlet boundary at the AIR cell centre (1st
    // order → the surface snaps to cell centres, visible voxel staircase). With
    // this on, a cheap per-step particle-ball level set places the zero-pressure
    // boundary at the actual sub-cell surface position: the fluid-air face's
    // diagonal coefficient is scaled by 1/theta (theta = fluid fraction along the
    // face from the level set) and the velocity update uses the matching ghost
    // pressure. Smooth, second-order surface; theta is clamped for stability.
    // Falls back to the p=0 boundary when false / no level set. CPU path.
    bool  ghost_fluid_surface = true;
    float surface_ball_radius = 0.9f;  // particle level-set ball radius (× voxel)

    // ── Material presets ───────────────────────────────────────────────────
    // Physically-motivated rheology presets for the common materials artists
    // reach for. Mirrors WaterWaveParams::WaterPreset: `current_preset` is a
    // UI/serialization convenience the solver never reads, set to Custom the
    // moment a rheology field is hand-edited. applyPreset() overwrites ONLY the
    // rheology fields (viscosity / friction / blend / damping / packing); the
    // domain, gravity, reseed, performance and free-surface settings are left
    // alone so a preset can be dropped onto an already-configured fluid.
    enum class FluidPreset : int {
        Custom = 0,
        Water,   // low-viscosity Newtonian, splashy (Houdini/Bridson defaults)
        Oil,     // mildly viscous, less splashy, slightly wetting walls
        Mud,     // heavy, dissipative slurry between oil and honey
        Honey,   // very viscous, slow, sticky threads
        Lava,    // extreme viscosity, very slow (renderer adds the glow)
        Sand     // granular approximation: high friction + strong packing
    };
    FluidPreset current_preset = FluidPreset::Water;

    void applyPreset(FluidPreset preset) {
        switch (preset) {
            case FluidPreset::Water:
                viscosity = 0.0f;  viscosity_iterations = 1;
                internal_friction = 0.5f;
                flip_blend = 0.97f; apic_blend = 0.95f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 0.5f;   wall_damping = 0.15f;
                affine_damping = 0.98f; max_velocity = 50.0f;
                break;
            case FluidPreset::Oil:
                viscosity = 3.0f;  viscosity_iterations = 2;
                internal_friction = 1.5f;
                flip_blend = 0.70f; apic_blend = 0.88f;
                velocity_damping = 0.997f;
                density_correction = 1.0f;
                air_drag = 0.8f;   wall_damping = 0.35f;
                affine_damping = 0.95f; max_velocity = 40.0f;
                break;
            case FluidPreset::Mud:
                viscosity = 10.0f; viscosity_iterations = 3;
                internal_friction = 3.5f;
                flip_blend = 0.40f; apic_blend = 0.70f;
                velocity_damping = 0.99f;
                density_correction = 1.2f;
                air_drag = 1.2f;   wall_damping = 0.60f;
                affine_damping = 0.90f; max_velocity = 25.0f;
                break;
            case FluidPreset::Honey:
                viscosity = 20.0f; viscosity_iterations = 4;
                internal_friction = 5.0f;
                flip_blend = 0.25f; apic_blend = 0.60f;
                velocity_damping = 0.99f;
                density_correction = 1.0f;
                air_drag = 1.5f;   wall_damping = 0.70f;
                affine_damping = 0.90f; max_velocity = 20.0f;
                break;
            case FluidPreset::Lava:
                viscosity = 30.0f; viscosity_iterations = 6;
                internal_friction = 6.0f;
                flip_blend = 0.15f; apic_blend = 0.50f;
                velocity_damping = 0.985f;
                density_correction = 1.0f;
                air_drag = 2.0f;   wall_damping = 0.80f;
                affine_damping = 0.85f; max_velocity = 12.0f;
                break;
            case FluidPreset::Sand:
                // Granular approximation in a liquid solver: no kinematic
                // viscosity (grain friction is not velocity-Laplacian), heavy
                // per-particle friction + damping so it stops quickly, low
                // APIC angular preservation, and strong density correction so
                // the pile resists collapse instead of flattening.
                viscosity = 0.0f;  viscosity_iterations = 1;
                internal_friction = 8.0f;
                flip_blend = 0.50f; apic_blend = 0.40f;
                velocity_damping = 0.96f;
                density_correction = 2.0f;
                air_drag = 1.0f;   wall_damping = 0.90f;
                affine_damping = 0.85f; max_velocity = 30.0f;
                break;
            case FluidPreset::Custom:
            default:
                return; // leave fields untouched
        }
        current_preset = preset;
    }
};

struct APICSolverStats {
    float total_ms = 0.0f;
    float forces_ms = 0.0f;
    float p2g_ms = 0.0f;
    float boundary_ms = 0.0f;
    float viscosity_ms = 0.0f;
    float pressure_ms = 0.0f;
    float g2p_ms = 0.0f;
    float advect_ms = 0.0f;
    float density_ms = 0.0f;
    float pressure_cg_dot_ms = 0.0f;
    double pressure_cg_final_relative_residual = 0.0;
    int   cpu_threads = 1;
    int   advect_substeps = 1;
    int   pressure_cg_iterations = 0;
    int   pressure_cg_max_iterations = 0;
    int   pressure_cg_dot_count = 0;
    bool  pressure_cg_multigrid = false;
    size_t particle_count = 0;
    size_t grid_cell_count = 0;
    size_t active_fluid_cells = 0;
    bool  density_on_gpu = false;
    bool  p2g_on_gpu = false;
    bool  g2p_on_gpu = false;
    bool  pressure_on_gpu = false;
    bool  forces_on_gpu = false;
    bool  gpu_requested = false;
    bool  gpu_compute_available = false;
    bool  gpu_fallback = false;
    std::string compute_device = "CPU";
    std::string gpu_status = "CPU reference path";
};

// Access the FLIP pre-pressure-projection velocity snapshot left in the
// internal static buffers by the most recent Fluid::step call that ran
// pressure. Valid until the next Fluid::step call.
bool          hasLastFlipPreSnapshot();
std::size_t   getLastFlipPreSnapshotSize(); // number of floats (vel_x faces)
const float*  getLastFlipPreSnapshotX();
const float*  getLastFlipPreSnapshotY();
const float*  getLastFlipPreSnapshotZ();

/// @brief Advance one liquid step. `grid` is used as scratch velocity field;
///        its density/temperature/fuel channels are not touched.
void step(FluidParticles& particles,
          FluidSim::FluidGrid& grid,
          const APICSolverParams& params,
          float dt,
          const SimulationForceFieldSnapshot* forces = nullptr,
          float time_seconds = 0.0f,
          APICSolverStats* stats = nullptr);

/// @brief Seed particles uniformly inside an AABB (jittered). Existing
///        particles are kept; positions are appended.
size_t estimateSeedBoxParticleCount(const Vec3& grid_origin,
                                    int nx,
                                    int ny,
                                    int nz,
                                    float voxel_size,
                                    const Vec3& min_world,
                                    const Vec3& max_world,
                                    int particles_per_cell);

size_t estimateSeedBoxParticleCount(const FluidSim::FluidGrid& grid,
                                    const Vec3& min_world,
                                    const Vec3& max_world,
                                    int particles_per_cell);

void seedBox(FluidParticles& particles,
             const FluidSim::FluidGrid& grid,
             const Vec3& min_world,
             const Vec3& max_world,
             int particles_per_cell,
             uint32_t seed = 0u,
             size_t max_new_particles = static_cast<size_t>(-1));

} // namespace Fluid
} // namespace RayTrophiSim
