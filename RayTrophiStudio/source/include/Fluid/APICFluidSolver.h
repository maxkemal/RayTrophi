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

enum class FluidChemistryPreset : int {
    Inert = 0,
    Water,
    Gasoline,
    Alcohol,
    Oil,
    Custom,
    Plastic,
    Wax
};

// Material-level phase/combustion description.  The APIC solver does not
// require these fields for ordinary liquids; keeping them here makes water,
// oil, honey and future fuels self-describing without forcing every domain to
// allocate gas channels.
struct FluidFuelProfile {
    bool flammable = false;
    bool extinguishing = false;
    float flash_temperature = 0.0f;
    float autoignition_temperature = 0.0f;
    float vaporization_rate = 0.0f;
    float heat_capacity = 1.0f;
    float latent_heat = 0.0f;
    float cooling_power = 0.0f;
    float oxygen_dilution = 0.0f;
    float flame_persistence = 0.0f;
};

struct APICSolverParams {
    enum class FluidPreset : int;
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
    // ── Viscosity ──────────────────────────────────────────────────────────
    // Kinematic viscosity ν in m²/s — a PHYSICAL quantity, not an artistic
    // 0..200 dial.
    //   water 1e-6 · olive oil 8e-5 · molten chocolate ~4e-3 · honey ~7e-3 ·
    //   lava 0.1 … 100
    // RENAMED, not repurposed. The old `viscosity` field drove a raw neighbour-
    // average lerp with NO 1/h² factor, so its meaning changed with voxel size,
    // and its strength saturated at clamp(v*dt, 0, 0.45) — 20 and 30 produced
    // nearly the same motion. A project written before this change must NOT
    // silently load 20.0 as 20 m²/s (that is lava, not honey), so the field name
    // changed with it and old files fall back to the preset default.
    float kinematic_viscosity = 0.0f;
    // Red-black Gauss-Seidel sweeps for the implicit diffusion solve. One sweep
    // is 2 dispatches covering all three MAC components. Under-converging does
    // not blow up — implicit diffusion is unconditionally stable — it merely
    // UNDER-applies the viscosity, so raise this if a thick preset still flows
    // too freely (especially at high resolution, where ν·dt/h² grows).
    int   viscosity_sweeps = 8;
    // Tangential wall condition for the viscous solve.
    //   0 = no-slip   — the fluid sticks to colliders and is dragged by moving
    //                   ones. Required for honey/chocolate to coat a surface.
    //   1 = free-slip — matches the pressure projection's own condition, which
    //                   is what water wants.
    // Values between blend the two. The pressure projection is unaffected;
    // this only enters the viscous stencil.
    float viscosity_wall_slip = 1.0f;
    float affine_damping = 0.98f;
    float max_affine = 80.0f;

    FluidFuelProfile fuel_profile;
    FluidChemistryPreset chemistry_preset = FluidChemistryPreset::Inert;

    void applyChemistryProfile(FluidChemistryPreset preset) {
        chemistry_preset = preset;
        fuel_profile = {};
        switch (preset) {
            case FluidChemistryPreset::Gasoline:
                fuel_profile.flammable = true;
                fuel_profile.flash_temperature = 0.35f;
                fuel_profile.autoignition_temperature = 0.80f;
                fuel_profile.vaporization_rate = 0.85f;
                fuel_profile.heat_capacity = 1.7f;
                fuel_profile.latent_heat = 0.9f;
                fuel_profile.flame_persistence = 0.55f;
                break;
            case FluidChemistryPreset::Alcohol:
                fuel_profile.flammable = true;
                fuel_profile.flash_temperature = 0.28f;
                fuel_profile.autoignition_temperature = 0.72f;
                fuel_profile.vaporization_rate = 1.20f;
                fuel_profile.heat_capacity = 2.4f;
                fuel_profile.latent_heat = 0.85f;
                fuel_profile.flame_persistence = 0.35f;
                break;
            case FluidChemistryPreset::Oil:
                fuel_profile.flammable = true;
                fuel_profile.flash_temperature = 0.65f;
                fuel_profile.autoignition_temperature = 1.10f;
                fuel_profile.vaporization_rate = 0.28f;
                fuel_profile.heat_capacity = 1.7f;
                fuel_profile.latent_heat = 1.2f;
                fuel_profile.flame_persistence = 0.90f;
                break;
            case FluidChemistryPreset::Plastic:
                fuel_profile.flammable = true;
                fuel_profile.flash_temperature = 0.92f;
                fuel_profile.autoignition_temperature = 1.10f;
                fuel_profile.vaporization_rate = 0.18f;
                fuel_profile.heat_capacity = 1.9f;
                fuel_profile.latent_heat = 1.35f;
                fuel_profile.flame_persistence = 1.10f;
                break;
            case FluidChemistryPreset::Wax:
                fuel_profile.flammable = true;
                fuel_profile.flash_temperature = 0.66f;
                fuel_profile.autoignition_temperature = 0.88f;
                fuel_profile.vaporization_rate = 0.12f;
                fuel_profile.heat_capacity = 2.1f;
                fuel_profile.latent_heat = 1.45f;
                fuel_profile.flame_persistence = 0.95f;
                break;
            case FluidChemistryPreset::Water:
                fuel_profile.extinguishing = true;
                fuel_profile.heat_capacity = 4.18f;
                fuel_profile.latent_heat = 2.26f;
                fuel_profile.cooling_power = 1.0f;
                fuel_profile.oxygen_dilution = 0.35f;
                break;
            case FluidChemistryPreset::Custom:
            default:
                break;
        }
    }

    // Backward-compatible bridge: old Oil fluid presets retain oil chemistry.
    void applyFuelProfile(FluidPreset preset) {
        applyChemistryProfile(preset == FluidPreset::Oil
            ? FluidChemistryPreset::Oil
            : FluidChemistryPreset::Inert);
    }

    // CPU reference path controls. Thread count 0 means automatic.
    int   cpu_threads = 0;
    int   parallel_particle_threshold = 32768;
    bool  external_forces_preintegrated = false;
    bool  p2g_precomputed = false;

    // (`skip_g2p` removed: no caller ever set it. The GPU split-step goes
    // through stop_before_pressure + pressure_g2p_precomputed, and the FLIP
    // snapshot it needed is now published unconditionally by step().)

    // GPU split-step second call: boundary+viscosity+pressure already done
    // externally; skip to air_drag → velocity_damping → advect → reseed only.
    bool  pressure_g2p_precomputed = false;
    // The caller runs the viscous diffusion itself (GPU). step() then does
    // P2G → FLIP snapshot → solid boundaries and stops, leaving grid.vel as the
    // POST-boundary, PRE-viscosity field for the device solve to consume.
    bool  viscosity_precomputed = false;
    // Vulkan tail dispatch already applied spray drag, damping, particle
    // advection and domain/collider boundaries. CPU retains only the
    // topology-changing reseed/reference fallback.
    bool  particle_tail_precomputed = false;

    // GPU split-step first call: run P2G → FLIP snapshot → solid boundaries →
    // (viscosity, unless viscosity_precomputed) and return, so the caller can
    // drive the device viscosity + pressure + G2P before calling back with
    // pressure_g2p_precomputed for the tail.
    // Was `stop_after_viscosity`: the stop point no longer implies viscosity ran
    // here, so the name had to move with the meaning.
    bool  stop_before_pressure = false;

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
    // ★ These presets used to differ almost entirely in DISSIPATION —
    // internal_friction, velocity_damping, air_drag, max_velocity — because the
    // viscosity dial barely worked (no 1/h², saturating clamp, and the FLIP
    // snapshot was taken AFTER it, which cancelled it out of the FLIP delta
    // entirely). Dissipation is not viscosity: it drags the whole body, so the
    // fluid reached a terminal fall speed of g/rate instead of accelerating.
    // Honey fell at ~1.8 m/s and Sand at ~0.9 m/s no matter the drop height, and
    // sand fell SLOWER than honey — the tell that the knob was wrong.
    // Real honey in free fall accelerates at g; it resists SHEAR, not motion.
    // So the thick presets now carry a physical ν and only a light residual
    // damping, and none of them clamp max_velocity below the safety value.
    enum class FluidPreset : int {
        Custom = 0,
        Water,     // ν≈1e-6, effectively inviscid at any renderable voxel size
        Oil,       // mildly viscous, wets walls a little
        Mud,       // heavy slurry (a real yield stress needs Drucker-Prager)
        Honey,     // very viscous, sticky threads, no-slip walls
        Lava,      // extreme viscosity, very slow (renderer adds the glow)
        Sand,      // granular APPROXIMATION — see the case body
        Chocolate  // molten couverture: between oil and honey, fully no-slip
    };
    FluidPreset current_preset = FluidPreset::Water;

    void applyPreset(FluidPreset preset) {
        switch (preset) {
            case FluidPreset::Water:
                // ν=0 skips the viscous solve outright: water's 1e-6 m²/s is
                // orders of magnitude below the numerical diffusion of any grid
                // coarse enough to render, so paying for the sweeps buys motion
                // you cannot see.
                kinematic_viscosity = 0.0f;   viscosity_sweeps = 1;
                viscosity_wall_slip = 1.0f;
                internal_friction = 0.5f;
                flip_blend = 0.97f; apic_blend = 0.95f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 0.5f;   wall_damping = 0.15f;
                affine_damping = 0.98f; max_velocity = 50.0f;
                break;
            case FluidPreset::Oil:
                kinematic_viscosity = 1.0e-4f; viscosity_sweeps = 6;
                viscosity_wall_slip = 0.6f;
                internal_friction = 0.15f;
                flip_blend = 0.95f; apic_blend = 0.95f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 0.6f;   wall_damping = 0.20f;
                affine_damping = 0.97f; max_velocity = 50.0f;
                break;
            case FluidPreset::Mud:
                // Mud is really a yield-stress (Bingham) material: it should
                // STOP, not creep slowly to a stop. Without a plastic return
                // mapping the solver cannot express that, so a little residual
                // internal_friction stands in for the yield surface. Marked so
                // the approximation is not mistaken for the physics.
                kinematic_viscosity = 2.0e-3f; viscosity_sweeps = 12;
                viscosity_wall_slip = 0.2f;
                internal_friction = 0.35f;
                flip_blend = 0.90f; apic_blend = 0.90f;
                velocity_damping = 0.999f;
                density_correction = 1.2f;
                air_drag = 0.8f;   wall_damping = 0.30f;
                affine_damping = 0.95f; max_velocity = 50.0f;
                break;
            case FluidPreset::Honey:
                kinematic_viscosity = 7.0e-3f; viscosity_sweeps = 16;
                viscosity_wall_slip = 0.0f;
                internal_friction = 0.2f;
                flip_blend = 0.90f; apic_blend = 0.90f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 0.8f;   wall_damping = 0.35f;
                affine_damping = 0.95f; max_velocity = 50.0f;
                break;
            case FluidPreset::Chocolate:
                // Molten couverture at ~40 °C: roughly half honey's kinematic
                // viscosity, and it wets everything it touches (no-slip). The
                // ribbon/coiling it is known for comes from the viscous solve;
                // the mound it holds after landing needs the yield stress this
                // solver does not have yet, so a thick pour reads better than a
                // slow drip until then.
                kinematic_viscosity = 4.0e-3f; viscosity_sweeps = 16;
                viscosity_wall_slip = 0.0f;
                internal_friction = 0.2f;
                flip_blend = 0.90f; apic_blend = 0.92f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 0.7f;   wall_damping = 0.40f;
                affine_damping = 0.95f; max_velocity = 50.0f;
                break;
            case FluidPreset::Lava:
                kinematic_viscosity = 0.5f;    viscosity_sweeps = 24;
                viscosity_wall_slip = 0.0f;
                internal_friction = 0.2f;
                flip_blend = 0.85f; apic_blend = 0.85f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 1.0f;   wall_damping = 0.50f;
                affine_damping = 0.92f; max_velocity = 50.0f;
                break;
            case FluidPreset::Sand:
                // ★ STILL AN APPROXIMATION, and knowingly so. Grain friction is
                // a SHEAR YIELD criterion (Drucker-Prager), not a velocity
                // Laplacian — no value of ν produces an angle of repose, so the
                // viscous solve is left off and per-particle friction + strong
                // density correction stand in. Consequence to expect: a sand
                // pile still flattens further than it should, and sand does not
                // fall at g. Fixing it properly needs per-particle plastic
                // state, which FluidParticles does not carry yet.
                kinematic_viscosity = 0.0f;   viscosity_sweeps = 1;
                viscosity_wall_slip = 0.0f;
                internal_friction = 4.0f;
                flip_blend = 0.60f; apic_blend = 0.40f;
                velocity_damping = 0.99f;
                density_correction = 2.0f;
                air_drag = 1.0f;   wall_damping = 0.90f;
                affine_damping = 0.85f; max_velocity = 50.0f;
                break;
            case FluidPreset::Custom:
            default:
                return; // leave fields untouched
        }
        current_preset = preset;
        applyFuelProfile(preset);
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
    size_t recovered_solid_particles = 0;
    // Sweeps the viscous solve actually ran this step (0 = ν was 0, i.e. the
    // solve was skipped). Reported so "Viscosity 0.00 ms" can be told apart
    // from "viscosity is switched off" — the same reading for two very
    // different states is how a dead knob stays invisible.
    int   viscosity_sweeps_run = 0;
    bool  viscosity_on_gpu = false;
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

/// @brief Integrate gravity + external force fields onto the particle velocities
///        (APIC step 1). Factored out of step() so the GPU pipeline can run this
///        cheap, CPU-only, correctness-critical pass (force-field/noise/wind
///        surface-drag eval is not ported to the device) and then upload the
///        post-force velocities for a GPU P2G — instead of falling the whole
///        solve back to the CPU whenever a force field is active. Reads `grid`
///        only for the wind free-surface band; writes only particle velocities.
void applyExternalForces(FluidParticles& particles,
                         const FluidSim::FluidGrid& grid,
                         const APICSolverParams& params,
                         const SimulationForceFieldSnapshot* forces,
                         float time_seconds,
                         float dt);

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
