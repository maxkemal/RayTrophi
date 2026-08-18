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
#include <algorithm>
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
    // v *= exp(-internal_friction * dt), applied to EVERY particle in
    // gridToParticle — bulk and spray alike, in whatever direction it is
    // moving.
    //
    // ★★★ THIS IS NOT VISCOSITY AND THE DEFAULT USED TO PRETEND IT WAS. It
    // brakes the whole body instead of resisting shear, so it also brakes free
    // FALL. It carried 0.5 from the era before `kinematic_viscosity` was
    // physical, when every thick preset had to fake thickness with it. The
    // thick presets were converted to a real ν; this default was not, so water
    // kept paying for thickness it does not have — a 2 m pour lost roughly a
    // third of its speed on the way down and landed without a splash, which
    // reads exactly as "even water is highly viscous now".
    //
    // ★ 0 is the honest default. Real water resists SHEAR, and shear is the
    // viscous solve's job. Anything non-zero here is a body drag no liquid
    // actually has; keep it for stylised or deliberately dead liquids.
    float internal_friction = 0.0f;

    // Drucker-Prager granular MPM parameters. They are inert unless the Sand
    // preset is active; liquid presets keep the existing Navier-Stokes path.
    bool  granular_enabled = false;
    float granular_friction_angle_degrees = 35.0f;
    float granular_cohesion = 0.0f;
    float granular_dilatancy_degrees = 5.0f;
    float granular_young_modulus = 2.0e5f;
    float granular_poisson_ratio = 0.25f;
    float granular_tensile_cutoff = 0.0f;
    float granular_hardening = 0.0f;
    float granular_fracture_strain = 0.04f;
    float granular_damage_rate = 6.0f;
    float granular_healing_rate = 0.0f;
    bool  granular_rebonding = false;
    int   granular_max_solver_substeps = 32;

    // ── Thermal / burn softening ─────────────────────────────────────────────
    // Melting a granular body is not a special-cased animation: it is the
    // skeleton losing the strength that held it up. These drive a PER-PARTICLE
    // multiplier (FluidParticles::granular_softening) from the particle's own
    // temperature and remaining mass, so a corner of a foam block can slump
    // while the rest stays cold — the entire point of doing it per particle.
    //
    // ★★★ THE AUTHORED VALUES ABOVE ARE NEVER MODIFIED BY THE RUNTIME. They are
    // hashed into the fluid coupling signature; writing a softened modulus back
    // into granular_young_modulus would re-key the bake every frame and the
    // rewind would restart from frame 0 forever. Softening lives beside them,
    // not on top of them.
    //
    // 0 K disables the whole path, which is the default: sand does not soften.
    float granular_softening_temperature = 0.0f;  // K, midpoint of the transition
    float granular_softening_range = 40.0f;       // K, width of the transition
    // Floor the skeleton keeps once fully softened. 0 = it can lose all
    // strength (a melt); a small value keeps a molten-but-not-liquid residue.
    float granular_residual_strength = 0.05f;
    // ★★★ Cohesion multiplier at peak tackiness. 1.0 = no hump (bonds simply
    // fade with stiffness — correct for a CHARRING material, wrong for a
    // thermoplastic). A melting plastic must get STICKIER before it flows, or
    // the block crumbles into separate grains instead of slumping as one body.
    // Both look like "lost its shape"; only one looks like melting.
    float granular_tack_peak = 1.0f;
    // Heat conduction rate, 1/s. Drives BOTH particle<->gas contact heating and
    // particle<->particle conduction. 0 = no conduction, which means a burning
    // body heats only at the surface and its interior never softens.
    float granular_thermal_conductivity = 0.0f;

    // Canonical validation used by UI, scripting/IPC and scene loading. Keep
    // every authoring surface on the same material contract.
    void sanitizeGranularMaterial() {
        granular_friction_angle_degrees = std::clamp(granular_friction_angle_degrees, 0.0f, 55.0f);
        granular_cohesion = std::max(granular_cohesion, 0.0f);
        granular_dilatancy_degrees = std::clamp(granular_dilatancy_degrees, 0.0f, 30.0f);
        granular_young_modulus = std::clamp(granular_young_modulus, 10.0f, 1.0e7f);
        granular_poisson_ratio = std::clamp(granular_poisson_ratio, 0.0f, 0.49f);
        granular_tensile_cutoff = std::max(granular_tensile_cutoff, 0.0f);
        granular_hardening = std::max(granular_hardening, 0.0f);
        granular_fracture_strain = std::clamp(granular_fracture_strain, 1.0e-4f, 1.0f);
        granular_damage_rate = std::clamp(granular_damage_rate, 0.0f, 100.0f);
        granular_healing_rate = std::clamp(granular_healing_rate, 0.0f, 20.0f);
        granular_max_solver_substeps = std::clamp(granular_max_solver_substeps, 1, 64);
        granular_softening_temperature = std::max(granular_softening_temperature, 0.0f);
        // A zero range would make the transition a step function at one exact
        // temperature, so a particle either never softens or softens completely
        // between two frames. Keep a floor.
        granular_softening_range = std::clamp(granular_softening_range, 1.0f, 2000.0f);
        granular_residual_strength = std::clamp(granular_residual_strength, 0.0f, 1.0f);
        granular_tack_peak = std::clamp(granular_tack_peak, 0.0f, 20.0f);
        granular_thermal_conductivity = std::clamp(granular_thermal_conductivity, 0.0f, 200.0f);
    }

    // Quadratic air drag on DETACHED droplets (an isolated 3x3x3 neighbourhood,
    // i.e. spray and splash debris — not merely a sparse cell). Applied
    // implicitly as v *= 1 / (1 + k|v|dt), unconditionally stable. 0 disables.
    //
    // ★★ k has units of 1/m and IS derivable: for a droplet of diameter d,
    // k = 3·ρ_air·C_d / (4·ρ_water·d). With C_d≈0.5 and d≈3 mm that is ≈0.15.
    // The old 0.5 corresponds to sub-millimetre mist, and it was chosen while
    // this knob was doubling as a thickness dial — so it was braking the very
    // droplets a splash is made of, on top of internal_friction braking them
    // again. Two brakes on one motion is why raising the old unitless viscosity
    // to 200 was the only way to feel a difference.
    float air_drag = 0.15f;

    // Particle redistribution (reseed). Crowded-cell removals fund replacements
    // in starved interior cells during the same step. Because P2G currently uses
    // unit-mass particles, reseeding is count-conservative and cannot invent
    // liquid; emitters and open boundaries remain the count-changing paths.
    bool  reseed_enabled = true;
    // Target particles per fluid cell. 0 = use particles_per_cell (the seed
    // density). Reseed kicks in when count falls below min_per_cell or rises
    // above max_per_cell; otherwise the cell is untouched.
    int   reseed_target_per_cell = 0;
    int   reseed_min_per_cell = 3;
    int   reseed_max_per_cell = 16;
    std::size_t max_particles = 1000000;
    // ── Material-coordinate refresh ────────────────────────────────────────
    // Solver steps between resets of the SAME coordinate generation. Two
    // generations run half a period out of phase and are blended, so the
    // texture is never carried by a map older than one period.
    //
    // ★ This is the ONE dial on the stretch/crossfade trade: larger carries the
    // pattern further with the material before refreshing it (more faithful
    // advection, more smearing at the end of each life), smaller keeps the map
    // crisp at the cost of more visible crossfading between the two.
    //
    // ★ Counted in STEPS, not seconds or frames, because the map degrades with
    // deformation and steps are what deform it. A scene run at a smaller dt
    // deforms less per step and correspondingly refreshes less often in wall
    // time, which is the behaviour you want and not a coincidence.
    int   uvw_refresh_period = 240;
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
    // Optional cell-centred kinematic viscosity in m^2/s, one float per grid
    // cell, produced by Fluid::buildSubstanceViscosityField from the per-
    // substance overrides. When null (the common case) the scalar
    // `kinematic_viscosity` above is used everywhere and nothing costs anything.
    //
    // ★★ NOT OWNED. The caller keeps it alive across the step; it is grow-only
    // function-static scratch on the ParticleSimulation side, like every other
    // per-step field this solver borrows.
    //
    // ★★★ THE PRESENCE OF THIS FIELD SWITCHES THE STAGE ON, even when the
    // domain scalar is 0. The alternative — gating on the scalar alone — means
    // binding honey to a substance in an inviscid domain authors a number that
    // is read, uploaded, and then silently ignored: the liquid stays thin, no
    // error is raised, and the only symptom is that a control does nothing.
    // The field itself is only ever built when some substance actually
    // overrides (see buildSubstanceViscosityField), so this cannot switch the
    // stage on for a scene that did not ask for it.
    const std::vector<float>* substance_viscosity = nullptr;
    // ── Solid-phase substances ─────────────────────────────────────────────
    // Tags of the substances the domain declares SOLID. Their parcels are
    // already stamped into grid.solid by the caller (see
    // Fluid::buildSubstanceSolidCells); this list is what lets the PARTICLE
    // stages tell "I am inside a solid" from "I AM the solid".
    //
    // ★★★ WITHOUT THIS THE SOLID EJECTS ITSELF. advectParticles treats a
    // parcel sitting in a solid cell as swallowed by a moving collider and
    // shoves it to the nearest free cell — correct for liquid, and for a solid
    // parcel it means the chunk explodes outward the instant it becomes thick
    // enough to fill its own cell. The symptom would look like a collision
    // instability, not like a missing exception.
    //
    // ★★ NOT OWNED and at most kMaxFluidSubstanceMaterials entries, so the
    // membership test is a short linear scan resolved ONCE per particle per
    // step into a mask rather than per substep.
    const std::vector<uint32_t>* solid_substance_tags = nullptr;
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
        Sand,      // dry granular: Drucker-Prager, cohesionless
        Chocolate, // molten couverture: between oil and honey, fully no-slip
        // ★ APPEND ONLY. SceneSerializer stores current_preset as a raw int, so
        // inserting anything above shifts every saved project's material.
        WetSand,      // capillary cohesion: clumps, holds a steeper wall
        Gravel,       // coarse, strongly dilatant, no cohesion
        CohesiveSoil, // clay-like: low friction, high cohesion, blocky failure
        MoltenPlastic // thermoplastic: rigid cold, TACKY hot, then viscous
    };
    FluidPreset current_preset = FluidPreset::Water;

    void applyPreset(FluidPreset preset) {
        switch (preset) {
            case FluidPreset::Water:
                granular_enabled = false;
                // ν=0 skips the viscous solve outright: water's 1e-6 m²/s is
                // orders of magnitude below the numerical diffusion of any grid
                // coarse enough to render, so paying for the sweeps buys motion
                // you cannot see.
                kinematic_viscosity = 0.0f;   viscosity_sweeps = 1;
                viscosity_wall_slip = 1.0f;
                // ★★★ BOTH DISSIPATION DIALS WERE LEFT AT THEIR FAKE-THICKNESS
                // VALUES when the thick presets were converted to a physical ν.
                // Water is the preset with NOTHING to fake, so it was the one
                // paying the whole cost: 0.5 body decay plus 0.5 droplet drag
                // meant a 2 m pour arrived slow and its spray died on contact.
                // Water resists shear, and its shear is below what any
                // renderable voxel can express — so both of these belong at (or
                // near) zero, and the splash comes back on its own.
                internal_friction = 0.0f;
                flip_blend = 0.97f; apic_blend = 0.95f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 0.15f;  wall_damping = 0.15f;
                affine_damping = 0.98f; max_velocity = 50.0f;
                break;
            case FluidPreset::Oil:
                granular_enabled = false;
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
                granular_enabled = false;
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
                granular_enabled = false;
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
                granular_enabled = false;
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
                granular_enabled = false;
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
                granular_enabled = true;
                granular_friction_angle_degrees = 35.0f;
                granular_cohesion = 0.0f;
                granular_dilatancy_degrees = 5.0f;
                granular_young_modulus = 2.0e5f;
                granular_poisson_ratio = 0.25f;
                granular_tensile_cutoff = 0.0f;
                granular_hardening = 0.0f;
                granular_fracture_strain = 0.02f;
                granular_damage_rate = 8.0f;
                granular_healing_rate = 0.0f;
                granular_rebonding = false;
                granular_max_solver_substeps = 32;
                // Grain friction is a SHEAR YIELD criterion (Drucker-Prager),
                // not a velocity Laplacian — no value of ν produces an angle of
                // repose. Both backends now run the rate-form Drucker-Prager
                // stress update and stress-divergence P2G, so there is no
                // viscous stand-in left to configure.
                kinematic_viscosity = 0.0f;   viscosity_sweeps = 1;
                viscosity_wall_slip = 0.0f;
                // ★★★ THE TWO DEAD DIALS, REMOVED RATHER THAN LEFT AT A VALUE
                // THAT NO LONGER REACHES ANYTHING.
                //
                // internal_friction carried 4.0 as the pre-Drucker-Prager
                // stand-in for grain friction. Both G2P paths now zero it when
                // granular_enabled, so 4.0 described a drag the solver does not
                // apply — a dial a user would tune for an hour with no effect.
                //
                // flip_blend carried 0.92, and FLIP is forced OFF for granular
                // on both paths: the elastic stress goes into the grid during
                // P2G, i.e. before the FLIP snapshot, so a non-zero blend
                // subtracts the stress back out. That is not a preference, it
                // is the reason a soft pile used to collapse. Zero here says so.
                internal_friction = 0.0f;
                flip_blend = 0.0f; apic_blend = 0.95f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 0.15f;  wall_damping = 0.40f;
                affine_damping = 0.98f; max_velocity = 50.0f;
                break;

            // ── Granular family ──────────────────────────────────────────────
            // ★ E IS CALIBRATED AGAINST PILE DEPTH, not picked for feel. The
            // corotational predictor needs E >= 10*rho*g*h to keep the bottom
            // layer's elastic strain inside small-strain (see
            // kGranularStiffnessLoadRatio), so each preset below states the
            // depth it is honest up to. Past that depth the panel's
            // "TOO SOFT FOR LOAD" row lights up rather than the pile quietly
            // sinking — raise E, or accept a shallower pour.
            //
            // ★★ granular_max_solver_substeps IS PART OF THE MATERIAL, because
            // it decides how much of the authored E the solver can deliver
            // (E_eff = E * (granted/needed)^2). The ceilings below are sized for
            // roughly 5 cm voxels at 24 fps, the common preview setup; coarser
            // voxels need fewer. Watch "Granular Young effective" in the panel —
            // if it sits below requested, this ceiling is the reason.
            case FluidPreset::WetSand:
                granular_enabled = true;
                // Capillary bridges: wet sand stands steeper than dry and holds
                // a shape, which is tensile strength, not extra friction. That
                // is what tensile_cutoff buys — and rebonding, because a
                // squeezed clump genuinely re-forms.
                granular_friction_angle_degrees = 37.0f;
                granular_cohesion = 1500.0f;      // apparent capillary cohesion
                granular_dilatancy_degrees = 6.0f;
                granular_young_modulus = 2.5e5f;  // honest to ~1.6 m of pile
                granular_poisson_ratio = 0.25f;
                granular_tensile_cutoff = 400.0f;
                granular_hardening = 0.0f;
                granular_fracture_strain = 0.010f;
                granular_damage_rate = 14.0f;     // clumps part cleanly
                granular_healing_rate = 0.5f;
                granular_rebonding = true;
                granular_max_solver_substeps = 32;
                kinematic_viscosity = 0.0f;   viscosity_sweeps = 1;
                viscosity_wall_slip = 0.0f;
                internal_friction = 0.0f;
                flip_blend = 0.0f; apic_blend = 0.95f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 0.12f;  wall_damping = 0.45f;
                affine_damping = 0.98f; max_velocity = 50.0f;
                break;
            case FluidPreset::Gravel:
                granular_enabled = true;
                // Coarse angular grains interlock: high friction, strong
                // dilatancy (the pile must loosen to shear at all), and no
                // cohesion whatsoever. Heavier grains also fall through air
                // more cleanly, hence the low drag.
                granular_friction_angle_degrees = 43.0f;
                granular_cohesion = 0.0f;
                granular_dilatancy_degrees = 12.0f;
                granular_young_modulus = 3.0e5f;  // honest to ~1.9 m of pile
                granular_poisson_ratio = 0.22f;
                granular_tensile_cutoff = 0.0f;
                granular_hardening = 0.0f;
                granular_fracture_strain = 0.02f;
                granular_damage_rate = 8.0f;
                granular_healing_rate = 0.0f;
                granular_rebonding = false;
                // ★ 40, not 32: at 5 cm / 24 fps this E needs 33 substeps, so a
                // 32 ceiling would silently deliver a softer gravel than the
                // number in the panel.
                granular_max_solver_substeps = 40;
                kinematic_viscosity = 0.0f;   viscosity_sweeps = 1;
                viscosity_wall_slip = 0.0f;
                internal_friction = 0.0f;
                flip_blend = 0.0f; apic_blend = 0.90f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 0.08f;  wall_damping = 0.50f;
                affine_damping = 0.96f; max_velocity = 50.0f;
                break;
            case FluidPreset::CohesiveSoil:
                granular_enabled = true;
                // Clay-like: the strength is cohesion, not friction, and it
                // does not dilate. Failure is blocky — a low fracture strain
                // with a fast damage rate localises cracks instead of spreading
                // damage through the body. No rebonding: once a clay block
                // parts it stays parted.
                granular_friction_angle_degrees = 20.0f;
                granular_cohesion = 12000.0f;
                granular_dilatancy_degrees = 0.0f;
                granular_young_modulus = 1.2e5f;  // honest to ~0.76 m of pile
                granular_poisson_ratio = 0.35f;   // nearly incompressible skeleton
                granular_tensile_cutoff = 3000.0f;
                granular_hardening = 0.5f;
                granular_fracture_strain = 0.006f;
                granular_damage_rate = 20.0f;
                granular_healing_rate = 0.0f;
                granular_rebonding = false;
                granular_max_solver_substeps = 32;
                kinematic_viscosity = 0.0f;   viscosity_sweeps = 1;
                viscosity_wall_slip = 0.0f;
                internal_friction = 0.0f;
                flip_blend = 0.0f; apic_blend = 0.95f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 0.10f;  wall_damping = 0.60f;  // clay clings
                affine_damping = 0.98f; max_velocity = 50.0f;
                break;
            case FluidPreset::MoltenPlastic:
                // ★★★ The whole point of this preset is that it TURNS ON the
                // thermal chain. Every dial below is inert on its own: without
                // granular_enabled there is no skeleton to lose, without
                // softening_temperature the thermal path is disabled, and
                // without thermal_conductivity only the surface ever heats.
                // Leaving a user to find three separate switches is how the
                // feature stayed invisible after it was written.
                granular_enabled = true;
                granular_friction_angle_degrees = 30.0f;
                granular_cohesion = 6000.0f;      // cold pellets already stick a little
                granular_dilatancy_degrees = 2.0f;
                granular_young_modulus = 2.0e5f;
                granular_poisson_ratio = 0.40f;   // polymers are nearly incompressible
                granular_tensile_cutoff = 1500.0f;
                granular_hardening = 0.0f;
                granular_fracture_strain = 0.02f;
                granular_damage_rate = 6.0f;
                granular_healing_rate = 2.0f;     // tacky material re-bonds on contact
                granular_rebonding = true;
                granular_max_solver_substeps = 32;
                // Softens around a typical thermoplastic working range, over a
                // wide window so the tacky phase is actually visible rather than
                // being crossed in a single frame.
                granular_softening_temperature = 420.0f;
                granular_softening_range = 90.0f;
                // ★ No residual skeleton: molten plastic must release fully into
                // viscous flow. A floor here would leave a weak solid that sags
                // forever and never behaves like a liquid. (A CHARRING material
                // is the opposite case and wants a real residual — that is the
                // per-substance split, not this preset.)
                granular_residual_strength = 0.0f;
                // ★★★ The correction this preset exists to carry: bonds PEAK
                // mid-transition instead of fading with stiffness, so the block
                // fuses and slumps as one body instead of crumbling into grains.
                granular_tack_peak = 3.5f;
                granular_thermal_conductivity = 2.5f;
                kinematic_viscosity = 0.0f;   viscosity_sweeps = 1;
                viscosity_wall_slip = 0.0f;
                internal_friction = 0.0f;
                flip_blend = 0.0f; apic_blend = 0.95f;
                velocity_damping = 0.999f;
                density_correction = 1.0f;
                air_drag = 0.05f;  wall_damping = 0.70f;
                affine_damping = 0.98f; max_velocity = 50.0f;
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
    float granular_constitutive_ms = 0.0f;
    float pressure_cg_dot_ms = 0.0f;
    double pressure_cg_final_relative_residual = 0.0;
    int   cpu_threads = 1;
    int   advect_substeps = 1;
    int   pressure_cg_iterations = 0;
    int   pressure_cg_max_iterations = 0;
    int   pressure_cg_dot_count = 0;
    bool  pressure_cg_multigrid = false;
    size_t particle_count = 0;
    // Dynamic reseeding may only replace unit-mass samples removed from
    // crowded cells in the same step; these counters expose that invariant.
    size_t reseed_added_particles = 0;
    size_t reseed_removed_particles = 0;
    size_t grid_cell_count = 0;
    size_t active_fluid_cells = 0;
    size_t recovered_solid_particles = 0;
    size_t granular_yielded_particles = 0;
    size_t granular_detached_particles = 0;
    size_t granular_invalid_particles = 0;
    size_t granular_sleeping_particles = 0;
    size_t granular_damaged_particles = 0;
    size_t granular_damage_over_10_particles = 0;
    size_t granular_damage_over_50_particles = 0;
    size_t granular_damage_over_90_particles = 0;
    float granular_max_damage = 0.0f;
    float granular_mean_damage = 0.0f;
    float granular_max_yield_value = 0.0f;
    float granular_max_plastic_increment = 0.0f;
    float granular_max_accumulated_plastic = 0.0f;
    float granular_mean_accumulated_plastic = 0.0f;
    float granular_max_fracture_history = 0.0f;
    float granular_mean_fracture_history = 0.0f;
    float granular_requested_young_modulus = 0.0f;
    float granular_effective_young_modulus = 0.0f;
    int granular_required_substeps = 1;
    int granular_solver_substeps = 1;
    bool granular_stiffness_capped = false;
    // The subcycle answers to two limits; reporting only the total hides which
    // one asked, and hides the case where NEITHER was granted because
    // granular_max_solver_substeps clamped the request.
    int granular_wave_substeps = 1;
    int granular_strain_substeps = 1;
    float granular_strain_rate = 0.0f;
    // Particles whose dt*C had to be clamped by the stress kernel. Non-zero
    // means the subcycle was too coarse for the motion, and the step survived
    // rather than being correct.
    size_t granular_strain_limited_particles = 0;
    // Particles whose elastic strain exceeded what a deformation gradient can
    // store, so the excess became permanent compaction. This is REAL
    // PLASTICITY on soft material, not an error -- it replaces the det(F)
    // reset that used to dump such a particle's whole stress in one step.
    size_t granular_compaction_capped_particles = 0;
    float granular_min_softening = 1.0f;
    size_t granular_softened_particles = 0;
    // Material validity, not stability: the load the domain puts on its own
    // bottom layer, and the stiffness the small-strain model needs to carry it.
    float granular_overburden_pressure = 0.0f;
    float granular_young_modulus_for_load = 0.0f;
    bool granular_stiffness_below_load = false;
    // Parcels of a SOLID-phase substance, and the cells they actually blocked
    // this step.
    //
    // ★★★ BOTH NUMBERS, because their disagreement is the diagnosis. Parcels
    // present with zero cells means the chunk is thinner than the voxel size
    // can express — the phase control did land, the grid just cannot hold it,
    // and raising the resolution fixes it. Reporting only the cells would make
    // that indistinguishable from "the binding never took", and the user would
    // go looking in the wrong layer.
    size_t solid_phase_particles = 0;
    size_t solid_phase_cells = 0;
    // Fluid regions the projection found with NO pressure reference this step,
    // and how many cells they held. A sealed region is one whose every face is
    // either fluid-fluid or closed (solid / closed domain wall): the matrix
    // block is singular and PCG grows its null-space component a little every
    // iteration, so the pressure -- and the kick it hands the particles --
    // scales with `pressure_iterations` instead of converging.
    //
    // ★★ REPORTED, NOT JUST FIXED. These two numbers are what tells "the
    // pocket fix did something" apart from "there was no pocket and the change
    // is inert". Without them the repair is unfalsifiable, which is exactly how
    // a calibration round starts.
    size_t sealed_pockets = 0;
    size_t sealed_pocket_cells = 0;
    // ★★★ A ZERO ABOVE IS ONLY A MEASUREMENT IF THIS IS TRUE. The pocket scan
    // lives in the CPU free-surface projection; the GPU MGPCG path and the
    // non-free-surface solver never run it, and there a zero would mean "not
    // looked at" while reading exactly like "looked and found none". That is the
    // silent failure this project keeps re-learning, so the two states are kept
    // apart in the data rather than in a comment.
    bool sealed_pockets_measured = false;
    // Fluid cells with liquid on all six sides, out of `active_fluid_cells`.
    //
    // \u2605\u2605\u2605 THE RESOLUTION READING. A cell touching air is pinned near p = 0 by
    // the free-surface condition, so only INTERIOR cells carry a pressure
    // field. Interior \u2248 0 with a healthy particle count means the liquid is
    // thinner than the grid can express: it is not a fluid to this solver, it
    // is loose particles, and it will fall without a splash and land in a heap
    // no matter what the viscosity says. The cure is mass per second or voxel
    // size -- and nothing else, which is exactly what makes this worth
    // reporting instead of inferring from the picture.
    size_t interior_fluid_cells = 0;
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

// Cell-scale heat conduction between particles. Call ONCE PER FRAME with the
// frame dt — see the definition for why per-substep would make the melt rate
// depend on the substep count. `conductivity` is 1/s; 0 disables it, which
// leaves a burning body heating only at its surface.
void diffuseParticleTemperature(FluidParticles& parts,
                                const FluidSim::FluidGrid& grid,
                                float conductivity,
                                float dt);

void seedBox(FluidParticles& particles,
             const FluidSim::FluidGrid& grid,
             const Vec3& min_world,
             const Vec3& max_world,
             int particles_per_cell,
             uint32_t seed = 0u,
             size_t max_new_particles = static_cast<size_t>(-1));

} // namespace Fluid
} // namespace RayTrophiSim
