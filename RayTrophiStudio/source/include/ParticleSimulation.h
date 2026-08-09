#pragma once

#include "Matrix4x4.h"
#include "FluidGrid.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/APICFluidSolver.h"
#include "Fluid/FluidLevelSet.h"
#include "Fluid/FluidFoam.h"
#include "Fluid/FluidRenderMode.h"
#include "GridFluidSolver.h"   // GridFluid::GasSolverStats (gas step telemetry)
#include "VolumeShader.h"
#include "SimulationWorld.h"

#include <memory>
#include <atomic>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <SurfaceMeshCache.h>
#include "ColliderMeshBVH.h"
#include "MaterialStateField.h"

namespace RayTrophiSim {

enum class ParticleEmitterSourceMode {
    Point,
    ObjectOrigin,
    ForceFieldOrigin
};

enum class ParticleColliderSourceMode {
    PlaneY,
    ObjectAABB,
    ObjectOBB,
    Sphere,
    Capsule,
    ObjectMeshSDF,
    ObjectConvexDecomp,
    ObjectMeshBVH
};

enum class ParticleEmitterSpawnMode {
    Center,
    ObjectAABBSurface,
    MeshSurface
};

enum class ParticlePhysicsMode {
    Spark,
    Granular,
    Fluid,
    Gas
};

enum class ParticleQualityMode {
    Realtime,
    Preview,
    Offline
};

// Tag for what kind of solver a grid domain feeds. Gas is the legacy gas/smoke
// path through GridFluid::step (advect → projection → buoyancy → fire). Fluid
// is the APIC liquid path (P2G → projection → G2P → advect particles). All
// shared infra (bounds, voxel_size, transform/gizmo, flow sources, colliders,
// serialization, timeline cache) lives on the same domain — the type just
// picks the per-step solver.
enum class SimulationDomainType {
    Gas,
    Fluid
};

enum class SimulationDomainBackend {
    CPU_Dense     = 0,
    // GPU_Compute: auto-selects CUDA if available, falls back to Vulkan.
    // UI shows "GPU Compute" — backend is transparent to the user.
    GPU_Compute   = 1,
    CPU_SparseVDB = 2,
    // GPU_Vulkan: forces Vulkan even when CUDA is present. No longer a testing-only
    // escape hatch — it is what new domains default to (defaultSimulationDomainBackend).
    GPU_Vulkan    = 3,
    // Legacy alias kept so old project files load correctly.
    GPU_CUDA      = GPU_Compute,
};

// Backend a newly created domain starts on, resolved from what this machine can
// actually run: Vulkan compute first, CUDA only when Vulkan compute is missing,
// CPU when neither answers. Vulkan leads on purpose — see the definition. The step
// loop falls back to CPU on its own anyway, so handing this out is safe even on a
// machine with no compute device. Defined out-of-line because it reads the runtime
// capability globals.
SimulationDomainBackend defaultSimulationDomainBackend();

enum class SimulationGridDomainSourceMode {
    ManualBox,
    ObjectBounds,
    Adaptive
};

enum class SimulationGridDomainBoundaryMode {
    Open,
    Closed,
    Periodic
};

// User-facing production intent. Custom preserves manual tuning; the other
// profiles provide predictable safety/cost defaults without hiding the actual
// resolution and memory budget from advanced users.
enum class SimulationDomainQualityProfile : uint32_t {
    Interactive = 0,
    Preview = 1,
    Final = 2,
    Cinema = 3,   // Cinema-grade: RAM limit lifted, disk bake mandatory, grid up to 2048
    Custom = 4
};

enum class SimulationFlowSourceMode {
    Point,
    ObjectBounds,
    MeshSurface
};

// How a parented emission source (flow source OR particle emitter) interprets
// its authored direction/velocity vector.
//   Local — the vector lives in the parent's frame and rotates with it, so a
//           nozzle keeps spraying out of its own muzzle as the prop turns.
//   World — the vector stays a world direction regardless of parent rotation
//           (matches the behaviour of an unparented source).
// Ignored entirely when the source has no parent.
enum class SimulationEmissionVelocitySpace : uint32_t {
    Local = 0,
    World = 1
};

enum class SimulationGridDomainChannelFlags : uint32_t {
    Density = 1u << 0u,
    Temperature = 1u << 1u,
    Velocity = 1u << 2u,
    Fuel = 1u << 3u,
    Pressure = 1u << 4u
};

inline uint32_t defaultGridDomainChannels() {
    return static_cast<uint32_t>(SimulationGridDomainChannelFlags::Density) |
           static_cast<uint32_t>(SimulationGridDomainChannelFlags::Temperature) |
           static_cast<uint32_t>(SimulationGridDomainChannelFlags::Velocity) |
           static_cast<uint32_t>(SimulationGridDomainChannelFlags::Pressure);
}

// How the APIC liquid is initialised when "Seed Fluid" runs.
//   SeedBox   — fill the user-positioned fluid_seed_min/max AABB (legacy; good
//               for a localized blob you then drop / emit from).
//   FillLevel — pre-fill the whole domain footprint from the floor up to
//               fluid_fill_level (fraction of domain height) as a resting tank.
//               Skips the long emission/settling transient for standing-water
//               setups ("dolu kap"); colliders then carve waves on top.
enum class FluidSeedMode : uint32_t {
    SeedBox = 0,
    FillLevel = 1
};

// Resting-tank fill region (single source of truth for the seeder, the viewport
// gizmo and the UI readout). Fills the full XZ footprint (inset by wall_margin)
// from the floor up to fill_level of the domain height, then caps the height to
// whatever `budget` particles afford at `ppc` (COMPLETE horizontal layers). ppc
// is a stability constant (> 1 for real internal pressure), so an under-budget
// fill loses HEIGHT, never density — the level just becomes what the budget
// supports. Returns the effective fill fraction actually seeded (0..fill_level).
inline float computeFluidFillSeedAABB(const Vec3& bounds_min,
                                      const Vec3& bounds_max,
                                      float voxel_size,
                                      float fill_level,
                                      float wall_margin,
                                      int ppc,
                                      std::size_t budget,
                                      Vec3& out_lo,
                                      Vec3& out_hi) {
    const float lvl = std::clamp(fill_level, 0.0f, 1.0f);
    const float m   = std::max(0.0f, wall_margin);
    const float height = std::max(0.0f, bounds_max.y - bounds_min.y);
    out_lo = Vec3(bounds_min.x + m, bounds_min.y, bounds_min.z + m);
    out_hi = Vec3(bounds_max.x - m, bounds_min.y + height * lvl, bounds_max.z - m);
    if (voxel_size <= 0.0f || ppc <= 0 || height <= 0.0f) return lvl;

    // Footprint = cells in one floor-level XZ layer.
    const float fx = std::max(0.0f, out_hi.x - out_lo.x);
    const float fz = std::max(0.0f, out_hi.z - out_lo.z);
    const std::size_t nx = static_cast<std::size_t>(std::floor(fx / voxel_size));
    const std::size_t nz = static_cast<std::size_t>(std::floor(fz / voxel_size));
    const std::size_t footprint = nx * nz;
    if (footprint == 0) return 0.0f;

    const std::size_t affordable_layers =
        (budget / static_cast<std::size_t>(ppc)) / footprint;
    const float budget_max_y =
        bounds_min.y + static_cast<float>(affordable_layers) * voxel_size;
    if (budget_max_y < out_hi.y) out_hi.y = budget_max_y;  // budget caps the level

    return (out_hi.y - bounds_min.y) / height;  // effective fill fraction
}

struct ParticlePhysicsSettings {
    ParticlePhysicsMode mode = ParticlePhysicsMode::Spark;
    ParticleQualityMode quality = ParticleQualityMode::Realtime;
    float particle_radius = 0.04f;
    bool self_collision_enabled = false;
    int solver_iterations = 1;
    int max_neighbors_per_particle = 32;
    float viscosity = 0.0f;
    float cohesion = 0.0f;
    float pressure_stiffness = 0.0f;
    float rest_density = 1000.0f;
    float buoyancy = 0.0f;
    float gravity_scale = 1.0f;
    float vorticity = 0.0f;

    // ── Particle → gas grid coupling ─────────────────────────────────────────
    // Discrete particles deposit into every Gas domain they fly through, so
    // scattering debris CARRIES fire and smoke instead of being a decorative
    // overlay on top of an unrelated volume. Rates are per SECOND (scaled by dt,
    // so the look no longer changes with framerate) and per particle.
    //   density     → smoke trail
    //   temperature → the ember glows and lifts the gas around it
    //   fuel        → the ember can IGNITE gas it passes through; this is what
    //                 makes an explosion's debris spread flame outward
    // Fuel deposit needs the domain's Fuel channel + fire_enabled to do anything.
    float grid_density_deposit = 0.0f;
    float grid_temperature_deposit = 0.0f;
    float grid_fuel_deposit = 0.0f;
    // Scale deposit by remaining life (full at birth → 0 at death). A cooling
    // ember should stop igniting things long before it disappears.
    bool grid_deposit_fade_with_age = true;
};

struct SimulationGridDomainDesc {
    std::string name = "Grid Domain";
    // Defaults to Gas so existing projects deserialize unchanged. Fluid domains
    // are created explicitly from the UI (or migrated from legacy FluidObject).
    SimulationDomainType type = SimulationDomainType::Gas;
    SimulationDomainBackend backend = SimulationDomainBackend::CPU_Dense;
    SimulationGridDomainSourceMode source_mode = SimulationGridDomainSourceMode::ManualBox;
    // Struct default stays Open so gas/smoke domains (the default type) let
    // their medium leave the box and existing projects deserialize unchanged.
    // Fluid domains are switched to Closed at creation/type-conversion in the UI
    // so liquid pools instead of silently draining through the walls.
    SimulationGridDomainBoundaryMode boundary_mode = SimulationGridDomainBoundaryMode::Open;
    std::string source_name;
    bool enabled = true;
    bool preserve_voxel_size_on_resize = true;
    bool use_sparse_tiles = true;
    bool render_to_nanovdb = true;
    Vec3 bounds_min = Vec3(-2.5f, 0.0f, -2.5f);
    Vec3 bounds_max = Vec3(2.5f, 5.0f, 2.5f);
    int resolution_x = 64;
    int resolution_y = 64;
    int resolution_z = 64;
    int max_auto_resolution = 128;
    SimulationDomainQualityProfile quality_profile = SimulationDomainQualityProfile::Preview;
    // Hard combined working-set budget for this domain. The solver clamps grid
    // dimensions before allocation; this is an execution guard, not just a UI
    // estimate. Zero disables the guard for deliberate expert/offline runs.
    uint32_t resource_budget_mb = 1024;
    bool enforce_resource_budget = true;
    // Cinema-grade volumetric disk cache. Directory auto-derived from project
    // path as <project>.volcache (same pattern as fluid's <project>.simcache).
    // When true the solver writes baked grid frames to disk instead of holding
    // the full working set in RAM — enables grid resolutions beyond what system
    // memory can sustain (512+). Set automatically by the Cinema profile.
    bool force_disk_cache = false;
    float voxel_size = 0.1f;
    float padding = 0.0f;
    bool adaptive_lock_floor = true;
    float adaptive_floor_y = 0.0f;
    uint32_t channels = defaultGridDomainChannels();
    // ── Fluid (APIC liquid) parameters. Only consumed when type == Fluid.
    //    Live alongside gas params so the same domain instance can be retyped
    //    later without losing settings. Defaults mirror the old FluidObject.
    Fluid::APICSolverParams fluid_params;
    Vec3 fluid_seed_min = Vec3(-0.5f, 1.0f, -0.5f);
    Vec3 fluid_seed_max = Vec3(0.5f, 1.5f, 0.5f);
    int  fluid_seed_particles_per_cell = 8;
    std::size_t fluid_max_particles = 100000;
    bool fluid_replace_on_seed = true;
    bool fluid_pending_seed = false;
    // Authored initial-state intent, kept separate from fluid_pending_seed
    // (a one-tick runtime command). Once a replace seed is requested, timeline
    // rewind/cache invalidation must reproduce it; continuous-flow domains leave
    // this false and therefore restart empty.
    bool fluid_reseed_on_reset = false;
    // Optional APIC-liquid -> gas combustion coupling. The liquid remains an
    // incompressible particle/SDF solve; only its GPU free-surface band becomes
    // a thermally responsive fuel source for overlapping Gas domains.
    bool  fluid_flammable = false;
    bool  fluid_extinguishing = false;
    bool  fluid_auto_ignite = false;
    float fluid_ignition_temperature = 0.8f;
    float fluid_evaporation_rate = 0.35f;
    float fluid_surface_fuel_capacity = 4.0f;
    float fluid_combustion_heat_release = 2.0f;
    float fluid_combustion_smoke_yield = 0.45f;
    float fluid_surface_cooling = 0.35f;
    float fluid_cooling_power = 0.0f;
    float fluid_oxygen_dilution = 0.0f;
    // Seed strategy. SeedBox (default) keeps old projects' behaviour. FillLevel
    // ignores fluid_seed_min/max and instead fills the domain footprint from the
    // floor up to fluid_fill_level of the domain height, optionally inset from
    // the side walls by fluid_fill_wall_margin (world units).
    FluidSeedMode fluid_seed_mode = FluidSeedMode::SeedBox;
    float fluid_fill_level = 0.5f;        // 0..1 fraction of domain height
    float fluid_fill_wall_margin = 0.0f;  // world-unit inset from the side walls
    // Translation anchors: previous frame's domain min/max corners. We use
    // BOTH so we can distinguish translation (both corners shift by the
    // same delta → seed follows) from resize (only one corner moves → seed
    // stays put). Sentinel x < -9999 means "uninitialized — adopt this
    // frame's bounds, no shift this tick; also drop a sensible default
    // seed AABB inside the domain".
    Vec3 fluid_seed_anchor_min = Vec3(-1.0e10f, 0.0f, 0.0f);
    Vec3 fluid_seed_anchor_max = Vec3(0.0f, 0.0f, 0.0f);
    // Combustion (opt-in). Requires the Fuel + Temperature channels and a fuel-
    // injecting flow source. See GridFluid::SolverParams for the model.
    bool fire_enabled = false;
    float ignition_temperature = 0.3f;
    float burn_rate = 1.5f;
    float heat_release = 2.0f;
    float smoke_generation = 0.6f;
    float flame_dissipation = 3.0f;
    float fire_max_temperature = 10.0f;
    // Gas motion belongs to the domain, not to the discrete-particle physics
    // preset. Hybrid effects (spark particles + fire grid) otherwise inherit
    // Spark's zero buoyancy and move only through combustion expansion.
    // ── Thermal boundary override (Phase 4) ──────────────────────────────────
    // A domain overrides the WORLD's ambient conditions inside its own box. Off
    // by default, so a domain that says nothing simply inherits the world and no
    // existing scene changes.
    //
    // ★ There is deliberately no per-domain `kelvin_per_unit`. That is the
    // normalized<->Kelvin calibration, and the char mask quantizes surface
    // temperature in absolute Kelvin — if two domains disagreed, the same object
    // would report a different temperature and glow differently depending on
    // which box it stood in. Ambient and oxygen are boundary conditions; the
    // unit mapping is not one. See WorldThermalState in MaterialStateField.h.
    bool  thermal_override_enabled = false;
    float thermal_ambient_kelvin = 293.0f;
    float thermal_oxygen = 1.0f;   // 0..1, throttles pyrolysis inside this box
    float gas_buoyancy_heat = 1.0f;
    float gas_buoyancy_density = 0.08f;
    float gas_vorticity = 0.35f;
    // Advection scheme. Plain semi-Lagrangian is first-order and smears smoke
    // into mush within a few dozen frames; MacCormack adds a limited
    // second-order correction (one extra pass) and is what keeps wisps and
    // flame fronts readable. Defaults to MacCormack for new domains — the
    // deserializer keeps whatever an older project stored.
    bool gas_maccormack_advection = true;
    // Thermal expansion: gas dilation driven by (temperature - ambient). Gives
    // fire its rolling billow and turns a sudden fuel ignition into a real
    // explosion blast. 0 = incompressible smoke (default; old projects unchanged).
    float fire_expansion = 0.0f;
    // Procedural curl-noise turbulence (Gas domains, dense CPU/GPU path). Adds
    // divergence-free FBM swirl modulated by local activity. 0 strength = off.
    // Not applied on the CPU_SparseVDB backend (which also skips vorticity).
    float turbulence_strength = 0.0f;
    float turbulence_scale = 1.2f;
    int   turbulence_octaves = 3;
    float turbulence_lacunarity = 2.0f;
    float turbulence_persistence = 0.5f;
    float turbulence_speed = 0.5f;
    // Per-domain volume render shader (host material data; created lazily by the
    // render bridge / UI). Travels with the domain for serialization and is
    // bound to the domain's live VDB volume. Not used by the solver.
    std::shared_ptr<VolumeShader> shader;

    // ── Fluid render mode (consumed when type == Fluid). ─────────────────────
    // Default is Volume so projects load unchanged. Other modes activate the
    // particle-sphere mirror (debug) or the level-set surface proxy.
    Fluid::FluidRenderMode fluid_render_mode = Fluid::FluidRenderMode::Volume;

    // Particles-mode render config (consumed only when fluid_render_mode ==
    // Particles). Mirrors the per-system render in ParticleSystemObject.
    Vec3  fluid_particle_color = Vec3(0.40f, 0.65f, 0.95f);
    float fluid_particle_radius_factor = 0.45f;
    float fluid_particle_size_multiplier = 1.0f;
    int   fluid_particle_subdivisions = 1;
    bool  fluid_particle_emissive = false;
    float fluid_particle_emission = 0.0f;
    // Optional: explicit MaterialManager material id for the particle spheres.
    // -1 (the default) falls back to the auto-synthesised PBR built from the
    // color / emissive fields above. When set, the user can author full water
    // PBR (high transmittance, IOR 1.33, low roughness) via the Materials
    // panel and have it actually drive the fluid sphere look.
    int   fluid_particle_material_id = -1;

    // SurfaceSDF-mode params (level-set narrow-band SDF + density-proxy band).
    Fluid::LevelSetParams fluid_level_set_params;
    float fluid_surface_band_voxels = 0.5f;
    // Whitewater (spray/foam/bubbles) generation — Ihmsen 2012. Off by default.
    Fluid::FoamParams fluid_foam_params;
    // Volume render mode for whitewater: white scattering NanoVDB shader.
    // Lazily created; editable in the foam UI like the liquid volume shader.
    std::shared_ptr<VolumeShader> foam_shader;
    // Index of refraction for the isosurface dielectric boundary. 1.33 = water,
    // 1.5 = glass, 1.0 = no bending. The absorption colour / coefficient that
    // tint the light by depth come from the domain's VolumeShader (NanoVDB
    // Render panel) so the whole "water material" is authored in one place.
    float fluid_surface_ior = 1.33f;
    // Surface roughness 0..1. 0 = mirror-smooth (still/glassy water), higher =
    // choppy / frosted (GGX normal perturbation on the dielectric boundary).
    float fluid_surface_roughness = 0.0f;
    // Whitewater/foam strength 0..1. Whitens high-curvature surface regions
    // (wave crests, breaking edges, splash) via the SDF Laplacian. 0 = off.
    float fluid_surface_foam = 0.0f;

    // ImGui debug overlay (cyan blob per particle on top of everything). Used
    // to preview fluid coverage / debug seeding when the RT route hasn't
    // converged yet. Off by default in the post-Phase-2 flow because the
    // Particles render mode draws RT instanced spheres for the same purpose
    // — the overlay on top would just double-paint.
    bool fluid_debug_overlay = false;
};

// Per-step telemetry for one Gas domain — the counterpart of
// Fluid::APICSolverStats, which is what the Fluid panel reports. The gas step
// is a hybrid: individual stages run on the device while the rest of the
// operator chain stays in GridFluid::step on the host, so every stage carries
// BOTH a time and where it ran. Without the flag a 0.00 ms row is ambiguous —
// it can mean "ran on the GPU", "was skipped because the channel is off", or
// "the stage is genuinely free at this resolution".
//
// All times are host wall time. The GPU rows are measured around the dispatch
// call, so they include whatever submit/fence the host waited on — that is the
// cost the frame actually pays, not the isolated kernel time.
struct SimulationGasStats {
    // False until a step ran. An idle domain (no source, no content) is skipped
    // wholesale, and reporting the previous step's numbers as if they were
    // current is how "the gas panel shows work but nothing moves" happens.
    bool  stepped = false;

    float total_ms = 0.0f;      // whole per-domain gas step, host wall time
    float voxelize_ms = 0.0f;   // collider stamp into grid.solid + face weights
    float analysis_ms = 0.0f;   // the post-step O(cells) field scan below

    // Device stages, in execution order.
    float gpu_collider_source_ms = 0.0f;   // collider Gas Interaction injection
    float gpu_msf_ms = 0.0f;               // Material State Field thermal gather
    float gpu_source_upload_ms = 0.0f;     // host scalar sources -> device
    float gpu_fluid_combustion_ms = 0.0f;  // combustible liquid surface deposit
    float gpu_velocity_advect_ms = 0.0f;
    float gpu_scalar_advect_ms = 0.0f;
    float gpu_combustion_ms = 0.0f;
    float gpu_body_forces_ms = 0.0f;       // vel upload + buoyancy/fields/vorticity/turbulence + readback
    float gpu_dissipation_ms = 0.0f;       // velocity loss + clamp
    float gpu_pressure_ms = 0.0f;          // projection
    float gpu_publish_ms = 0.0f;           // final scalar publication for the RT bridge
    float gpu_majorant_ms = 0.0f;          // per-block density max for the RT empty-space skip

    // Host operator chain that still had to run (stages the device covered are
    // switched off inside it, so its rows are the true CPU residual).
    GridFluid::GasSolverStats cpu;

    // Where each stage ran this step. A false flag with a 0 ms CPU row means the
    // stage did not run at all (channel off / feature disabled).
    bool velocity_advect_on_gpu = false;
    bool scalar_advect_on_gpu = false;
    bool combustion_on_gpu = false;
    bool buoyancy_on_gpu = false;
    bool force_fields_on_gpu = false;
    bool vorticity_on_gpu = false;
    bool turbulence_on_gpu = false;
    bool dissipation_on_gpu = false;
    bool pressure_on_gpu = false;
    bool fluid_combustion_on_gpu = false;

    // ── Field measurements (post-step scan) ──────────────────────────────────
    std::size_t cell_count = 0;
    std::size_t active_fuel_cells = 0;   // fuel > 1e-4
    std::size_t burning_cells = 0;       // flame/interaction field lit
    std::size_t solid_cells = 0;         // collider-occupied cells
    float max_temperature = 0.0f;
    float total_density = 0.0f;          // smoke mass proxy (sum over cells)
    float total_fuel = 0.0f;
    float max_speed = 0.0f;              // max |v| over MAC faces, world units/s
    // max_speed * dt / voxel_size. Above 1 the semi-Lagrangian trace jumps more
    // than a cell per step: detail smears and the projection fights it.
    float cfl = 0.0f;
    // Dense host storage for this domain's channels (grid.* vectors).
    std::size_t grid_memory_bytes = 0;
};

struct SimulationGridDomainState {
    // Mirror of the desc's type so consumers (renderer, timeline cache, UI)
    // can branch on type without holding a desc reference.
    SimulationDomainType type = SimulationDomainType::Gas;
    Vec3 bounds_min = Vec3(-2.5f, 0.0f, -2.5f);
    Vec3 bounds_max = Vec3(2.5f, 5.0f, 2.5f);
    int resolution_x = 0;
    int resolution_y = 0;
    int resolution_z = 0;
    float voxel_size = 0.1f;
    uint32_t channels = 0u;
    bool valid = false;
    uint64_t version = 0;
    // MAC staggered storage. For Gas this is the source of truth (advected
    // density/temperature/fuel). For Fluid this is per-step scratch for the
    // pressure projection; the source of truth is `particles`.
    FluidSim::FluidGrid grid;
    std::size_t active_density_cells = 0;
    float max_density = 0.0f;
    // Inclusive active-density cell bounds. Empty when max < min. Used by the
    // dense Vulkan RT bridge to ray-march only the occupied sub-domain.
    int active_density_min[3] = {0, 0, 0};
    int active_density_max[3] = {-1, -1, -1};
    // Gas runtime telemetry. The authored domain.backend is only the requested
    // backend; this reports what the most recent solver step actually executed.
    bool gas_gpu_requested = false;
    bool gas_gpu_active = false;
    bool gas_gpu_partial = false;
    std::string gas_compute_status = "Not stepped";
    SimulationGasStats gas_stats;
    // Fluid-only runtime state. Empty for Gas domains.
    Fluid::FluidParticles particles;
    Fluid::APICSolverStats fluid_stats;
    // Whitewater secondary particles (spray/foam/bubbles). Render-only — never
    // fed back into the pressure solve, so it cannot affect liquid mass.
    Fluid::FoamParticles foam;
    Fluid::FoamStats     foam_stats;
    Vec3 domain_motion_delta = Vec3(0.0f, 0.0f, 0.0f);
};

struct SimulationGridDomainMGLevelBuffers {
    ComputeBufferHandle mask; // >0.5 fluid, 0 air, <-0.5 solid
    ComputeBufferHandle rhs;
    ComputeBufferHandle z;
    ComputeBufferHandle diag;
    int nx = 0;
    int ny = 0;
    int nz = 0;
};

struct SimulationGpuFoamRenderBuffer {
    // Packed float4 spheres: xyz = world-space centre, w = sphere radius.
    // Produced by simulation compute and intended for render-backend interop.
    ComputeBufferHandle spheres;
    std::size_t count = 0;
    std::size_t capacity = 0;
    uint64_t version = 0;
    float radius = 0.0f;

    bool valid() const { return spheres.valid() && count > 0; }
};

// Backend-neutral publication record for Vulkan RT live-volume consumption.
// Addresses are zero until buffers are resident on an address-capable Vulkan
// compute backend. NanoVDB remains the bake/export path.
struct SimulationGasGpuFieldView {
    uint64_t density_address = 0;
    uint64_t temperature_address = 0;
    uint64_t fuel_address = 0;
    uint64_t flame_address = 0;
    // Per-block maximum density (kGasMajorantBlock^3 cells per entry). Lets the
    // RT volume march skip empty blocks; a dense domain has no other
    // empty-space acceleration. Zero when the majorant pass has not run, and
    // the shader must then fall back to marching every step — a stale or
    // missing majorant may never be treated as "empty".
    uint64_t majorant_address = 0;
    // Emitting-block list ([0] = count, [1..] = block indices) for volume
    // emission NEE. Shares the majorant's block layout.
    uint64_t emissive_list_address = 0;
    int emissive_capacity = 0;
    int majorant_dim_x = 0;
    int majorant_dim_y = 0;
    int majorant_dim_z = 0;
    int majorant_block = 0;
    int resolution_x = 0;
    int resolution_y = 0;
    int resolution_z = 0;
    Vec3 origin = Vec3(0.0f);
    float voxel_size = 0.0f;
    uint64_t version = 0;

    bool valid() const {
        return density_address != 0 &&
               resolution_x > 0 && resolution_y > 0 && resolution_z > 0;
    }
};

// Cells per axis in one majorant block. Must match MAJORANT_BLOCK in
// shaders/sim_gas_majorant.comp and the DDA in volume_closesthit.rchit.
inline constexpr int kGasMajorantBlock = 8;
// Emitting blocks the volume march can sample. Bounded because the list is
// walked per scatter sample; overflow keeps the fire lit from a subset
// rather than dropping it.
inline constexpr int kGasEmissiveListCapacity = 4096;

struct SimulationGridDomainComputeBuffers {
    ComputeBufferHandle vel_x;
    ComputeBufferHandle vel_y;
    ComputeBufferHandle vel_z;
    ComputeBufferHandle density;
    ComputeBufferHandle temperature;
    ComputeBufferHandle fuel;
    ComputeBufferHandle interaction;
    ComputeBufferHandle pressure;
    ComputeBufferHandle divergence;
    // Per-block density maximum consumed by the RT volume march. Sized from the
    // block resolution, which is tracked separately from the cell resolution so
    // a domain resize reallocates it instead of publishing a majorant that
    // describes the old grid.
    ComputeBufferHandle gas_majorant;
    int gas_majorant_dim[3] = {0, 0, 0};
    bool gas_majorant_valid = false;
    // Compact list of emitting blocks: [0] = count, [1..] = block indices.
    // Lets the RT volume march sample the fire directly instead of waiting for
    // a random bounce to land in it. Device-resident end to end — the host
    // never reads it, so producing it costs no sync.
    ComputeBufferHandle gas_emissive_list;
    bool gas_emissive_valid = false;
    ComputeBufferHandle scratch_vel_x;
    ComputeBufferHandle scratch_vel_y;
    ComputeBufferHandle scratch_vel_z;
    ComputeBufferHandle scratch_scalar;
    // MacCormack second-pass targets. The correction kernels read the original
    // field AND the forward-advected field while writing a third, so the result
    // cannot alias either input.
    ComputeBufferHandle scratch_scalar2;
    ComputeBufferHandle scratch2_vel_x;
    ComputeBufferHandle scratch2_vel_y;
    ComputeBufferHandle scratch2_vel_z;
    ComputeBufferHandle fluid_positions;
    ComputeBufferHandle fluid_velocities;
    ComputeBufferHandle fluid_affine;
    // GPU mirror of FluidParticles::mass_fraction. Kept separate so existing
    // APIC kernels retain their buffer ABI while lifecycle support is added.
    ComputeBufferHandle fluid_mass_fraction;
    ComputeBufferHandle foam_positions;
    SimulationGpuFoamRenderBuffer foam_render;
    // Whitewater spawn-potential pass (GPU port of stepFoam's phase 3, which
    // measured 66.5% of the whole CPU foam cost and scales with the FLUID
    // particle count). foam_bin_counts holds one occupancy counter per cell plus
    // a trailing OVERFLOW tally; foam_bin_items is the fixed-capacity bucket grid
    // that replaces the host's serial CSR (no prefix scan, no extra sync).
    ComputeBufferHandle foam_bin_counts;
    ComputeBufferHandle foam_bin_items;
    ComputeBufferHandle foam_expected;
    int foam_bin_cell_capacity = 0;   // allocated max_per_cell
    int foam_bin_cell_count = 0;      // allocated cell count
    std::size_t foam_expected_capacity = 0;  // particles the expected buffer holds
    // One-frame pipeline: a dispatch is in flight and its result is read at the
    // START of the next step, by which time the GPU has long finished it. Reading
    // it in the same step cost 17.29 ms of pure waiting — the kernel itself is
    // ~1-2 ms of work and the transfer ~0.1 ms.
    bool foam_expected_pending = false;
    std::size_t foam_expected_pending_count = 0;
    // Foam-side classification gather (sim_foam_neigh). Once the criterion moved to
    // the device this became the dominant remaining cost — 11.35 ms of 14.98 ms —
    // and it reuses the bucket grid above, so it only adds a foam-position upload.
    // Dispatched at the END of a step over the final foam array and consumed at the
    // start of the next, which is the only ordering where the indices still line up:
    // nothing mutates foam in between. The generation stamp catches a cache scrub or
    // reset landing in that gap, which a length check alone would miss.
    ComputeBufferHandle foam_neigh;
    std::size_t foam_positions_capacity = 0;
    std::size_t foam_neigh_capacity = 0;
    bool foam_neigh_pending = false;
    std::size_t foam_neigh_pending_count = 0;
    uint64_t foam_neigh_pending_generation = 0;
    // Set by runGpuFoamCriteria once this step's bins are built. The neighbour pass
    // reads those bins, so without a fresh build it must not run at all (stale bins
    // would silently classify against last step's liquid, or a resized grid).
    bool foam_bin_ready_this_step = false;
    // Float buffer (0.0f = air, 1.0f = fluid cell). Rebuilt from particle
    // positions every step before GPU pressure projection.
    ComputeBufferHandle fluid_mask;
    // APIC Wind surface-drag: one ordered-float height per XZ grid column.
    ComputeBufferHandle fluid_surface_columns;
    // Two float planes per fluid cell: thermally accumulated surface
    // temperature followed by remaining combustible fuel. The fuel plane uses
    // -1 as a lazy initialization marker so material edits/reset/resize are
    // deterministic without a CPU readback.
    ComputeBufferHandle fluid_combustion_state;
    bool fluid_combustion_state_needs_reset = true;
    // Gas collider coupling (all GPU backends): cell-centred solid occupancy
    // (0=open, 1=solid) plus moving-wall linear velocity components.
    ComputeBufferHandle gas_solid_mask;
    ComputeBufferHandle gas_solid_vel_x;
    ComputeBufferHandle gas_solid_vel_y;
    ComputeBufferHandle gas_solid_vel_z;
    ComputeBufferHandle gas_source_density;
    ComputeBufferHandle gas_source_temperature;
    ComputeBufferHandle gas_source_fuel;
    ComputeBufferHandle gas_source_flame;
    ComputeBufferHandle gas_source_band;
    // Material State Field scatter accumulators, one uint per cell. Fixed-point
    // because GL_EXT_shader_atomic_float is optional and Vulkan is the primary
    // backend — hardware independence outranks the convenience. Cleared by the
    // resolve pass itself, so they need no separate zero-fill dispatch.
    ComputeBufferHandle msf_accum_fuel;
    ComputeBufferHandle msf_accum_density;
    ComputeBufferHandle msf_accum_heat;
    ComputeBufferHandle msf_accum_flame;
    // GPU MGPCG (Layer A: Jacobi-preconditioned CG) scratch. Solves the same
    // free-surface Poisson system as the SOR path; allocated lazily alongside
    // the cell buffers. cg_partials holds per-block double partial sums for the
    // dot-product reductions (downloaded + summed on host for stability).
    ComputeBufferHandle cg_residual;   // r
    ComputeBufferHandle cg_z;          // z = M^-1 r
    ComputeBufferHandle cg_search;     // s (search direction)
    ComputeBufferHandle cg_As;         // As = A*s
    ComputeBufferHandle cg_diag;       // diagonal (in-bounds neighbour count)
    ComputeBufferHandle cg_partials;   // double[] block partial sums
    // Device-resident CG scalars (Vulkan fast path): [0]=sigma [1]=sigma0
    // [2]=sAs [3]=alpha [4]=beta [5]=sigma_new [6]=degenerate flag. Keeping
    // alpha/beta on the GPU collapses the per-dot submit+fence round-trips
    // (the dominant Vulkan MGPCG cost) into one tiny download every K iters.
    ComputeBufferHandle cg_scalars;    // double[7]
    // Variational solid coupling (GPU Stage 1): MAC-face fractional open weights
    // (uint8_t->float conversion happens on upload) and per-cell solid velocity.
    ComputeBufferHandle var_u_weight;   // float[(nx+1)*ny*nz]
    ComputeBufferHandle var_v_weight;   // float[nx*(ny+1)*nz]
    ComputeBufferHandle var_w_weight;   // float[nx*ny*(nz+1)]
    ComputeBufferHandle var_svx;        // float[nx*ny*nz]
    ComputeBufferHandle var_svy;        // float[nx*ny*nz]
    ComputeBufferHandle var_svz;        // float[nx*ny*nz]
    ComputeBufferHandle var_fluid_phi;  // float[nx*ny*nz] (GFM level-set narrow-band)
    // MGPCG Layer B: geometric multigrid V-cycle coarse levels used as the
    // pressure preconditioner on CUDA. Empty or partially invalid => Layer A
    // Jacobi preconditioner remains the fallback.
    std::vector<SimulationGridDomainMGLevelBuffers> mg_levels;
    std::size_t fluid_particle_capacity = 0;
    int resolution_x = 0;
    int resolution_y = 0;
    int resolution_z = 0;
    ComputeBackendType backend = ComputeBackendType::CPU;
    // GPU-resident simulation fields flag. When true, GPU compute kernels
    // operate directly between persistent VRAM buffers without host round-trips.
    bool gpu_resident_fields_valid = false;
};

struct SimulationFlowSourceDesc {
    struct Keyframe {
        bool has_enabled = false;
        bool has_position = false;
        bool has_velocity = false;
        bool has_radius = false;
        bool has_density = false;
        bool has_temperature = false;
        bool has_fuel = false;
        bool has_falloff = false;
        bool has_velocity_coupling = false;
        // Liquid emission rate. Keying this is what animates a hose valve —
        // open at one frame, throttled at another — which is the whole point of
        // a keyable flow source on a Fluid domain.
        bool has_flow_rate = false;
        bool enabled = true;
        Vec3 position = Vec3(0.0f);
        Vec3 velocity = Vec3(0.0f);
        float radius = 0.35f;
        float density = 1.0f;
        float temperature = 0.0f;
        float fuel = 0.0f;
        float falloff = 1.0f;
        float velocity_coupling = 8.0f;
        float flow_rate = 1000.0f;
    };
    std::string name = "Flow Source";
    uint64_t timeline_uid = 0;
    SimulationFlowSourceMode source_mode = SimulationFlowSourceMode::Point;
    std::string source_name;
    int domain_index = 0;
    bool enabled = true;

    // ── Object binding (parenting) ───────────────────────────────────────────
    // Deliberately ORTHOGONAL to source_mode: a Point source and a MeshSurface
    // source can both be parented. When `parent_object` names a scene node the
    // source rides that node's transform every step, so it can be carried by a
    // keyframed prop OR by a Jolt rigid body with no extra authoring — a lit
    // match that is thrown and falls under physics keeps its flame on the tip.
    //
    // ★ Parenting REINTERPRETS `position` and `velocity` as parent-LOCAL rather
    // than adding a second offset field. One field, one meaning, and the
    // existing keyframe channels (Keyframe::position / ::velocity) keep working
    // untouched — a keyed local offset animates the flame along the object.
    // The consequence, which is the whole point: a parented source follows its
    // object instead of staying nailed to the world coordinates it was authored
    // at. The UI converts world -> local once at the moment of parenting so the
    // source does not visibly jump.
    std::string parent_object;
    SimulationEmissionVelocitySpace velocity_space =
        SimulationEmissionVelocitySpace::Local;
    // Fraction of the parent's own velocity added to the emitted medium. Without
    // it a waved match leaves its flame behind and a moving hose pours a
    // vertical wall of liquid instead of an arc. 1 = fully carried.
    float inherit_velocity = 1.0f;

    // Runtime-only motion state — NEVER serialized, reset on rewind. The
    // sentinel x < -9.99e9 means "no previous sample yet" so the first step
    // after load/rewind/enable inherits ZERO instead of a huge bogus velocity
    // measured from the origin.
    Vec3 parent_prev_position = Vec3(-1.0e10f, 0.0f, 0.0f);
    Vec3 parent_velocity = Vec3(0.0f);

    // World space, or parent-local when `parent_object` is set (see above).
    Vec3 position = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 velocity = Vec3(0.0f, 1.0f, 0.0f);
    // Per-second relaxation toward `velocity`. The source is a boundary/inflow
    // target, not a force; additive velocity injection accumulates energy until
    // the gas becomes numerically unstable.
    float velocity_coupling = 8.0f;
    float radius = 0.35f;
    // Gas (target domain type == Gas): per-second injection amounts into the
    // density/temperature/fuel channels.
    float density = 1.0f;
    float temperature = 0.0f;
    float fuel = 0.0f;
    float falloff = 1.0f;
    // Fluid (target domain type == Fluid): continuous particle spawn rate.
    // The spawn volume is taken from source_mode (Point → sphere around
    // position; ObjectBounds → resolved AABB; MeshSurface → sampled points
    // on the source mesh). Initial particle velocity = `velocity` above.
    float fluid_particles_per_second = 1000.0f;
    // Emission velocity spread (fluid only). All particles otherwise inherit the
    // single `velocity` above verbatim, so an APIC liquid — which has no surface
    // tension or turbulence to break a laminar stream apart — keeps the emitted
    // mass in perfect formation: it falls as a coherent slab/sheet and only
    // scatters once it slams into a collider. This fraction adds a per-particle
    // random velocity perturbation of magnitude (spread * |velocity|), breaking
    // that symmetry at the source so the stream looks like flowing water rather
    // than a falling plate. 0 = laminar (old behaviour), ~0.1-0.3 = natural.
    float fluid_velocity_spread = 0.15f;
    // MeshSurface mode only: when true the emission velocity is redirected
    // along each spawn point's surface normal (magnitude = |velocity|), so the
    // liquid sprays outward off the geometry like a hose/fountain following the
    // shape. When false every particle uses the single `velocity` vector above.
    bool  fluid_emit_along_normal = false;
    // Per-source accumulator for fractional emit counts (kept in the desc so
    // it survives step boundaries; reset on disable).
    float fluid_emit_accumulator = 0.0f;
    // Dynamic emission limits (Houdini/Blender style flow controls)
    bool use_time_limit = false;
    float start_time = 0.0f;
    float end_time = 5.0f;
    bool use_particle_limit = false;
    int max_emitted_particles = 100000;
    int total_emitted_particles = 0;
    std::map<int, Keyframe> keyframes;
};

// Evaluates the independently keyed flow-source channels at a timeline frame.
// Shared by the solver and authoring UI so displayed values match simulation.
SimulationFlowSourceDesc::Keyframe evaluateSimulationFlowSource(
    const SimulationFlowSourceDesc& source, int frame);

// A flow source's authored channels after keyframe evaluation AND object
// parenting have BOTH been applied — i.e. exactly what the solver injects.
// The viewport gizmo and the panel readout resolve through the same function,
// so a parented source is drawn where it actually emits instead of at the local
// offset it was authored with.
struct SimulationFlowSourceFrame {
    SimulationFlowSourceDesc::Keyframe keyed;
    Vec3 position = Vec3(0.0f);   // world space
    Vec3 velocity = Vec3(0.0f);   // world space, inherited motion folded in
    bool parented = false;        // parent was named AND resolved this frame
    // A parent was named but could not be resolved (renamed/deleted object).
    // The solver skips such a source entirely rather than interpreting its
    // parent-local offset as world coordinates — that would drop a flame at
    // roughly the world origin, a ghost with no visible cause.
    bool parent_missing = false;
};

struct ParticleSurfaceSample {
    Vec3 position = Vec3(0.0f);
    Vec3 normal = Vec3(0.0f, 1.0f, 0.0f);
};

// Sentinel emitter index: "no owning emitter" (scripted//API spawns, or an
// emitter that has since been removed). Such particles use the system rates.
inline constexpr uint16_t kNoEmitterIndex = 0xFFFFu;

struct ParticleSpawnDesc {
    uint16_t emitter_index = kNoEmitterIndex;
    Vec3 position = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 velocity = Vec3(0.0f, 0.0f, 0.0f);
    float lifetime_seconds = 5.0f;
    float mass = 1.0f;
    // Visual attributes evolve linearly from start (birth) to end (death) across
    // the particle's lifetime. Renderers read the current values from the SoA.
    float start_size = 0.05f;
    float end_size = 0.05f;
    float start_opacity = 1.0f;
    float end_opacity = 0.0f;
    Vec3 start_color = Vec3(1.0f, 1.0f, 1.0f);
    Vec3 end_color = Vec3(1.0f, 1.0f, 1.0f);
    float rotation = 0.0f;          // initial angle (radians)
    float angular_velocity = 0.0f;  // spin (radians/sec)
};

struct ParticleEmitterDesc {
    // Independently-keyable channels, mirroring SimulationFlowSourceDesc so the
    // two emission families behave the same on the timeline. Before this, a
    // particle emitter could only be switched on and off by hand: there was no
    // way to say "the embers start at the frame the match catches".
    struct Keyframe {
        bool has_enabled = false;
        bool has_rate = false;
        bool has_speed = false;
        bool has_spread = false;
        bool has_point = false;
        bool has_direction = false;
        bool enabled = true;
        float rate_per_second = 32.0f;
        float speed = 2.0f;
        float spread = 0.35f;
        Vec3 point = Vec3(0.0f, 1.0f, 0.0f);
        Vec3 direction = Vec3(0.0f, 1.0f, 0.0f);
    };

    std::string name = "Particle Emitter";
    uint64_t timeline_uid = 0;
    std::map<int, Keyframe> keyframes;

    // ── Object binding (parenting) ───────────────────────────────────────────
    // Same contract as SimulationFlowSourceDesc: `point` and `direction` become
    // parent-LOCAL while parented, and the emitter rides the object's full
    // transform (rotation included) rather than the AABB centre that
    // ParticleEmitterSourceMode::ObjectOrigin resolves. That AABB path cannot
    // put sparks on a match TIP or turn them as the match turns.
    std::string parent_object;
    SimulationEmissionVelocitySpace velocity_space =
        SimulationEmissionVelocitySpace::Local;
    float inherit_velocity = 1.0f;
    // Runtime-only motion state, never serialized. Sentinel x < -9.99e9.
    Vec3 parent_prev_position = Vec3(-1.0e10f, 0.0f, 0.0f);
    Vec3 parent_velocity = Vec3(0.0f);

    ParticleEmitterSourceMode source_mode = ParticleEmitterSourceMode::Point;
    ParticleEmitterSpawnMode spawn_mode = ParticleEmitterSpawnMode::Center;
    std::string source_name;
    Vec3 point = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 local_offset = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 direction = Vec3(0.0f, 1.0f, 0.0f);
    float surface_offset = 0.02f;
    float rate_per_second = 32.0f;
    // One-shot spawn count. Consumed via `burst_consumed` at runtime so the desc
    // itself survives replay and serialization — zeroing this directly made the
    // Explosion preset fire once and then stay dead forever (including on disk).
    int burst_count = 0;
    // Runtime-only: set when the burst has fired, cleared by clear()/rewind so
    // the same explosion replays. Never serialized.
    bool burst_consumed = false;
    float speed = 2.0f;
    float spread = 0.35f;
    float lifetime_seconds = 4.0f;
    float mass = 1.0f;
    // Visual attributes pushed onto spawned particles (over-life start -> end).
    float start_size = 0.06f;
    float end_size = 0.02f;
    float size_jitter = 0.0f;          // +/- random fraction of size at spawn
    float start_opacity = 1.0f;
    float end_opacity = 0.0f;
    Vec3 start_color = Vec3(1.0f, 0.85f, 0.5f);
    Vec3 end_color = Vec3(1.0f, 0.25f, 0.08f);
    float angular_velocity = 0.0f;     // mean spin (radians/sec)
    float angular_jitter = 0.0f;       // +/- random spin added at spawn
    bool enabled = true;
    float accumulator = 0.0f;
    uint32_t seed = 1;

    // ── Per-emitter particle -> gas deposit override ─────────────────────────
    // The system-wide rates in ParticlePhysicsSettings stay the default. When
    // this is on, particles born from THIS emitter use the rates below instead.
    // That is what lets one system carry both igniting embers (fuel > 0) and
    // inert smoke (fuel = 0) — the match scenario needs exactly that.
    bool  override_grid_deposit = false;
    float grid_density_deposit = 0.0f;
    float grid_temperature_deposit = 0.0f;
    float grid_fuel_deposit = 0.0f;
};

// Keyframe evaluation + object parenting for a particle emitter, resolved
// through one function so the viewport gizmo, the panel and the spawner agree.
struct ParticleEmitterFrame {
    ParticleEmitterDesc::Keyframe keyed;
    Vec3 position = Vec3(0.0f);            // world space
    Vec3 direction = Vec3(0.0f, 1.0f, 0.0f); // world space, normalized-ish
    Vec3 inherited_velocity = Vec3(0.0f);  // parent motion * inherit factor
    bool parented = false;
    bool parent_missing = false;           // named a parent that no longer exists
};

ParticleEmitterDesc::Keyframe evaluateParticleEmitter(
    const ParticleEmitterDesc& emitter, int frame);

struct ParticleColliderDesc {
    std::string name = "Particle Collider";
    ParticleColliderSourceMode source_mode = ParticleColliderSourceMode::PlaneY;
    std::string source_name;
    bool enabled = true;
    float plane_y = 0.0f;
    Vec3 sphere_center = Vec3(0.0f, 1.0f, 0.0f);
    float sphere_radius = 1.0f;
    Vec3 capsule_start = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 capsule_end = Vec3(0.0f, 2.0f, 0.0f);
    float capsule_radius = 0.5f;
    Vec3 bounds_min = Vec3(-1.0f, -1.0f, -1.0f);
    Vec3 bounds_max = Vec3(1.0f, 1.0f, 1.0f);
    float restitution = 0.35f;
    float friction = 0.0f;
    float thickness = 0.0f;
    // Optional gas-surface source. Disabled by default so ordinary particle /
    // fluid colliders remain pure boundaries with zero scalar injection.
    bool gas_interaction_enabled = false;
    float gas_density_rate = 0.0f;
    float gas_temperature_rate = 0.0f;
    float gas_fuel_rate = 0.0f;
    float gas_flame_rate = 0.0f;
    float gas_surface_band_voxels = 1.0f;
    bool gas_ignite_on_contact = false;
    // What this object is made of. Resolves through the built-in substance
    // library; every burning/thermal number is DERIVED from real physical
    // constants in Kelvin. See MaterialStateField.h for the unit decision and
    // why AtmosphereParams.temperature is deliberately not the thermal authority.
    std::string msf_substance = "Wood (Oak)";
    // Per-object deviation from that substance. Defaults are a no-op, so an
    // untouched collider behaves exactly as its substance says.
    SubstanceOverride msf_override;
    // UV-space char mask resolution. Elements ARE the texels of this mask, so
    // this sets both the burn-mark detail and the simulation cost. 0 forces the
    // blocky per-triangle fallback.
    int msf_mask_resolution = 128;
    // Whether this object gets a VISIBLE burn mark at all. Off means the mask is
    // never built: no res*res texture, no per-texel elements, and the object
    // falls back to blocky per-triangle thermal elements. It still heats, burns,
    // releases fuel and can ignite its neighbours — it just does not carry a
    // scorch texture. Worth turning off for anything that only needs to
    // PARTICIPATE in the fire (a collider that blocks flow, an object that will
    // be hidden or destroyed anyway), since the mask is the expensive part.
    bool msf_generate_char_mask = true;
    // Opt-in automatic molten-reservoir -> APIC bridge. Disabled by default:
    // enabling it adds one MSF readback per simulated frame for this workflow.
    bool msf_auto_transfer = false;
    std::string msf_transfer_domain;
    float msf_transfer_rate_kg_s = 0.10f;
    float msf_transfer_min_mass_kg = 0.01f;
    float msf_transfer_particles_per_kg = 2048.0f;
    uint32_t msf_transfer_max_batch_particles = 256u;
    Vec3 msf_transfer_velocity = Vec3(0.0f, -0.1f, 0.0f);
    bool msf_melt_flow_enabled = true;
    float msf_melt_height_loss = 0.85f;
    float msf_melt_spread = 1.50f;
    // Keep a mesh-SDF collider close to the visibly melted surface without
    // cooking a 3D field every render frame. mask_revision is advanced by the
    // thermal readback, so this interval remains idle while simulation is idle.
    bool msf_melt_sdf_refresh = true;
    uint32_t msf_melt_sdf_revision_interval = 4u;
    float msf_melt_sdf_change_threshold = 0.025f;

    // Advanced complex object settings
    int sdf_resolution_mode = 1;       // 0: Low (32^3), 1: Med (64^3), 2: High (128^3)
    float decimation_ratio = 1.0f;     // Decimation ratio
    bool draw_wireframe = true;        // Draw wireframe overlay
    bool draw_slice_preview = false;   // Draw 2D voxel slice preview
    float slice_plane_distance = 0.5f; // Deepness distance (0.0 to 1.0)
    int slice_axis = 1;                // 0: X, 1: Y, 2: Z

    // Cached SDF grid data for simulation lookup
    std::shared_ptr<std::vector<float>> sdf_grid_data;
    Vec3 sdf_origin = Vec3(0.0f);
    Vec3 sdf_extents = Vec3(0.0f);
    int sdf_nx = 0;
    int sdf_ny = 0;
    int sdf_nz = 0;
    // Runtime-only bake guards. Copies of a collider intentionally share these
    // tokens: an older detached cook may finish, but can never publish over a
    // newer request. The busy flag coalesces automatic melt refreshes.
    std::shared_ptr<std::atomic<uint64_t>> sdf_bake_serial =
        std::make_shared<std::atomic<uint64_t>>(0u);
    std::shared_ptr<std::atomic<bool>> sdf_bake_busy =
        std::make_shared<std::atomic<bool>>(false);

    // Local mesh cache for ConvexDecomp / BVH
    std::shared_ptr<std::vector<SurfaceMeshTriangle>> local_triangles_cache;
    std::shared_ptr<std::vector<Vec3>> octant_min_cache;
    std::shared_ptr<std::vector<Vec3>> octant_max_cache;
    std::shared_ptr<std::vector<bool>> octant_active_cache;
    // Accelerated nearest-triangle queries for ObjectMeshBVH voxelization. Built
    // alongside local_triangles_cache (same version gate) so the per-step solid
    // stamp is O(cells·log tris) instead of the old linear scan over every
    // triangle, and uses the exact same BVH math as the SDF cook (consistency).
    std::shared_ptr<ColliderMeshBVH> mesh_bvh_cache;
    uint64_t last_mesh_cache_version = 0;
};

struct ParticleColliderOBB {
    Vec3 local_bounds_min = Vec3(-1.0f, -1.0f, -1.0f);
    Vec3 local_bounds_max = Vec3(1.0f, 1.0f, 1.0f);
    Matrix4x4 local_to_world = Matrix4x4::identity();
};

struct ParticleSoABuffers {
    std::vector<float> position_x;
    std::vector<float> position_y;
    std::vector<float> position_z;
    std::vector<float> velocity_x;
    std::vector<float> velocity_y;
    std::vector<float> velocity_z;
    std::vector<float> age_seconds;
    std::vector<float> lifetime_seconds;
    std::vector<float> inverse_mass;
    std::vector<uint8_t> alive;
    // Which emitter spawned this particle, so the particle -> gas deposit can be
    // authored PER EMITTER instead of only per system. Without it a system can
    // only say "all my particles ignite" or "none do" — a lit match cannot have
    // igniting embers and non-igniting smoke at once.
    // kNoEmitterIndex (or a stale index after the emitter is removed) falls back
    // to the system-wide rates, so this never becomes a dangling reference.
    std::vector<uint16_t> emitter_index;

    // Visual attributes — current values written each step, consumed by renderers.
    std::vector<float> size;
    std::vector<float> rotation;          // radians
    std::vector<float> angular_velocity;  // radians/sec
    std::vector<float> color_r;
    std::vector<float> color_g;
    std::vector<float> color_b;
    std::vector<float> opacity;

    // Over-life endpoints captured at spawn (current = lerp(start, end, age/life)).
    std::vector<float> start_size;
    std::vector<float> end_size;
    std::vector<float> start_opacity;
    std::vector<float> end_opacity;
    std::vector<float> start_color_r;
    std::vector<float> start_color_g;
    std::vector<float> start_color_b;
    std::vector<float> end_color_r;
    std::vector<float> end_color_g;
    std::vector<float> end_color_b;
};

struct ParticleComputeBuffers {
    ComputeBufferHandle position_x;
    ComputeBufferHandle position_y;
    ComputeBufferHandle position_z;
    ComputeBufferHandle velocity_x;
    ComputeBufferHandle velocity_y;
    ComputeBufferHandle velocity_z;
    ComputeBufferHandle age_seconds;
    ComputeBufferHandle lifetime_seconds;
    ComputeBufferHandle inverse_mass;
    ComputeBufferHandle alive;
    std::size_t capacity = 0;
    uint64_t source_version = 0;
};

struct ParticleSimulationStats {
    // ── Particle -> gas deposit telemetry ────────────────────────────────────
    // "I set a fuel deposit rate and no fire appeared" had four indistinguishable
    // causes, all of them a silent zero. These make the break point observable;
    // read them in order, like MaterialStateFieldBridgeStats.
    //   landed             — deposits that reached at least one gas domain
    //   dropped_no_domain  — particles flying where no gas box exists
    //   dropped_no_channel — reached a domain that cannot hold Fuel
    uint32_t grid_deposit_landed = 0;
    uint32_t grid_deposit_dropped_no_domain = 0;
    uint32_t grid_deposit_dropped_no_channel = 0;

    float total_ms = 0.0f;
    float emit_ms = 0.0f;
    float integrate_ms = 0.0f;
    float self_collision_ms = 0.0f;
    float grid_domain_ms = 0.0f;
    float upload_ms = 0.0f;
    std::size_t alive_count = 0;
    std::size_t capacity = 0;
    std::size_t emitter_count = 0;
    std::size_t collider_count = 0;
    std::size_t domain_count = 0;
};

struct MoltenMassTransferRequest {
    uint64_t sequence = 0;
    std::string object_key;
    std::string preferred_domain;
    float requested_mass = 0.0f;
    float particles_per_kg = 32.0f;
    Vec3 velocity = Vec3(0.0f);
    uint32_t max_batch_particles = 0u;
    bool configure_domain_chemistry = false;
};

struct MoltenMassTransferStats {
    uint64_t queued = 0;
    uint64_t completed = 0;
    uint64_t deferred_no_domain = 0;
    uint64_t deferred_no_capacity = 0;
    float requested_mass = 0.0f;
    float transferred_mass = 0.0f;
    uint64_t spawned_particles = 0;
    std::string last_object;
    std::string last_domain;
    std::string last_substance;
    float last_temperature_kelvin = 0.0f;
    float last_combustible_fraction = 0.0f;
};

// Runtime-only counters that are required to continue a simulation from a
// timeline snapshot. Particle/grid data alone is not sufficient: restoring a
// cached frame while leaving these counters in their rewind state re-arms burst
// emitters and makes the first uncached frame behave like frame zero.
struct ParticleSimulationRuntimeState {
    std::vector<float> emitter_accumulators;
    std::vector<uint8_t> emitter_burst_consumed;
    std::vector<float> flow_emit_accumulators;
    std::vector<int> flow_total_emitted_particles;
    std::vector<Vec3> previous_collider_centers;
    std::vector<uint8_t> previous_collider_center_valid;
    std::vector<Matrix4x4> previous_collider_transforms;
    std::vector<uint8_t> previous_collider_transform_valid;
    uint32_t emitter_spawn_serial = 1;
};

class ParticleSimulationSystem final : public ISimulationSystem {
public:
    const char* name() const override { return "Particles"; }
    SimulationSystemKind kind() const override { return SimulationSystemKind::Particle; }
    int order() const override { return 300; }
    bool enabled() const override;
    void step(const SimulationContext& context) override;

    void setEnabled(bool enabled);
    void reserve(std::size_t capacity);
    void clear();
    // Overwrite the live discrete-particle SoA with a captured snapshot (timeline
    // frame cache replay). Restores the alive count and forces a GPU re-upload so a
    // cached-frame scrub/loop-back shows the exact particles that were baked, instead
    // of an empty SoA.
    void restoreSoA(const ParticleSoABuffers& src, std::size_t alive_count);
    ParticleSimulationRuntimeState captureRuntimeState() const;
    void restoreRuntimeState(const ParticleSimulationRuntimeState& state);
    void releaseComputeResources(SimulationComputeContext& compute);
    std::size_t spawn(const ParticleSpawnDesc& desc);
    bool kill(std::size_t index);

    std::vector<ParticleEmitterDesc>& emitters();
    const std::vector<ParticleEmitterDesc>& emitters() const;
    ParticleEmitterDesc& addEmitter(const ParticleEmitterDesc& desc);
    bool removeEmitter(std::size_t index);
    void clearEmitters();
    void setEmitterSourceResolver(std::function<bool(const ParticleEmitterDesc&, Vec3&, Vec3&)> resolver);
    void setEmitterBoundsResolver(std::function<bool(const ParticleEmitterDesc&, Vec3&, Vec3&)> resolver);
    void setEmitterSurfaceSampler(std::function<bool(const ParticleEmitterDesc&, uint32_t, ParticleSurfaceSample&)> sampler);

    std::vector<ParticleColliderDesc>& colliders();
    const std::vector<ParticleColliderDesc>& colliders() const;
    ParticleColliderDesc& addCollider(const ParticleColliderDesc& desc);
    bool removeCollider(std::size_t index);
    void clearColliders();
    void setColliderBoundsResolver(std::function<bool(const ParticleColliderDesc&, Vec3&, Vec3&)> resolver);
    void setColliderOBBResolver(std::function<bool(const ParticleColliderDesc&, ParticleColliderOBB&)> resolver);
    void setColliderMeshResolver(std::function<bool(const ParticleColliderDesc&, std::vector<SurfaceMeshTriangle>&, uint64_t&)> resolver);

    // ── Material State Field ─────────────────────────────────────────────────
    // Persistent per-object thermal/pyrolysis surface state — the sole owner of
    // surface burning. Per-collider "Ignite on Contact" is the toggle; there is
    // no separate system-wide enable any more.
    // Stats require a device readback; ask for one only when something is
    // actually going to look at the numbers (stats panel / debug view).
    void requestMaterialStateFieldReadback() { material_state_fields_.requestReadback(); }
    const MaterialStateFieldStats& materialStateFieldStats() const {
        return material_state_fields_.stats();
    }
    void resetMaterialStateFields() { material_state_fields_.resetState(); }
    // Clear one object's accumulated burn/heat damage. Keyed by the collider's
    // source_name, the same key the field is stored under.
    bool clearMaterialStateField(const std::string& object_key) {
        return material_state_fields_.clearField(object_key);
    }
    // Read-only view for the render bridge (mask upload). The bridge needs the
    // host mirror, which only exists after a requested readback.
    const std::unordered_map<std::string, MaterialStateField>& materialStateFields() const {
        return material_state_fields_.fields();
    }
    bool hasMaterialStateFields() const { return !material_state_fields_.fields().empty(); }
    uint64_t queueMoltenMassTransfer(const MoltenMassTransferRequest& request);
    const MoltenMassTransferStats& moltenMassTransferStats() const {
        return molten_mass_transfer_stats_;
    }
    // ── MSF frame cache (Phase 4b) ───────────────────────────────────────────
    // Burn/heat damage is per-object runtime state and does NOT live under a
    // domain (Phase 4: an object outside every domain is still simulated), so it
    // is cached alongside the grid states rather than inside them — the same
    // parallel-map shape rigid/soft/particle snapshots already use.
    std::vector<MaterialStateFieldSnapshot> captureMaterialStateFieldsForCache(
        SimulationComputeContext& compute) {
        return material_state_fields_.captureSnapshot(compute);
    }
    void restoreMaterialStateFields(
        const std::vector<MaterialStateFieldSnapshot>& snapshot,
        SimulationComputeContext& compute) {
        material_state_fields_.restoreSnapshot(snapshot, compute);
    }
    // ── World thermal boundary conditions (Phase 4) ──────────────────────────
    // Ambient temperature, the Kelvin calibration, convection and oxygen. This
    // is what gives an object that is inside NO domain a defined temperature, so
    // a burning log carried out of the smoke box cools toward the room instead of
    // freezing mid-burn.
    WorldThermalState& worldThermal() { return world_thermal_; }
    const WorldThermalState& worldThermal() const { return world_thermal_; }
    // Derived Kelvin <-> normalized mapping. Returned BY VALUE: the authoring
    // values live on WorldThermalState and there must stay exactly one place
    // that can be edited, or the two would drift.
    MaterialTemperatureScale materialTemperatureScale() const {
        return world_thermal_.scale();
    }

    std::vector<SimulationGridDomainDesc>& gridDomains();
    const std::vector<SimulationGridDomainDesc>& gridDomains() const;
    const std::vector<SimulationGridDomainState>& gridDomainStates() const;
    std::vector<SimulationGridDomainState> captureGridDomainStatesForCache(
        SimulationComputeContext& compute) const;
    SimulationGasGpuFieldView gasGpuFieldView(
        std::size_t domain_index,
        const SimulationComputeContext& compute) const;
    const SimulationGpuFoamRenderBuffer* gridDomainFoamRenderBuffer(std::size_t domain_index) const;
    void setGridDomainStates(const std::vector<SimulationGridDomainState>& states); // timeline cache restore
    SimulationGridDomainDesc& addGridDomain(const SimulationGridDomainDesc& desc);
    bool removeGridDomain(
        std::size_t index,
        SimulationComputeContext* compute = nullptr);
    void clearGridDomains();
    void resetGridDomainStates();
    void setGridDomainBoundsResolver(std::function<bool(const SimulationGridDomainDesc&, Vec3&, Vec3&)> resolver);
    /// Run a sync pass immediately (resize state, apply any pending fluid
    /// seeds, etc.) without waiting for the next sim tick. Needed when the
    /// timeline is stopped so UI actions like "Seed Fluid" apply right away.
    void synchronizeGridDomainsNow();
    bool injectGasPressurePulse(const std::string& domain, const Vec3& center,
                                float radius, float peak_pressure_kpa);

    /// @brief Export a Gas grid domain's live fields to an OpenVDB (.vdb) file.
    /// Writes density / temperature / fuel / flame FloatGrids (channel-aware)
    /// with a world-space linear transform. Returns false for an out-of-range
    /// or non-Gas domain, an invalid state, or on I/O error. Without
    /// OPENVDB_ENABLED a raw binary fallback is written instead.
    bool exportGridDomainToVDB(std::size_t domain_index, const std::string& filepath) const;

    std::vector<SimulationFlowSourceDesc>& flowSources();
    const std::vector<SimulationFlowSourceDesc>& flowSources() const;
    SimulationFlowSourceDesc& addFlowSource(const SimulationFlowSourceDesc& desc);
    bool removeFlowSource(std::size_t index);
    void clearFlowSources();
    void setFlowSourceBoundsResolver(std::function<bool(const SimulationFlowSourceDesc&, Vec3&, Vec3&)> resolver);
    void setFlowSourceSurfaceSampler(std::function<bool(const SimulationFlowSourceDesc&, uint32_t, ParticleSurfaceSample&)> sampler);
    /// Node-name -> current world matrix, used to parent a flow source to an
    /// object. Kept separate from the bounds resolver on purpose: parenting
    /// wants the object's TRANSFORM (which physics writes back into), not a
    /// box derived from its geometry.
    void setFlowSourceTransformResolver(std::function<bool(const std::string&, Matrix4x4&)> resolver);
    /// Keyframe evaluation + parenting for a particle emitter. Const, like
    /// resolveFlowSourceFrame — safe to call from UI as often as needed.
    ParticleEmitterFrame resolveParticleEmitterFrame(
        const ParticleEmitterDesc& emitter, int frame) const;
    void advanceEmitterMotion(float dt, int frame);
    /// Keyframe evaluation + object parenting in one place. Const: it never
    /// touches the per-source motion state, so UI/gizmo callers can resolve a
    /// source as often as they like without perturbing inherited velocity.
    SimulationFlowSourceFrame resolveFlowSourceFrame(
        const SimulationFlowSourceDesc& source, int frame) const;
    /// Samples every parented source's world position for this step so
    /// inherited velocity can be differenced. Must run before the injection
    /// filters — see the definition.
    void advanceFlowSourceMotion(float dt, int frame);

    std::size_t capacity() const;
    std::size_t aliveCount() const;
    const ParticleSoABuffers& buffers() const;
    const ParticleComputeBuffers& computeBuffers() const;
    const ParticleSimulationStats& stats() const;

    void setGravity(const Vec3& gravity);
    void setLinearDrag(float drag);
    void setCollisionPlane(float y, bool enabled, float restitution = 0.35f);
    Vec3 gravity() const;
    float linearDrag() const;
    bool collisionPlaneEnabled() const;
    float collisionPlaneY() const;
    float collisionRestitution() const;
    ParticlePhysicsSettings& physicsSettings();
    const ParticlePhysicsSettings& physicsSettings() const;
    void applyPhysicsModePreset(ParticlePhysicsMode mode);
    void applyQualityModePreset(ParticleQualityMode quality);

private:
    struct NeighborGridEntry {
        uint64_t key = 0;
        int x = 0;
        int y = 0;
        int z = 0;
        std::size_t index = 0;
    };

    struct ResolvedCollider {
        ParticleColliderDesc desc;
        ParticleColliderOBB obb;
        bool has_obb = false;
    };

    void resizeStorage(std::size_t capacity);
    std::size_t findDeadSlot() const;
    bool hasActiveEmitters() const;
    bool hasActiveGridSimulation() const;
    void emitFromEmitters(const SimulationContext& context);
    void refreshResolvedColliders(float particle_radius);
    void applyColliders(Vec3& position, Vec3& velocity, const Vec3* previous_position = nullptr) const;
    void synchronizeGridDomains();
    void stepGridDomains(const SimulationContext& context);
    void injectFlowSourcesIntoGridDomains(float dt,
                                          float time_seconds,
                                          int frame,
                                          SimulationComputeContext* compute);
    void buildNeighborGrid(float cell_size);
    void solveSelfCollisions(float dt);
    void uploadToCompute(const SimulationContext& context);
    void ensureComputeBuffer(SimulationComputeContext& compute,
                             ComputeBufferHandle& handle,
                             const char* name,
                             std::size_t size_bytes,
                             ComputeBufferUsage usage);
    bool ensureGridDomainComputeBuffers(SimulationComputeContext& compute,
                                        SimulationGridDomainComputeBuffers& buffers,
                                        const FluidSim::FluidGrid& grid);
    void releaseGridDomainComputeBuffers(SimulationComputeContext& compute,
                                         SimulationGridDomainComputeBuffers& buffers);
    void processMoltenMassTransfers(SimulationComputeContext& compute);
    void queueAutomaticMoltenMassTransfers(float dt);

    // GPU MGPCG (Layer A) correctness self-test. Builds a synthetic free-surface
    // pressure problem on a small grid, solves it with the GPU Jacobi-PCG path,
    // and reports the GPU residual ‖b-Ap‖/‖b‖ plus the gap to a reference CPU CG
    // (same matrix-free operator). Isolated — touches no live simulation state.
    // Invoked once when the RAYTROPHI_MGPCG_SELFTEST env var is set.
    bool validateGpuFluidMGPCG(SimulationComputeContext* compute);

    ParticleSoABuffers buffers_;
    ParticleComputeBuffers compute_buffers_;
    std::vector<NeighborGridEntry> neighbor_grid_;
    std::vector<ParticleEmitterDesc> emitters_;
    std::vector<ParticleColliderDesc> colliders_;
    std::vector<ResolvedCollider> resolved_colliders_;
    // Moving-collider momentum transfer (grid-domain fluid). Per-collider linear
    // velocity = (resolved centre this step - last step) / dt, recomputed once
    // per stepGridDomains and stamped into FluidGrid::solid_vel by voxelization.
    std::vector<Vec3>    collider_velocities_;
    std::vector<Vec3>    collider_angular_velocities_;
    std::vector<Vec3>    prev_collider_centers_;
    std::vector<uint8_t> prev_collider_center_valid_;
    std::vector<Matrix4x4> prev_collider_transforms_;
    std::vector<uint8_t> prev_collider_transform_valid_;
    std::vector<SimulationGridDomainDesc> grid_domains_;
    std::vector<SimulationGridDomainState> grid_domain_states_;
    std::vector<SimulationGridDomainComputeBuffers> grid_domain_compute_buffers_;
    std::vector<SimulationFlowSourceDesc> flow_sources_;
    std::function<bool(const ParticleEmitterDesc&, Vec3&, Vec3&)> emitter_source_resolver_;
    std::function<bool(const ParticleEmitterDesc&, Vec3&, Vec3&)> emitter_bounds_resolver_;
    std::function<bool(const ParticleEmitterDesc&, uint32_t, ParticleSurfaceSample&)> emitter_surface_sampler_;
    std::function<bool(const ParticleColliderDesc&, Vec3&, Vec3&)> collider_bounds_resolver_;
    std::function<bool(const ParticleColliderDesc&, ParticleColliderOBB&)> collider_obb_resolver_;
    std::function<bool(const ParticleColliderDesc&, std::vector<SurfaceMeshTriangle>&, uint64_t&)> collider_mesh_resolver_;
    // Material State Field — persistent per-object surface burn state.
    MaterialStateFieldSystem material_state_fields_;
    std::vector<MoltenMassTransferRequest> molten_mass_transfer_queue_;
    MoltenMassTransferStats molten_mass_transfer_stats_;
    uint64_t next_molten_mass_transfer_sequence_ = 1;
    float automatic_molten_readback_seconds_ = 0.0f;
    WorldThermalState world_thermal_;
    std::function<bool(const SimulationGridDomainDesc&, Vec3&, Vec3&)> grid_domain_bounds_resolver_;
    std::function<bool(const SimulationFlowSourceDesc&, Vec3&, Vec3&)> flow_source_bounds_resolver_;
    std::function<bool(const std::string&, Matrix4x4&)> flow_source_transform_resolver_;
    std::function<bool(const SimulationFlowSourceDesc&, uint32_t, ParticleSurfaceSample&)> flow_source_surface_sampler_;
    bool enabled_ = true;
    bool collision_plane_enabled_ = false;
    float collision_plane_y_ = 0.0f;
    float collision_restitution_ = 0.35f;
    Vec3 gravity_ = Vec3(0.0f, -9.81f, 0.0f);
    float linear_drag_ = 0.0f;
    ParticlePhysicsSettings physics_settings_;
    ParticleSimulationStats stats_;
    std::size_t alive_count_ = 0;
    uint64_t data_version_ = 1;
    uint32_t emitter_spawn_serial_ = 1;
};

} // namespace RayTrophiSim
