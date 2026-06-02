#pragma once

#include "Matrix4x4.h"
#include "FluidGrid.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/APICFluidSolver.h"
#include "Fluid/FluidLevelSet.h"
#include "Fluid/FluidFoam.h"
#include "Fluid/FluidRenderMode.h"
#include "VolumeShader.h"
#include "SimulationWorld.h"

#include <memory>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <SurfaceMeshCache.h>

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
    // GPU_Vulkan: forces Vulkan even when CUDA is present (for testing).
    GPU_Vulkan    = 3,
    // Legacy alias kept so old project files load correctly.
    GPU_CUDA      = GPU_Compute,
};

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

enum class SimulationFlowSourceMode {
    Point,
    ObjectBounds,
    MeshSurface
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

struct SimulationGridDomainComputeBuffers {
    ComputeBufferHandle vel_x;
    ComputeBufferHandle vel_y;
    ComputeBufferHandle vel_z;
    ComputeBufferHandle density;
    ComputeBufferHandle temperature;
    ComputeBufferHandle fuel;
    ComputeBufferHandle pressure;
    ComputeBufferHandle divergence;
    ComputeBufferHandle scratch_vel_x;
    ComputeBufferHandle scratch_vel_y;
    ComputeBufferHandle scratch_vel_z;
    ComputeBufferHandle scratch_scalar;
    ComputeBufferHandle fluid_positions;
    ComputeBufferHandle fluid_velocities;
    ComputeBufferHandle fluid_affine;
    ComputeBufferHandle foam_positions;
    SimulationGpuFoamRenderBuffer foam_render;
    // Float buffer (0.0f = air, 1.0f = fluid cell). Rebuilt from particle
    // positions every step before GPU pressure projection.
    ComputeBufferHandle fluid_mask;
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
};

struct SimulationFlowSourceDesc {
    std::string name = "Flow Source";
    SimulationFlowSourceMode source_mode = SimulationFlowSourceMode::Point;
    std::string source_name;
    int domain_index = 0;
    bool enabled = true;
    Vec3 position = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 velocity = Vec3(0.0f, 1.0f, 0.0f);
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
};

struct ParticleSurfaceSample {
    Vec3 position = Vec3(0.0f);
    Vec3 normal = Vec3(0.0f, 1.0f, 0.0f);
};

struct ParticleSpawnDesc {
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
    std::string name = "Particle Emitter";
    ParticleEmitterSourceMode source_mode = ParticleEmitterSourceMode::Point;
    ParticleEmitterSpawnMode spawn_mode = ParticleEmitterSpawnMode::Center;
    std::string source_name;
    Vec3 point = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 local_offset = Vec3(0.0f, 0.0f, 0.0f);
    Vec3 direction = Vec3(0.0f, 1.0f, 0.0f);
    float surface_offset = 0.02f;
    float rate_per_second = 32.0f;
    int burst_count = 0;
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
};

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

    // Local mesh cache for ConvexDecomp / BVH
    std::shared_ptr<std::vector<SurfaceMeshTriangle>> local_triangles_cache;
    std::shared_ptr<std::vector<Vec3>> octant_min_cache;
    std::shared_ptr<std::vector<Vec3>> octant_max_cache;
    std::shared_ptr<std::vector<bool>> octant_active_cache;
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

    std::vector<SimulationGridDomainDesc>& gridDomains();
    const std::vector<SimulationGridDomainDesc>& gridDomains() const;
    const std::vector<SimulationGridDomainState>& gridDomainStates() const;
    const SimulationGpuFoamRenderBuffer* gridDomainFoamRenderBuffer(std::size_t domain_index) const;
    void setGridDomainStates(const std::vector<SimulationGridDomainState>& states); // timeline cache restore
    SimulationGridDomainDesc& addGridDomain(const SimulationGridDomainDesc& desc);
    bool removeGridDomain(std::size_t index);
    void clearGridDomains();
    void resetGridDomainStates();
    void setGridDomainBoundsResolver(std::function<bool(const SimulationGridDomainDesc&, Vec3&, Vec3&)> resolver);
    /// Run a sync pass immediately (resize state, apply any pending fluid
    /// seeds, etc.) without waiting for the next sim tick. Needed when the
    /// timeline is stopped so UI actions like "Seed Fluid" apply right away.
    void synchronizeGridDomainsNow();

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
    void injectFlowSourcesIntoGridDomains(float dt, float time_seconds);
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
    std::vector<Vec3>    prev_collider_centers_;
    std::vector<uint8_t> prev_collider_center_valid_;
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
    std::function<bool(const SimulationGridDomainDesc&, Vec3&, Vec3&)> grid_domain_bounds_resolver_;
    std::function<bool(const SimulationFlowSourceDesc&, Vec3&, Vec3&)> flow_source_bounds_resolver_;
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
