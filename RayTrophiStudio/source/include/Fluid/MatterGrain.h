#pragma once

#include "../SimulationCompute.h"
#include "../Vec3.h"
#include "MatterConstitutive.h"
#include "FluidParticles.h"
#include <json.hpp>
#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
struct SimulationGridDomainComputeBuffers;
struct ParticleColliderDesc;
struct SurfaceMeshTriangle;
struct SubstanceProfile;
struct SimulationFlowSourceDesc;
class ParticleSimulationSystem;
namespace Fluid {
class FluidParticles;
class MatterGrainMpmContact;
struct APICSolverParams;
struct MatterGrainCommonDriver;

// Opt-in dry DEM candidate. Physical radius is independent of render detail.
struct MatterGrainParams {
    // `enabled` and the material below (restitution, friction, rolling and
    // twisting friction, tangential ratio, packing, wet grains, water capacity,
    // absorption, drying, surface tension, contact angle, real radius) are
    // DERIVED every step from the domain's DEM substance (matterGrainOwnership,
    // applyMatterGrainSubstance). Authored here: radius, stiffness scale,
    // numerics, coupling and sleep (docs/dev/MADDE_UI_TEK_OTORITE.md).
    bool enabled = false;
    float radius_m = 0.025f;
    // Contact stiffness follows the radius (k ~ r keeps the impact overlap
    // fraction): stiffness_n_m = stiffness_scale * kMatterGrainStiffnessPerRadius
    // * radius_m, recomputed whenever either changes. Scale 1 = 2e4 N/m at 25 mm.
    float stiffness_scale = 1.0f;
    float stiffness_n_m = 20000.0f;
    // Coefficient of restitution of a normal impact, 0.01..1 (a measurable
    // material property; sand ~.5, glass beads ~.9). The step derives the
    // damping ratio zeta = -ln e / sqrt(pi^2 + ln^2 e) and gives every
    // contact c = 2 zeta sqrt(k m_eff) with its own effective mass, so a
    // grain-grain and a grain-wall impact rebound alike. Replaces the old
    // per-domain normal damping in N s/m (loaded scenes are converted).
    float restitution = 0.5f;
    float sliding_damping_n_s_m = 4.0f;
    float friction = 0.5f;
    float rolling_friction = 0.02f;
    float twisting_friction = 0.0f;
    // Cundall-Strack tangential spring as a fraction of normal stiffness.
    // 2/7 matches the tangential and normal contact periods of a solid
    // sphere; 0 is the old kinetic-only (viscous Coulomb) sliding.
    float tangential_stiffness_ratio = 2.0f / 7.0f;
    // Substeps per binary collision duration (accuracy, not stability).
    int contact_resolution = 24;
    // Grain mass = substance bulk density / packing_fraction * sphere volume,
    // so a packed bed of grains reproduces the substance's bulk density.
    float packing_fraction = 0.6f;
    // Two-way liquid coupling when fluid carriers share the domain: implicit
    // pairwise drag (Di Felice voidage law) + Archimedes buoyancy, with the
    // equal and opposite impulse returned to the liquid parcels. Off = grains
    // and liquid pass through each other (A/B and diagnosis only).
    bool fluid_coupling = true;
    // Drag viscosity is the liquid's own: substance liquid_kinematic_viscosity
    // x liquid_density of the parcels around each grain (no second field).
    // B5: the liquid's projection sees the grains' volume (pore-fraction face
    // weights, grain velocity on the closed part) and grains take the
    // solver's pressure gradient instead of hydrostatic buoyancy. Needs
    // fluid_coupling. Off = B3 behaviour (pile is drag-only to the liquid).
    bool volume_exclusion = true;
    // B6 wet grains. Water a grain holds lives in the canonical
    // pore_water_mass_kg sidecar (one owner, the ledger counts it once) and
    // is absorbed from liquid parcels in the grain's cells, momentum-exact.
    // Wet grains pull on each other through pendular liquid bridges.
    bool wet_grains = false;
    float water_capacity_fraction = 0.05f;  // of the sphere volume
    float absorption_rate_per_s = 4.0f;     // of the free capacity, when submerged
    float drying_rate_per_s = 0.0f;         // of the held water; leaves the domain
    float surface_tension_n_m = 0.072f;
    float contact_angle_deg = 20.0f;
    // Real grain radius one simulated grain stands for (0 = radius_m). A
    // coarse grain keeps the real Bond number: bridge force x (radius_m / it)^2.
    float represented_grain_radius_m = 0.0f;
    int max_substeps = 512;
    // Sleeping grains (docs/dev/DEM_UYUYAN_TANELER.md): a grain slower than
    // sleep_speed_m_s (translation, and spin x radius) for sleep_time_s stops
    // computing contacts between periodic audits. Sleep also requires balanced
    // residual force/torque; contact/bridge disturbances wake connected grains.
    // Absolute units: the threshold is a property of the motion, not of the
    // solver settings. Never with moving colliders, MPM contact, the shared
    // clock, liquid coupling or a force field on the grain.
    bool sleep = true;
    float sleep_speed_m_s = 0.002f;
    float sleep_time_s = 0.2f;
};

// Host CFL and GLSL history slots share this budget (sim_matter_grain.glsl).
inline constexpr int kMatterGrainContactBudget = 24;
// 17: 2-D dispatch index, 18: cost counters, 19: Verlet neighbour list,
// 20: single-bank in-place contact history in host-assigned blocks,
// 21: sleeping grains (rest counter per block, 128 B push constants),
// 22: sleep audits/force balance, bridge wake and atomic rest-word commit.
inline constexpr uint32_t kMatterGrainShaderRevision = 24;
// Neighbour hash: one table of fixed-capacity buckets, filled only when the
// neighbour lists are rebuilt (docs/dev/DEM_VERLET_LISTESI.md).
inline constexpr uint32_t kMatterGrainBucketCapacity = 16;
inline constexpr uint32_t kMatterGrainBucketTables = 1;
// Per-grain Verlet list: neighbours within 2r + bridge rupture cap + skin,
// rebuilt once any grain has moved skin/2 since the last build. More
// neighbours than the capacity refuse publication (diagnostics bit 8).
inline constexpr uint32_t kMatterGrainListCapacity = 32;
inline constexpr float kMatterGrainListSkinRadii = 0.5f;
// Grains are re-sorted by cell only when the population changed or more than
// this fraction of neighbours in the array are out of cell order. Each re-sort
// re-uploads the whole state (88 MB at 1.3M grains); a settled pile keeps its
// order and stays resident on the device.
inline constexpr float kMatterGrainResortDisorder = 0.02f;
// Collider faces: index width only, not a tuning budget. Contact history keys
// carry the surface patch id (a face index) in 30 bits (PATCH_KEY in
// sim_matter_grain.glsl). Device memory (storage-buffer limit) binds first and
// reports itself; there is no face decimation, so the mesh is used as given.
inline constexpr std::size_t kMatterGrainMaxColliderFaces = std::size_t{1} << 30;

inline constexpr float kMatterGrainStiffnessPerRadius = 8.0e5f;  // N/m per m of radius
bool patchMatterGrainParams(const nlohmann::json& patch, MatterGrainParams& params,
                           std::string& error);
// Copies the DEM substance's grain material (and the liquid's surface tension)
// into the domain's grain solver params; `wet` is the derived wet_grains.
void applyMatterGrainSubstance(MatterGrainParams& params, const SubstanceProfile& grain,
                               const SubstanceProfile* liquid, bool wet);
nlohmann::json matterGrainParamsToJson(const MatterGrainParams& params);
// `dropped_keys` (optional) lists saved grain-material keys that the substance
// owns now; the loader logs them.
MatterGrainParams matterGrainParamsFromJson(const nlohmann::json& json,
                                            std::string* dropped_keys = nullptr);
// Why the grain solver cannot run on this domain as configured; empty = ready.
// The single rule: validation, the domain panel's locks and
// fluid.matter_models grain_readiness all read it. `code` is stable
// (not_matter, backend, boundary, pore_exchange, wet_response,
// thermal_liquid, solid_phase); `message` names the fix.
// Damping ratio of a restitution coefficient (spring-dashpot contact).
float matterGrainDampingRatio(float restitution);

struct MatterGrainBlocker {
    std::string code;
    std::string message;
};
std::vector<MatterGrainBlocker> matterGrainBlockers(const SimulationGridDomainDesc& domain);
bool validateMatterGrainDomain(const SimulationGridDomainDesc& domain,
                              const MatterGrainParams& params, std::string& error);
// Birth mass of one physical grain; the emitter is the only writer.
float matterGrainRestMassKg(const MatterGrainParams& params, uint32_t substance_tag,
                            const SubstanceProfile* domain_substance,
                            MatterConstitutiveModel model, bool legacy_granular);
// Water one grain can hold, kg: water_capacity_fraction of its sphere volume.
float matterGrainWaterCapacityKg(const MatterGrainParams& params);
// Gives a just-emitted grain its birth water, capacity and water energy.
// birth_saturation: the flow source's grain_birth_saturation (0..1 of capacity).
void initMatterGrainBirthWater(FluidParticles& particles, std::size_t index, float birth_saturation,
                               const MatterGrainParams& params);
// Fills missing (<= 0) rest masses of granular carriers with the grain mass.
std::size_t ensureMatterGrainRestMasses(FluidParticles& particles,
                                        const MatterGrainParams& params,
                                        const SubstanceProfile* domain_substance,
                                        bool legacy_granular);

struct MatterGrainStepReport {
    int substeps = 0;
    int dispatches = 0;
    float substep_dt = 0.0f;
    std::string limit; // accuracy | stability | damping | travel
    uint32_t max_contacts = 0;
    // Contacts per grain the stability bound assumed this frame, and whether
    // the frame was re-run at the full budget because more were measured
    // (the re-run starts without contact history).
    uint32_t cfl_contacts = 0;
    bool cfl_budget_retry = false;
    // Cost counters of the last substep, summed over grains: neighbour-list
    // entries read, grain-grain contacts found, contact-history slots read,
    // grains with no contact at all, collider BVH nodes visited.
    uint64_t neighbour_candidates = 0;
    uint64_t grain_pairs = 0;
    uint64_t history_probes = 0;
    uint64_t contactless_grains = 0;
    uint64_t collider_nodes_visited = 0;
    // Neighbour-list builds in this frame (one is forced at the frame start).
    uint32_t list_rebuilds = 0;
    uint32_t sticking_contacts = 0;
    uint32_t contacts = 0;
    bool history_reset = false;
    // allocation | host_state_changed | first_step | "" (history carried over)
    std::string history_reset_reason;
    bool history_remapped = false;  // order changed; grain -> history block map re-uploaded
    // B4 / H1-C7 transfer accounting for this step (grain path only).
    bool state_resident = false;    // bank 0 reused, no state upload
    std::size_t upload_bytes = 0;
    std::size_t download_bytes = 0;
    int transfer_batches = 0;       // host<->device synchronisation points
    // Host wall time of one grain step by stage (ms): owner split + cell
    // order + subset copies, upload preparation (identity remap, residency
    // compare, uploads), dispatch + download (includes waiting for the GPU),
    // validation and publication copies, merge back into the domain.
    float host_order_ms = 0.0f;
    float host_prepare_ms = 0.0f;
    float gpu_wait_ms = 0.0f;
    float host_publish_ms = 0.0f;
    float host_merge_ms = 0.0f;
    // Liquid-field binning, drag/pressure preparation, reaction and water
    // exchange around the grain step (host, every coupled frame).
    float host_coupling_ms = 0.0f;
    // Collider gather + vertex velocities + force-field evaluation.
    float host_motion_ms = 0.0f;
    // Inside prepare / gpu_wait: the state upload batch and the readback batch.
    float upload_ms = 0.0f;
    float download_ms = 0.0f;
    // The whole grain-domain step (all of the above plus the liquid lane): a
    // frame's wall time minus this is spent outside the matter step.
    float host_step_ms = 0.0f;
    // Fraction of adjacent grains out of (cell, identity) order this frame, and
    // whether they were re-sorted (a re-sort forces the full state upload).
    float cell_order_disorder = 0.0f;
    bool cell_resorted = false;
    // Grains held asleep on the last substep (contacts audited, no integration).
    uint32_t sleeping_grains = 0;
    // Moving colliders and force fields seen by this step.
    std::size_t collider_faces = 0;
    float collider_speed_max = 0.0f;       // fastest collider vertex, m/s
    float field_acceleration_max = 0.0f;   // largest force-field acceleration on a grain, m/s^2
    // Grain-substeps whose collider contacts exceeded 8 features / 4 supports;
    // the deepest were kept (a dense tessellated curve, not an error).
    uint32_t collider_manifold_truncated = 0;
    // Liquid parcel speed entering the frame (m/s): the liquid lane's CFL
    // substeps follow the maximum, so a few outliers far above the 99th
    // percentile multiply the pressure solves of the whole frame.
    float liquid_speed_max = 0.0f;
    float liquid_speed_p99 = 0.0f;
    int liquid_substeps = 0;
    std::size_t working_set_bytes = 0;
    // Separate fluid and continuum MPM owners beside the DEM grains.
    std::size_t grains = 0;
    std::size_t liquid_parcels = 0;
    std::size_t mpm_parcels = 0;
    uint64_t mpm_contact_events = 0;
    uint32_t mpm_contact_max_neighbours = 0;
    double mpm_contact_momentum_residual = 0.0;
    double mpm_contact_impulse = 0.0;
    bool common_clock = false;
    bool liquid_reaction_on_gpu = false;
    bool liquid_support_dynamic = false;
    bool coupling_enabled = false;
    std::size_t coupled_grains = 0;
    Vec3 drag_impulse;        // sum over grains of the liquid's drag, N*s
    Vec3 buoyancy_impulse;    // sum over grains, N*s
    Vec3 liquid_reaction;     // impulse returned to the liquid parcels, N*s
    double momentum_residual = 0.0;  // |grain gain + liquid gain|, N*s
    double unmatched_impulse = 0.0;  // reaction with no liquid mass to take it
    float max_drag_coefficient = 0.0f;
    float max_submerged_fraction = 0.0f;
    // B5 volume exclusion
    bool volume_exclusion = false;
    bool pressure_force = false;      // false: hydrostatic buoyancy was used
    std::size_t porous_cells = 0;
    float max_solid_fraction = 0.0f;
    Vec3 pressure_impulse;            // sum over grains, N*s
    // B6 wet grains
    bool wet_grains = false;
    std::size_t wet_grain_count = 0;
    uint32_t liquid_bridges = 0;      // active bridges, last substep
    double grain_water_kg = 0.0;      // after this step
    double absorbed_kg = 0.0;         // liquid -> grains this step
    double evaporated_kg = 0.0;       // grains -> outside this step
    double water_balance_error_kg = 0.0;  // (liquid + grain water) change - (-evaporated)
    float max_grain_saturation = 0.0f;
};

// Per-grain liquid coupling, one frame. Inputs are frozen for the frame
// (staggered with the liquid step); the shader integrates a private liquid
// lump of mass `lump_mass_kg` with each grain so the drag pair is implicit
// and momentum-exact. Outputs come back with the grain publication.
struct MatterGrainCouplingInput {
    Vec3 lump_velocity;          // liquid velocity at the grain, m/s
    float drag_coefficient = 0;  // beta, kg/s (0 = uncoupled)
    Vec3 buoyancy_acceleration;  // m/s^2, opposite to gravity
    float lump_mass_kg = 0;      // liquid mass that answers this grain's drag
};
struct MatterGrainCouplingOutput {
    Vec3 drag_impulse;           // impulse the liquid gave this grain, N*s
};

struct MatterGrainGpuRuntime {
    ComputeBufferHandle positions;
    ComputeBufferHandle velocities;
    ComputeBufferHandle affines;
    ComputeBufferHandle coupling;
    ComputeBufferHandle bucket_counts;
    ComputeBufferHandle bucket_slots;
    ComputeBufferHandle mass;
    ComputeBufferHandle ids;
    ComputeBufferHandle scratch;
    ComputeBufferHandle history;
    // Grain -> history block map, then {owner id, occupied-slot mask} per block.
    ComputeBufferHandle history_blocks;
    ComputeBufferHandle diagnostics;
    ComputeBufferHandle triangles;
    ComputeBufferHandle collider_nodes;
    ComputeBufferHandle collider_patches;
    // Verlet list: per grain {count, kMatterGrainListCapacity indices}, and
    // the positions the list was built from.
    ComputeBufferHandle neighbour_list;
    ComputeBufferHandle build_positions;
    uint64_t collider_fingerprint = 0;
    uint32_t collider_node_count = 0;
    // Float offset of the vertex velocities in `triangles` (0 = static).
    uint32_t collider_velocity_offset = 0;
    bool collider_uploaded = false;
    // Contacts per grain the CFL assumes next frame: last frame's measured
    // maximum + 4, clamped to [12, kMatterGrainContactBudget]. A frame that
    // measures more is re-run at the full budget (stepMatterGrainGpu).
    uint32_t cfl_contact_hint = static_cast<uint32_t>(kMatterGrainContactBudget);
    bool history_fresh = true;
    // The device liquid-coupling rows are known to be all zero (a dry frame's
    // upload landed and nothing coupled has written them since).
    bool coupling_zero = false;
    // Exact bits of the last published sleep/force law and substep duration.
    // An edit or collider change invalidates equilibrium, without changing the
    // contact history or allocating another per-grain bank.
    std::array<uint32_t, 24> sleep_context{};
    uint64_t sleep_collider_fingerprint = 0;
    bool sleep_context_valid = false;
    // Device contact history is valid only for the grain state it was
    // published with. Reset, timeline scrub, cache restore or a script edit
    // present a different state (ids restart at 1 after a reset, so identity
    // alone matches the wrong grains): any surviving grain whose position is
    // not bit-identical to the published one drops all history. A new ORDER
    // of the same grains (cell sort, births, removals) keeps it: every grain
    // keeps its history block and only the grain -> block map is uploaded (B8).
    std::vector<uint64_t> published_ids;
    // Block of each published grain (no FRESH bit), parallel to published_ids.
    std::vector<uint32_t> history_block;
    std::vector<Vec3> published_positions;
    // B4: the rest of the published state, to skip re-uploading it.
    std::vector<Vec3> published_velocities;
    std::vector<AffineC> published_affines;
    std::vector<float> published_masses;
    std::size_t capacity = 0;
    std::size_t triangle_capacity = 0;
    uint32_t buckets = 0;
    // Moving colliders: each collider's flat vertices at the previous frame,
    // keyed by collider, so a vertex velocity is (now - then) / dt for rigid,
    // rotating and skinned colliders alike. Valid only for the next step in
    // time (scrub/reset drop it: a jump is not a velocity).
    std::map<std::string, std::vector<Vec3>> collider_previous_vertices;
    double collider_previous_time = -1.0;
};

// Per-frame motion inputs of one grain step (optional, empty = none).
struct MatterGrainMotion {
    // One velocity per triangle vertex, same order as the triangles (m/s).
    // The shader places each triangle at end - v * (frame time left), so the
    // collider sweeps through the substeps instead of jumping per frame.
    std::vector<Vec3> triangle_velocity;
    // Force-field acceleration per grain (m/s^2), held over the frame.
    std::vector<Vec3> external_acceleration;
};

nlohmann::json matterGrainDiagnostics(const FluidParticles& particles,
                                     const MatterGrainParams& params,
                                     const MatterGrainStepReport* report = nullptr,
                                     const APICSolverParams* owner_params = nullptr);
// Free-standing pile shape from grain centres: radial rings of one diameter
// around the horizontal centroid, ring surface = highest grain top above the
// lowest grain bottom. The repose angle is the least-squares slope of the
// rings whose surface lies between 20% and 80% of the peak (cap and toe
// excluded). Wall-confined or multi-pile layouts are not a repose measurement.
nlohmann::json matterGrainPileProfile(const std::vector<Vec3>& centres, float radius);
// Domain panel: grain material (Matter tab), solver and coupling (Solvers
// tab, with the readiness line) and the last step's report (Measure tab).
// `ownership` is matterGrainOwnership() for this domain (MatterSubstanceState.h).
struct MatterGrainOwnership;
void drawMatterGrainMaterial(const SimulationGridDomainDesc& domain,
                             const MatterGrainOwnership& ownership);
void drawMatterGrainSolver(const SimulationGridDomainDesc& domain,
                           const SimulationGridDomainState* state,
                           const MatterGrainOwnership& ownership);
// Flow source panel: which solver this source's granular substance gets in
// its target domain (DEM grains, or MPM and why). Nothing for other substances.
void drawMatterGrainSourceLine(const ParticleSimulationSystem& system,
                               SimulationFlowSourceDesc& source);
void drawMatterGrainReport(const SimulationGridDomainDesc& domain,
                           const SimulationGridDomainState* state);

void releaseMatterGrainGpu(SimulationComputeContext& compute, MatterGrainGpuRuntime& runtime);

// Steps `grains` -- the grain-owned carriers of one domain, in a stable
// (identity-sorted) order -- inside the closed box [low, high]. The grain
// runtime owns its own device copy of the grain state; liquid parcels of the
// same domain never enter these buffers. Contact history is keyed by index
// and stable identity, so a stable order keeps static friction across frames.
// `coupling` (optional, one entry per grain) adds the liquid drag/buoyancy;
// `drag_out` then receives the impulse each grain took from the liquid.
// All transport remains on device between contact substeps; the grain state
// is published to the host once at frame end.
bool stepMatterGrainGpu(FluidParticles& grains, const Vec3& low, const Vec3& high,
    const MatterGrainParams& params, float dt, const Vec3& gravity,
    std::size_t budget_bytes, SimulationComputeContext& compute,
    MatterGrainGpuRuntime& runtime, const std::vector<SurfaceMeshTriangle>& triangles,
    const std::vector<MatterGrainCouplingInput>* coupling,
    std::vector<MatterGrainCouplingOutput>* drag_out,
    MatterGrainStepReport& report, std::string& error,
    const MatterGrainMotion* motion = nullptr,
    MatterGrainMpmContact* mpm_contact = nullptr,
    const MatterGrainCommonDriver* common_driver = nullptr);

} // namespace Fluid
} // namespace RayTrophiSim
