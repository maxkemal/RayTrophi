#pragma once

#include "../SimulationCompute.h"
#include "../Vec3.h"
#include "MatterConstitutive.h"
#include <json.hpp>
#include <cstddef>
#include <string>
#include <vector>

namespace RayTrophiSim {
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
struct SimulationGridDomainComputeBuffers;
struct ParticleColliderDesc;
struct SurfaceMeshTriangle;
namespace Fluid {
class FluidParticles;
enum class FluidChemistryPreset : int;

// Opt-in dry DEM candidate. Physical radius is independent of render detail.
struct MatterGrainParams {
    bool enabled = false;
    float radius_m = 0.025f;
    float stiffness_n_m = 20000.0f;
    float normal_damping_n_s_m = 4.0f;
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
    float drag_viscosity_pa_s = 1.0e-3f;
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
    // Water a newly emitted grain already holds, as a fraction of capacity
    // (an emitter of wet sand). Needs wet_grains.
    float birth_saturation = 0.0f;
    int max_substeps = 512;
    // H1-G0 candidate comparison: "dem" (force spring-dashpot, contact
    // history) or "xpbd" (small-steps positional contact, inelastic normal,
    // positional Coulomb friction). Same hash, colliders, coupling, owners.
    std::string solver_kind = "dem";
    int xpbd_substeps = 20;  // per frame minimum; travel bound may add more
};

// Host CFL and GLSL history slots share this budget (sim_matter_grain.glsl).
inline constexpr int kMatterGrainContactBudget = 24;
inline constexpr uint32_t kMatterGrainShaderRevision = 9;
// Neighbour hash: three rotating tables of fixed-capacity buckets.
inline constexpr uint32_t kMatterGrainBucketCapacity = 16;
inline constexpr uint32_t kMatterGrainBucketTables = 3;

bool patchMatterGrainParams(const nlohmann::json& patch, MatterGrainParams& params,
                           std::string& error);
nlohmann::json matterGrainParamsToJson(const MatterGrainParams& params);
MatterGrainParams matterGrainParamsFromJson(const nlohmann::json& json);
bool validateMatterGrainDomain(const SimulationGridDomainDesc& domain,
                              const MatterGrainParams& params, std::string& error);
// Birth mass of one physical grain; the emitter is the only writer.
float matterGrainRestMassKg(const MatterGrainParams& params, uint32_t substance_tag,
                            FluidChemistryPreset chemistry_preset,
                            MatterConstitutiveModel model, bool legacy_granular);
// Water one grain can hold, kg: water_capacity_fraction of its sphere volume.
float matterGrainWaterCapacityKg(const MatterGrainParams& params);
// Gives a just-emitted grain its birth water, capacity and water energy.
void initMatterGrainBirthWater(FluidParticles& particles, std::size_t index,
                               const MatterGrainParams& params);
// Fills missing (<= 0) rest masses of granular carriers with the grain mass.
std::size_t ensureMatterGrainRestMasses(FluidParticles& particles,
                                        const MatterGrainParams& params,
                                        FluidChemistryPreset chemistry_preset,
                                        bool legacy_granular);

struct MatterGrainStepReport {
    int substeps = 0;
    int dispatches = 0;
    float substep_dt = 0.0f;
    std::string limit; // accuracy | stability | damping | travel | xpbd_substeps
    std::string solver_kind = "dem";
    uint32_t max_contacts = 0;
    uint32_t sticking_contacts = 0;
    uint32_t contacts = 0;
    bool history_reset = false;
    // allocation | host_state_changed | first_step | "" (history carried over)
    std::string history_reset_reason;
    std::size_t working_set_bytes = 0;
    // Same-domain liquid (transport owner mpm) beside the grains.
    std::size_t grains = 0;
    std::size_t liquid_parcels = 0;
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
    ComputeBufferHandle history_owner;
    ComputeBufferHandle diagnostics;
    ComputeBufferHandle triangles;
    ComputeBufferHandle collider_nodes;
    ComputeBufferHandle collider_patches;
    uint64_t collider_fingerprint = 0;
    uint32_t collider_node_count = 0;
    bool collider_uploaded = false;
    bool history_fresh = true;
    // Device contact history is valid only for the exact grain state it was
    // published with. Reset, timeline scrub, cache restore, a script edit or
    // a reordering all present a different host state: the springs of that
    // other state must not be applied to it (ids restart at 1 after a reset,
    // so an identity check alone matches the wrong grains).
    uint64_t published_signature = 0;
    std::size_t published_count = 0;
    std::size_t capacity = 0;
    std::size_t triangle_capacity = 0;
    uint32_t buckets = 0;
};

nlohmann::json matterGrainDiagnostics(const FluidParticles& particles,
                                     const MatterGrainParams& params,
                                     const MatterGrainStepReport* report = nullptr);
// Free-standing pile shape from grain centres: radial rings of one diameter
// around the horizontal centroid, ring surface = highest grain top above the
// lowest grain bottom. The repose angle is the least-squares slope of the
// rings whose surface lies between 20% and 80% of the peak (cap and toe
// excluded). Wall-confined or multi-pile layouts are not a repose measurement.
nlohmann::json matterGrainPileProfile(const std::vector<Vec3>& centres, float radius);
void drawMatterGrainControls(const SimulationGridDomainDesc& domain);

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
    MatterGrainStepReport& report, std::string& error);

} // namespace Fluid
} // namespace RayTrophiSim
