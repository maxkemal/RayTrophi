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
    int max_substeps = 512;
};

// Host CFL and GLSL history slots share this budget (sim_matter_grain.glsl).
inline constexpr int kMatterGrainContactBudget = 24;
inline constexpr uint32_t kMatterGrainShaderRevision = 6;
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
// Fills missing (<= 0) rest masses of granular carriers with the grain mass.
std::size_t ensureMatterGrainRestMasses(FluidParticles& particles,
                                        const MatterGrainParams& params,
                                        FluidChemistryPreset chemistry_preset,
                                        bool legacy_granular);

struct MatterGrainStepReport {
    int substeps = 0;
    int dispatches = 0;
    float substep_dt = 0.0f;
    std::string limit; // accuracy | stability | damping | travel
    uint32_t max_contacts = 0;
    uint32_t sticking_contacts = 0;
    uint32_t contacts = 0;
    bool history_reset = false;
};

struct MatterGrainGpuRuntime {
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

// All canonical transport remains on device between contact microsteps.
// The existing host publication/cache/render bridge is still used at frame end.
bool stepMatterGrainGpu(SimulationGridDomainState& state, const MatterGrainParams& params,
    float dt, const Vec3& gravity, std::size_t budget_bytes,
    SimulationComputeContext& compute, SimulationGridDomainComputeBuffers& buffers,
    const std::vector<SurfaceMeshTriangle>& triangles, std::string& error);

} // namespace Fluid
} // namespace RayTrophiSim
