#pragma once

#include "MatterConstitutive.h"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>
#include <string>

namespace RayTrophiSim {
struct SubstanceProfile;
struct SimulationGridDomainDesc;
struct SimulationGridDomainState;
struct SimulationFlowSourceDesc;
class ParticleSimulationSystem;
namespace Fluid {
struct APICSolverParams;
class FluidParticles;
constexpr uint64_t kMatterSubstanceStateRevision = 9u;  // 9: fusion latent heat

enum class MatterTransportOwner : uint8_t { Fluid, Grain, Mpm, Obstacle };
struct MatterOwnerSummary {
    std::array<std::size_t, 4> particles{};
    bool ready = true;
    std::string reason;
};

// Whether a Matter domain runs DEM grains. Not an authored switch: it follows
// the substances that can enter the domain (its Default Substance, its bound
// substances and every flow source aimed at it). One that is granular with
// granular_transport=dem asks for grains. The step writes `enabled` into
// grain.enabled; panel and IPC show the same struct, so a substance that asks
// for DEM but runs as MPM always says why.
struct MatterGrainOwnership {
    bool wanted = false;          // a DEM substance is in reach of the domain
    bool enabled = false;         // what the step runs (grain.enabled)
    bool reset_pending = false;   // live particles keep the old owner until reset
    std::string substance;        // the DEM substance whose grain material runs
    // WHERE that substance entered the domain: "flow source '<name>'",
    // "substance binding" or "default substance". A material binding of a
    // DEM substance turns grains on (and locks the backend to Vulkan) with
    // zero particles of it present; without this the lock had no visible cause.
    std::string origin;
    std::vector<std::string> blockers;  // why a wanted DEM runs as MPM instead
    // Liquid in reach (its surface tension feeds the bridges); empty = none.
    std::string liquid;
    // Derived wet grains: the grain substance holds water (capacity > 0) and
    // the domain has liquid or a source pours wet grains. Live grains keep
    // wet on until a reset (they may hold water).
    bool wet_grains = false;
    // Informational: e.g. a second DEM substance whose material is not used
    // (one grain material per domain).
    std::vector<std::string> notes;
};
MatterGrainOwnership matterGrainOwnership(const SimulationGridDomainDesc& domain,
                                          const std::vector<SimulationFlowSourceDesc>& sources,
                                          int domain_index, bool has_particles);
// The same for domain `index` of a running system (panel and IPC read this).
MatterGrainOwnership matterGrainOwnership(const ParticleSimulationSystem& system,
                                          std::size_t index);
// Grains run in `domain` or a substance there asks for them. Authoring guards
// use it, so a domain is not moved off what grains need before the first step.
bool matterGrainsInUse(const ParticleSimulationSystem& system,
                       const SimulationGridDomainDesc& domain);

void configureBuiltinMatterTransport(std::vector<SubstanceProfile>& profiles);
MatterTransportOwner substanceTransportOwner(uint32_t tag, MatterConstitutiveModel model,
                                            const APICSolverParams& params);
MatterOwnerSummary inspectMatterOwners(const FluidParticles& particles,
                                      const APICSolverParams& params);
MatterOwnerSummary inspectMatterDomainOwners(const FluidParticles& particles,
                                            const SimulationGridDomainDesc& domain);
// `key` names the birth-fixed field being edited (for the message).
bool validateMatterTransportEdit(const std::string& substance,
                                const std::vector<SimulationGridDomainDesc>& domains,
                                const std::vector<SimulationGridDomainState>& states,
                                std::string& error,
                                const char* key = "granular_transport");

// One substance lookup policy for birth, freezing and viscosity. Tag zero
// means the domain default; an unknown nonzero tag retains that same fallback.
// No substance at all (a domain whose Default Substance is missing) has no
// freeze point: substanceFreezeKelvin is -inf, never a domain-wide copy.
const SubstanceProfile* particleSubstance(uint32_t tag, const APICSolverParams& params);
// A meltable substance whose solid is a mechanical skeleton (granular or
// elastic: Snow, Ice). Its solid and liquid are two constitutive models of one
// parcel; a meltable liquid (Water, Wax) freezes in place instead.
bool substanceHasSolidSkeleton(const SubstanceProfile* profile);
// A parcel is in the solid state while at least this share of it is solid
// (fusion_solid_fraction). The latent heat of half the parcel separates the
// two switch points, so the state does not flicker at the melt point.
constexpr float kFusionSolidThreshold = 0.5f;
MatterConstitutiveModel substanceBirthModel(uint32_t tag, const APICSolverParams& params,
                                          float kelvin);
// A source that does not set its own temperature pours the substance as
// itself: the domain ambient, except that a meltable granular or elastic
// substance (Snow, Ice) is born kSolidBirthBelowMeltKelvin under its melt point
// when the ambient is warmer. Snow poured into a 20 C room is snow that melts
// over time, not water. A liquid is likewise never born above its boiling
// point (born at Tb, then boils as heat arrives). `lowered` reports whether
// the ambient was overridden.
constexpr float kSolidBirthBelowMeltKelvin = 10.0f;
float substanceAmbientBirthKelvin(uint32_t tag, const APICSolverParams& params,
                                  float ambient_kelvin, bool* lowered = nullptr);
void initializeMatterBirthModels(FluidParticles& particles, std::size_t begin,
                                const APICSolverParams& params);
void refreshMatterConstitutiveModels(FluidParticles& particles,
                                    const APICSolverParams& params);

// Static block bindings and supported thermal freezing are distinct producers
// of the same solver obstacle. Dead parcels never produce an obstacle.
bool isMatterObstacle(const FluidParticles& particles, std::size_t index,
                      const uint32_t* static_tags, std::size_t tag_count,
                      bool include_frozen = true);
bool buildMatterObstacleMask(const FluidParticles& particles,
                            const std::vector<uint32_t>* static_tags,
                            std::vector<uint8_t>& mask, bool& any_frozen);
void collectMatterStaticTags(const SimulationGridDomainDesc& domain,
                            std::vector<uint32_t>& tags);

float substanceFreezeKelvin(uint32_t tag, const APICSolverParams& params);
float substanceThermalViscosity(uint32_t tag, float kelvin,
                               const APICSolverParams& params);

} // namespace Fluid
} // namespace RayTrophiSim
