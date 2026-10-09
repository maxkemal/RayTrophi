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
constexpr uint64_t kMatterSubstanceStateRevision = 8u;

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
const SubstanceProfile* particleSubstance(uint32_t tag, const APICSolverParams& params);
MatterConstitutiveModel substanceBirthModel(uint32_t tag, const APICSolverParams& params,
                                          float kelvin);
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
float substanceMeltReleaseKelvin(uint32_t tag, const APICSolverParams& params);
float substanceThermalViscosity(uint32_t tag, float kelvin,
                               const APICSolverParams& params);

} // namespace Fluid
} // namespace RayTrophiSim
