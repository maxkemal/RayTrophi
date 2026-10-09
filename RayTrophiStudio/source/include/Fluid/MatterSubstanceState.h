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

void configureBuiltinMatterTransport(std::vector<SubstanceProfile>& profiles);
MatterTransportOwner substanceTransportOwner(uint32_t tag, MatterConstitutiveModel model,
                                            const APICSolverParams& params);
MatterOwnerSummary inspectMatterOwners(const FluidParticles& particles,
                                      const APICSolverParams& params);
MatterOwnerSummary inspectMatterDomainOwners(const FluidParticles& particles,
                                            const SimulationGridDomainDesc& domain);
bool validateMatterTransportEdit(const std::string& substance,
                                const std::vector<SimulationGridDomainDesc>& domains,
                                const std::vector<SimulationGridDomainState>& states,
                                std::string& error);

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
