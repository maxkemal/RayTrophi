#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace RayTrophiSim {
struct SubstanceProfile;
}

namespace RayTrophiSim::Fluid {
class FluidParticles;
struct MatterPoreParams;

// Shader ABI: friction multiplier, dilatancy multiplier, additive cohesion Pa,
// local positive pore pressure Pa. Zero saturation is exactly the dry model.
using MatterWetResponse = std::array<float, 4>;
static_assert(sizeof(MatterWetResponse) == 16);
// Rest skeleton volume m / rho. An untagged parcel IS the domain's Default
// Substance (that is what its rest mass was computed from), so pass it: the
// old 1600 kg/m^3 fallback gave seeded 400 kg/m^3 Snow a quarter of its volume
// and therefore a quarter of its stress force.
float matterParticleDryVolume(const FluidParticles& particles, std::size_t index,
                              const SubstanceProfile* domain_substance = nullptr);
float matterParticleSaturation(const FluidParticles& particles, std::size_t index);
// Upper-edge quantization: exactly dry stays band 0; positive saturation reaches
// a wet band. The error in normalized appearance response is at most 1/7.
int matterWetAppearanceBand(float saturation, float full_wet_saturation = 1.0f);
inline constexpr const char* kMatterWetAppearanceQuantization =
    "normalized_saturation_upper_edge_v2";
MatterWetResponse evaluateMatterWetResponse(float saturation, float voxel_size,
                                           const MatterPoreParams& params);
bool buildMatterWetResponses(const FluidParticles& particles, bool legacy_granular,
                            float voxel_size, const MatterPoreParams& params,
                            std::vector<MatterWetResponse>& responses, std::string& error);
} // namespace RayTrophiSim::Fluid
