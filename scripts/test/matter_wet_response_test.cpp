// Regression source for the user's test target; do not compile via Codex.
#include "Fluid/MatterWetResponse.h"
#include "Fluid/MatterPoreAuthoring.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"

#include <cassert>
#include <cmath>
#include <limits>

using namespace RayTrophiSim::Fluid;

int main() {
    assert(matterWetAppearanceBand(0.0f) == 0);
    assert(matterWetAppearanceBand(0.000001f) == 1);
    assert(matterWetAppearanceBand(0.047456805f) == 1);
    assert(matterWetAppearanceBand(0.5f) == 4);
    assert(matterWetAppearanceBand(1.0f) == 7);
    assert(matterWetAppearanceBand(-0.1f) == 0);
    assert(matterWetAppearanceBand(std::numeric_limits<float>::quiet_NaN()) == 0);
    assert(matterWetAppearanceBand(0.0f, 0.05f) == 0);
    assert(matterWetAppearanceBand(0.01f, 0.05f) == 2);
    assert(matterWetAppearanceBand(0.047456805f, 0.05f) == 7);
    assert(matterWetAppearanceBand(1.0f, 0.05f) == 7);
    assert(matterWetAppearanceBand(0.05f, 0.0f) == 0);
    MatterPoreParams params;
    params.wet_response_enabled = true;
    const auto dry = evaluateMatterWetResponse(0.0f, 0.1f, params);
    const auto damp = evaluateMatterWetResponse(0.5f, 0.1f, params);
    const auto saturated = evaluateMatterWetResponse(1.0f, 0.1f, params);
    assert((dry == MatterWetResponse{1.0f, 1.0f, 0.0f, 0.0f}));
    assert(std::abs(damp[0] - 0.8f) < 1e-6f);
    assert(std::abs(damp[1] - 0.625f) < 1e-6f);
    assert(damp[2] == params.capillary_cohesion_pa);
    assert(saturated[2] == 0.0f);
    assert(saturated[0] < damp[0]);
    params.wet_appearance_full_saturation = 1.0f;
    assert(evaluateMatterWetResponse(0.5f, 0.1f, params) == damp);
    params.wet_appearance_full_saturation = 0.05f;
    assert(std::abs(saturated[3] - damp[3] * 4.0f) < 1e-3f);
    assert(evaluateMatterWetResponse(1.0f, 0.2f, params)[3] > saturated[3]);
    params.wet_response_enabled = false;
    assert(evaluateMatterWetResponse(1.0f, 0.1f, params) == dry);
    params.wet_response_enabled = true;

    FluidParticles particles;
    particles.emit(Vec3(0.0f), Vec3(0.0f), 293.15f, 0.0f, substanceTag("Sand"),
        nullptr, nullptr, 0.2f, MatterConstitutiveModel::Granular);
    particles.emit(Vec3(1.0f), Vec3(0.0f), 293.15f, 0.0f, substanceTag("Water"),
        nullptr, nullptr, 0.125f, MatterConstitutiveModel::Fluid);
    particles.pore_capacity_kg[0] = 0.1f;
    particles.pore_water_mass_kg[0] = 0.05f;
    const float volume = matterParticleDryVolume(particles, 0);
    assert(std::abs(volume - 0.000125f) < 1e-8f);
    particles.pore_water_mass_kg[0] = 0.09f;
    assert(matterParticleDryVolume(particles, 0) == volume);
    particles.pore_water_mass_kg[0] = 0.05f;
    const auto ids = particles.particle_id;
    const auto masses = particles.rest_mass_kg;
    std::vector<MatterWetResponse> responses;
    std::string error;
    assert(buildMatterWetResponses(particles, false, 0.1f, params, responses, error));
    assert(responses[0] == damp && responses[1] == dry);
    assert(particles.particle_id == ids && particles.rest_mass_kg == masses);
    particles.constitutive_model[0] = static_cast<uint8_t>(MatterConstitutiveModel::Auto);
    assert(buildMatterWetResponses(particles, true, 0.1f, params, responses, error));
    assert(responses[0] == damp);
    assert(buildMatterWetResponses(particles, false, 0.1f, params, responses, error));
    assert(responses[0] == dry);
    particles.constitutive_model[0] = static_cast<uint8_t>(MatterConstitutiveModel::Granular);
    particles.pore_water_mass_kg[0] = 0.2f;
    assert(!buildMatterWetResponses(particles, false, 0.1f, params, responses, error));
    particles.pore_water_mass_kg[0] = std::numeric_limits<float>::quiet_NaN();
    assert(!buildMatterWetResponses(particles, false, 0.1f, params, responses, error));

    const auto original = matterPoreParamsToJson(params);
    assert(!patchMatterPoreParams({{"wet_friction_scale", -1.0f}}, params, error));
    assert(matterPoreParamsToJson(params) == original);
    assert(!patchMatterPoreParams({{"wet_response_enabled", 1}}, params, error));
    assert(!patchMatterPoreParams({{"wet_typo", 0}}, params, error));
    assert(!patchMatterPoreParams({{"wet_color_scale", 0.4f},
        {"wet_appearance_full_saturation", 0.0f}}, params, error));
    assert(!patchMatterPoreParams({{"wet_appearance_full_saturation", "0.05"}}, params, error));
    assert(matterPoreParamsToJson(params) == original);
    assert(matterPoreParamsToJson(matterPoreParamsFromJson(original)) == original);
    const auto legacy = matterPoreParamsFromJson({{"enabled", true}});
    assert(!legacy.wet_response_enabled && !legacy.wet_appearance_enabled);
    assert(legacy.wet_appearance_full_saturation == 0.05f);
    auto sensitivity = params;
    sensitivity.wet_appearance_full_saturation = 0.1f;
    assert(matterPoreSettingsHash(sensitivity) != matterPoreSettingsHash(params));
    auto changed = params;
    changed.wet_color_scale = 0.7f;
    assert(matterPoreSettingsHash(changed) != matterPoreSettingsHash(params));
}
