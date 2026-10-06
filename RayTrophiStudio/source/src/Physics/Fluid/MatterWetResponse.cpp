#include "Fluid/MatterWetResponse.h"
#include "Fluid/MatterPoreExchange.h"
#include "Fluid/FluidParticles.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim::Fluid {

int matterWetAppearanceBand(float saturation, float full_wet_saturation) {
    if (!std::isfinite(saturation) || saturation <= 0.0f ||
        !std::isfinite(full_wet_saturation) || full_wet_saturation <= 0.0f) {
        return 0;
    }
    const float appearance = std::min(saturation / full_wet_saturation, 1.0f);
    return std::clamp(static_cast<int>(std::ceil(
        appearance * 7.0f)), 1, 7);
}

float matterParticleDryVolume(const FluidParticles& particles, std::size_t index) {
    if (index >= particles.rest_mass_kg.size() || index >= particles.mass_fraction.size()) {
        return -1.0f;
    }
    const auto* profile = index < particles.substance_tag.size()
        ? tryFindSubstanceByTag(particles.substance_tag[index]) : nullptr;
    const float density = profile && std::isfinite(profile->density) && profile->density > 0.0f
        ? profile->density : 1600.0f; // Untagged legacy stress used the Sand density.
    const float mass = particles.rest_mass_kg[index] * particles.mass_fraction[index];
    return std::isfinite(mass) && mass >= 0.0f ? mass / density : -1.0f;
}

float matterParticleSaturation(const FluidParticles& particles, std::size_t index) {
    if (index >= particles.pore_water_mass_kg.size() ||
        index >= particles.pore_capacity_kg.size()) {
        return 0.0f;
    }
    const float capacity = particles.pore_capacity_kg[index];
    const float mass = particles.pore_water_mass_kg[index];
    if (!std::isfinite(capacity) || !std::isfinite(mass) || capacity <= 0.0f) {
        return 0.0f;
    }
    return std::clamp(mass / capacity, 0.0f, 1.0f);
}

MatterWetResponse evaluateMatterWetResponse(float saturation, float voxel_size,
                                           const MatterPoreParams& params) {
    if (!params.wet_response_enabled) {
        return {1.0f, 1.0f, 0.0f, 0.0f};
    }
    const float s = std::isfinite(saturation) ? std::clamp(saturation, 0.0f, 1.0f) : 0.0f;
    const auto* water = tryFindSubstance("Water");
    const float density = water ? water->liquid_density : 1000.0f;
    // Empirical local-head closure, not a pressure PDE. Saturation controls
    // head; capillary bonds peak in the partly saturated state and vanish at 1.
    const float head = std::max(voxel_size, 0.0f) * s * s;
    return {1.0f + (params.wet_friction_scale - 1.0f) * s,
        1.0f + (params.wet_dilatancy_scale - 1.0f) * s,
        params.capillary_cohesion_pa * 4.0f * s * (1.0f - s),
        params.pore_pressure_scale * density * params.gravity_m_s2 * head};
}

bool buildMatterWetResponses(const FluidParticles& particles, bool legacy_granular,
                            float voxel_size, const MatterPoreParams& params,
                            std::vector<MatterWetResponse>& responses, std::string& error) {
    if (!validateMatterPoreParams(params, error) || !std::isfinite(voxel_size) ||
        voxel_size <= 0.0f) {
        error = error.empty() ? "C6 requires finite positive voxel size" : error;
        return false;
    }
    const auto count = particles.size();
    responses.assign(count, {1.0f, 1.0f, 0.0f, 0.0f});
    if (!params.wet_response_enabled) {
        return true;
    }
    if (particles.pore_capacity_kg.size() != count ||
        particles.pore_water_mass_kg.size() != count ||
        particles.constitutive_model.size() != count || particles.substance_tag.size() != count) {
        error = "C6 canonical pore sidecar cardinality is invalid";
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        auto model = static_cast<MatterConstitutiveModel>(particles.constitutive_model[i]);
        if (model == MatterConstitutiveModel::Auto) {
            model = legacy_granular ? MatterConstitutiveModel::Granular
                : MatterConstitutiveModel::Fluid;
        }
        const auto* profile = tryFindSubstanceByTag(particles.substance_tag[i]);
        if (model != MatterConstitutiveModel::Granular || !profile ||
            profile->default_constitutive_model != MatterConstitutiveModel::Granular) {
            continue;
        }
        const float mass = particles.pore_water_mass_kg[i];
        const float capacity = particles.pore_capacity_kg[i];
        if (!std::isfinite(mass) || !std::isfinite(capacity) || mass < 0.0f || capacity < 0.0f ||
            mass > capacity * 1.00001f || (mass > 0.0f && capacity <= 0.0f)) {
            error = "C6 pore mass/capacity is invalid";
            return false;
        }
        responses[i] = evaluateMatterWetResponse(matterParticleSaturation(particles, i),
            voxel_size, params);
        if (!std::all_of(responses[i].begin(), responses[i].end(), [](float value) {
                return std::isfinite(value);
            })) {
            error = "C6 derived wet response is nonfinite";
            return false;
        }
    }
    error.clear();
    return true;
}
} // namespace RayTrophiSim::Fluid
