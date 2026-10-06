#include "Fluid/FluidPhysicalMass.h"

#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"

#include <algorithm>
#include <cmath>

namespace RayTrophiSim {
namespace Fluid {

namespace {

const SubstanceProfile* presetProfile(FluidChemistryPreset preset) {
    switch (preset) {
        case FluidChemistryPreset::Water: return tryFindSubstance("Water");
        case FluidChemistryPreset::Gasoline: return tryFindSubstance("Gasoline");
        case FluidChemistryPreset::Alcohol: return tryFindSubstance("Alcohol");
        case FluidChemistryPreset::Oil: return tryFindSubstance("Oil");
        case FluidChemistryPreset::Plastic: return tryFindSubstance("Plastic (PE)");
        case FluidChemistryPreset::Wax: return tryFindSubstance("Wax");
        default: return nullptr;
    }
}

float densityForParticle(uint32_t tag, const SubstanceProfile* fallback,
                         MatterConstitutiveModel model, bool legacy_granular) {
    const SubstanceProfile* profile = tryFindSubstanceByTag(tag);
    if (!profile) {
        profile = fallback;
    }
    if (model == MatterConstitutiveModel::Auto) {
        model = legacy_granular ? MatterConstitutiveModel::Granular
            : MatterConstitutiveModel::Fluid;
    }
    const float density = profile
        ? (model == MatterConstitutiveModel::Granular ? profile->density : profile->liquid_density)
        : 1000.0f;
    return std::isfinite(density) && density > 0.0f ? density : 1000.0f;
}

} // namespace

const SubstanceProfile* resolveFluidSubstanceProfile(
    uint32_t substance_tag,
    FluidChemistryPreset chemistry_preset) {
    const SubstanceProfile* profile = tryFindSubstanceByTag(substance_tag);
    return profile ? profile : presetProfile(chemistry_preset);
}

FluidPhysicalMassStats ensureFluidParticleRestMasses(
    FluidParticles& particles,
    FluidChemistryPreset chemistry_preset,
    float voxel_size,
    int particles_per_cell, bool legacy_granular) {
    FluidPhysicalMassStats stats;
    const std::size_t count = particles.size();
    particles.rest_mass_kg.resize(count, 0.0f);
    const SubstanceProfile* fallback = presetProfile(chemistry_preset);
    const float h = std::isfinite(voxel_size) && voxel_size > 0.0f
        ? voxel_size : 0.1f;
    const float parcel_volume = h * h * h /
        static_cast<float>(std::max(1, particles_per_cell));

    for (std::size_t i = 0; i < count; ++i) {
        float& rest_mass = particles.rest_mass_kg[i];
        if (!std::isfinite(rest_mass) || rest_mass <= 0.0f) {
            const uint32_t tag = i < particles.substance_tag.size()
                ? particles.substance_tag[i] : kSubstanceUntagged;
            const auto model = i < particles.constitutive_model.size()
                ? static_cast<MatterConstitutiveModel>(particles.constitutive_model[i])
                : MatterConstitutiveModel::Auto;
            rest_mass = densityForParticle(tag, fallback, model, legacy_granular) * parcel_volume;
            ++stats.initialized_particles;
        }
        const float fraction = i < particles.mass_fraction.size() &&
            std::isfinite(particles.mass_fraction[i])
            ? std::clamp(particles.mass_fraction[i], 0.0f, 1.0f) : 1.0f;
        stats.rest_mass_kg += static_cast<double>(rest_mass);
        stats.current_mass_kg += static_cast<double>(rest_mass) * fraction;
    }
    return stats;
}

float fluidParticleRestMassKg(uint32_t tag, FluidChemistryPreset preset,
                             float voxel_size, int particles_per_cell,
                             MatterConstitutiveModel model, bool legacy_granular) {
    const float h = std::isfinite(voxel_size) && voxel_size > 0.0f
        ? voxel_size : 0.1f;
    return densityForParticle(tag, presetProfile(preset), model, legacy_granular) * h * h * h /
        static_cast<float>(std::max(1, particles_per_cell));
}

} // namespace Fluid
} // namespace RayTrophiSim
