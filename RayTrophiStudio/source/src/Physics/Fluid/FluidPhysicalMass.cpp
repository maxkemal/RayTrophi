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

float densityForParticle(uint32_t tag, const SubstanceProfile* fallback) {
    const SubstanceProfile* profile = tryFindSubstanceByTag(tag);
    if (!profile) profile = fallback;
    if (!profile || !std::isfinite(profile->liquid_density) ||
        profile->liquid_density <= 0.0f) {
        return 1000.0f;
    }
    return profile->liquid_density;
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
    int particles_per_cell) {
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
            rest_mass = densityForParticle(tag, fallback) * parcel_volume;
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

} // namespace Fluid
} // namespace RayTrophiSim
