#include "Fluid/MatterConstitutiveState.h"
#include "MaterialStateField.h"

#include <cmath>

namespace RayTrophiSim::Fluid {

bool hasMatterSolidSkeleton(const SubstanceProfile* profile) {
    return profile && profile->meltable &&
        (profile->default_constitutive_model == MatterConstitutiveModel::Granular ||
         profile->default_constitutive_model == MatterConstitutiveModel::Elastic);
}

MatterConstitutiveState resolveMatterConstitutiveState(const SubstanceProfile* profile,
    bool granular_fallback, float kelvin, float solid_fraction) {
    MatterConstitutiveState result;
    result.model = granular_fallback ? MatterConstitutiveModel::Granular
                                    : MatterConstitutiveModel::Fluid;
    if (profile && profile->default_constitutive_model != MatterConstitutiveModel::Auto) {
        result.model = profile->default_constitutive_model;
        result.basis = "profile_default";
        if (hasMatterSolidSkeleton(profile)) {
            if (std::isfinite(solid_fraction)) {
                result.model = solid_fraction >= kFusionSolidThreshold
                    ? profile->default_constitutive_model : MatterConstitutiveModel::Fluid;
                result.basis = "fusion_solid_fraction";
                result.fusion_fraction_used = true;
            } else if (std::isfinite(kelvin) && kelvin >= profile->melt_kelvin) {
                result.model = MatterConstitutiveModel::Fluid;
                result.basis = "birth_temperature_above_melt";
            }
        }
    }
    // Phase is a separate observation. A partially molten parcel can still
    // use the skeleton model; a supported frozen fluid remains a fluid carrier.
    if (profile && profile->meltable) {
        if (std::isfinite(solid_fraction) && solid_fraction >= 0.0f && solid_fraction <= 1.0f) {
            result.phase_basis = "fusion_solid_fraction";
            result.phase = solid_fraction == 0.0f ? "liquid"
                : solid_fraction == 1.0f ? "solid" : "mixed_solid_liquid";
        } else if (std::isfinite(kelvin) && kelvin > 0.0f) {
            result.phase_basis = "temperature_inferred";
            result.phase = kelvin < profile->melt_kelvin ? "solid"
                : kelvin > profile->melt_kelvin ? "liquid" : "fusion_unresolved";
        }
    } else if (result.model == MatterConstitutiveModel::Granular ||
               result.model == MatterConstitutiveModel::Elastic) {
        result.phase = "solid";
        result.phase_basis = "nonmeltable_skeleton";
    }
    return result;
}

nlohmann::json describeMatterPhaseMechanics(const SubstanceProfile& profile) {
    const auto model = profile.default_constitutive_model;
    const bool skeleton = model == MatterConstitutiveModel::Granular ||
        model == MatterConstitutiveModel::Elastic;
    const bool fluid = model == MatterConstitutiveModel::Fluid;
    const char* solid = skeleton ? matterConstitutiveModelName(model)
        : fluid && profile.meltable ? "supported_pinned_carrier" : "not_defined";
    const char* liquid = fluid || (skeleton && profile.meltable) ? "fluid" : "not_defined";
    return {{"scope", "particle_mechanics_not_surface_object_mechanics"},
        {"solid", solid}, {"liquid", liquid},
        {"solid_deformation_defined", skeleton &&
            (model == MatterConstitutiveModel::Elastic ||
             profile.granular_transport == MatterGranularTransport::Mpm)},
        {"fusion_model_switch", hasMatterSolidSkeleton(&profile)},
        {"solid_transport", skeleton
            ? model == MatterConstitutiveModel::Elastic ? "mpm"
                : matterGranularTransportName(profile.granular_transport)
            : "not_applicable"},
        {"solid_fraction_switch_threshold", kFusionSolidThreshold},
        {"pinned_freeze_requires", fluid && profile.meltable
            ? "thermal_chain_and_supported_contact" : "not_applicable"},
        {"cold_fluid_birth_remains_fluid_carrier", fluid && profile.meltable},
        {"gas_mechanics", "separate_gas_domain"}};
}

} // namespace RayTrophiSim::Fluid
