#include "Api/RtApi.h"
#include "MaterialStateField.h"

namespace rtapi {

Result getMaterialSubstance(const std::string& name, SubstanceProfileInfo& out_info) {
    const RayTrophiSim::SubstanceProfile* profile =
        RayTrophiSim::tryFindSubstance(name);
    if (!profile) return Result::fail("Unknown substance: " + name);

    out_info = {};
    out_info.name = profile->name;
    out_info.default_constitutive_model =
        RayTrophiSim::Fluid::matterConstitutiveModelName(
            profile->default_constitutive_model);
    out_info.density = profile->density;
    out_info.liquid_density = profile->liquid_density;
    out_info.specific_heat = profile->specific_heat;
    out_info.conductivity = profile->conductivity;
    out_info.liquid_kinematic_viscosity = profile->liquid_kinematic_viscosity;
    out_info.combustible = profile->combustible;
    out_info.fluid_flammable = profile->fluid_flammable;
    out_info.fluid_extinguishing = profile->fluid_extinguishing;
    out_info.meltable = profile->meltable;
    out_info.ignition_kelvin = profile->ignition_kelvin;
    out_info.flash_kelvin = profile->flash_kelvin;
    out_info.autoignition_kelvin = profile->autoignition_kelvin;
    out_info.melt_kelvin = profile->melt_kelvin;
    out_info.boiling_kelvin = profile->boiling_kelvin;
    out_info.latent_heat_fusion = profile->latent_heat_fusion;
    out_info.latent_heat_vaporization = profile->latent_heat_vaporization;
    out_info.vaporization_rate = profile->vaporization_rate;
    out_info.cooling_power = profile->cooling_power;
    out_info.oxygen_dilution = profile->oxygen_dilution;
    out_info.flame_persistence = profile->flame_persistence;
    out_info.granular_friction_degrees = profile->granular_friction_degrees;
    out_info.granular_cohesion = profile->granular_cohesion;
    return Result::success();
}

} // namespace rtapi
