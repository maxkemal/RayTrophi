#include "Fluid/BuiltinPresetPhysics.h"
#include "MaterialStateField.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace RayTrophiSim::Fluid {
namespace {

SubstanceProfile& require(std::vector<SubstanceProfile>& profiles, const char* name) {
    const auto found = std::find_if(profiles.begin(), profiles.end(),
        [name](const SubstanceProfile& profile) { return profile.name == name; });
    if (found == profiles.end()) {
        throw std::logic_error(std::string("Missing built-in preset: ") + name);
    }
    return *found;
}

} // namespace

void configureBuiltinPresetPhysics(std::vector<SubstanceProfile>& profiles) {
    const auto& water = require(profiles, "Water");
    // The zero used to suppress the solid solver also erased meltwater viscosity.
    // The granular domain resolver already suppresses liquid viscosity itself.
    for (const char* name : {"Snow", "Ice"}) {
        auto& profile = require(profiles, name);
        profile.liquid_density = water.liquid_density;
        profile.liquid_kinematic_viscosity = water.liquid_kinematic_viscosity;
    }

    // These presets already use shear viscosity. Body drag changes free fall
    // and rigid translation too; it is not an additional material rheology.
    for (const char* name : {"Honey", "Oil", "Chocolate"}) {
        require(profiles, name).solver_internal_friction = 0.0f;
    }

    // Reuse the existing Lava authoring starting point, not the dimensionless
    // MSF melt_viscosity. Stone is a broad example, not a fitted magma sample.
    auto& stone = require(profiles, "Stone");
    stone.liquid_kinematic_viscosity = 0.5f;
    stone.liquid_density = stone.density; // explicit equal-density approximation
    stone.solver_viscosity_sweeps = 24.0f;
    stone.solver_viscosity_wall_slip = 0.0f;

    // The existing freeze curve must not thin a built-in as it cools. This
    // conservative floor is not a newly fitted temperature-viscosity law.
    for (auto& profile : profiles) {
        if (profile.meltable) {
            profile.liquid_cold_viscosity = std::max(profile.liquid_cold_viscosity,
                profile.liquid_kinematic_viscosity);
        }
    }

    // Gasoline: NIOSH Pocket Guide, flash point -45 F -> 230.37 K.
    // Alcohol's density/molar mass/boiling point already describe ethanol.
    // Ethanol: NIOSH flash point 55 F; NOAA CAMEO autoignition 689 F.
    require(profiles, "Gasoline").flash_kelvin = 230.37f;
    auto& ethanol = require(profiles, "Alcohol");
    ethanol.flash_kelvin = 285.93f;
    ethanol.autoignition_kelvin = 638.15f;
    // Authoring recipe, not a fitted gum sample: hot gum retains a small
    // skeleton while yielding and relaxing faster. Other recipes keep defaults.
    auto& gum = require(profiles, "Chewing Gum");
    gum.granular_residual_strength = 0.02f;
    gum.elastic_thermal_viscosity_power = 1.0f;
    gum.elastic_thermal_yield_power = 1.0f;
    gum.elastic_thermal_relaxation_power = 1.0f;
}

} // namespace RayTrophiSim::Fluid
