// User-built regression: link BuiltinPresetPhysics.cpp and Math/Vec3.cpp.
#include "Fluid/BuiltinPresetPhysics.h"
#include "MaterialStateField.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

using RayTrophiSim::SubstanceProfile;

SubstanceProfile& get(std::vector<SubstanceProfile>& profiles, const char* name) {
    const auto found = std::find_if(profiles.begin(), profiles.end(),
        [name](const SubstanceProfile& profile) { return profile.name == name; });
    assert(found != profiles.end());
    return *found;
}

int main() {
    std::vector<SubstanceProfile> profiles;
    for (const char* name : {"Water", "Snow", "Ice", "Honey", "Oil", "Chocolate",
                            "Stone", "Plastic (PE)", "Gasoline", "Alcohol", "Paper",
                            "Chewing Gum"}) {
        SubstanceProfile profile;
        profile.name = name;
        profiles.push_back(profile);
    }
    get(profiles, "Water").liquid_density = 997.0f;
    get(profiles, "Water").liquid_kinematic_viscosity = 1.0e-6f;
    for (const char* name : {"Snow", "Ice"}) {
        get(profiles, name).liquid_kinematic_viscosity = 0.0f;
        get(profiles, name).default_constitutive_model =
            RayTrophiSim::Fluid::MatterConstitutiveModel::Granular;
    }
    for (const char* name : {"Honey", "Oil", "Chocolate"}) {
        get(profiles, name).solver_internal_friction = 0.2f;
    }
    auto& polymer = get(profiles, "Plastic (PE)");
    polymer.meltable = true;
    polymer.liquid_kinematic_viscosity = 0.30f;
    polymer.liquid_cold_viscosity = 0.05f;
    auto& stone = get(profiles, "Stone");
    stone.meltable = true;
    stone.density = 2600.0f;
    const auto paper = get(profiles, "Paper");
    const auto size = profiles.size();
    RayTrophiSim::Fluid::configureBuiltinPresetPhysics(profiles);
    assert(profiles.size() == size); // does not create aliases or extra profiles
    for (const char* name : {"Snow", "Ice"}) {
        const auto& profile = get(profiles, name);
        assert(profile.liquid_kinematic_viscosity ==
            get(profiles, "Water").liquid_kinematic_viscosity);
        assert(profile.liquid_density == get(profiles, "Water").liquid_density);
        assert(profile.default_constitutive_model ==
            RayTrophiSim::Fluid::MatterConstitutiveModel::Granular);
    }
    for (const char* name : {"Honey", "Oil", "Chocolate"}) {
        assert(get(profiles, name).solver_internal_friction == 0.0f);
    }
    assert(stone.liquid_density == stone.density);
    assert(stone.liquid_kinematic_viscosity > get(profiles, "Water").liquid_kinematic_viscosity);
    for (const auto& profile : profiles) {
        assert(!profile.meltable ||
            profile.liquid_cold_viscosity >= profile.liquid_kinematic_viscosity);
    }
    assert(get(profiles, "Alcohol").flash_kelvin < get(profiles, "Alcohol").boiling_kelvin);
    assert(std::abs(get(profiles, "Gasoline").flash_kelvin -
        ((-45.0f - 32.0f) * 5.0f / 9.0f + 273.15f)) < 0.01f);
    assert(std::abs(get(profiles, "Alcohol").autoignition_kelvin -
        ((689.0f - 32.0f) * 5.0f / 9.0f + 273.15f)) < 0.01f);
    assert(get(profiles, "Paper").density == paper.density);
    assert(get(profiles, "Paper").ignition_kelvin == paper.ignition_kelvin);
    assert(get(profiles, "Paper").burn_rate == paper.burn_rate);
    // Publication must be idempotent, without progressively inflating viscosity.
    RayTrophiSim::Fluid::configureBuiltinPresetPhysics(profiles);
    assert(polymer.liquid_cold_viscosity == 0.30f);
    assert(stone.liquid_cold_viscosity == 0.5f);
}
