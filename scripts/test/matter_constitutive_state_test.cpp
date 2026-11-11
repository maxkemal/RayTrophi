// User-built pure selection test. Link MatterConstitutiveState.cpp and Vec3.cpp.
#include "Fluid/MatterConstitutiveState.h"
#include "MaterialStateField.h"
#include <cassert>
#include <cmath>
#include <limits>
#include <string>

namespace F = RayTrophiSim::Fluid;
using F::MatterConstitutiveModel;

int main() {
    RayTrophiSim::SubstanceProfile snow;
    snow.default_constitutive_model = MatterConstitutiveModel::Granular;
    snow.meltable = true;
    snow.melt_kelvin = 273.15f;
    auto state = F::resolveMatterConstitutiveState(&snow, false, 260.0f);
    assert(state.model == MatterConstitutiveModel::Granular);
    assert(std::string(state.phase) == "solid");
    assert(std::string(state.phase_basis) == "temperature_inferred");
    state = F::resolveMatterConstitutiveState(&snow, false, 280.0f);
    assert(state.model == MatterConstitutiveModel::Fluid);
    // Latent state overrides the thermometer in both directions.
    state = F::resolveMatterConstitutiveState(&snow, false, 280.0f, 0.8f);
    assert(state.model == MatterConstitutiveModel::Granular);
    assert(state.fusion_fraction_used);
    assert(std::string(state.phase_basis) == "fusion_solid_fraction");
    assert(std::string(state.phase) == "mixed_solid_liquid");
    state = F::resolveMatterConstitutiveState(&snow, false, 260.0f, 0.2f);
    assert(state.model == MatterConstitutiveModel::Fluid);
    assert(F::resolveMatterConstitutiveState(&snow, false, 280.0f, 0.5f).model ==
        MatterConstitutiveModel::Granular);
    assert(F::resolveMatterConstitutiveState(&snow, false, 260.0f,
        std::nextafter(0.5f, 0.0f)).model == MatterConstitutiveModel::Fluid);
    // Unknown fusion falls back to the existing temperature/default rule.
    const float unknown = std::numeric_limits<float>::quiet_NaN();
    assert(F::resolveMatterConstitutiveState(&snow, false, 280.0f, unknown).model ==
        MatterConstitutiveModel::Fluid);
    assert(F::resolveMatterConstitutiveState(&snow, false, unknown).model ==
        MatterConstitutiveModel::Granular);
    snow.default_constitutive_model = MatterConstitutiveModel::Elastic;
    assert(F::resolveMatterConstitutiveState(&snow, false, 280.0f, 0.8f).model ==
        MatterConstitutiveModel::Elastic);

    RayTrophiSim::SubstanceProfile water;
    water.meltable = true;
    water.melt_kelvin = 273.15f;
    state = F::resolveMatterConstitutiveState(&water, false, 260.0f, 1.0f);
    assert(state.model == MatterConstitutiveModel::Fluid);
    assert(std::string(state.phase) == "solid");
    assert(!state.fusion_fraction_used);
    const auto definition = F::describeMatterPhaseMechanics(water);
    assert(definition.at("solid") == "supported_pinned_carrier");
    assert(!definition.at("solid_deformation_defined").get<bool>());

    RayTrophiSim::SubstanceProfile thermal_only;
    assert(F::describeMatterPhaseMechanics(thermal_only).at("solid") == "not_defined");
    assert(F::resolveMatterConstitutiveState(nullptr, true, 300.0f).model ==
        MatterConstitutiveModel::Granular);
    assert(F::resolveMatterConstitutiveState(nullptr, false, 300.0f).model ==
        MatterConstitutiveModel::Fluid);
    thermal_only.default_constitutive_model = MatterConstitutiveModel::Auto;
    assert(F::resolveMatterConstitutiveState(&thermal_only, true, 300.0f).model ==
        MatterConstitutiveModel::Granular);
}
