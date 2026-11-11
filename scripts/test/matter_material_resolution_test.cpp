// User-built regression; link with the application's physics/library test target.
// Codex does not build or run this target.
#include "Fluid/MatterMaterialResolution.h"
#include "Fluid/FluidDomainSubstance.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"
#include "ParticleSimulation.h"

#include <cassert>
#include <cmath>
#include <limits>

namespace F = RayTrophiSim::Fluid;
using namespace RayTrophiSim;

int main() {
    SubstanceProfile oil;
    oil.liquid_kinematic_viscosity = 1.0e-4f;
    auto result = F::resolveSubstanceViscosity(oil, 0.1f, false, false);
    assert(result.requested == 1.0e-4f && result.effective == 0.0f);
    assert(std::string(result.reason) == "below_resolution_threshold");
    assert(F::resolveSubstanceViscosity(oil, 0.01f, false, false).effective == 1.0e-4f);
    assert(F::resolveSubstanceViscosity(oil, 0.1f, true, false).effective == 1.0e-4f);
    assert(F::resolveSubstanceViscosity(oil, 0.1f, true, true).effective == 0.0f);
    // Metamorphic check: fixed nu/h^2 preserves the skip decision.
    oil.liquid_kinematic_viscosity *= 4.0f;
    assert(F::resolveSubstanceViscosity(oil, 0.2f, false, false).effective == 0.0f);

    SimulationGridDomainDesc domain;
    domain.type = SimulationDomainType::Matter;
    domain.voxel_size = 0.1f;
    domain.fluid_params.default_substance = "Oil";
    domain.fluid_params.kinematic_viscosity = 0.0f;
    domain.fluid_params.thermal_liquid_enabled = false;
    nlohmann::json report;
    std::string error;
    assert(F::inspectMatterMaterialResolution(domain, nullptr, report, error));
    assert(!report.at("carriers_measured").get<bool>());
    assert(report.at("backend_executed").is_null());
    assert(report.at("viscosity").at("configuration_matches_policy").get<bool>());

    SimulationGridDomainState state;
    state.valid = true;
    state.particles.position.resize(2);
    state.particles.constitutive_model.assign(2,
        static_cast<uint8_t>(F::MatterConstitutiveModel::Fluid));
    state.particles.temperature = {300.0f, 0.0f};
    state.particles.fusion_solid_fraction.assign(2,
        std::numeric_limits<float>::quiet_NaN());
    assert(F::inspectMatterMaterialResolution(domain, &state, report, error));
    assert(report.at("carriers").size() == 1);
    const auto& carrier = report.at("carriers").at(0);
    assert(carrier.at("particles") == 2);
    assert(carrier.at("mean_temperature_k") == 300.0);
    assert(carrier.at("temperature_known_particles") == 1);
    assert(carrier.at("mean_fusion_solid_fraction").is_null());
    assert(state.particles.temperature[1] == 0.0f); // read-only
    assert(std::isnan(state.particles.fusion_solid_fraction[0]));

    const auto sentinel = report;
    state.particles.temperature[0] = -1.0f;
    assert(!F::inspectMatterMaterialResolution(domain, &state, report, error));
    assert(report == sentinel && !error.empty());
    state.particles.temperature[0] = 300.0f;
    state.particles.fusion_solid_fraction[0] = 1.1f;
    assert(!F::inspectMatterMaterialResolution(domain, &state, report, error));
    assert(report == sentinel);
    state.particles.fusion_solid_fraction[0] =
        std::numeric_limits<float>::quiet_NaN();
    state.particles.constitutive_model[0] = 255u;
    assert(!F::inspectMatterMaterialResolution(domain, &state, report, error));
    assert(report == sentinel);
    state.particles.constitutive_model[0] =
        static_cast<uint8_t>(F::MatterConstitutiveModel::Fluid);
    state.particles.temperature = {300.0f};
    assert(!F::inspectMatterMaterialResolution(domain, &state, report, error));
    assert(report == sentinel);
    domain.fluid_params.default_substance = "__unknown_material_resolution__";
    assert(!F::inspectMatterMaterialResolution(domain, nullptr, report, error));
    assert(report == sentinel);
    domain.fluid_params.default_substance = "Oil";
    domain.voxel_size = std::numeric_limits<float>::infinity();
    assert(!F::inspectMatterMaterialResolution(domain, nullptr, report, error));
    assert(report == sentinel);
}
