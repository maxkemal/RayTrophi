// Source regression for the user's test build; Codex does not compile it.
#include "Fluid/MatterSubstanceState.h"
#include "Fluid/FluidThermalLiquid.h"
#include "Fluid/SubstanceTag.h"
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/MatterGrainBirth.h"
#include "MaterialStateField.h"
#include "ParticleSimulation.h"

#include <cassert>
#include <cmath>
#include <vector>

using namespace RayTrophiSim;
using namespace RayTrophiSim::Fluid;

int main() {
    APICSolverParams params;
    params.default_substance = "Water";
    params.thermal_liquid_enabled = true;
    params.thermal_freeze_kelvin = 999.0f;
    params.thermal_cold_viscosity = 999.0f;
    const auto water = substanceTag("Water");
    const auto wax = substanceTag("Wax");
    const auto ice = substanceTag("Ice");
    assert(std::abs(substanceFreezeKelvin(water, params) - 273.15f) < 1.0e-4f);
    assert(std::abs(substanceFreezeKelvin(wax, params) - 330.0f) < 1.0e-4f);
    assert(substanceFreezeKelvin(0u, params) == substanceFreezeKelvin(water, params));
    assert(!std::isfinite(substanceFreezeKelvin(substanceTag("Sand"), params)));
    assert(substanceBirthModel(ice, params, 263.0f) == MatterConstitutiveModel::Granular);
    assert(substanceBirthModel(ice, params, 300.0f) == MatterConstitutiveModel::Fluid);

    FluidParticles particles;
    particles.emit(Vec3(0.15f), Vec3(1.0f, 2.0f, 3.0f), 263.0f, 0.0f, ice,
        nullptr, nullptr, 0.01f, MatterConstitutiveModel::Granular);
    const auto identity = particles.particle_id[0];
    particles.granular_stress_diag[0] = Vec3(123.0f);
    particles.temperature[0] = 300.0f;
    refreshMatterConstitutiveModels(particles, params);
    assert(particles.constitutive_model[0] == static_cast<uint8_t>(MatterConstitutiveModel::Fluid));
    assert(particles.particle_id[0] == identity && particles.rest_mass_kg[0] == 0.01f);
    assert(particles.velocity[0].y == 2.0f && particles.temperature[0] == 300.0f);
    assert(particles.granular_stress_diag[0].length() == 0.0f);

    const std::vector<uint32_t> static_tags{ice};
    std::vector<uint8_t> mask;
    bool any_frozen = true;
    assert(buildMatterObstacleMask(particles, &static_tags, mask, any_frozen));
    assert(mask[0] == 1u && !any_frozen);
    particles.substance_tag[0] = 0u;
    assert(!buildMatterObstacleMask(particles, &static_tags, mask, any_frozen));
    assert(mask.empty());
    particles.flags[0] |= kParticleFlagFrozen;
    assert(buildMatterObstacleMask(particles, nullptr, mask, any_frozen) && any_frozen);
    particles.mass_fraction[0] = 0.0f;
    assert(!buildMatterObstacleMask(particles, &static_tags, mask, any_frozen));
    assert(!any_frozen && mask.empty());

    // Independent material curves are evaluated before spatial averaging.
    // The old domain-threshold implementation thickened the water as wax.
    FluidParticles mixture;
    mixture.emit(Vec3(0.15f), Vec3(0.0f), 300.0f, 0.0f, water);
    mixture.emit(Vec3(0.15f), Vec3(0.0f), 300.0f, 0.0f, wax);
    FluidSim::FluidGrid grid(3, 3, 3, 0.1f);
    std::vector<float> viscosity;
    assert(buildThermalViscosityField(mixture, grid, params, nullptr, viscosity));
    const auto* water_profile = tryFindSubstance("Water");
    const auto* wax_profile = tryFindSubstance("Wax");
    assert(water_profile && wax_profile);
    const float expected = 0.5f * (water_profile->liquid_kinematic_viscosity +
                                  wax_profile->liquid_cold_viscosity);
    assert(std::abs(viscosity[grid.cellIndex(1, 1, 1)] - expected) < 1.0e-6f);

    params.grain.enabled = true;
    assert(substanceTransportOwner(substanceTag("Sand"), MatterConstitutiveModel::Granular,
        params) == MatterTransportOwner::Grain);
    assert(substanceTransportOwner(substanceTag("Soil"), MatterConstitutiveModel::Granular,
        params) == MatterTransportOwner::Mpm);
    FluidParticles owners;
    owners.emit(Vec3(0.0f), Vec3(0.0f), 293.15f, 0.0f, substanceTag("Sand"),
        nullptr, nullptr, 0.01f, MatterConstitutiveModel::Granular);
    owners.emit(Vec3(0.0f), Vec3(0.0f), 293.15f, 0.0f, substanceTag("Soil"),
        nullptr, nullptr, 0.01f, MatterConstitutiveModel::Granular);
    const auto summary = inspectMatterOwners(owners, params);
    assert(summary.particles[1] == 1 && summary.particles[2] == 1);
    assert(summary.ready && summary.reason.empty());

    owners.rest_mass_kg[0] = 0.0f;
    owners.rest_mass_kg[1] = 0.0f;
    ensureMatterGrainRestMasses(owners, params.grain, water_profile, false);
    assert(owners.rest_mass_kg[0] > 0.0f && owners.rest_mass_kg[1] == 0.0f);
    ensureFluidParticleRestMasses(owners, water_profile, 0.1f, 8, false);
    assert(std::abs(owners.rest_mass_kg[1] - 0.18125f) < 1.0e-6f);

    // Only DEM spheres constrain DEM births. MPM parcels and consumed grains
    // cannot silently starve an emitter through the sphere-spacing filter.
    FluidParticles occupied;
    occupied.emit(Vec3(0.15f), Vec3(0.0f), 293.15f, 0.0f, substanceTag("Soil"),
        nullptr, nullptr, 0.01f, MatterConstitutiveModel::Granular);
    MatterGrainBirthFilter mpm_birth(occupied, grid, 0.025f, &params);
    assert(mpm_birth.accept(Vec3(0.15f)));
    occupied.substance_tag[0] = substanceTag("Sand");
    MatterGrainBirthFilter dem_birth(occupied, grid, 0.025f, &params);
    assert(!dem_birth.accept(Vec3(0.15f)));
    occupied.mass_fraction[0] = 0.0f;
    MatterGrainBirthFilter consumed_birth(occupied, grid, 0.025f, &params);
    assert(consumed_birth.accept(Vec3(0.15f)));
}
