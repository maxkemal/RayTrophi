// Source regression for the user's test build; Codex does not compile it.
#include "Fluid/FluidPhysicalMass.h"
#include "Fluid/SubstanceTag.h"
#include "MaterialStateField.h"

#include <cassert>
#include <cmath>

using namespace RayTrophiSim::Fluid;

int main() {
    const auto* preset = RayTrophiSim::tryFindSubstance("Water");
    assert(preset);
    const auto soil = substanceTag("Soil");
    const auto water = substanceTag("Water");
    const auto granular = MatterConstitutiveModel::Granular;
    const auto fluid = MatterConstitutiveModel::Fluid;
    const auto automatic = MatterConstitutiveModel::Auto;
    const auto mass = [&](uint32_t tag, MatterConstitutiveModel model, bool legacy = false) {
        return fluidParticleRestMassKg(tag, preset, 0.1f, 8, model, legacy);
    };
    assert(std::abs(mass(soil, granular) - 0.18125f) < 1e-6f);
    assert(std::abs(mass(substanceTag("Sand"), granular) - 0.2f) < 1e-6f);
    assert(std::abs(mass(substanceTag("Gravel"), granular) - 0.21875f) < 1e-6f);
    assert(std::abs(mass(soil, fluid) - 0.125f) < 1e-6f);
    assert(mass(soil, automatic, true) == mass(soil, granular));
    assert(mass(soil, automatic, false) == mass(soil, fluid));
    assert(mass(water, fluid, true) == mass(water, fluid, false));

    FluidParticles particles;
    particles.emit(Vec3(0.0f), Vec3(0.0f), 293.15f, 0.0f, soil,
        nullptr, nullptr, 0.0f, granular);
    particles.emit(Vec3(0.0f), Vec3(0.0f), 293.15f, 0.0f, water,
        nullptr, nullptr, 0.004f, fluid);
    particles.mass_fraction[1] = 0.25f;
    auto stats = ensureFluidParticleRestMasses(particles, preset, 0.1f, 8);
    assert(stats.initialized_particles == 1);
    assert(std::abs(particles.rest_mass_kg[0] - 0.18125f) < 1e-6f);
    assert(particles.rest_mass_kg[1] == 0.004f);
    assert(particles.mass_fraction[1] == 0.25f);
    stats = ensureFluidParticleRestMasses(particles, preset, 0.1f, 8, true);
    assert(stats.initialized_particles == 0);
}
