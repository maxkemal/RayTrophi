// User-built regression in the physics test target; no build run by Codex.
#include "Fluid/MatterMpmSoftening.h"
#include "Fluid/APICFluidSolver.h"
#include "Fluid/GranularConstitutive.h"
#include <cassert>
#include <cmath>

namespace F = RayTrophiSim::Fluid;
namespace G = F::Granular;

Vec3 stress(float soft, float viscosity) {
    G::StressUpdateParams params;
    params.dt = 1.0e-4f;
    params.young_modulus = 2.0e5f;
    params.poisson_ratio = 0.45f;
    params.elastic_solid = true;
    params.elastic_viscosity = viscosity;
    params.elastic_relaxation_s = 1.5f;
    params.elastic_yield_strain = 0.5f;
    F::AffineC rate;
    rate.col1.x = 0.1f;
    Vec3 f0(1, 0, 0), f1(0, 1, 0), f2(0, 0, 1), diagonal, shear;
    float volume = 1.0f, damage = 0.0f, hardening = 0.0f, history = 0.0f;
    float yield = 0.0f, plastic = 0.0f;
    uint32_t flags = 0;
    G::elasticStressUpdateParticle(rate, params, soft, f0, f1, f2,
        diagonal, shear, volume, damage, hardening, history, yield, plastic, flags);
    assert((flags & G::kFlagInvalid) == 0u);
    return shear;
}

int main() {
    F::FluidParticles particles;
    particles.position.resize(4);
    particles.temperature = {293.15f, 340.0f, 310.0f, 340.0f};
    particles.mass_fraction.assign(4, 1.0f);
    particles.constitutive_model.assign(4,
        static_cast<uint8_t>(F::MatterConstitutiveModel::Elastic));
    particles.constitutive_model[3] = static_cast<uint8_t>(F::MatterConstitutiveModel::Fluid);
    particles.ensureGranularStateSize();
    particles.granular_softening[3] = 0.77f;
    particles.granular_bond_scale[3] = 0.42f;
    F::APICSolverParams params;
    params.granular_enabled = false; // the actual elastic/mixed route
    params.elastic_solid = true;
    params.granular_softening_temperature = 310.0f;
    params.granular_softening_range = 15.0f;
    params.granular_residual_strength = 0.2f;
    assert(F::updateMatterMpmSoftening(particles, params) == 3);
    assert(particles.granular_softening[0] == 1.0f);
    assert(std::abs(particles.granular_softening[1] - 0.2f) < 1.0e-6f);
    assert(std::abs(particles.granular_softening[2] - 0.6f) < 1.0e-6f);
    assert(particles.granular_softening[3] == 0.77f);
    assert(particles.granular_bond_scale[3] == 0.42f);

    // Exercise the real constitutive law, isolating the viscous stress by
    // subtracting the same deformation response with eta = 0.
    const Vec3 cold_viscous = stress(1.0f, 30.0f) - stress(1.0f, 0.0f);
    const Vec3 hot_viscous = stress(0.2f, 30.0f) - stress(0.2f, 0.0f);
    assert(cold_viscous.length() > 0.01f);
    assert(std::abs(hot_viscous.length() / cold_viscous.length() - 0.2f) < 1.0e-3f);
    // A second frame must not progressively soften already softened values.
    F::updateMatterMpmSoftening(particles, params);
    assert(std::abs(particles.granular_softening[1] - 0.2f) < 1.0e-6f);
    params.granular_softening_temperature = 0.0f;
    F::updateMatterMpmSoftening(particles, params);
    assert(particles.granular_softening[1] == 1.0f);
    assert(particles.granular_softening[3] == 0.77f);
    // Granular MPM also works while its domain-wide switch is disabled.
    particles.constitutive_model[0] = static_cast<uint8_t>(F::MatterConstitutiveModel::Granular);
    params.granular_softening_temperature = 310.0f;
    assert(F::updateMatterMpmSoftening(particles, params) == 3);
}
