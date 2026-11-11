// User-built regression. Link MudMaterial.cpp and source/src/Math/Vec3.cpp.
#include "Fluid/MudMaterial.h"
#include "Fluid/GranularConstitutive.h"
#include "MaterialStateField.h"
#include <cassert>
#include <cmath>

namespace F = RayTrophiSim::Fluid;
namespace G = F::Granular;

struct Response {
    Vec3 stress;
    float yield = 0.0f;
    float plastic = 0.0f;
    uint32_t flags = 0;
};

Response evaluate(const RayTrophiSim::SubstanceProfile& p, float strain,
    const F::AffineC& affine, float viscosity) {
    G::StressUpdateParams pc;
    pc.dt = 1.0e-4f;
    pc.young_modulus = p.elastic_young_modulus;
    pc.poisson_ratio = p.elastic_poisson_ratio;
    pc.elastic_solid = true;
    pc.elastic_yield_strain = p.elastic_yield_strain;
    pc.elastic_relaxation_s = p.elastic_relaxation_s;
    pc.elastic_viscosity = viscosity;
    pc.elastic_fracture_strain = p.elastic_fracture_strain;
    pc.damage_rate = p.elastic_damage_rate;
    Vec3 f0(std::exp(strain), 0.0f, 0.0f);
    Vec3 f1(0.0f, std::exp(-strain), 0.0f);
    Vec3 f2(0.0f, 0.0f, 1.0f), diagonal, shear;
    float volume = 1.0f, damage = 0.0f, hardening = 0.0f, history = 0.0f;
    Response result;
    G::elasticStressUpdateParticle(affine, pc, 1.0f, f0, f1, f2,
        diagonal, shear, volume, damage, hardening, history,
        result.yield, result.plastic, result.flags);
    result.stress = shear;
    assert(std::isfinite(diagonal.length()) && std::isfinite(shear.length()));
    assert((result.flags & G::kFlagInvalid) == 0u);
    return result;
}

int main() {
    const auto mud = F::makeDenseMud();
    const auto slurry = F::makeMudSlurry();
    assert(mud.name == "Mud");
    assert(mud.default_constitutive_model == F::MatterConstitutiveModel::Elastic);
    assert(mud.elastic_yield_strain > 0.0f);
    assert(mud.elastic_viscosity_pa_s > 0.0f);
    assert(mud.elastic_relaxation_s == 0.0f);
    assert(mud.elastic_fracture_strain > 0.0f);
    assert(slurry.name == "Mud Slurry");
    assert(slurry.default_constitutive_model == F::MatterConstitutiveModel::Fluid);
    assert(slurry.liquid_kinematic_viscosity > 0.0f);
    F::AffineC still;
    const auto small = evaluate(mud, 0.005f, still, mud.elastic_viscosity_pa_s);
    assert(small.yield == 0.0f && small.plastic == 0.0f);
    const auto large = evaluate(mud, 0.10f, still, mud.elastic_viscosity_pa_s);
    assert(large.yield > 0.0f && large.plastic > 0.0f);
    assert((large.flags & G::kFlagYielded) != 0u);
    F::AffineC shearing;
    shearing.col1 = Vec3(2.0f, 0.0f, 0.0f);
    const auto inviscid = evaluate(mud, 0.0f, shearing, 0.0f);
    const auto viscous = evaluate(mud, 0.0f, shearing, mud.elastic_viscosity_pa_s);
    assert(std::abs(viscous.stress.x) > std::abs(inviscid.stress.x));
}
