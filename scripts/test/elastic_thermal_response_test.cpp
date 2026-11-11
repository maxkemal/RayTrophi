// User build: link Math/Vec3.cpp. Exercises the actual CPU constitutive law.
#include "Fluid/GranularConstitutive.h"
#include <cassert>
#include <cmath>

namespace G = RayTrophiSim::Fluid::Granular;

struct Response {
    float stretch = 0.0f;
    float plastic = 0.0f;
};

Response evaluate(float soft, float yield_power, float relaxation_power, bool relax) {
    G::StressUpdateParams pc;
    pc.dt = 0.01f;
    pc.young_modulus = 2.0e5f;
    pc.elastic_solid = true;
    pc.elastic_yield_strain = relax ? 0.0f : 0.5f;
    pc.elastic_relaxation_s = relax ? 1.5f : 0.0f;
    pc.elastic_thermal_yield_power = yield_power;
    pc.elastic_thermal_relaxation_power = relaxation_power;
    RayTrophiSim::Fluid::AffineC rate;
    Vec3 f0(std::exp(0.2f), 0, 0), f1(0, std::exp(-0.2f), 0), f2(0, 0, 1);
    Vec3 diag, shear;
    float volume = 1.0f, damage = 0.0f, hardening = 0.0f, history = 0.0f;
    float yield = 0.0f, plastic = 0.0f;
    uint32_t flags = 0;
    G::elasticStressUpdateParticle(rate, pc, soft, f0, f1, f2, diag, shear,
        volume, damage, hardening, history, yield, plastic, flags);
    assert((flags & G::kFlagInvalid) == 0u);
    return {f0.length(), plastic};
}

int main() {
    assert(G::elasticThermalScale(0.02f, 0.0f) == 1.0f);
    assert(G::elasticThermalScale(0.02f, 1.0f) == 0.02f);
    assert(std::abs(G::elasticThermalScale(0.2f, 2.0f) - 0.04f) < 1.0e-6f);
    assert(evaluate(1.0f, 1.0f, 1.0f, false).plastic == 0.0f);
    assert(evaluate(0.02f, 0.0f, 0.0f, false).plastic == 0.0f);
    assert(evaluate(0.02f, 1.0f, 0.0f, false).plastic > 0.0f);
    const auto fixed = evaluate(0.02f, 0.0f, 0.0f, true);
    const auto thermal = evaluate(0.02f, 0.0f, 1.0f, true);
    assert(thermal.stretch < fixed.stretch);
    assert(std::abs(evaluate(1.0f, 1.0f, 1.0f, true).stretch -
        evaluate(1.0f, 0.0f, 0.0f, true).stretch) < 1.0e-6f);
    // Zero base disables the term; hot scaling must not accidentally enable it.
    assert(std::isfinite(evaluate(0.0f, 4.0f, 4.0f, true).stretch));
}
