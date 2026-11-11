#include "Fluid/DemContactTimeBounds.h"

#include <cassert>
#include <cmath>
#include <cstring>

int main() {
    using namespace RayTrophiSim::Fluid;
    DemContactTimeInput input;
    input.mass_kg = 0.01;
    input.normal_stiffness_n_m = 1000.0;
    input.contacts = 12.0;
    input.contact_resolution = 20.0;
    const auto dry = demExplicitContactTimeBounds(input, 1.0);
    const double omega = std::sqrt(2.0 * 12.0 * 1000.0 / 0.01);
    assert(std::abs(dry.stability * omega - 1.0) < 1e-12);
    // Stiffness quadrupling halves both elastic time scales.
    const auto stiff = demExplicitContactTimeBounds(input, 4.0);
    assert(std::abs(stiff.stability / dry.stability - 0.5) < 1e-12);
    assert(std::abs(stiff.accuracy / dry.accuracy - 0.5) < 1e-12);
    // Four times the mass doubles both scales at unchanged damping ratio.
    input.mass_kg *= 4.0;
    const auto heavy = demExplicitContactTimeBounds(input, 1.0);
    assert(std::abs(heavy.stability / dry.stability - 2.0) < 1e-12);
    input.mass_kg /= 4.0;
    input.tangential_ratio = 1.0;
    assert(std::strcmp(demExplicitContactTimeBounds(input, 1.0).source,
        "tangential") == 0);
    input.tangential_ratio = 0.0;
    input.rolling_friction = 1.0;
    assert(std::strcmp(demExplicitContactTimeBounds(input, 1.0).source, "rolling") == 0);
    input.bridge_stiffness_n_m = 10000.0;
    const auto wet = demExplicitContactTimeBounds(input, 1.0);
    const auto softened = demExplicitContactTimeBounds(input, 0.01);
    assert(std::strcmp(wet.source, "bridge") == 0);
    assert(wet.k_effective == softened.k_effective);
    input.damping_ratio = 0.7;
    const auto damped = demExplicitContactTimeBounds(input, 1.0);
    assert(damped.stability < wet.stability);
    // The selected dt is half the coupled spring/dashpot stability limit.
    // At twice dt, the symplectic Euler boundary must evaluate to four.
    const double w = std::sqrt(2.0 * input.contacts * damped.k_effective / input.mass_kg);
    const double h = 2.0 * damped.stability;
    assert(std::abs(w * w * h * h + 4.0 * damped.aggregate_zeta * w * h - 4.0) < 1e-12);
    input.damping_ratio = 1e12;
    const auto extreme = demExplicitContactTimeBounds(input, 1.0);
    assert(std::isfinite(extreme.stability) && extreme.stability > 0.0);
}
