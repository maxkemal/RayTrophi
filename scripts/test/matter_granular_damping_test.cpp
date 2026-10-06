#include "Fluid/GranularStepPolicy.h"

#include <cassert>
#include <cmath>
#include <initializer_list>

int main() {
    using RayTrophiSim::Fluid::Granular::substepDamping;
    for (const float multiplier : {0.0f, 0.5f, 0.98f, 0.999f, 1.0f}) {
        for (const int steps : {1, 2, 9, 17, 43, 128}) {
            const double substep = substepDamping(multiplier, steps);
            const double combined = std::pow(substep, steps);
            assert(std::abs(combined - multiplier) < 1e-5);
        }
    }
    assert(substepDamping(-1.0f, 17) == 0.0f);
    assert(substepDamping(2.0f, 17) == 1.0f);
    assert(substepDamping(0.98f, 0) == substepDamping(0.98f, 1));
    using RayTrophiSim::Fluid::Granular::timeScaledSubstepDamping;
    for (const float multiplier : {0.98f, 0.999f, 1.0f}) {
        const double expected = std::pow(multiplier, 60);
        for (const int outer_steps : {24, 60, 120}) {
            for (const int substeps : {1, 9, 17, 43}) {
                const double factor = timeScaledSubstepDamping(
                    multiplier, 1.0f / outer_steps, substeps);
                const double retained = std::pow(factor, outer_steps * substeps);
                assert(std::abs(retained - expected) < 1e-4);
            }
        }
    }
    assert(timeScaledSubstepDamping(0.98f, 0.0f, 17) == 1.0f);
}
