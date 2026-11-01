// Build/run in the user's standalone C++ test target, never by Codex.
#include "Fluid/MatterCommonClock.h"
#include <cassert>
#include <limits>

using namespace RayTrophiSim::Fluid;

int main() {
    std::string error;
    uint32_t steps = 0;
    MatterCommonClockHooks clock;
    clock.authored_max_substeps = 512;
    clock.contact_substeps = 512;
    assert(resolveMatterCommonSubsteps(40.0, &clock, steps, error) && steps == 64);
    clock.contact_substeps = 502;
    assert(resolveMatterCommonSubsteps(40.0, &clock, steps, error) && steps == 42);
    clock.contact_substeps = 80;
    assert(resolveMatterCommonSubsteps(40.0, &clock, steps, error) && steps == 40);
    clock.contact_substeps = 0;
    clock.minimum_substeps = 190;
    clock.authored_max_substeps = 512;
    assert(resolveMatterCommonSubsteps(16.0, &clock, steps, error) && steps == 190);
    assert(resolveMatterCommonSubsteps(235.1, &clock, steps, error) && steps == 236);
    clock.authored_max_substeps = 191;
    assert(!resolveMatterCommonSubsteps(191.0, &clock, steps, error));
    clock.authored_max_substeps = 0;
    assert(resolveMatterCommonSubsteps(100000.0, &clock, steps, error) && steps == 100000);
    assert(!resolveMatterCommonSubsteps(std::numeric_limits<double>::infinity(),
        &clock, steps, error));
    assert(!resolveMatterCommonSubsteps(std::numeric_limits<double>::quiet_NaN(),
        &clock, steps, error));
    assert(!resolveMatterCommonSubsteps(-1.0, &clock, steps, error));
    assert(!resolveMatterCommonSubsteps(double(std::numeric_limits<int>::max()),
        &clock, steps, error));

    // Independent exhaustive oracle: select the first even continuum count
    // whose rounded DEM product fits. No CFL interval may become longer.
    for (uint32_t limit = 2; limit <= 128; ++limit) {
        for (uint32_t contact = 2; contact <= limit; contact += 2) {
            for (uint32_t requested = 2; requested <= limit; requested += 2) {
                uint32_t expected = requested;
                while (expected <= limit &&
                       expected * ((contact + expected - 1) / expected) > limit) {
                    expected += 2;
                }
                uint32_t actual = requested;
                assert(fitMatterContactSubsteps(contact, limit, actual, error));
                assert(actual == expected && actual >= requested);
                assert(actual * ((contact + actual - 1) / actual) <= limit);
            }
        }
    }
    uint32_t invalid = 0;
    assert(!fitMatterContactSubsteps(8, 16, invalid, error));
    invalid = 4;
    assert(!fitMatterContactSubsteps(9, 16, invalid, error));
    assert(invalid == 4);

    // Multirate grid refresh never exceeds the requested CFL interval. Contact
    // and both geometries still advance every micro tick, exactly one frame.
    for (uint32_t n : {2u, 190u, 236u, 10000u}) {
        for (uint32_t request : {1u, 2u, 16u, 189u}) {
            if (request > n) {
                continue;
            }
            const uint32_t stride = std::max(1u, n / request);
            uint32_t covered = 0, refreshes = 0;
            for (uint32_t i = 0; i < n; i += stride) {
                const auto length = std::min(stride, n - i);
                assert(double(length) / n <= 1.0 / request + 1e-12);
                covered += length;
                ++refreshes;
            }
            assert(covered == n && refreshes >= request);
            assert(refreshes == 1u + (n - 1u) / stride);
        }
    }
}
