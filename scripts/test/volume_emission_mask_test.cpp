// Standalone shared-core regression test. Build/run by the user, never by the agent.
#include "../../RayTrophiStudio/source/include/VolumeEmissionMask.h"
#include <cassert>
#include <cmath>
#include <initializer_list>

int main() {
    float mask[4] = {1.0f, 0.2f, 0.1f, 1.0f};
    assert(std::abs(volumeEmissionMaskFactor(0.15f, 1.0f, mask) - 1.0f) < 1e-5f);
    assert(std::abs(volumeEmissionMaskFactor(0.2f, 1.0f, mask) - 0.5f) < 1e-5f);
    assert(std::abs(volumeEmissionMaskFactor(0.25f, 1.0f, mask)) < 1e-5f);
    float previous = 1.0f;
    for (int i = 0; i <= 1000; ++i) {
        float density = float(i) * 0.001f;
        float weight = volumeEmissionMaskFactor(density, 1.0f, mask);
        assert(weight >= 0.0f && weight <= previous + 1e-6f);
        previous = weight;
        for (float multiplier : {0.1f, 1.0f, 10.417f, 13.194f, 100.0f}) {
            float scaled = volumeEmissionMaskFactor(density * multiplier, multiplier, mask);
            assert(std::abs(weight - scaled) < 1e-5f);
        }
    }
    mask[3] = 0.4f;
    assert(std::abs(volumeEmissionMaskFactor(1.0f, 1.0f, mask) - 0.6f) < 1e-5f);
    mask[3] = 0.0f;
    assert(volumeEmissionMaskFactor(1.0f, 1.0f, mask) == 1.0f);
    mask[0] = 0.0f;
    mask[3] = 1.0f;
    assert(volumeEmissionMaskFactor(100.0f, 1.0f, mask) == 1.0f);
    // Masking the physical source precedes the display/radiance limit.
    mask[0] = 1.0f;
    float source = 100.0f;
    float weight = volumeEmissionMaskFactor(0.2f, 1.0f, mask);
    assert(std::abs(std::fmin(source * weight, 64.0f) - 50.0f) < 1e-3f);
    assert(std::abs(std::fmin(source, 64.0f) * weight - 32.0f) < 1e-3f);
}
