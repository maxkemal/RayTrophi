// User-run C++17 scalar test; no project/GPU build is run by the agent.
#include "VolumeEmissionTransport.h"
#include <cassert>
#include <cmath>
#include <initializer_list>

struct Color {
    float x, y, z;
};

int main() {
    float response[5] = {3.0f, 2.0f, 1.861f, 0.0f, 1.0f};
    Color color{1.0f, 0.25f, 0.05f};
    Color reference{};
    for (float multiplier : {0.0f, 0.001f, 1.0f, 10.0f, 60.0f}) {
        float density = 0.4f * volumeEmissionSamplingMultiplier(multiplier, response[0]);
        Color j = volumeEmissionBuild(color, 1.0f,
            volumeEmissionInputDensity(density, multiplier, response[0]),
            0.7f, 10.0f, response, true);
        if (multiplier == 0.0f) reference = j;
        assert(std::abs(j.x - reference.x) < 1e-5f);
        assert(std::abs(j.y - reference.y) < 1e-5f);
        assert(std::abs(j.z - reference.z) < 1e-5f);
        float optical = volumeEmissionOpticalDensity(density, multiplier, response[0]);
        assert(multiplier != 0.0f || optical == 0.0f);
    }
    Color j{2.0f, 0.5f, 0.1f};
    Color vacuum = volumeEmissionIntegrate(j, j, 0.0f, 3.0f, 3.0f);
    assert(vacuum.x == 6.0f && vacuum.y == 1.5f);
    float previous = 1e9f;
    for (float sigma : {0.0f, 1e-8f, 0.1f, 1.0f, 100.0f, 1e8f}) {
        float expected = sigma > 0.0f ? -std::expm1(-sigma * 3.0f) / sigma : 3.0f;
        Color result = volumeEmissionIntegrate(j, j, sigma, 3.0f, 3.0f);
        assert(result.x > 0.0f);
        assert(std::abs(result.x - 2.0f * expected) < 1e-5f);
        assert(result.x <= previous);
        previous = result.x;
        for (int count : {32, 128, 256}) {
            float dt = 3.0f / float(count);
            float transmittance = 1.0f;
            float sum = 0.0f;
            for (int step = 0; step < count; ++step) {
                sum += transmittance * j.x * volumeEmissionIntegral(sigma, dt, 3.0f);
                transmittance *= std::exp(-sigma * dt);
            }
            assert(std::abs(sum - result.x) < 1e-4f);
        }
    }
    assert(!volumeEmissionIndependent(0.0f));
    assert(!volumeEmissionIndependent(1.0f));
    assert(volumeEmissionIndependent(2.0f));
    assert(volumeEmissionIndependent(3.0f));
    assert(volumeEmissionInputDensity(8.0f, 4.0f, 1.0f) == 8.0f);
    return 0;
}
