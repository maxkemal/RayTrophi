// User-run C++ numerical regression. Codex does not compile project sources.
#include "Fluid/FogOpticalBounds.h"
#include <cassert>
#include <limits>

using namespace RayTrophiSim::Fluid;

int main() {
    constexpr int n = 32;
    std::vector<float> density(n * n * n, 1e-5f);
    auto at = [&](int x, int y, int z) -> float& {
        return density[(z * n + y) * n + x];
    };
    VolumeShader shader;
    shader.density.cutoff_threshold = 0.01f;
    at(16, 16, 16) = 0.1f;
    auto bounds = fogOpticalBounds(density, n, n, n, &shader);
    assert(!bounds.empty && !bounds.conservative_fallback);
    assert(bounds.first[0] == 8 && bounds.end[0] == 24);
    // Preserve the entire trilinear support of a contributor on a block edge.
    at(16, 16, 16) = 1e-5f;
    at(7, 7, 7) = 0.1f;
    bounds = fogOpticalBounds(density, n, n, n, &shader);
    assert(bounds.first[0] == 0 && bounds.end[0] == 16);
    at(7, 7, 7) = 1e-5f;
    assert(fogOpticalBounds(density, n, n, n, &shader).empty);
    // Cutoff precedes multiplier: scaling a rejected raw value cannot revive it.
    shader.density.multiplier = 10000.0f;
    assert(fogOpticalBounds(density, n, n, n, &shader).empty);
    at(16, 16, 16) = 0.004f;
    shader.density.curve.enabled = true;
    shader.density.curve.values[1] = 0.4f;
    for (int i = 2; i < 8; ++i) shader.density.curve.values[i] = 0.5f + i * 0.05f;
    assert(!fogOpticalBounds(density, n, n, n, &shader).empty);
    const auto key = fogOpticalBoundsKey(&shader, 1.0f);
    shader.density.cutoff_threshold = 0.001f;
    assert(key != fogOpticalBoundsKey(&shader, 1.0f));
    shader.material_graph = "Spatial Density";
    bounds = fogOpticalBounds(density, n, n, n, &shader);
    assert(bounds.conservative_fallback && bounds.first[0] == 0 && bounds.end[0] == n);
    shader.material_graph.clear();
    shader.material_program.density_noise_enabled = true;
    assert(fogOpticalBounds(density, n, n, n, &shader).conservative_fallback);
    shader.material_program.density_noise_enabled = false;
    shader.density.remap_low = -0.5f;
    assert(fogOpticalBounds(density, n, n, n, &shader).conservative_fallback);
    shader.density.remap_low = 0.0f;
    shader.density.multiplier = 0.0f;
    shader.emission.mode = VolumeEmissionMode::Blackbody;
    assert(fogOpticalBounds(density, n, n, n, &shader).empty);
    const auto zeroSmokeKey = fogOpticalBoundsKey(&shader, 1.0f);
    shader.emission_response.independent_emission = true;
    assert(!fogOpticalBounds(density, n, n, n, &shader).empty);
    assert(!fogOpticalBounds(density, n, n, n, &shader, 0.0f).empty);
    assert(zeroSmokeKey != fogOpticalBoundsKey(&shader, 1.0f));
    // A zero smoke curve must not cull a pre-curve independent emitter.
    for (float& value : shader.density.curve.values) value = 0.0f;
    assert(!fogOpticalBounds(density, n, n, n, &shader).empty);
    shader.emission_response.independent_emission = false;
    assert(fogOpticalBounds(density, n, n, n, &shader).empty);
    shader.emission_response.independent_emission = true;
    at(0, 0, 0) = std::numeric_limits<float>::quiet_NaN();
    assert(fogOpticalBounds(density, n, n, n, &shader).conservative_fallback);
    return 0;
}
