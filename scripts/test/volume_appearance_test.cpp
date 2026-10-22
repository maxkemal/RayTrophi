// User-run numerical regression; link the project's Vec3 implementation.
#include "VolumeAppearance.h"
#include "Atmosphere/AtmosphereClouds.h"

#include <cassert>
#include <cmath>

namespace {

void near(float actual, float expected) {
    assert(std::abs(actual - expected) < 1.0e-5f);
}

} // namespace

int main() {
    const Vec3 color(0.8f, 0.2f, 0.1f);
    const Vec3 neutral = volume_appearance::scatterColor(color, 1.0f);
    assert(neutral.x == color.x && neutral.y == color.y && neutral.z == color.z);

    const Vec3 grey = volume_appearance::scatterColor(color, 0.0f);
    near(grey.x, 0.32034f);
    near(grey.y, grey.x);
    near(grey.z, grey.x);

    const Vec3 saturated = volume_appearance::scatterColor(Vec3(1.0f, 0.0f, 0.0f), 2.0f);
    assert(saturated.x == 1.0f && saturated.y == 0.0f && saturated.z == 0.0f);

    const Vec3 emitted = volume_appearance::emissionColor(color, 1.0f, 1.0f);
    near(emitted.x, 1.6f);
    near(emitted.y, 0.4f);
    near(emitted.z, 0.2f);
    const Vec3 dimmed = volume_appearance::emissionColor(color, 1.0f, -1.0f);
    near(dimmed.x, 0.4f);
    near(dimmed.y, 0.1f);

    const Vec3 hdr = volume_appearance::emissionColor(Vec3(8.0f, 2.0f, 1.0f), 1.0f, 0.0f);
    assert(hdr.x == 8.0f && hdr.y == 2.0f && hdr.z == 1.0f);
    const Vec3 black = volume_appearance::emissionColor(Vec3(0.0f), 2.0f, 16.0f);
    assert(black.x == 0.0f && black.y == 0.0f && black.z == 0.0f);

    atmosphere::CloudState state;
    state.material.scattering_color = color;
    state.material.scattering_saturation = 0.0f;
    state.material.emission_color = Vec3(1.0f, 0.0f, 0.0f);
    state.material.emission_strength = 2.0f;
    state.material.emission_exposure_ev = 1.0f;
    const auto master = atmosphere::resolvedCloudMaterial(state.material);
    const auto twice = atmosphere::resolvedCloudMaterial(master);
    near(master.scattering_color.x, grey.x);
    assert(twice.emission_color.x == master.emission_color.x);
    assert(twice.scattering_color.x == master.scattering_color.x);

    auto& layer = state.layers[0];
    layer.use_custom_material = true;
    layer.material.scattering_color = Vec3(0.5f);
    layer.material.emission_color = Vec3(0.0f, 1.0f, 0.0f);
    layer.material.emission_strength = 3.0f;
    layer.material.emission_exposure_ev = -1.0f;
    const auto combined = atmosphere::layerMaterial(state, 0);
    near(combined.scattering_color.x, grey.x * 0.5f);
    near(combined.emission_color.x * combined.emission_strength, 4.0f);
    near(combined.emission_color.y * combined.emission_strength, 1.5f);
    // Resolution must not modify the authored state or apply master exposure
    // to the local emission contribution.
    assert(state.material.emission_color.x == 1.0f);
    assert(state.material.emission_exposure_ev == 1.0f);
    return 0;
}
