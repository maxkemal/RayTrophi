// User-run C++17 test. Link VolumeMaterialAuthoring.cpp and the normal Vec3.cpp.
#include "VolumeMaterialAuthoring.h"

#include <cassert>
#include <limits>

int main() {
    VolumeShader shader;
    shader.density.multiplier = 2.5f;
    shader.density.channel = "custom_density";
    shader.emission.temperature_channel = "heat";
    std::string error;
    assert(volume_material::configure(shader, {{"preset", "physical_cloud"}}, error));
    assert(shader.emission.mode == VolumeEmissionMode::None);
    assert(shader.scattering.random_walk);
    assert(shader.scattering.color.x == 1.0f);
    assert(shader.absorption.coefficient < 0.01f);
    assert(shader.scattering.random_walk_exact_events ==
           shader.scattering.random_walk_max_events);
    assert(shader.density.multiplier == 2.5f);
    assert(shader.density.channel == "custom_density");
    assert(shader.emission.temperature_channel == "heat");

    assert(volume_material::configure(shader,
        {{"preset", "artistic_fire"}, {"emission_exposure_ev", 2.0f}}, error));
    assert(shader.emission.color_ramp.stops.size() == 8);
    assert(shader.emission_response.enabled);
    assert(shader.emission_response.exposure_ev == 2.0f);
    assert(shader.emission_response.temperature_power == 1.5f);
    assert(shader.quality.internal_light_steps == 0);
    assert(shader.quality.internal_light_stride == 4);
    assert(!shader.scattering.random_walk);
    assert(volume_material::configure(shader, {{"density_curve", {
        {"enabled", true}, {"values", {0, 0.03, 0.1, 0.25, 0.45, 0.65, 0.85, 1}}
    }}}, error));
    assert(shader.density.curve.evaluate(0) == 0);
    assert(shader.density.curve.evaluate(0.25f) < 0.25f);
    float packed[12];
    shader.density.curve.pack(packed);
    assert(packed[0] == 1 && packed[1] == 0 && packed[8] == 1);
    for (int i = 1; i <= 200; ++i) {
        assert(shader.density.curve.evaluate(float(i)/100) >=
            shader.density.curve.evaluate(float(i-1)/100));
    }
    const auto saved = shader.toJson();
    const auto roundTrip = volume_material::inspect(shader);
    assert(volume_material::configure(shader, roundTrip, error));
    assert(shader.toJson() == saved);
    VolumeShader loaded;
    loaded.fromJson(saved);
    assert(loaded.density.curve.toJson() == shader.density.curve.toJson());
    assert(loaded.emission_response.toJson() == shader.emission_response.toJson());
    auto legacy = saved;
    legacy.erase("emission_response");
    loaded.fromJson(legacy);
    assert(!loaded.emission_response.enabled);

    const auto before = shader.toJson();
    const nlohmann::json invalid[] = {
        {{"preset", "physical_cloud"}, {"density_power", 5.0f}},
        {{"density_curve", {{"enabled", true}, {"values", {0, .2, .1, .3, .4, .5, .6, 1}}}}},
        {{"density_curve", {{"enabled", true}, {"values", {.1, .2, .3, .4, .5, .6, .7, 1}}}}},
        {{"density_curve", {{"enabled", true}, {"values", {0, 1}}}}},
        {{"temperature_power", -1.0f}}, {{"emission_saturation", true}},
        {{"scatter_color", {1.0f, 2.0f, 1.0f}}},
        {{"density_remap_high", 0.0f}}, {{"temperature_max", 0.0f}},
        {{"max_scatter_events", 8.5f}}, {{"unknown", 1}},
        {{"internal_light_steps", 33}}, {{"internal_light_stride", 0}},
        {{"emission_exposure_ev", std::numeric_limits<float>::infinity()}},
        {{"color_ramp", {{0.5f, 1, 0, 0}, {0.5f, 0, 1, 0}}}}
    };
    for (const auto& patch : invalid) {
        assert(!volume_material::configure(shader, patch, error));
        assert(!error.empty());
        assert(shader.toJson() == before);
    }
    return 0;
}
