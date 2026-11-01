#include "VolumeMaterialAuthoring.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace volume_material {
namespace {

float number(const nlohmann::json& value, float lo, float hi, const std::string& key) {
    if (!value.is_number()) {
        throw std::invalid_argument(key + " must be numeric");
    }
    const float result = value.get<float>();
    if (!std::isfinite(result) || result < lo || result > hi) {
        throw std::invalid_argument(key + " is non-finite or outside its allowed range");
    }
    return result;
}

Vec3 color(const nlohmann::json& value, float hi, const std::string& key) {
    if (!value.is_array() || value.size() != 3) {
        throw std::invalid_argument(key + " must be [r,g,b]");
    }
    return Vec3(number(value[0], 0.0f, hi, key), number(value[1], 0.0f, hi, key),
                number(value[2], 0.0f, hi, key));
}

bool boolean(const nlohmann::json& value, const std::string& key) {
    if (!value.is_boolean()) {
        throw std::invalid_argument(key + " must be boolean");
    }
    return value.get<bool>();
}

int integer(const nlohmann::json& value, int lo, int hi, const std::string& key) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument(key + " must be an integer");
    }
    return static_cast<int>(number(value, static_cast<float>(lo), static_cast<float>(hi), key));
}

void preset(VolumeShader& shader, const std::string& name) {
    const int temperatureUnits = shader.emission_response.temperature_units;
    const auto previousResponse = shader.emission_response;
    if (name == "physical_cloud") {
        shader.scattering.color = Vec3(1.0f);
        shader.scattering.coefficient = 1.0f;
        shader.scattering.anisotropy = 0.75f;
        shader.scattering.anisotropy_back = -0.3f;
        shader.scattering.lobe_mix = 1.0f;
        shader.scattering.multi_scatter = 0.8f;
        shader.scattering.random_walk = true;
        shader.scattering.random_walk_max_events = 128;
        shader.scattering.random_walk_exact_events = 128;
        shader.absorption.color = Vec3(0.0f);
        shader.absorption.coefficient = 0.001f;
        shader.emission.mode = VolumeEmissionMode::None;
        shader.emission.intensity = 0.0f;
        shader.emission_response = VolumeEmissionResponse{};
        shader.quality.shadow_strength = 1.0f;
        shader.quality.internal_light_steps = 0;
        shader.quality.shadow_steps = 24;
        shader.quality.shadow_stride = 1;
        shader.quality.max_steps = 1024;
        shader.quality.adaptive_stepping = true;
        shader.quality.voxel_step_multiplier = 0.5f;
    } else if (name == "blackbody_fire" || name == "artistic_fire") {
        shader.scattering.random_walk = false;
        shader.emission.mode = VolumeEmissionMode::Blackbody;
        shader.emission.temperature_scale = 1.0f;
        shader.emission.blackbody_intensity = 15.0f;
        shader.emission_response = VolumeEmissionResponse{};
        shader.emission_response.enabled = true;
        shader.emission_response.temperature_power = name == "blackbody_fire" ? 4.0f : 1.5f;
        shader.emission_response.density_power = name == "blackbody_fire" ? 0.0f : 1.0f;
        shader.emission.color_ramp.enabled = name == "artistic_fire";
        shader.quality.internal_light_steps = 0;
        shader.quality.internal_light_stride = 4;
        shader.emission.color_ramp.stops = {
            {0.0f, Vec3(0.0f), 0.0f},
            {0.12f, Vec3(0.10f, 0.002f, 0.0f), 1.0f},
            {0.28f, Vec3(0.55f, 0.025f, 0.001f), 1.0f},
            {0.45f, Vec3(1.0f, 0.12f, 0.005f), 1.0f},
            {0.62f, Vec3(1.0f, 0.32f, 0.025f), 1.0f},
            {0.78f, Vec3(1.0f, 0.60f, 0.12f), 1.0f},
            {0.90f, Vec3(1.0f, 0.82f, 0.42f), 1.0f},
            {1.0f, Vec3(1.0f, 0.96f, 0.80f), 1.0f}
        };
    } else {
        throw std::invalid_argument("unknown preset: " + name);
    }
    shader.emission_response.temperature_units = temperatureUnits;
    shader.emission_response.mask_enabled = previousResponse.mask_enabled;
    shader.emission_response.mask_center = previousResponse.mask_center;
    shader.emission_response.mask_width = previousResponse.mask_width;
    shader.emission_response.mask_strength = previousResponse.mask_strength;
    shader.emission_response.independent_emission = previousResponse.independent_emission;
    shader.emission_response.soot_burnout = previousResponse.soot_burnout;
    shader.emission_response.soot_burnout_start = previousResponse.soot_burnout_start;
    shader.emission_response.soot_burnout_end = previousResponse.soot_burnout_end;
    shader.emission_response.temperature_gamma = previousResponse.temperature_gamma;
    shader.emission_response.flame_emission = previousResponse.flame_emission;
}

} // namespace

SmokeAppearance smokeAppearance(const VolumeShader& s) {
    const float scatter = (std::max)(s.scattering.coefficient, 0.0f);
    const float absorption = (std::max)(s.absorption.coefficient, 0.0f);
    SmokeAppearance out;
    out.opacity = scatter + absorption;
    const float share = out.opacity > 0.0f ? scatter / out.opacity : 1.0f;
    out.color = s.scattering.color * share;
    const Vec3& tint = s.absorption.color;
    const float peak = (std::max)({tint.x, tint.y, tint.z});
    // Black absorption means neutral in the renderers.
    out.tint = peak > 1e-5f ? tint / peak : Vec3(1.0f, 1.0f, 1.0f);
    return out;
}

void applySmokeAppearance(VolumeShader& s, const SmokeAppearance& appearance) {
    const Vec3& c = appearance.color;
    const float peak = (std::max)({c.x, c.y, c.z});
    s.scattering.coefficient = appearance.opacity * peak;
    s.absorption.coefficient = appearance.opacity * (1.0f - peak);
    // Extinction ignores scatter color, so albedo per channel is exactly c.
    s.scattering.color = peak > 1e-5f ? c / peak : Vec3(1.0f, 1.0f, 1.0f);
    const Vec3& tint = appearance.tint;
    const float tintPeak = (std::max)({tint.x, tint.y, tint.z});
    s.absorption.color = tintPeak > 1e-5f ? tint / tintPeak : Vec3(1.0f, 1.0f, 1.0f);
}

nlohmann::json inspect(const VolumeShader& s) {
    nlohmann::json ramp = nlohmann::json::array();
    for (const auto& stop : s.emission.color_ramp.stops) {
        ramp.push_back({stop.position, stop.color.x, stop.color.y, stop.color.z, stop.alpha});
    }
    const SmokeAppearance appearance = smokeAppearance(s);
    return {{"temperature_units", s.emission_response.temperature_units},
        {"density_multiplier", s.density.multiplier},
        {"density_cutoff", s.density.cutoff_threshold},
        {"density_curve", s.density.curve.toJson()},
        {"density_remap_low", s.density.remap_low}, {"density_remap_high", s.density.remap_high},
        {"shadow_strength", s.quality.shadow_strength},
        {"shadow_steps", s.quality.shadow_steps}, {"shadow_stride", s.quality.shadow_stride},
        {"internal_light_steps", s.quality.internal_light_steps},
        {"internal_light_stride", s.quality.internal_light_stride},
        {"max_steps", s.quality.max_steps},
        {"adaptive_stepping", s.quality.adaptive_stepping},
        {"voxel_step_multiplier", s.quality.voxel_step_multiplier},
        {"scatter_color", {s.scattering.color.x, s.scattering.color.y, s.scattering.color.z}},
        {"scatter_coefficient", s.scattering.coefficient},
        {"anisotropy", s.scattering.anisotropy}, {"back_anisotropy", s.scattering.anisotropy_back},
        {"lobe_mix", s.scattering.lobe_mix}, {"multi_scatter", s.scattering.multi_scatter},
        {"random_walk", s.scattering.random_walk},
        {"max_scatter_events", s.scattering.random_walk_max_events},
        {"exact_scatter_events", s.scattering.random_walk_exact_events},
        {"absorption_coefficient", s.absorption.coefficient},
        {"absorption_color", {s.absorption.color.x, s.absorption.color.y, s.absorption.color.z}},
        {"emission_mode", static_cast<int>(s.emission.mode)},
        {"emission_strength", s.emission.intensity},
        {"emission_color", {s.emission.color.x, s.emission.color.y, s.emission.color.z}},
        {"blackbody_intensity", s.emission.blackbody_intensity},
        {"temperature_min", s.emission.temperature_min},
        {"temperature_max", s.emission.temperature_max},
        {"temperature_scale", s.emission.temperature_scale},
        {"response_enabled", s.emission_response.enabled},
        {"temperature_power", s.emission_response.temperature_power},
        {"density_power", s.emission_response.density_power},
        {"emission_exposure_ev", s.emission_response.exposure_ev},
        {"emission_saturation", s.emission_response.saturation},
        {"emission_mask_enabled", s.emission_response.mask_enabled},
        {"emission_mask_center", s.emission_response.mask_center},
        {"emission_mask_width", s.emission_response.mask_width},
        {"emission_mask_strength", s.emission_response.mask_strength},
        {"independent_emission", s.emission_response.independent_emission},
        {"soot_burnout", s.emission_response.soot_burnout},
        {"soot_burnout_start", s.emission_response.soot_burnout_start},
        {"soot_burnout_end", s.emission_response.soot_burnout_end},
        {"temperature_gamma", s.emission_response.temperature_gamma},
        {"flame_emission", s.emission_response.flame_emission},
        {"smoke_color", {appearance.color.x, appearance.color.y, appearance.color.z}},
        {"smoke_opacity", appearance.opacity},
        {"smoke_tint", {appearance.tint.x, appearance.tint.y, appearance.tint.z}},
        {"ramp_enabled", s.emission.color_ramp.enabled}, {"color_ramp", ramp}};
}

bool configure(VolumeShader& shader, const nlohmann::json& patch, std::string& error) {
    try {
        if (!patch.is_object()) {
            throw std::invalid_argument("material settings must be an object");
        }
        VolumeShader next = shader;
        if (patch.contains("preset")) {
            preset(next, patch.at("preset").get<std::string>());
        }
        for (auto it = patch.begin(); it != patch.end(); ++it) {
            const auto& k = it.key();
            const auto& v = it.value();
            if (k == "preset") {
                continue;
            }
            if (k == "density_multiplier") {
                next.density.multiplier = number(v, 0, 10000, k);
            } else if (k == "shadow_strength") {
                next.quality.shadow_strength = number(v, 0, 1, k);
            } else if (k == "shadow_steps") {
                next.quality.shadow_steps = integer(v, 0, 48, k);
            } else if (k == "shadow_stride") {
                next.quality.shadow_stride = integer(v, 1, 16, k);
            } else if (k == "internal_light_steps") {
                next.quality.internal_light_steps = integer(v, 0, 32, k);
            } else if (k == "internal_light_stride") {
                next.quality.internal_light_stride = integer(v, 1, 32, k);
            } else if (k == "max_steps") {
                next.quality.max_steps = integer(v, 16, 1024, k);
            } else if (k == "voxel_step_multiplier") {
                next.quality.voxel_step_multiplier = number(v, 0.1f, 8, k);
            } else if (k == "adaptive_stepping") {
                next.quality.adaptive_stepping = boolean(v, k);
            } else if (k == "density_cutoff") {
                next.density.cutoff_threshold = number(v, 0, 1, k);
            } else if (k == "density_curve") {
                next.density.curve.fromJson(v);
            } else if (k == "density_remap_low") {
                next.density.remap_low = number(v, 0, 10000, k);
            } else if (k == "density_remap_high") {
                next.density.remap_high = number(v, 0, 10000, k);
            } else if (k == "scatter_color") {
                next.scattering.color = color(v, 1, k);
            } else if (k == "scatter_coefficient") {
                next.scattering.coefficient = number(v, 0, 10000, k);
            } else if (k == "anisotropy") {
                next.scattering.anisotropy = number(v, -0.99f, 0.99f, k);
            } else if (k == "back_anisotropy") {
                next.scattering.anisotropy_back = number(v, -0.99f, 0, k);
            } else if (k == "lobe_mix") {
                next.scattering.lobe_mix = number(v, 0, 1, k);
            } else if (k == "multi_scatter") {
                next.scattering.multi_scatter = number(v, 0, 1, k);
            } else if (k == "random_walk") {
                next.scattering.random_walk = boolean(v, k);
            } else if (k == "max_scatter_events") {
                next.scattering.random_walk_max_events = integer(v, 8, 512, k);
            } else if (k == "exact_scatter_events") {
                next.scattering.random_walk_exact_events = integer(v, 1, 512, k);
            } else if (k == "absorption_coefficient") {
                next.absorption.coefficient = number(v, 0, 10000, k);
            } else if (k == "absorption_color") {
                next.absorption.color = color(v, 1, k);
            } else if (k == "emission_mode") {
                next.emission.mode = VolumeEmissionMode(integer(v, 0, 3, k));
            } else if (k == "emission_strength") {
                next.emission.intensity = number(v, 0, 100000, k);
            } else if (k == "emission_color") {
                next.emission.color = color(v, 10000, k);
            } else if (k == "blackbody_intensity") {
                next.emission.blackbody_intensity = number(v, 0, 100000, k);
            } else if (k == "temperature_min") {
                next.emission.temperature_min = number(v, 0, 100000, k);
            } else if (k == "temperature_max") {
                next.emission.temperature_max = number(v, 0, 100000, k);
            } else if (k == "temperature_scale") {
                next.emission.temperature_scale = number(v, 0.001f, 100, k);
            } else if (k == "response_enabled") {
                next.emission_response.enabled = boolean(v, k);
            } else if (k == "temperature_units") {
                next.emission_response.temperature_units = integer(v, 0, 2, k);
            } else if (k == "temperature_power") {
                next.emission_response.temperature_power = number(v, 0, 8, k);
            } else if (k == "density_power") {
                next.emission_response.density_power = number(v, 0, 4, k);
            } else if (k == "emission_exposure_ev") {
                next.emission_response.exposure_ev = number(v, -16, 16, k);
            } else if (k == "emission_saturation") {
                next.emission_response.saturation = number(v, 0, 2, k);
            } else if (k == "independent_emission") {
                next.emission_response.independent_emission = boolean(v, k);
            } else if (k == "soot_burnout") {
                next.emission_response.soot_burnout = number(v, 0, 1, k);
            } else if (k == "soot_burnout_start") {
                next.emission_response.soot_burnout_start = number(v, 0, 1, k);
            } else if (k == "soot_burnout_end") {
                next.emission_response.soot_burnout_end = number(v, 0, 1, k);
            } else if (k == "temperature_gamma") {
                next.emission_response.temperature_gamma = number(v, 0.1f, 8, k);
            } else if (k == "flame_emission") {
                next.emission_response.flame_emission = number(v, 0, 100000, k);
            } else if (k == "smoke_color" || k == "smoke_opacity" || k == "smoke_tint") {
                // Applied after the loop: three keys of one appearance.
            } else if (k == "emission_mask_enabled") {
                next.emission_response.mask_enabled = boolean(v, k);
            } else if (k == "emission_mask_center") {
                next.emission_response.mask_center = number(v, 0, 100000, k);
            } else if (k == "emission_mask_width") {
                next.emission_response.mask_width = number(v, 1e-6f, 100000, k);
            } else if (k == "emission_mask_strength") {
                next.emission_response.mask_strength = number(v, 0, 1, k);
            } else if (k == "ramp_enabled") {
                next.emission.color_ramp.enabled = boolean(v, k);
            } else if (k == "color_ramp") {
                if (!v.is_array() || v.size() < 2 || v.size() > 256) {
                    throw std::invalid_argument(
                        "color_ramp requires 2..256 [position,r,g,b] stops");
                }
                next.emission.color_ramp.stops.clear();
                float previous = -1.0f;
                for (const auto& stop : v) {
                    if (!stop.is_array() || (stop.size() != 4 && stop.size() != 5)) {
                        throw std::invalid_argument(
                            "ramp stop must be [position,r,g,b,optional alpha]");
                    }
                    const float p = number(stop[0], 0, 1, k);
                    if (p <= previous) {
                        throw std::invalid_argument("ramp positions must be strictly increasing");
                    }
                    next.emission.color_ramp.stops.push_back({p,
                        Vec3(number(stop[1], 0, 10000, k), number(stop[2], 0, 10000, k),
                             number(stop[3], 0, 10000, k)),
                        stop.size() == 5 ? number(stop[4], 0, 1, "ramp alpha") : 1.0f});
                    previous = p;
                }
            } else {
                throw std::invalid_argument("unknown volume material setting: " + k);
            }
        }
        const bool smokeKeys = patch.contains("smoke_color") ||
            patch.contains("smoke_opacity") || patch.contains("smoke_tint");
        if (smokeKeys) {
            if (patch.contains("scatter_color") || patch.contains("scatter_coefficient") ||
                patch.contains("absorption_color") || patch.contains("absorption_coefficient")) {
                throw std::invalid_argument(
                    "smoke_* keys cannot be combined with raw scatter or absorption keys");
            }
            SmokeAppearance appearance = smokeAppearance(next);
            if (patch.contains("smoke_color")) {
                appearance.color = color(patch.at("smoke_color"), 1, "smoke_color");
            }
            if (patch.contains("smoke_opacity")) {
                appearance.opacity = number(patch.at("smoke_opacity"), 0, 10000, "smoke_opacity");
            }
            if (patch.contains("smoke_tint")) {
                appearance.tint = color(patch.at("smoke_tint"), 1, "smoke_tint");
            }
            applySmokeAppearance(next, appearance);
        }
        if (next.density.remap_high <= next.density.remap_low ||
            next.emission.temperature_max <= next.emission.temperature_min) {
            throw std::invalid_argument("density and temperature ranges must have max > min");
        }
        if (next.emission_response.soot_burnout_end <= next.emission_response.soot_burnout_start) {
            throw std::invalid_argument("soot_burnout_end must be above soot_burnout_start");
        }
        shader = std::move(next);
        error.clear();
        return true;
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }
}

} // namespace volume_material
