// Measured distribution of a live gas field through the authored emission
// chain. Peak and mean cannot show where a transfer function squeezes the data,
// and the fog inspector refuses large domains on its memory budget, so this
// walks the host grid in place and keeps one float per active cell per channel.
#include "RtApiInternal.h"
#include "ParticleSimulation.h"
#include "SimulationDomainLanes.h"
#include "VolumeMaterialAuthoring.h"
#include "VolumeMaterialService.h"
#include "VolumeShader.h"
#include "VolumeTemperatureMapping.h"
#include <algorithm>
#include <cmath>

namespace rtapi {

namespace {

// Same conversion as the fog publishers: solver heat to Kelvin.
constexpr float kHeatToKelvin = 3000.0f;

GasFieldPercentiles percentiles(std::vector<float>& values) {
    GasFieldPercentiles out;
    if (values.empty()) return out;
    std::sort(values.begin(), values.end());
    const auto at = [&](double p) {
        return values[static_cast<std::size_t>(p * static_cast<double>(values.size() - 1) + 0.5)];
    };
    double sum = 0.0;
    for (const float value : values) sum += static_cast<double>(value);
    out.mean = static_cast<float>(sum / static_cast<double>(values.size()));
    out.minimum = values.front();
    out.maximum = values.back();
    out.p1 = at(0.01);
    out.p5 = at(0.05);
    out.p25 = at(0.25);
    out.p50 = at(0.50);
    out.p75 = at(0.75);
    out.p95 = at(0.95);
    out.p99 = at(0.99);
    return out;
}

std::vector<uint64_t> histogram(const std::vector<float>& sorted, const GasFieldPercentiles& range,
                                int bins) {
    std::vector<uint64_t> out(static_cast<std::size_t>(bins), 0u);
    const float width = range.maximum - range.minimum;
    if (sorted.empty() || !(width > 0.0f)) return out;
    for (const float value : sorted) {
        const int bin = static_cast<int>((value - range.minimum) / width * static_cast<float>(bins));
        ++out[static_cast<std::size_t>(std::min(bin, bins - 1))];
    }
    return out;
}

float densityCutoff(float density, float cutoff) {
    if (density <= cutoff) return 0.0f;
    if (cutoff <= 0.0f) return density;
    const float x = std::clamp((density - cutoff) / cutoff, 0.0f, 1.0f);
    return density * (x * x * (3.0f - 2.0f * x));
}

struct GasFieldView {
    const RayTrophiSim::SimulationGridDomainDesc* desc = nullptr;
    const RayTrophiSim::SimulationGridDomainState* state = nullptr;
};

// Prefers a stepped domain over a merely declared one, like measureGasPlume.
GasFieldView resolve(const std::string& name) {
    GasFieldView view;
    auto consider = [&](RayTrophiSim::ParticleSimulationSystem& system) {
        const auto& domains = system.gridDomains();
        const auto& states = system.gridDomainStates();
        for (std::size_t i = 0; i < domains.size(); ++i) {
            if (domains[i].name != name) continue;
            const bool live = i < states.size() && states[i].valid;
            if (view.desc && !live) continue;
            view.desc = &domains[i];
            view.state = live ? &states[i] : nullptr;
            if (live) return true;
        }
        return false;
    };
    for (auto& system : g_ctx->scene.particle_systems) {
        if (system.runtime && consider(*system.runtime)) return view;
    }
    if (!view.desc) consider(g_ctx->scene.ensureParticleSimulationSystem());
    return view;
}

bool thermalChain(const VolumeShader* shader) {
    return shader && shader->emission_response.independent_emission &&
        shader->emission.color_ramp.enabled && !shader->emission.color_ramp.stops.empty() &&
        (shader->emission.mode == VolumeEmissionMode::Blackbody ||
         shader->emission.mode == VolumeEmissionMode::ChannelDriven);
}

} // namespace

Result measureGasFields(const std::string& domain, float density_threshold, int bins,
                        GasFieldStats& out) {
    out = GasFieldStats{};
    out.threshold = density_threshold > 0.0f ? density_threshold : 0.01f;
    if (!g_ctx) return notBound();
    if (bins < 2) return Result::fail("bins must be at least 2");
    const GasFieldView view = resolve(domain);
    if (!view.desc) return Result::fail("gas domain not found: " + domain);
    if (!RayTrophiSim::simulationDomainHasGas(*view.desc)) {
        return Result::fail("not a gas domain: " + domain);
    }
    if (!view.state) return Result::success();
    const auto& grid = view.state->grid;
    const std::size_t cells = static_cast<std::size_t>(grid.nx) *
        static_cast<std::size_t>(grid.ny) * static_cast<std::size_t>(grid.nz);
    if (cells == 0u || grid.density.size() != cells) return Result::success();
    const bool hasTemperature = grid.temperature.size() == cells;
    const VolumeShader* shader = view.desc->shader.get();
    const bool thermal = hasTemperature && thermalChain(shader);

    const bool hasFlame = grid.interaction.size() == cells;
    std::vector<float> density, kelvin, flame, ramp, luminance, depth;
    float response[5] = {};
    float low = 0.0f, high = 1.0f, extinction = 0.0f, scale = 1.0f;
    float multiplier = 1.0f;
    if (thermal) {
        const auto& r = shader->emission_response;
        r.pack(response);
        low = std::max(shader->emission.temperature_min, 0.0f);
        high = shader->emission.temperature_max > low + 1.0f
            ? shader->emission.temperature_max : low + 1500.0f;
        scale = shader->emission.temperature_scale;
        multiplier = shader->density.multiplier;
        // The strongest RGB channel: absorption color is normalized by its peak.
        extinction = shader->scattering.coefficient + shader->absorption.coefficient;
    }
    for (std::size_t c = 0; c < cells; ++c) {
        const float d = grid.density[c];
        if (!(d > out.threshold)) continue;
        density.push_back(d);
        const float burning = hasFlame ? std::clamp(grid.interaction[c], 0.0f, 1.0f) : 0.0f;
        if (burning > 1e-3f) flame.push_back(burning);
        if (!hasTemperature) continue;
        const float k = grid.temperature[c] * kHeatToKelvin;
        kelvin.push_back(k);
        if (!thermal) continue;
        const auto& r = shader->emission_response;
        const float range = std::max(shader->density.remap_high - shader->density.remap_low, 1e-6f);
        const float normalized = std::max((d - shader->density.remap_low) / range, 0.0f);
        const float cutoff = shader->density.cutoff_threshold;
        const float source = densityCutoff(normalized, cutoff);
        const float optical = densityCutoff(shader->density.curve.evaluate(normalized), cutoff) *
            multiplier;
        const VolumeTemperatureSample mapped =
            volumeMapTemperature(k, source, low, high, 1, true);
        float coordinate = 0.0f;
        float emitted = 0.0f;
        if (mapped.valid) {
            const float u = std::pow(mapped.normalized, r.temperature_gamma);
            coordinate = std::clamp(u * scale, 0.0f, 1.0f);
            const Vec3 color = shader->emission.color_ramp.sample(coordinate);
            Vec3 j = volumeEmissionBuild(color, 1.0f, source, u,
                shader->emission.blackbody_intensity, response, true);
            const float mask = r.maskFactor(source * multiplier, multiplier);
            emitted = (j.x * 0.2126f + j.y * 0.7152f + j.z * 0.0722f) * mask;
            if (r.flame_emission > 0.0f && burning > 0.0f) {
                const Vec3 flameColor = shader->emission.color_ramp.sample(
                    std::max(coordinate, burning));
                emitted += (flameColor.x * 0.2126f + flameColor.y * 0.7152f +
                    flameColor.z * 0.0722f) * mask * volumeFlameRadiance(burning,
                        r.flame_emission, shader->emission.blackbody_intensity, response);
            }
        }
        ramp.push_back(coordinate);
        luminance.push_back(emitted);
        if (emitted > 64.0f) ++out.limited_cells;
        if (emitted > 0.0f) ++out.emitting_cells;
        depth.push_back(optical * extinction * grid.voxel_size * std::min(
            volumeSootExtinctionScale(coordinate, r.soot_burnout, r.soot_burnout_start,
                r.soot_burnout_end, r.policyFlags()),
            1.0f - r.soot_burnout * burning));
    }
    out.measured = true;
    out.active_cells = density.size();
    out.density = percentiles(density);
    out.density_histogram = histogram(density, out.density, bins);
    if (hasFlame) {
        out.flame_measured = true;
        out.burning_cells = flame.size();
        out.flame = percentiles(flame);
    }
    if (hasTemperature) {
        out.temperature_measured = true;
        out.temperature_kelvin = percentiles(kelvin);
        out.temperature_histogram = histogram(kelvin, out.temperature_kelvin, bins);
    }
    if (thermal) {
        out.emission_measured = true;
        out.ramp_coordinate = percentiles(ramp);
        out.ramp_histogram = histogram(ramp, out.ramp_coordinate, bins);
        out.emission_luminance = percentiles(luminance);
        out.voxel_optical_depth = percentiles(depth);
    }
    return Result::success();
}

Result fitGasTemperatureRange(const std::string& domain, const GasTemperatureFitRequest& request,
                              GasTemperatureFit& out) {
    out = GasTemperatureFit{};
    if (!g_ctx) return notBound();
    if (!(request.low_percentile >= 0.0f && request.low_percentile < request.high_percentile &&
          request.high_percentile <= 100.0f)) {
        return Result::fail("percentiles must satisfy 0 <= low < high <= 100");
    }
    if (!(request.median_target > 0.0f && request.median_target < 1.0f)) {
        return Result::fail("median_target must be inside (0, 1)");
    }
    const GasFieldView view = resolve(domain);
    if (!view.desc) return Result::fail("gas domain not found: " + domain);
    if (!view.state || view.state->grid.temperature.empty()) {
        return Result::fail("gas domain has no measured temperature field: " + domain);
    }
    // Cold smoke is most of the domain and would pin the low end to zero. The
    // floor keeps the fit on gas that can glow at all.
    std::vector<float> kelvin;
    for (const float value : view.state->grid.temperature) {
        const float k = value * kHeatToKelvin;
        if (k >= request.floor_kelvin) kelvin.push_back(k);
    }
    if (kelvin.size() < 2u) return Result::fail("no gas above floor_kelvin in: " + domain);
    std::sort(kelvin.begin(), kelvin.end());
    const auto at = [&](float percent) {
        return kelvin[static_cast<std::size_t>(percent * 0.01 *
            static_cast<double>(kelvin.size() - 1) + 0.5)];
    };
    const float low = at(request.low_percentile);
    const float high = std::max(at(request.high_percentile), low + 2.0f);
    const float median = std::clamp((at(50.0f) - low) / (high - low), 1e-4f, 0.9999f);
    float gamma = 1.0f;
    if (request.fit_gamma) {
        gamma = std::clamp(std::log(request.median_target) / std::log(median), 0.1f, 8.0f);
    }
    nlohmann::json patch = {{"temperature_min", low}, {"temperature_max", high}};
    if (request.fit_gamma) patch["temperature_gamma"] = gamma;
    const nlohmann::json result = volume_material::execute(*g_ctx, "set", "gas:" + domain, patch);
    if (result.contains("__error")) return Result::fail(result["__error"].get<std::string>());
    out.fitted = true;
    out.temperature_min = low;
    out.temperature_max = high;
    out.temperature_gamma = gamma;
    out.median_before_gamma = median;
    out.cells = kelvin.size();
    return Result::success();
}

} // namespace rtapi
