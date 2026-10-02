#include "Atmosphere/AtmosphereClouds.h"
#include "World.h"

#include <algorithm>
#include <cmath>

namespace atmosphere {

namespace {

bool fail(std::string* error, const std::string& what) {
    if (error) *error = what;
    return false;
}

bool finite(float v) { return std::isfinite(v); }

bool inRange(float v, float lo, float hi) { return finite(v) && v >= lo && v <= hi; }

// Legacy density multiplier <-> extinction. The old volume shader used a
// scattering coefficient of ~0.042 per unit density; its 0.35 default read as
// fair-weather cumulus. 0.12 1/m per unit maps that default to ~0.042 1/m.
// Approximate on purpose: the legacy path is a stand-in until Faz 3b, and
// OptiX (frozen) only needs a plausible look.
constexpr float kLegacyExtinctionPerDensity = 0.12f;
// Legacy cloud_scale 0.45 corresponded to ~1.8 km cells.
constexpr float kLegacyScaleCellProduct = 0.45f * 1800.0f;

float lerpf(float a, float b, float t) { return a + (b - a) * t; }

} // namespace

bool CloudState::anyEnabled() const {
    for (const auto& l : layers) {
        if (l.enabled && l.coverage > 0.0f && l.extinction_per_m > 0.0f) return true;
    }
    return cirrus.enabled && cirrus.coverage > 0.0f && cirrus.opacity > 0.0f;
}

bool validateClouds(const CloudState& c, std::string* error) {
    for (int i = 0; i < kMaxCloudLayers; ++i) {
        const CloudLayer& l = c.layers[i];
        const std::string p = "layers[" + std::to_string(i) + "].";
        if (!inRange(l.base_altitude_m, -500.0f, 20000.0f))
            return fail(error, p + "base_altitude_m must be within -500..20000 m");
        if (!inRange(l.thickness_m, 10.0f, 15000.0f))
            return fail(error, p + "thickness_m must be within 10..15000 m");
        if (!inRange(l.coverage, 0.0f, 1.0f)) return fail(error, p + "coverage must be within 0..1");
        if (!inRange(l.type, 0.0f, 1.0f)) return fail(error, p + "type must be within 0..1 (0 stratus, 0.5 cumulus, 1 cumulonimbus)");
        if (!inRange(l.extinction_per_m, 0.0f, 2.0f))
            return fail(error, p + "extinction_per_m is 1/m and must be within 0..2 (cumulus ~0.05)");
        if (!inRange(l.droplet_diameter_um, 5.0f, 50.0f))
            return fail(error, p + "droplet_diameter_um must be within 5..50 (phase fit range)");
        if (!inRange(l.erosion, 0.0f, 1.0f)) return fail(error, p + "erosion must be within 0..1");
        if (!inRange(l.cell_size_m, 50.0f, 50000.0f)) return fail(error, p + "cell_size_m must be within 50..50000 m");
        if (!inRange(l.detail_size_m, 5.0f, 5000.0f)) return fail(error, p + "detail_size_m must be within 5..5000 m");
    }
    // Enabled layers must not overlap: two shells sharing altitude would be
    // marched twice and double their density where they meet.
    for (int i = 0; i < kMaxCloudLayers; ++i) {
        for (int k = i + 1; k < kMaxCloudLayers; ++k) {
            const CloudLayer& a = c.layers[i];
            const CloudLayer& b = c.layers[k];
            if (!a.enabled || !b.enabled) continue;
            const bool overlap = a.base_altitude_m < b.base_altitude_m + b.thickness_m &&
                                 b.base_altitude_m < a.base_altitude_m + a.thickness_m;
            if (overlap)
                return fail(error, "enabled layers " + std::to_string(i) + " and " + std::to_string(k) +
                                   " overlap in altitude");
        }
    }
    if (!inRange(c.precipitation.rate_mm_h, 0.0f, 200.0f)) return fail(error, "precipitation.rate_mm_h must be within 0..200");
    if (!inRange(c.precipitation.ground_fraction, 0.0f, 1.0f)) return fail(error, "precipitation.ground_fraction must be within 0..1");
    if (!inRange(c.cirrus.altitude_m, 3000.0f, 20000.0f)) return fail(error, "cirrus.altitude_m must be within 3000..20000 m");
    if (!inRange(c.cirrus.coverage, 0.0f, 1.0f)) return fail(error, "cirrus.coverage must be within 0..1");
    if (!inRange(c.cirrus.opacity, 0.0f, 5.0f)) return fail(error, "cirrus.opacity (optical depth) must be within 0..5");
    if (!inRange(c.cirrus.scale_m, 100.0f, 100000.0f)) return fail(error, "cirrus.scale_m must be within 100..100000 m");
    if (!inRange(c.weather.extent_m, 5000.0f, 1000000.0f)) return fail(error, "weather.extent_m must be within 5000..1000000 m");
    if (!inRange(c.weather.feature_size_m, 500.0f, 200000.0f)) return fail(error, "weather.feature_size_m must be within 500..200000 m");
    if (!inRange(c.weather.type_variation, 0.0f, 1.0f)) return fail(error, "weather.type_variation must be within 0..1");
    if (!inRange(c.weather.evolution_speed, 0.0f, 1.0f)) return fail(error, "weather.evolution_speed must be within 0..1");
    if (c.quality.rt_steps < 16 || c.quality.rt_steps > 1024)
        return fail(error, "quality.rt_steps must be within 16..1024");
    if (c.quality.rt_max_bounces < 1 || c.quality.rt_max_bounces > 1024)
        return fail(error, "quality.rt_max_bounces must be within 1..1024");
    if (c.quality.realtime_steps < 8 || c.quality.realtime_steps > 512)
        return fail(error, "quality.realtime_steps must be within 8..512");
    if (c.quality.realtime_resolution_divisor < 1 || c.quality.realtime_resolution_divisor > 8)
        return fail(error, "quality.realtime_resolution_divisor must be within 1..8");
    return true;
}

void cloudsToJson(const CloudState& c, nlohmann::json& j) {
    nlohmann::json layers = nlohmann::json::array();
    for (const auto& l : c.layers) {
        layers.push_back({
            {"enabled", l.enabled}, {"base_altitude_m", l.base_altitude_m},
            {"thickness_m", l.thickness_m}, {"coverage", l.coverage}, {"type", l.type},
            {"extinction_per_m", l.extinction_per_m}, {"droplet_diameter_um", l.droplet_diameter_um},
            {"erosion", l.erosion}, {"cell_size_m", l.cell_size_m}, {"detail_size_m", l.detail_size_m}});
    }
    j["layers"] = layers;
    j["derive_from_climate"] = c.derive_from_climate;
    j["precipitation"] = {{"rate_mm_h", c.precipitation.rate_mm_h}, {"snow", c.precipitation.snow},
                          {"ground_fraction", c.precipitation.ground_fraction}};
    j["cirrus"] = {{"enabled", c.cirrus.enabled}, {"altitude_m", c.cirrus.altitude_m},
                   {"coverage", c.cirrus.coverage}, {"opacity", c.cirrus.opacity},
                   {"scale_m", c.cirrus.scale_m}};
    j["weather"] = {{"seed", c.weather.seed}, {"extent_m", c.weather.extent_m},
                    {"feature_size_m", c.weather.feature_size_m},
                    {"type_variation", c.weather.type_variation},
                    {"evolution_speed", c.weather.evolution_speed}};
    j["quality"] = {{"rt_steps", c.quality.rt_steps},
                    {"rt_reference_path_trace", c.quality.rt_reference_path_trace},
                    {"rt_max_bounces", c.quality.rt_max_bounces},
                    {"realtime_steps", c.quality.realtime_steps},
                    {"realtime_resolution_divisor", c.quality.realtime_resolution_divisor},
                    {"rt_secondary_full_march", c.quality.rt_secondary_full_march}};
}

bool cloudsFromJson(const nlohmann::json& j, CloudState& out, std::string* error) {
    CloudState c;
    c.derive_from_climate = j.value("derive_from_climate", c.derive_from_climate);
    if (j.contains("layers") && j["layers"].is_array()) {
        const auto& arr = j["layers"];
        for (size_t i = 0; i < arr.size() && i < static_cast<size_t>(kMaxCloudLayers); ++i) {
            const auto& s = arr[i];
            CloudLayer& l = c.layers[i];
            l.enabled = s.value("enabled", l.enabled);
            l.base_altitude_m = s.value("base_altitude_m", l.base_altitude_m);
            l.thickness_m = s.value("thickness_m", l.thickness_m);
            l.coverage = s.value("coverage", l.coverage);
            l.type = s.value("type", l.type);
            l.extinction_per_m = s.value("extinction_per_m", l.extinction_per_m);
            l.droplet_diameter_um = s.value("droplet_diameter_um", l.droplet_diameter_um);
            l.erosion = s.value("erosion", l.erosion);
            l.cell_size_m = s.value("cell_size_m", l.cell_size_m);
            l.detail_size_m = s.value("detail_size_m", l.detail_size_m);
        }
    }
    if (j.contains("precipitation") && j["precipitation"].is_object()) {
        const auto& s = j["precipitation"];
        c.precipitation.rate_mm_h = s.value("rate_mm_h", c.precipitation.rate_mm_h);
        c.precipitation.snow = s.value("snow", c.precipitation.snow);
        c.precipitation.ground_fraction = s.value("ground_fraction", c.precipitation.ground_fraction);
    }
    if (j.contains("cirrus") && j["cirrus"].is_object()) {
        const auto& s = j["cirrus"];
        c.cirrus.enabled = s.value("enabled", c.cirrus.enabled);
        c.cirrus.altitude_m = s.value("altitude_m", c.cirrus.altitude_m);
        c.cirrus.coverage = s.value("coverage", c.cirrus.coverage);
        c.cirrus.opacity = s.value("opacity", c.cirrus.opacity);
        c.cirrus.scale_m = s.value("scale_m", c.cirrus.scale_m);
    }
    if (j.contains("weather") && j["weather"].is_object()) {
        const auto& s = j["weather"];
        c.weather.seed = s.value("seed", c.weather.seed);
        c.weather.extent_m = s.value("extent_m", c.weather.extent_m);
        c.weather.feature_size_m = s.value("feature_size_m", c.weather.feature_size_m);
        c.weather.type_variation = s.value("type_variation", c.weather.type_variation);
        c.weather.evolution_speed = s.value("evolution_speed", c.weather.evolution_speed);
    }
    if (j.contains("quality") && j["quality"].is_object()) {
        const auto& s = j["quality"];
        c.quality.rt_steps = s.value("rt_steps", c.quality.rt_steps);
        c.quality.rt_reference_path_trace = s.value("rt_reference_path_trace", c.quality.rt_reference_path_trace);
        c.quality.rt_max_bounces = s.value("rt_max_bounces", c.quality.rt_max_bounces);
        c.quality.realtime_steps = s.value("realtime_steps", c.quality.realtime_steps);
        c.quality.realtime_resolution_divisor =
            s.value("realtime_resolution_divisor", c.quality.realtime_resolution_divisor);
        c.quality.rt_secondary_full_march =
            s.value("rt_secondary_full_march", c.quality.rt_secondary_full_march);
    }
    if (!validateClouds(c, error)) {
        out = CloudState{};
        return false;
    }
    out = c;
    return true;
}


CloudState lerpClouds(const CloudState& a, const CloudState& b, float t) {
    CloudState r = a;
    for (int i = 0; i < kMaxCloudLayers; ++i) {
        const CloudLayer& la = a.layers[i];
        const CloudLayer& lb = b.layers[i];
        CloudLayer& l = r.layers[i];
        l.enabled = t < 1.0f ? la.enabled : lb.enabled;
        l.base_altitude_m = lerpf(la.base_altitude_m, lb.base_altitude_m, t);
        l.thickness_m = lerpf(la.thickness_m, lb.thickness_m, t);
        l.coverage = lerpf(la.coverage, lb.coverage, t);
        l.type = lerpf(la.type, lb.type, t);
        l.extinction_per_m = lerpf(la.extinction_per_m, lb.extinction_per_m, t);
        l.droplet_diameter_um = lerpf(la.droplet_diameter_um, lb.droplet_diameter_um, t);
        l.erosion = lerpf(la.erosion, lb.erosion, t);
        l.cell_size_m = lerpf(la.cell_size_m, lb.cell_size_m, t);
        l.detail_size_m = lerpf(la.detail_size_m, lb.detail_size_m, t);
    }
    r.precipitation.rate_mm_h = lerpf(a.precipitation.rate_mm_h, b.precipitation.rate_mm_h, t);
    r.precipitation.snow = t < 1.0f ? a.precipitation.snow : b.precipitation.snow;
    r.precipitation.ground_fraction = lerpf(a.precipitation.ground_fraction, b.precipitation.ground_fraction, t);
    r.cirrus.enabled = t < 1.0f ? a.cirrus.enabled : b.cirrus.enabled;
    r.cirrus.altitude_m = lerpf(a.cirrus.altitude_m, b.cirrus.altitude_m, t);
    r.cirrus.coverage = lerpf(a.cirrus.coverage, b.cirrus.coverage, t);
    r.cirrus.opacity = lerpf(a.cirrus.opacity, b.cirrus.opacity, t);
    r.cirrus.scale_m = lerpf(a.cirrus.scale_m, b.cirrus.scale_m, t);
    r.weather.type_variation = lerpf(a.weather.type_variation, b.weather.type_variation, t);
    r.weather.evolution_speed = lerpf(a.weather.evolution_speed, b.weather.evolution_speed, t);
    r.weather.feature_size_m = lerpf(a.weather.feature_size_m, b.weather.feature_size_m, t);
    // extent_m, seed and quality are not interpolated: changing them re-lays
    // the whole map, which a blend cannot express.
    return r;
}

float cloudPhaseForwardG(float d) {
    d = std::clamp(d, 5.0f, 50.0f);
    return std::exp(-0.0990567f / (d - 1.67154f));
}

void cloudsToLegacyPacket(const CloudState& c, float wind_offset_x_m, float wind_offset_z_m,
                          NishitaSkyParams& n) {
    const CloudLayer& a = c.layers[0];
    const CloudLayer& b = c.layers[1];
    n.clouds_enabled = a.enabled ? 1 : 0;
    n.cloud_coverage = a.coverage;
    n.cloud_density = a.extinction_per_m / kLegacyExtinctionPerDensity;
    n.cloud_scale = kLegacyScaleCellProduct / std::max(a.cell_size_m, 1.0f);
    n.cloud_height_min = a.base_altitude_m;
    n.cloud_height_max = a.base_altitude_m + a.thickness_m;
    // The legacy offset is in noise units (map extent normalized); world
    // metres over the legacy cell size is the closest equivalent.
    n.cloud_offset_x = wind_offset_x_m / std::max(a.cell_size_m, 1.0f);
    n.cloud_offset_z = wind_offset_z_m / std::max(a.cell_size_m, 1.0f);
    n.cloud_seed = static_cast<int>(c.weather.seed);
    n.cloud_layer2_enabled = b.enabled ? 1 : 0;
    n.cloud2_coverage = b.coverage;
    n.cloud2_density = b.extinction_per_m / kLegacyExtinctionPerDensity;
    n.cloud2_scale = kLegacyScaleCellProduct / std::max(b.cell_size_m, 1.0f);
    n.cloud2_height_min = b.base_altitude_m;
    n.cloud2_height_max = b.base_altitude_m + b.thickness_m;
    n.cloud_detail = std::clamp(1.0f - a.erosion * 0.5f, 0.0f, 1.0f);
    // Lighting: the legacy renderer's own neutral defaults. These are no
    // longer authored anywhere; the physical renderer (3b) derives them.
    n.cloud_quality = 1.0f;
    n.cloud_base_steps = 8;
    n.cloud_light_steps = 0;
    n.cloud_shadow_strength = 0.35f;
    n.cloud_ambient_strength = 1.0f;
    n.cloud_silver_intensity = 0.25f;
    n.cloud_absorption = 1.0f;
    n.cloud_anisotropy = cloudPhaseForwardG(a.droplet_diameter_um);
    n.cloud_anisotropy_back = -0.2f;
    n.cloud_lobe_mix = 0.75f;
    n.cloud_emissive_intensity = 0.0f;
    n.cloud_emissive_color = make_float3(1.0f, 1.0f, 1.0f);
    n.cloud_use_fft = 0;
}

std::vector<std::string> cloudPresetNames() {
    return {"clear", "fair_weather_cumulus", "scattered", "broken", "overcast_stratus",
            "storm", "high_cirrus", "sunset_altocumulus"};
}

bool cloudPreset(const std::string& name, CloudState& out) {
    CloudState c;   // defaults: everything off
    c.derive_from_climate = false;   // a preset is an explicit look
    CloudLayer& low = c.layers[0];
    CloudLayer& mid = c.layers[1];
    if (name == "clear") {
        // nothing enabled
    } else if (name == "fair_weather_cumulus") {
        low.enabled = true; low.base_altitude_m = 1200.0f; low.thickness_m = 1200.0f;
        low.coverage = 0.25f; low.type = 0.5f; low.extinction_per_m = 0.05f;
        low.droplet_diameter_um = 10.0f; low.erosion = 0.55f;
        low.cell_size_m = 1500.0f; low.detail_size_m = 150.0f;
    } else if (name == "scattered") {
        low.enabled = true; low.base_altitude_m = 1400.0f; low.thickness_m = 1800.0f;
        low.coverage = 0.45f; low.type = 0.55f; low.extinction_per_m = 0.06f;
        low.cell_size_m = 2200.0f; low.detail_size_m = 200.0f;
        c.cirrus.enabled = true; c.cirrus.coverage = 0.2f; c.cirrus.opacity = 0.2f;
    } else if (name == "broken") {
        low.enabled = true; low.base_altitude_m = 1200.0f; low.thickness_m = 2000.0f;
        low.coverage = 0.7f; low.type = 0.45f; low.extinction_per_m = 0.07f;
        low.cell_size_m = 3000.0f; low.detail_size_m = 250.0f;
        mid.enabled = true; mid.base_altitude_m = 4200.0f; mid.thickness_m = 800.0f;
        mid.coverage = 0.35f; mid.type = 0.2f; mid.extinction_per_m = 0.03f;
        mid.cell_size_m = 5000.0f; mid.detail_size_m = 300.0f;
    } else if (name == "overcast_stratus") {
        low.enabled = true; low.base_altitude_m = 600.0f; low.thickness_m = 700.0f;
        low.coverage = 1.0f; low.type = 0.0f; low.extinction_per_m = 0.04f;
        low.droplet_diameter_um = 8.0f; low.erosion = 0.25f;
        low.cell_size_m = 6000.0f; low.detail_size_m = 400.0f;
        c.weather.type_variation = 0.05f;
    } else if (name == "storm") {
        low.enabled = true; low.base_altitude_m = 800.0f; low.thickness_m = 9000.0f;
        low.coverage = 0.65f; low.type = 0.95f; low.extinction_per_m = 0.12f;
        low.droplet_diameter_um = 20.0f; low.erosion = 0.45f;
        low.cell_size_m = 8000.0f; low.detail_size_m = 400.0f;
        c.cirrus.enabled = true; c.cirrus.altitude_m = 11000.0f; c.cirrus.coverage = 0.6f;
        c.cirrus.opacity = 0.8f;   // anvil outflow
    } else if (name == "high_cirrus") {
        c.cirrus.enabled = true; c.cirrus.altitude_m = 9500.0f; c.cirrus.coverage = 0.45f;
        c.cirrus.opacity = 0.35f; c.cirrus.scale_m = 12000.0f;
    } else if (name == "sunset_altocumulus") {
        mid.enabled = true; mid.base_altitude_m = 3500.0f; mid.thickness_m = 600.0f;
        mid.coverage = 0.55f; mid.type = 0.3f; mid.extinction_per_m = 0.035f;
        mid.droplet_diameter_um = 8.0f; mid.erosion = 0.7f;
        mid.cell_size_m = 900.0f; mid.detail_size_m = 90.0f;
        c.cirrus.enabled = true; c.cirrus.coverage = 0.25f; c.cirrus.opacity = 0.25f;
    } else {
        return false;
    }
    std::string error;
    if (!validateClouds(c, &error)) return false;
    out = c;
    return true;
}

} // namespace atmosphere
