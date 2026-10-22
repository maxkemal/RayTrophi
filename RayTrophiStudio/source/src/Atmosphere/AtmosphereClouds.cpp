#include "Atmosphere/AtmosphereClouds.h"
#include "Atmosphere/AtmosphereLocalWeather.h"
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

void configureStormLayers(CloudState& c, bool storm) {
    const float bases[kMaxCloudLayers] = {800.0f, 2400.0f, 9500.0f};
    const float thicknesses[kMaxCloudLayers] = {1600.0f, 7100.0f, 3000.0f};
    const float types[kMaxCloudLayers] = {0.40f, 1.0f, 0.30f};
    const float periods[kMaxCloudLayers] = {14000.0f, 4200.0f, 7000.0f};
    const float details[kMaxCloudLayers] = {650.0f, 450.0f, 700.0f};
    const float extinctions[kMaxCloudLayers] = {0.012f, 0.008f, 0.004f};
    for (int i = 0; i < kMaxCloudLayers; ++i) {
        CloudLayer& l = c.layers[i];
        l.enabled = true;
        l.base_altitude_m = bases[i];
        l.thickness_m = thicknesses[i];
        l.coverage = storm ? 0.65f : 0.45f;
        l.type = types[i];
        l.cell_size_m = periods[i];
        l.detail_size_m = details[i];
        l.extinction_per_m = extinctions[i];
        l.erosion = i == 1 ? 0.70f : 0.50f;
        l.droplet_diameter_um = i == 2 ? 8.0f : 18.0f;
        l.formation_group = 1;
        l.formation_role = i + 1;
        l.use_custom_material = true;
        l.material.density_multiplier = i == 0 ? 1.15f : (i == 2 ? 0.65f : 1.0f);
        l.material.absorption_coefficient = i == 0 ? 0.035f : 0.005f;
        l.material.scattering_color = i == 0 ? Vec3(0.92f, 0.94f, 0.96f) : Vec3(1.0f);
    }
    c.cirrus.enabled = storm;
    c.cirrus.altitude_m = 14000.0f;
    c.cirrus.coverage = storm ? 0.30f : 0.0f;
    c.cirrus.opacity = 0.30f;
}

} // namespace

ColumnLevels columnLevels(const CloudState& c) {
    ColumnLevels lv;
    for (int i = 0; i < kMaxCloudLayers; ++i) {
        const CloudLayer& l = c.layers[i];
        if (!l.enabled || l.extinction_per_m <= 0.0f) continue;
        if (lv.host_layer < 0 || l.base_altitude_m < c.layers[lv.host_layer].base_altitude_m) lv.host_layer = i;
    }
    if (lv.host_layer < 0) return lv;
    lv.valid = true;
    lv.lcl_m = c.layers[lv.host_layer].base_altitude_m;
    lv.tropopause_m = (std::max)(c.storms.column_top_m, lv.lcl_m + 1500.0f);
    // The same linear profile the precipitation forms follow (cloudAirTempC).
    const float t0 = c.precipitation.air_temperature_c;
    const float lapse = c.precipitation.lapse_rate_k_per_m;
    if (t0 <= 0.0f) lv.freezing_m = 0.0f;
    else if (lapse <= 1e-6f) lv.freezing_m = lv.tropopause_m;
    else lv.freezing_m = (std::min)(t0 / lapse, 11000.0f);
    return lv;
}

bool CloudState::anyEnabled() const {
    if (material.density_multiplier <= 0.0f) {
        return false;
    }
    for (int i = 0; i < kMaxCloudLayers; ++i) {
        const CloudLayer& l = layers[i];
        const CloudMaterial m = layerMaterial(*this, i);
        if (l.enabled && l.coverage > 0.0f && l.extinction_per_m > 0.0f &&
            m.density_multiplier > 0.0f &&
            m.scattering_coefficient + m.absorption_coefficient > 0.0f) {
            return true;
        }
    }
    return cirrus.enabled && cirrus.coverage > 0.0f && cirrus.opacity > 0.0f &&
           material.scattering_coefficient + material.absorption_coefficient > 0.0f;
}

bool validateCloudMaterial(const CloudMaterial& m, std::string* error) {
    if (!inRange(m.scattering_saturation, 0.0f, 2.0f) ||
        !inRange(m.emission_saturation, 0.0f, 2.0f)) {
        return fail(error, "material scattering/emission saturation must be within 0..2");
    }
    if (!inRange(m.emission_exposure_ev, -16.0f, 16.0f)) {
        return fail(error, "material.emission_exposure_ev must be within -16..16");
    }
    if (!inRange(m.density_multiplier, 0.0f, 100.0f)) {
        return fail(error, "material.density_multiplier must be within 0..100");
    }
    if (!inRange(m.scattering_coefficient, 0.0f, 100.0f) ||
        !inRange(m.absorption_coefficient, 0.0f, 100.0f)) {
        return fail(error, "material scattering/absorption coefficients must be within 0..100");
    }
    if (!inRange(m.scattering_color.x, 0.0f, 1.0f) ||
        !inRange(m.scattering_color.y, 0.0f, 1.0f) ||
        !inRange(m.scattering_color.z, 0.0f, 1.0f)) {
        return fail(error, "material.scattering_color must be linear RGB within 0..1");
    }
    if (!inRange(m.anisotropy, -0.99f, 0.99f)) {
        return fail(error, "material.anisotropy must be within -0.99..0.99");
    }
    if (!inRange(m.emission_color.x, 0.0f, 1.0f) ||
        !inRange(m.emission_color.y, 0.0f, 1.0f) ||
        !inRange(m.emission_color.z, 0.0f, 1.0f) ||
        !inRange(m.emission_strength, 0.0f, 100000.0f)) {
        return fail(error, "material emission colour must be RGB 0..1, strength within 0..100000");
    }
    return true;
}

bool validateClouds(const CloudState& c, std::string* error) {
    if (!validateCloudMaterial(c.material, error)) {
        return false;
    }
    const CloudLightning& lightning = c.lightning;
    if (!inRange(lightning.rate_per_minute, 0.0f, 600.0f) ||
        !inRange(lightning.intensity, 0.0f, 10000.0f) ||
        !inRange(lightning.ground_altitude_m, -500.0f, 20000.0f) ||
        !inRange(lightning.ground_strike_fraction, 0.0f, 1.0f) ||
        !inRange(lightning.upward_fraction, 0.0f, 1.0f) ||
        !inRange(lightning.bolt_radius_m, 0.1f, 30.0f) ||
        !inRange(lightning.duration_s, 0.08f, 3.0f) ||
        !inRange(lightning.leader_fraction, 0.05f, 0.65f) ||
        !inRange(lightning.horizontal_span_m, 100.0f, 50000.0f) ||
        !inRange(lightning.branching, 0.0f, 1.0f) ||
        !inRange(lightning.channel_brightness, 0.0f, 1000.0f)) {
        return fail(error, "lightning requires finite rate_per_minute 0..600, "
                           "intensity 0..10000, "
                           "ground_altitude_m -500..20000, ground_strike_fraction 0..1, "
                           "upward_fraction 0..1, bolt_radius_m 0.1..30, duration_s 0.08..3, "
                           "leader_fraction 0.05..0.65, horizontal_span_m 100..50000, branching 0..1, "
                           "channel_brightness 0..1000");
    }
    for (float v : {lightning.color.x, lightning.color.y, lightning.color.z}) {
        if (!inRange(v, 0.0f, 1.0f)) {
            return fail(error, "lightning.color must be linear RGB within 0..1");
        }
    }
    const CloudStorms& storms = c.storms;
    if (!inRange(storms.center_m.x, -1000000.0f, 1000000.0f) ||
        !inRange(storms.center_m.z, -1000000.0f, 1000000.0f) ||
        !inRange(storms.extent_m, 1000.0f, 500000.0f) ||
        storms.max_cells < 0 || storms.max_cells > kMaxAutoStormCells ||
        !inRange(storms.cell_radius_m, 500.0f, 20000.0f) ||
        !inRange(storms.lifetime_s, 120.0f, 14400.0f) ||
        !inRange(storms.intensity, 0.0f, 1.0f) ||
        !inRange(storms.stratiform_rain, 0.0f, 1.0f) ||
        !inRange(storms.rotation, -1.0f, 1.0f) ||
        !inRange(storms.turrets, 0.0f, 1.0f) ||
        !inRange(storms.column_top_m, 2000.0f, 20000.0f) ||
        !inRange(storms.anvil, 0.0f, 1.0f) ||
        !inRange(storms.wall_cloud, 0.0f, 1.0f) ||
        !inRange(storms.mammatus, 0.0f, 1.0f)) {
        return fail(error, "storms requires center_m x/z within +/-1000000, extent_m 1000..500000, "
                           "max_cells 0..16, cell_radius_m 500..20000, lifetime_s 120..14400, "
                           "intensity 0..1, stratiform_rain 0..1, rotation -1..1, turrets 0..1, "
                           "column_top_m 2000..20000, anvil/wall_cloud/mammatus 0..1");
    }
    if (storms.cells.size() > static_cast<size_t>(kMaxManualStormCells)) {
        return fail(error, "storms.cells holds at most 8 authored cells");
    }
    if (!inRange(c.local_weather.valley_fog, 0.0f, 1.0f) ||
        !inRange(c.local_weather.valley_fog_lift, 0.0f, 1.0f) ||
        !inRange(c.local_weather.post_rain_mist, 0.0f, 1.0f)) {
        return fail(error, "local_weather.valley_fog / valley_fog_lift / post_rain_mist must be within 0..1");
    }
    if (!inRange(c.local_weather.fog_strength, 0.0f, 10.0f)) {
        return fail(error, "local_weather.fog_strength must be within 0..10");
    }
    if (c.volumes.slots < 1 || c.volumes.slots > kMaxCloudVolumeSlots ||
        !inRange(c.volumes.voxel_m, 20.0f, 400.0f) || !inRange(c.volumes.skin_m, 5.0f, 500.0f) ||
        !inRange(c.volumes.min_tower, 0.0f, 1.0f)) {
        return fail(error, "volumes: slots 1..16, voxel_m 20..400, skin_m 5..500, min_tower 0..1");
    }
    for (size_t i = 0; i < storms.cells.size(); ++i) {
        const StormCell& cell = storms.cells[i];
        if (!inRange(cell.position_m.x, -1000000.0f, 1000000.0f) ||
            !inRange(cell.position_m.z, -1000000.0f, 1000000.0f) ||
            !inRange(cell.radius_m, 200.0f, 30000.0f) ||
            !inRange(cell.intensity, 0.0f, 1.0f) ||
            !inRange(cell.birth_s, -1.0e7f, 1.0e7f) ||
            !inRange(cell.lifetime_s, 0.0f, 14400.0f)) {
            return fail(error, "storms.cells[" + std::to_string(i) + "] requires position_m x/z "
                               "within +/-1000000, radius_m 200..30000, intensity 0..1, finite "
                               "birth_s, lifetime_s 0..14400 (0 = permanently mature)");
        }
    }
    for (int i = 0; i < kMaxCloudLayers; ++i) {
        const CloudLayer& l = c.layers[i];
        const std::string p = "layers[" + std::to_string(i) + "].";
        if (l.formation_role < 0 || l.formation_role > 3 ||
            l.formation_group < 0 || l.formation_group > 3) {
            return fail(error, p + "formation_role/group must be within 0..3");
        }
        if (!validateCloudMaterial(l.material, error)) {
            if (error) {
                *error = p + *error;
            }
            return false;
        }
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
        if (!inRange(l.base_raggedness, 0.0f, 1.0f))
            return fail(error, p + "base_raggedness must be within 0..1");
        if (!inRange(l.cell_size_m, 50.0f, 50000.0f)) return fail(error, p + "cell_size_m must be within 50..50000 m");
        if (!inRange(l.detail_size_m, 5.0f, 5000.0f)) return fail(error, p + "detail_size_m must be within 5..5000 m");
        if (!inRange(l.rotation_degrees, -360.0f, 360.0f)) {
            return fail(error, p + "rotation_degrees must be within -360..360");
        }
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
    if (!inRange(c.precipitation.near_range_m, 2.0f, 60.0f)) return fail(error, "precipitation.near_range_m must be within 2..60 m");
    if (!inRange(c.shutter_s, 0.0f, 0.5f)) return fail(error, "shutter_s must be within 0..0.5 s");
    if (c.precipitation.form < kPrecipAuto || c.precipitation.form > kPrecipIcePellets)
        return fail(error, "precipitation.form must be auto|rain|drizzle|snow|sleet|graupel|hail|freezing_rain|ice_pellets");
    if (!inRange(c.cirrus.altitude_m, 3000.0f, 20000.0f)) return fail(error, "cirrus.altitude_m must be within 3000..20000 m");
    if (!inRange(c.cirrus.coverage, 0.0f, 1.0f)) return fail(error, "cirrus.coverage must be within 0..1");
    if (!inRange(c.cirrus.opacity, 0.0f, 5.0f)) return fail(error, "cirrus.opacity (optical depth) must be within 0..5");
    if (!inRange(c.cirrus.scale_m, 100.0f, 100000.0f)) return fail(error, "cirrus.scale_m must be within 100..100000 m");
    if (!inRange(c.cirrus.rotation_degrees, -360.0f, 360.0f)) {
        return fail(error, "cirrus.rotation_degrees must be within -360..360");
    }
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
    if (c.quality.realtime_interleave < 1 || c.quality.realtime_interleave > 4)
        return fail(error, "quality.realtime_interleave must be within 1..4");
    return true;
}

void cloudMaterialToJson(const CloudMaterial& m, nlohmann::json& j) {
    j = {
        {"density_multiplier", m.density_multiplier},
        {"scattering_saturation", m.scattering_saturation},
        {"emission_saturation", m.emission_saturation},
        {"emission_exposure_ev", m.emission_exposure_ev},
        {"scattering_coefficient", m.scattering_coefficient},
        {"scattering_color", {m.scattering_color.x, m.scattering_color.y,
                              m.scattering_color.z}},
        {"absorption_coefficient", m.absorption_coefficient},
        {"override_anisotropy", m.override_anisotropy}, {"anisotropy", m.anisotropy},
        {"emission_color", {m.emission_color.x, m.emission_color.y, m.emission_color.z}},
        {"emission_strength", m.emission_strength}
    };
}

void cloudsToJson(const CloudState& c, nlohmann::json& j) {
    cloudMaterialToJson(c.material, j["material"]);
    const CloudLightning& lightning = c.lightning;
    j["lightning"] = {
        {"enabled", lightning.enabled},
        {"derive_rate_from_climate", lightning.derive_rate_from_climate},
        {"attach_to_scene", lightning.attach_to_scene},
        {"rate_per_minute", lightning.rate_per_minute},
        {"intensity", lightning.intensity},
        {"color", {lightning.color.x, lightning.color.y, lightning.color.z}},
        {"ground_altitude_m", lightning.ground_altitude_m},
        {"ground_strike_fraction", lightning.ground_strike_fraction},
        {"upward_fraction", lightning.upward_fraction},
        {"bolt_radius_m", lightning.bolt_radius_m},
        {"duration_s", lightning.duration_s},
        {"leader_fraction", lightning.leader_fraction},
        {"horizontal_span_m", lightning.horizontal_span_m},
        {"branching", lightning.branching},
        {"channel_brightness", lightning.channel_brightness}
    };
    {
        const CloudStorms& s = c.storms;
        nlohmann::json cells = nlohmann::json::array();
        for (const StormCell& cell : s.cells) {
            cells.push_back({{"position_m", {cell.position_m.x, cell.position_m.y, cell.position_m.z}},
                             {"radius_m", cell.radius_m}, {"intensity", cell.intensity},
                             {"birth_s", cell.birth_s}, {"lifetime_s", cell.lifetime_s}});
        }
        j["storms"] = {
            {"enabled", s.enabled}, {"derive_from_climate", s.derive_from_climate},
            {"center_m", {s.center_m.x, s.center_m.y, s.center_m.z}}, {"extent_m", s.extent_m},
            {"max_cells", s.max_cells}, {"cell_radius_m", s.cell_radius_m},
            {"lifetime_s", s.lifetime_s}, {"intensity", s.intensity},
            {"stratiform_rain", s.stratiform_rain}, {"rotation", s.rotation},
            {"turrets", s.turrets}, {"column", s.column}, {"column_top_m", s.column_top_m},
            {"anvil", s.anvil}, {"wall_cloud", s.wall_cloud}, {"mammatus", s.mammatus},
            {"cells", cells}};
    }
    j["local_weather"] = {{"terrain_suitability", c.local_weather.terrain_suitability},
                          {"fog_enabled", c.local_weather.fog_enabled},
                          {"fog_strength", c.local_weather.fog_strength},
                          {"fog_manual", c.local_weather.fog_manual},
                          {"valley_fog", c.local_weather.valley_fog},
                          {"valley_fog_lift", c.local_weather.valley_fog_lift},
                          {"post_rain_mist", c.local_weather.post_rain_mist}};
    j["volumes"] = {{"enabled", c.volumes.enabled}, {"slots", c.volumes.slots},
                    {"voxel_m", c.volumes.voxel_m}, {"skin_m", c.volumes.skin_m},
                    {"min_tower", c.volumes.min_tower}};
    {
        // Read-only (ignored by cloudsFromJson): the derived column levels.
        const ColumnLevels lv = columnLevels(c);
        j["column_levels"] = {{"valid", lv.valid}, {"lcl_m", lv.lcl_m},
                              {"freezing_m", lv.freezing_m}, {"tropopause_m", lv.tropopause_m}};
    }
    nlohmann::json layers = nlohmann::json::array();
    for (const auto& l : c.layers) {
        nlohmann::json material;
        cloudMaterialToJson(l.material, material);
        layers.push_back({
            {"enabled", l.enabled}, {"base_altitude_m", l.base_altitude_m},
            {"thickness_m", l.thickness_m}, {"coverage", l.coverage}, {"type", l.type},
            {"extinction_per_m", l.extinction_per_m}, {"droplet_diameter_um", l.droplet_diameter_um},
            {"erosion", l.erosion}, {"base_raggedness", l.base_raggedness},
            {"cell_size_m", l.cell_size_m},
            {"detail_size_m", l.detail_size_m}, {"rotation_degrees", l.rotation_degrees},
            {"formation_role", l.formation_role}, {"formation_group", l.formation_group},
            {"use_custom_material", l.use_custom_material}, {"material", material}});
    }
    j["layers"] = layers;
    j["derive_from_climate"] = c.derive_from_climate;
    j["precipitation"] = {{"rate_mm_h", c.precipitation.rate_mm_h},
                          {"form", precipFormName(c.precipitation.form)},
                          {"ground_fraction", c.precipitation.ground_fraction},
                          {"air_temperature_c", c.precipitation.air_temperature_c},
                          {"lapse_rate_k_per_m", c.precipitation.lapse_rate_k_per_m},
                          {"snow", c.precipitation.snow},
                          {"near_enabled", c.precipitation.near_enabled},
                          {"near_range_m", c.precipitation.near_range_m}};
    j["shutter_s"] = c.shutter_s;
    j["cirrus"] = {{"enabled", c.cirrus.enabled}, {"altitude_m", c.cirrus.altitude_m},
                   {"coverage", c.cirrus.coverage}, {"opacity", c.cirrus.opacity},
                   {"scale_m", c.cirrus.scale_m},
                   {"rotation_degrees", c.cirrus.rotation_degrees}};
    j["weather"] = {{"seed", c.weather.seed}, {"extent_m", c.weather.extent_m},
                    {"feature_size_m", c.weather.feature_size_m},
                    {"type_variation", c.weather.type_variation},
                    {"evolution_speed", c.weather.evolution_speed}};
    j["quality"] = {{"rt_steps", c.quality.rt_steps},
                    {"rt_reference_path_trace", c.quality.rt_reference_path_trace},
                    {"rt_max_bounces", c.quality.rt_max_bounces},
                    {"realtime_steps", c.quality.realtime_steps},
                    {"realtime_resolution_divisor", c.quality.realtime_resolution_divisor},
                    {"realtime_interleave", c.quality.realtime_interleave},
                    {"rt_secondary_full_march", c.quality.rt_secondary_full_march}};
}

bool cloudMaterialFromJson(const nlohmann::json& s, CloudMaterial& out,
                          std::string* error) {
    CloudMaterial m;
    if (!s.is_object()) {
        return fail(error, "material must be an object");
    }
    m.density_multiplier = s.value("density_multiplier", m.density_multiplier);
    for (const char* key : {"scattering_saturation", "emission_saturation",
                            "emission_exposure_ev"}) {
        if (s.contains(key) && !s[key].is_number()) {
            return fail(error, std::string("material.") + key + " must be a number");
        }
    }
    m.scattering_saturation = s.value("scattering_saturation", m.scattering_saturation);
    m.emission_saturation = s.value("emission_saturation", m.emission_saturation);
    m.emission_exposure_ev = s.value("emission_exposure_ev", m.emission_exposure_ev);
    m.scattering_coefficient = s.value("scattering_coefficient", m.scattering_coefficient);
    m.absorption_coefficient = s.value("absorption_coefficient", m.absorption_coefficient);
    m.override_anisotropy = s.value("override_anisotropy", m.override_anisotropy);
    m.anisotropy = s.value("anisotropy", m.anisotropy);
    m.emission_strength = s.value("emission_strength", m.emission_strength);
    auto readColor = [&](const char* key, Vec3& color) {
        if (!s.contains(key)) {
            return true;
        }
        const auto& v = s[key];
        if (!v.is_array() || v.size() != 3 || !v[0].is_number() ||
            !v[1].is_number() || !v[2].is_number()) {
            return fail(error, std::string("material.") + key + " must be [r,g,b]");
        }
        color = Vec3(v[0].get<float>(), v[1].get<float>(), v[2].get<float>());
        return true;
    };
    if (!readColor("scattering_color", m.scattering_color) ||
        !readColor("emission_color", m.emission_color)) {
        return false;
    }
    if (!validateCloudMaterial(m, error)) {
        return false;
    }
    out = m;
    return true;
}

bool cloudsFromJson(const nlohmann::json& j, CloudState& out, std::string* error) {
    CloudState c;
    if (j.contains("material") &&
        !cloudMaterialFromJson(j["material"], c.material, error)) {
        return false;
    }
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
            l.base_raggedness = s.value("base_raggedness", l.base_raggedness);
            l.cell_size_m = s.value("cell_size_m", l.cell_size_m);
            l.detail_size_m = s.value("detail_size_m", l.detail_size_m);
            l.rotation_degrees = s.value("rotation_degrees", l.rotation_degrees);
            for (const char* key : {"formation_role", "formation_group"}) {
                if (s.contains(key) && !s[key].is_number_integer()) {
                    return fail(error, std::string("layer.") + key + " must be an integer");
                }
            }
            l.formation_role = s.value("formation_role", l.formation_role);
            l.formation_group = s.value("formation_group", l.formation_group);
            l.use_custom_material = s.value("use_custom_material", l.use_custom_material);
            if (s.contains("material") &&
                !cloudMaterialFromJson(s["material"], l.material, error)) {
                if (error) {
                    *error = "layers[" + std::to_string(i) + "]." + *error;
                }
                return false;
            }
        }
    }
    if (j.contains("lightning")) {
        const auto& s = j["lightning"];
        if (!s.is_object()) {
            return fail(error, "lightning must be an object");
        }
        for (auto it = s.begin(); it != s.end(); ++it) {
            const bool flag = it.key() == "enabled" || it.key() == "derive_rate_from_climate" ||
                              it.key() == "attach_to_scene";
            // center_m / extent_m: the storm square before 2026-10-12, read
            // into storms.* below (projects saved before storm cells).
            const bool vector = it.key() == "color" || it.key() == "center_m";
            const bool number = it.key() == "rate_per_minute" || it.key() == "intensity" ||
                it.key() == "extent_m" || it.key() == "ground_altitude_m" ||
                it.key() == "ground_strike_fraction" || it.key() == "bolt_radius_m" ||
                it.key() == "upward_fraction" ||
                it.key() == "duration_s" || it.key() == "leader_fraction" ||
                it.key() == "horizontal_span_m" ||
                it.key() == "branching" || it.key() == "shutter_s" ||
                it.key() == "channel_brightness";
            if ((!flag && !vector && !number) || (flag && !it.value().is_boolean()) ||
                (number && !it.value().is_number())) {
                return fail(error, "invalid lightning field or type: " + it.key());
            }
        }
        CloudLightning& lightning = c.lightning;
        lightning.enabled = s.value("enabled", lightning.enabled);
        lightning.derive_rate_from_climate = s.value(
            "derive_rate_from_climate", lightning.derive_rate_from_climate);
        lightning.attach_to_scene = s.value("attach_to_scene", lightning.attach_to_scene);
        lightning.rate_per_minute = s.value("rate_per_minute", lightning.rate_per_minute);
        lightning.intensity = s.value("intensity", lightning.intensity);
        lightning.ground_altitude_m = s.value("ground_altitude_m", lightning.ground_altitude_m);
        lightning.ground_strike_fraction = s.value(
            "ground_strike_fraction", lightning.ground_strike_fraction);
        lightning.upward_fraction = s.value("upward_fraction", lightning.upward_fraction);
        lightning.bolt_radius_m = s.value("bolt_radius_m", lightning.bolt_radius_m);
        lightning.duration_s = s.value("duration_s", lightning.duration_s);
        lightning.leader_fraction = s.value("leader_fraction", lightning.leader_fraction);
        lightning.horizontal_span_m = s.value("horizontal_span_m", lightning.horizontal_span_m);
        lightning.branching = s.value("branching", lightning.branching);
        // lightning.shutter_s before 2026-10-14: now the shared CloudState::
        // shutter_s (a top-level key, when present, wins below).
        c.shutter_s = s.value("shutter_s", c.shutter_s);
        lightning.channel_brightness = s.value("channel_brightness", lightning.channel_brightness);
        for (const char* key : {"color", "center_m"}) {
            if (!s.contains(key)) {
                continue;
            }
            const auto& value = s[key];
            if (!value.is_array() || value.size() != 3 ||
                !value[0].is_number() || !value[1].is_number() || !value[2].is_number()) {
                return fail(error, std::string("lightning.") + key + " must be three numbers");
            }
            const Vec3 vector(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
            if (std::string(key) == "color") {
                lightning.color = vector;
            } else {
                c.storms.center_m = vector;
            }
        }
        c.storms.extent_m = s.value("extent_m", c.storms.extent_m);
    }
    if (j.contains("storms")) {
        const auto& s = j["storms"];
        if (!s.is_object()) {
            return fail(error, "storms must be an object");
        }
        CloudStorms& storms = c.storms;
        try {
            storms.enabled = s.value("enabled", storms.enabled);
            storms.derive_from_climate = s.value("derive_from_climate", storms.derive_from_climate);
            if (s.contains("center_m")) {
                const auto& v = s["center_m"];
                if (!v.is_array() || v.size() != 3) return fail(error, "storms.center_m must be three numbers");
                storms.center_m = Vec3(v[0].get<float>(), v[1].get<float>(), v[2].get<float>());
            }
            storms.extent_m = s.value("extent_m", storms.extent_m);
            storms.max_cells = s.value("max_cells", storms.max_cells);
            storms.cell_radius_m = s.value("cell_radius_m", storms.cell_radius_m);
            storms.lifetime_s = s.value("lifetime_s", storms.lifetime_s);
            storms.intensity = s.value("intensity", storms.intensity);
            storms.stratiform_rain = s.value("stratiform_rain", storms.stratiform_rain);
            storms.rotation = s.value("rotation", storms.rotation);
            storms.turrets = s.value("turrets", storms.turrets);
            storms.column = s.value("column", storms.column);
            storms.column_top_m = s.value("column_top_m", storms.column_top_m);
            storms.anvil = s.value("anvil", storms.anvil);
            storms.wall_cloud = s.value("wall_cloud", storms.wall_cloud);
            storms.mammatus = s.value("mammatus", storms.mammatus);
            if (s.contains("cells")) {
                const auto& arr = s["cells"];
                if (!arr.is_array()) return fail(error, "storms.cells must be an array");
                storms.cells.clear();
                for (const auto& e : arr) {
                    if (!e.is_object()) return fail(error, "storms.cells entries must be objects");
                    StormCell cell;
                    if (e.contains("position_m")) {
                        const auto& v = e["position_m"];
                        if (!v.is_array() || v.size() != 3)
                            return fail(error, "storms.cells[].position_m must be three numbers");
                        cell.position_m = Vec3(v[0].get<float>(), v[1].get<float>(), v[2].get<float>());
                    }
                    cell.radius_m = e.value("radius_m", cell.radius_m);
                    cell.intensity = e.value("intensity", cell.intensity);
                    cell.birth_s = e.value("birth_s", cell.birth_s);
                    cell.lifetime_s = e.value("lifetime_s", cell.lifetime_s);
                    storms.cells.push_back(cell);
                }
            }
        } catch (const std::exception& e) {
            return fail(error, std::string("storms: ") + e.what());
        }
    }
    if (j.contains("local_weather")) {
        const auto& s = j["local_weather"];
        if (!s.is_object()) return fail(error, "local_weather must be an object");
        try {
            CloudLocalWeather& lw = c.local_weather;
            lw.terrain_suitability = s.value("terrain_suitability", lw.terrain_suitability);
            lw.fog_enabled = s.value("fog_enabled", lw.fog_enabled);
            lw.fog_strength = s.value("fog_strength", lw.fog_strength);
            lw.fog_manual = s.value("fog_manual", lw.fog_manual);
            lw.valley_fog = s.value("valley_fog", lw.valley_fog);
            lw.valley_fog_lift = s.value("valley_fog_lift", lw.valley_fog_lift);
            lw.post_rain_mist = s.value("post_rain_mist", lw.post_rain_mist);
        } catch (const std::exception& e) {
            return fail(error, std::string("local_weather: ") + e.what());
        }
    }
    if (j.contains("volumes")) {
        const auto& s = j["volumes"];
        if (!s.is_object()) return fail(error, "volumes must be an object");
        try {
            CloudVolumes& v = c.volumes;
            v.enabled = s.value("enabled", v.enabled);
            v.slots = s.value("slots", v.slots);
            v.voxel_m = s.value("voxel_m", v.voxel_m);
            v.skin_m = s.value("skin_m", v.skin_m);
            v.min_tower = s.value("min_tower", v.min_tower);
        } catch (const std::exception& e) {
            return fail(error, std::string("volumes: ") + e.what());
        }
    }
    if (j.contains("precipitation") && j["precipitation"].is_object()) {
        const auto& s = j["precipitation"];
        c.precipitation.rate_mm_h = s.value("rate_mm_h", c.precipitation.rate_mm_h);
        c.precipitation.ground_fraction = s.value("ground_fraction", c.precipitation.ground_fraction);
        c.precipitation.near_enabled = s.value("near_enabled", c.precipitation.near_enabled);
        c.precipitation.near_range_m = s.value("near_range_m", c.precipitation.near_range_m);
        // air_temperature_c / lapse_rate_k_per_m / snow are derived from the
        // climate (World::rederiveClouds): read back, but every sync rewrites them.
        c.precipitation.air_temperature_c = s.value("air_temperature_c", c.precipitation.air_temperature_c);
        c.precipitation.lapse_rate_k_per_m = s.value("lapse_rate_k_per_m", c.precipitation.lapse_rate_k_per_m);
        if (s.contains("form")) {
            if (!s["form"].is_string()) return fail(error, "precipitation.form must be a string");
            const int form = precipFormFromName(s["form"].get<std::string>());
            if (form < 0)
                return fail(error, "precipitation.form must be auto|rain|drizzle|snow|sleet|graupel|hail|"
                                   "freezing_rain|ice_pellets");
            c.precipitation.form = form;
        } else if (s.value("snow", false)) {
            // Before 2026-10-12 `snow` was the authored form switch.
            c.precipitation.form = kPrecipSnow;
        }
    }
    if (j.contains("cirrus") && j["cirrus"].is_object()) {
        const auto& s = j["cirrus"];
        c.cirrus.enabled = s.value("enabled", c.cirrus.enabled);
        c.cirrus.altitude_m = s.value("altitude_m", c.cirrus.altitude_m);
        c.cirrus.coverage = s.value("coverage", c.cirrus.coverage);
        c.cirrus.opacity = s.value("opacity", c.cirrus.opacity);
        c.cirrus.scale_m = s.value("scale_m", c.cirrus.scale_m);
        c.cirrus.rotation_degrees = s.value("rotation_degrees", c.cirrus.rotation_degrees);
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
        c.quality.realtime_interleave =
            s.value("realtime_interleave", c.quality.realtime_interleave);
        c.quality.rt_secondary_full_march =
            s.value("rt_secondary_full_march", c.quality.rt_secondary_full_march);
    }
    if (j.contains("shutter_s")) {
        if (!j["shutter_s"].is_number()) return fail(error, "shutter_s must be a number");
        c.shutter_s = j["shutter_s"].get<float>();
    }
    if (!validateClouds(c, error)) {
        out = CloudState{};
        return false;
    }
    out = c;
    return true;
}


CloudMaterial lerpCloudMaterial(const CloudMaterial& a, const CloudMaterial& b, float t) {
    CloudMaterial r = a;
    r.density_multiplier =
        lerpf(a.density_multiplier, b.density_multiplier, t);
    r.scattering_coefficient =
        lerpf(a.scattering_coefficient, b.scattering_coefficient, t);
    r.absorption_coefficient =
        lerpf(a.absorption_coefficient, b.absorption_coefficient, t);
    r.scattering_color = a.scattering_color * (1.0f - t) + b.scattering_color * t;
    r.scattering_saturation = lerpf(a.scattering_saturation, b.scattering_saturation, t);
    r.emission_saturation = lerpf(a.emission_saturation, b.emission_saturation, t);
    r.emission_exposure_ev = lerpf(a.emission_exposure_ev, b.emission_exposure_ev, t);
    r.override_anisotropy = t < 1.0f ? a.override_anisotropy : b.override_anisotropy;
    r.anisotropy = lerpf(a.anisotropy, b.anisotropy, t);
    r.emission_color = a.emission_color * (1.0f - t) + b.emission_color * t;
    r.emission_strength =
        lerpf(a.emission_strength, b.emission_strength, t);
    return r;
}

CloudState lerpClouds(const CloudState& a, const CloudState& b, float t) {
    CloudState r = a;
    r.material = lerpCloudMaterial(a.material, b.material, t);
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
        l.base_raggedness = lerpf(la.base_raggedness, lb.base_raggedness, t);
        l.cell_size_m = lerpf(la.cell_size_m, lb.cell_size_m, t);
        l.detail_size_m = lerpf(la.detail_size_m, lb.detail_size_m, t);
        l.rotation_degrees = lerpf(la.rotation_degrees, lb.rotation_degrees, t);
        l.formation_role = t < 1.0f ? la.formation_role : lb.formation_role;
        l.formation_group = t < 1.0f ? la.formation_group : lb.formation_group;
        l.use_custom_material = t < 1.0f ? la.use_custom_material : lb.use_custom_material;
        l.material = lerpCloudMaterial(la.material, lb.material, t);
    }
    r.precipitation.rate_mm_h = lerpf(a.precipitation.rate_mm_h, b.precipitation.rate_mm_h, t);
    r.lightning = t < 1.0f ? a.lightning : b.lightning;
    r.lightning.rate_per_minute = lerpf(
        a.lightning.rate_per_minute, b.lightning.rate_per_minute, t);
    r.lightning.intensity = lerpf(a.lightning.intensity, b.lightning.intensity, t);
    r.lightning.color = a.lightning.color * (1.0f - t) + b.lightning.color * t;
    r.storms = t < 1.0f ? a.storms : b.storms;
    r.storms.center_m = a.storms.center_m * (1.0f - t) + b.storms.center_m * t;
    r.storms.extent_m = lerpf(a.storms.extent_m, b.storms.extent_m, t);
    r.storms.cell_radius_m = lerpf(a.storms.cell_radius_m, b.storms.cell_radius_m, t);
    r.storms.intensity = lerpf(a.storms.intensity, b.storms.intensity, t);
    r.storms.stratiform_rain = lerpf(a.storms.stratiform_rain, b.storms.stratiform_rain, t);
    r.storms.rotation = lerpf(a.storms.rotation, b.storms.rotation, t);
    r.storms.turrets = lerpf(a.storms.turrets, b.storms.turrets, t);
    r.storms.column_top_m = lerpf(a.storms.column_top_m, b.storms.column_top_m, t);
    r.storms.anvil = lerpf(a.storms.anvil, b.storms.anvil, t);
    r.storms.wall_cloud = lerpf(a.storms.wall_cloud, b.storms.wall_cloud, t);
    r.local_weather = t < 1.0f ? a.local_weather : b.local_weather;
    r.local_weather.fog_strength = lerpf(a.local_weather.fog_strength, b.local_weather.fog_strength, t);
    // Keyed authored fog: a morning sequence fades the valley fog and lifts it.
    r.local_weather.valley_fog = lerpf(a.local_weather.valley_fog, b.local_weather.valley_fog, t);
    r.local_weather.valley_fog_lift = lerpf(a.local_weather.valley_fog_lift, b.local_weather.valley_fog_lift, t);
    r.local_weather.post_rain_mist = lerpf(a.local_weather.post_rain_mist, b.local_weather.post_rain_mist, t);
    // Volume settings are a resource configuration, not a look: no blend.
    r.volumes = t < 1.0f ? a.volumes : b.volumes;
    r.storms.mammatus = lerpf(a.storms.mammatus, b.storms.mammatus, t);
    r.lightning.upward_fraction = lerpf(a.lightning.upward_fraction, b.lightning.upward_fraction, t);
    r.lightning.ground_altitude_m = lerpf(
        a.lightning.ground_altitude_m, b.lightning.ground_altitude_m, t);
    r.lightning.ground_strike_fraction = lerpf(
        a.lightning.ground_strike_fraction, b.lightning.ground_strike_fraction, t);
    r.lightning.bolt_radius_m = lerpf(a.lightning.bolt_radius_m, b.lightning.bolt_radius_m, t);
    r.lightning.duration_s = lerpf(a.lightning.duration_s, b.lightning.duration_s, t);
    r.lightning.leader_fraction = lerpf(a.lightning.leader_fraction,
                                      b.lightning.leader_fraction, t);
    r.lightning.horizontal_span_m = lerpf(
        a.lightning.horizontal_span_m, b.lightning.horizontal_span_m, t);
    r.lightning.branching = lerpf(a.lightning.branching, b.lightning.branching, t);
    r.precipitation.snow = t < 1.0f ? a.precipitation.snow : b.precipitation.snow;
    r.precipitation.form = t < 1.0f ? a.precipitation.form : b.precipitation.form;
    r.precipitation.air_temperature_c = lerpf(a.precipitation.air_temperature_c, b.precipitation.air_temperature_c, t);
    r.precipitation.lapse_rate_k_per_m = lerpf(a.precipitation.lapse_rate_k_per_m, b.precipitation.lapse_rate_k_per_m, t);
    r.precipitation.ground_fraction = lerpf(a.precipitation.ground_fraction, b.precipitation.ground_fraction, t);
    r.precipitation.near_enabled = t < 1.0f ? a.precipitation.near_enabled : b.precipitation.near_enabled;
    r.precipitation.near_range_m = lerpf(a.precipitation.near_range_m, b.precipitation.near_range_m, t);
    r.shutter_s = lerpf(a.shutter_s, b.shutter_s, t);
    r.cirrus.enabled = t < 1.0f ? a.cirrus.enabled : b.cirrus.enabled;
    r.cirrus.altitude_m = lerpf(a.cirrus.altitude_m, b.cirrus.altitude_m, t);
    r.cirrus.coverage = lerpf(a.cirrus.coverage, b.cirrus.coverage, t);
    r.cirrus.opacity = lerpf(a.cirrus.opacity, b.cirrus.opacity, t);
    r.cirrus.scale_m = lerpf(a.cirrus.scale_m, b.cirrus.scale_m, t);
    r.cirrus.rotation_degrees =
        lerpf(a.cirrus.rotation_degrees, b.cirrus.rotation_degrees, t);
    r.weather.type_variation = lerpf(a.weather.type_variation, b.weather.type_variation, t);
    r.weather.evolution_speed = lerpf(a.weather.evolution_speed, b.weather.evolution_speed, t);
    r.weather.feature_size_m = lerpf(a.weather.feature_size_m, b.weather.feature_size_m, t);
    // extent_m, seed and quality are not interpolated: changing them re-lays
    // the whole map, which a blend cannot express.
    return r;
}

namespace {
uint32_t stormHash(uint32_t v) {
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; return v ^ (v >> 16);
}
float stormRandom(uint32_t& seed) {
    seed = stormHash(seed + 0x9e3779b9u);
    return static_cast<float>(seed & 0x00ffffffu) / 16777216.0f;
}
float stormSmooth(float a, float b, float x) {
    const float t = std::clamp((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
// Phase curves of a cell over its life s = age / lifetime (ATMOSPHERE_STORMS.md §3).
void stormPhase(float s, float peak, StormCellState& c) {
    c.life = s;
    c.peak = peak;
    const float grow = stormSmooth(0.0f, 0.25f, s);
    c.coverage = peak * grow * (1.0f - 0.4f * stormSmooth(0.7f, 1.0f, s));   // anvil remnant
    c.tower = peak * stormSmooth(0.1f, 0.3f, s) * (1.0f - stormSmooth(0.65f, 0.95f, s));
    c.convective_rain = peak * stormSmooth(0.22f, 0.35f, s) * (1.0f - stormSmooth(0.6f, 0.85f, s));
    c.stratiform_rain = peak * 0.35f * stormSmooth(0.6f, 0.75f, s) * (1.0f - stormSmooth(0.9f, 1.0f, s));
    c.lightning = peak * stormSmooth(0.25f, 0.32f, s) * (1.0f - stormSmooth(0.62f, 0.7f, s));
}
} // namespace

namespace {
const char* const kPrecipFormNames[] = {"auto", "rain", "drizzle", "snow", "sleet", "graupel",
                                        "hail", "freezing_rain", "ice_pellets"};
}

const char* precipFormName(int form) {
    return form >= kPrecipAuto && form <= kPrecipIcePellets ? kPrecipFormNames[form] : "auto";
}

int precipFormFromName(const std::string& name) {
    for (int i = kPrecipAuto; i <= kPrecipIcePellets; ++i)
        if (name == kPrecipFormNames[i]) return i;
    return -1;
}

int freezeStormCells(CloudState& clouds, float time_s, const Vec3& wind_mps, int first) {
    StormCellState live[kMaxStormCells];
    const int n = evaluateStormCells(clouds, time_s, wind_mps, live);
    int order[kMaxStormCells];
    int m = 0;
    if (first >= 0 && first < n && !live[first].manual) order[m++] = first;
    for (int i = 0; i < n; ++i)
        if (!live[i].manual && i != first) order[m++] = i;
    CloudStorms& s = clouds.storms;
    int added = 0;
    for (int k = 0; k < m && s.cells.size() < static_cast<size_t>(kMaxManualStormCells); ++k) {
        const StormCellState& c = live[order[k]];
        StormCell cell;
        // Authored cells store the birth position and drift from there.
        cell.position_m = Vec3(c.x - wind_mps.x * c.age_s, 0.0f, c.z - wind_mps.z * c.age_s);
        cell.radius_m = std::clamp(c.radius_m, 200.0f, 30000.0f);
        cell.intensity = std::clamp(c.peak, 0.0f, 1.0f);
        cell.birth_s = time_s - c.age_s;
        cell.lifetime_s = c.life > 1e-3f ? std::clamp(c.age_s / c.life, 1.0f, 14400.0f) : s.lifetime_s;
        s.cells.push_back(cell);
        ++added;
    }
    s.max_cells = 0;
    s.derive_from_climate = false;
    return added;
}

int evaluateStormCells(const CloudState& clouds, float time_s, const Vec3& wind_mps,
                       StormCellState* out) {
    const CloudStorms& s = clouds.storms;
    if (!s.enabled || !std::isfinite(time_s)) return 0;
    int n = 0;
    // Birth suitability (ATMOSPHERE_LOCAL_WINDS_FOG.md §2): only with a
    // terrain. Without one the draws below are the old ones (positions equal
    // up to float rounding of the reordered sum).
    std::shared_ptr<const LocalWeatherEnvironment> env;
    if (clouds.local_weather.terrain_suitability) {
        env = localWeatherEnvironment();
        if (env && (!env->terrain || !env->terrain->valid)) env.reset();
    }
    // Automatic lanes, staggered like the lightning lanes: lane k cycles a
    // cell every `period`; a gap between lives keeps the count near max_cells.
    const double period = static_cast<double>(s.lifetime_s) * 1.4;
    for (int k = 0; k < s.max_cells && n < kMaxStormCells; ++k) {
        const double offset = period * static_cast<double>(k) / (std::max)(s.max_cells, 1);
        const double slot = std::floor((static_cast<double>(time_s) - offset) / period);
        const double wrapped = slot - std::floor(slot / 4294967296.0) * 4294967296.0;
        uint32_t seed = stormHash(static_cast<uint32_t>(wrapped) ^ (clouds.weather.seed * 0x2c1b3c6du) ^
                                  stormHash(static_cast<uint32_t>(k + 101)));
        const uint32_t id = seed;
        const float life = s.lifetime_s * (0.7f + 0.6f * stormRandom(seed));
        const double birth = offset + slot * period + period * 0.15 * stormRandom(seed);
        const float age = static_cast<float>(static_cast<double>(time_s) - birth);
        if (age < 0.0f || age >= life) continue;
        // Born upwind of the domain centre by half a life of drift, so cells
        // cross the domain instead of all leaving it on one side.
        float ux = stormRandom(seed);
        float uz = stormRandom(seed);
        const float upwindX = s.center_m.x - wind_mps.x * life * 0.5f;
        const float upwindZ = s.center_m.z - wind_mps.z * life * 0.5f;
        if (env) {
            // Rejection on the terrain's odds: up to 6 candidates from an
            // own stream (the cell's other draws stay unchanged); none
            // accepted -> the best candidate, so the cell count holds.
            uint32_t retry = stormHash(id ^ 0xa511e9b3u);
            float bestU = ux, bestV = uz, best = -1.0f;
            for (int attempt = 0; attempt < 6; ++attempt) {
                const float cu = attempt == 0 ? ux : stormRandom(retry);
                const float cv = attempt == 0 ? uz : stormRandom(retry);
                const float suit = birthSuitability(env.get(), upwindX + (cu - 0.5f) * s.extent_m,
                                                    upwindZ + (cv - 0.5f) * s.extent_m, wind_mps).value;
                if (suit > best) { best = suit; bestU = cu; bestV = cv; }
                if (stormRandom(retry) < suit) { bestU = cu; bestV = cv; break; }
            }
            ux = bestU;
            uz = bestV;
        }
        const float bx = upwindX + (ux - 0.5f) * s.extent_m;
        const float bz = upwindZ + (uz - 0.5f) * s.extent_m;
        StormCellState& c = out[n++];
        c.x = bx + wind_mps.x * age;
        c.z = bz + wind_mps.z * age;
        c.radius_m = s.cell_radius_m * (0.7f + 0.6f * stormRandom(seed));
        stormPhase(age / life, s.intensity * (0.6f + 0.4f * stormRandom(seed)), c);
        c.age_s = age;
        c.id = id;
        c.manual = false;
        c.manual_index = -1;
    }
    for (size_t m = 0; m < s.cells.size(); ++m) {
        const StormCell& cell = s.cells[m];
        if (n >= kMaxStormCells) break;
        const float age = time_s - cell.birth_s;
        float lifeFraction = 0.45f;   // permanent: held mid-maturity
        if (cell.lifetime_s > 0.0f) {
            if (age < 0.0f || age >= cell.lifetime_s) continue;
            lifeFraction = age / cell.lifetime_s;
        }
        StormCellState& c = out[n++];
        const float drift = cell.lifetime_s > 0.0f ? age : 0.0f;
        c.x = cell.position_m.x + wind_mps.x * drift;
        c.z = cell.position_m.z + wind_mps.z * drift;
        c.radius_m = cell.radius_m;
        stormPhase(lifeFraction, cell.intensity, c);
        // A permanent cell still cycles its turrets: wrap the age to a day.
        c.age_s = cell.lifetime_s > 0.0f ? age : age - 86400.0f * std::floor(age / 86400.0f);
        c.id = stormHash(static_cast<uint32_t>(m) * 0x9e3779b9u ^ 0x51ed270bu);
        c.manual = true;
        c.manual_index = static_cast<int>(m);
    }
    return n;
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
        configureStormLayers(c, true);
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
