#include "Atmosphere/AtmosphereClimate.h"

#include <algorithm>
#include <cmath>
#include <mutex>

namespace atmosphere {

namespace {

std::mutex g_ambient_mutex;
AmbientSnapshot g_ambient;

bool fail(std::string* error, const char* what) {
    if (error) *error = what;
    return false;
}

bool finite(float v) { return std::isfinite(v); }

} // namespace

bool validateClimate(const ClimateState& c, std::string* error) {
    if (!finite(c.surface_temperature_k) || c.surface_temperature_k < 150.0f || c.surface_temperature_k > 400.0f)
        return fail(error, "surface_temperature_k is in KELVIN and must be within 150..400");
    if (!finite(c.lapse_rate_k_per_m) || c.lapse_rate_k_per_m < -0.02f || c.lapse_rate_k_per_m > 0.02f)
        return fail(error, "lapse_rate_k_per_m must be within -0.02..0.02 K/m (ISA is 0.0065)");
    // The troposphere must not reach absolute zero before the tropopause.
    if (c.surface_temperature_k - c.lapse_rate_k_per_m * kTropopauseAltitudeM < 100.0f)
        return fail(error, "lapse_rate_k_per_m cools the air below 100 K before the tropopause");
    if (!finite(c.surface_relative_humidity) || c.surface_relative_humidity < 0.0f || c.surface_relative_humidity > 1.0f)
        return fail(error, "surface_relative_humidity must be within 0..1");
    if (!finite(c.surface_pressure_pa) || c.surface_pressure_pa < 30000.0f || c.surface_pressure_pa > 120000.0f)
        return fail(error, "surface_pressure_pa is in PASCAL and must be within 30000..120000");
    if (!finite(c.wind_speed_mps) || c.wind_speed_mps < 0.0f || c.wind_speed_mps > 150.0f)
        return fail(error, "wind_speed_mps must be within 0..150 m/s");
    if (!finite(c.wind_direction.x) || !finite(c.wind_direction.y) || !finite(c.wind_direction.z))
        return fail(error, "wind_direction must be finite");
    const float horizontal = std::sqrt(c.wind_direction.x * c.wind_direction.x +
                                       c.wind_direction.z * c.wind_direction.z);
    if (horizontal < 1e-6f)
        return fail(error, "wind_direction needs a horizontal (x/z) component");
    if (!finite(c.instability) || c.instability < 0.0f || c.instability > 1.0f)
        return fail(error, "instability must be within 0..1");
    return true;
}

bool normalizeWindDirection(ClimateState& c) {
    const float horizontal = std::sqrt(c.wind_direction.x * c.wind_direction.x +
                                       c.wind_direction.z * c.wind_direction.z);
    if (!(horizontal >= 1e-6f)) return false;
    c.wind_direction = Vec3(c.wind_direction.x / horizontal, 0.0f, c.wind_direction.z / horizontal);
    return true;
}

void publishAmbient(const ClimateState& c, float altitude_offset_m) {
    std::lock_guard<std::mutex> lock(g_ambient_mutex);
    g_ambient.climate = c;
    g_ambient.altitude_offset_m = std::isfinite(altitude_offset_m) ? altitude_offset_m : 0.0f;
    ++g_ambient.revision;
}

AmbientSnapshot ambientSnapshot() {
    std::lock_guard<std::mutex> lock(g_ambient_mutex);
    return g_ambient;
}

ClimateSample sampleAmbient(const Vec3& scene_pos) {
    const AmbientSnapshot s = ambientSnapshot();
    return sampleClimate(s.climate, scene_pos.y + s.altitude_offset_m);
}

float ambientSurfaceKelvin() {
    return sampleAmbient(Vec3(0.0f, 0.0f, 0.0f)).temperature_k;
}

Vec3 ambientWindMps() {
    const AmbientSnapshot s = ambientSnapshot();
    return s.climate.wind_direction * s.climate.wind_speed_mps;
}

ClimateSample sampleClimate(const ClimateState& c, float altitude_m) {
    ClimateSample s;
    s.altitude_m = altitude_m;

    const float T0 = c.surface_temperature_k;
    const float p0 = c.surface_pressure_pa;
    const float L  = c.lapse_rate_k_per_m;
    // g*M/R, the exponent scale of the barometric formula.
    const float gMR = kGravity * kMolarMassAir / kUniversalGasConstant;

    // Troposphere up to the tropopause, isothermal above it (ISA layer 1).
    const float hTrop = std::min(altitude_m, kTropopauseAltitudeM);
    float T = T0 - L * hTrop;
    float p;
    if (std::fabs(L) > 1e-7f) {
        p = p0 * std::pow(T / T0, gMR / L);
    } else {
        p = p0 * std::exp(-gMR * hTrop / T0);
    }
    if (altitude_m > kTropopauseAltitudeM) {
        p *= std::exp(-gMR * (altitude_m - kTropopauseAltitudeM) / T);
    }

    s.temperature_k     = T;
    s.pressure_pa       = p;
    s.air_density_kg_m3 = p / (kSpecificGasConstantAir * T);
    s.relative_humidity = c.surface_relative_humidity;
    s.wind_mps          = c.wind_direction * c.wind_speed_mps;
    return s;
}

ClimateState climateFromKeyed(const ClimateState& current,
                              float temperature_k, float lapse_rate_k_per_m,
                              float relative_humidity, float pressure_pa,
                              const Vec3& wind_direction, float wind_speed_mps) {
    ClimateState c = current;
    c.surface_temperature_k     = temperature_k;
    c.lapse_rate_k_per_m        = lapse_rate_k_per_m;
    c.surface_relative_humidity = relative_humidity;
    c.surface_pressure_pa       = pressure_pa;
    c.wind_speed_mps            = wind_speed_mps;
    c.wind_direction            = wind_direction;
    if (!normalizeWindDirection(c)) c.wind_direction = current.wind_direction;
    return c;
}

float hygroscopicMieScale(float relative_humidity) {
    constexpr float kGamma = 0.5f;   // continental aerosol, extinction
    const float rh = std::clamp(relative_humidity, 0.0f, 0.95f);
    return std::pow(1.0f - rh, -kGamma);
}

namespace {
float smooth01(float e0, float e1, float x) {
    const float t = std::min(1.0f, std::max(0.0f, (x - e0) / (e1 - e0)));
    return t * t * (3.0f - 2.0f * t);
}
float mixf(float a, float b, float t) { return a + (b - a) * t; }
} // namespace

DerivedWeather deriveWeather(const ClimateState& c, float surface_altitude_m) {
    DerivedWeather w;
    const float T = c.surface_temperature_k;
    const float Tc = T - kKelvinOffset;
    const float rh = std::min(1.0f, std::max(0.01f, c.surface_relative_humidity));
    // Dew point: Magnus (Alduchov & Eskridge 1996).
    const float g = std::log(rh) + 17.62f * Tc / (243.12f + Tc);
    const float TdC = 243.12f * g / (17.62f - g);
    w.dew_point_k = TdC + kKelvinOffset;
    // Lifting condensation level (Espy): ~125 m per kelvin of dew-point spread.
    const float lcl = std::max(0.0f, 125.0f * (Tc - TdC));
    w.cloud_base_m = surface_altitude_m + std::max(150.0f, lcl);

    const float I = c.instability;
    // Genus: stable air -> stratiform; unstable + moist -> convective. Dry air
    // cannot build towers however unstable it is.
    w.cloud_type = std::min(1.0f, std::max(0.0f, I * mixf(0.45f, 1.05f, smooth01(0.35f, 0.8f, rh))));
    // Coverage from humidity (RH >= ~0.95: overcast); convection leaves gaps.
    w.cloud_coverage = smooth01(0.5f, 0.95f, rh) * mixf(1.0f, 0.75f, smooth01(0.5f, 0.9f, w.cloud_type));
    // Depth: a stratus sheet ~400 m, cumulus ~2 km, a cumulonimbus to the
    // tropopause.
    const float tropo = kTropopauseAltitudeM;
    float thick = mixf(400.0f, 2000.0f, smooth01(0.0f, 0.6f, w.cloud_type));
    thick = mixf(thick, std::max(2000.0f, tropo - w.cloud_base_m), smooth01(0.65f, 1.0f, w.cloud_type));
    w.cloud_thickness_m = std::max(100.0f, std::min(thick, tropo - w.cloud_base_m));

    // Freezing level from the lapse rate (isothermal/inverted: never, or at
    // the ground when the surface is already below freezing).
    if (Tc <= 0.0f) w.freezing_level_m = surface_altitude_m;
    else if (c.lapse_rate_k_per_m > 1e-5f) w.freezing_level_m = surface_altitude_m + Tc / c.lapse_rate_k_per_m;
    else w.freezing_level_m = 1e6f;
    // Snow survives to the ground when the surface is at most ~+1 C.
    w.snow = Tc <= 1.0f;

    // Precipitation: convective showers under deep cloud, drizzle from a
    // saturated stratus. Peak rate under a cluster core (the weather map's B
    // channel scales it spatially). ~25 mm/h is a heavy thunderstorm.
    const float convective = 25.0f * smooth01(0.5f, 1.0f, w.cloud_type) *
                             w.cloud_coverage * smooth01(0.65f, 0.95f, rh);
    const float drizzle = 1.0f * (1.0f - w.cloud_type) * smooth01(0.9f, 1.0f, rh) * w.cloud_coverage;
    w.precipitation_mm_h = convective + drizzle;
    // Flashes per minute from a mature cumulonimbus.
    w.lightning_per_minute = 10.0f * smooth01(0.75f, 1.0f, w.cloud_type) * smooth01(0.6f, 1.0f, I);
    return w;
}

void derivedWeatherToJson(const DerivedWeather& w, nlohmann::json& j) {
    j["dew_point_k"]          = w.dew_point_k;
    j["cloud_base_m"]         = w.cloud_base_m;
    j["cloud_thickness_m"]    = w.cloud_thickness_m;
    j["cloud_type"]           = w.cloud_type;
    j["cloud_coverage"]       = w.cloud_coverage;
    j["freezing_level_m"]     = w.freezing_level_m;
    j["precipitation_mm_h"]   = w.precipitation_mm_h;
    j["snow"]                 = w.snow;
    j["lightning_per_minute"] = w.lightning_per_minute;
}

void climateToJson(const ClimateState& c, nlohmann::json& j) {
    j["surface_temperature_k"]     = c.surface_temperature_k;
    j["lapse_rate_k_per_m"]        = c.lapse_rate_k_per_m;
    j["surface_relative_humidity"] = c.surface_relative_humidity;
    j["surface_pressure_pa"]       = c.surface_pressure_pa;
    j["wind_direction"]            = { c.wind_direction.x, c.wind_direction.y, c.wind_direction.z };
    j["wind_speed_mps"]            = c.wind_speed_mps;
    j["instability"]               = c.instability;
}

bool climateFromJson(const nlohmann::json& j, ClimateState& out, std::string* error) {
    ClimateState c;
    c.surface_temperature_k     = j.value("surface_temperature_k", c.surface_temperature_k);
    c.lapse_rate_k_per_m        = j.value("lapse_rate_k_per_m", c.lapse_rate_k_per_m);
    c.surface_relative_humidity = j.value("surface_relative_humidity", c.surface_relative_humidity);
    c.surface_pressure_pa       = j.value("surface_pressure_pa", c.surface_pressure_pa);
    c.wind_speed_mps            = j.value("wind_speed_mps", c.wind_speed_mps);
    c.instability               = j.value("instability", c.instability);
    if (j.contains("wind_direction") && j["wind_direction"].is_array() && j["wind_direction"].size() == 3) {
        const auto& w = j["wind_direction"];
        c.wind_direction = Vec3(w[0].get<float>(), w[1].get<float>(), w[2].get<float>());
    }
    if (!validateClimate(c, error)) {
        out = ClimateState{};
        return false;
    }
    normalizeWindDirection(c);
    out = c;
    return true;
}

} // namespace atmosphere
