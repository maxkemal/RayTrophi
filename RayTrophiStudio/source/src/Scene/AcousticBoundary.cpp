#include "Audio/AcousticBoundary.h"
#include "SubstanceLibrary.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace audio {
namespace {
using json = nlohmann::json;
constexpr double pi = 3.14159265358979323846;
constexpr std::array<unsigned, 6> frequencies{125, 250, 500, 1000, 2000, 4000};

std::array<float, 6> losses(const RayTrophiSim::SubstanceProfile& p) {
    return {p.acoustic_medium_loss_125, p.acoustic_medium_loss_250,
        p.acoustic_medium_loss_500, p.acoustic_medium_loss_1000,
        p.acoustic_medium_loss_2000, p.acoustic_medium_loss_4000};
}
json pressure(const std::complex<double>& value) {
    const double magnitude = std::abs(value);
    return {{"real", value.real()}, {"imaginary", value.imag()},
        {"magnitude", magnitude},
        {"phase_radians", magnitude > 1e-15 ? json(std::arg(value)) : json()}};
}
double number(const json& params, const char* key, double fallback, double low, double high) {
    const auto value = params.value(key, json(fallback));
    if (!value.is_number() || !std::isfinite(value.get<double>()) ||
        value.get<double>() < low || value.get<double>() > high) {
        throw std::invalid_argument(std::string(key) + " is outside finite bounds");
    }
    return value.get<double>();
}
const RayTrophiSim::SubstanceProfile& medium(const json& params, const char* key) {
    if (!params.contains(key) || !params.at(key).is_string()) {
        throw std::invalid_argument(std::string(key) + " requires a substance name");
    }
    const auto* profile = RayTrophiSim::tryFindSubstance(params.at(key).get<std::string>());
    if (!profile) {
        throw std::invalid_argument(std::string(key) + " names an unknown substance");
    }
    if (!profile->acoustic_medium_defined) {
        throw std::invalid_argument(std::string(key) + " has no defined acoustic medium");
    }
    if (!std::isfinite(profile->acoustic_medium_density) ||
        !std::isfinite(profile->acoustic_medium_speed) ||
        profile->acoustic_medium_density < 1e-6f || profile->acoustic_medium_speed < 1e-3f) {
        throw std::invalid_argument(std::string(key) + " has invalid acoustic medium data");
    }
    return *profile;
}
} // namespace

nlohmann::json acousticMediumState(const RayTrophiSim::SubstanceProfile& profile) {
    json result = {{"defined", profile.acoustic_medium_defined},
        {"model", "nondispersive_equivalent_fluid"}};
    if (!profile.acoustic_medium_defined) {
        return result;
    }
    const double speed = profile.acoustic_medium_speed;
    result["density_kg_m3"] = profile.acoustic_medium_density;
    result["sound_speed_mps"] = speed;
    result["impedance_rayl"] = speed * profile.acoustic_medium_density;
    result["bands"] = json::array();
    const auto attenuation = losses(profile);
    for (unsigned i = 0; i < frequencies.size(); ++i) {
        result["bands"].push_back({{"frequency_hz", frequencies[i]},
            {"wavelength_meters", speed / frequencies[i]},
            {"loss_db_per_meter", attenuation[i]}});
    }
    return result;
}

nlohmann::json queryAcousticBoundary(const nlohmann::json& params) {
    if (!params.is_object()) {
        throw std::invalid_argument("acoustics_boundary requires an object");
    }
    for (const auto& [key, value] : params.items()) {
        if (key != "from_substance" && key != "to_substance" &&
            key != "incidence_degrees" && key != "transmitted_distance_meters") {
            throw std::invalid_argument("Unknown acoustic boundary parameter: " + key);
        }
    }
    const auto& from = medium(params, "from_substance");
    const auto& to = medium(params, "to_substance");
    const double angle = number(params, "incidence_degrees", 0, 0, 89.999);
    const double distance = number(params, "transmitted_distance_meters", 0, 0, 1e6);
    const double c1 = from.acoustic_medium_speed, c2 = to.acoustic_medium_speed;
    const double z1 = from.acoustic_medium_density * c1;
    const double z2 = to.acoustic_medium_density * c2;
    const double cosIncident = std::cos(angle * pi / 180);
    const double sinTransmitted = (c2 / c1) * std::sin(angle * pi / 180);
    const bool evanescent = sinTransmitted > 1.0;
    // exp(-i omega t): positive imaginary transmitted wave number decays into medium 2.
    const std::complex<double> cosTransmitted = std::sqrt(
        std::complex<double>(1 - sinTransmitted * sinTransmitted, 0));
    const auto denominator = z2 * cosIncident + z1 * cosTransmitted;
    const auto reflected = (z2 * cosIncident - z1 * cosTransmitted) / denominator;
    const auto transmitted = (2 * z2 * cosIncident) / denominator;
    const double reflectedEnergy = std::clamp(std::norm(reflected), 0.0, 1.0);
    const double transmittedEnergy = evanescent ? 0 : std::clamp(
        (z1 / z2) * cosTransmitted.real() / cosIncident * std::norm(transmitted), 0.0, 1.0);
    json result = {{"model", "lossless_planar_fluid_interface_with_bulk_loss"},
        {"from_substance", from.name}, {"to_substance", to.name},
        {"from_medium", acousticMediumState(from)}, {"to_medium", acousticMediumState(to)},
        {"incidence_degrees", angle}, {"total_internal_reflection", evanescent},
        {"transmitted_angle_degrees", evanescent ? json() :
            json(std::asin(std::clamp(sinTransmitted, 0.0, 1.0)) * 180 / pi)},
        {"critical_incidence_degrees", c2 > c1 ? json(std::asin(c1 / c2) * 180 / pi) : json()},
        {"reflection_energy", reflectedEnergy}, {"transmission_energy", transmittedEnergy},
        {"energy_balance_error", reflectedEnergy + transmittedEnergy - 1},
        {"reflected_pressure", pressure(reflected)}, {"transmitted_pressure", pressure(transmitted)},
        {"energy_quantity", "normal_time_averaged_flux_ratio"},
        {"time_convention", "exp(-i*omega*t)"}, {"transmitted_distance_meters", distance},
        {"delay_seconds", evanescent ? json() : json(distance / c2)},
        {"diffraction", false}, {"elastic_mode_conversion", false},
        {"gpu_refracted_paths", false}, {"bands", json::array()}};
    const auto attenuation = losses(to);
    for (unsigned i = 0; i < frequencies.size(); ++i) {
        const double amplitudeLoss = std::pow(10.0, -attenuation[i] * distance / 20.0);
        const double phase = std::remainder(2 * pi * frequencies[i] * distance / c2, 2 * pi);
        result["bands"].push_back({{"frequency_hz", frequencies[i]},
            {"incident_wavelength_meters", c1 / frequencies[i]},
            {"transmitted_wavelength_meters", c2 / frequencies[i]},
            {"bulk_pressure_ratio", evanescent ? json() : json(amplitudeLoss)},
            {"transmitted_energy_after_distance", transmittedEnergy * amplitudeLoss * amplitudeLoss},
            {"transmitted_pressure_after_distance", evanescent ? json() :
                pressure(transmitted * std::polar(amplitudeLoss, phase))}});
    }
    return result;
}
} // namespace audio
