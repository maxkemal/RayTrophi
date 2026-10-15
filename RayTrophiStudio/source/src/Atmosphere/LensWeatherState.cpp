#include "Atmosphere/LensWeatherState.h"
#include <cmath>
#include <atomic>
#include <stdexcept>

namespace atmosphere {
std::uint64_t nextLensWeatherIdentity() {
    static std::atomic<std::uint64_t> sequence{0};
    return ++sequence;
}

LensWeatherSettings lensWeatherFromCameraJson(const std::string& text) {
    const auto camera = nlohmann::json::parse(text);
    return lensWeatherFromProjectJson(camera.value("lens_weather", nlohmann::json::object()));
}

LensWeatherSettings lensWeatherFromProjectJson(nlohmann::json settings) {
    if (settings.is_object()) {
        settings.erase("glass_temperature_c"); // Retired local override; inherit climate.
    }
    return patchLensWeather({}, settings);
}

nlohmann::json lensWeatherJson(const LensWeatherSettings& s) {
    return {{"enabled", s.enabled}, {"coating", s.hydrophobic ? "hydrophobic" : "bare"},
        {"hood_deg", s.hood_deg}, {"diameter_mm", s.diameter_mm}, {"seed", s.seed}};
}

LensWeatherSettings patchLensWeather(const LensWeatherSettings& current,
                                    const nlohmann::json& patch) {
    const auto require = [](bool valid, const char* message) {
        if (!valid) {
            throw std::invalid_argument(message);
        }
    };
    require(patch.is_object(), "Lens weather patch must be an object");
    auto candidate = current;
    for (auto field = patch.begin(); field != patch.end(); ++field) {
        const auto& key = field.key();
        const auto& value = field.value();
        if (key == "enabled") {
            require(value.is_boolean(), "enabled must be boolean");
            candidate.enabled = value.get<bool>();
        } else if (key == "coating") {
            require(value.is_string() && (value == "bare" || value == "hydrophobic"),
                    "coating must be bare or hydrophobic");
            candidate.hydrophobic = value == "hydrophobic";
        } else if (key == "seed") {
            require(value.is_number_integer() && value.get<double>() >= 0 &&
                    value.get<double>() <= 4294967295.0, "seed must be uint32");
            candidate.seed = value.get<std::uint32_t>();
        } else if (key == "glass_temperature_c") {
            throw std::invalid_argument(
                "glass_temperature_c was removed; use world.set_climate surface_temperature_k");
        } else if (key == "hood_deg" || key == "diameter_mm") {
            require(value.is_number(), "Lens dimensions must be numeric");
            const auto number = value.get<double>();
            const double low = key == "hood_deg" ? 0 : 10;
            const double high = key == "hood_deg" ? 89 : 200;
            require(std::isfinite(number) && number >= low && number <= high,
                    "hood_deg must be 0..89; diameter_mm must be 10..200");
            if (key == "hood_deg") {
                candidate.hood_deg = static_cast<float>(number);
            } else {
                candidate.diameter_mm = static_cast<float>(number);
            }
        } else {
            throw std::invalid_argument("Unknown lens weather field: " + key);
        }
    }
    if (lensWeatherJson(candidate) != lensWeatherJson(current)) {
        ++candidate.history_revision;
        if (candidate.enabled != current.enabled || candidate.seed != current.seed ||
            candidate.diameter_mm != current.diameter_mm) {
            ++candidate.revision;
        }
    }
    return candidate;
}
} // namespace atmosphere
