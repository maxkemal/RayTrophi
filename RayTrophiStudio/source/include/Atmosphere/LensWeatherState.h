#pragma once

#include "json.hpp"
#include "Atmosphere/LensWeatherSettings.h"

namespace atmosphere {
LensWeatherSettings lensWeatherFromCameraJson(const std::string& text);
LensWeatherSettings lensWeatherFromProjectJson(nlohmann::json settings);
nlohmann::json lensWeatherJson(const LensWeatherSettings& settings);
LensWeatherSettings patchLensWeather(const LensWeatherSettings& current,
                                    const nlohmann::json& patch);
} // namespace atmosphere
