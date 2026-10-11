#pragma once

#include "Api/RtApi.h"

struct WeatherParams;

namespace rtapi {
Result getWeatherEffectsJson(std::string& out);
Result setWeatherEffectsJson(const std::string& patch);
Result setWeatherEffects(const WeatherParams& weather);
} // namespace rtapi
