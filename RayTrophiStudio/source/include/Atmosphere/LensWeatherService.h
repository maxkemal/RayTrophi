#pragma once
#include "Atmosphere/LensWeatherState.h"
#include <string>
class Camera;
struct UIContext;
namespace atmosphere {
nlohmann::json setLensWeather(UIContext& context, Camera& camera,
                              const nlohmann::json& patch);
nlohmann::json resetLensWeather(UIContext& context, Camera& camera);
nlohmann::json sampleLensWeather(UIContext& context, const std::string& backend);
} // namespace atmosphere
