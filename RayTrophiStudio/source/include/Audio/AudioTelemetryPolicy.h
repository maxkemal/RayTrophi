#pragma once
#include "json.hpp"

namespace audio {
nlohmann::json validateTelemetryPolicy(const nlohmann::json& value);
nlohmann::json updateTelemetryPolicy(const nlohmann::json& current,
                                     const nlohmann::json& patch);
}
