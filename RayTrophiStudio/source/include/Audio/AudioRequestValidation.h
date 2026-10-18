#pragma once
#include "json.hpp"
#include <string>

namespace audio {
// Validate the request envelope before any operation performs typed JSON access.
void validateAudioRequest(const std::string& operation, const nlohmann::json& params);
}
