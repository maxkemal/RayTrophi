#pragma once
#include "json.hpp"
#include <string>
#include <vector>

namespace audio {
std::vector<std::string> audioEmitterParameters(bool includeId);
nlohmann::json validateAudioVector(const nlohmann::json& params, const char* key,
                                    const nlohmann::json& fallback);
nlohmann::json normalizeAudioEmitter(const nlohmann::json& params);
}
