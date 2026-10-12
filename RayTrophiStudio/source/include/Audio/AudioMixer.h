#pragma once
#include "json.hpp"
#include <array>
#include <string>

namespace audio {
class AudioMixer {
public:
    static const std::array<std::string, 5>& names();
    static nlohmann::json validate(const nlohmann::json& input);
    static nlohmann::json update(const nlohmann::json& current, const nlohmann::json& patch);
    static float effectiveGain(const nlohmann::json& config, const std::string& bus);
    static float directMasterGain(const nlohmann::json& config);
};
}
