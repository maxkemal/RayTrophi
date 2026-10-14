#pragma once
#include "json.hpp"

namespace audio {
nlohmann::json synthesisPresets();
// Bounded, deterministic offline synthesis. Never runs in the device callback.
nlohmann::json generateSyntheticAudio(const nlohmann::json& params);
}
