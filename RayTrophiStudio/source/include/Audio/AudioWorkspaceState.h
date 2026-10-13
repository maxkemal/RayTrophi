#pragma once
#include "json.hpp"

namespace audio {
nlohmann::json defaultAudioWorkspace();
nlohmann::json updateAudioWorkspace(const nlohmann::json& current,
                                    const nlohmann::json& patch,
                                    const nlohmann::json& clips);
}
