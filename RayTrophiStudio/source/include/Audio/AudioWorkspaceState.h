#pragma once
#include "json.hpp"

namespace audio {
nlohmann::json defaultAudioWorkspace();
void pruneAudioSelections(nlohmann::json& workspace, const nlohmann::json& clips,
                          const nlohmann::json& tracks);
nlohmann::json updateAudioWorkspace(const nlohmann::json& current,
                                    const nlohmann::json& patch,
                                    const nlohmann::json& clips,
                                    const nlohmann::json& tracks);
}
