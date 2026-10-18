#pragma once
#include "Audio/AcousticPreview.h"

namespace audio {
nlohmann::json validateAcousticSettings(const nlohmann::json& input);
nlohmann::json validateSourceAcoustics(const nlohmann::json& input);
void reconcileAcousticSettings(nlohmann::json& scene, AcousticPreview& preview);
AcousticPreview stageAcousticEmitter(const nlohmann::json& scene,
    const AcousticPreview& preview, const nlohmann::json& emitter);
nlohmann::json acousticSettingsOperation(nlohmann::json& scene, AcousticPreview& preview,
    const std::string& operation, const nlohmann::json& params);
}
