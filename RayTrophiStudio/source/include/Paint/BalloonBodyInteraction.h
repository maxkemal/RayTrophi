#pragma once

#include "Paint/BalloonInteraction.h"

namespace Paint {
// Real and synthetic poses arrive through the same Input::BodyPoseRuntime.
void tickBalloonBodyInteraction(SceneData& scene, bool enabled,
                               const std::vector<BalloonInteractionTarget>& targets);
void resetBalloonBodyInteraction();
nlohmann::json balloonBodyInteractionStatus(bool enabled);
bool balloonBodyControlCommand(nlohmann::json& config, const nlohmann::json& input,
                               nlohmann::json& output, std::string& error);
} // namespace Paint
