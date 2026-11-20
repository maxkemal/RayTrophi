#pragma once

#include "Vec3.h"
#include "json.hpp"
#include <string>
#include <vector>

struct SceneData;

namespace Paint {

struct BalloonInteractionTarget {
    int id = -1;
    Vec3 position;
    float radius = 0.4f;
    float carry_radius = 0.4f;
};

struct BalloonInteractionResult {
    int move = -1;
    int pop = -1;
    Vec3 position;
};

int pickBalloonTarget(SceneData& scene, float x, float y,
                      const std::vector<BalloonInteractionTarget>& targets);
// Main-thread service. Owner identity lets future tracked hands use the same
// begin/move/release lifecycle without taking a mouse-owned hold over.
BalloonInteractionResult balloonPointerCommand(
    SceneData& scene, const nlohmann::json& input, const std::string& mode,
    const std::vector<BalloonInteractionTarget>& targets,
    const std::vector<std::string>& visual_nodes);
nlohmann::json balloonInteractionStatus();
bool balloonIsHeld(int id);
int heldBalloon();
void cancelBalloonInteraction();

} // namespace Paint
