#pragma once

#include "Vec3.h"
#include "json.hpp"
#include <string>
#include <vector>

struct SceneData;
class TriangleMesh;
class Matrix4x4;

namespace Paint {

struct BalloonInteractionTarget {
    int id = -1;
    Vec3 position;
    float radius = 0.4f;
    float carry_radius = 0.4f;
    const TriangleMesh* visual_mesh = nullptr;
    float billboard_radius = 0.0f;
};

struct BalloonDisplacement {
    int id;
    Vec3 position;
};

struct BalloonInteractionResult {
    int move = -1;
    int pop = -1;
    Vec3 position;
    std::vector<BalloonDisplacement> pushed;
};

int pickBalloonTarget(SceneData& scene, float x, float y,
                      const std::vector<BalloonInteractionTarget>& targets);
// Main-thread service. Owner identity lets future tracked hands use the same
// begin/move/release lifecycle without taking a mouse-owned hold over.
BalloonInteractionResult balloonPointerCommand(
    SceneData& scene, const nlohmann::json& input, const std::string& mode,
    const std::vector<BalloonInteractionTarget>& targets,
    const std::vector<std::string>& visual_nodes,
    const std::vector<Vec3>* barrier = nullptr);
nlohmann::json balloonInteractionStatus();
bool balloonIsHeld(int id);
std::vector<Vec3> balloonRoomTriangles(SceneData& scene,
                                       const std::vector<std::string>& visual_nodes);
int heldBalloon();
void cancelBalloonInteraction();
float balloonCarryCompression();
void applyBalloonCarryCompression(int id, Matrix4x4& transform,
                                  float axial = 0.0f, float lateral = 1.0f);
int takeReleasedCompressedBalloon();
int updateBalloonCrush(float dt);
Vec3 balloonCrushNormal();

} // namespace Paint
