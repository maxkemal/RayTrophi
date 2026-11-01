#pragma once
#include "json.hpp"
#include <string>

struct AnimationData;
namespace RayTrophi {
class NodeHierarchy;
}

namespace RigAuthoring {
struct RigAnatomy;
struct HumanWalkRecipe {
    float fps = 30.f;
    float cadence = 100.f;   // Steps per minute.
    int cycles = 2;          // One cycle contains left and right steps.
    float stride = .35f;     // Fraction of character height.
    float stepHeight = .06f; // Fraction of character height.
    float bodyBounce = 0.f;  // Fraction of character height, added to the stance-leg height.
    float armSwing = .7f;    // 0..1 multiplier.
    float bodyMotion = .75f; // 0..1 pelvis and torso motion.
    float legReach = .998f;  // Largest hip-to-ankle distance as a fraction of leg length.
    float stanceKnee = 18.f; // Loading knee flexion of the supporting leg, degrees.
    float heelStrike = 15.f; // Toe-up degrees at heel contact.
    float toeOff = 30.f;     // Heel-up degrees at toe-off.
};

bool inspectHumanWalkRecipe(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                            const HumanWalkRecipe &recipe, nlohmann::json &output,
                            std::string &error);
bool buildHumanWalkClip(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                        const HumanWalkRecipe &recipe, AnimationData &output, std::string &error);
} // namespace RigAuthoring
