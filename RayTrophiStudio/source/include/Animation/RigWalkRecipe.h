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
    float bodyBounce = .02f; // Fraction of character height.
    float armSwing = .7f;    // 0..1 multiplier.
    float bodyMotion = .75f; // 0..1 pelvis and torso motion.
};

bool inspectHumanWalkRecipe(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                            const HumanWalkRecipe &recipe, nlohmann::json &output,
                            std::string &error);
bool buildHumanWalkClip(const RayTrophi::NodeHierarchy &hierarchy, const RigAnatomy &anatomy,
                        const HumanWalkRecipe &recipe, AnimationData &output, std::string &error);
} // namespace RigAuthoring
