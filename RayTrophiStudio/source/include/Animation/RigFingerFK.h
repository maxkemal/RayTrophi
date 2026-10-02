#pragma once

#include "Animation/NodeHierarchy.h"
#include "Animation/RigAnatomy.h"
#include <string>
#include <vector>

namespace RigAuthoring {

bool poseDetailedHumanoidFingers(const RayTrophi::NodeHierarchy& input,
                                 const RigAnatomy& anatomy, const std::string& side,
                                 float curl, float spread, float thumb,
                                 RayTrophi::NodeHierarchy& output,
                                 std::vector<std::string>& affectedBones,
                                 std::string& error);

} // namespace RigAuthoring
