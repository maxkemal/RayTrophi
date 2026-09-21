#pragma once
#include "Animation/RigIK.h"

namespace RigAuthoring {
// Called by the common IK evaluator after definition/runtime validation.
bool solveChainIK(const RayTrophi::NodeHierarchy& input, const IKControl& control,
                  const IKPose& pose, const Matrix4x4& placement, RayTrophi::NodeHierarchy& output,
                  std::string& error);
}
